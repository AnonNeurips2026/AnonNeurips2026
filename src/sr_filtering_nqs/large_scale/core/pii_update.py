from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as fcore
from netket.stats import statistics

from advanced_drivers._src.callbacks.base import AbstractCallback
from sr_filtering_nqs.large_scale.core.multilambda_sr import (
    batch_residual_metrics,
    current_cached_batch,
    tree_l2_norm,
)


def _energy_shift_for_matrix(local_loss, *, matrix_side: int, tau: float) -> jnp.ndarray:
    energies = jnp.real(jnp.asarray(local_loss)).reshape(-1)
    shift = energies - float(tau)
    if shift.shape[0] == matrix_side:
        return shift
    if 2 * shift.shape[0] == matrix_side:
        return jnp.repeat(shift, 2)
    raise ValueError(
        "local_loss and NTK matrix shape disagree: "
        f"local_loss has {shift.shape[0]} entries but matrix side is {matrix_side}"
    )


def current_energy_tau(local_loss, weights=None) -> float:
    """Current-energy shift used by Rayleigh Gauss-Newton."""
    energies = np.real(np.asarray(local_loss)).reshape(-1)
    if weights is None:
        return float(np.mean(energies))
    weights_array = np.real(np.asarray(weights)).reshape(-1)
    if weights_array.shape != energies.shape:
        raise ValueError(
            "local_loss and weights shape disagree: "
            f"{energies.shape} != {weights_array.shape}"
        )
    weight_sum = float(np.sum(weights_array))
    if weight_sum == 0.0:
        raise ValueError("Cannot compute current-energy tau with zero total weight")
    return float(np.sum(energies * weights_array) / weight_sum)


def compute_energy_shifted_ntk_matrix(
    ntk_matrix,
    local_loss,
    *,
    tau: float,
    diag_shift: float = 0.0,
) -> jnp.ndarray:
    """Sample-space PII matrix for ``[H(theta) - tau S(theta)] xi = grad``.

    This is the on-the-fly NTK analogue of the projected inverse-iteration
    preconditioner from arXiv:2507.10835. The large-scale trainers only expose
    the centered sample-space NTK, so we use the symmetric local-energy weighted
    form ``0.5 * ((E_i - tau) + (E_j - tau)) * K_ij``.
    """
    ntk = jnp.asarray(ntk_matrix)
    if ntk.ndim != 2 or ntk.shape[0] != ntk.shape[1]:
        raise ValueError(f"Expected a square NTK matrix, got shape={ntk.shape}")
    energy_shift = _energy_shift_for_matrix(
        local_loss,
        matrix_side=int(ntk.shape[0]),
        tau=float(tau),
    )
    matrix = 0.5 * (energy_shift[:, None] + energy_shift[None, :]) * ntk
    if float(diag_shift) != 0.0:
        matrix = matrix + float(diag_shift) * jnp.eye(ntk.shape[0], dtype=ntk.dtype)
    return 0.5 * (matrix + matrix.T)


def compute_rgn_ntk_matrix(
    ntk_matrix,
    local_loss,
    *,
    weights=None,
    diag_shift: float = 0.0,
) -> jnp.ndarray:
    """Rayleigh Gauss-Newton matrix using ``tau = E(theta)``."""
    tau = current_energy_tau(local_loss, weights)
    return compute_energy_shifted_ntk_matrix(
        ntk_matrix,
        local_loss,
        tau=tau,
        diag_shift=diag_shift,
    )


def make_pii_solver(
    *,
    local_loss,
    tau: float,
    diag_shift: float,
    base_solver_fn: Callable[[Any, Any], Any] | None = None,
):
    def pii_solver(ntk_matrix, dv):
        matrix = compute_energy_shifted_ntk_matrix(
            ntk_matrix,
            local_loss,
            tau=float(tau),
            diag_shift=float(diag_shift),
        )
        if base_solver_fn is None:
            return jnp.linalg.solve(matrix, dv)
        return base_solver_fn(matrix, dv)

    return pii_solver


def make_rgn_solver(
    *,
    local_loss,
    weights=None,
    diag_shift: float,
    base_solver_fn: Callable[[Any, Any], Any] | None = None,
):
    tau = current_energy_tau(local_loss, weights)
    return make_pii_solver(
        local_loss=local_loss,
        tau=tau,
        diag_shift=diag_shift,
        base_solver_fn=base_solver_fn,
    )


def _is_signature_mismatch(exc: TypeError) -> bool:
    text = str(exc)
    return (
        "unexpected keyword argument" in text
        or "missing 1 required keyword-only argument: 'n_replicas'" in text
    )


def _compute_delta_with_solver(driver, batch, *, solver_fn):
    local_grad = batch.local_grad
    variables = batch.variables
    model_state, params = fcore.pop(variables, "params")

    common_kwargs = {
        "diag_shift": 0.0,
        "solver_fn": solver_fn,
        "mode": driver.mode,
        "proj_reg": None,
        "momentum": None,
        "old_updates": None,
        "chunk_size": driver.chunk_size_bwd,
        "collect_quadratic_model": False,
        "collect_gradient_statistics": False,
        "collect_residual_info": True,
    }

    weight_kwargs_options = [
        {"importance_sampling_weights": batch.weights},
        {"pdf": jnp.asarray(batch.weights) / float(batch.sample_count)},
        {},
    ]
    replica_kwargs_options = []
    if batch.n_replicas is not None:
        replica_kwargs_options.append({"n_replicas": int(batch.n_replicas)})
    replica_kwargs_options.append({})

    last_exc = None
    for replica_kwargs in replica_kwargs_options:
        for weight_kwargs in weight_kwargs_options:
            try:
                delta, old_updates, info = driver.update_fn(
                    batch.afun,
                    local_grad,
                    params,
                    model_state,
                    batch.samples,
                    **weight_kwargs,
                    **replica_kwargs,
                    **common_kwargs,
                )
                driver._old_updates = old_updates
                last_exc = None
                break
            except TypeError as exc:
                if not _is_signature_mismatch(exc):
                    raise
                last_exc = exc
        if last_exc is None:
            break

    if last_exc is not None:
        raise last_exc
    if info is None:
        info = {}
    loss_stats = statistics(batch.local_loss * batch.weights)
    jax.block_until_ready(jax.tree_util.tree_leaves(delta))
    return delta, loss_stats, info


def _compute_energy_shifted_update(
    driver,
    *,
    tau: float,
    diag_shift: float = 0.0,
    method: str,
    batch=None,
) -> tuple[Any, dict[str, Any]]:
    if not getattr(driver, "use_ntk", False) or not getattr(driver, "on_the_fly", False):
        raise NotImplementedError(
            f"{method.upper()} currently requires use_ntk=True and on_the_fly=True"
        )

    batch = current_cached_batch(driver) if batch is None else batch
    base_solver = getattr(driver, "_linear_solver_fn", None)
    pii_solver = make_pii_solver(
        local_loss=batch.local_loss,
        tau=float(tau),
        diag_shift=float(diag_shift),
        base_solver_fn=base_solver,
    )
    delta, loss_stats, info = _compute_delta_with_solver(driver, batch, solver_fn=pii_solver)
    driver._loss_stats = loss_stats

    ntk_matrix = info.get("ntk_matrix")
    matrix_shape = None
    energy_shift_min = float("nan")
    energy_shift_max = float("nan")
    if ntk_matrix is not None:
        matrix_shape = [int(x) for x in np.asarray(ntk_matrix).shape]
        energy_shift = np.asarray(
            _energy_shift_for_matrix(
                batch.local_loss,
                matrix_side=matrix_shape[0],
                tau=float(tau),
            )
        )
        energy_shift_min = float(np.min(energy_shift))
        energy_shift_max = float(np.max(energy_shift))

    train_metrics = batch_residual_metrics(driver, batch, delta)
    if method == "rgn":
        reference = "arXiv:2106.10558"
        equation = "[H(theta) - E(theta) S(theta)] xi = grad"
        tau_source = "current_energy"
    else:
        reference = "arXiv:2507.10835"
        equation = "[H(theta) - tau S(theta)] xi = grad"
        tau_source = "fixed"
    metadata = {
        "method": method,
        "reference": reference,
        "equation": equation,
        "h_estimator": "symmetrized_energy_shifted_ntk",
        "tau_source": tau_source,
        "tau": float(tau),
        "pii_diag_shift": float(diag_shift),
        "matrix_shape": matrix_shape,
        "energy_shift_min": energy_shift_min,
        "energy_shift_max": energy_shift_max,
        "delta_l2_norm": float(tree_l2_norm(delta)),
        "train_metrics": {key: float(value) for key, value in train_metrics.items()},
    }
    return delta, metadata


def compute_pii_update(
    driver,
    *,
    tau: float,
    diag_shift: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    return _compute_energy_shifted_update(
        driver,
        tau=float(tau),
        diag_shift=diag_shift,
        method="pii",
    )


def compute_rgn_update(
    driver,
    *,
    diag_shift: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    batch = current_cached_batch(driver)
    tau = current_energy_tau(batch.local_loss, batch.weights)
    return _compute_energy_shifted_update(
        driver,
        tau=tau,
        diag_shift=diag_shift,
        method="rgn",
        batch=batch,
    )


class PIIUpdateCallback(AbstractCallback):
    tau: float = 0.0
    diag_shift: float = 0.0
    order: int = -100

    def __init__(self, *, tau: float, diag_shift: float = 0.0, order: int = -100):
        self.tau = float(tau)
        self.diag_shift = float(diag_shift)
        self.order = int(order)

    @property
    def callback_order(self) -> int:
        return self.order

    def on_compute_update_end(self, step, log_data, driver):
        delta, metadata = compute_pii_update(
            driver,
            tau=self.tau,
            diag_shift=self.diag_shift,
        )
        driver._dp = delta
        info = {} if getattr(driver, "info", None) is None else dict(driver.info)
        info["pii"] = metadata
        driver.info = info
        log_data["ngd_method"] = "pii"
        log_data["pii_tau"] = metadata["tau"]
        log_data["pii_diag_shift"] = metadata["pii_diag_shift"]


class RGNUpdateCallback(AbstractCallback):
    diag_shift: float = 0.0
    order: int = -100

    def __init__(self, *, diag_shift: float = 0.0, order: int = -100):
        self.diag_shift = float(diag_shift)
        self.order = int(order)

    @property
    def callback_order(self) -> int:
        return self.order

    def on_compute_update_end(self, step, log_data, driver):
        delta, metadata = compute_rgn_update(
            driver,
            diag_shift=self.diag_shift,
        )
        driver._dp = delta
        info = {} if getattr(driver, "info", None) is None else dict(driver.info)
        info["rgn"] = metadata
        driver.info = info
        log_data["ngd_method"] = "rgn"
        log_data["pii_tau"] = metadata["tau"]
        log_data["pii_diag_shift"] = metadata["pii_diag_shift"]
