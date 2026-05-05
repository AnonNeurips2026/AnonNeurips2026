from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as fcore
from netket.stats import statistics
from scipy.optimize import minimize

from advanced_drivers._src.callbacks.base import AbstractCallback
from sr_filtering_nqs.large_scale.core.common import (
    SR_VARIANTS,
    SR_WEIGHT_SCHEMES,
    parse_sr_lambda_grid,
)
from sr_filtering_nqs.large_scale.core.multidelta_metrics import center_samples
from sr_filtering_nqs.large_scale.core import sr_diagnostics


GCV_DENOM_EPS = 1e-8
OBJECTIVE_PENALTY = 1e24
SPECTRUM_FLOOR_REL = 1e-12


@dataclass
class PreparedBatch:
    afun: Any
    variables: Any
    samples: Any
    weights: Any
    extra_args: tuple[Any, ...]
    local_grad: Any
    local_loss: Any
    n_replicas: int | None

    @property
    def sample_count(self) -> int:
        return int(np.asarray(self.samples).shape[0])


@dataclass
class BaseSolveRecord:
    lambda_value: float
    delta: Any
    info: dict[str, Any]
    train_metrics: dict[str, float]
    training_prediction: np.ndarray | None = None
    training_target: np.ndarray | None = None
    ntk_eigenvalues: np.ndarray | None = None


def validate_sr_configuration(sr_variant: str, sr_weight_scheme: str) -> None:
    if sr_variant not in SR_VARIANTS:
        raise ValueError(f"Unsupported sr_variant={sr_variant!r}")
    if sr_weight_scheme not in SR_WEIGHT_SCHEMES:
        raise ValueError(f"Unsupported sr_weight_scheme={sr_weight_scheme!r}")
    if sr_variant == "single" and sr_weight_scheme != "uniform":
        raise ValueError("sr_variant='single' only supports sr_weight_scheme='uniform'")
    if sr_variant == "same_batch" and sr_weight_scheme == "stacking":
        raise ValueError("same_batch only supports uniform or gcv weights")


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    vector = np.asarray(weights, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("Cannot project an empty vector onto the simplex")
    if np.all(vector >= 0.0) and math.isclose(float(vector.sum()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return vector

    u = np.sort(vector)[::-1]
    cssv = np.cumsum(u) - 1.0
    rho = np.nonzero(u - cssv / np.arange(1, vector.size + 1) > 0.0)[0]
    if rho.size == 0:
        return np.full(vector.size, 1.0 / float(vector.size), dtype=np.float64)
    theta = cssv[rho[-1]] / float(rho[-1] + 1)
    projected = np.maximum(vector - theta, 0.0)
    total = float(projected.sum())
    if total <= 0.0:
        return np.full(vector.size, 1.0 / float(vector.size), dtype=np.float64)
    return projected / total


def uniform_mixture_weights(count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    return np.full(int(count), 1.0 / float(count), dtype=np.float64)


def spectrum_quantile_lambda_grid(
    spectrum,
    *,
    quantiles: tuple[float, ...],
    floor_rel: float = SPECTRUM_FLOOR_REL,
) -> tuple[float, ...]:
    eigenvalues = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    positive = np.clip(eigenvalues, 0.0, None)
    positive = positive[positive > 0.0]
    if positive.size == 0:
        raise ValueError("The NTK spectrum has no strictly positive eigenvalues")

    lambda_max = float(np.max(positive))
    min_positive = float(np.min(positive))
    floor = max(min_positive, float(floor_rel) * lambda_max)
    values: list[float] = []
    for quantile in quantiles:
        value = float(np.quantile(positive, float(quantile)))
        if not np.isfinite(value) or value <= 0.0:
            value = floor
        values.append(float(max(value, floor)))
    return tuple(values)


def _validate_ema_rho(rho: float) -> float:
    value = float(rho)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"EMA rho must be in [0, 1], got {rho}")
    return value


def ema_positive_vector(raw_values, previous_values, *, rho: float | None) -> tuple[float, ...]:
    raw = np.asarray(raw_values, dtype=np.float64).reshape(-1)
    if np.any(~np.isfinite(raw)) or np.any(raw <= 0.0):
        raise ValueError(f"EMA raw values must be finite and positive, got {raw}")
    if rho is None or previous_values is None:
        return tuple(float(value) for value in raw)
    weight = _validate_ema_rho(rho)
    previous = np.asarray(previous_values, dtype=np.float64).reshape(-1)
    if previous.shape != raw.shape:
        raise ValueError(f"EMA shape mismatch: previous={previous.shape} raw={raw.shape}")
    if np.any(~np.isfinite(previous)) or np.any(previous <= 0.0):
        raise ValueError(f"EMA previous values must be finite and positive, got {previous}")
    smoothed = weight * previous + (1.0 - weight) * raw
    return tuple(float(value) for value in smoothed)


def ema_simplex_weights(raw_weights, previous_weights, *, rho: float | None) -> np.ndarray:
    raw = project_to_simplex(np.asarray(raw_weights, dtype=np.float64).reshape(-1))
    if rho is None or previous_weights is None:
        return raw
    weight = _validate_ema_rho(rho)
    previous = project_to_simplex(np.asarray(previous_weights, dtype=np.float64).reshape(-1))
    if previous.shape != raw.shape:
        raise ValueError(f"EMA weight shape mismatch: previous={previous.shape} raw={raw.shape}")
    return project_to_simplex(weight * previous + (1.0 - weight) * raw)


def spectrum_summary(spectrum) -> dict[str, float | int]:
    values = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        return {
            "count": int(values.size),
            "positive_count": 0,
            "min_positive": float("nan"),
            "max": float("nan"),
            "median_positive": float("nan"),
        }
    return {
        "count": int(values.size),
        "positive_count": int(positive.size),
        "min_positive": float(np.min(positive)),
        "max": float(np.max(positive)),
        "median_positive": float(np.median(positive)),
    }


def _simplex_minimize(objective, dimension: int, *, x0: np.ndarray | None = None) -> np.ndarray:
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    start = (
        project_to_simplex(np.asarray(x0, dtype=np.float64))
        if x0 is not None
        else uniform_mixture_weights(dimension)
    )
    bounds = [(0.0, 1.0)] * int(dimension)
    constraints = (
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones_like(w, dtype=np.float64),
        },
    )
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        if result.x is None:
            raise RuntimeError(f"Simplex optimization failed: {result.message}")
        return project_to_simplex(np.asarray(result.x, dtype=np.float64))
    return project_to_simplex(np.asarray(result.x, dtype=np.float64))


def compute_stacking_mixture_weights(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Solve simplex-constrained least-squares weights for an ensemble prediction matrix."""
    targets = np.asarray(y, dtype=np.float64).reshape(-1)
    predictions = np.asarray(Z, dtype=np.float64)
    if predictions.ndim != 2:
        raise ValueError(f"Expected Z with shape (n_samples, m), got {predictions.shape}")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(
            f"y and Z disagree on sample dimension: {targets.shape[0]} != {predictions.shape[0]}"
        )

    def objective(weights: np.ndarray) -> float:
        residual = targets - predictions @ weights
        return float(np.dot(residual, residual))

    return _simplex_minimize(objective, predictions.shape[1])


def compute_gcv_mixture_weights(
    y: np.ndarray,
    Z_train: np.ndarray,
    mu: np.ndarray,
    lambdas: np.ndarray,
) -> np.ndarray:
    """Return simplex-constrained GCV-optimal mixture weights for same-batch ridge updates."""
    targets = np.asarray(y, dtype=np.float64).reshape(-1)
    predictions = np.asarray(Z_train, dtype=np.float64)
    eigenvalues = np.asarray(mu, dtype=np.float64).reshape(-1)
    penalties = np.asarray(lambdas, dtype=np.float64).reshape(-1)

    if predictions.ndim != 2:
        raise ValueError(f"Expected Z_train with shape (N_s, m), got {predictions.shape}")
    if predictions.shape[0] != targets.shape[0]:
        raise ValueError(
            f"y and Z_train disagree on sample dimension: {targets.shape[0]} != {predictions.shape[0]}"
        )
    if predictions.shape[1] != penalties.shape[0]:
        raise ValueError(
            f"Z_train and lambdas disagree on model dimension: {predictions.shape[1]} != {penalties.shape[0]}"
        )
    if eigenvalues.size == 0:
        raise ValueError("mu must contain at least one empirical NTK eigenvalue")
    if np.any(penalties <= 0.0):
        raise ValueError(f"All lambdas must be positive, got {penalties}")

    ns = float(targets.shape[0])
    df_columns = (
        eigenvalues[:, None] / (eigenvalues[:, None] + penalties[None, :])
    ).sum(axis=0)

    def objective(weights: np.ndarray) -> float:
        residual = targets - predictions @ weights
        df = float(np.dot(df_columns, weights))
        denom = 1.0 - df / ns
        if denom <= GCV_DENOM_EPS:
            return OBJECTIVE_PENALTY + (GCV_DENOM_EPS - denom) ** 2
        numer = float(np.dot(residual, residual) / ns)
        return numer / (denom**2)

    return _simplex_minimize(objective, predictions.shape[1])


def numpy_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def jax_tree(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


def stack_tree(pytrees):
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *pytrees)


def weighted_tree_sum(pytrees, weights: np.ndarray):
    coeffs = np.asarray(weights, dtype=np.float64)
    if coeffs.ndim != 1:
        raise ValueError(f"Expected 1D weights, got shape {coeffs.shape}")
    return jax.tree_util.tree_map(
        lambda *xs: jnp.asarray(np.tensordot(coeffs, np.stack([np.asarray(x) for x in xs], axis=0), axes=(0, 0))),
        *pytrees,
    )


def tree_l2_norm(delta) -> float:
    leaves = [np.asarray(leaf, dtype=np.float64).ravel() for leaf in jax.tree_util.tree_leaves(delta)]
    if not leaves:
        return 0.0
    flat = np.concatenate(leaves, axis=0)
    return float(np.linalg.norm(flat))


def _flatten_samples(samples):
    array = jnp.asarray(samples)
    return jax.lax.collapse(array, 0, array.ndim - 1)


def _resolve_distribution(driver, variables):
    state = driver.state
    distribution = getattr(driver, "original_distribution", None)
    if distribution is None:
        return state._apply_fun, variables
    return distribution(state._apply_fun, variables)


def _weights_for_samples(samples, *, dtype) -> Any:
    return jnp.ones((int(np.asarray(samples).shape[0]),), dtype=dtype)


def _prepared_batch_from_samples(driver, samples, *, variables=None) -> PreparedBatch:
    state = driver.state
    variables = state.variables if variables is None else variables
    afun, variables = _resolve_distribution(driver, variables)
    samples_flat = _flatten_samples(samples)
    local_grad, local_loss = driver._kernel(
        afun,
        variables,
        samples_flat,
        driver._ham,
    )
    local_grad_array = jnp.asarray(local_grad)
    weights = _weights_for_samples(local_grad_array, dtype=local_grad_array.dtype)
    n_replicas = getattr(state, "_n_replicas", None)
    if n_replicas is not None:
        n_replicas = int(n_replicas)
    return PreparedBatch(
        afun=afun,
        variables=variables,
        samples=samples_flat,
        weights=weights,
        extra_args=(driver._ham,),
        local_grad=local_grad,
        local_loss=local_loss,
        n_replicas=n_replicas,
    )


def current_cached_batch(driver) -> PreparedBatch:
    return _prepared_batch_from_samples(driver, driver.state.samples)


def sample_prepared_batch(driver, *, n_samples: int | None = None, chain_name: str | None = None) -> PreparedBatch:
    state = driver.state
    variables = state.variables
    afun, variables = _resolve_distribution(driver, variables)
    if n_samples is None:
        samples = state.samples_distribution(
            afun,
            variables=variables,
            chain_name=chain_name,
        )
    else:
        samples = state.sample_distribution(
            afun,
            variables=variables,
            n_samples=int(n_samples),
            chain_name=chain_name,
        )
    return _prepared_batch_from_samples(driver, samples, variables=variables)


def _channel_array(values, *, mode: str, sample_count: int) -> np.ndarray:
    vector = np.asarray(values)
    if mode == "complex":
        return np.stack(
            [
                np.asarray(np.real(vector[:sample_count]), dtype=np.float64),
                np.asarray(np.real(vector[sample_count : 2 * sample_count]), dtype=np.float64),
            ],
            axis=0,
        )
    return np.asarray(np.real(vector[:sample_count]), dtype=np.float64)[None, :]


def centered_target_vector(local_loss, *, mode: str, n_replicas: int | None) -> tuple[np.ndarray, float]:
    local_loss = np.asarray(local_loss)
    if mode == "complex":
        channels = np.stack(
            [
                np.asarray(np.real(local_loss).reshape(-1), dtype=np.float64),
                np.asarray(np.imag(local_loss).reshape(-1), dtype=np.float64),
            ],
            axis=0,
        )
    else:
        channels = np.asarray(np.real(local_loss).reshape(-1), dtype=np.float64)[None, :]

    centered, total_count = center_samples(channels, n_replicas=n_replicas)
    centered_flat = centered.reshape(channels.shape[0], -1)
    energy_variance = float(np.square(centered).sum() / float(total_count))
    if mode == "complex":
        target = 2.0 * np.concatenate([centered_flat[0], centered_flat[1]], axis=0) / math.sqrt(float(total_count))
    else:
        target = 2.0 * centered_flat[0] / math.sqrt(float(total_count))
    return target, energy_variance


def centered_prediction_vector(
    driver,
    batch: PreparedBatch,
    delta,
) -> np.ndarray:
    model_state, params = fcore.pop(batch.variables, "params")
    prediction = sr_diagnostics.compute_O_delta_jvp(
        batch.afun,
        params,
        model_state,
        jnp.asarray(batch.samples),
        delta,
        mode=driver.mode,
        chunk_size=driver.chunk_size_bwd,
    )
    channels = _channel_array(
        np.asarray(prediction),
        mode=driver.mode,
        sample_count=batch.sample_count,
    )
    centered, total_count = center_samples(channels, n_replicas=batch.n_replicas)
    centered_flat = centered.reshape(channels.shape[0], -1)
    if driver.mode == "complex":
        return np.concatenate([centered_flat[0], centered_flat[1]], axis=0) / math.sqrt(float(total_count))
    return centered_flat[0] / math.sqrt(float(total_count))


def batch_residual_metrics(
    driver,
    batch: PreparedBatch,
    delta,
) -> dict[str, float]:
    target_vector, energy_variance = centered_target_vector(
        batch.local_loss,
        mode=driver.mode,
        n_replicas=batch.n_replicas,
    )
    prediction_vector = centered_prediction_vector(driver, batch, delta)
    residual = prediction_vector - target_vector
    residual_raw = float(np.dot(residual, residual))
    target_norm_sq = float(np.dot(target_vector, target_vector))
    residual_norm = residual_raw / target_norm_sq if target_norm_sq > 1e-12 else float("inf")
    return {
        "residual_raw": residual_raw,
        "residual_norm": residual_norm,
        "target_norm_sq": target_norm_sq,
        "energy_variance": float(energy_variance),
    }


def _resolve_driver_scalar(value, *, step: int):
    if callable(value):
        return value(step)
    return value


def solve_sr_on_batch(
    driver,
    batch: PreparedBatch,
    *,
    diag_shift: float,
    collect_residual_info: bool,
):
    local_grad = batch.local_grad
    variables = batch.variables
    model_state, params = fcore.pop(variables, "params")

    proj_reg = _resolve_driver_scalar(getattr(driver, "proj_reg", None), step=driver.step_count)
    momentum = _resolve_driver_scalar(getattr(driver, "momentum", None), step=driver.step_count)

    common_kwargs = {
        "diag_shift": float(diag_shift),
        "solver_fn": driver._linear_solver_fn,
        "mode": driver.mode,
        "proj_reg": proj_reg,
        "momentum": momentum,
        "old_updates": driver._old_updates,
        "chunk_size": driver.chunk_size_bwd,
        "collect_quadratic_model": False,
        "collect_gradient_statistics": False,
        "collect_residual_info": collect_residual_info,
    }

    def _is_signature_mismatch(exc: TypeError) -> bool:
        text = str(exc)
        return (
            "unexpected keyword argument" in text
            or "missing 1 required keyword-only argument: 'n_replicas'" in text
        )

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
                delta, _, info = driver.update_fn(
                    batch.afun,
                    local_grad,
                    params,
                    model_state,
                    batch.samples,
                    **weight_kwargs,
                    **replica_kwargs,
                    **common_kwargs,
                )
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


def _prediction_from_residual_info(info: dict[str, Any]) -> np.ndarray | None:
    aux_vector = info.get("aux_vector")
    ntk_matrix = info.get("ntk_matrix")
    if aux_vector is None or ntk_matrix is None:
        return None
    return np.asarray(ntk_matrix @ aux_vector, dtype=np.float64)


def _eigenvalues_from_residual_info(info: dict[str, Any]) -> np.ndarray | None:
    ntk_matrix = info.get("ntk_matrix")
    if ntk_matrix is None:
        return None
    eigvals = np.linalg.eigvalsh(np.asarray(ntk_matrix, dtype=np.float64))
    return eigvals[::-1]


def base_solve_record(
    driver,
    batch: PreparedBatch,
    *,
    diag_shift: float,
    collect_residual_info: bool,
) -> BaseSolveRecord:
    delta, _, info = solve_sr_on_batch(
        driver,
        batch,
        diag_shift=diag_shift,
        collect_residual_info=collect_residual_info,
    )
    train_metrics = batch_residual_metrics(driver, batch, delta)
    return BaseSolveRecord(
        lambda_value=float(diag_shift),
        delta=delta,
        info=info,
        train_metrics=train_metrics,
        training_prediction=_prediction_from_residual_info(info),
        training_target=(
            None if info.get("dv") is None else np.asarray(info["dv"], dtype=np.float64)
        ),
        ntk_eigenvalues=_eigenvalues_from_residual_info(info),
    )


def snapshot_driver_state(driver) -> dict[str, Any]:
    state = driver.state
    return {
        "driver_dp": getattr(driver, "_dp", None),
        "driver_old_updates": getattr(driver, "_old_updates", None),
        "driver_loss_stats": getattr(driver, "_loss_stats", None),
        "driver_info": getattr(driver, "info", None),
        "driver_collect_residual_info": getattr(driver, "collect_residual_info", None),
        "vstate_samples": copy.copy(getattr(state, "_samples", None)),
        "vstate_sampler_state": copy.copy(getattr(state, "sampler_state", None)),
        "vstate_sampler_state_previous": copy.copy(getattr(state, "_sampler_state_previous", None)),
        "vstate_sampler_states": copy.copy(getattr(state, "sampler_states", {})),
        "vstate_sampler_states_previous": copy.copy(getattr(state, "_sampler_states_previous", {})),
        "vstate_samples_distributions": copy.copy(getattr(state, "_samples_distributions", {})),
        "vstate_log_probabilities_distributions": copy.copy(
            getattr(state, "_log_probabilities_distributions", {})
        ),
        "vstate_samples_distribution_resampling_cache": copy.copy(
            getattr(state, "_samples_distribution_resampling_cache", {})
        ),
        "vstate_log_probabilities_distribution_resampling_cache": copy.copy(
            getattr(state, "_log_probabilities_distribution_resampling_cache", {})
        ),
        "vstate_sampler_seed": copy.copy(getattr(state, "_sampler_seed", None)),
    }


def restore_driver_state(driver, snapshot: dict[str, Any]) -> None:
    state = driver.state
    driver._dp = snapshot["driver_dp"]
    driver._old_updates = snapshot["driver_old_updates"]
    driver._loss_stats = snapshot["driver_loss_stats"]
    if hasattr(driver, "info"):
        driver.info = snapshot["driver_info"]
    if snapshot["driver_collect_residual_info"] is not None:
        driver.collect_residual_info = snapshot["driver_collect_residual_info"]

    state._samples = snapshot["vstate_samples"]
    if hasattr(state, "sampler_state"):
        state.sampler_state = snapshot["vstate_sampler_state"]
    if hasattr(state, "_sampler_state_previous"):
        state._sampler_state_previous = snapshot["vstate_sampler_state_previous"]
    if hasattr(state, "sampler_states"):
        state.sampler_states = snapshot["vstate_sampler_states"]
    if hasattr(state, "_sampler_states_previous"):
        state._sampler_states_previous = snapshot["vstate_sampler_states_previous"]
    if hasattr(state, "_samples_distributions"):
        state._samples_distributions = snapshot["vstate_samples_distributions"]
    if hasattr(state, "_log_probabilities_distributions"):
        state._log_probabilities_distributions = snapshot["vstate_log_probabilities_distributions"]
    if hasattr(state, "_samples_distribution_resampling_cache"):
        state._samples_distribution_resampling_cache = snapshot[
            "vstate_samples_distribution_resampling_cache"
        ]
    if hasattr(state, "_log_probabilities_distribution_resampling_cache"):
        state._log_probabilities_distribution_resampling_cache = snapshot[
            "vstate_log_probabilities_distribution_resampling_cache"
        ]
    if hasattr(state, "_sampler_seed"):
        state._sampler_seed = snapshot["vstate_sampler_seed"]


def _same_batch_records(
    driver,
    lambda_grid: tuple[float, ...],
    *,
    current_batch: PreparedBatch | None = None,
):
    batch = current_batch if current_batch is not None else current_cached_batch(driver)
    return [
        base_solve_record(
            driver,
            batch,
            diag_shift=lam,
            collect_residual_info=True,
        )
        for lam in lambda_grid
    ], batch


def _independent_batch_records(
    driver,
    lambda_grid: tuple[float, ...],
    *,
    n_samples: int | None = None,
):
    records = []
    batches = []
    for index, lam in enumerate(lambda_grid):
        batch = sample_prepared_batch(
            driver,
            n_samples=n_samples,
            chain_name=f"multilambda_fit_{index}",
        )
        records.append(
            base_solve_record(
                driver,
                batch,
                diag_shift=lam,
                collect_residual_info=False,
            )
        )
        batches.append(batch)
    return records, batches


def _gcv_weights_from_records(records: list[BaseSolveRecord]) -> np.ndarray:
    reference = records[0]
    if reference.training_target is None or reference.ntk_eigenvalues is None:
        raise RuntimeError("GCV weights require residual-info targets and NTK eigenvalues")
    Z_train = np.column_stack([record.training_prediction for record in records])
    lambdas = np.asarray([record.lambda_value for record in records], dtype=np.float64)
    return compute_gcv_mixture_weights(
        reference.training_target,
        Z_train,
        reference.ntk_eigenvalues,
        lambdas,
    )


def _same_batch_weights(records: list[BaseSolveRecord], *, weight_scheme: str) -> np.ndarray:
    if weight_scheme == "uniform":
        return uniform_mixture_weights(len(records))
    if weight_scheme != "gcv":
        raise ValueError(f"Unsupported same-batch weight scheme: {weight_scheme}")
    return _gcv_weights_from_records(records)


def _independent_batch_weights(
    driver,
    records: list[BaseSolveRecord],
    *,
    weight_scheme: str,
    weight_fit_samples: int,
    pilot_batch_samples: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if weight_scheme == "uniform":
        return uniform_mixture_weights(len(records)), {
            "weight_fit_samples": 0,
            "weight_fit_target_norm_sq": float("nan"),
            "weight_fit_energy_variance": float("nan"),
        }
    if weight_scheme == "gcv":
        pilot_batch = sample_prepared_batch(
            driver,
            n_samples=int(pilot_batch_samples),
            chain_name="multilambda_gcv_pilot",
        )
        pilot_records = [
            base_solve_record(
                driver,
                pilot_batch,
                diag_shift=float(record.lambda_value),
                collect_residual_info=True,
            )
            for record in records
        ]
        return _gcv_weights_from_records(pilot_records), {
            "weight_fit_samples": int(pilot_batch.sample_count),
            "weight_fit_target_norm_sq": float(
                pilot_records[0].train_metrics["target_norm_sq"]
                if pilot_records
                else float("nan")
            ),
            "weight_fit_energy_variance": float(
                pilot_records[0].train_metrics["energy_variance"]
                if pilot_records
                else float("nan")
            ),
        }
    if weight_scheme != "stacking":
        raise ValueError(f"Unsupported independent-batch weight scheme: {weight_scheme}")
    weight_batch = sample_prepared_batch(
        driver,
        n_samples=int(weight_fit_samples),
        chain_name="multilambda_weight_fit",
    )
    target_vector, energy_variance = centered_target_vector(
        weight_batch.local_loss,
        mode=driver.mode,
        n_replicas=weight_batch.n_replicas,
    )
    prediction_matrix = np.column_stack(
        [centered_prediction_vector(driver, weight_batch, record.delta) for record in records]
    )
    weights = compute_stacking_mixture_weights(target_vector, prediction_matrix)
    return weights, {
        "weight_fit_samples": int(weight_batch.sample_count),
        "weight_fit_target_norm_sq": float(np.dot(target_vector, target_vector)),
        "weight_fit_energy_variance": float(energy_variance),
    }


def compute_multilambda_update(
    driver,
    *,
    sr_variant: str,
    sr_weight_scheme: str,
    sr_lambda_grid,
    sr_weight_fit_samples: int | None = None,
    sr_lambda_quantiles=None,
    sr_lambda_ema_rho: float | None = None,
    sr_weight_ema_rho: float | None = None,
    previous_lambda_grid=None,
    previous_weights=None,
) -> tuple[Any, dict[str, Any]]:
    validate_sr_configuration(sr_variant, sr_weight_scheme)
    lambda_grid = tuple(float(value) for value in parse_sr_lambda_grid(sr_lambda_grid))
    lambda_quantiles = (
        None
        if sr_lambda_quantiles is None
        else tuple(float(value) for value in parse_sr_lambda_grid(sr_lambda_quantiles))
    )

    if sr_variant == "single":
        metadata = {
            "variant": "single",
            "weight_scheme": "uniform",
            "lambda_grid": [float(getattr(driver, "diag_shift"))],
            "weights": [1.0],
            "weight_fit_samples": 0,
            "per_lambda": [],
        }
        return driver._dp, metadata

    weight_fit_samples = (
        int(sr_weight_fit_samples)
        if sr_weight_fit_samples is not None
        else int(np.asarray(current_cached_batch(driver).samples).shape[0])
    )

    snapshot = snapshot_driver_state(driver)
    current_batch = current_cached_batch(driver)

    try:
        lambda_meta: dict[str, Any] = {
            "lambda_source": "static",
            "lambda_grid_raw": [float(value) for value in lambda_grid],
            "lambda_quantiles": [],
            "lambda_ema_rho": None,
            "lambda_grid_previous": [],
            "ntk_spectrum_summary": {},
        }
        if lambda_quantiles is not None:
            probe_record = base_solve_record(
                driver,
                current_batch,
                diag_shift=max(lambda_grid),
                collect_residual_info=True,
            )
            if probe_record.ntk_eigenvalues is None:
                raise RuntimeError("Dynamic NTK-quantile lambda grid requires NTK residual info")
            raw_lambda_grid = spectrum_quantile_lambda_grid(
                probe_record.ntk_eigenvalues,
                quantiles=lambda_quantiles,
            )
            lambda_grid = ema_positive_vector(
                raw_lambda_grid,
                previous_lambda_grid,
                rho=sr_lambda_ema_rho,
            )
            lambda_meta = {
                "lambda_source": "ntk_quantile",
                "lambda_grid_raw": [float(value) for value in raw_lambda_grid],
                "lambda_quantiles": [float(value) for value in lambda_quantiles],
                "lambda_ema_rho": None if sr_lambda_ema_rho is None else float(sr_lambda_ema_rho),
                "lambda_grid_previous": (
                    []
                    if previous_lambda_grid is None
                    else [float(value) for value in np.asarray(previous_lambda_grid, dtype=np.float64).reshape(-1)]
                ),
                "ntk_spectrum_summary": spectrum_summary(probe_record.ntk_eigenvalues),
            }

        if sr_variant == "same_batch":
            records, _ = _same_batch_records(driver, lambda_grid, current_batch=current_batch)
            weights = _same_batch_weights(records, weight_scheme=sr_weight_scheme)
            weight_fit_meta = {
                "weight_fit_samples": int(current_batch.sample_count),
                "weight_fit_target_norm_sq": float(
                    records[0].train_metrics["target_norm_sq"]
                    if records
                    else float("nan")
                ),
                "weight_fit_energy_variance": float(
                    records[0].train_metrics["energy_variance"]
                    if records
                    else float("nan")
                ),
            }
        else:
            records, _ = _independent_batch_records(
                driver,
                lambda_grid,
                n_samples=int(current_batch.sample_count),
            )
            weights, weight_fit_meta = _independent_batch_weights(
                driver,
                records,
                weight_scheme=sr_weight_scheme,
                weight_fit_samples=weight_fit_samples,
                pilot_batch_samples=int(current_batch.sample_count),
            )

        raw_weights = np.asarray(weights, dtype=np.float64)
        weights = ema_simplex_weights(
            raw_weights,
            previous_weights,
            rho=sr_weight_ema_rho,
        )
        mixed_delta = weighted_tree_sum([record.delta for record in records], weights)
        train_metrics = batch_residual_metrics(driver, current_batch, mixed_delta)
    finally:
        restore_driver_state(driver, snapshot)

    metadata = {
        "variant": sr_variant,
        "weight_scheme": sr_weight_scheme,
        "lambda_grid": [float(value) for value in lambda_grid],
        "weights": [float(value) for value in np.asarray(weights, dtype=np.float64)],
        "weights_raw": [float(value) for value in np.asarray(raw_weights, dtype=np.float64)],
        "weight_ema_rho": None if sr_weight_ema_rho is None else float(sr_weight_ema_rho),
        "weights_previous": (
            []
            if previous_weights is None
            else [float(value) for value in np.asarray(previous_weights, dtype=np.float64).reshape(-1)]
        ),
        "weight_fit_samples": int(weight_fit_meta["weight_fit_samples"]),
        "train_metrics": {
            key: float(value) for key, value in train_metrics.items()
        },
        "mixed_delta_l2_norm": float(tree_l2_norm(mixed_delta)),
        "per_lambda": [
            {
                "lambda": float(record.lambda_value),
                "delta_l2_norm": float(tree_l2_norm(record.delta)),
                "train_residual_raw": float(record.train_metrics["residual_raw"]),
                "train_residual_norm": float(record.train_metrics["residual_norm"]),
                "train_target_norm_sq": float(record.train_metrics["target_norm_sq"]),
                "energy_variance": float(record.train_metrics["energy_variance"]),
            }
            for record in records
        ],
        "weight_fit_meta": {
            key: float(value) for key, value in weight_fit_meta.items()
        },
        **lambda_meta,
    }
    return mixed_delta, metadata


class MultiLambdaSRCallback(AbstractCallback):
    sr_variant: str = "single"
    sr_weight_scheme: str = "uniform"
    sr_lambda_grid: tuple[float, ...] = ()
    sr_weight_fit_samples: int | None = None
    sr_lambda_quantiles: tuple[float, ...] | None = None
    sr_lambda_ema_rho: float | None = None
    sr_weight_ema_rho: float | None = None
    _previous_lambda_grid: tuple[float, ...] | None = None
    _previous_weights: tuple[float, ...] | None = None
    order: int = -100

    def __init__(
        self,
        *,
        sr_variant: str,
        sr_weight_scheme: str,
        sr_lambda_grid,
        sr_weight_fit_samples: int | None = None,
        sr_lambda_quantiles=None,
        sr_lambda_ema_rho: float | None = None,
        sr_weight_ema_rho: float | None = None,
        order: int = -100,
    ):
        validate_sr_configuration(sr_variant, sr_weight_scheme)
        self.sr_variant = sr_variant
        self.sr_weight_scheme = sr_weight_scheme
        self.sr_lambda_grid = tuple(parse_sr_lambda_grid(sr_lambda_grid))
        self.sr_weight_fit_samples = (
            None if sr_weight_fit_samples is None else int(sr_weight_fit_samples)
        )
        self.sr_lambda_quantiles = (
            None
            if sr_lambda_quantiles is None
            else tuple(parse_sr_lambda_grid(sr_lambda_quantiles))
        )
        self.sr_lambda_ema_rho = (
            None if sr_lambda_ema_rho is None else _validate_ema_rho(sr_lambda_ema_rho)
        )
        self.sr_weight_ema_rho = (
            None if sr_weight_ema_rho is None else _validate_ema_rho(sr_weight_ema_rho)
        )
        self._previous_lambda_grid = None
        self._previous_weights = None
        self.order = int(order)

    @property
    def callback_order(self) -> int:
        return self.order

    def on_compute_update_end(self, step, log_data, driver):
        if self.sr_variant == "single":
            return
        mixed_delta, metadata = compute_multilambda_update(
            driver,
            sr_variant=self.sr_variant,
            sr_weight_scheme=self.sr_weight_scheme,
            sr_lambda_grid=self.sr_lambda_grid,
            sr_weight_fit_samples=self.sr_weight_fit_samples,
            sr_lambda_quantiles=self.sr_lambda_quantiles,
            sr_lambda_ema_rho=self.sr_lambda_ema_rho,
            sr_weight_ema_rho=self.sr_weight_ema_rho,
            previous_lambda_grid=self._previous_lambda_grid,
            previous_weights=self._previous_weights,
        )
        self._previous_lambda_grid = tuple(float(value) for value in metadata["lambda_grid"])
        self._previous_weights = tuple(float(value) for value in metadata["weights"])
        driver._dp = mixed_delta
        info = {} if getattr(driver, "info", None) is None else dict(driver.info)
        info["multilambda"] = metadata
        driver.info = info
        log_data["multilambda_weight_scheme"] = metadata["weight_scheme"]
        log_data["multilambda_variant"] = metadata["variant"]
        log_data["multilambda_lambda_source"] = metadata["lambda_source"]
