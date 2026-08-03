"""Paper-faithful, diagnostic-free SR step primitives for cost benchmarking.

The production multi-lambda callback was written for experimentation and adds
an ordinary SR solve, training-residual diagnostics, and (for stacking) a
separate weight-fit batch.  Those operations make it unsuitable for measuring
the algorithmic cost stated in the paper.  This module implements the three
update constructors needed by the rebuttal cost benchmark:

* one-batch SR;
* K-batch fixed-shift bagged SR;
* K-batch spectrum-quantile MS-SR with exact leave-one-batch-out stacking.

All three paths omit optional residual diagnostics.  MS-SR reuses the K solve
batches and their local energies, fuses the first NTK eigendecomposition with
the first shifted solve, and evaluates only the K(K-1) off-diagonal LOO
predictions in candidate-batched JVP calls.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as fcore
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

from sr_filtering_nqs.large_scale.core import sr_diagnostics
from sr_filtering_nqs.large_scale.core.multilambda_sr import PreparedBatch, sample_prepared_batch


DEFAULT_QUANTILES = (0.9, 0.7, 0.4, 0.1)
LOO_DENOM_EPS = 1e-12
OBJECTIVE_PENALTY = 1e24
SPECTRUM_FLOOR_REL = 1e-12


@dataclass(frozen=True)
class OptimizedStepResult:
    """One synchronized update and its component timings."""

    delta: Any
    timings: dict[str, float]
    metadata: dict[str, Any]


def _eigh_solve_from_factors(eigenvalues, eigenvectors, rhs, *, rtol: float):
    scale = jnp.max(jnp.abs(eigenvalues))
    inverse = jnp.where(
        jnp.abs(eigenvalues) > float(rtol) * scale,
        jnp.reciprocal(eigenvalues),
        0.0,
    )
    return eigenvectors @ (inverse * (eigenvectors.conj().T @ rhs))


def eigh_pinv_solver(A, b, *, rtol: float = 1e-12):
    """Hermitian pseudoinverse solver used identically by all benchmark arms."""

    rhs, unravel = ravel_pytree(b)
    eigenvalues, eigenvectors = jnp.linalg.eigh(A)
    solution = _eigh_solve_from_factors(
        eigenvalues,
        eigenvectors,
        rhs,
        rtol=rtol,
    )
    return unravel(solution), None


def _positive_spectrum_quantiles(
    eigenvalues,
    *,
    quantiles: Sequence[float],
    floor_rel: float,
):
    positive = jnp.where(eigenvalues > 0.0, eigenvalues, jnp.nan)
    values = jnp.nanquantile(
        positive,
        jnp.asarray(tuple(quantiles), dtype=eigenvalues.dtype),
    )
    minimum = jnp.nanmin(positive)
    maximum = jnp.nanmax(positive)
    floor = jnp.maximum(minimum, float(floor_rel) * maximum)
    return jnp.maximum(values, floor)


def spectrum_quantile_eigh_pinv_solver(
    A,
    b,
    *,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    rtol: float = 1e-12,
    floor_rel: float = SPECTRUM_FLOOR_REL,
):
    """Choose the shift grid and solve candidate 1 with one decomposition.

    ``A`` is the unshifted empirical NTK.  Its eigendecomposition supplies the
    positive-spectrum quantiles and is immediately reused to solve with the
    first quantile shift.  Returning only the four shifts avoids retaining or
    transferring the full spectrum after the solve.
    """

    rhs, unravel = ravel_pytree(b)
    eigenvalues, eigenvectors = jnp.linalg.eigh(A)
    lambda_grid = _positive_spectrum_quantiles(
        eigenvalues,
        quantiles=quantiles,
        floor_rel=floor_rel,
    )
    shifted_eigenvalues = eigenvalues + lambda_grid[0]
    solution = _eigh_solve_from_factors(
        shifted_eigenvalues,
        eigenvectors,
        rhs,
        rtol=rtol,
    )
    return unravel(solution), {"lambda_grid": lambda_grid}


def make_spectrum_quantile_solver(
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    *,
    rtol: float = 1e-12,
):
    return partial(
        spectrum_quantile_eigh_pinv_solver,
        quantiles=tuple(float(value) for value in quantiles),
        rtol=float(rtol),
    )


def _synchronize_tree(tree) -> None:
    jax.block_until_ready(jax.tree_util.tree_leaves(tree))


def _driver_scalar(value, *, step: int):
    return value(step) if callable(value) else value


def solve_prepared_batch(
    driver,
    batch: PreparedBatch,
    *,
    diag_shift: float,
    solver_fn: Callable,
) -> tuple[Any, dict[str, Any]]:
    """Run one SR solve without loss statistics or residual diagnostics."""

    model_state, params = fcore.pop(batch.variables, "params")
    delta, _, info = driver.update_fn(
        batch.afun,
        batch.local_grad,
        params,
        model_state,
        batch.samples,
        n_replicas=int(batch.n_replicas),
        diag_shift=float(diag_shift),
        solver_fn=solver_fn,
        mode=driver.mode,
        proj_reg=_driver_scalar(getattr(driver, "proj_reg", None), step=driver.step_count),
        momentum=_driver_scalar(getattr(driver, "momentum", None), step=driver.step_count),
        old_updates=driver._old_updates,
        chunk_size=driver.chunk_size_bwd,
        collect_quadratic_model=False,
        collect_gradient_statistics=False,
        collect_residual_info=False,
    )
    _synchronize_tree(delta)
    return delta, {} if info is None else info


def draw_prepared_batches(
    driver,
    count: int,
    *,
    n_samples: int,
    chain_prefix: str,
) -> list[PreparedBatch]:
    batches = []
    for index in range(int(count)):
        batch = sample_prepared_batch(
            driver,
            n_samples=int(n_samples),
            chain_name=f"{chain_prefix}_{index}",
        )
        jax.block_until_ready((batch.samples, batch.local_loss))
        batches.append(batch)
    return batches


def weighted_delta(deltas: Sequence[Any], weights: Sequence[float]):
    coefficients = jnp.asarray(weights)
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.tensordot(
            coefficients.astype(jnp.asarray(leaves[0]).dtype),
            jnp.stack(leaves, axis=0),
            axes=(0, 0),
        ),
        *deltas,
    )


@partial(
    jax.jit,
    static_argnames=("mode", "sample_count", "n_replicas"),
)
def _prediction_sufficient_statistics_on_device(
    prediction_values,
    local_loss,
    *,
    mode: str,
    sample_count: int,
    n_replicas: int | None,
):
    """Center a prediction bank and reduce it before any host transfer."""

    if mode == "complex":
        prediction_channels = jnp.stack(
            [
                jnp.real(prediction_values[:, :sample_count]),
                jnp.real(prediction_values[:, sample_count : 2 * sample_count]),
            ],
            axis=1,
        )
        target_channels = jnp.stack(
            [jnp.real(local_loss).reshape(-1), jnp.imag(local_loss).reshape(-1)],
            axis=0,
        )
    else:
        prediction_channels = jnp.real(prediction_values[:, :sample_count])[:, None, :]
        target_channels = jnp.real(local_loss).reshape(1, -1)

    if n_replicas is not None:
        samples_per_replica = sample_count // int(n_replicas)
        prediction_channels = prediction_channels.reshape(
            prediction_channels.shape[:-1] + (int(n_replicas), samples_per_replica)
        )
        target_channels = target_channels.reshape(
            target_channels.shape[:-1] + (int(n_replicas), samples_per_replica)
        )
    centered_predictions = prediction_channels - jnp.mean(
        prediction_channels,
        axis=-1,
        keepdims=True,
    )
    centered_target = target_channels - jnp.mean(
        target_channels,
        axis=-1,
        keepdims=True,
    )
    predictions = centered_predictions.reshape(prediction_values.shape[0], -1) / math.sqrt(
        float(sample_count)
    )
    target = 2.0 * centered_target.reshape(-1) / math.sqrt(float(sample_count))
    return predictions @ predictions.T, predictions @ target, jnp.dot(target, target)


def _loo_objective_from_sufficient_statistics(
    weights: np.ndarray,
    gram_matrices: np.ndarray,
    target_projections: np.ndarray,
    target_norms: np.ndarray,
) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    total = 0.0
    for heldout in range(weights.size):
        denom = 1.0 - float(weights[heldout])
        if denom <= LOO_DENOM_EPS:
            return OBJECTIVE_PENALTY + (LOO_DENOM_EPS - denom) ** 2
        coefficients = np.delete(weights, heldout) / denom
        total += float(
            coefficients @ gram_matrices[heldout] @ coefficients
            - 2.0 * coefficients @ target_projections[heldout]
            + target_norms[heldout]
        )
    return total


def exact_loo_weights_from_sufficient_statistics(
    gram_matrices: np.ndarray,
    target_projections: np.ndarray,
    target_norms: np.ndarray,
) -> np.ndarray:
    """Solve the exact LOO simplex problem from small sufficient statistics."""

    gram_matrices = np.asarray(gram_matrices, dtype=np.float64)
    target_projections = np.asarray(target_projections, dtype=np.float64)
    target_norms = np.asarray(target_norms, dtype=np.float64)
    count = int(gram_matrices.shape[0])
    initial = np.full(count, 1.0 / float(count), dtype=np.float64)
    result = minimize(
        lambda weights: _loo_objective_from_sufficient_statistics(
            weights,
            gram_matrices,
            target_projections,
            target_norms,
        ),
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=(
            {
                "type": "eq",
                "fun": lambda weights: float(np.sum(weights) - 1.0),
                "jac": lambda weights: np.ones_like(weights, dtype=np.float64),
            },
        ),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"LOO simplex optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    return weights / float(np.sum(weights))


def optimized_exact_loo_weights(
    driver,
    batches: Sequence[PreparedBatch],
    deltas: Sequence[Any],
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit exact LOO weights using K batched calls and K(K-1) directions."""

    count = len(batches)
    if count != len(deltas) or count < 2:
        raise ValueError("LOO stacking requires equally sized batch/delta lists with K >= 2")

    gram_matrices = []
    target_projections = []
    target_norms = []
    prediction_start = time.perf_counter()
    for heldout, batch in enumerate(batches):
        other_deltas = [delta for index, delta in enumerate(deltas) if index != heldout]
        delta_bank = jax.tree_util.tree_map(
            lambda *leaves: jnp.stack(leaves, axis=0),
            *other_deltas,
        )
        model_state, params = fcore.pop(batch.variables, "params")
        values = sr_diagnostics.compute_O_delta_jvp_many(
            batch.afun,
            params,
            model_state,
            jnp.asarray(batch.samples),
            delta_bank,
            mode=driver.mode,
            chunk_size=driver.chunk_size_bwd,
        )
        gram, projection, target_norm = _prediction_sufficient_statistics_on_device(
            values,
            batch.local_loss,
            mode=driver.mode,
            sample_count=batch.sample_count,
            n_replicas=batch.n_replicas,
        )
        gram_matrices.append(gram)
        target_projections.append(projection)
        target_norms.append(target_norm)
    gram_matrices_array, target_projections_array, target_norms_array = jax.block_until_ready(
        (
            jnp.stack(gram_matrices),
            jnp.stack(target_projections),
            jnp.stack(target_norms),
        )
    )
    prediction_seconds = time.perf_counter() - prediction_start

    fit_start = time.perf_counter()
    weights = exact_loo_weights_from_sufficient_statistics(
        np.asarray(gram_matrices_array),
        np.asarray(target_projections_array),
        np.asarray(target_norms_array),
    )
    fit_seconds = time.perf_counter() - fit_start
    return weights, {
        "cross_prediction_s": float(prediction_seconds),
        "simplex_fit_s": float(fit_seconds),
        "cross_prediction_launches": int(count),
        "cross_prediction_directions": int(count * (count - 1)),
    }


def _apply_update(driver, delta) -> None:
    driver._dp = delta
    driver.update_parameters(delta)
    _synchronize_tree(driver.state.parameters)
    driver._step_count += 1


def run_optimized_step(
    driver,
    *,
    method: str,
    n_samples: int,
    k: int = 4,
    diag_shift: float = 1e-4,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    solver_fn: Callable = eigh_pinv_solver,
    spectrum_solver_fn: Callable | None = None,
    chain_prefix: str = "optimized_cost",
    apply_update: bool = True,
) -> OptimizedStepResult:
    """Construct one synchronized SR/bagged/MS-SR update and time components."""

    if method not in {"sr", "bagged", "mssr"}:
        raise ValueError(f"Unknown benchmark method {method!r}")
    if int(k) < 2:
        raise ValueError(f"K must be at least 2, got {k}")
    if method == "mssr" and len(tuple(quantiles)) != int(k):
        raise ValueError("MS-SR needs exactly K spectrum quantiles")
    if getattr(driver, "momentum", None) is not None or getattr(driver, "proj_reg", None) is not None:
        raise ValueError("The fair SR cost benchmark requires plain SR without momentum or proj_reg")
    if spectrum_solver_fn is None:
        spectrum_solver_fn = make_spectrum_quantile_solver(quantiles)

    total_start = time.perf_counter()
    driver.reset_step()
    candidate_count = 1 if method == "sr" else int(k)

    prepare_start = time.perf_counter()
    batches = draw_prepared_batches(
        driver,
        candidate_count,
        n_samples=int(n_samples),
        # Keep chain names method-independent so separate benchmark processes
        # with the same checkpoint/seed consume paired candidate batches.
        chain_prefix=chain_prefix,
    )
    prepare_seconds = time.perf_counter() - prepare_start

    solve_start = time.perf_counter()
    deltas = []
    lambda_grid: tuple[float, ...]
    if method == "mssr":
        first_delta, first_info = solve_prepared_batch(
            driver,
            batches[0],
            diag_shift=0.0,
            solver_fn=spectrum_solver_fn,
        )
        lambda_grid_array = np.asarray(first_info["lambda_grid"], dtype=np.float64)
        lambda_grid = tuple(float(value) for value in lambda_grid_array)
        deltas.append(first_delta)
        for index in range(1, int(k)):
            delta, _ = solve_prepared_batch(
                driver,
                batches[index],
                diag_shift=lambda_grid[index],
                solver_fn=solver_fn,
            )
            deltas.append(delta)
    else:
        lambda_grid = tuple(float(diag_shift) for _ in range(candidate_count))
        for batch in batches:
            delta, _ = solve_prepared_batch(
                driver,
                batch,
                diag_shift=float(diag_shift),
                solver_fn=solver_fn,
            )
            deltas.append(delta)
    solve_seconds = time.perf_counter() - solve_start

    cross_prediction_seconds = 0.0
    simplex_fit_seconds = 0.0
    cross_prediction_launches = 0
    cross_prediction_directions = 0
    if method == "mssr":
        weights, stacking_meta = optimized_exact_loo_weights(driver, batches, deltas)
        cross_prediction_seconds = float(stacking_meta["cross_prediction_s"])
        simplex_fit_seconds = float(stacking_meta["simplex_fit_s"])
        cross_prediction_launches = int(stacking_meta["cross_prediction_launches"])
        cross_prediction_directions = int(stacking_meta["cross_prediction_directions"])
    else:
        weights = np.full(candidate_count, 1.0 / float(candidate_count), dtype=np.float64)

    mix_start = time.perf_counter()
    mixed_delta = deltas[0] if candidate_count == 1 else weighted_delta(deltas, weights)
    _synchronize_tree(mixed_delta)
    mix_seconds = time.perf_counter() - mix_start

    apply_start = time.perf_counter()
    if apply_update:
        _apply_update(driver, mixed_delta)
    apply_seconds = time.perf_counter() - apply_start
    total_seconds = time.perf_counter() - total_start

    # Computed after the timed region.  This makes cross-process batch pairing
    # directly auditable without charging host hashing to any method.
    sample_fingerprints = [
        hashlib.sha256(np.asarray(batch.samples).tobytes()).hexdigest()[:16]
        for batch in batches
    ]

    timings = {
        "prepare_batches_s": float(prepare_seconds),
        "candidate_solves_s": float(solve_seconds),
        "cross_prediction_s": float(cross_prediction_seconds),
        "simplex_fit_s": float(simplex_fit_seconds),
        "mix_s": float(mix_seconds),
        "apply_update_s": float(apply_seconds),
        "total_s": float(total_seconds),
    }
    metadata = {
        "method": method,
        "candidate_batches": int(candidate_count),
        "candidate_solves": int(candidate_count),
        "lambda_grid": [float(value) for value in lambda_grid],
        "lambda_source": "ntk_quantile_fused" if method == "mssr" else "fixed",
        "weights": [float(value) for value in weights],
        "sample_fingerprints": sample_fingerprints,
        "stacking_protocol": "exact_loo" if method == "mssr" else "uniform",
        "diagnostic_jvps": 0,
        "cross_prediction_launches": cross_prediction_launches,
        "cross_prediction_directions": cross_prediction_directions,
    }
    return OptimizedStepResult(delta=mixed_delta, timings=timings, metadata=metadata)
