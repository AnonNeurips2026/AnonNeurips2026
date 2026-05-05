"""
SR Diagnostic Utilities for Ridge-Induced Benign Overfitting

Reusable functions for computing two-batch similarity and gap ratio
diagnostics during VMC optimization. These measure whether the SR update
generalizes across independent MCMC batches.

Key metrics:
  - twobatch_sim: cos(delta_A, delta_B) — do independent batches produce similar updates?
  - gap_ratio: r_val / r_train — does the update fit one batch but not another?
  - r_train: ||O_A delta_A - eps_A||^2 — unnormalized training residual
  - r_val: ||O_B delta_A - eps_B||^2 — unnormalized validation residual
  - *_norm: corresponding residuals normalized by the centered energy variance scale

Adapted from archive/train_with_residuals.py.
"""

import time
from functools import partial

import jax
import jax.numpy as jnp
from flax import core as fcore


def _flatten_samples_array(samples):
    """Collapse all leading sample dimensions into a single batch axis."""
    return jax.lax.collapse(samples, 0, samples.ndim - 1)


def flatten_params(params):
    """Flatten pytree to 1D array."""
    leaves = jax.tree_util.tree_leaves(params)
    return jnp.concatenate([l.ravel() for l in leaves])


def compute_cosine_similarity(delta1, delta2):
    """Cosine similarity between two parameter update pytrees.

    Handles both real and complex parameter vectors correctly.
    For complex vectors, uses the Hermitian inner product.
    """
    v1 = flatten_params(delta1)
    v2 = flatten_params(delta2)
    dot = jnp.sum(v1 * jnp.conj(v2)).real
    norm1 = jnp.sqrt(jnp.sum(jnp.abs(v1)**2))
    norm2 = jnp.sqrt(jnp.sum(jnp.abs(v2)**2))
    return float(dot / (norm1 * norm2 + 1e-12))


# ── Module-level JIT for JVP computation ─────────────────────────────────────
#
# The original closure-based approach created new Python function objects each
# call, which busted JAX's JIT trace cache. Worse, samples captured in closures
# became constants in the jaxpr, forcing XLA recompilation for every different
# batch of samples.
#
# This module-level @jax.jit ensures:
#   - apply_fn (a HashablePartial) is static → stable JIT cache key
#   - samples, params, delta are dynamic traced args → different values reuse
#     the same compiled XLA binary
#   - One compilation per (apply_fn, mode) pair, cached for the process lifetime

@partial(jax.jit, static_argnames=("apply_fn", "mode"))
def _jvp_O_delta_jitted(apply_fn, params, model_state, samples, delta, mode):
    """Compute O @ delta via JVP. Module-level JIT for stable cache.

    All arguments except apply_fn and mode are dynamic (traced), so
    different sample batches / parameter values reuse the compiled binary.
    """
    from netket import jax as nkjax

    params_real, reconstruct = nkjax.tree_to_real(params)
    delta_real, _ = nkjax.tree_to_real(delta)

    def f(p_real):
        variables = {"params": reconstruct(p_real), **model_state}
        return apply_fn(variables, samples)

    _, O_delta = jax.jvp(f, (params_real,), (delta_real,))

    if mode == 'complex':
        O_delta = jnp.concatenate([jnp.real(O_delta), jnp.imag(O_delta)])
    else:
        O_delta = jnp.real(O_delta)
    return O_delta


@partial(jax.jit, static_argnames=("apply_fn", "mode"))
def _jvp_O_delta_many_jitted(apply_fn, params, model_state, samples, delta_bank, mode):
    """Compute a stacked bank of O @ delta predictions via batched JVPs."""

    def single_prediction(delta):
        return _jvp_O_delta_jitted(apply_fn, params, model_state, samples, delta, mode)

    return jax.vmap(single_prediction)(delta_bank)


def compute_O_delta_jvp(apply_fn, params, model_state, samples, delta,
                         mode='complex', chunk_size=None):
    """Compute O @ delta via JVP without materializing the full Jacobian O.

    Chunking is handled at Python level: each chunk calls the same JIT-compiled
    function with identical shapes, so only the first chunk triggers compilation.

    Args:
        apply_fn: Model apply function (variables, x) -> log_psi.
        params: The 'params' subtree of variables.
        model_state: Everything in variables except 'params'.
        samples: Flattened samples, shape (N_s, n_sites).
        delta: Parameter update pytree (same structure as params).
        mode: 'complex' -> concat [Re, Im]; else -> Re only.
        chunk_size: Optional int for chunked evaluation.

    Returns:
        1D array, length N_s (mode='real') or 2*N_s (mode='complex').
    """
    if chunk_size is not None and chunk_size < samples.shape[0]:
        chunks = []
        for i in range(0, samples.shape[0], chunk_size):
            chunk_result = _jvp_O_delta_jitted(
                apply_fn, params, model_state,
                samples[i:i + chunk_size], delta, mode)
            chunks.append(chunk_result)
        return jnp.concatenate(chunks, axis=0)
    return _jvp_O_delta_jitted(apply_fn, params, model_state, samples, delta, mode)


def compute_O_delta_jvp_many(
    apply_fn,
    params,
    model_state,
    samples,
    delta_bank,
    *,
    mode='complex',
    chunk_size=None,
):
    """Compute a stacked bank of ``O @ delta`` predictions via JVP.

    Args:
        apply_fn: Model apply function.
        params: Parameter pytree.
        model_state: Non-parameter variables.
        samples: Flattened samples, shape ``(N_s, n_sites)``.
        delta_bank: Pytree whose leaves have leading shape ``(m, ...)``.
        mode: ``'complex'`` or ``'real'``.
        chunk_size: Optional sample-side chunk size.

    Returns:
        Array of shape ``(m, N_s)`` for real mode or ``(m, 2 * N_s)`` for
        complex mode.
    """
    if chunk_size is not None and chunk_size < samples.shape[0]:
        chunks = []
        for i in range(0, samples.shape[0], chunk_size):
            chunk_result = _jvp_O_delta_many_jitted(
                apply_fn,
                params,
                model_state,
                samples[i:i + chunk_size],
                delta_bank,
                mode,
            )
            chunks.append(chunk_result)
        return jnp.concatenate(chunks, axis=-1)
    return _jvp_O_delta_many_jitted(
        apply_fn,
        params,
        model_state,
        samples,
        delta_bank,
        mode,
    )


def compute_centered_local_energies(driver, vstate, samples):
    """Compute centered local energies matching SR normalization.

    Args:
        driver: VMC_NG driver instance.
        vstate: MCState.
        samples: Raw samples (will be flattened internally).

    Returns:
        dv: Centered+scaled local energies, shape (N_s,) or (2*N_s,).
            Scaling: 2 * (E_loc - mean) / sqrt(N_s).
        local_energies: Raw local energies (uncentered).
    """
    samples_flat = _flatten_samples_array(samples)
    N_s = samples_flat.shape[0]

    local_energies, _ = driver._kernel(
        vstate._apply_fun,
        vstate.variables,
        samples_flat,
        driver._ham,
    )

    local_grad = local_energies.flatten()
    de = local_grad - jnp.mean(local_grad)
    dv = 2.0 * de / jnp.sqrt(N_s)

    if driver.mode == 'complex':
        dv = jnp.concatenate([jnp.real(dv), jnp.imag(dv)])
    else:
        dv = jnp.real(dv)

    return dv, local_energies


def compute_residual_metrics(driver, vstate, samples, delta):
    """Compute raw and normalized residual metrics on a sample batch.

    The SR solver centers Re and Im parts of the Jacobian separately
    (before collapsing), so we must do the same here.

    Args:
        driver: VMC_NG driver.
        vstate: MCState.
        samples: Raw samples.
        delta: Parameter update pytree.

    Returns:
        Dict with residual_raw, residual_norm, target_norm_sq, energy_variance.
    """
    samples_flat = _flatten_samples_array(samples)
    N_s = samples_flat.shape[0]

    dv, local_energies = compute_centered_local_energies(driver, vstate, samples)

    variables = vstate.variables
    model_state, params = fcore.pop(variables, 'params')
    O_delta = compute_O_delta_jvp(
        vstate._apply_fun, params, model_state, samples_flat,
        delta, mode=driver.mode, chunk_size=driver.chunk_size_bwd
    )

    # Center and scale O_delta to match SR normalization.
    # For complex mode, the vector is [Re_0..Re_{N-1}, Im_0..Im_{N-1}].
    # The SR solver centers Re and Im parts SEPARATELY before collapsing,
    # so we must do the same here.
    if driver.mode == 'complex':
        half = O_delta.shape[0] // 2
        O_re = O_delta[:half] - jnp.mean(O_delta[:half])
        O_im = O_delta[half:] - jnp.mean(O_delta[half:])
        O_delta = jnp.concatenate([O_re, O_im])
    else:
        O_delta = O_delta - jnp.mean(O_delta)
    O_delta = O_delta / jnp.sqrt(N_s)

    residual = O_delta - dv
    residual_norm_sq = float(jnp.sum(jnp.abs(residual)**2))
    target_norm_sq = float(jnp.sum(jnp.abs(dv)**2))
    de = local_energies.flatten() - jnp.mean(local_energies.flatten())
    energy_variance = float(jnp.mean(jnp.abs(de) ** 2))

    if target_norm_sq > 1e-12:
        residual_norm = residual_norm_sq / target_norm_sq
    else:
        residual_norm = float('inf')

    return {
        "residual_raw": residual_norm_sq,
        "residual_norm": residual_norm,
        "target_norm_sq": target_norm_sq,
        "energy_variance": energy_variance,
    }


def compute_residual(driver, vstate, samples, delta, normalize=False):
    """Compute residual in raw or normalized units.

    Args:
        normalize: If True, return ||residual||^2 / ||target||^2.
            Otherwise return the unnormalized ||residual||^2.
    """
    metrics = compute_residual_metrics(driver, vstate, samples, delta)
    if normalize:
        return metrics["residual_norm"]
    return metrics["residual_raw"]


def compute_training_residual_from_info(info):
    """Compute training residual metrics from NTK residual info.

    When the NGD driver runs with ``collect_residual_info=True``, the SR solve
    already exposes the centered/scaled target vector ``dv`` and the centered
    NTK representation needed to reconstruct ``O_c delta`` on the training
    batch. This avoids an additional local-energy + JVP pass.

    Returns:
        Dict with residual_raw, residual_norm, target_norm_sq, energy_variance.
    """
    if info is None:
        return None

    aux_vector = info.get("aux_vector")
    ntk_matrix = info.get("ntk_matrix")
    dv = info.get("dv")
    if aux_vector is None or ntk_matrix is None or dv is None:
        return None

    O_delta = ntk_matrix @ aux_vector
    residual = O_delta - dv
    residual_raw = float(jnp.sum(jnp.abs(residual) ** 2))
    target_norm_sq = float(jnp.sum(jnp.abs(dv) ** 2))
    residual_norm = (
        residual_raw / target_norm_sq if target_norm_sq > 1e-12 else float("inf")
    )
    energy_variance = target_norm_sq / 4.0
    return {
        "residual_raw": residual_raw,
        "residual_norm": residual_norm,
        "target_norm_sq": target_norm_sq,
        "energy_variance": energy_variance,
    }


def solve_independent_sr(
    driver,
    vstate,
    *,
    return_diagnostics=False,
    diag_shift_override=None,
):
    """Draw a fresh batch and compute a pure SR solve on it.

    Uses driver.compute_loss_and_update() instead of driver.update_fn()
    directly. This leverages the JIT-cached training path (~4.5s) rather
    than triggering fresh JIT compilation (~1730s) due to nested @jax.jit
    decorators with static_argnames in the netket driver internals.

    Saves and restores driver/vstate state to avoid corrupting training.

    Args:
        driver: VMC_NG driver with update_fn, _kernel, _ham, etc.
        vstate: MCState.

    Returns:
        delta_B: Parameter update pytree from independent batch.
        samples_B: Raw samples from the independent batch.
    """
    # Save state that compute_loss_and_update will overwrite
    samples_original = vstate._samples
    dp_original = driver._dp
    old_updates_original = driver._old_updates
    loss_stats_original = getattr(driver, "_loss_stats", None)
    info_original = getattr(driver, "info", None)
    diag_shift_original = getattr(driver, "diag_shift", None)

    diagnostics = None
    try:
        if diag_shift_override is not None:
            driver.diag_shift = float(diag_shift_override)

        # Draw fresh MCMC samples via the driver's normal reset path
        driver.reset_step()
        samples_B = vstate.samples

        # Compute SR update using the JIT-cached training code path
        driver.compute_loss_and_update()
        delta_B = driver._dp
        if return_diagnostics:
            diagnostics = {
                "loss_stats": getattr(driver, "_loss_stats", None),
                "info": getattr(driver, "info", None),
            }
    finally:
        # Restore state
        vstate._samples = samples_original
        driver._dp = dp_original
        driver._old_updates = old_updates_original
        driver._loss_stats = loss_stats_original
        if hasattr(driver, "info"):
            driver.info = info_original
        if hasattr(driver, "diag_shift"):
            driver.diag_shift = diag_shift_original

    if return_diagnostics:
        return delta_B, samples_B, diagnostics
    return delta_B, samples_B


def compute_diagnostics(driver, vstate):
    """Compute all benign overfitting diagnostics.

    This is the master entry point called by the callback. It performs one
    extra SR solve on an independent batch to measure generalization.

    Args:
        driver: VMC_NG driver (must have _dp set from current step).
        vstate: MCState.

    Returns:
        dict with keys:
            twobatch_sim: Cosine similarity between delta_A and delta_B.
            gap_ratio: r_val / r_train in raw units.
            gap_ratio_norm: r_val_norm / r_train_norm.
            r_train: Unnormalized residual on training batch.
            r_val: Unnormalized residual on validation batch.
    """
    delta_A = driver._dp
    if delta_A is None:
        return None

    # Solve SR on independent batch B
    t1 = time.time()
    delta_B, samples_B = solve_independent_sr(driver, vstate)
    jax.block_until_ready(jax.tree.leaves(delta_B))
    dt_sr = time.time() - t1

    # Two-batch similarity
    twobatch_sim = compute_cosine_similarity(delta_A, delta_B)

    # Training residual: delta_A on batch A (current training samples)
    t2 = time.time()
    train_metrics = compute_training_residual_from_info(getattr(driver, "info", None))
    if train_metrics is None:
        train_metrics = compute_residual_metrics(driver, vstate, vstate.samples, delta_A)
    dt_rtrain = time.time() - t2

    # Validation residual: delta_A on batch B
    t3 = time.time()
    val_metrics = compute_residual_metrics(driver, vstate, samples_B, delta_A)
    dt_rval = time.time() - t3

    gap_ratio = (
        val_metrics["residual_raw"] / train_metrics["residual_raw"]
        if train_metrics["residual_raw"] > 1e-12 else float('inf')
    )
    gap_ratio_norm = (
        val_metrics["residual_norm"] / train_metrics["residual_norm"]
        if train_metrics["residual_norm"] > 1e-12 else float('inf')
    )
    # Keep the normalized version explicitly for comparisons that factor out
    # energy-scale changes between batches.

    dt_total = dt_sr + dt_rtrain + dt_rval
    print(f"\n    [diag timing: sr={dt_sr:.1f}s r_train={dt_rtrain:.1f}s "
          f"r_val={dt_rval:.1f}s total={dt_total:.1f}s]", end="", flush=True)

    return {
        "twobatch_sim": twobatch_sim,
        "gap_ratio": gap_ratio,
        "gap_ratio_norm": gap_ratio_norm,
        "r_train": train_metrics["residual_raw"],
        "r_train_norm": train_metrics["residual_norm"],
        "r_train_target_norm_sq": train_metrics["target_norm_sq"],
        "energy_variance_train": train_metrics["energy_variance"],
        "r_val": val_metrics["residual_raw"],
        "r_val_norm": val_metrics["residual_norm"],
        "r_val_target_norm_sq": val_metrics["target_norm_sq"],
        "energy_variance_val": val_metrics["energy_variance"],
    }
