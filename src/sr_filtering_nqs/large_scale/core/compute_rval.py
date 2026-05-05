"""
Recompute held-out `r_val` on fresh validation samples.

This module still supports direct one-off `r_val` recomputation for ViT, FNQS,
RBM, and Jastrow checkpoints. In the active Section 4 workflow, Figure 4 reads
canonical checkpoint-local diagnostics outputs directly.
"""

import argparse
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sr_filtering_nqs.large_scale.core.common import (
    ACTIVE_SYSTEM_CHOICES,
    HELDOUT_SAMPLES,
    LAMBDA_GRID,
    LATTICE_SIZE,
    TRAIN_SAMPLES,
    VIT_EXPANSION,
    canonical_results_root_candidates,
    heldout_samples_for_system,
    legacy_system_names,
    make_system,
    model_type_for_system,
    normalize_system_name,
)
from sr_filtering_nqs.large_scale.core.diagnostic_core import (
    process_fnqs_checkpoint_metrics,
    process_vit_checkpoint_metrics,
)
from sr_filtering_nqs.large_scale.core.diagnostic_state import atomic_save

# ── Constants ────────────────────────────────────────────────────────────────

N_VAL_DEFAULT = HELDOUT_SAMPLES
N_SAMPLES = TRAIN_SAMPLES
LAMBDA_EVAL = LAMBDA_GRID
JVP_CHUNK = 8192
ELOC_CHUNK = 8192
RESULT_NAME = "rval_step{step:06d}.pkl"


def _diagnostics_output_dir(ckpt_path):
    ckpt_path = Path(ckpt_path)
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent / "diagnostics"
    return ckpt_path.parent


def _parse_lambda_token(token):
    try:
        return float(token)
    except ValueError:
        return None


def _selected_checkpoint_paths(ckpt_dir, step_stride=None, min_step=None):
    ckpts = sorted(Path(ckpt_dir).glob("checkpoint_step*.pkl"))
    if not ckpts:
        return []

    if step_stride is None:
        return [ckpts[-1]]

    ckpt_map = {}
    for path in ckpts:
        step_str = path.stem.replace("checkpoint_step", "")
        try:
            step = int(step_str)
        except ValueError:
            continue
        ckpt_map[step] = path

    if not ckpt_map:
        return []

    latest_step = max(ckpt_map)
    selected = {latest_step}
    for step in sorted(ckpt_map):
        if step % step_stride == 0:
            selected.add(step)

    if min_step is not None:
        selected = {step for step in selected if step >= min_step}

    return [ckpt_map[step] for step in sorted(selected)]


# ── Drawing samples ──────────────────────────────────────────────────────────

def draw_samples(vstate, n_target):
    """Draw n_target MCMC samples via multiple rounds.

    Saves and restores vstate._samples to avoid corrupting training state.

    Returns:
        numpy array of shape (n_target, n_sites).
    """
    n_chains = vstate.n_samples
    n_rounds = max(1, -(-n_target // n_chains))
    actual_n = n_rounds * n_chains

    original_samples = vstate._samples

    rounds = []
    for i in range(n_rounds):
        vstate.reset()
        vstate.sample()
        s = np.asarray(vstate.samples)
        # Flatten leading dims to (n_chains, n_sites)
        s = s.reshape(-1, s.shape[-1])
        rounds.append(s)
        if (i + 1) % 50 == 0:
            print(f"    round {i+1}/{n_rounds}", flush=True)

    vstate._samples = original_samples

    samples = np.concatenate(rounds, axis=0)
    assert samples.shape[0] == actual_n
    return samples[:n_target]


# ── Local energy computation (chunked get_conn_padded) ────────────────────

def compute_local_energies_chunked(apply_fn, variables, samples, hamiltonian,
                                    conn_chunk=ELOC_CHUNK, eval_chunk=ELOC_CHUNK):
    """Compute local energies with chunked get_conn_padded to limit memory.

    Chunks over samples for both get_conn_padded and network evaluation.

    Returns:
        Complex array of shape (N,).
    """
    samples_np = np.asarray(samples)
    N = samples_np.shape[0]
    n_sites = samples_np.shape[1]

    eloc_parts = []
    for i in range(0, N, conn_chunk):
        chunk = samples_np[i:i + conn_chunk]
        n_chunk = len(chunk)

        x_primes, mels = hamiltonian.get_conn_padded(chunk)
        n_conn = x_primes.shape[1]

        log_psi_x = np.array(apply_fn(variables, jnp.array(chunk)))

        x_flat = np.array(x_primes).reshape(-1, n_sites)
        log_psi_conn_parts = []
        for j in range(0, len(x_flat), eval_chunk):
            lp = apply_fn(variables, jnp.array(x_flat[j:j + eval_chunk]))
            log_psi_conn_parts.append(np.array(lp))
        log_psi_conn = np.concatenate(log_psi_conn_parts).reshape(n_chunk, n_conn)

        eloc_chunk = np.sum(
            np.array(mels) * np.exp(log_psi_conn - log_psi_x[:, None]),
            axis=1,
        )
        eloc_parts.append(eloc_chunk)

    return np.concatenate(eloc_parts)


# ── JVP for ViT (mode='complex', tree_to_real) ──────────────────────────────

def compute_residual_vit_jvp(driver, vstate, samples_val, eloc_val, delta,
                              chunk_size=JVP_CHUNK):
    """Compute residual metrics for ViT using the existing JVP infrastructure.

    Uses sr_diagnostics.compute_O_delta_jvp with mode='complex' and
    centers Re/Im halves separately to match SR normalization.

    Returns:
        Dict with residual_raw, residual_norm, target_norm_sq, energy_variance.
    """
    from sr_filtering_nqs.large_scale.core import sr_diagnostics
    from flax import core as fcore

    N = len(samples_val)
    samples_jnp = jnp.array(samples_val)

    variables = vstate.variables
    model_state, params = fcore.pop(variables, 'params')

    # Centered local energies (target vector)
    y = eloc_val - np.mean(eloc_val)
    dv = 2.0 * y / np.sqrt(N)
    if driver.mode == 'complex':
        dv = np.concatenate([np.real(dv), np.imag(dv)])
    else:
        dv = np.real(dv)

    # JVP: O @ delta
    O_delta = sr_diagnostics.compute_O_delta_jvp(
        vstate._apply_fun, params, model_state, samples_jnp,
        delta, mode=driver.mode, chunk_size=chunk_size)
    O_delta = np.array(O_delta)

    # Center Re/Im halves separately
    if driver.mode == 'complex':
        half = len(O_delta) // 2
        O_re = O_delta[:half] - np.mean(O_delta[:half])
        O_im = O_delta[half:] - np.mean(O_delta[half:])
        O_delta = np.concatenate([O_re, O_im])
    else:
        O_delta = O_delta - np.mean(O_delta)
    O_delta = O_delta / np.sqrt(N)

    residual = O_delta - dv
    residual_raw = float(np.sum(np.abs(residual)**2))
    target_norm_sq = float(np.sum(np.abs(dv)**2))
    energy_variance = float(np.mean(np.abs(y) ** 2))
    residual_norm = residual_raw / (target_norm_sq + 1e-30)
    return {
        "residual_raw": residual_raw,
        "residual_norm": residual_norm,
        "target_norm_sq": target_norm_sq,
        "energy_variance": energy_variance,
    }


# ── JVP for holomorphic models (RBM, Jastrow) ───────────────────────────────

@partial(jax.jit, static_argnames=("apply_fn",))
def _jvp_holomorphic(apply_fn, params, model_state, samples, delta_pytree):
    """Compute O @ delta via holomorphic JVP (no tree_to_real needed)."""
    def f(p):
        return apply_fn({"params": p, **model_state}, samples)
    _, O_delta = jax.jvp(f, (params,), (delta_pytree,))
    return O_delta


def unflatten_delta(delta_flat, params):
    """Unflatten (P,) complex array to match params pytree structure."""
    leaves, treedef = jax.tree.flatten(params)
    sizes = [l.size for l in leaves]
    shapes = [l.shape for l in leaves]
    splits = np.cumsum(sizes[:-1])
    delta_leaves = np.split(delta_flat, splits)
    return treedef.unflatten([d.reshape(s) for d, s in zip(delta_leaves, shapes)])


def compute_residual_holomorphic_jvp(apply_fn, params, model_state,
                                      samples_val, eloc_val, delta_pytree,
                                      chunk_size=JVP_CHUNK):
    """Compute residual metrics for holomorphic models (RBM/Jastrow) via JVP.

    Residual: ||X_c @ delta - y||^2 in raw units, with an explicit normalized
    companion based on ||y||^2.

    Returns:
        Dict with residual_raw, residual_norm, target_norm_sq, energy_variance.
    """
    N = len(samples_val)
    y = eloc_val - np.mean(eloc_val)

    # Chunked JVP
    Od_parts = []
    for i in range(0, N, chunk_size):
        chunk = jnp.array(samples_val[i:i + chunk_size])
        Od = _jvp_holomorphic(apply_fn, params, model_state, chunk, delta_pytree)
        Od_parts.append(np.array(Od))
    Od_all = np.concatenate(Od_parts)
    X_delta = Od_all - np.mean(Od_all)  # center

    residual = X_delta - y
    target_norm_sq = float(np.sum(np.abs(y)**2))
    residual_raw = float(np.sum(np.abs(residual)**2))
    residual_norm = (
        residual_raw / target_norm_sq if target_norm_sq > 1e-30 else float("inf")
    )
    energy_variance = float(np.mean(np.abs(y) ** 2))
    return {
        "residual_raw": residual_raw,
        "residual_norm": residual_norm,
        "target_norm_sq": target_norm_sq,
        "energy_variance": energy_variance,
    }


# ── ViT checkpoint processing ────────────────────────────────────────────────

def process_vit_checkpoint(ckpt_path, n_val=N_VAL_DEFAULT):
    result = process_vit_checkpoint_metrics(
        ckpt_path,
        n_val=n_val,
        validation_round_samples=None,
        jvp_chunk=JVP_CHUNK,
        conn_chunk=ELOC_CHUNK,
        eval_chunk=ELOC_CHUNK,
        chunk_size_bwd=None,
        use_prep_cache=True,
        want_rval=True,
        want_twobatch=False,
    )
    return result["rval"]


def process_fnqs_checkpoint(ckpt_path, n_val=None):
    result = process_fnqs_checkpoint_metrics(
        ckpt_path,
        n_val=n_val,
        validation_round_samples=None,
        jvp_chunk=JVP_CHUNK,
        chunk_size_bwd=None,
        use_prep_cache=True,
        want_rval=True,
        want_twobatch=False,
    )
    return result["rval"]


# ── Holomorphic (RBM/Jastrow) checkpoint processing ─────────────────────────

def process_holomorphic_checkpoint(ckpt_path, model_type, system_name,
                                    alpha=None, n_val=N_VAL_DEFAULT,
                                    lambda_values=None):
    """Process a single RBM or Jastrow checkpoint.

    1. Reconstruct model + vstate
    2. Draw training batch A → holomorphic Jacobian → S, F
    3. Draw n_val validation samples (once)
    4. Compute local energies on validation set (once)
    5. For each lambda: solve, unflatten delta, compute r_train, r_val (JVP)

    Returns:
        List of result dicts.
    """
    import netket as nk
    from netket import jax as nkjax

    if lambda_values is None:
        lambda_values = LAMBDA_EVAL
    system_name = normalize_system_name(system_name)

    ckpt_path = Path(ckpt_path)
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    step = ckpt["step"]
    parameters = ckpt["parameters"]

    # Check existing
    out_path = _diagnostics_output_dir(ckpt_path) / RESULT_NAME.format(step=step)
    if out_path.exists():
        print(f"  Already exists: {out_path.name}, skipping.")
        with open(out_path, "rb") as f:
            return pickle.load(f)

    label = f"{model_type}"
    if alpha is not None:
        label += f" alpha={alpha}"
    print(f"\n  {label}: {system_name} step={step}")

    # Reconstruct model
    system = make_system(system_name, L=LATTICE_SIZE)
    hamiltonian = system.hamiltonian

    if model_type == "rbm":
        model = nk.models.RBM(alpha=alpha, param_dtype=complex)
    elif model_type == "jastrow":
        model = nk.models.Jastrow(param_dtype=complex)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    sampler = nk.sampler.MetropolisExchange(
        system.hilbert_space, graph=system.graph, d_max=2,
        n_chains=N_SAMPLES,
    )
    vstate = nk.vqs.MCState(
        sampler, model=model, n_samples=N_SAMPLES, chunk_size=4096,
    )
    vstate.parameters = parameters
    n_params = sum(p.size for p in jax.tree.leaves(vstate.parameters))
    print(f"    P={n_params}, P/Ns={n_params/N_SAMPLES:.3f}")

    # ── Training batch A: Jacobian + local energies ──────────────────────
    print(f"    Computing Jacobian on batch A ({N_SAMPLES} samples)...",
          end="", flush=True)
    t0 = time.time()

    vstate.reset()
    vstate.sample()
    samples_A = np.asarray(jax.lax.collapse(vstate.samples, 0, vstate.samples.ndim - 1))
    N_s = samples_A.shape[0]

    J_A = np.array(nkjax.jacobian(
        vstate._apply_fun, vstate.parameters, jnp.array(samples_A),
        vstate.model_state, mode="holomorphic", dense=True, center=False,
    ))
    eloc_A = compute_local_energies_chunked(
        vstate._apply_fun, vstate.variables, samples_A, hamiltonian)

    X_A = J_A - np.mean(J_A, axis=0, keepdims=True)
    y_A = eloc_A - np.mean(eloc_A)

    S = (X_A.conj().T @ X_A) / N_s
    F = (X_A.conj().T @ y_A) / N_s

    dt = time.time() - t0
    print(f" done ({dt:.1f}s)")

    # ── Validation samples ───────────────────────────────────────────────
    n_rounds = max(1, -(-n_val // vstate.n_samples))
    actual_n_val = n_rounds * vstate.n_samples
    print(f"    Drawing {actual_n_val} validation samples ({n_rounds} rounds)...",
          end="", flush=True)
    t1 = time.time()
    samples_val = draw_samples(vstate, n_val)
    dt = time.time() - t1
    print(f" done ({dt:.1f}s)")

    # ── Local energies on validation set ─────────────────────────────────
    print(f"    Computing local energies on {len(samples_val)} samples...",
          end="", flush=True)
    t2 = time.time()
    eloc_val = compute_local_energies_chunked(
        vstate._apply_fun, vstate.variables, samples_val, hamiltonian)
    dt = time.time() - t2
    print(f" done ({dt:.1f}s)")

    # ── Lambda sweep ─────────────────────────────────────────────────────
    P = S.shape[0]
    eye_P = np.eye(P)
    train_target_norm_sq = float(np.sum(np.abs(y_A)**2))
    energy_variance_train = float(np.mean(np.abs(y_A) ** 2))

    params = vstate.parameters
    model_state = vstate.model_state or {}

    print(f"    {'lambda':>10s} | {'r_train':>10s} | {'r_val':>10s} | {'gap':>10s}")
    print(f"    {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    results = []
    for lam in lambda_values:
        S_reg = S + lam * eye_P
        delta_flat = np.linalg.solve(S_reg, F)

        # r_train on batch A (explicit matrix multiply)
        resid_A = X_A @ delta_flat - y_A
        r_train_raw = float(np.sum(np.abs(resid_A)**2))
        r_train_norm = (
            r_train_raw / train_target_norm_sq
            if train_target_norm_sq > 1e-30 else float("inf")
        )

        # Unflatten delta for JVP
        delta_pytree = unflatten_delta(delta_flat, params)

        # r_val via holomorphic JVP on held-out samples
        val_metrics = compute_residual_holomorphic_jvp(
            vstate._apply_fun, params, model_state,
            samples_val, eloc_val, delta_pytree,
            chunk_size=JVP_CHUNK)

        gap = (
            val_metrics["residual_raw"] / r_train_raw
            if r_train_raw > 1e-12 else float("inf")
        )
        gap_norm = (
            val_metrics["residual_norm"] / r_train_norm
            if r_train_norm > 1e-12 else float("inf")
        )

        print(
            f"    {lam:10.0e} | {r_train_raw:10.6f} | "
            f"{val_metrics['residual_raw']:10.6f} | {gap:10.2f}"
        )

        results.append({
            "model_type": model_type,
            "system": system_name,
            "lambda": lam,
            "step": step,
            "P": n_params,
            "P_over_Ns": n_params / N_SAMPLES,
            "r_train": r_train_raw,
            "r_train_norm": r_train_norm,
            "r_train_target_norm_sq": train_target_norm_sq,
            "energy_variance_train": energy_variance_train,
            "r_val": val_metrics["residual_raw"],
            "r_val_norm": val_metrics["residual_norm"],
            "r_val_target_norm_sq": val_metrics["target_norm_sq"],
            "energy_variance_val": val_metrics["energy_variance"],
            "gap_ratio": gap,
            "gap_ratio_norm": gap_norm,
            "n_val": len(samples_val),
        })
        if alpha is not None:
            results[-1]["alpha"] = alpha

    # Save all lambda results for this checkpoint
    atomic_save(results, out_path)
    print(f"    Saved: {out_path.name}")
    return results


# ── Checkpoint discovery ─────────────────────────────────────────────────────

def discover_vit_checkpoints(input_dir, system, step_stride=None, diag_shift=None,
                             min_step=None):
    """Find checkpoints from the active canonical results trees."""
    base = Path(input_dir)
    system = normalize_system_name(system)
    results = []
    seen = set()

    for canonical_root in canonical_results_root_candidates(base, system):
        if not canonical_root.exists():
            continue
        for lam_dir in sorted(canonical_root.iterdir(), key=lambda p: p.name):
            if not lam_dir.is_dir():
                continue
            lam = _parse_lambda_token(lam_dir.name)
            if lam is None:
                continue
            if diag_shift is not None and not np.isclose(lam, diag_shift, rtol=1e-6, atol=0.0):
                continue
            ckpt_dir = lam_dir / "results" / "checkpoints"
            for ckpt in _selected_checkpoint_paths(ckpt_dir, step_stride, min_step):
                resolved = str(ckpt.resolve())
                if resolved not in seen:
                    results.append(ckpt)
                    seen.add(resolved)

    return sorted(results, key=lambda p: str(p))


def discover_fnqs_checkpoints(input_dir, system, step_stride=None, diag_shift=None, min_step=None):
    return discover_vit_checkpoints(
        input_dir,
        system,
        step_stride=step_stride,
        diag_shift=diag_shift,
        min_step=min_step,
    )


def discover_rbm_checkpoints(input_dir, system):
    """Find latest checkpoint per alpha directory for RBM."""
    base = Path(input_dir)
    system = normalize_system_name(system)
    results = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        for prefix in legacy_system_names(system):
            if not d.name.startswith(f"{prefix}_alpha"):
                continue
            ckpts = sorted(d.glob("checkpoint_step*.pkl"))
            if ckpts:
                alpha_str = d.name.split("alpha")[-1]
                alpha = float(alpha_str)
                results.append((alpha, ckpts[-1]))
            break
    return results


def discover_jastrow_checkpoints(input_dir, system):
    """Find latest checkpoint for Jastrow."""
    base = Path(input_dir)
    system = normalize_system_name(system)
    for prefix in legacy_system_names(system):
        d = base / prefix
        if not d.exists():
            continue
        ckpts = sorted(d.glob("checkpoint_step*.pkl"))
        if ckpts:
            return [ckpts[-1]]
    return []


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute held-out residuals on fresh validation samples"
    )
    parser.add_argument("--model-type", required=True,
                        choices=["vit", "fnqs", "rbm", "jastrow"])
    parser.add_argument("--system", type=str, choices=ACTIVE_SYSTEM_CHOICES,
                        help="System name (sweeps all relevant checkpoints)")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to a single checkpoint file")
    parser.add_argument("--input-dir", type=str,
                        help="Base directory containing run subdirs")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Validation sample count (default: per-system canonical value)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="RBM alpha (for single checkpoint mode)")
    parser.add_argument("--step-stride", type=int, default=None,
                        help="Process checkpoints at multiples of this stride "
                             "(default: latest only)")
    parser.add_argument("--diag-shift", type=float, default=None,
                        help="Only process this specific lambda value")
    parser.add_argument("--min-step", type=int, default=None,
                        help="Only process checkpoints with step >= this value")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── compute ──────────────────────────────────────────────────────────
    model_type = args.model_type
    n_val = args.n_val

    if args.checkpoint:
        # Single checkpoint mode
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"Error: checkpoint not found: {ckpt_path}")
            sys.exit(1)

        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        config = ckpt["config"]
        system_name = config["system"]

        if model_type == "vit":
            process_vit_checkpoint(
                ckpt_path,
                n_val=(heldout_samples_for_system(system_name) if n_val is None else n_val),
            )
        elif model_type == "fnqs":
            process_fnqs_checkpoint(ckpt_path, n_val=n_val)
        else:
            # Need system and alpha info
            config = ckpt["config"]
            alpha = args.alpha or config.get("alpha")
            process_holomorphic_checkpoint(
                ckpt_path, model_type, system_name,
                alpha=alpha,
                n_val=(heldout_samples_for_system(system_name) if n_val is None else n_val),
            )

    elif args.system and args.input_dir:
        # System sweep mode
        system = args.system

        if n_val is None:
            n_val = heldout_samples_for_system(system)

        if model_type == "vit":
            ckpts = discover_vit_checkpoints(
                args.input_dir, system,
                step_stride=args.step_stride,
                diag_shift=args.diag_shift,
                min_step=args.min_step)
            print(f"Found {len(ckpts)} ViT checkpoints for {system}")
            for cp in ckpts:
                try:
                    process_vit_checkpoint(cp, n_val=n_val)
                except Exception as e:
                    print(f"  FAILED: {cp}: {e}")
                    import traceback
                    traceback.print_exc()

        elif model_type == "fnqs":
            ckpts = discover_fnqs_checkpoints(
                args.input_dir,
                system,
                step_stride=args.step_stride,
                diag_shift=args.diag_shift,
                min_step=args.min_step,
            )
            print(f"Found {len(ckpts)} FNQS checkpoints for {system}")
            for cp in ckpts:
                try:
                    process_fnqs_checkpoint(cp, n_val=n_val)
                except Exception as e:
                    print(f"  FAILED: {cp}: {e}")
                    import traceback
                    traceback.print_exc()

        elif model_type == "rbm":
            ckpts = discover_rbm_checkpoints(args.input_dir, system)
            print(f"Found {len(ckpts)} RBM checkpoints for {system}")
            for alpha, cp in ckpts:
                try:
                    process_holomorphic_checkpoint(
                        cp, "rbm", system, alpha=alpha, n_val=n_val)
                except Exception as e:
                    print(f"  FAILED: {cp}: {e}")
                    import traceback
                    traceback.print_exc()

        elif model_type == "jastrow":
            ckpts = discover_jastrow_checkpoints(args.input_dir, system)
            print(f"Found {len(ckpts)} Jastrow checkpoints for {system}")
            for cp in ckpts:
                try:
                    process_holomorphic_checkpoint(
                        cp, "jastrow", system, n_val=n_val)
                except Exception as e:
                    print(f"  FAILED: {cp}: {e}")
                    import traceback
                    traceback.print_exc()

    else:
        print("Error: specify --checkpoint or --system + --input-dir")
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
