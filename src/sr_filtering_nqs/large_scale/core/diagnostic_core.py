from __future__ import annotations

import math
import pickle
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as fcore

from sr_filtering_nqs.large_scale.core import sr_diagnostics
from sr_filtering_nqs.large_scale.core.common import (
    default_delta_bank_chunk_size_bwd,
    format_lambda_dir,
    heldout_samples_for_system,
    large_scale_root_dir,
    normalize_system_name,
)
from sr_filtering_nqs.large_scale.core.diagnostic_state import EPS, atomic_save
from sr_filtering_nqs.large_scale.core.multidelta_metrics import (
    finalize_multidelta_rval,
    finalize_multidelta_twobatch,
)


DELTA_BANK_NAME = "delta_bank_step{step:06d}.pkl"
FNQS_DIAGNOSTICS_VERSION = 2


class MissingDiagnosticsCacheError(FileNotFoundError):
    pass


def diagnostics_output_dir(ckpt_path):
    ckpt_path = Path(ckpt_path)
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent / "diagnostics"
    return ckpt_path.parent


def result_output_path(ckpt_path, name_template: str, step: int) -> Path:
    return diagnostics_output_dir(ckpt_path) / name_template.format(step=step)


def delta_bank_output_path(ckpt_path, step: int) -> Path:
    return diagnostics_output_dir(ckpt_path) / DELTA_BANK_NAME.format(step=step)


def delta_bank_output_path_for_results(results_dir: Path, step: int) -> Path:
    return Path(results_dir) / "diagnostics" / DELTA_BANK_NAME.format(step=step)


def numpy_tree(tree):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)


def jax_tree(tree):
    return jax.tree_util.tree_map(jnp.asarray, tree)


def stack_tree(pytrees):
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs, axis=0), *pytrees)


def slice_tree(tree, start: int, stop: int):
    return jax.tree_util.tree_map(lambda x: x[start:stop], tree)


def tree_mean(tree):
    return jax.tree_util.tree_map(lambda x: np.mean(np.asarray(x), axis=0), tree)


def nanmean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def count_parameters(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def loss_stats_variance(loss_stats) -> float:
    if loss_stats is None:
        return float("nan")
    try:
        return float(np.real(complex(loss_stats.Variance)))
    except Exception:
        return float("nan")


def train_metrics_from_checkpoint(ckpt):
    row = dict(ckpt.get("train_metrics") or {})
    if not row:
        return None
    return {
        "residual_raw": float(row.get("r_train", float("nan"))),
        "residual_norm": float(row.get("r_train_norm", float("nan"))),
        "target_norm_sq": float(row.get("r_train_target_norm_sq", float("nan"))),
        "energy_variance": float(row.get("energy_variance_train", float("nan"))),
    }


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _delta_bank_matches(path: Path, *, m_deltas: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = _load_pickle(path)
    except Exception:
        return False
    return int(payload.get("m_deltas", -1)) == int(m_deltas) and "delta_bank" in payload


def resolve_delta_bank_chunk_size_bwd(config: dict, *, override: int | None = None) -> int | None:
    if override is not None:
        return int(override)
    return default_delta_bank_chunk_size_bwd(
        config["system"],
        n_samples=config.get("n_samples"),
    )


def resolve_solve_lambda(config: dict, *, diag_shift_override: float | None = None) -> float:
    if diag_shift_override is not None:
        return float(diag_shift_override)
    return float(config["diag_shift"])


def _compute_deltas_command(ckpt_path, ckpt, *, m_deltas: int) -> str:
    step = int(ckpt["step"])
    config = ckpt["config"]
    system_name = normalize_system_name(config["system"])
    large_scale_root = large_scale_root_dir(ckpt_path)
    chunk_size_bwd = resolve_delta_bank_chunk_size_bwd(config)
    return (
        f'uv run python "{large_scale_root / "cli" / "compute_deltas.py"}" compute '
        f"--system {system_name} "
        f'--input-dir "{large_scale_root}" '
        f"--diag-shift {float(config['diag_shift']):.12g} "
        f"--min-step {step} "
        f"--m-deltas {int(m_deltas)} "
        f"--chunk-size-bwd {int(chunk_size_bwd)}"
    )


def load_required_delta_bank(*, ckpt_path, ckpt, m_deltas: int):
    step = int(ckpt["step"])
    bank_path = delta_bank_output_path(ckpt_path, step)
    if not _delta_bank_matches(bank_path, m_deltas=m_deltas):
        command = _compute_deltas_command(ckpt_path, ckpt, m_deltas=m_deltas)
        raise MissingDiagnosticsCacheError(
            f"Missing compatible diagnostic delta bank for {ckpt_path}: {bank_path}. "
            f"Diagnostics are cache-only. Run: {command}"
        )
    return _load_pickle(bank_path)


def sample_validation_round(vstate, n_target: int, *, n_replicas: int | None = None):
    vstate.reset()
    vstate.sample()
    samples = np.asarray(vstate.samples).reshape(-1, vstate.samples.shape[-1])
    if samples.shape[0] < n_target:
        raise ValueError(f"Requested {n_target} samples, sampler produced {samples.shape[0]}")
    if samples.shape[0] == n_target:
        return samples
    if n_replicas is None:
        return samples[:n_target]
    if n_target % n_replicas != 0:
        raise ValueError(
            f"n_target={n_target} must be divisible by n_replicas={n_replicas}"
        )
    per_replica_have = samples.shape[0] // n_replicas
    per_replica_take = n_target // n_replicas
    return (
        samples.reshape(n_replicas, per_replica_have, -1)[:, :per_replica_take, :]
        .reshape(n_target, samples.shape[-1])
    )


def resolve_round_samples(
    *,
    system_name: str,
    total_n_val: int,
    requested_round_samples: int | None,
    default_round_samples: int,
    n_replicas: int | None = None,
):
    round_samples = (
        int(requested_round_samples)
        if requested_round_samples is not None
        else int(default_round_samples)
    )
    if round_samples <= 0:
        raise ValueError(f"validation_round_samples must be positive, got {round_samples}")
    if n_replicas is not None and round_samples % n_replicas != 0:
        raise ValueError(
            f"validation_round_samples={round_samples} must be divisible by n_replicas={n_replicas} "
            f"for {system_name} diagnostics"
        )
    return round_samples


def chunk_size_for_round(round_size: int, requested: int | None):
    if requested is None:
        return None
    return min(int(requested), int(round_size))


def compute_vit_local_energies_round(
    apply_fn,
    variables,
    samples,
    hamiltonian,
    *,
    conn_chunk: int,
    eval_chunk: int,
):
    samples_np = np.asarray(samples)
    n_samples = samples_np.shape[0]
    n_sites = samples_np.shape[1]

    outputs = []
    for start in range(0, n_samples, conn_chunk):
        batch = samples_np[start : start + conn_chunk]
        x_primes, mels = hamiltonian.get_conn_padded(batch)
        log_psi_batch = np.asarray(apply_fn(variables, jnp.asarray(batch)))
        x_flat = np.asarray(x_primes).reshape(-1, n_sites)
        conn_parts = []
        for inner in range(0, x_flat.shape[0], eval_chunk):
            chunk = x_flat[inner : inner + eval_chunk]
            conn_parts.append(np.asarray(apply_fn(variables, jnp.asarray(chunk))))
        log_psi_conn = np.concatenate(conn_parts, axis=0).reshape(x_primes.shape[:2])
        eloc = np.sum(
            np.asarray(mels) * np.exp(log_psi_conn - log_psi_batch[:, None]),
            axis=1,
        )
        outputs.append(eloc)
    return np.concatenate(outputs, axis=0)


def compute_driver_local_energies_round(driver, vstate, samples):
    samples_jax = jnp.asarray(samples)
    local_energies, _ = driver._kernel(
        vstate._apply_fun,
        vstate.variables,
        samples_jax,
        driver._ham,
    )
    return np.asarray(local_energies).reshape(-1)


def compute_train_metrics_fallback(driver, vstate, delta):
    metrics = sr_diagnostics.compute_residual_metrics(driver, vstate, vstate.samples, delta)
    return {
        "residual_raw": float(metrics["residual_raw"]),
        "residual_norm": float(metrics["residual_norm"]),
        "target_norm_sq": float(metrics["target_norm_sq"]),
        "energy_variance": float(metrics["energy_variance"]),
    }


def _prediction_channels_from_bank(values, *, mode: str, sample_count: int):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected predictions with shape (m, d), got {values.shape}")
    if mode == "complex":
        return np.stack(
            [values[:, :sample_count], values[:, sample_count:2 * sample_count]],
            axis=1,
        )
    return np.real(values[:, :sample_count])[:, None, :]


def _energy_channels_from_eloc(eloc, *, mode: str):
    eloc = np.asarray(eloc)
    if mode == "complex":
        return np.stack(
            [
                np.asarray(np.real(eloc), dtype=np.float64),
                np.asarray(np.imag(eloc), dtype=np.float64),
            ],
            axis=0,
        )
    return np.asarray(np.real(eloc), dtype=np.float64)[None, :]


def _store_validation_block(
    buffer,
    values,
    *,
    sample_offset: int,
    n_replicas: int | None,
):
    """Store one validation round while preserving replica-major layout.

    FNQS centering reshapes the flat sample axis into ``(n_replicas, n_per_replica)``.
    When validation is streamed over multiple rounds, appending each round contiguously
    would create a round-major layout and scramble the per-replica centering. For
    replica-conditioned systems we therefore place each round into the per-replica slice
    of the final buffer.
    """

    values = np.asarray(values, dtype=np.float64)
    take = int(values.shape[-1])

    if n_replicas is None:
        buffer[..., sample_offset : sample_offset + take] = values
        return

    total_count = int(buffer.shape[-1])
    if total_count % n_replicas != 0:
        raise ValueError(
            f"Buffer sample axis {total_count} is not divisible by n_replicas={n_replicas}"
        )
    if take % n_replicas != 0:
        raise ValueError(
            f"Round sample axis {take} is not divisible by n_replicas={n_replicas}"
        )
    if sample_offset % n_replicas != 0:
        raise ValueError(
            f"sample_offset={sample_offset} is not divisible by n_replicas={n_replicas}"
        )

    per_replica_total = total_count // n_replicas
    per_replica_take = take // n_replicas
    per_replica_offset = sample_offset // n_replicas
    if per_replica_offset + per_replica_take > per_replica_total:
        raise ValueError(
            "Validation block would exceed the per-replica buffer extent: "
            f"offset={per_replica_offset} take={per_replica_take} total={per_replica_total}"
        )

    buffer_view = buffer.reshape(buffer.shape[:-1] + (n_replicas, per_replica_total))
    values_view = values.reshape(values.shape[:-1] + (n_replicas, per_replica_take))
    buffer_view[..., per_replica_offset : per_replica_offset + per_replica_take] = values_view


def build_rval_result(
    *,
    model_type: str,
    system_name: str,
    lam: float,
    step: int,
    n_params: int,
    n_train: int,
    n_val: int,
    m_deltas: int,
    train_metrics: dict,
    val_metrics: dict,
    metadata: dict | None = None,
):
    gap = val_metrics["residual_raw"] / (float(train_metrics["residual_raw"]) + EPS)
    gap_norm = val_metrics["residual_norm"] / (float(train_metrics["residual_norm"]) + EPS)
    result = {
        "model_type": model_type,
        "system": system_name,
        "lambda": float(lam),
        "step": int(step),
        "P": int(n_params),
        "P_over_Ns": float(n_params) / float(n_train),
        "r_train": float(train_metrics["residual_raw"]),
        "r_train_norm": float(train_metrics["residual_norm"]),
        "r_train_target_norm_sq": float(train_metrics["target_norm_sq"]),
        "energy_variance_train": float(train_metrics["energy_variance"]),
        "r_val": float(val_metrics["residual_raw"]),
        "r_val_norm": float(val_metrics["residual_norm"]),
        "r_val_target_norm_sq": float(val_metrics["target_norm_sq"]),
        "energy_variance_val": float(val_metrics["energy_variance"]),
        "gap_ratio": float(gap),
        "gap_ratio_norm": float(gap_norm),
        "n_val": int(n_val),
        "m_deltas": int(m_deltas),
        "diagnostics_version": (
            int(FNQS_DIAGNOSTICS_VERSION) if model_type == "fnqs" else 1
        ),
    }
    if metadata:
        result.update(dict(metadata))
    return result


def build_twobatch_result(
    *,
    system_name: str,
    model_type: str,
    lam: float,
    step: int,
    n_val: int,
    m_deltas: int,
    train_target_norm_sq: float,
    energy_variance_train: float,
    twobatch_metrics: dict,
    metadata: dict | None = None,
):
    result = {
        "system": system_name,
        "lambda": float(lam),
        "step": int(step),
        "var_hat_raw": float(twobatch_metrics["var_hat_raw"]),
        "var_hat_norm": float(twobatch_metrics["var_hat_norm"]),
        "train_target_norm_sq": float(train_target_norm_sq),
        "energy_variance_train": float(energy_variance_train),
        "snorm_diff_sq": float(twobatch_metrics["snorm_diff_sq"]),
        "n_val": int(n_val),
        "m_deltas": int(m_deltas),
        "diagnostics_version": (
            int(FNQS_DIAGNOSTICS_VERSION) if model_type == "fnqs" else 1
        ),
    }
    if metadata:
        result.update(dict(metadata))
    return result


def _train_target_metrics_from_solve(diagnostics) -> tuple[float, float]:
    metrics = sr_diagnostics.compute_training_residual_from_info(diagnostics.get("info"))
    if metrics is not None:
        return float(metrics["target_norm_sq"]), float(metrics["energy_variance"])
    energy_variance_train = loss_stats_variance(diagnostics.get("loss_stats"))
    if np.isfinite(energy_variance_train):
        return float(4.0 * energy_variance_train), float(energy_variance_train)
    return float("nan"), float("nan")


def compute_and_save_delta_bank(
    *,
    ckpt_path,
    ckpt,
    driver,
    vstate,
    m_deltas: int,
    diag_shift_override: float | None = None,
    output_path=None,
    metadata: dict | None = None,
):
    step = int(ckpt["step"])
    config = ckpt["config"]
    system_name = normalize_system_name(config["system"])
    solve_lambda = resolve_solve_lambda(config, diag_shift_override=diag_shift_override)
    path = Path(output_path) if output_path is not None else delta_bank_output_path(ckpt_path, step)

    delta_rows = []
    train_target_norm_sq_per_delta = []
    energy_variance_train_per_delta = []
    for index in range(int(m_deltas)):
        delta_i, _, diagnostics = sr_diagnostics.solve_independent_sr(
            driver,
            vstate,
            return_diagnostics=True,
            diag_shift_override=diag_shift_override,
        )
        jax.block_until_ready(jax.tree_util.tree_leaves(delta_i))
        delta_rows.append(numpy_tree(delta_i))
        target_norm_sq_i, energy_variance_i = _train_target_metrics_from_solve(diagnostics)
        train_target_norm_sq_per_delta.append(target_norm_sq_i)
        energy_variance_train_per_delta.append(energy_variance_i)
        print(
            f"      delta {index + 1}/{m_deltas} solved",
            flush=True,
        )

    delta_bank = stack_tree(delta_rows)
    delta_mean = tree_mean(delta_bank)
    payload = {
        "system": system_name,
        "lambda": float(solve_lambda),
        "source_training_lambda": float(config["diag_shift"]),
        "step": int(step),
        "mode": str(driver.mode),
        "m_deltas": int(m_deltas),
        "delta_bank": delta_bank,
        "delta_mean": delta_mean,
        "train_target_norm_sq_per_delta": np.asarray(
            train_target_norm_sq_per_delta, dtype=np.float64
        ),
        "energy_variance_train_per_delta": np.asarray(
            energy_variance_train_per_delta, dtype=np.float64
        ),
        "train_target_norm_sq_mean": nanmean(train_target_norm_sq_per_delta),
        "energy_variance_train_mean": nanmean(energy_variance_train_per_delta),
        "source": "offline",
        "seed_base": int(config.get("seed", 0)),
    }
    if metadata:
        payload.update(dict(metadata))
    atomic_save(payload, path)
    return payload


def precompute_vit_checkpoint_delta_bank(
    ckpt_path,
    *,
    m_deltas: int,
    chunk_size_bwd: int | None = None,
    diag_shift_override: float | None = None,
    output_path=None,
    metadata: dict | None = None,
):
    from sr_filtering_nqs.large_scale.core.run_diagnostics import load_checkpoint, reconstruct_driver

    ckpt_path = Path(ckpt_path)
    ckpt = load_checkpoint(ckpt_path)
    step = int(ckpt["step"])
    bank_path = Path(output_path) if output_path is not None else delta_bank_output_path(ckpt_path, step)
    need_bank = not _delta_bank_matches(bank_path, m_deltas=m_deltas)

    result = {
        "checkpoint": ckpt_path,
        "step": step,
        "delta_bank_path": bank_path,
        "computed_delta_bank": False,
        "chunk_size_bwd": resolve_delta_bank_chunk_size_bwd(
            ckpt["config"],
            override=chunk_size_bwd,
        ),
    }
    if not need_bank:
        return result

    driver, vstate = reconstruct_driver(
        ckpt["config"],
        ckpt["parameters"],
        variables=ckpt.get("variables"),
        step=step,
        chunk_size_bwd_override=resolve_delta_bank_chunk_size_bwd(
            ckpt["config"],
            override=chunk_size_bwd,
        ),
    )
    compute_and_save_delta_bank(
        ckpt_path=ckpt_path,
        ckpt=ckpt,
        driver=driver,
        vstate=vstate,
        m_deltas=m_deltas,
        diag_shift_override=diag_shift_override,
        output_path=bank_path,
        metadata=metadata,
    )
    result["computed_delta_bank"] = True
    return result


def precompute_fnqs_checkpoint_delta_bank(
    ckpt_path,
    *,
    m_deltas: int,
    chunk_size_bwd: int | None = None,
    diag_shift_override: float | None = None,
    output_path=None,
    metadata: dict | None = None,
):
    from sr_filtering_nqs.large_scale.core.tfim_fnqs import reconstruct_training_driver

    ckpt_path = Path(ckpt_path)
    ckpt = _load_pickle(ckpt_path)
    step = int(ckpt["step"])
    bank_path = Path(output_path) if output_path is not None else delta_bank_output_path(ckpt_path, step)
    need_bank = not _delta_bank_matches(bank_path, m_deltas=m_deltas)

    result = {
        "checkpoint": ckpt_path,
        "step": step,
        "delta_bank_path": bank_path,
        "computed_delta_bank": False,
        "chunk_size_bwd": resolve_delta_bank_chunk_size_bwd(
            ckpt["config"],
            override=chunk_size_bwd,
        ),
    }
    if not need_bank:
        return result

    driver, vstate = reconstruct_training_driver(ckpt)
    chunk_size_bwd = resolve_delta_bank_chunk_size_bwd(
        ckpt["config"],
        override=chunk_size_bwd,
    )
    if chunk_size_bwd is not None:
        driver.chunk_size_bwd = int(chunk_size_bwd)
    compute_and_save_delta_bank(
        ckpt_path=ckpt_path,
        ckpt=ckpt,
        driver=driver,
        vstate=vstate,
        m_deltas=m_deltas,
        diag_shift_override=diag_shift_override,
        output_path=bank_path,
        metadata=metadata,
    )
    result["computed_delta_bank"] = True
    return result


def _load_existing_result(
    path: Path,
    *,
    n_val: int,
    m_deltas: int,
    required_diagnostics_version: int | None = None,
):
    if not path.exists():
        return None
    payload = _load_pickle(path)
    if int(payload.get("n_val", -1)) != int(n_val):
        return None
    if int(payload.get("m_deltas", -1)) != int(m_deltas):
        return None
    if required_diagnostics_version is not None:
        if int(payload.get("diagnostics_version", -1)) != int(required_diagnostics_version):
            return None
    return payload


def _stream_prediction_cache(
    *,
    driver,
    validation_vstate,
    delta_bank,
    mode: str,
    n_val: int,
    round_samples: int,
    jvp_chunk: int | None,
    delta_batch_size: int,
    n_replicas: int | None,
    compute_eloc_round,
):
    variables = validation_vstate.variables
    model_state, params = fcore.pop(variables, "params")
    m_deltas = int(np.asarray(jax.tree_util.tree_leaves(delta_bank)[0]).shape[0])
    n_channels = 2 if mode == "complex" else 1
    jvp_round_chunk = chunk_size_for_round(round_samples, jvp_chunk)

    with tempfile.TemporaryDirectory(prefix="large_scale_diag_cache_") as tmpdir:
        cache_dir = Path(tmpdir)
        prediction_bank = np.memmap(
            cache_dir / "predictions.dat",
            dtype=np.float64,
            mode="w+",
            shape=(m_deltas, n_channels, int(n_val)),
        )
        energy_channels = (
            np.empty((n_channels, int(n_val)), dtype=np.float64)
            if compute_eloc_round is not None
            else None
        )

        remaining = int(n_val)
        offset = 0
        round_index = 0
        total_rounds = int(math.ceil(float(n_val) / float(round_samples)))
        print(
            f"    Streaming validation: n_val={n_val} round_samples={round_samples} rounds={total_rounds}",
            flush=True,
        )

        while remaining > 0:
            round_index += 1
            take = min(round_samples, remaining)
            if n_replicas is not None and take % n_replicas != 0:
                raise ValueError(
                    f"Validation round size {take} must be divisible by n_replicas={n_replicas}"
                )
            samples_round = sample_validation_round(
                validation_vstate,
                take,
                n_replicas=n_replicas,
            )

            if compute_eloc_round is not None:
                eloc_round = compute_eloc_round(samples_round)
                _store_validation_block(
                    energy_channels,
                    _energy_channels_from_eloc(eloc_round, mode=mode),
                    sample_offset=offset,
                    n_replicas=n_replicas,
                )

            for start in range(0, m_deltas, int(delta_batch_size)):
                stop = min(start + int(delta_batch_size), m_deltas)
                delta_block = jax_tree(slice_tree(delta_bank, start, stop))
                predictions = sr_diagnostics.compute_O_delta_jvp_many(
                    validation_vstate._apply_fun,
                    params,
                    model_state,
                    jnp.asarray(samples_round),
                    delta_block,
                    mode=mode,
                    chunk_size=chunk_size_for_round(take, jvp_round_chunk),
                )
                _store_validation_block(
                    prediction_bank[start:stop],
                    _prediction_channels_from_bank(
                        np.asarray(predictions),
                        mode=mode,
                        sample_count=take,
                    ),
                    sample_offset=offset,
                    n_replicas=n_replicas,
                )

            remaining -= take
            offset += take
            print(
                f"      round {round_index}/{total_rounds} processed ({take} samples, remaining={remaining})",
                flush=True,
            )

        prediction_bank.flush()
        prediction_bank_array = np.asarray(prediction_bank)
        energy_array = None if energy_channels is None else np.asarray(energy_channels)
        return prediction_bank_array, energy_array


def process_vit_checkpoint_metrics(
    ckpt_path,
    *,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    jvp_chunk: int | None = None,
    conn_chunk: int,
    eval_chunk: int,
    chunk_size_bwd: int | None = None,
    delta_batch_size: int = 10,
    m_deltas: int = 100,
    use_prep_cache: bool = True,
    want_rval: bool,
    want_twobatch: bool,
    diag_shift_override: float | None = None,
    bank_payload: dict | None = None,
    output_paths: dict | None = None,
    result_metadata: dict | None = None,
    output_root=None,
    output_roots: dict | None = None,
    validation_seed_offset: int = 0,
):
    from sr_filtering_nqs.large_scale.core.run_diagnostics import (
        load_checkpoint,
        make_vit_state,
        reconstruct_driver,
        resolve_n_samples_for_config,
    )

    _ = (use_prep_cache, output_root, output_roots)

    ckpt_path = Path(ckpt_path)
    ckpt = load_checkpoint(ckpt_path)
    step = int(ckpt["step"])
    config = ckpt["config"]
    system_name = normalize_system_name(config["system"])
    if system_name != "j1j2":
        raise ValueError(f"ViT diagnostics only support j1j2, got {system_name}")

    out_rval = (
        Path(output_paths["rval"])
        if output_paths is not None and "rval" in output_paths
        else result_output_path(ckpt_path, "rval_step{step:06d}.pkl", step)
    )
    out_twobatch = (
        Path(output_paths["twobatch"])
        if output_paths is not None and "twobatch" in output_paths
        else result_output_path(ckpt_path, "twobatch_step{step:06d}.pkl", step)
    )

    if n_val is None:
        n_val = heldout_samples_for_system(system_name)

    rval_result = None
    twobatch_result = None
    if want_rval:
        rval_result = _load_existing_result(out_rval, n_val=n_val, m_deltas=m_deltas)
    if want_twobatch:
        twobatch_result = _load_existing_result(out_twobatch, n_val=n_val, m_deltas=m_deltas)

    need_rval = bool(want_rval and rval_result is None)
    need_twobatch = bool(want_twobatch and twobatch_result is None)
    if not need_rval and not need_twobatch:
        return {"rval": rval_result, "twobatch": twobatch_result}

    n_train = resolve_n_samples_for_config(config, step=step)
    if bank_payload is None:
        bank_payload = load_required_delta_bank(ckpt_path=ckpt_path, ckpt=ckpt, m_deltas=m_deltas)
    delta_bank = bank_payload["delta_bank"]
    delta_mean = jax_tree(bank_payload["delta_mean"])

    driver, vstate = reconstruct_driver(
        config,
        ckpt["parameters"],
        variables=ckpt.get("variables"),
        step=step,
        chunk_size_bwd_override=chunk_size_bwd,
    )

    train_metrics = train_metrics_from_checkpoint(ckpt)
    if train_metrics is None:
        train_metrics = compute_train_metrics_fallback(driver, vstate, delta_mean)

    round_samples = resolve_round_samples(
        system_name=system_name,
        total_n_val=n_val,
        requested_round_samples=validation_round_samples,
        default_round_samples=n_train,
    )
    if round_samples == n_train and validation_seed_offset == 0:
        validation_vstate = vstate
    else:
        _, _, validation_vstate = make_vit_state(
            config,
            ckpt["parameters"],
            variables=ckpt.get("variables"),
            step=step,
            n_samples_override=round_samples,
            chunk_size_override=min(int(config.get("chunk_size", 4096)), round_samples),
            seed_override=int(config.get("seed", 0)) + 20_000 + step + validation_seed_offset,
        )

    compute_eloc_round = None
    if need_rval:
        def compute_eloc_round(samples_round):
            return compute_vit_local_energies_round(
                validation_vstate._apply_fun,
                validation_vstate.variables,
                samples_round,
                driver._ham,
                conn_chunk=conn_chunk,
                eval_chunk=eval_chunk,
            )

    prediction_bank, energy_channels = _stream_prediction_cache(
        driver=driver,
        validation_vstate=validation_vstate,
        delta_bank=delta_bank,
        mode=driver.mode,
        n_val=int(n_val),
        round_samples=int(round_samples),
        jvp_chunk=jvp_chunk,
        delta_batch_size=delta_batch_size,
        n_replicas=None,
        compute_eloc_round=compute_eloc_round,
    )

    n_params = count_parameters(vstate.parameters)
    lam = resolve_solve_lambda(config, diag_shift_override=diag_shift_override)
    if need_rval:
        val_metrics = finalize_multidelta_rval(
            prediction_bank,
            energy_channels,
            n_replicas=None,
        )
        rval_result = build_rval_result(
            model_type="vit",
            system_name=system_name,
            lam=lam,
            step=step,
            n_params=n_params,
            n_train=int(n_train),
            n_val=int(n_val),
            m_deltas=int(bank_payload["m_deltas"]),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metadata=result_metadata,
        )
        atomic_save(rval_result, out_rval)

    if need_twobatch:
        twobatch_metrics = finalize_multidelta_twobatch(
            prediction_bank,
            n_replicas=None,
            train_target_norm_sq=float(bank_payload["train_target_norm_sq_mean"]),
        )
        twobatch_result = build_twobatch_result(
            system_name=system_name,
            model_type="vit",
            lam=lam,
            step=step,
            n_val=int(n_val),
            m_deltas=int(bank_payload["m_deltas"]),
            train_target_norm_sq=float(bank_payload["train_target_norm_sq_mean"]),
            energy_variance_train=float(bank_payload["energy_variance_train_mean"]),
            twobatch_metrics=twobatch_metrics,
            metadata=result_metadata,
        )
        atomic_save(twobatch_result, out_twobatch)

    return {"rval": rval_result, "twobatch": twobatch_result}


def process_fnqs_checkpoint_metrics(
    ckpt_path,
    *,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    jvp_chunk: int | None = None,
    chunk_size_bwd: int | None = None,
    delta_batch_size: int = 10,
    m_deltas: int = 100,
    use_prep_cache: bool = True,
    want_rval: bool,
    want_twobatch: bool,
    diag_shift_override: float | None = None,
    bank_payload: dict | None = None,
    output_paths: dict | None = None,
    result_metadata: dict | None = None,
    output_root=None,
    output_roots: dict | None = None,
    validation_seed_offset: int = 0,
):
    from sr_filtering_nqs.large_scale.core.tfim_fnqs import (
        count_parameters as fnqs_count_parameters,
        load_or_create_heldout_h_values,
        make_state_from_config,
        reconstruct_training_driver,
    )

    _ = (use_prep_cache, output_root, output_roots)

    ckpt_path = Path(ckpt_path)
    ckpt = _load_pickle(ckpt_path)
    step = int(ckpt["step"])
    config = ckpt["config"]
    system_name = normalize_system_name(config["system"])
    if system_name != "tfim":
        raise ValueError(f"FNQS diagnostics only support tfim, got {system_name}")

    diagnostic_steps = {int(x) for x in config.get("diagnostic_steps", [])}
    if diagnostic_steps and step not in diagnostic_steps:
        return {"rval": None, "twobatch": None}

    out_rval = (
        Path(output_paths["rval"])
        if output_paths is not None and "rval" in output_paths
        else result_output_path(ckpt_path, "rval_step{step:06d}.pkl", step)
    )
    out_twobatch = (
        Path(output_paths["twobatch"])
        if output_paths is not None and "twobatch" in output_paths
        else result_output_path(ckpt_path, "twobatch_step{step:06d}.pkl", step)
    )

    if n_val is None:
        n_val = int(config.get("validation_n_samples", heldout_samples_for_system(system_name)))

    rval_result = None
    twobatch_result = None
    if want_rval:
        rval_result = _load_existing_result(
            out_rval,
            n_val=n_val,
            m_deltas=m_deltas,
            required_diagnostics_version=FNQS_DIAGNOSTICS_VERSION,
        )
    if want_twobatch:
        twobatch_result = _load_existing_result(
            out_twobatch,
            n_val=n_val,
            m_deltas=m_deltas,
            required_diagnostics_version=FNQS_DIAGNOSTICS_VERSION,
        )

    need_rval = bool(want_rval and rval_result is None)
    need_twobatch = bool(want_twobatch and twobatch_result is None)
    if not need_rval and not need_twobatch:
        return {"rval": rval_result, "twobatch": twobatch_result}

    if bank_payload is None:
        bank_payload = load_required_delta_bank(ckpt_path=ckpt_path, ckpt=ckpt, m_deltas=m_deltas)
    delta_bank = bank_payload["delta_bank"]
    delta_mean = jax_tree(bank_payload["delta_mean"])

    driver, vstate = reconstruct_training_driver(ckpt)
    if chunk_size_bwd is not None:
        driver.chunk_size_bwd = int(chunk_size_bwd)

    train_metrics = train_metrics_from_checkpoint(ckpt)
    if train_metrics is None:
        train_metrics = compute_train_metrics_fallback(driver, vstate, delta_mean)

    heldout_h = load_or_create_heldout_h_values(large_scale_root_dir(ckpt_path), config)
    n_replicas = int(len(heldout_h))
    round_samples = resolve_round_samples(
        system_name=system_name,
        total_n_val=n_val,
        requested_round_samples=validation_round_samples,
        default_round_samples=int(config["n_samples"]),
        n_replicas=n_replicas,
    )
    if n_val % n_replicas != 0:
        raise ValueError(
            f"n_val={n_val} must be divisible by validation_h_count={n_replicas}"
        )

    validation_vstate = make_state_from_config(
        config,
        parameter_array=heldout_h,
        n_samples=round_samples,
        seed=int(config.get("seed", 0)) + 20_000 + step + validation_seed_offset,
        chunk_size=min(
            int(config.get("validation_chunk_size", config.get("chunk_size", round_samples))),
            round_samples,
        ),
    )
    validation_vstate.variables = vstate.variables

    compute_eloc_round = None
    if need_rval:
        def compute_eloc_round(samples_round):
            return compute_driver_local_energies_round(
                driver,
                validation_vstate,
                samples_round,
            )

    prediction_bank, energy_channels = _stream_prediction_cache(
        driver=driver,
        validation_vstate=validation_vstate,
        delta_bank=delta_bank,
        mode=driver.mode,
        n_val=int(n_val),
        round_samples=int(round_samples),
        jvp_chunk=jvp_chunk,
        delta_batch_size=delta_batch_size,
        n_replicas=n_replicas,
        compute_eloc_round=compute_eloc_round,
    )

    n_params = fnqs_count_parameters(vstate.parameters)
    lam = resolve_solve_lambda(config, diag_shift_override=diag_shift_override)
    if need_rval:
        val_metrics = finalize_multidelta_rval(
            prediction_bank,
            energy_channels,
            n_replicas=n_replicas,
        )
        rval_result = build_rval_result(
            model_type="fnqs",
            system_name=system_name,
            lam=lam,
            step=step,
            n_params=n_params,
            n_train=int(config["n_samples"]),
            n_val=int(n_val),
            m_deltas=int(bank_payload["m_deltas"]),
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            metadata=result_metadata,
        )
        atomic_save(rval_result, out_rval)

    if need_twobatch:
        twobatch_metrics = finalize_multidelta_twobatch(
            prediction_bank,
            n_replicas=n_replicas,
            train_target_norm_sq=float(bank_payload["train_target_norm_sq_mean"]),
        )
        twobatch_result = build_twobatch_result(
            system_name=system_name,
            model_type="fnqs",
            lam=lam,
            step=step,
            n_val=int(n_val),
            m_deltas=int(bank_payload["m_deltas"]),
            train_target_norm_sq=float(bank_payload["train_target_norm_sq_mean"]),
            energy_variance_train=float(bank_payload["energy_variance_train_mean"]),
            twobatch_metrics=twobatch_metrics,
            metadata=result_metadata,
        )
        atomic_save(twobatch_result, out_twobatch)

    return {"rval": rval_result, "twobatch": twobatch_result}
