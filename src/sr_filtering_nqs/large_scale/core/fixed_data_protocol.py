from __future__ import annotations

import json
import math
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import core as fcore
from scipy.optimize import minimize
from nqs_support_core import distributed

from sr_filtering_nqs.large_scale.core.common import (
    SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA,
    SHARED_FIG2_DIAGNOSTIC_LAMBDAS,
    default_delta_bank_chunk_size_bwd,
    format_lambda_dir,
    heldout_samples_for_system,
    large_scale_root_dir,
    normalize_system_name,
)
from sr_filtering_nqs.large_scale.core.diagnostic_core import (
    _stream_prediction_cache,
    compute_driver_local_energies_round,
    compute_vit_local_energies_round,
    sample_validation_round,
)
from sr_filtering_nqs.large_scale.core.diagnostic_state import atomic_save
from sr_filtering_nqs.large_scale.core.multidelta_metrics import center_samples
from sr_filtering_nqs.large_scale.core.multilambda_sr import (
    PreparedBatch,
    _prepared_batch_from_samples,
    base_solve_record,
    centered_prediction_vector,
    centered_target_vector,
    compute_stacking_mixture_weights,
    jax_tree,
    restore_driver_state,
    sample_prepared_batch,
    snapshot_driver_state,
    stack_tree,
    uniform_mixture_weights,
    weighted_tree_sum,
)
from sr_filtering_nqs.large_scale.core.run_diagnostics import (
    load_checkpoint as load_vit_checkpoint,
    make_vit_state,
    reconstruct_driver as reconstruct_vit_driver,
    resolve_n_samples_for_config,
)
from sr_filtering_nqs.large_scale.core.shared_checkpoint_protocol import (
    checkpoint_step_from_path,
    discover_source_checkpoints,
)
from sr_filtering_nqs.large_scale.core.sr_diagnostics import compute_O_delta_jvp_many
from sr_filtering_nqs.large_scale.core.tfim_fnqs import (
    count_parameters as count_tfim_parameters,
    load_or_create_heldout_h_values,
    make_state_from_config as make_tfim_state,
    reconstruct_training_driver,
)


WORKSPACE_NAME = os.environ.get("FIXED_DATA_WORKSPACE_NAME", "20260422")
SCHEMA_VERSION = 1
FIXED_DATA_REPEAT_COUNT = 20
FIXED_DATA_BATCH_SLOTS = 4
FIG3_STACKING_HOLDOUT_FRACTION = 0.25
FIG3_LAMBDA_QUANTILES = (0.9, 0.7, 0.4, 0.1)
FIG3_METHOD_IDS = (
    "single_best_fixed_lambda",
    "same_batch_multilambda_uniform_k4",
    "same_batch_multilambda_stacking_k4",
    "indep_single_lambda_uniform_k4",
    "indep_multilambda_uniform_k4",
    "indep_multilambda_stacking_k4",
)
SIMPLEX_OBJECTIVE_PENALTY = 1e24
LOO_DENOM_EPS = 1e-12
TRAIN_SEED_OFFSET = 10_000
VALIDATION_SEED_OFFSET = 20_000
SPECTRUM_SEED_OFFSET = 30_000


def workspace_root(base_root) -> Path:
    return large_scale_root_dir(base_root) / WORKSPACE_NAME


def cache_root(base_root) -> Path:
    return workspace_root(base_root) / "cache"


def cache_step_dir(base_root, system_name: str, step: int) -> Path:
    system_name = normalize_system_name(system_name)
    return cache_root(base_root) / system_name / f"step{int(step):06d}"


def train_batches_cache_path(base_root, system_name: str, step: int) -> Path:
    return cache_step_dir(base_root, system_name, step) / "train_batches.pkl"


def validation_cache_path(base_root, system_name: str, step: int) -> Path:
    return cache_step_dir(base_root, system_name, step) / "validation.pkl"


def ntk_cache_dir(base_root, system_name: str, step: int) -> Path:
    return cache_step_dir(base_root, system_name, step) / "ntk"


def spectrum_ref_path(base_root, system_name: str, step: int) -> Path:
    return ntk_cache_dir(base_root, system_name, step) / "spectrum_ref.npy"


def spectrum_ref_meta_path(base_root, system_name: str, step: int) -> Path:
    return ntk_cache_dir(base_root, system_name, step) / "spectrum_ref_meta.json"


def results_root(base_root) -> Path:
    return workspace_root(base_root) / "results"


def plots_root(base_root) -> Path:
    return workspace_root(base_root) / "plots"


def logs_root(base_root) -> Path:
    return workspace_root(base_root) / "logs"


def fig2_results_dir(base_root, system_name: str) -> Path:
    system_name = normalize_system_name(system_name)
    return results_root(base_root) / "fig2" / system_name


def fig2_step_dir(base_root, system_name: str, step: int) -> Path:
    return fig2_results_dir(base_root, system_name) / f"step{int(step):06d}"


def fig2_checkpoint_lambda_path(base_root, system_name: str, step: int, diagnostic_lambda: float) -> Path:
    token = format_lambda_dir(float(diagnostic_lambda))
    return fig2_step_dir(base_root, system_name, step) / f"diag_{token}.pkl"


def best_fixed_lambda_path(base_root) -> Path:
    return results_root(base_root) / "fig2" / "best_fixed_lambda.json"


def fig3_results_dir(base_root, system_name: str) -> Path:
    system_name = normalize_system_name(system_name)
    return results_root(base_root) / "fig3" / system_name


def fig3_step_dir(base_root, system_name: str, step: int) -> Path:
    return fig3_results_dir(base_root, system_name) / f"step{int(step):06d}"


def fig3_checkpoint_eval_path(base_root, system_name: str, step: int, *, k: int = 4) -> Path:
    return fig3_step_dir(base_root, system_name, step) / f"k{k}_eval.pkl"


def fig2_result_files(base_root, system_name: str) -> list[Path]:
    return sorted(fig2_results_dir(base_root, system_name).glob("step*/diag_*.pkl"))


def fig3_result_files(base_root, system_name: str, *, k: int = 4) -> list[Path]:
    return sorted(fig3_results_dir(base_root, system_name).glob(f"step*/k{k}_eval.pkl"))


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _atomic_json_dump(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json.tmp",
        dir=str(path.parent),
        delete=False,
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def _atomic_numpy_save(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".npy.tmp",
        dir=str(path.parent),
        delete=False,
    ) as handle:
        np.save(handle, np.asarray(array))
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def count_parameters(params) -> int:
    return int(sum(np.prod(np.asarray(leaf).shape) for leaf in jax.tree_util.tree_leaves(params)))


def default_diagnostic_lambdas() -> tuple[float, ...]:
    return tuple(float(value) for value in SHARED_FIG2_DIAGNOSTIC_LAMBDAS)


def _default_seed_base(config: dict[str, Any], step: int, offset: int) -> int:
    return int(config.get("seed", 0)) + int(offset) + int(step)


def _resolve_seed_base(
    config: dict[str, Any],
    step: int,
    requested_seed_base: int | None,
    offset: int,
) -> int:
    if requested_seed_base is not None:
        return int(requested_seed_base)
    return _default_seed_base(config, int(step), int(offset))


def _clear_vstate_sample_caches(vstate) -> None:
    if hasattr(vstate, "_samples"):
        vstate._samples = None
    for attr in (
        "_samples_distributions",
        "_log_probabilities_distributions",
        "_samples_distribution_resampling_cache",
        "_log_probabilities_distribution_resampling_cache",
    ):
        cache = getattr(vstate, attr, None)
        if isinstance(cache, dict):
            cache.clear()


def reset_vstate_sampler_to_seed(vstate, seed: int | None) -> None:
    if seed is None:
        return
    seed = int(seed)
    key = jax.random.PRNGKey(seed)
    try:
        sampler = vstate.sampler
        model = getattr(vstate, "_sampler_model", getattr(vstate, "_model", None))
        variables = getattr(vstate, "_sampler_variables", getattr(vstate, "variables", None))
        sampler_state = sampler.init_state(model, variables, seed=key)
        if hasattr(vstate, "sampler_states"):
            vstate.sampler_states = {}
        if hasattr(vstate, "_sampler_states_previous"):
            vstate._sampler_states_previous = {}
        vstate.sampler_state = sampler_state
        if hasattr(vstate, "_sampler_state_previous"):
            vstate._sampler_state_previous = sampler_state
        if hasattr(vstate, "_sampler_seed"):
            vstate._sampler_seed = key
    except Exception:
        if not hasattr(vstate, "replace_sampler_seed"):
            raise
        vstate.replace_sampler_seed(seed)
    _clear_vstate_sample_caches(vstate)


def _live_batch_seed(train_seed_base: int, *, repeat_index: int, slot_index: int, batch_slots: int) -> int:
    return int(train_seed_base) + int(repeat_index) * int(batch_slots) + int(slot_index)


def _sample_live_prepared_batch(driver, *, seed: int, chain_name: str) -> PreparedBatch:
    reset_vstate_sampler_to_seed(driver.state, int(seed))
    return sample_prepared_batch(driver, chain_name=chain_name)


def figure3_method_ids() -> tuple[str, ...]:
    return FIG3_METHOD_IDS


def canonical_fixed_data_checkpoints(
    base_root,
    system_name: str,
    *,
    source_training_lambda: float = SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA,
    source_steps: str | None = "all",
    latest_only: bool = False,
) -> list[tuple[int, Path]]:
    return discover_source_checkpoints(
        base_root,
        normalize_system_name(system_name),
        source_training_lambda=float(source_training_lambda),
        source_steps=source_steps,
        latest_only=latest_only,
    )


def _load_training_driver(ckpt_path, *, system_name: str) -> dict[str, Any]:
    system_name = normalize_system_name(system_name)
    if system_name == "tfim":
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        driver, vstate = reconstruct_training_driver(ckpt)
        return {
            "system": system_name,
            "config": ckpt["config"],
            "step": int(ckpt["step"]),
            "driver": driver,
            "vstate": vstate,
            "checkpoint": ckpt,
        }

    ckpt = load_vit_checkpoint(ckpt_path)
    driver, vstate = reconstruct_vit_driver(
        ckpt["config"],
        ckpt["parameters"],
        variables=ckpt.get("variables"),
        step=int(ckpt["step"]),
        chunk_size_bwd_override=ckpt["config"].get("chunk_size_bwd"),
    )
    return {
        "system": system_name,
        "config": ckpt["config"],
        "step": int(ckpt["step"]),
        "driver": driver,
        "vstate": vstate,
        "checkpoint": ckpt,
    }


def _train_batches_cache_compatible(payload: dict[str, Any], *, repeat_count: int, batch_slots: int) -> bool:
    return (
        int(payload.get("schema_version", -1)) == SCHEMA_VERSION
        and int(payload.get("repeat_count", -1)) == int(repeat_count)
        and int(payload.get("batch_slots", -1)) == int(batch_slots)
    )


def ensure_train_batches_cache(
    ckpt_path,
    *,
    base_root,
    system_name: str,
    repeat_count: int = FIXED_DATA_REPEAT_COUNT,
    batch_slots: int = FIXED_DATA_BATCH_SLOTS,
) -> dict[str, Any]:
    system_name = normalize_system_name(system_name)
    step = int(checkpoint_step_from_path(ckpt_path))
    path = train_batches_cache_path(base_root, system_name, step)
    if path.exists():
        payload = _load_pickle(path)
        if _train_batches_cache_compatible(payload, repeat_count=repeat_count, batch_slots=batch_slots):
            return payload

    training = _load_training_driver(ckpt_path, system_name=system_name)
    driver = training["driver"]
    snapshot = snapshot_driver_state(driver)
    repeats = []
    try:
        for repeat_index in range(int(repeat_count)):
            slots = []
            for slot_index in range(int(batch_slots)):
                batch = sample_prepared_batch(
                    driver,
                    chain_name=f"fixed_data_r{repeat_index:03d}_s{slot_index:02d}",
                )
                slots.append(
                    {
                        "samples": np.asarray(batch.samples),
                        "local_loss": np.asarray(batch.local_loss),
                        "sample_count": int(batch.sample_count),
                        "n_replicas": None if batch.n_replicas is None else int(batch.n_replicas),
                    }
                )
            repeats.append({"slots": slots})
    finally:
        restore_driver_state(driver, snapshot)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "workspace": WORKSPACE_NAME,
        "system": system_name,
        "step": int(step),
        "repeat_count": int(repeat_count),
        "batch_slots": int(batch_slots),
        "sample_count_per_slot": int(repeats[0]["slots"][0]["sample_count"]) if repeats else 0,
        "source_checkpoint": str(Path(ckpt_path).resolve()),
        "source_training_lambda": float(training["config"]["diag_shift"]),
        "repeats": repeats,
    }
    atomic_save(payload, path)
    return payload


def _validation_cache_compatible(payload: dict[str, Any], *, n_val: int, round_samples: int) -> bool:
    return (
        int(payload.get("schema_version", -1)) == SCHEMA_VERSION
        and int(payload.get("n_val", -1)) == int(n_val)
        and int(payload.get("round_samples", -1)) == int(round_samples)
    )


def _store_sample_block(buffer: np.ndarray, values: np.ndarray, *, sample_offset: int, n_replicas: int | None) -> None:
    values = np.asarray(values)
    take = int(values.shape[0])

    if n_replicas is None:
        buffer[sample_offset : sample_offset + take] = values
        return

    total_count = int(buffer.shape[0])
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
    buffer_view = buffer.reshape(n_replicas, per_replica_total, *buffer.shape[1:])
    value_view = values.reshape(n_replicas, per_replica_take, *values.shape[1:])
    buffer_view[:, per_replica_offset : per_replica_offset + per_replica_take, ...] = value_view


def _store_channel_block(
    buffer: np.ndarray,
    values: np.ndarray,
    *,
    sample_offset: int,
    n_replicas: int | None,
) -> None:
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
    buffer_view = buffer.reshape(buffer.shape[:-1] + (n_replicas, per_replica_total))
    value_view = values.reshape(values.shape[:-1] + (n_replicas, per_replica_take))
    buffer_view[..., per_replica_offset : per_replica_offset + per_replica_take] = value_view


def _energy_channels_from_eloc(eloc, *, mode: str | None) -> np.ndarray:
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


def ensure_validation_cache(
    ckpt_path,
    *,
    base_root,
    system_name: str,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    conn_chunk: int = 4096,
    eval_chunk: int = 4096,
) -> dict[str, Any]:
    system_name = normalize_system_name(system_name)
    step = int(checkpoint_step_from_path(ckpt_path))
    training = _load_training_driver(ckpt_path, system_name=system_name)
    config = training["config"]
    if n_val is None:
        n_val = int(heldout_samples_for_system(system_name))

    if system_name == "tfim":
        default_round_samples = int(config["n_samples"])
    else:
        default_round_samples = int(resolve_n_samples_for_config(config, step=step))
    round_samples = (
        int(validation_round_samples)
        if validation_round_samples is not None
        else default_round_samples
    )

    path = validation_cache_path(base_root, system_name, step)
    if path.exists():
        payload = _load_pickle(path)
        if _validation_cache_compatible(payload, n_val=n_val, round_samples=round_samples):
            return payload

    if system_name == "tfim":
        heldout_h = load_or_create_heldout_h_values(base_root, config)
        n_replicas = int(len(heldout_h))
        if int(n_val) % n_replicas != 0:
            raise ValueError(
                f"n_val={n_val} must be divisible by validation_h_count={n_replicas}"
            )
        if int(round_samples) % n_replicas != 0:
            raise ValueError(
                f"validation_round_samples={round_samples} must be divisible by validation_h_count={n_replicas}"
            )
        validation_vstate = make_tfim_state(
            config,
            parameter_array=heldout_h,
            n_samples=int(round_samples),
            seed=int(config.get("seed", 0)) + 20_000 + step,
            chunk_size=min(
                int(config.get("validation_chunk_size", config.get("chunk_size", round_samples))),
                int(round_samples),
            ),
        )
        validation_vstate.variables = training["vstate"].variables
        compute_eloc_round = lambda samples_round: compute_driver_local_energies_round(  # noqa: E731
            training["driver"],
            validation_vstate,
            samples_round,
        )
        mode = training["driver"].mode
    else:
        n_replicas = None
        ckpt = training["checkpoint"]
        _, _, validation_vstate = make_vit_state(
            config,
            ckpt["parameters"],
            variables=ckpt.get("variables"),
            step=step,
            n_samples_override=int(round_samples),
            chunk_size_override=min(int(config.get("chunk_size", 4096)), int(round_samples)),
            seed_override=int(config.get("seed", 0)) + 20_000 + step,
        )
        compute_eloc_round = lambda samples_round: compute_vit_local_energies_round(  # noqa: E731
            validation_vstate._apply_fun,
            validation_vstate.variables,
            samples_round,
            training["driver"]._ham,
            conn_chunk=int(conn_chunk),
            eval_chunk=int(eval_chunk),
        )
        mode = training["driver"].mode

    remaining = int(n_val)
    offset = 0
    samples_buffer = None
    energy_buffer = None

    while remaining > 0:
        take = min(int(round_samples), remaining)
        samples_round = sample_validation_round(
            validation_vstate,
            take,
            n_replicas=n_replicas,
        )
        if samples_buffer is None:
            samples_buffer = np.empty((int(n_val), samples_round.shape[-1]), dtype=np.asarray(samples_round).dtype)
            n_channels = 2 if mode == "complex" else 1
            energy_buffer = np.empty((n_channels, int(n_val)), dtype=np.float64)

        _store_sample_block(samples_buffer, np.asarray(samples_round), sample_offset=offset, n_replicas=n_replicas)
        eloc_round = compute_eloc_round(samples_round)
        _store_channel_block(
            energy_buffer,
            _energy_channels_from_eloc(eloc_round, mode=mode),
            sample_offset=offset,
            n_replicas=n_replicas,
        )
        remaining -= take
        offset += take

    if samples_buffer is None or energy_buffer is None:
        raise RuntimeError(f"Failed to cache validation payload for {ckpt_path}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "workspace": WORKSPACE_NAME,
        "system": system_name,
        "step": int(step),
        "n_val": int(n_val),
        "round_samples": int(round_samples),
        "n_replicas": None if n_replicas is None else int(n_replicas),
        "mode": mode,
        "samples": samples_buffer,
        "energy_channels": energy_buffer,
        "source_checkpoint": str(Path(ckpt_path).resolve()),
        "source_training_lambda": float(config["diag_shift"]),
    }
    atomic_save(payload, path)
    return payload


def ensure_reference_spectrum(
    ckpt_path,
    *,
    base_root,
    system_name: str,
    spectrum_seed_base: int | None = None,
) -> np.ndarray:
    system_name = normalize_system_name(system_name)
    step = int(checkpoint_step_from_path(ckpt_path))
    npy_path = spectrum_ref_path(base_root, system_name, step)
    meta_path = spectrum_ref_meta_path(base_root, system_name, step)
    if npy_path.exists() and meta_path.exists():
        if spectrum_seed_base is None:
            return np.asarray(np.load(npy_path), dtype=np.float64)
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        if meta.get("spectrum_seed") == int(spectrum_seed_base):
            return np.asarray(np.load(npy_path), dtype=np.float64)

    training = _load_training_driver(ckpt_path, system_name=system_name)
    driver = training["driver"]
    spectrum_seed = _resolve_seed_base(
        training["config"],
        int(step),
        spectrum_seed_base,
        SPECTRUM_SEED_OFFSET,
    )
    snapshot = snapshot_driver_state(driver)
    try:
        batch = _sample_live_prepared_batch(
            driver,
            seed=spectrum_seed,
            chain_name="fixed_data_spectrum_ref",
        )
        record = base_solve_record(
            driver,
            batch,
            diag_shift=float(training["config"]["diag_shift"]),
            collect_residual_info=True,
        )
    finally:
        restore_driver_state(driver, snapshot)

    if record.ntk_eigenvalues is None:
        raise RuntimeError(f"Failed to compute NTK spectrum for {ckpt_path}")

    spectrum = np.asarray(record.ntk_eigenvalues, dtype=np.float64)
    _atomic_numpy_save(spectrum, npy_path)
    _atomic_json_dump(
        {
            "schema_version": SCHEMA_VERSION,
            "workspace": WORKSPACE_NAME,
            "system": system_name,
            "step": int(step),
            "source_checkpoint": str(Path(ckpt_path).resolve()),
            "source_training_lambda": float(training["config"]["diag_shift"]),
            "reference_lambda": float(training["config"]["diag_shift"]),
            "sample_count": int(batch.sample_count),
            "n_replicas": None if batch.n_replicas is None else int(batch.n_replicas),
            "spectrum_size": int(spectrum.shape[0]),
            "spectrum_seed": int(spectrum_seed),
        },
        meta_path,
    )
    return spectrum


def load_cached_train_batches(base_root, system_name: str, step: int) -> dict[str, Any]:
    path = train_batches_cache_path(base_root, system_name, step)
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed-data training batches: {path}")
    return _load_pickle(path)


def load_cached_validation_payload(base_root, system_name: str, step: int) -> dict[str, Any]:
    path = validation_cache_path(base_root, system_name, step)
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed-data validation payload: {path}")
    return _load_pickle(path)


def load_cached_spectrum(base_root, system_name: str, step: int) -> np.ndarray:
    path = spectrum_ref_path(base_root, system_name, step)
    if not path.exists():
        raise FileNotFoundError(f"Missing fixed-data NTK spectrum: {path}")
    return np.asarray(np.load(path), dtype=np.float64)


def build_prepared_batch_from_slot(driver, slot_payload: dict[str, Any]) -> PreparedBatch:
    batch = _prepared_batch_from_samples(
        driver,
        jnp.asarray(np.asarray(slot_payload["samples"])),
    )
    if "local_loss" in slot_payload:
        batch.local_loss = jnp.asarray(np.asarray(slot_payload["local_loss"]))
    if slot_payload.get("n_replicas") is not None:
        batch.n_replicas = int(slot_payload["n_replicas"])
    return batch


def holdout_partition_counts(
    total_count: int,
    *,
    n_replicas: int | None,
    holdout_fraction: float = FIG3_STACKING_HOLDOUT_FRACTION,
    sample_axis_granularity: int = 1,
) -> dict[str, int | None]:
    total_count = int(total_count)
    sample_axis_granularity = max(1, int(sample_axis_granularity))
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")

    def _nearest_aligned_count(target: int, *, minimum: int, maximum: int, granularity: int) -> int:
        if minimum > maximum:
            raise ValueError(
                f"Cannot fit an aligned holdout split inside [{minimum}, {maximum}] "
                f"with granularity={granularity}"
            )
        if granularity <= 1:
            return min(max(int(target), int(minimum)), int(maximum))

        lower = (int(target) // int(granularity)) * int(granularity)
        upper = lower if lower == int(target) else lower + int(granularity)
        candidates = [value for value in (lower, upper) if minimum <= value <= maximum]
        if not candidates:
            raise ValueError(
                f"No aligned split near target={target} within [{minimum}, {maximum}] "
                f"for granularity={granularity}"
            )
        return min(candidates, key=lambda value: (abs(value - int(target)), -value))

    if n_replicas is None:
        if total_count % sample_axis_granularity != 0:
            raise ValueError(
                f"total_count={total_count} must be divisible by sample_axis_granularity="
                f"{sample_axis_granularity}"
            )
        holdout_count = _nearest_aligned_count(
            int(round(float(total_count) * float(holdout_fraction))),
            minimum=sample_axis_granularity,
            maximum=total_count - sample_axis_granularity,
            granularity=sample_axis_granularity,
        )
        return {
            "solve_count": int(total_count - holdout_count),
            "holdout_count": int(holdout_count),
            "solve_n_replicas": None,
            "holdout_n_replicas": None,
        }

    n_replicas = int(n_replicas)
    if total_count % n_replicas != 0:
        raise ValueError(
            f"total_count={total_count} must be divisible by n_replicas={n_replicas}"
        )
    if total_count % sample_axis_granularity != 0:
        raise ValueError(
            f"total_count={total_count} must be divisible by sample_axis_granularity="
            f"{sample_axis_granularity}"
        )
    per_replica = total_count // n_replicas
    replica_granularity = max(1, sample_axis_granularity // math.gcd(sample_axis_granularity, per_replica))
    holdout_replicas = _nearest_aligned_count(
        int(round(float(n_replicas) * float(holdout_fraction))),
        minimum=replica_granularity,
        maximum=n_replicas - replica_granularity,
        granularity=replica_granularity,
    )
    solve_replicas = n_replicas - holdout_replicas
    return {
        "solve_count": int(solve_replicas * per_replica),
        "holdout_count": int(holdout_replicas * per_replica),
        "solve_n_replicas": int(solve_replicas),
        "holdout_n_replicas": int(holdout_replicas),
    }


def split_prepared_batch_for_same_batch_stacking(
    driver,
    batch: PreparedBatch,
    *,
    holdout_fraction: float = FIG3_STACKING_HOLDOUT_FRACTION,
) -> tuple[PreparedBatch, PreparedBatch, dict[str, int | None]]:
    sample_axis_granularity = 1
    if distributed.mode() == "sharding":
        sample_axis_granularity = max(1, int(distributed.device_count()))
    counts = holdout_partition_counts(
        int(batch.sample_count),
        n_replicas=batch.n_replicas,
        holdout_fraction=holdout_fraction,
        sample_axis_granularity=sample_axis_granularity,
    )
    solve_count = int(counts["solve_count"])
    samples = batch.samples
    solve_batch = _prepared_batch_from_samples(
        driver,
        samples[:solve_count],
        variables=batch.variables,
    )
    holdout_batch = _prepared_batch_from_samples(
        driver,
        samples[solve_count : solve_count + int(counts["holdout_count"])],
        variables=batch.variables,
    )
    solve_batch.n_replicas = counts["solve_n_replicas"]
    holdout_batch.n_replicas = counts["holdout_n_replicas"]
    return solve_batch, holdout_batch, counts


def _project_to_simplex(weights: np.ndarray) -> np.ndarray:
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


def _simplex_minimize(objective, dimension: int, *, x0: np.ndarray | None = None) -> np.ndarray:
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension}")
    start = (
        _project_to_simplex(np.asarray(x0, dtype=np.float64))
        if x0 is not None
        else uniform_mixture_weights(dimension)
    )
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * int(dimension),
        constraints=(
            {
                "type": "eq",
                "fun": lambda w: float(np.sum(w) - 1.0),
                "jac": lambda w: np.ones_like(w, dtype=np.float64),
            },
        ),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        if result.x is None:
            raise RuntimeError(f"Simplex optimization failed: {result.message}")
        return _project_to_simplex(np.asarray(result.x, dtype=np.float64))
    return _project_to_simplex(np.asarray(result.x, dtype=np.float64))


def exact_loo_stacking_weights(
    targets: np.ndarray,
    cross_predictions: np.ndarray,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.float64)
    cross_predictions = np.asarray(cross_predictions, dtype=np.float64)
    if targets.ndim != 2:
        raise ValueError(f"Expected targets with shape (k, d), got {targets.shape}")
    if cross_predictions.ndim != 3:
        raise ValueError(
            f"Expected cross_predictions with shape (k, k, d), got {cross_predictions.shape}"
        )
    if cross_predictions.shape[0] != targets.shape[0] or cross_predictions.shape[1] != targets.shape[0]:
        raise ValueError(
            "cross_predictions must have shape (k, k, d) with the same k as targets"
        )
    if cross_predictions.shape[2] != targets.shape[1]:
        raise ValueError(
            "cross_predictions and targets disagree on feature dimension"
        )

    def objective(weights: np.ndarray) -> float:
        weights = np.asarray(weights, dtype=np.float64)
        total = 0.0
        for j in range(targets.shape[0]):
            denom = 1.0 - float(weights[j])
            if denom <= LOO_DENOM_EPS:
                return SIMPLEX_OBJECTIVE_PENALTY + (LOO_DENOM_EPS - denom) ** 2
            coeffs = np.asarray(weights, dtype=np.float64) / denom
            coeffs[j] = 0.0
            prediction = np.tensordot(coeffs, cross_predictions[j], axes=(0, 0))
            residual = prediction - targets[j]
            total += float(np.dot(residual, residual))
        return total

    return _simplex_minimize(objective, int(targets.shape[0]))


def compute_same_batch_stacking_weights(
    driver,
    holdout_batch: PreparedBatch,
    deltas: list[Any],
) -> tuple[np.ndarray, dict[str, float]]:
    target_vector, energy_variance = centered_target_vector(
        holdout_batch.local_loss,
        mode=driver.mode,
        n_replicas=holdout_batch.n_replicas,
    )
    prediction_matrix = np.column_stack(
        [centered_prediction_vector(driver, holdout_batch, delta) for delta in deltas]
    )
    weights = compute_stacking_mixture_weights(target_vector, prediction_matrix)
    return weights, {
        "weight_fit_samples": int(holdout_batch.sample_count),
        "weight_fit_target_norm_sq": float(np.dot(target_vector, target_vector)),
        "weight_fit_energy_variance": float(energy_variance),
    }


def _compatible_chunk_divisor(sample_count: int, preferred: int | None) -> int | None:
    sample_count = int(sample_count)
    if sample_count <= 0:
        raise ValueError(f"sample_count must be positive, got {sample_count}")
    if preferred is None:
        return None
    preferred = int(preferred)
    if preferred <= 0:
        return None
    if sample_count % preferred == 0:
        return preferred
    divisor = math.gcd(sample_count, preferred)
    return divisor if divisor > 1 else sample_count


def _compatible_chunk_size_bwd(driver, *, system_name: str, sample_count: int) -> int | None:
    sample_count = int(sample_count)
    chunk_size_bwd = getattr(driver, "chunk_size_bwd", None)
    if chunk_size_bwd is None:
        return None
    chunk_size_bwd = int(chunk_size_bwd)
    if sample_count % chunk_size_bwd == 0:
        return chunk_size_bwd
    fallback = default_delta_bank_chunk_size_bwd(system_name, n_samples=sample_count)
    compatible_fallback = _compatible_chunk_divisor(sample_count, fallback)
    if compatible_fallback is not None:
        return compatible_fallback
    return _compatible_chunk_divisor(sample_count, chunk_size_bwd)


def compute_exact_loo_weights(
    driver,
    batches: list[PreparedBatch],
    deltas: list[Any],
) -> tuple[np.ndarray, dict[str, float]]:
    targets = []
    cross_predictions = []
    for batch_j in batches:
        target_j, _ = centered_target_vector(
            batch_j.local_loss,
            mode=driver.mode,
            n_replicas=batch_j.n_replicas,
        )
        targets.append(target_j)
        row = []
        for delta_i in deltas:
            row.append(centered_prediction_vector(driver, batch_j, delta_i))
        cross_predictions.append(row)

    targets_array = np.asarray(targets, dtype=np.float64)
    cross_predictions_array = np.asarray(cross_predictions, dtype=np.float64)
    weights = exact_loo_stacking_weights(targets_array, cross_predictions_array)
    return weights, {
        "loo_objective": float(
            sum(
                np.dot(
                    np.tensordot(
                        np.where(
                            np.arange(len(weights)) == j,
                            0.0,
                            weights / max(1.0 - weights[j], LOO_DENOM_EPS),
                        ),
                        cross_predictions_array[j],
                        axes=(0, 0),
                    )
                    - targets_array[j],
                    np.tensordot(
                        np.where(
                            np.arange(len(weights)) == j,
                            0.0,
                            weights / max(1.0 - weights[j], LOO_DENOM_EPS),
                        ),
                        cross_predictions_array[j],
                        axes=(0, 0),
                    )
                    - targets_array[j],
                )
                for j in range(len(weights))
            )
        ),
    }


def prediction_channels_from_bank_values(values, *, mode: str | None, sample_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"Expected predictions with shape (m, d), got {values.shape}")
    if mode == "complex":
        return np.stack(
            [
                values[:, :sample_count],
                values[:, sample_count : 2 * sample_count],
            ],
            axis=1,
        )
    return np.real(values[:, :sample_count])[:, None, :]


def target_vector_from_energy_channels(
    energy_channels,
    *,
    mode: str | None,
    n_replicas: int | None,
) -> tuple[np.ndarray, float]:
    centered, total_count = center_samples(energy_channels, n_replicas=n_replicas)
    centered_flat = centered.reshape(centered.shape[0], -1)
    energy_variance = float(np.square(centered).sum() / float(total_count))
    if mode == "complex":
        vector = 2.0 * np.concatenate([centered_flat[0], centered_flat[1]], axis=0) / math.sqrt(float(total_count))
    else:
        vector = 2.0 * centered_flat[0] / math.sqrt(float(total_count))
    return vector, energy_variance


def prediction_vectors_from_bank(
    prediction_bank,
    *,
    mode: str | None,
    n_replicas: int | None,
) -> np.ndarray:
    centered, total_count = center_samples(prediction_bank, n_replicas=n_replicas)
    centered_flat = centered.reshape(centered.shape[0], centered.shape[1], -1)
    if mode == "complex":
        return np.concatenate([centered_flat[:, 0, :], centered_flat[:, 1, :]], axis=1) / math.sqrt(float(total_count))
    return centered_flat[:, 0, :] / math.sqrt(float(total_count))


def metrics_from_prediction_bank(
    prediction_bank,
    energy_channels,
    *,
    mode: str | None,
    n_replicas: int | None,
) -> dict[str, Any]:
    target_vector, energy_variance = target_vector_from_energy_channels(
        energy_channels,
        mode=mode,
        n_replicas=n_replicas,
    )
    prediction_vectors = prediction_vectors_from_bank(
        prediction_bank,
        mode=mode,
        n_replicas=n_replicas,
    )
    residual_vectors = prediction_vectors - target_vector[None, :]
    residual_raw_per_repeat = np.einsum("ij,ij->i", residual_vectors, residual_vectors)
    target_norm_sq = float(np.dot(target_vector, target_vector))
    residual_norm_per_repeat = residual_raw_per_repeat / target_norm_sq if target_norm_sq > 1e-12 else np.full_like(
        residual_raw_per_repeat,
        np.inf,
    )
    if prediction_vectors.shape[0] > 1:
        centered_predictions = prediction_vectors - prediction_vectors.mean(axis=0, keepdims=True)
        var_hat_raw = float(np.square(centered_predictions).sum() / float(prediction_vectors.shape[0] - 1))
    else:
        var_hat_raw = 0.0
    return {
        "r_val": float(np.mean(residual_raw_per_repeat)),
        "r_val_raw": float(np.mean(residual_raw_per_repeat)),
        "r_val_norm": float(np.mean(residual_norm_per_repeat)),
        "r_val_raw_per_repeat": [float(x) for x in residual_raw_per_repeat],
        "r_val_norm_per_repeat": [float(x) for x in residual_norm_per_repeat],
        "target_norm_sq": float(target_norm_sq),
        "energy_variance_eval": float(energy_variance),
        "var_hat_raw": float(var_hat_raw),
        "var_hat_norm": float(var_hat_raw / target_norm_sq) if target_norm_sq > 1e-12 else float("inf"),
        "snorm_diff_sq": float(2.0 * var_hat_raw),
        "repeat_count": int(prediction_vectors.shape[0]),
    }


def _slice_tree(tree, start: int, stop: int):
    return jax.tree_util.tree_map(lambda x: x[start:stop], tree)


def prediction_bank_on_validation(
    driver,
    validation_payload: dict[str, Any],
    delta_bank,
    *,
    sample_chunk_size: int,
    jvp_chunk: int | None,
    delta_batch_size: int,
) -> np.ndarray:
    variables = driver.state.variables
    model_state, params = fcore.pop(variables, "params")
    samples = np.asarray(validation_payload["samples"])
    mode = validation_payload.get("mode", driver.mode)
    n_val = int(samples.shape[0])
    m_deltas = int(np.asarray(jax.tree_util.tree_leaves(delta_bank)[0]).shape[0])
    n_channels = 2 if mode == "complex" else 1
    prediction_bank = np.empty((m_deltas, n_channels, n_val), dtype=np.float64)
    chunk_size = min(int(sample_chunk_size), int(n_val))

    for sample_start in range(0, n_val, chunk_size):
        sample_stop = min(sample_start + chunk_size, n_val)
        samples_chunk = jnp.asarray(samples[sample_start:sample_stop])
        for delta_start in range(0, m_deltas, int(delta_batch_size)):
            delta_stop = min(delta_start + int(delta_batch_size), m_deltas)
            delta_block = jax_tree(_slice_tree(delta_bank, delta_start, delta_stop))
            predictions = compute_O_delta_jvp_many(
                driver.state._apply_fun,
                params,
                model_state,
                samples_chunk,
                delta_block,
                mode=mode,
                chunk_size=jvp_chunk,
            )
            prediction_bank[
                delta_start:delta_stop,
                :,
                sample_start:sample_stop,
            ] = prediction_channels_from_bank_values(
                np.asarray(predictions),
                mode=mode,
                sample_count=int(sample_stop - sample_start),
            )
    return prediction_bank


def _live_validation_context(
    training: dict[str, Any],
    *,
    base_root,
    system_name: str,
    n_val: int,
    validation_round_samples: int | None,
    validation_seed_base: int | None,
    conn_chunk: int = 4096,
    eval_chunk: int = 4096,
) -> dict[str, Any]:
    system_name = normalize_system_name(system_name)
    step = int(training["step"])
    config = training["config"]
    validation_seed = _resolve_seed_base(
        config,
        step,
        validation_seed_base,
        VALIDATION_SEED_OFFSET,
    )

    if system_name == "tfim":
        heldout_h = load_or_create_heldout_h_values(base_root, config)
        n_replicas = int(len(heldout_h))
        if int(n_val) % n_replicas != 0:
            raise ValueError(
                f"n_val={n_val} must be divisible by validation_h_count={n_replicas}"
            )
        round_samples = (
            int(validation_round_samples)
            if validation_round_samples is not None
            else int(config["n_samples"])
        )
        if int(round_samples) % n_replicas != 0:
            raise ValueError(
                f"validation_round_samples={round_samples} must be divisible by validation_h_count={n_replicas}"
            )
        validation_vstate = make_tfim_state(
            config,
            parameter_array=heldout_h,
            n_samples=int(round_samples),
            seed=int(validation_seed),
            chunk_size=min(
                int(config.get("validation_chunk_size", config.get("chunk_size", round_samples))),
                int(round_samples),
            ),
        )
        validation_vstate.variables = training["vstate"].variables

        def compute_eloc_round(samples_round):
            return compute_driver_local_energies_round(
                training["driver"],
                validation_vstate,
                samples_round,
            )

        return {
            "driver": training["driver"],
            "validation_vstate": validation_vstate,
            "round_samples": int(round_samples),
            "n_replicas": int(n_replicas),
            "mode": training["driver"].mode,
            "compute_eloc_round": compute_eloc_round,
            "validation_seed": int(validation_seed),
            "n_val": int(n_val),
        }

    ckpt = training["checkpoint"]
    n_replicas = None
    n_train = resolve_n_samples_for_config(config, step=step)
    round_samples = (
        int(validation_round_samples)
        if validation_round_samples is not None
        else int(n_train)
    )
    _, _, validation_vstate = make_vit_state(
        config,
        ckpt["parameters"],
        variables=ckpt.get("variables"),
        step=step,
        n_samples_override=int(round_samples),
        chunk_size_override=min(int(config.get("chunk_size", 4096)), int(round_samples)),
        seed_override=int(validation_seed),
    )

    def compute_eloc_round(samples_round):
        return compute_vit_local_energies_round(
            validation_vstate._apply_fun,
            validation_vstate.variables,
            samples_round,
            training["driver"]._ham,
            conn_chunk=int(conn_chunk),
            eval_chunk=int(eval_chunk),
        )

    return {
        "driver": training["driver"],
        "validation_vstate": validation_vstate,
        "round_samples": int(round_samples),
        "n_replicas": n_replicas,
        "mode": training["driver"].mode,
        "compute_eloc_round": compute_eloc_round,
        "validation_seed": int(validation_seed),
        "n_val": int(n_val),
    }


def prediction_bank_on_live_validation(
    training: dict[str, Any],
    *,
    base_root,
    system_name: str,
    delta_bank,
    n_val: int,
    validation_round_samples: int | None,
    validation_sample_chunk: int,
    validation_seed_base: int | None,
    jvp_chunk: int | None,
    delta_batch_size: int,
    conn_chunk: int = 4096,
    eval_chunk: int = 4096,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    effective_round_samples = (
        int(validation_round_samples)
        if validation_round_samples is not None
        else int(validation_sample_chunk)
    )
    context = _live_validation_context(
        training,
        base_root=base_root,
        system_name=system_name,
        n_val=int(n_val),
        validation_round_samples=effective_round_samples,
        validation_seed_base=validation_seed_base,
        conn_chunk=conn_chunk,
        eval_chunk=eval_chunk,
    )
    prediction_bank, energy_channels = _stream_prediction_cache(
        driver=context["driver"],
        validation_vstate=context["validation_vstate"],
        delta_bank=delta_bank,
        mode=context["mode"],
        n_val=int(n_val),
        round_samples=int(context["round_samples"]),
        jvp_chunk=jvp_chunk,
        delta_batch_size=int(delta_batch_size),
        n_replicas=context["n_replicas"],
        compute_eloc_round=context["compute_eloc_round"],
    )
    return prediction_bank, energy_channels, context


def spectrum_quantile_lambda_grid(
    spectrum,
    *,
    quantiles: tuple[float, ...] = FIG3_LAMBDA_QUANTILES,
) -> tuple[float, ...]:
    eigenvalues = np.asarray(spectrum, dtype=np.float64).reshape(-1)
    clipped = np.clip(eigenvalues, 0.0, None)
    positive = clipped[clipped > 0.0]
    if positive.size == 0:
        raise ValueError("The cached NTK spectrum has no strictly positive eigenvalues")
    lambda_max = float(np.max(positive))
    min_positive = float(np.min(positive))
    floor = max(min_positive, 1e-12 * lambda_max)
    selected = []
    for quantile in quantiles:
        value = float(np.quantile(positive, float(quantile)))
        if not np.isfinite(value) or value <= 0.0:
            value = floor
        selected.append(float(value))
    return tuple(selected)


def _parameter_count_for_training(training: dict[str, Any]) -> int:
    if training["system"] == "tfim":
        return int(count_tfim_parameters(training["vstate"].parameters))
    return int(count_parameters(training["vstate"].parameters))


def evaluate_fixed_data_fig2_checkpoint(
    ckpt_path,
    *,
    base_root,
    system_name: str,
    diagnostic_lambdas: tuple[float, ...],
    repeat_count: int = FIXED_DATA_REPEAT_COUNT,
    batch_slots: int = FIXED_DATA_BATCH_SLOTS,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    validation_sample_chunk: int | None = None,
    jvp_chunk: int | None = None,
    conn_chunk: int = 4096,
    eval_chunk: int = 4096,
    delta_batch_size: int = 10,
    train_seed_base: int | None = None,
    validation_seed_base: int | None = None,
    spectrum_seed_base: int | None = None,
) -> list[dict[str, Any]]:
    system_name = normalize_system_name(system_name)
    step = int(checkpoint_step_from_path(ckpt_path))
    if n_val is None:
        n_val = int(heldout_samples_for_system(system_name))
    if validation_sample_chunk is None:
        if validation_round_samples is not None:
            validation_sample_chunk = int(validation_round_samples)
        elif system_name == "tfim":
            validation_sample_chunk = 12000
        else:
            validation_sample_chunk = 4000

    training = _load_training_driver(ckpt_path, system_name=system_name)
    driver = training["driver"]
    resolved_train_seed_base = _resolve_seed_base(
        training["config"],
        step,
        train_seed_base,
        TRAIN_SEED_OFFSET,
    )
    snapshot = snapshot_driver_state(driver)
    deltas_by_lambda = {float(lam): [] for lam in diagnostic_lambdas}
    sample_count_per_slot = None
    try:
        for repeat_index in range(int(repeat_count)):
            shared_batch = _sample_live_prepared_batch(
                driver,
                seed=_live_batch_seed(
                    resolved_train_seed_base,
                    repeat_index=repeat_index,
                    slot_index=0,
                    batch_slots=batch_slots,
                ),
                chain_name=f"fixed_data_live_r{repeat_index:03d}_s00",
            )
            if sample_count_per_slot is None:
                sample_count_per_slot = int(shared_batch.sample_count)
            for lam in diagnostic_lambdas:
                record = base_solve_record(
                    driver,
                    shared_batch,
                    diag_shift=float(lam),
                    collect_residual_info=False,
                )
                deltas_by_lambda[float(lam)].append(jax.tree_util.tree_map(np.asarray, record.delta))
    finally:
        restore_driver_state(driver, snapshot)
    if sample_count_per_slot is None:
        raise RuntimeError(f"No training batches were generated for {ckpt_path}")

    parameter_count = _parameter_count_for_training(training)
    results = []
    for lam in diagnostic_lambdas:
        delta_bank = stack_tree([jax_tree(delta) for delta in deltas_by_lambda[float(lam)]])
        prediction_bank, energy_channels, validation_context = prediction_bank_on_live_validation(
            training,
            base_root=base_root,
            system_name=system_name,
            delta_bank=delta_bank,
            n_val=int(n_val),
            validation_round_samples=validation_round_samples,
            validation_sample_chunk=int(validation_sample_chunk),
            validation_seed_base=validation_seed_base,
            jvp_chunk=jvp_chunk,
            delta_batch_size=delta_batch_size,
            conn_chunk=conn_chunk,
            eval_chunk=eval_chunk,
        )
        metrics = metrics_from_prediction_bank(
            prediction_bank,
            energy_channels,
            mode=validation_context["mode"],
            n_replicas=validation_context["n_replicas"],
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "workspace": WORKSPACE_NAME,
            "system": system_name,
            "step": int(step),
            "lambda": float(lam),
            "diagnostic_lambda": float(lam),
            "source_step": int(step),
            "repeat_count": int(repeat_count),
            "m_deltas": int(repeat_count),
            "batch_slots": int(batch_slots),
            "shared_slot": 0,
            "n_val": int(n_val),
            "P": int(parameter_count),
            "P_over_Ns": float(parameter_count) / float(sample_count_per_slot),
            "source_checkpoint": str(Path(ckpt_path).resolve()),
            "source_training_lambda": float(training["config"]["diag_shift"]),
            "sample_materialization": "live_sampler",
            "train_seed_base": int(resolved_train_seed_base),
            "validation_seed": int(validation_context["validation_seed"]),
            "validation_round_samples": int(validation_context["round_samples"]),
            "validation_path": None,
            "train_batches_path": None,
            "spectrum_ref_path": str(spectrum_ref_path(base_root, system_name, step)),
        }
        payload.update(metrics)
        result_path = fig2_checkpoint_lambda_path(base_root, system_name, step, float(lam))
        atomic_save(payload, result_path)
        results.append(payload)
    return results


def select_best_fixed_lambda(fig2_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not fig2_payloads:
        raise ValueError("Expected at least one Figure 2 payload to choose lambda*")
    rows = sorted(
        (
            float(payload["r_val_raw"]),
            float(payload["lambda"]),
            payload,
        )
        for payload in fig2_payloads
    )
    _, _, winner = rows[0]
    return {
        "lambda_star": float(winner["lambda"]),
        "r_val_raw": float(winner["r_val_raw"]),
        "r_val_norm": float(winner["r_val_norm"]),
    }


def update_best_fixed_lambda_summary(
    base_root,
    *,
    system_name: str,
    checkpoint_summaries: dict[int, dict[str, Any]],
    diagnostic_lambdas: tuple[float, ...],
) -> dict[str, Any]:
    path = best_fixed_lambda_path(base_root)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "workspace": WORKSPACE_NAME,
            "diagnostic_lambdas": [float(lam) for lam in diagnostic_lambdas],
            "systems": {},
        }
    systems = dict(payload.get("systems", {}))
    system_block = dict(systems.get(system_name, {}))
    for step, summary in checkpoint_summaries.items():
        system_block[str(int(step))] = {
            "lambda_star": float(summary["lambda_star"]),
            "r_val_raw": float(summary["r_val_raw"]),
            "r_val_norm": float(summary["r_val_norm"]),
        }
    systems[system_name] = system_block
    payload["systems"] = systems
    payload["diagnostic_lambdas"] = [float(lam) for lam in diagnostic_lambdas]
    _atomic_json_dump(payload, path)
    return payload


def load_best_fixed_lambda_summary(base_root) -> dict[str, Any]:
    path = best_fixed_lambda_path(base_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing Figure 2 best-fixed-lambda summary: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def best_fixed_lambda_for_checkpoint(base_root, system_name: str, step: int) -> float:
    payload = load_best_fixed_lambda_summary(base_root)
    try:
        return float(payload["systems"][normalize_system_name(system_name)][str(int(step))]["lambda_star"])
    except KeyError as exc:
        raise KeyError(
            f"Missing lambda* for system={system_name} step={step} in {best_fixed_lambda_path(base_root)}"
        ) from exc


def _method_metrics_from_final_deltas(
    driver,
    validation_payload: dict[str, Any],
    final_deltas: list[Any],
    *,
    validation_sample_chunk: int,
    jvp_chunk: int | None,
    delta_batch_size: int,
) -> dict[str, Any]:
    delta_bank = stack_tree([jax_tree(delta) for delta in final_deltas])
    prediction_bank = prediction_bank_on_validation(
        driver,
        validation_payload,
        delta_bank,
        sample_chunk_size=int(validation_sample_chunk),
        jvp_chunk=jvp_chunk,
        delta_batch_size=delta_batch_size,
    )
    return metrics_from_prediction_bank(
        prediction_bank,
        validation_payload["energy_channels"],
        mode=validation_payload.get("mode", driver.mode),
        n_replicas=validation_payload.get("n_replicas"),
    )


def _method_metrics_from_final_deltas_live(
    training: dict[str, Any],
    *,
    base_root,
    system_name: str,
    final_deltas: list[Any],
    n_val: int,
    validation_round_samples: int | None,
    validation_sample_chunk: int,
    validation_seed_base: int | None,
    jvp_chunk: int | None,
    delta_batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    delta_bank = stack_tree([jax_tree(delta) for delta in final_deltas])
    prediction_bank, energy_channels, validation_context = prediction_bank_on_live_validation(
        training,
        base_root=base_root,
        system_name=system_name,
        delta_bank=delta_bank,
        n_val=int(n_val),
        validation_round_samples=validation_round_samples,
        validation_sample_chunk=int(validation_sample_chunk),
        validation_seed_base=validation_seed_base,
        jvp_chunk=jvp_chunk,
        delta_batch_size=delta_batch_size,
    )
    metrics = metrics_from_prediction_bank(
        prediction_bank,
        energy_channels,
        mode=validation_context["mode"],
        n_replicas=validation_context["n_replicas"],
    )
    return metrics, validation_context


def evaluate_fixed_data_fig3_checkpoint(
    ckpt_path,
    *,
    base_root,
    system_name: str,
    k: int = 4,
    repeat_count: int = FIXED_DATA_REPEAT_COUNT,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    validation_sample_chunk: int | None = None,
    jvp_chunk: int | None = None,
    delta_batch_size: int = 10,
    train_seed_base: int | None = None,
    validation_seed_base: int | None = None,
    spectrum_seed_base: int | None = None,
) -> dict[str, Any]:
    system_name = normalize_system_name(system_name)
    if int(k) != 4:
        raise ValueError(f"Only k=4 is supported by the 20260422 fixed-data workflow, got k={k}")
    step = int(checkpoint_step_from_path(ckpt_path))
    if n_val is None:
        n_val = int(heldout_samples_for_system(system_name))
    ensure_reference_spectrum(
        ckpt_path,
        base_root=base_root,
        system_name=system_name,
        spectrum_seed_base=spectrum_seed_base,
    )
    spectrum = load_cached_spectrum(base_root, system_name, step)
    lambda_star = best_fixed_lambda_for_checkpoint(base_root, system_name, step)
    lambda_grid = spectrum_quantile_lambda_grid(spectrum)

    if validation_sample_chunk is None:
        if system_name == "tfim":
            validation_sample_chunk = 12000
        else:
            validation_sample_chunk = 4000
    if validation_round_samples is None:
        validation_round_samples = int(validation_sample_chunk)

    training = _load_training_driver(ckpt_path, system_name=system_name)
    driver = training["driver"]
    resolved_train_seed_base = _resolve_seed_base(
        training["config"],
        step,
        train_seed_base,
        TRAIN_SEED_OFFSET,
    )
    snapshot = snapshot_driver_state(driver)
    original_chunk_size_bwd = getattr(driver, "chunk_size_bwd", None)

    method_deltas: dict[str, list[Any]] = {method_id: [] for method_id in FIG3_METHOD_IDS}
    method_weights: dict[str, list[list[float]]] = {method_id: [] for method_id in FIG3_METHOD_IDS}
    method_meta: dict[str, dict[str, Any]] = {
        "same_batch_multilambda_uniform_k4": {},
        "same_batch_multilambda_stacking_k4": {},
        "indep_multilambda_stacking_k4": {},
    }

    try:
        for repeat_index in range(int(repeat_count)):
            full_batches = [
                _sample_live_prepared_batch(
                    driver,
                    seed=_live_batch_seed(
                        resolved_train_seed_base,
                        repeat_index=repeat_index,
                        slot_index=slot_index,
                        batch_slots=int(k),
                    ),
                    chain_name=f"fixed_data_live_r{repeat_index:03d}_s{slot_index:02d}",
                )
                for slot_index in range(int(k))
            ]
            shared_full_batch = full_batches[0]

            single_record = base_solve_record(
                driver,
                shared_full_batch,
                diag_shift=float(lambda_star),
                collect_residual_info=False,
            )
            method_deltas["single_best_fixed_lambda"].append(jax.tree_util.tree_map(np.asarray, single_record.delta))
            method_weights["single_best_fixed_lambda"].append([1.0])

            driver.chunk_size_bwd = _compatible_chunk_size_bwd(
                driver,
                system_name=system_name,
                sample_count=int(shared_full_batch.sample_count),
            )
            same_batch_records = [
                base_solve_record(
                    driver,
                    shared_full_batch,
                    diag_shift=float(lam),
                    collect_residual_info=False,
                )
                for lam in lambda_grid
            ]
            same_batch_deltas = [record.delta for record in same_batch_records]

            same_uniform_weights = uniform_mixture_weights(len(same_batch_records))
            same_uniform_delta = weighted_tree_sum(same_batch_deltas, same_uniform_weights)
            method_deltas["same_batch_multilambda_uniform_k4"].append(
                jax.tree_util.tree_map(np.asarray, same_uniform_delta)
            )
            method_weights["same_batch_multilambda_uniform_k4"].append(
                [float(value) for value in same_uniform_weights]
            )

            same_batch_weight_batch = full_batches[1]
            driver.chunk_size_bwd = _compatible_chunk_size_bwd(
                driver,
                system_name=system_name,
                sample_count=int(same_batch_weight_batch.sample_count),
            )
            same_stack_weights, same_stack_meta = compute_same_batch_stacking_weights(
                driver,
                same_batch_weight_batch,
                same_batch_deltas,
            )
            same_stack_delta = weighted_tree_sum(same_batch_deltas, same_stack_weights)
            method_deltas["same_batch_multilambda_stacking_k4"].append(
                jax.tree_util.tree_map(np.asarray, same_stack_delta)
            )
            method_weights["same_batch_multilambda_stacking_k4"].append(
                [float(value) for value in same_stack_weights]
            )
            method_meta["same_batch_multilambda_uniform_k4"] = {
                "solve_sample_count": int(shared_full_batch.sample_count),
                "solve_n_replicas": (
                    None if shared_full_batch.n_replicas is None else int(shared_full_batch.n_replicas)
                ),
            }
            method_meta["same_batch_multilambda_stacking_k4"] = {
                **method_meta["same_batch_multilambda_uniform_k4"],
                **same_stack_meta,
                "weight_fit_source": "independent_batch",
                "weight_fit_batch_slot": 1,
                "weight_fit_n_replicas": (
                    None
                    if same_batch_weight_batch.n_replicas is None
                    else int(same_batch_weight_batch.n_replicas)
                ),
            }

            driver.chunk_size_bwd = original_chunk_size_bwd
            indep_single_records = [
                base_solve_record(
                    driver,
                    batch,
                    diag_shift=float(lambda_star),
                    collect_residual_info=False,
                )
                for batch in full_batches
            ]
            indep_single_weights = uniform_mixture_weights(len(indep_single_records))
            indep_single_delta = weighted_tree_sum(
                [record.delta for record in indep_single_records],
                indep_single_weights,
            )
            method_deltas["indep_single_lambda_uniform_k4"].append(
                jax.tree_util.tree_map(np.asarray, indep_single_delta)
            )
            method_weights["indep_single_lambda_uniform_k4"].append(
                [float(value) for value in indep_single_weights]
            )

            indep_multi_records = [
                base_solve_record(
                    driver,
                    batch,
                    diag_shift=float(lam),
                    collect_residual_info=False,
                )
                for batch, lam in zip(full_batches, lambda_grid, strict=True)
            ]
            indep_multi_deltas = [record.delta for record in indep_multi_records]
            indep_uniform_weights = uniform_mixture_weights(len(indep_multi_records))
            indep_uniform_delta = weighted_tree_sum(indep_multi_deltas, indep_uniform_weights)
            method_deltas["indep_multilambda_uniform_k4"].append(
                jax.tree_util.tree_map(np.asarray, indep_uniform_delta)
            )
            method_weights["indep_multilambda_uniform_k4"].append(
                [float(value) for value in indep_uniform_weights]
            )

            loo_weights, loo_meta = compute_exact_loo_weights(
                driver,
                full_batches,
                indep_multi_deltas,
            )
            loo_delta = weighted_tree_sum(indep_multi_deltas, loo_weights)
            method_deltas["indep_multilambda_stacking_k4"].append(
                jax.tree_util.tree_map(np.asarray, loo_delta)
            )
            method_weights["indep_multilambda_stacking_k4"].append(
                [float(value) for value in loo_weights]
            )
            method_meta["indep_multilambda_stacking_k4"] = loo_meta
    finally:
        driver.chunk_size_bwd = original_chunk_size_bwd
        restore_driver_state(driver, snapshot)

    methods = {}
    validation_context = None
    for method_id in FIG3_METHOD_IDS:
        metrics, validation_context = _method_metrics_from_final_deltas_live(
            training,
            base_root=base_root,
            system_name=system_name,
            final_deltas=method_deltas[method_id],
            n_val=int(n_val),
            validation_round_samples=validation_round_samples,
            validation_sample_chunk=int(validation_sample_chunk),
            validation_seed_base=validation_seed_base,
            jvp_chunk=jvp_chunk,
            delta_batch_size=delta_batch_size,
        )
        weights = method_weights[method_id]
        methods[method_id] = {
            "method_id": method_id,
            "weights_per_repeat": weights,
            "mean_weights": [
                float(value)
                for value in np.mean(np.asarray(weights, dtype=np.float64), axis=0)
            ],
            **metrics,
        }
        if method_id in method_meta and method_meta[method_id]:
            methods[method_id]["method_meta"] = method_meta[method_id]
    if validation_context is None:
        raise RuntimeError(f"No Figure 3 methods were evaluated for {ckpt_path}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "workspace": WORKSPACE_NAME,
        "system": system_name,
        "step": int(step),
        "source_step": int(step),
        "k": int(k),
        "repeat_count": int(repeat_count),
        "holdout_fraction": 0.0,
        "same_batch_weight_fit": "independent_batch",
        "lambda_star": float(lambda_star),
        "lambda_grid": [float(value) for value in lambda_grid],
        "lambda_quantiles": [float(value) for value in FIG3_LAMBDA_QUANTILES],
        "source_checkpoint": str(Path(ckpt_path).resolve()),
        "source_training_lambda": float(training["config"]["diag_shift"]),
        "n_val": int(n_val),
        "sample_materialization": "live_sampler",
        "train_seed_base": int(resolved_train_seed_base),
        "validation_seed": int(validation_context["validation_seed"]),
        "validation_round_samples": int(validation_context["round_samples"]),
        "validation_path": None,
        "train_batches_path": None,
        "spectrum_ref_path": str(spectrum_ref_path(base_root, system_name, step)),
        "methods": methods,
    }
    result_path = fig3_checkpoint_eval_path(base_root, system_name, step, k=int(k))
    atomic_save(payload, result_path)
    return payload


def run_fixed_data_fig2(
    base_root,
    *,
    system_name: str,
    source_training_lambda: float = SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA,
    source_steps: str | None = "all",
    latest_only: bool = False,
    repeat_count: int = FIXED_DATA_REPEAT_COUNT,
    batch_slots: int = FIXED_DATA_BATCH_SLOTS,
    diagnostic_lambdas: tuple[float, ...] | None = None,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    validation_sample_chunk: int | None = None,
    jvp_chunk: int | None = None,
    conn_chunk: int = 4096,
    eval_chunk: int = 4096,
    delta_batch_size: int = 10,
    train_seed_base: int | None = None,
    validation_seed_base: int | None = None,
    spectrum_seed_base: int | None = None,
) -> list[dict[str, Any]]:
    system_name = normalize_system_name(system_name)
    selected_lambdas = diagnostic_lambdas or default_diagnostic_lambdas()
    checkpoints = canonical_fixed_data_checkpoints(
        base_root,
        system_name,
        source_training_lambda=float(source_training_lambda),
        source_steps=source_steps,
        latest_only=latest_only,
    )
    if not checkpoints:
        raise SystemExit(
            f"No source checkpoints found for {system_name} at lambda_train={source_training_lambda:.3e}"
        )

    checkpoint_summaries = {}
    all_payloads = []
    for step, ckpt_path in checkpoints:
        payloads = evaluate_fixed_data_fig2_checkpoint(
            ckpt_path,
            base_root=base_root,
            system_name=system_name,
            diagnostic_lambdas=tuple(float(lam) for lam in selected_lambdas),
            repeat_count=repeat_count,
            batch_slots=batch_slots,
            n_val=n_val,
            validation_round_samples=validation_round_samples,
            validation_sample_chunk=validation_sample_chunk,
            jvp_chunk=jvp_chunk,
            conn_chunk=conn_chunk,
            eval_chunk=eval_chunk,
            delta_batch_size=delta_batch_size,
            train_seed_base=train_seed_base,
            validation_seed_base=validation_seed_base,
            spectrum_seed_base=spectrum_seed_base,
        )
        checkpoint_summaries[int(step)] = select_best_fixed_lambda(payloads)
        all_payloads.extend(payloads)

    update_best_fixed_lambda_summary(
        base_root,
        system_name=system_name,
        checkpoint_summaries=checkpoint_summaries,
        diagnostic_lambdas=tuple(float(lam) for lam in selected_lambdas),
    )
    return all_payloads


def run_fixed_data_fig3(
    base_root,
    *,
    system_name: str,
    source_training_lambda: float = SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA,
    source_steps: str | None = "all",
    latest_only: bool = False,
    repeat_count: int = FIXED_DATA_REPEAT_COUNT,
    k: int = 4,
    n_val: int | None = None,
    validation_round_samples: int | None = None,
    validation_sample_chunk: int | None = None,
    jvp_chunk: int | None = None,
    delta_batch_size: int = 10,
    train_seed_base: int | None = None,
    validation_seed_base: int | None = None,
    spectrum_seed_base: int | None = None,
) -> list[dict[str, Any]]:
    system_name = normalize_system_name(system_name)
    checkpoints = canonical_fixed_data_checkpoints(
        base_root,
        system_name,
        source_training_lambda=float(source_training_lambda),
        source_steps=source_steps,
        latest_only=latest_only,
    )
    if not checkpoints:
        raise SystemExit(
            f"No source checkpoints found for {system_name} at lambda_train={source_training_lambda:.3e}"
        )

    load_best_fixed_lambda_summary(base_root)
    results = []
    for step, ckpt_path in checkpoints:
        results.append(
            evaluate_fixed_data_fig3_checkpoint(
                ckpt_path,
                base_root=base_root,
                system_name=system_name,
                k=k,
                repeat_count=repeat_count,
                n_val=n_val,
                validation_round_samples=validation_round_samples,
                validation_sample_chunk=validation_sample_chunk,
                jvp_chunk=jvp_chunk,
                delta_batch_size=delta_batch_size,
                train_seed_base=train_seed_base,
                validation_seed_base=validation_seed_base,
                spectrum_seed_base=spectrum_seed_base,
            )
        )
    return results
