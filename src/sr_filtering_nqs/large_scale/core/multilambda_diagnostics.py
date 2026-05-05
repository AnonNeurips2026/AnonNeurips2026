from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from sr_filtering_nqs.large_scale.core.common import (
    canonical_mixture_results_root_candidates,
    lambda_grid_tag,
    model_type_for_system,
    normalize_system_name,
    parse_sr_lambda_grid,
)
from sr_filtering_nqs.large_scale.core.diagnostic_core import (
    _stream_prediction_cache,
    atomic_save,
    compute_driver_local_energies_round,
    compute_vit_local_energies_round,
    diagnostics_output_dir,
    jax_tree,
    resolve_round_samples,
    stack_tree,
)
from sr_filtering_nqs.large_scale.core.multidelta_metrics import center_samples
from sr_filtering_nqs.large_scale.core.multilambda_sr import (
    base_solve_record,
    compute_gcv_mixture_weights,
    compute_stacking_mixture_weights,
    numpy_tree,
    restore_driver_state,
    sample_prepared_batch,
    snapshot_driver_state,
    tree_l2_norm,
    uniform_mixture_weights,
    weighted_tree_sum,
)
from sr_filtering_nqs.large_scale.core.run_diagnostics import (
    load_checkpoint as load_vit_checkpoint,
    make_vit_state,
    reconstruct_driver as reconstruct_vit_driver,
    resolve_n_samples_for_config,
)
from sr_filtering_nqs.large_scale.core.tfim_fnqs import (
    load_or_create_heldout_h_values,
    make_state_from_config as make_tfim_state,
    reconstruct_training_driver,
)


MIXTURE_BANK_NAME = "mixture_bank_step{step:06d}.pkl"
MIXTURE_EVAL_NAME = "mixture_eval_step{step:06d}.pkl"
MIXTURE_METHOD_IDS = (
    "same_batch_uniform",
    "same_batch_gcv",
    "indep_uniform",
    "indep_batch_gcv",
    "indep_stacking",
)


def parse_mixture_method_id(method_id: str) -> dict[str, Any]:
    legacy_specs = {
        "same_batch_uniform": {
            "branch": "same_batch",
            "batch_mode": "same_batch",
            "weight_scheme": "uniform",
            "family": "same_batch_multilambda",
            "k": None,
        },
        "same_batch_gcv": {
            "branch": "same_batch",
            "batch_mode": "same_batch",
            "weight_scheme": "gcv",
            "family": "same_batch_multilambda",
            "k": None,
        },
        "indep_uniform": {
            "branch": "indep_batch",
            "batch_mode": "indep_batch",
            "weight_scheme": "uniform",
            "family": "indep_multilambda",
            "k": None,
        },
        "indep_batch_gcv": {
            "branch": "indep_batch",
            "batch_mode": "indep_batch",
            "weight_scheme": "gcv",
            "family": "indep_multilambda",
            "k": None,
        },
        "indep_stacking": {
            "branch": "indep_batch",
            "batch_mode": "indep_batch",
            "weight_scheme": "stacking",
            "family": "indep_multilambda",
            "k": None,
        },
        "single_oracle": {
            "branch": "single_oracle",
            "batch_mode": "single_oracle",
            "weight_scheme": "oracle",
            "family": "single_lambda",
            "k": 1,
        },
    }
    if method_id in legacy_specs:
        spec = dict(legacy_specs[method_id])
        spec["method_id"] = method_id
        return spec

    k_prefixes = (
        ("indep_single_lambda_k", "indep_single_lambda", "indep_single_lambda", "uniform", "single_lambda"),
        (
            "same_batch_multilambda_uniform_k",
            "same_batch",
            "same_batch",
            "uniform",
            "same_batch_multilambda",
        ),
        (
            "same_batch_multilambda_gcv_k",
            "same_batch",
            "same_batch",
            "gcv",
            "same_batch_multilambda",
        ),
        (
            "indep_multilambda_uniform_k",
            "indep_batch",
            "indep_batch",
            "uniform",
            "indep_multilambda",
        ),
        (
            "indep_multilambda_gcv_k",
            "indep_batch",
            "indep_batch",
            "gcv",
            "indep_multilambda",
        ),
    )
    for prefix, branch, batch_mode, weight_scheme, family in k_prefixes:
        if not method_id.startswith(prefix):
            continue
        k_text = method_id[len(prefix) :]
        if not k_text.isdigit():
            raise ValueError(f"Malformed k-suffixed mixture method id: {method_id}")
        return {
            "method_id": method_id,
            "branch": branch,
            "batch_mode": batch_mode,
            "weight_scheme": weight_scheme,
            "family": family,
            "k": int(k_text),
        }
    raise ValueError(f"Unknown mixture method id: {method_id}")


def validate_mixture_method_ids(method_ids, *, lambda_grid, single_lambda_count: int | None = None) -> tuple[str, ...]:
    methods = tuple(str(method_id) for method_id in method_ids)
    expected_k = int(single_lambda_count) if single_lambda_count is not None else int(len(tuple(lambda_grid)))
    for method_id in methods:
        spec = parse_mixture_method_id(method_id)
        if spec["k"] is not None and int(spec["k"]) not in (1, expected_k):
            raise ValueError(
                f"Method {method_id!r} expects k={int(spec['k'])}, but the active grid/single-lambda count uses k={expected_k}"
            )
    return methods


def mixture_method_requires_weight_fit(method_id: str) -> bool:
    return parse_mixture_method_id(method_id)["weight_scheme"] == "stacking"


def mixture_method_requires_oracle(method_id: str) -> bool:
    return parse_mixture_method_id(method_id)["branch"] in ("single_oracle", "indep_single_lambda")


def required_mixture_branches(method_ids) -> tuple[str, ...]:
    branches = set()
    for method_id in method_ids:
        spec = parse_mixture_method_id(method_id)
        branches.add(spec["branch"])
        # Independent-batch GCV reuses same-batch pilot solves to fit weights.
        if spec["branch"] == "indep_batch" and spec["weight_scheme"] == "gcv":
            branches.add("same_batch")
    return tuple(sorted(branches))


def mixture_bank_output_path(ckpt_path, step: int) -> Path:
    return diagnostics_output_dir(ckpt_path) / MIXTURE_BANK_NAME.format(step=step)


def mixture_eval_output_path(ckpt_path, step: int) -> Path:
    return diagnostics_output_dir(ckpt_path) / MIXTURE_EVAL_NAME.format(step=step)


def _selected_checkpoint_rows(
    ckpt_dir: Path,
    *,
    latest_only: bool,
    min_step: int | None,
) -> list[tuple[int, Path]]:
    checkpoints = sorted(Path(ckpt_dir).glob("checkpoint_step*.pkl"))
    if not checkpoints:
        return []

    if latest_only:
        latest = checkpoints[-1]
        step = int(latest.stem.replace("checkpoint_step", ""))
        if min_step is not None and step < min_step:
            return []
        return [(step, latest)]

    rows = []
    for checkpoint in checkpoints:
        try:
            step = int(checkpoint.stem.replace("checkpoint_step", ""))
        except ValueError:
            continue
        if min_step is not None and step < min_step:
            continue
        rows.append((step, checkpoint))
    return rows


def discover_mixture_checkpoints(
    system_name: str,
    input_dir,
    *,
    sr_variant: str | None = None,
    sr_weight_scheme: str | None = None,
    lambda_grid=None,
    min_step: int | None = None,
    latest_only: bool = False,
) -> list[tuple[str, str, str, int, Path]]:
    system_name = normalize_system_name(system_name)
    grid_filter = None if lambda_grid is None else lambda_grid_tag(lambda_grid)
    rows: list[tuple[str, str, str, int, Path]] = []
    seen = set()

    for root in canonical_mixture_results_root_candidates(input_dir, system_name):
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            try:
                variant, weight_scheme, grid_tag = run_dir.name.split("__", 2)
            except ValueError:
                continue
            if sr_variant is not None and variant != sr_variant:
                continue
            if sr_weight_scheme is not None and weight_scheme != sr_weight_scheme:
                continue
            if grid_filter is not None and grid_tag != grid_filter:
                continue

            ckpt_dir = run_dir / "results" / "checkpoints"
            for step, checkpoint in _selected_checkpoint_rows(
                ckpt_dir,
                latest_only=latest_only,
                min_step=min_step,
            ):
                resolved = str(checkpoint.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                rows.append((variant, weight_scheme, grid_tag, step, checkpoint))

    return sorted(rows, key=lambda row: (row[0], row[1], row[2], row[3], str(row[4])))


def _mixture_bank_compatible(
    payload: dict[str, Any],
    *,
    repeat_count: int,
    lambda_grid: tuple[float, ...],
    oracle_lambda: float | None = None,
    single_lambda_count: int | None = None,
    required_branches: tuple[str, ...] = (),
) -> bool:
    if int(payload.get("repeat_count", -1)) != int(repeat_count):
        return False
    stored = tuple(float(x) for x in payload.get("lambda_grid", ()))
    if stored != tuple(float(x) for x in lambda_grid):
        return False
    if oracle_lambda is not None:
        if not math.isclose(
            float(payload.get("oracle_lambda", float("nan"))),
            float(oracle_lambda),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            return False
    if single_lambda_count is not None:
        branch = payload.get("indep_single_lambda")
        if branch is not None and int(branch.get("group_size", -1)) != int(single_lambda_count):
            return False
    return all(branch in payload for branch in required_branches)


def _load_training_driver(ckpt_path, *, system_name: str):
    system_name = normalize_system_name(system_name)
    if model_type_for_system(system_name) == "fnqs":
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        driver, _ = reconstruct_training_driver(ckpt)
        return {
            "system": system_name,
            "config": ckpt["config"],
            "step": int(ckpt["step"]),
            "driver": driver,
        }

    ckpt = load_vit_checkpoint(ckpt_path)
    driver, _ = reconstruct_vit_driver(
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
    }


def _load_vit_validation_context(
    ckpt_path,
    *,
    n_val: int,
    validation_round_samples: int | None,
    validation_seed_offset: int,
):
    ckpt = load_vit_checkpoint(ckpt_path)
    step = int(ckpt["step"])
    config = ckpt["config"]
    driver, vstate = reconstruct_vit_driver(
        config,
        ckpt["parameters"],
        variables=ckpt.get("variables"),
        step=step,
        chunk_size_bwd_override=config.get("chunk_size_bwd"),
    )
    n_train = resolve_n_samples_for_config(config, step=step)
    round_samples = resolve_round_samples(
        system_name="j1j2",
        total_n_val=int(n_val),
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

    def compute_eloc_round(samples_round):
        return compute_vit_local_energies_round(
            validation_vstate._apply_fun,
            validation_vstate.variables,
            samples_round,
            driver._ham,
            conn_chunk=4096,
            eval_chunk=4096,
        )

    return {
        "system": "j1j2",
        "config": config,
        "step": step,
        "driver": driver,
        "validation_vstate": validation_vstate,
        "round_samples": int(round_samples),
        "n_replicas": None,
        "compute_eloc_round": compute_eloc_round,
        "mode": driver.mode,
    }


def _load_tfim_validation_context(
    ckpt_path,
    *,
    n_val: int,
    validation_round_samples: int | None,
    validation_seed_offset: int,
):
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)
    step = int(ckpt["step"])
    config = ckpt["config"]
    driver, vstate = reconstruct_training_driver(ckpt)
    heldout_h = load_or_create_heldout_h_values(Path(ckpt_path).parents[2], config)
    n_replicas = int(len(heldout_h))
    round_samples = resolve_round_samples(
        system_name="tfim",
        total_n_val=int(n_val),
        requested_round_samples=validation_round_samples,
        default_round_samples=int(config["n_samples"]),
        n_replicas=n_replicas,
    )
    validation_vstate = make_tfim_state(
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

    def compute_eloc_round(samples_round):
        return compute_driver_local_energies_round(
            driver,
            validation_vstate,
            samples_round,
        )

    return {
        "system": "tfim",
        "config": config,
        "step": step,
        "driver": driver,
        "validation_vstate": validation_vstate,
        "round_samples": int(round_samples),
        "n_replicas": n_replicas,
        "compute_eloc_round": compute_eloc_round,
        "mode": driver.mode,
    }


def _load_validation_context(
    ckpt_path,
    *,
    system_name: str,
    n_val: int,
    validation_round_samples: int | None,
    validation_seed_offset: int,
):
    system_name = normalize_system_name(system_name)
    if system_name == "tfim":
        return _load_tfim_validation_context(
            ckpt_path,
            n_val=n_val,
            validation_round_samples=validation_round_samples,
            validation_seed_offset=validation_seed_offset,
        )
    return _load_vit_validation_context(
        ckpt_path,
        n_val=n_val,
        validation_round_samples=validation_round_samples,
        validation_seed_offset=validation_seed_offset,
    )


def _vector_from_energy_channels(
    energy_channels,
    *,
    mode: str,
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


def _matrix_from_prediction_bank(
    prediction_bank,
    *,
    mode: str,
    n_replicas: int | None,
) -> np.ndarray:
    centered, total_count = center_samples(prediction_bank, n_replicas=n_replicas)
    centered_flat = centered.reshape(centered.shape[0], centered.shape[1], -1)
    if mode == "complex":
        return np.concatenate([centered_flat[:, 0, :], centered_flat[:, 1, :]], axis=1) / math.sqrt(float(total_count))
    return centered_flat[:, 0, :] / math.sqrt(float(total_count))


def _final_prediction_metrics(
    target_vector: np.ndarray,
    prediction_vectors: np.ndarray,
    *,
    energy_variance: float,
) -> dict[str, object]:
    target_norm_sq = float(np.dot(target_vector, target_vector))
    residual_vectors = prediction_vectors - target_vector[None, :]
    residual_raw_per_repeat = np.einsum("ij,ij->i", residual_vectors, residual_vectors)
    if target_norm_sq > 1e-12:
        residual_norm_per_repeat = residual_raw_per_repeat / target_norm_sq
    else:
        residual_norm_per_repeat = np.full_like(residual_raw_per_repeat, np.inf)

    if prediction_vectors.shape[0] > 1:
        centered = prediction_vectors - prediction_vectors.mean(axis=0, keepdims=True)
        var_hat_raw = float(np.square(centered).sum() / float(prediction_vectors.shape[0] - 1))
    else:
        var_hat_raw = 0.0

    return {
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


def _record_summary(record) -> dict[str, Any]:
    return {
        "lambda": float(record.lambda_value),
        "delta_l2_norm": float(tree_l2_norm(record.delta)),
        "train_residual_raw": float(record.train_metrics["residual_raw"]),
        "train_residual_norm": float(record.train_metrics["residual_norm"]),
        "train_target_norm_sq": float(record.train_metrics["target_norm_sq"]),
        "energy_variance": float(record.train_metrics["energy_variance"]),
    }


def _same_batch_repeat_payload(driver, lambda_grid: tuple[float, ...]) -> dict[str, Any]:
    batch = sample_prepared_batch(driver, chain_name="mixture_same_batch")
    records = [
        base_solve_record(
            driver,
            batch,
            diag_shift=float(lam),
            collect_residual_info=True,
        )
        for lam in lambda_grid
    ]
    return {
        "batch_mode": "same_batch",
        "deltas": [numpy_tree(record.delta) for record in records],
        "per_lambda": [_record_summary(record) for record in records],
        "gcv_y": np.asarray(records[0].training_target, dtype=np.float64),
        "gcv_mu": np.asarray(records[0].ntk_eigenvalues, dtype=np.float64),
        "gcv_z": np.column_stack([record.training_prediction for record in records]),
    }


def _indep_batch_repeat_payload(driver, lambda_grid: tuple[float, ...]) -> dict[str, Any]:
    records = []
    for index, lam in enumerate(lambda_grid):
        batch = sample_prepared_batch(driver, chain_name=f"mixture_indep_{index}")
        records.append(
            base_solve_record(
                driver,
                batch,
                diag_shift=float(lam),
                collect_residual_info=False,
            )
        )
    return {
        "batch_mode": "indep_batch",
        "deltas": [numpy_tree(record.delta) for record in records],
        "per_lambda": [_record_summary(record) for record in records],
    }


def _single_oracle_repeat_payload(driver, oracle_lambda: float) -> dict[str, Any]:
    batch = sample_prepared_batch(driver, chain_name="mixture_single_oracle")
    record = base_solve_record(
        driver,
        batch,
        diag_shift=float(oracle_lambda),
        collect_residual_info=False,
    )
    return {
        "batch_mode": "single_oracle",
        "deltas": [numpy_tree(record.delta)],
        "per_lambda": [_record_summary(record)],
    }


def _indep_single_lambda_repeat_payload(
    driver,
    *,
    oracle_lambda: float,
    group_size: int,
) -> dict[str, Any]:
    records = []
    for index in range(int(group_size)):
        batch = sample_prepared_batch(driver, chain_name=f"mixture_indep_single_{index}")
        records.append(
            base_solve_record(
                driver,
                batch,
                diag_shift=float(oracle_lambda),
                collect_residual_info=False,
            )
        )
    return {
        "batch_mode": "indep_single_lambda",
        "deltas": [numpy_tree(record.delta) for record in records],
        "per_lambda": [_record_summary(record) for record in records],
    }


def precompute_mixture_bank(
    ckpt_path,
    *,
    system_name: str,
    repeat_count: int,
    lambda_grid,
    method_ids=None,
    oracle_lambda: float | None = None,
    single_lambda_count: int | None = None,
    output_path=None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    system_name = normalize_system_name(system_name)
    lambda_grid = tuple(float(x) for x in parse_sr_lambda_grid(lambda_grid))
    methods = tuple(MIXTURE_METHOD_IDS if method_ids is None else method_ids)
    methods = validate_mixture_method_ids(
        methods,
        lambda_grid=lambda_grid,
        single_lambda_count=single_lambda_count,
    )
    required_branches = required_mixture_branches(methods)
    needs_single_oracle = "single_oracle" in required_branches
    needs_indep_single = "indep_single_lambda" in required_branches
    if needs_indep_single and single_lambda_count is None:
        single_lambda_count = len(lambda_grid)
    if any(mixture_method_requires_oracle(method_id) for method_id in methods) and oracle_lambda is None:
        raise ValueError("oracle_lambda is required for single-oracle shared-checkpoint methods")

    training = _load_training_driver(ckpt_path, system_name=system_name)
    bank_path = (
        Path(output_path)
        if output_path is not None
        else mixture_bank_output_path(ckpt_path, int(training["step"]))
    )
    if bank_path.exists():
        with bank_path.open("rb") as f:
            payload = pickle.load(f)
        if _mixture_bank_compatible(
            payload,
            repeat_count=repeat_count,
            lambda_grid=lambda_grid,
            oracle_lambda=oracle_lambda,
            single_lambda_count=single_lambda_count,
            required_branches=required_branches,
        ):
            return payload

    driver = training["driver"]
    snapshot = snapshot_driver_state(driver)
    try:
        payload = {
            "system": system_name,
            "step": int(training["step"]),
            "lambda": float(training["config"]["diag_shift"]),
            "source_training_lambda": float(training["config"]["diag_shift"]),
            "lambda_grid": [float(x) for x in lambda_grid],
            "lambda_grid_tag": lambda_grid_tag(lambda_grid),
            "repeat_count": int(repeat_count),
        }
        if oracle_lambda is not None:
            payload["oracle_lambda"] = float(oracle_lambda)
        if metadata:
            payload.update(dict(metadata))
        if "same_batch" in required_branches:
            payload["same_batch"] = {
                "batch_mode": "same_batch",
                "repeats": [],
            }
        if "indep_batch" in required_branches:
            payload["indep_batch"] = {
                "batch_mode": "indep_batch",
                "repeats": [],
            }
        if needs_single_oracle:
            payload["single_oracle"] = {
                "batch_mode": "single_oracle",
                "lambda_value": float(oracle_lambda),
                "repeats": [],
            }
        if needs_indep_single:
            payload["indep_single_lambda"] = {
                "batch_mode": "indep_single_lambda",
                "lambda_value": float(oracle_lambda),
                "group_size": int(single_lambda_count),
                "repeats": [],
            }
        for _ in range(int(repeat_count)):
            if "same_batch" in required_branches:
                payload["same_batch"]["repeats"].append(_same_batch_repeat_payload(driver, lambda_grid))
            if "indep_batch" in required_branches:
                payload["indep_batch"]["repeats"].append(_indep_batch_repeat_payload(driver, lambda_grid))
            if needs_single_oracle:
                payload["single_oracle"]["repeats"].append(
                    _single_oracle_repeat_payload(driver, float(oracle_lambda))
                )
            if needs_indep_single:
                payload["indep_single_lambda"]["repeats"].append(
                    _indep_single_lambda_repeat_payload(
                        driver,
                        oracle_lambda=float(oracle_lambda),
                        group_size=int(single_lambda_count),
                    )
                )
    finally:
        restore_driver_state(driver, snapshot)

    atomic_save(payload, bank_path)
    return payload


def _prediction_bank_for_deltas(
    ckpt_path,
    *,
    system_name: str,
    delta_bank,
    n_val: int,
    validation_round_samples: int | None,
    validation_seed_offset: int,
    jvp_chunk: int | None,
    delta_batch_size: int,
):
    context = _load_validation_context(
        ckpt_path,
        system_name=system_name,
        n_val=n_val,
        validation_round_samples=validation_round_samples,
        validation_seed_offset=validation_seed_offset,
    )
    prediction_bank, energy_channels = _stream_prediction_cache(
        driver=context["driver"],
        validation_vstate=context["validation_vstate"],
        delta_bank=delta_bank,
        mode=context["mode"],
        n_val=int(n_val),
        round_samples=int(context["round_samples"]),
        jvp_chunk=jvp_chunk,
        delta_batch_size=delta_batch_size,
        n_replicas=context["n_replicas"],
        compute_eloc_round=context["compute_eloc_round"],
    )
    return prediction_bank, energy_channels, context


def _fit_stacking_weights(
    ckpt_path,
    *,
    system_name: str,
    repeat_payload: dict[str, Any],
    weight_fit_samples: int,
    validation_round_samples: int | None,
    jvp_chunk: int | None,
) -> np.ndarray:
    delta_bank = stack_tree([jax_tree(delta) for delta in repeat_payload["deltas"]])
    prediction_bank, energy_channels, context = _prediction_bank_for_deltas(
        ckpt_path,
        system_name=system_name,
        delta_bank=delta_bank,
        n_val=int(weight_fit_samples),
        validation_round_samples=validation_round_samples,
        validation_seed_offset=31_000,
        jvp_chunk=jvp_chunk,
        delta_batch_size=1,
    )
    y_weight, _ = _vector_from_energy_channels(
        energy_channels,
        mode=context["mode"],
        n_replicas=context["n_replicas"],
    )
    z_weight = _matrix_from_prediction_bank(
        prediction_bank,
        mode=context["mode"],
        n_replicas=context["n_replicas"],
    ).T
    return compute_stacking_mixture_weights(y_weight, z_weight)


def _method_weights(
    bank_payload: dict[str, Any],
    method_id: str,
    *,
    ckpt_path,
    system_name: str,
    weight_fit_samples: int,
    validation_round_samples: int | None,
    jvp_chunk: int | None,
) -> list[np.ndarray]:
    spec = parse_mixture_method_id(method_id)
    branch = spec["branch"]
    weight_scheme = spec["weight_scheme"]
    if branch == "single_oracle":
        return [np.array([1.0], dtype=np.float64) for _ in bank_payload["single_oracle"]["repeats"]]
    if branch == "indep_single_lambda":
        return [
            uniform_mixture_weights(len(repeat["deltas"]))
            for repeat in bank_payload["indep_single_lambda"]["repeats"]
        ]
    if branch == "same_batch" and weight_scheme == "uniform":
        return [uniform_mixture_weights(len(repeat["deltas"])) for repeat in bank_payload["same_batch"]["repeats"]]
    if branch == "same_batch" and weight_scheme == "gcv":
        weights = []
        lambdas = np.asarray(bank_payload["lambda_grid"], dtype=np.float64)
        for repeat in bank_payload["same_batch"]["repeats"]:
            weights.append(
                compute_gcv_mixture_weights(
                    repeat["gcv_y"],
                    repeat["gcv_z"],
                    repeat["gcv_mu"],
                    lambdas,
                )
            )
        return weights
    if branch == "indep_batch" and weight_scheme == "uniform":
        return [uniform_mixture_weights(len(repeat["deltas"])) for repeat in bank_payload["indep_batch"]["repeats"]]
    if branch == "indep_batch" and weight_scheme == "gcv":
        weights = []
        lambdas = np.asarray(bank_payload["lambda_grid"], dtype=np.float64)
        for repeat in bank_payload["same_batch"]["repeats"]:
            weights.append(
                compute_gcv_mixture_weights(
                    repeat["gcv_y"],
                    repeat["gcv_z"],
                    repeat["gcv_mu"],
                    lambdas,
                )
            )
        return weights
    if branch == "indep_batch" and weight_scheme == "stacking":
        return [
            _fit_stacking_weights(
                ckpt_path,
                system_name=system_name,
                repeat_payload=repeat,
                weight_fit_samples=weight_fit_samples,
                validation_round_samples=validation_round_samples,
                jvp_chunk=jvp_chunk,
            )
            for repeat in bank_payload["indep_batch"]["repeats"]
        ]
    raise ValueError(f"Unknown mixture method id: {method_id}")


def _final_deltas_for_method(
    bank_payload: dict[str, Any],
    method_id: str,
    weights_per_repeat: list[np.ndarray],
):
    branch = parse_mixture_method_id(method_id)["branch"]
    final_deltas = []
    for repeat_payload, weights in zip(bank_payload[branch]["repeats"], weights_per_repeat, strict=True):
        delta_trees = [jax_tree(delta) for delta in repeat_payload["deltas"]]
        final_deltas.append(weighted_tree_sum(delta_trees, weights))
    return final_deltas


def _evaluate_method(
    ckpt_path,
    *,
    system_name: str,
    bank_payload: dict[str, Any],
    method_id: str,
    weights_per_repeat: list[np.ndarray],
    eval_samples: int,
    validation_round_samples: int | None,
    jvp_chunk: int | None,
) -> dict[str, Any]:
    spec = parse_mixture_method_id(method_id)
    final_deltas = _final_deltas_for_method(bank_payload, method_id, weights_per_repeat)
    delta_bank = stack_tree(final_deltas)
    prediction_bank, energy_channels, context = _prediction_bank_for_deltas(
        ckpt_path,
        system_name=system_name,
        delta_bank=delta_bank,
        n_val=int(eval_samples),
        validation_round_samples=validation_round_samples,
        validation_seed_offset=47_000,
        jvp_chunk=jvp_chunk,
        delta_batch_size=1,
    )
    target_vector, energy_variance = _vector_from_energy_channels(
        energy_channels,
        mode=context["mode"],
        n_replicas=context["n_replicas"],
    )
    prediction_vectors = _matrix_from_prediction_bank(
        prediction_bank,
        mode=context["mode"],
        n_replicas=context["n_replicas"],
    )
    metrics = _final_prediction_metrics(
        target_vector,
        prediction_vectors,
        energy_variance=energy_variance,
    )
    metrics.update(
        {
            "method_id": method_id,
            "batch_mode": spec["batch_mode"],
            "weight_scheme": spec["weight_scheme"],
            "method_family": spec["family"],
            "k": int(spec["k"]) if spec["k"] is not None else int(len(weights_per_repeat[0])),
            "weights_per_repeat": [[float(x) for x in weights] for weights in weights_per_repeat],
            "mean_weights": [
                float(x)
                for x in np.mean(np.asarray(weights_per_repeat, dtype=np.float64), axis=0)
            ],
        }
    )
    return metrics


def evaluate_mixture_bank(
    ckpt_path,
    *,
    system_name: str,
    repeat_count: int,
    lambda_grid,
    n_val: int,
    weight_fit_samples: int | None = None,
    validation_round_samples: int | None = None,
    jvp_chunk: int | None = None,
    method_ids=None,
    oracle_lambda: float | None = None,
    single_lambda_count: int | None = None,
    bank_output_path=None,
    output_path=None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    system_name = normalize_system_name(system_name)
    lambda_grid = tuple(float(x) for x in parse_sr_lambda_grid(lambda_grid))
    methods_to_run = tuple(MIXTURE_METHOD_IDS if method_ids is None else method_ids)
    if single_lambda_count is None:
        single_lambda_count = len(lambda_grid)
    methods_to_run = validate_mixture_method_ids(
        methods_to_run,
        lambda_grid=lambda_grid,
        single_lambda_count=single_lambda_count,
    )
    training = _load_training_driver(ckpt_path, system_name=system_name)
    bank_payload = precompute_mixture_bank(
        ckpt_path,
        system_name=system_name,
        repeat_count=repeat_count,
        lambda_grid=lambda_grid,
        method_ids=methods_to_run,
        oracle_lambda=oracle_lambda,
        single_lambda_count=single_lambda_count,
        output_path=bank_output_path,
        metadata=metadata,
    )

    if weight_fit_samples is None:
        if any(mixture_method_requires_weight_fit(method_id) for method_id in methods_to_run):
            weight_fit_samples = int(training["config"]["n_samples"])
        else:
            weight_fit_samples = 0
    weight_fit_samples = int(weight_fit_samples)
    if weight_fit_samples < 0:
        raise ValueError(f"weight_fit_samples must be non-negative, got {weight_fit_samples}")
    if any(mixture_method_requires_weight_fit(method_id) for method_id in methods_to_run) and weight_fit_samples == 0:
        raise ValueError("Stacking-based mixture methods require weight_fit_samples > 0")
    eval_samples = int(n_val) - int(weight_fit_samples)
    if eval_samples <= 0:
        raise ValueError(
            f"n_val={n_val} must exceed weight_fit_samples={weight_fit_samples} for mixture evaluation"
        )
    if system_name == "tfim":
        validation_h_count = int(training["config"]["validation_h_count"])
        if weight_fit_samples % validation_h_count != 0 or eval_samples % validation_h_count != 0:
            raise ValueError(
                f"weight_fit_samples={weight_fit_samples} and eval_samples={eval_samples} "
                f"must both be divisible by validation_h_count={validation_h_count}"
            )

    methods = {}
    for method_id in methods_to_run:
        weights_per_repeat = _method_weights(
            bank_payload,
            method_id,
            ckpt_path=ckpt_path,
            system_name=system_name,
            weight_fit_samples=weight_fit_samples,
            validation_round_samples=validation_round_samples,
            jvp_chunk=jvp_chunk,
        )
        methods[method_id] = _evaluate_method(
            ckpt_path,
            system_name=system_name,
            bank_payload=bank_payload,
            method_id=method_id,
            weights_per_repeat=weights_per_repeat,
            eval_samples=eval_samples,
            validation_round_samples=validation_round_samples,
            jvp_chunk=jvp_chunk,
        )

    payload = {
        "system": system_name,
        "step": int(training["step"]),
        "lambda": float(training["config"]["diag_shift"]),
        "source_training_lambda": float(training["config"]["diag_shift"]),
        "lambda_grid": [float(x) for x in lambda_grid],
        "lambda_grid_tag": lambda_grid_tag(lambda_grid),
        "repeat_count": int(repeat_count),
        "split_sizes": {
            "b_weight": int(weight_fit_samples),
            "c_eval": int(eval_samples),
        },
        "methods": methods,
    }
    if oracle_lambda is not None:
        payload["oracle_lambda"] = float(oracle_lambda)
    if metadata:
        payload.update(dict(metadata))
    output_eval_path = (
        Path(output_path)
        if output_path is not None and str(output_path).endswith(".pkl")
        else mixture_eval_output_path(ckpt_path, int(training["step"]))
    )
    atomic_save(payload, output_eval_path)
    return payload
