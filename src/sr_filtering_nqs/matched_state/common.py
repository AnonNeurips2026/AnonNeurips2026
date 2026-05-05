from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "True")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGES = PROJECT_ROOT / "nqs_support_core" / "packages"
if str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import netket as nk
import nqs_support_core as nkp
import numpy as np
import scipy.sparse.linalg as spla
from advanced_drivers.driver import InfidelityOptimizerNG_FS

from sr_filtering_nqs.small_scale.common import (
    checkpoint_metric_value,
    compute_exact_regression_diagnostics,
    copy_variables,
    make_model,
    run_trials_for_checkpoint,
    save_pickle,
)


def build_fullsum_state(
    hilbert: Any,
    model: Any,
    variables: dict[str, Any],
) -> nk.vqs.FullSumState:
    return nk.vqs.FullSumState(hilbert=hilbert, model=model, variables=variables)


def normalized_state_vector(vstate: nk.vqs.FullSumState) -> np.ndarray:
    psi = np.asarray(vstate.to_array(normalize=False), dtype=np.complex128)
    norm = np.linalg.norm(psi)
    if norm == 0.0:
        raise ValueError("Encountered zero-norm state while computing exact fidelity.")
    return psi / norm


def exact_overlap_fidelity(
    target_state: nk.vqs.FullSumState,
    variational_state: nk.vqs.FullSumState,
) -> dict[str, float]:
    psi_target = normalized_state_vector(target_state)
    psi_var = normalized_state_vector(variational_state)
    overlap = np.vdot(psi_target, psi_var)
    fidelity = float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))
    return {
        "overlap_real": float(np.real(overlap)),
        "overlap_imag": float(np.imag(overlap)),
        "overlap_abs": float(np.abs(overlap)),
        "fidelity": fidelity,
        "infidelity_direct": float(max(0.0, 1.0 - fidelity)),
    }


def exact_vector_fidelity(
    target_vector: np.ndarray,
    variational_state: nk.vqs.FullSumState,
) -> dict[str, float]:
    psi_target = np.asarray(target_vector, dtype=np.complex128)
    norm = np.linalg.norm(psi_target)
    if norm == 0.0:
        raise ValueError("Encountered zero-norm target vector while computing exact fidelity.")
    psi_target = psi_target / norm
    psi_var = normalized_state_vector(variational_state)
    overlap = np.vdot(psi_target, psi_var)
    fidelity = float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))
    return {
        "overlap_real": float(np.real(overlap)),
        "overlap_imag": float(np.imag(overlap)),
        "overlap_abs": float(np.abs(overlap)),
        "fidelity": fidelity,
        "infidelity_direct": float(max(0.0, 1.0 - fidelity)),
    }


def exact_operator_infidelity(
    target_state: nk.vqs.FullSumState,
    variational_state: nk.vqs.FullSumState,
) -> float:
    infidelity_op = nkp.infidelity.InfidelityOperator(target_state)
    stats = variational_state.expect(infidelity_op)
    return float(max(0.0, stats.mean))


def evaluate_state_match(
    target_state: nk.vqs.FullSumState,
    variational_state: nk.vqs.FullSumState,
) -> dict[str, float]:
    metrics = exact_overlap_fidelity(target_state, variational_state)
    metrics["infidelity_operator"] = exact_operator_infidelity(target_state, variational_state)
    metrics["infidelity_abs_error"] = float(
        abs(metrics["infidelity_direct"] - metrics["infidelity_operator"])
    )
    return metrics


def compute_exact_ground_state(
    hamiltonian: Any,
) -> dict[str, Any]:
    sparse = hamiltonian.to_sparse().astype(np.complex128)
    eigenvalues, eigenvectors = spla.eigsh(sparse, k=1, which="SA")
    energy = float(np.real(eigenvalues[0]))
    vector = np.asarray(eigenvectors[:, 0], dtype=np.complex128)
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        raise ValueError("Encountered zero-norm eigenvector while diagonalizing Hamiltonian.")
    vector /= norm
    return {
        "energy": energy,
        "vector": vector,
    }


def select_checkpoint_by_step(
    checkpoint_summaries: list[dict[str, Any]],
    step: int,
) -> tuple[int, dict[str, Any]]:
    for idx, summary in enumerate(checkpoint_summaries):
        if int(summary["step"]) == int(step):
            return idx, summary
    available = [int(item["step"]) for item in checkpoint_summaries]
    raise ValueError(f"Requested step {step} not found. Available steps: {available}")


def run_infidelity_matching(
    *,
    hilbert: Any,
    graph: Any,
    square_L: int,
    j_couplings: tuple[float, float],
    model_name: str,
    target_state: nk.vqs.FullSumState,
    learning_rate: float,
    diag_shift: float,
    match_steps: int,
    seed: int,
    fidelity_threshold: float | None = None,
    initial_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model, model_desc = make_model(
        model_name,
        hilbert=hilbert,
        graph=graph,
        square_L=square_L,
        j_couplings=j_couplings,
    )
    variational_state = nk.vqs.FullSumState(hilbert=hilbert, model=model, seed=seed)
    if initial_variables is not None:
        variational_state.variables = copy_variables(initial_variables)
    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = InfidelityOptimizerNG_FS(
        target_state=target_state,
        optimizer=optimizer,
        variational_state=variational_state,
        diag_shift=diag_shift,
    )

    history: list[dict[str, Any]] = []
    best_variables = copy_variables(variational_state.variables)
    best_history_index = 0
    best_metrics: dict[str, float] | None = None
    stop_reason = "max_steps_reached"

    for step in range(match_steps + 1):
        if step > 0:
            driver.run(1, show_progress=False)

        metrics = evaluate_state_match(target_state, variational_state)
        entry = {"step": int(step), **metrics}
        history.append(entry)

        if best_metrics is None or metrics["infidelity_direct"] < best_metrics["infidelity_direct"]:
            best_metrics = dict(entry)
            best_variables = copy_variables(variational_state.variables)
            best_history_index = len(history) - 1

        if fidelity_threshold is not None and metrics["fidelity"] >= fidelity_threshold:
            stop_reason = "threshold_reached"
            break

    assert best_metrics is not None
    final_metrics = dict(history[-1])
    success = (
        True
        if fidelity_threshold is None
        else bool(best_metrics["fidelity"] >= fidelity_threshold)
    )

    return {
        "model": model,
        "model_desc": model_desc,
        "history": history,
        "best_variables": best_variables,
        "best_history_index": int(best_history_index),
        "best_step": int(best_metrics["step"]),
        "best_fidelity": float(best_metrics["fidelity"]),
        "best_infidelity": float(best_metrics["infidelity_direct"]),
        "best_operator_infidelity": float(best_metrics["infidelity_operator"]),
        "final_step": int(final_metrics["step"]),
        "final_fidelity": float(final_metrics["fidelity"]),
        "final_infidelity": float(final_metrics["infidelity_direct"]),
        "final_operator_infidelity": float(final_metrics["infidelity_operator"]),
        "initial_step": int(history[0]["step"]),
        "initial_fidelity": float(history[0]["fidelity"]),
        "initial_infidelity": float(history[0]["infidelity_direct"]),
        "initial_operator_infidelity": float(history[0]["infidelity_operator"]),
        "consistency_max_abs_error": float(
            max(item["infidelity_abs_error"] for item in history)
        ),
        "fidelity_threshold": None if fidelity_threshold is None else float(fidelity_threshold),
        "success": bool(success),
        "stop_reason": stop_reason,
        "steps_completed": int(len(history) - 1),
    }


def strip_large_exact_fields(exact_diag: dict[str, Any]) -> dict[str, Any]:
    stored = dict(exact_diag)
    stored.pop("S_exact", None)
    return stored


def materialize_model_result(
    *,
    model: Any,
    model_name: str,
    model_desc: str,
    state_origin: str,
    variables: dict[str, Any],
    hamiltonian: Any,
    all_states: np.ndarray,
    lambda_grid: np.ndarray,
    n_samples: int,
    n_trials: int,
    trial_seed: int,
    checkpoint_step: int,
    selected_index: int,
    selected_reason: str,
    selection_metric: str,
    selection_metric_value: float,
    checkpoint_summaries: list[dict[str, Any]],
    jac_chunk_size: int,
    apply_chunk_size: int,
) -> dict[str, Any]:
    exact_diag = compute_exact_regression_diagnostics(
        model,
        variables,
        hamiltonian,
        all_states,
        jac_chunk_size=jac_chunk_size,
        apply_chunk_size=apply_chunk_size,
        store_arrays=True,
        store_matrix=True,
    )
    exact_diag["step"] = int(checkpoint_step)
    exact_diag["selected_reason"] = selected_reason
    exact_diag["selection_metric"] = selection_metric
    exact_diag["selection_metric_value"] = float(selection_metric_value)

    trials, summary, representative = run_trials_for_checkpoint(
        model=model,
        variables=variables,
        all_states=all_states,
        exact_diag=exact_diag,
        lambda_grid=lambda_grid,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=trial_seed,
        jac_chunk_size=jac_chunk_size,
        apply_chunk_size=apply_chunk_size,
        feature_matrix_centered=None,
    )

    stored_diag = strip_large_exact_fields(exact_diag)
    return {
        "model_name": model_name,
        "model_desc": model_desc,
        "state_origin": state_origin,
        "P_real": int(stored_diag["p_real"]),
        "P_over_Ns": float(stored_diag["p_real"] / n_samples),
        "checkpoint_step": int(checkpoint_step),
        "checkpoint_energy": float(stored_diag["energy"]),
        "noise_fraction": float(stored_diag["noise_fraction"]),
        "sigma_gap_sq": float(stored_diag["sigma_gap_sq"]),
        "target_var": float(stored_diag["target_var"]),
        "selected_index": int(selected_index),
        "selected_reason": selected_reason,
        "selection_metric": selection_metric,
        "selection_metric_value": float(selection_metric_value),
        "checkpoint_summaries": checkpoint_summaries,
        "selected_checkpoint": stored_diag,
        "trials": trials,
        "summary": summary,
        "representative": representative,
    }


def rbm_selection_value(
    *,
    checkpoint_summary: dict[str, Any],
    selection_metric: str,
    fallback_step: int,
) -> float:
    if selection_metric == "step":
        return float(fallback_step)
    return checkpoint_metric_value(checkpoint_summary, selection_metric)


def build_checkpoint_payload(
    *,
    system_meta: dict[str, Any],
    model_name: str,
    model_desc: str,
    state_origin: str,
    step: int,
    variables: dict[str, Any],
    result: dict[str, Any],
    match_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "system": dict(system_meta),
        "model_name": model_name,
        "model_desc": model_desc,
        "state_origin": state_origin,
        "step": int(step),
        "variables": copy_variables(variables),
        "exact_diagnostics": {
            "checkpoint_energy": float(result["checkpoint_energy"]),
            "sigma_gap_sq": float(result["sigma_gap_sq"]),
            "noise_fraction": float(result["noise_fraction"]),
            "target_var": float(result["target_var"]),
            "P_real": int(result["P_real"]),
            "P_over_Ns": float(result["P_over_Ns"]),
        },
    }
    if match_metrics is not None:
        payload["match_metrics"] = dict(match_metrics)
    return payload


def save_checkpoint_artifact(path: Path, payload: dict[str, Any]) -> None:
    save_pickle(path, payload)
