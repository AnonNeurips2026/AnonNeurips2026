from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE2_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sr_filtering_nqs.small_scale.common import (
    compute_exact_regression_diagnostics,
    load_pickle,
    make_model,
    make_square_j1j2,
    now,
    precompute_centered_feature_matrix,
    run_trials_for_checkpoint,
    save_pickle,
)
from sr_filtering_nqs.small_scale.plot_figure1 import plot_main_figure
from sr_filtering_nqs.matched_state.common import (
    build_fullsum_state,
    compute_exact_ground_state,
    evaluate_state_match,
    exact_vector_fidelity,
)


DEFAULT_RBM_CHECKPOINT = (
    FIGURE2_DIR
    / "checkpoints_pairmax_until_0p99_from_900459"
    / "alternating_stage23_rbm_a2_to_vit_d2_m24_h4_e4_best.pkl"
)
DEFAULT_VIT_CHECKPOINT = (
    FIGURE2_DIR / "checkpoints" / "fullsum_exact_until_0p99_from_913152_best.pkl"
)
DEFAULT_OUTPUT = FIGURE2_DIR / "results" / "best_fidelity_pair_figure1_style.pkl"
DEFAULT_FIGURE_DIR = FIGURE2_DIR / "figures_best_fidelity_pair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a Figure-1-style one-step SR scatter plot for two saved Figure 2 "
            "checkpoint payloads."
        ),
    )
    parser.add_argument(
        "--rbm-checkpoint",
        type=str,
        default=str(DEFAULT_RBM_CHECKPOINT),
        help="Saved RBM checkpoint payload.",
    )
    parser.add_argument(
        "--vit-checkpoint",
        type=str,
        default=str(DEFAULT_VIT_CHECKPOINT),
        help="Saved ViT checkpoint payload.",
    )
    parser.add_argument("--ns", type=int, default=4096, help="Born-resampled training size.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of resampling trials.")
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=(1e-9, 1e-1),
        help="Lambda grid used in the one-step SR sweep.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed.")
    parser.add_argument("--jac-chunk-size", type=int, default=4096)
    parser.add_argument("--apply-chunk-size", type=int, default=4096)
    parser.add_argument(
        "--no-precompute-features",
        action="store_true",
        help="Avoid materializing the full centered feature matrix during trial evaluation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output result pickle.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=str(DEFAULT_FIGURE_DIR),
        help="Directory for figure1_main.pdf.",
    )
    return parser.parse_args()


def validate_same_system(rbm_payload: dict[str, Any], vit_payload: dict[str, Any]) -> dict[str, Any]:
    rbm_system = dict(rbm_payload["system"])
    vit_system = dict(vit_payload["system"])
    for key in ("name", "L", "J1", "J2"):
        if rbm_system.get(key) != vit_system.get(key):
            raise ValueError(
                f"Checkpoint system mismatch on {key}: "
                f"{rbm_system.get(key)} != {vit_system.get(key)}"
            )
    return rbm_system


def materialize_model_result(
    *,
    payload: dict[str, Any],
    role: str,
    model: Any,
    model_desc: str,
    hamiltonian: Any,
    all_states: np.ndarray,
    lambda_grid: np.ndarray,
    n_samples: int,
    n_trials: int,
    seed: int,
    jac_chunk_size: int,
    apply_chunk_size: int,
    precompute_features: bool,
) -> dict[str, Any]:
    t0 = time.time()
    exact_diag = compute_exact_regression_diagnostics(
        model,
        payload["variables"],
        hamiltonian,
        all_states,
        jac_chunk_size=jac_chunk_size,
        apply_chunk_size=apply_chunk_size,
        store_arrays=True,
        store_matrix=not precompute_features,
    )
    exact_diag["timing_seconds"] = float(time.time() - t0)
    exact_diag["step"] = int(payload["step"])
    exact_diag["selected_reason"] = "best_fidelity_checkpoint_pair"
    exact_diag["selection_metric"] = "checkpoint_pair"
    exact_diag["selection_metric_value"] = float(payload["step"])

    feature_matrix_centered = None
    if precompute_features:
        feature_matrix_centered = precompute_centered_feature_matrix(
            model,
            payload["variables"],
            all_states,
            np.asarray(exact_diag["feature_mean"], dtype=np.float64),
            jac_chunk_size=jac_chunk_size,
        )

    trials, summary, representative = run_trials_for_checkpoint(
        model=model,
        variables=payload["variables"],
        all_states=all_states,
        exact_diag=exact_diag,
        lambda_grid=lambda_grid,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=seed,
        jac_chunk_size=jac_chunk_size,
        apply_chunk_size=apply_chunk_size,
        feature_matrix_centered=feature_matrix_centered,
    )

    checkpoint_summary = {
        "step": int(payload["step"]),
        "energy": float(exact_diag["energy"]),
        "target_var": float(exact_diag["target_var"]),
        "sigma_gap_sq": float(exact_diag["sigma_gap_sq"]),
        "noise_fraction": float(exact_diag["noise_fraction"]),
        "timing_seconds": float(exact_diag["timing_seconds"]),
    }
    return {
        "model_name": payload["model_name"],
        "model_desc": payload.get("model_desc", model_desc),
        "state_origin": payload.get("state_origin", role),
        "P_real": int(exact_diag["p_real"]),
        "P_over_Ns": float(exact_diag["p_real"] / n_samples),
        "checkpoint_step": int(payload["step"]),
        "checkpoint_energy": float(exact_diag["energy"]),
        "noise_fraction": float(exact_diag["noise_fraction"]),
        "sigma_gap_sq": float(exact_diag["sigma_gap_sq"]),
        "target_var": float(exact_diag["target_var"]),
        "selected_index": 0,
        "selected_reason": "best_fidelity_checkpoint_pair",
        "selection_metric": "checkpoint_pair",
        "selection_metric_value": float(payload["step"]),
        "checkpoint_summaries": [checkpoint_summary],
        "checkpoint_path": payload.get("checkpoint_path"),
        "selected_checkpoint": exact_diag,
        "trials": trials,
        "summary": summary,
        "representative": representative,
    }


def main() -> None:
    args = parse_args()
    rbm_checkpoint = Path(args.rbm_checkpoint)
    vit_checkpoint = Path(args.vit_checkpoint)
    output_path = Path(args.output)
    figure_dir = Path(args.figure_dir)
    lambda_grid = np.asarray(args.lambda_values, dtype=np.float64)
    precompute_features = not args.no_precompute_features

    rbm_payload = load_pickle(rbm_checkpoint)
    vit_payload = load_pickle(vit_checkpoint)
    rbm_payload["checkpoint_path"] = str(rbm_checkpoint)
    vit_payload["checkpoint_path"] = str(vit_checkpoint)
    system = validate_same_system(rbm_payload, vit_payload)

    hilbert, hamiltonian, graph, system_meta = make_square_j1j2(
        L=int(system["L"]),
        J1=float(system["J1"]),
        J2=float(system["J2"]),
    )
    all_states = np.asarray(hilbert.all_states())
    square_L = int(system["L"])
    j_couplings = (float(system["J1"]), float(system["J2"]))

    rbm_model, rbm_desc = make_model(
        rbm_payload["model_name"],
        hilbert=hilbert,
        graph=graph,
        square_L=square_L,
        j_couplings=j_couplings,
    )
    vit_model, vit_desc = make_model(
        vit_payload["model_name"],
        hilbert=hilbert,
        graph=graph,
        square_L=square_L,
        j_couplings=j_couplings,
    )
    rbm_state = build_fullsum_state(hilbert, rbm_model, rbm_payload["variables"])
    vit_state = build_fullsum_state(hilbert, vit_model, vit_payload["variables"])
    pair_match = evaluate_state_match(rbm_state, vit_state)
    ground_state = compute_exact_ground_state(hamiltonian)
    ground_fidelities = {
        "fid_rbm_ground": exact_vector_fidelity(ground_state["vector"], rbm_state)["fidelity"],
        "fid_vit_ground": exact_vector_fidelity(ground_state["vector"], vit_state)["fidelity"],
        "ground_energy": float(ground_state["energy"]),
    }

    print(
        "Rendering best-fidelity pair: "
        f"F(RBM,ViT)={pair_match['fidelity']:.12f}, "
        f"Ns={args.ns}, n_trials={args.n_trials}, lambdas={lambda_grid}"
    )
    rbm_result = materialize_model_result(
        payload=rbm_payload,
        role="rbm",
        model=rbm_model,
        model_desc=rbm_desc,
        hamiltonian=hamiltonian,
        all_states=all_states,
        lambda_grid=lambda_grid,
        n_samples=args.ns,
        n_trials=args.n_trials,
        seed=args.seed + 100_000,
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
        precompute_features=precompute_features,
    )
    print(
        f"Prepared RBM: sigma_gap_sq={rbm_result['sigma_gap_sq']:.6e}, "
        f"noise_fraction={rbm_result['noise_fraction']:.6f}"
    )
    vit_result = materialize_model_result(
        payload=vit_payload,
        role="vit",
        model=vit_model,
        model_desc=vit_desc,
        hamiltonian=hamiltonian,
        all_states=all_states,
        lambda_grid=lambda_grid,
        n_samples=args.ns,
        n_trials=args.n_trials,
        seed=args.seed + 200_000,
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
        precompute_features=precompute_features,
    )
    print(
        f"Prepared ViT: sigma_gap_sq={vit_result['sigma_gap_sq']:.6e}, "
        f"noise_fraction={vit_result['noise_fraction']:.6f}"
    )

    results = {
        "format_version": 1,
        "experiment": "figure2_best_fidelity_pair_figure1_style",
        "created_at_utc": now(),
        "system": dict(system_meta),
        "ns": int(args.ns),
        "train_config": {
            "seed": int(args.seed),
            "jac_chunk_size": int(args.jac_chunk_size),
            "apply_chunk_size": int(args.apply_chunk_size),
            "precompute_features": bool(precompute_features),
        },
        "selection": {
            "mode": "best_saved_fidelity_pair",
            "rbm_checkpoint": str(rbm_checkpoint),
            "vit_checkpoint": str(vit_checkpoint),
        },
        "matching": {
            "matched_fidelity": float(pair_match["fidelity"]),
            "matched_infidelity_direct": float(pair_match["infidelity_direct"]),
            "matched_infidelity_operator": float(pair_match["infidelity_operator"]),
            "matched_infidelity_abs_error": float(pair_match["infidelity_abs_error"]),
            "matched_overlap_abs": float(pair_match["overlap_abs"]),
            **ground_fidelities,
        },
        "lambda_grid": np.asarray(lambda_grid, dtype=np.float64),
        "models": [rbm_result, vit_result],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(output_path, results)
    plot_main_figure(results, figure_dir / "figure1_main.pdf")
    print(f"Saved result pickle to {output_path}")
    print(f"Saved figure to {figure_dir / 'figure1_main.pdf'}")


if __name__ == "__main__":
    main()
