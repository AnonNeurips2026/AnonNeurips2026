from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE2_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sr_filtering_nqs.small_scale.common import (
    make_square_j1j2,
    make_model,
    now,
    save_pickle,
    select_checkpoint_target,
    train_model_checkpoints,
)
from sr_filtering_nqs.matched_state.common import (
    build_fullsum_state,
    build_checkpoint_payload,
    evaluate_state_match,
    materialize_model_result,
    rbm_selection_value as compute_rbm_selection_value,
    run_infidelity_matching,
    save_checkpoint_artifact,
    select_checkpoint_by_step,
)


DEFAULT_OUTPUT_TEMPLATE = FIGURE2_DIR / "results" / "j1j2_matched_state_ns{ns}.pkl"
SELECTION_METRICS = ("noise_fraction", "sigma_gap_sq")
DEFAULT_CHECKPOINT_DIR = FIGURE2_DIR / "checkpoints"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def checkpoint_path_for_model(
    checkpoint_dir: Path,
    *,
    model_name: str,
    suffix: str,
) -> Path:
    return checkpoint_dir / f"{sanitize_name(model_name)}_{suffix}.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 2 exact matched-state RBM vs ViT experiment on 4x4 J1-J2.",
    )
    parser.add_argument("--L", type=int, default=4, help="Square lattice linear size.")
    parser.add_argument("--J1", type=float, default=1.0, help="Nearest-neighbor coupling.")
    parser.add_argument("--J2", type=float, default=0.5, help="Next-nearest-neighbor coupling.")
    parser.add_argument(
        "--rbm-model",
        type=str,
        default="RBM_a1",
        help="Underparameterized RBM model spec.",
    )
    parser.add_argument(
        "--vit-model",
        type=str,
        default="ViT_d2_m24_h4_e4",
        help="Overparameterized ViT model spec.",
    )
    parser.add_argument("--ns", type=int, default=2048, help="Fixed exact sample count.")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=4096,
        help="MCState samples during RBM training.",
    )
    parser.add_argument("--train-steps", type=int, default=200, help="Maximum RBM VMC steps.")
    parser.add_argument("--diag-every", type=int, default=10, help="Exact diagnostic interval.")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="RBM VMC learning rate.",
    )
    parser.add_argument(
        "--train-diag-shift",
        type=float,
        default=1e-3,
        help="RBM VMC training regularization.",
    )
    parser.add_argument(
        "--rbm-selection-metric",
        choices=SELECTION_METRICS,
        default="noise_fraction",
        help="Metric used to choose the RBM target checkpoint.",
    )
    parser.add_argument(
        "--rbm-target-value",
        type=float,
        default=0.30,
        help="Target metric value used when selecting the RBM checkpoint.",
    )
    parser.add_argument(
        "--rbm-step",
        type=int,
        default=None,
        help="Optional explicit RBM checkpoint step override.",
    )
    parser.add_argument(
        "--match-max-steps",
        dest="match_max_steps",
        type=int,
        default=2000,
        help="Maximum exact full-sum infidelity optimization steps for the ViT.",
    )
    parser.add_argument(
        "--match-steps",
        dest="match_max_steps",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--match-learning-rate",
        type=float,
        default=5e-2,
        help="Learning rate for exact infidelity minimization.",
    )
    parser.add_argument(
        "--match-diag-shift",
        type=float,
        default=1e-4,
        help="Diagonal shift for exact infidelity minimization.",
    )
    parser.add_argument(
        "--match-fidelity-threshold",
        type=float,
        default=0.99,
        help="Exact fidelity threshold for freezing the matched ViT checkpoint.",
    )
    parser.add_argument(
        "--retry-vit-model",
        type=str,
        default="ViT_d2_m32_h4_e4",
        help="Optional larger ViT used if the primary match model misses the fidelity threshold.",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of exact Born trials.")
    parser.add_argument("--n-lambdas", type=int, default=17, help="Number of lambda grid points.")
    parser.add_argument(
        "--lambda-min-exp",
        type=float,
        default=-9.0,
        help="Minimum base-10 exponent for lambda grid.",
    )
    parser.add_argument(
        "--lambda-max-exp",
        type=float,
        default=-1.0,
        help="Maximum base-10 exponent for lambda grid.",
    )
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=None,
        help="Optional explicit lambda grid.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--jac-chunk-size",
        type=int,
        default=4096,
        help="Chunk size for exact Jacobian evaluations.",
    )
    parser.add_argument(
        "--apply-chunk-size",
        type=int,
        default=4096,
        help="Chunk size for exact forward and JVP evaluations.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle path. Defaults to figure2/results/j1j2_matched_state_ns{ns}.pkl",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory where standalone frozen checkpoint artifacts are written.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save standalone checkpoint files for the frozen RBM and ViT states.",
    )
    parser.add_argument(
        "--strict-match",
        action="store_true",
        help="Fail the run if the exact fidelity threshold is never reached.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambda_grid = (
        np.asarray(args.lambda_values, dtype=np.float64)
        if args.lambda_values is not None
        else np.logspace(args.lambda_min_exp, args.lambda_max_exp, args.n_lambdas)
    )
    hilbert, hamiltonian, graph, system_meta = make_square_j1j2(L=args.L, J1=args.J1, J2=args.J2)
    all_states = np.asarray(hilbert.all_states())
    output_path = (
        Path(args.output)
        if args.output
        else Path(str(DEFAULT_OUTPUT_TEMPLATE).format(ns=args.ns))
    )
    checkpoint_dir = Path(args.checkpoint_dir)

    rbm_model, rbm_desc = make_model(
        args.rbm_model,
        hilbert=hilbert,
        graph=graph,
        square_L=args.L,
        j_couplings=(args.J1, args.J2),
    )
    print("Figure 2 exact matched-state study")
    print(f"System: {system_meta}")
    print(f"RBM target model: {args.rbm_model} ({rbm_desc})")
    print(f"ViT match model: {args.vit_model}")
    print(f"Fidelity threshold: {args.match_fidelity_threshold:.3f}")
    print(f"Output: {output_path}")

    rbm_training = train_model_checkpoints(
        model=rbm_model,
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        all_states=all_states,
        train_samples=args.train_samples,
        train_steps=args.train_steps,
        diag_every=args.diag_every,
        learning_rate=args.learning_rate,
        train_diag_shift=args.train_diag_shift,
        seed=args.seed,
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
        store_checkpoint_arrays=False,
    )

    if args.rbm_step is not None:
        rbm_selected_idx, rbm_selected_summary = select_checkpoint_by_step(
            rbm_training["checkpoint_summaries"],
            args.rbm_step,
        )
        rbm_selected_reason = "explicit_step"
        rbm_selection_metric = "step"
        rbm_selected_metric_value = float(args.rbm_step)
    else:
        rbm_selected_idx, rbm_selected_reason = select_checkpoint_target(
            rbm_training["checkpoint_summaries"],
            metric_key=args.rbm_selection_metric,
            target_value=args.rbm_target_value,
        )
        rbm_selected_summary = rbm_training["checkpoint_summaries"][rbm_selected_idx]
        rbm_selection_metric = args.rbm_selection_metric
        rbm_selected_metric_value = compute_rbm_selection_value(
            checkpoint_summary=rbm_selected_summary,
            selection_metric=rbm_selection_metric,
            fallback_step=int(rbm_selected_summary["step"]),
        )

    rbm_selected_runtime = rbm_training["checkpoint_runtime"][rbm_selected_idx]
    rbm_selected_step = int(rbm_selected_runtime["step"])
    rbm_selected_vars = rbm_selected_runtime["variables"]
    rbm_target_state = build_fullsum_state(hilbert, rbm_model, rbm_selected_vars)

    rbm_result = materialize_model_result(
        model=rbm_model,
        model_name=args.rbm_model,
        model_desc=rbm_desc,
        state_origin="rbm_vmc_checkpoint",
        variables=rbm_selected_vars,
        hamiltonian=hamiltonian,
        all_states=all_states,
        lambda_grid=lambda_grid,
        n_samples=args.ns,
        n_trials=args.n_trials,
        trial_seed=args.seed + 100_000,
        checkpoint_step=rbm_selected_step,
        selected_index=int(rbm_selected_idx),
        selected_reason=rbm_selected_reason,
        selection_metric=rbm_selection_metric,
        selection_metric_value=rbm_selected_metric_value,
        checkpoint_summaries=rbm_training["checkpoint_summaries"],
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
    )
    checkpoint_artifacts: dict[str, str | None] = {
        "rbm_checkpoint": None,
        "vit_checkpoint": None,
    }
    rbm_result["checkpoint_artifact"] = None
    if args.save_checkpoints:
        rbm_checkpoint_path = checkpoint_path_for_model(
            checkpoint_dir,
            model_name=args.rbm_model,
            suffix=f"step{rbm_selected_step:04d}",
        )
        rbm_checkpoint_payload = build_checkpoint_payload(
            system_meta=system_meta,
            model_name=args.rbm_model,
            model_desc=rbm_desc,
            state_origin="rbm_vmc_checkpoint",
            step=rbm_selected_step,
            variables=rbm_selected_vars,
            result=rbm_result,
        )
        save_checkpoint_artifact(rbm_checkpoint_path, rbm_checkpoint_payload)
        checkpoint_artifacts["rbm_checkpoint"] = str(rbm_checkpoint_path)
        rbm_result["checkpoint_artifact"] = str(rbm_checkpoint_path)

    matching_attempts = []
    matching_candidates = [args.vit_model]
    retry_model = args.retry_vit_model.strip() if args.retry_vit_model else ""
    if retry_model and retry_model != args.vit_model:
        matching_candidates.append(retry_model)

    successful_matching = None
    for attempt_idx, candidate_model in enumerate(matching_candidates):
        attempt = run_infidelity_matching(
            hilbert=hilbert,
            graph=graph,
            square_L=args.L,
            j_couplings=(args.J1, args.J2),
            model_name=candidate_model,
            target_state=rbm_target_state,
            learning_rate=args.match_learning_rate,
            diag_shift=args.match_diag_shift,
            match_steps=args.match_max_steps,
            seed=args.seed + 50_000 + 10_000 * attempt_idx,
            fidelity_threshold=args.match_fidelity_threshold,
        )
        attempt["attempt_index"] = int(attempt_idx)
        attempt["requested_model_name"] = candidate_model
        if attempt_idx > 0 and not attempt["success"]:
            attempt["stop_reason"] = "max_steps_retry"
        matching_attempts.append(attempt)
        if attempt["success"]:
            successful_matching = attempt
            break

    if successful_matching is None:
        successful_matching = min(
            matching_attempts,
            key=lambda item: item["best_infidelity"],
        )
        failed_stop_reason = "failed_to_match"
    else:
        failed_stop_reason = successful_matching["stop_reason"]

    matched_vit_state = build_fullsum_state(
        hilbert,
        successful_matching["model"],
        successful_matching["best_variables"],
    )
    matched_metrics = evaluate_state_match(rbm_target_state, matched_vit_state)

    results = {
        "format_version": 1,
        "experiment": "figure2_matched_state",
        "created_at_utc": now(),
        "system": dict(system_meta),
        "ns": int(args.ns),
        "success": bool(successful_matching["success"]),
        "train_config": {
            "train_samples": int(args.train_samples),
            "train_steps": int(args.train_steps),
            "diag_every": int(args.diag_every),
            "learning_rate": float(args.learning_rate),
            "train_diag_shift": float(args.train_diag_shift),
            "seed": int(args.seed),
            "jac_chunk_size": int(args.jac_chunk_size),
            "apply_chunk_size": int(args.apply_chunk_size),
        },
        "selection": {
            "target_model_name": args.rbm_model,
            "metric": rbm_selection_metric,
            "target_value": None if args.rbm_step is not None else float(args.rbm_target_value),
            "explicit_step": None if args.rbm_step is None else int(args.rbm_step),
            "selected_index": int(rbm_selected_idx),
            "selected_step": int(rbm_selected_step),
            "selected_reason": rbm_selected_reason,
            "selected_metric_value": float(rbm_selected_metric_value),
        },
        "match_config": {
            "target_model_name": args.rbm_model,
            "match_model_name": args.vit_model,
            "match_max_steps": int(args.match_max_steps),
            "match_learning_rate": float(args.match_learning_rate),
            "match_diag_shift": float(args.match_diag_shift),
            "match_fidelity_threshold": float(args.match_fidelity_threshold),
            "retry_vit_model": None if not retry_model else retry_model,
            "seed": int(args.seed + 50_000),
            "strict_match": bool(args.strict_match),
        },
        "checkpoint_artifacts": checkpoint_artifacts,
        "matching": {
            "success": bool(successful_matching["success"]),
            "stop_reason": failed_stop_reason,
            "fidelity_threshold": float(args.match_fidelity_threshold),
            "attempt_count": int(len(matching_attempts)),
            "attempts": [
                {
                    "attempt_index": int(item["attempt_index"]),
                    "requested_model_name": item["requested_model_name"],
                    "model_desc": item["model_desc"],
                    "success": bool(item["success"]),
                    "stop_reason": item["stop_reason"],
                    "steps_completed": int(item["steps_completed"]),
                    "best_step": int(item["best_step"]),
                    "best_fidelity": float(item["best_fidelity"]),
                    "best_infidelity": float(item["best_infidelity"]),
                    "best_operator_infidelity": float(item["best_operator_infidelity"]),
                    "final_step": int(item["final_step"]),
                    "final_fidelity": float(item["final_fidelity"]),
                    "final_infidelity": float(item["final_infidelity"]),
                    "final_operator_infidelity": float(item["final_operator_infidelity"]),
                    "initial_step": int(item["initial_step"]),
                    "initial_fidelity": float(item["initial_fidelity"]),
                    "initial_infidelity": float(item["initial_infidelity"]),
                    "initial_operator_infidelity": float(item["initial_operator_infidelity"]),
                    "consistency_max_abs_error": float(item["consistency_max_abs_error"]),
                    "history": item["history"],
                }
                for item in matching_attempts
            ],
            "target_model_name": args.rbm_model,
            "target_model_desc": rbm_desc,
            "target_checkpoint_step": int(rbm_selected_step),
            "target_checkpoint_energy": float(rbm_result["checkpoint_energy"]),
            "match_model_name": successful_matching["requested_model_name"],
            "match_model_desc": successful_matching["model_desc"],
            "history": successful_matching["history"],
            "best_history_index": int(successful_matching["best_history_index"]),
            "best_step": int(successful_matching["best_step"]),
            "best_fidelity": float(successful_matching["best_fidelity"]),
            "best_infidelity": float(successful_matching["best_infidelity"]),
            "best_operator_infidelity": float(successful_matching["best_operator_infidelity"]),
            "final_step": int(successful_matching["final_step"]),
            "final_fidelity": float(successful_matching["final_fidelity"]),
            "final_infidelity": float(successful_matching["final_infidelity"]),
            "final_operator_infidelity": float(successful_matching["final_operator_infidelity"]),
            "initial_step": int(successful_matching["initial_step"]),
            "initial_fidelity": float(successful_matching["initial_fidelity"]),
            "initial_infidelity": float(successful_matching["initial_infidelity"]),
            "initial_operator_infidelity": float(successful_matching["initial_operator_infidelity"]),
            "consistency_max_abs_error": float(successful_matching["consistency_max_abs_error"]),
            "matched_overlap_real": float(matched_metrics["overlap_real"]),
            "matched_overlap_imag": float(matched_metrics["overlap_imag"]),
            "matched_overlap_abs": float(matched_metrics["overlap_abs"]),
            "matched_fidelity": float(matched_metrics["fidelity"]),
            "matched_infidelity_direct": float(matched_metrics["infidelity_direct"]),
            "matched_infidelity_operator": float(matched_metrics["infidelity_operator"]),
            "matched_infidelity_abs_error": float(matched_metrics["infidelity_abs_error"]),
        },
        "lambda_grid": np.asarray(lambda_grid, dtype=np.float64),
        "models": [rbm_result],
    }

    if not successful_matching["success"]:
        save_pickle(output_path, results)
        print(
            "Matched-state summary: "
            f"best fidelity={results['matching']['best_fidelity']:.6f}, "
            f"threshold={args.match_fidelity_threshold:.6f}"
        )
        print(f"Saved partial failed-match results to {output_path}")
        if args.strict_match:
            raise RuntimeError(
                "Exact fidelity threshold was not reached; strict-match requested failure."
            )
        return

    vit_result = materialize_model_result(
        model=successful_matching["model"],
        model_name=successful_matching["requested_model_name"],
        model_desc=successful_matching["model_desc"],
        state_origin="vit_infidelity_checkpoint",
        variables=successful_matching["best_variables"],
        hamiltonian=hamiltonian,
        all_states=all_states,
        lambda_grid=lambda_grid,
        n_samples=args.ns,
        n_trials=args.n_trials,
        trial_seed=args.seed + 200_000,
        checkpoint_step=successful_matching["best_step"],
        selected_index=successful_matching["best_history_index"],
        selected_reason="best_infidelity",
        selection_metric="infidelity",
        selection_metric_value=successful_matching["best_infidelity"],
        checkpoint_summaries=successful_matching["history"],
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
    )
    if args.save_checkpoints:
        vit_checkpoint_path = checkpoint_path_for_model(
            checkpoint_dir,
            model_name=successful_matching["requested_model_name"],
            suffix="matched_best",
        )
        vit_checkpoint_payload = build_checkpoint_payload(
            system_meta=system_meta,
            model_name=successful_matching["requested_model_name"],
            model_desc=successful_matching["model_desc"],
            state_origin="vit_infidelity_checkpoint",
            step=successful_matching["best_step"],
            variables=successful_matching["best_variables"],
            result=vit_result,
            match_metrics={
                "fidelity": float(successful_matching["best_fidelity"]),
                "infidelity_direct": float(successful_matching["best_infidelity"]),
                "infidelity_operator": float(successful_matching["best_operator_infidelity"]),
            },
        )
        save_checkpoint_artifact(vit_checkpoint_path, vit_checkpoint_payload)
        checkpoint_artifacts["vit_checkpoint"] = str(vit_checkpoint_path)
        vit_result["checkpoint_artifact"] = str(vit_checkpoint_path)
    else:
        vit_result["checkpoint_artifact"] = None

    results["models"].append(vit_result)

    save_pickle(output_path, results)
    print(
        "Matched-state summary: "
        f"best fidelity={results['matching']['best_fidelity']:.6f}, "
        f"RBM noise_fraction={rbm_result['noise_fraction']:.3f}, "
        f"ViT noise_fraction={vit_result['noise_fraction']:.3f}"
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
