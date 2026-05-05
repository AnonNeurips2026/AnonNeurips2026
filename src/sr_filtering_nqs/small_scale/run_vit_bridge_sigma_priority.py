from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE1_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sr_filtering_nqs.small_scale.common import make_square_j1j2, run_family_experiment, save_pickle
from sr_filtering_nqs.small_scale.plot_figure1 import plot_main_figure


DEFAULT_MODELS = ["RBM_a2", "ViT_d2_m24_h4_e4"]
DEFAULT_OUTPUT = FIGURE1_DIR / "results" / "j1j2_rbm_a2_vit_sigma_priority.pkl"
DEFAULT_FIGURES_DIR = FIGURE1_DIR / "figures_j1j2_rbm_a2_vit_sigma_priority"
DEFAULT_CHECKPOINT_ROOT = FIGURE1_DIR / "checkpoints" / "j1j2_rbm_a2_vit_sigma_priority"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain RBM_a2 and ViT on the exact 4x4 J1-J2 bridge, then select checkpoint pairs "
            "by prioritizing sigma_gap_sq matching and using noise_fraction as a secondary tie-break."
        ),
    )
    parser.add_argument("--L", type=int, default=4, help="Square lattice linear size.")
    parser.add_argument("--J1", type=float, default=1.0, help="Nearest-neighbor coupling.")
    parser.add_argument("--J2", type=float, default=0.5, help="Next-nearest-neighbor coupling.")
    parser.add_argument(
        "--models",
        nargs=2,
        default=DEFAULT_MODELS,
        help=f"Exactly two models, default: {DEFAULT_MODELS}",
    )
    parser.add_argument(
        "--ns",
        type=int,
        default=4096,
        help="Exact diagnostic sample count. Kept equal to train-samples by default.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=4096,
        help="MCState samples used during training.",
    )
    parser.add_argument("--train-steps", type=int, default=1000, help="Maximum SR training steps.")
    parser.add_argument("--diag-every", type=int, default=10, help="Exact diagnostic interval.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="SR optimizer learning rate.")
    parser.add_argument("--train-diag-shift", type=float, default=1e-3, help="Training regularization.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of exact Born trials.")
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=[1e-9, 1e-1],
        help="Explicit diagnostic lambda values. Default: 1e-9 1e-1",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--match-min-value",
        type=float,
        default=None,
        help="Optional lower bound on sigma_gap_sq when searching for matched checkpoint pairs.",
    )
    parser.add_argument(
        "--match-target-value",
        type=float,
        default=None,
        help="Optional target sigma_gap_sq around which to search for matched pairs.",
    )
    parser.add_argument("--jac-chunk-size", type=int, default=4096, help="Chunk size for Jacobian evaluations.")
    parser.add_argument("--apply-chunk-size", type=int, default=4096, help="Chunk size for forward/JVP evaluations.")
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help=f"Directory used to store resumable per-step training checkpoints. Default: {DEFAULT_CHECKPOINT_ROOT}",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any saved training checkpoints and restart from step 0.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output pickle path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(DEFAULT_FIGURES_DIR),
        help=f"Directory where figure1_main.pdf will be written. Default: {DEFAULT_FIGURES_DIR}",
    )
    return parser.parse_args()


def attach_pair_steps(results: dict) -> None:
    pair_metadata = results.get("selection", {}).get("pair_metadata")
    if not pair_metadata:
        return

    checkpoint_steps_by_model = []
    for model in results["models"]:
        checkpoint_steps_by_model.append(
            [int(summary["step"]) for summary in model["checkpoint_summaries"]]
        )

    def indices_to_steps(indices: list[int]) -> list[int]:
        return [
            checkpoint_steps_by_model[model_idx][int(checkpoint_idx)]
            for model_idx, checkpoint_idx in enumerate(indices)
        ]

    pair_metadata["selected_steps"] = indices_to_steps(pair_metadata["selected_indices"])
    pair_metadata["top_candidates_with_steps"] = [
        {
            **candidate,
            "steps": indices_to_steps(candidate["indices"]),
        }
        for candidate in pair_metadata.get("top_candidates", [])
    ]


def main() -> None:
    args = parse_args()
    lambda_grid = np.asarray(args.lambda_values, dtype=np.float64)
    output_path = Path(args.output)
    figures_dir = Path(args.figures_dir)
    checkpoint_root = Path(args.checkpoint_root)

    hilbert, hamiltonian, graph, system_meta = make_square_j1j2(L=args.L, J1=args.J1, J2=args.J2)

    print("Figure 1 sigma-priority bridge")
    print(f"System: {system_meta}")
    print(f"Models: {args.models}")
    print(f"Diagnostic lambdas: {lambda_grid.tolist()}")
    print(f"Output pickle: {output_path}")
    print(f"Output figures: {figures_dir}")
    print(f"Training checkpoints: {checkpoint_root}")
    print(f"Resume enabled: {not args.no_resume}")

    results = run_family_experiment(
        experiment_name="figure1_vit_bridge_sigma_priority",
        output_path=output_path,
        system_meta=system_meta,
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        graph=graph,
        model_specs=list(args.models),
        train_samples=args.train_samples,
        train_steps=args.train_steps,
        diag_every=args.diag_every,
        learning_rate=args.learning_rate,
        train_diag_shift=args.train_diag_shift,
        n_samples=args.ns,
        n_trials=args.n_trials,
        lambda_grid=lambda_grid,
        seed=args.seed,
        selection_mode="paired_match_sigma_priority",
        selection_metric="sigma_gap_sq",
        match_target_value=args.match_target_value,
        match_min_value=args.match_min_value,
        band_low=0.15,
        band_high=0.30,
        target_noise=0.35,
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
        store_checkpoint_arrays=False,
        training_checkpoint_root=checkpoint_root,
        resume_training=not args.no_resume,
        square_L=args.L,
        j_couplings=(args.J1, args.J2),
    )

    attach_pair_steps(results)
    save_pickle(output_path, results)
    plot_main_figure(results, figures_dir / "figure1_main.pdf")

    print("\nSelected checkpoints")
    for model in results["models"]:
        print(
            f"  {model['model_name']} step {model['checkpoint_step']}: "
            f"sigma_gap_sq={model['sigma_gap_sq']:.6f}, "
            f"noise_fraction={model['noise_fraction']:.6f}, "
            f"P/Ns={model['P_over_Ns']:.6f}"
        )

    pair_metadata = results["selection"]["pair_metadata"]
    print("\nBest pair metadata")
    print(
        "  "
        f"sigma_gap_sq abs diff={pair_metadata['selected_sigma_gap_sq_abs_diff']:.6e}, "
        f"rel diff={pair_metadata['selected_sigma_gap_sq_rel_diff']:.6e}"
    )
    print(
        "  "
        f"noise_fraction abs diff={pair_metadata['selected_noise_fraction_abs_diff']:.6e}, "
        f"rel diff={pair_metadata['selected_noise_fraction_rel_diff']:.6e}"
    )
    print(f"Saved figure to {figures_dir / 'figure1_main.pdf'}")


if __name__ == "__main__":
    main()
