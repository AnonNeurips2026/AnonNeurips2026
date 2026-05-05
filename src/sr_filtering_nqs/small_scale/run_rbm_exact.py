from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE1_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sr_filtering_nqs.small_scale.common import make_tfim_chain, run_family_experiment


DEFAULT_ALPHAS = [1, 2, 4, 8]
DEFAULT_OUTPUT_TEMPLATE = FIGURE1_DIR / "results" / "rbm_exact_L{L}_ns{ns}.pkl"
SELECTION_MODES = ("independent_band", "independent_target", "paired_match")
SELECTION_METRICS = ("noise_fraction", "sigma_gap_sq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure 1 exact fixed-Ns RBM study on 1D TFIM.",
    )
    parser.add_argument("--L", type=int, default=16, help="TFIM chain length.")
    parser.add_argument("--h", type=float, default=1.0, help="Transverse field.")
    parser.add_argument("--J", type=float, default=1.0, help="Ising coupling.")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=int,
        default=DEFAULT_ALPHAS,
        help=f"RBM widths to run. Default: {DEFAULT_ALPHAS}",
    )
    parser.add_argument("--ns", type=int, default=2048, help="Fixed exact sample count.")
    parser.add_argument(
        "--train-samples",
        type=int,
        default=4096,
        help="MCState sample count used during checkpoint training.",
    )
    parser.add_argument("--train-steps", type=int, default=200, help="Maximum SR training steps.")
    parser.add_argument("--diag-every", type=int, default=10, help="Exact diagnostic interval.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="SR optimizer learning rate.")
    parser.add_argument(
        "--train-diag-shift",
        type=float,
        default=1e-3,
        help="Regularization used during checkpoint training.",
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
        help="Optional explicit lambda grid. Overrides --n-lambdas and exponent bounds.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default="independent_target",
        help=(
            "Checkpoint selection rule. `independent_target` picks each model near a target metric; "
            "`paired_match` jointly matches two models; `independent_band` preserves the older band rule."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="noise_fraction",
        help="Checkpoint metric used by the target or paired matcher.",
    )
    parser.add_argument(
        "--match-target-value",
        type=float,
        default=None,
        help=(
            "Optional target value for checkpoint matching. For `independent_target`, this overrides "
            "`--target-noise` when matching by `noise_fraction`."
        ),
    )
    parser.add_argument(
        "--match-min-value",
        type=float,
        default=None,
        help=(
            "Optional lower bound for paired matching. If omitted and matching by noise fraction, "
            "the shared logic prefers values >= 0.30."
        ),
    )
    parser.add_argument("--band-low", type=float, default=0.15, help="Noise-band lower bound.")
    parser.add_argument("--band-high", type=float, default=0.30, help="Noise-band upper bound.")
    parser.add_argument(
        "--target-noise",
        type=float,
        default=0.35,
        help="Default target noise fraction used by independent target selection.",
    )
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
        "--store-checkpoint-arrays",
        action="store_true",
        help="Store full exact arrays for every diagnostic checkpoint, not just the selected one.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle path. Defaults to figure1/results/rbm_exact_L{L}_ns{ns}.pkl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambda_grid = (
        np.asarray(args.lambda_values, dtype=np.float64)
        if args.lambda_values is not None
        else np.logspace(args.lambda_min_exp, args.lambda_max_exp, args.n_lambdas)
    )

    hilbert, hamiltonian, graph, system_meta = make_tfim_chain(L=args.L, h=args.h, J=args.J)
    model_specs = [f"RBM_a{alpha}" for alpha in args.alphas]
    output_path = (
        Path(args.output)
        if args.output
        else Path(str(DEFAULT_OUTPUT_TEMPLATE).format(L=args.L, ns=args.ns))
    )

    print("Figure 1 exact RBM study")
    print(f"System: {system_meta}")
    print(f"Models: {model_specs}")
    print(f"Fixed N_s: {args.ns}")
    print(f"Output: {output_path}")

    run_family_experiment(
        experiment_name="figure1_rbm_exact",
        output_path=output_path,
        system_meta=system_meta,
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        graph=graph,
        model_specs=model_specs,
        train_samples=args.train_samples,
        train_steps=args.train_steps,
        diag_every=args.diag_every,
        learning_rate=args.learning_rate,
        train_diag_shift=args.train_diag_shift,
        n_samples=args.ns,
        n_trials=args.n_trials,
        lambda_grid=lambda_grid,
        seed=args.seed,
        selection_mode=args.selection_mode,
        selection_metric=args.selection_metric,
        match_target_value=args.match_target_value,
        match_min_value=args.match_min_value,
        band_low=args.band_low,
        band_high=args.band_high,
        target_noise=args.target_noise,
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
        store_checkpoint_arrays=args.store_checkpoint_arrays,
    )

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
