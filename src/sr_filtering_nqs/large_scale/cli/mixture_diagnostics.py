"""
Checkpoint-local diagnostics for multi-lambda SR runs.

Examples (run from ``nqs_support_core/``):

    uv run python ../large_scale/cli/mixture_diagnostics.py --system j1j2 --metric all
    uv run python ../large_scale/cli/mixture_diagnostics.py --system tfim --metric eval --latest-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core.common import (
    ACTIVE_SYSTEM_CHOICES,
    heldout_samples_for_system,
    normalize_system_name,
)
from sr_filtering_nqs.large_scale.core.multilambda_diagnostics import (
    discover_mixture_checkpoints,
    evaluate_mixture_bank,
    precompute_mixture_bank,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint-local diagnostics for multi-lambda SR runs")
    parser.add_argument("--system", required=True, choices=ACTIVE_SYSTEM_CHOICES)
    parser.add_argument("--metric", default="all", choices=["all", "bank", "eval"])
    parser.add_argument("--input-dir", default=str(ROOT))
    parser.add_argument("--sr-variant", default=None, choices=["same_batch", "indep_batch"])
    parser.add_argument("--sr-weight-scheme", default=None, choices=["uniform", "gcv", "stacking"])
    parser.add_argument(
        "--sr-lambda-grid",
        type=str,
        default=None,
        help="Optional comma-separated lambda grid tag filter.",
    )
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--weight-fit-samples", type=int, default=None)
    parser.add_argument("--validation-round-samples", type=int, default=None)
    parser.add_argument("--jvp-chunk", type=int, default=None)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--latest-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    system_name = normalize_system_name(args.system)
    n_val = int(heldout_samples_for_system(system_name) if args.n_val is None else args.n_val)

    checkpoints = discover_mixture_checkpoints(
        system_name,
        args.input_dir,
        sr_variant=args.sr_variant,
        sr_weight_scheme=args.sr_weight_scheme,
        lambda_grid=args.sr_lambda_grid,
        min_step=args.min_step,
        latest_only=args.latest_only,
    )
    if not checkpoints:
        raise SystemExit(f"No mixture checkpoints found for {system_name} under {args.input_dir}")

    print(f"Found {len(checkpoints)} mixture checkpoints for {system_name}", flush=True)
    for variant, weight_scheme, grid_tag, step, ckpt_path in checkpoints:
        print(
            f"  run={variant}__{weight_scheme}__{grid_tag} step={step} path={ckpt_path.name}",
            flush=True,
        )
        if args.metric in ("all", "bank"):
            precompute_mixture_bank(
                ckpt_path,
                system_name=system_name,
                repeat_count=args.repeat_count,
                lambda_grid=args.sr_lambda_grid,
            )
        if args.metric in ("all", "eval"):
            evaluate_mixture_bank(
                ckpt_path,
                system_name=system_name,
                repeat_count=args.repeat_count,
                lambda_grid=args.sr_lambda_grid,
                n_val=n_val,
                weight_fit_samples=args.weight_fit_samples,
                validation_round_samples=args.validation_round_samples,
                jvp_chunk=args.jvp_chunk,
            )


if __name__ == "__main__":
    main()
