"""
Shared-checkpoint Figure 3 ablations on standard source checkpoints.

Examples (run from ``nqs_support_core/``):

    uv run python ../large_scale/cli/shared_checkpoint_mixture.py --system j1j2
    uv run python ../large_scale/cli/shared_checkpoint_mixture.py --system tfim --k 5 --methods single_oracle,indep_single_lambda_k5,indep_multilambda_gcv_k5
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core.common import (
    ACTIVE_SYSTEM_CHOICES,
    SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA,
    canonical_shared_fig3_lambda_grid,
    heldout_samples_for_system,
    lambda_grid_tag,
    normalize_system_name,
    shared_fig3_default_method_ids,
)
from sr_filtering_nqs.large_scale.core.multilambda_diagnostics import (
    evaluate_mixture_bank,
    parse_mixture_method_id,
    precompute_mixture_bank,
    validate_mixture_method_ids,
)
from sr_filtering_nqs.large_scale.core.shared_checkpoint_protocol import (
    checkpoint_step_from_path,
    discover_source_checkpoints,
    repeat_count_tag,
    shared_fig3_bank_path,
    shared_fig3_eval_path,
    shared_protocol_step_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run shared-checkpoint Figure 3 ablations from standard source checkpoints"
    )
    parser.add_argument("--system", required=True, choices=ACTIVE_SYSTEM_CHOICES)
    parser.add_argument("--metric", default="all", choices=["all", "bank", "eval"])
    parser.add_argument("--input-dir", default=str(ROOT))
    parser.add_argument("--source-train-lambda", type=float, default=SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA)
    parser.add_argument(
        "--source-steps",
        type=str,
        default="all",
        help="Comma-separated source checkpoint steps or 'all' for the canonical diagnostic steps.",
    )
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--lambda-grid",
        type=str,
        default=None,
        help="Optional comma-separated override for the Figure 3 lambda grid.",
    )
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--protocol-id", type=str, default=None)
    parser.add_argument("--oracle-protocol-id", type=str, default="shared_fig2_single")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated method ids; defaults to the main-paper shared Figure 3 set for the chosen k.",
    )
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--weight-fit-samples", type=int, default=None)
    parser.add_argument("--validation-round-samples", type=int, default=None)
    parser.add_argument("--jvp-chunk", type=int, default=None)
    return parser.parse_args()


def _selected_methods(raw: str | None, *, k: int, lambda_grid) -> tuple[str, ...]:
    if raw is None:
        methods = shared_fig3_default_method_ids(k)
    else:
        methods = tuple(part.strip() for part in raw.split(",") if part.strip())
        if not methods:
            raise ValueError("At least one Figure 3 method id must be provided")
    return validate_mixture_method_ids(methods, lambda_grid=lambda_grid, single_lambda_count=k)


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _effective_n_val(args, system_name: str) -> int:
    if args.n_val is not None:
        return int(args.n_val)
    return int(heldout_samples_for_system(system_name))


def _oracle_lambda_for_checkpoint(
    ckpt_path,
    *,
    oracle_protocol_id: str,
    source_training_lambda: float,
    source_step: int,
    repeat_count: int,
) -> float:
    step_dir = shared_protocol_step_dir(
        ckpt_path,
        protocol_id=oracle_protocol_id,
        source_training_lambda=float(source_training_lambda),
        source_step=int(source_step),
    )
    if not step_dir.exists():
        raise FileNotFoundError(
            f"Missing Figure 2 shared outputs for step={source_step} under protocol={oracle_protocol_id}"
        )
    payloads = []
    for path in sorted(step_dir.glob(f"rval__diag_*__{repeat_count_tag(repeat_count)}.pkl")):
        payload = _load_pickle(path)
        payloads.append((float(payload["r_val"]), float(payload["lambda"]), path))
    if not payloads:
        raise FileNotFoundError(
            "No shared Figure 2 residual payloads found for "
            f"step={source_step} repeat_count={repeat_count} under protocol={oracle_protocol_id}"
        )
    payloads.sort(key=lambda row: (row[0], row[1]))
    return float(payloads[0][1])


def _shared_metadata(
    ckpt_path,
    *,
    protocol_id: str,
    source_training_lambda: float,
    source_step: int,
    repeat_count: int,
    lambda_grid,
    oracle_lambda: float,
    method_ids: tuple[str, ...],
) -> dict[str, object]:
    return {
        "protocol_id": str(protocol_id),
        "source_training_lambda": float(source_training_lambda),
        "source_step": int(source_step),
        "source_checkpoint": str(Path(ckpt_path).resolve()),
        "repeat_count": int(repeat_count),
        "lambda_grid_tag": lambda_grid_tag(lambda_grid),
        "oracle_lambda": float(oracle_lambda),
        "method_ids": list(method_ids),
        "k": int(len(tuple(lambda_grid))),
    }


def main():
    args = parse_args()
    system_name = normalize_system_name(args.system)
    protocol_id = args.protocol_id or f"shared_fig3_k{int(args.k)}"
    lambda_grid = (
        canonical_shared_fig3_lambda_grid(args.k)
        if args.lambda_grid is None
        else tuple(float(part) for part in args.lambda_grid.split(",") if part.strip())
    )
    if len(tuple(lambda_grid)) != int(args.k):
        raise ValueError(f"Expected a lambda grid of length k={args.k}, got {lambda_grid}")
    method_ids = _selected_methods(args.methods, k=int(args.k), lambda_grid=lambda_grid)
    n_val = _effective_n_val(args, system_name)
    checkpoints = discover_source_checkpoints(
        args.input_dir,
        system_name,
        source_training_lambda=float(args.source_train_lambda),
        source_steps=args.source_steps,
        latest_only=args.latest_only,
    )
    if not checkpoints:
        raise SystemExit(
            f"No shared source checkpoints found for {system_name} at lambda_train={args.source_train_lambda:.3e}"
        )

    print(
        f"Found {len(checkpoints)} shared source checkpoints for {system_name} "
        f"at lambda_train={args.source_train_lambda:.3e}",
        flush=True,
    )
    print(
        "Figure 3 methods: " + ", ".join(method_ids),
        flush=True,
    )
    print(
        "Lambda grid: " + ", ".join(f"{lam:.3e}" for lam in lambda_grid),
        flush=True,
    )

    for source_step, ckpt_path in checkpoints:
        source_step = int(source_step)
        if source_step != checkpoint_step_from_path(ckpt_path):
            raise RuntimeError(f"Checkpoint step mismatch for {ckpt_path}")
        oracle_lambda = _oracle_lambda_for_checkpoint(
            ckpt_path,
            oracle_protocol_id=args.oracle_protocol_id,
            source_training_lambda=float(args.source_train_lambda),
            source_step=source_step,
            repeat_count=int(args.repeat_count),
        )
        metadata = _shared_metadata(
            ckpt_path,
            protocol_id=protocol_id,
            source_training_lambda=float(args.source_train_lambda),
            source_step=source_step,
            repeat_count=int(args.repeat_count),
            lambda_grid=lambda_grid,
            oracle_lambda=oracle_lambda,
            method_ids=method_ids,
        )
        bank_path = shared_fig3_bank_path(
            ckpt_path,
            protocol_id=protocol_id,
            source_training_lambda=float(args.source_train_lambda),
            source_step=source_step,
            lambda_grid=lambda_grid,
            repeat_count=int(args.repeat_count),
            oracle_lambda=oracle_lambda,
        )
        eval_path = shared_fig3_eval_path(
            ckpt_path,
            protocol_id=protocol_id,
            source_training_lambda=float(args.source_train_lambda),
            source_step=source_step,
            lambda_grid=lambda_grid,
            repeat_count=int(args.repeat_count),
            oracle_lambda=oracle_lambda,
        )

        print(
            f"\nCheckpoint step={source_step} oracle_lambda={oracle_lambda:.3e} path={ckpt_path.name}",
            flush=True,
        )
        if args.metric in ("all", "bank"):
            precompute_mixture_bank(
                ckpt_path,
                system_name=system_name,
                repeat_count=int(args.repeat_count),
                lambda_grid=lambda_grid,
                method_ids=method_ids,
                oracle_lambda=oracle_lambda,
                single_lambda_count=int(args.k),
                output_path=bank_path,
                metadata=metadata,
            )
        if args.metric in ("all", "eval"):
            evaluate_mixture_bank(
                ckpt_path,
                system_name=system_name,
                repeat_count=int(args.repeat_count),
                lambda_grid=lambda_grid,
                n_val=n_val,
                weight_fit_samples=args.weight_fit_samples,
                validation_round_samples=args.validation_round_samples,
                jvp_chunk=args.jvp_chunk,
                method_ids=method_ids,
                oracle_lambda=oracle_lambda,
                single_lambda_count=int(args.k),
                bank_output_path=bank_path,
                output_path=eval_path,
                metadata=metadata,
            )


if __name__ == "__main__":
    main()
