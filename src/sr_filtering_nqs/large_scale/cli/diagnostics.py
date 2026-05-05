"""
Canonical validation-risk diagnostics entrypoint for the large-scale runs.

Examples:

    python -m sr_filtering_nqs.large_scale.cli.diagnostics --system j1j2
    python -m sr_filtering_nqs.large_scale.cli.diagnostics --system tfim
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
    J1J2_DIAGNOSTIC_STEPS,
    model_type_for_system,
    normalize_system_name,
)
from sr_filtering_nqs.large_scale.core.compute_rval import discover_fnqs_checkpoints, discover_vit_checkpoints
from sr_filtering_nqs.large_scale.core.diagnostic_core import (
    process_fnqs_checkpoint_metrics,
    process_vit_checkpoint_metrics,
)
from sr_filtering_nqs.large_scale.core.tfim_fnqs import TFIM_DIAGNOSTIC_STEPS


def parse_args():
    parser = argparse.ArgumentParser(description="Canonical validation-risk diagnostics")
    parser.add_argument("--system", required=True, choices=ACTIVE_SYSTEM_CHOICES)
    parser.add_argument("--metric", default="rval", choices=["all", "rval"])
    parser.add_argument("--input-dir", default=str(ROOT))
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        help="Compatibility option from the old raw-run workflow; ignored.",
    )
    parser.add_argument("--diag-shift", type=float, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument("--validation-round-samples", type=int, default=None)
    parser.add_argument("--jvp-chunk", type=int, default=None)
    parser.add_argument("--conn-chunk", type=int, default=8192)
    parser.add_argument("--eval-chunk", type=int, default=8192)
    parser.add_argument("--chunk-size-bwd", type=int, default=None)
    parser.add_argument("--delta-batch-size", type=int, default=10)
    parser.add_argument("--m-deltas", type=int, default=100)
    parser.add_argument(
        "--prep-cache",
        dest="prep_cache",
        action="store_true",
        help="Compatibility no-op; diagnostics are delta-bank cache-only.",
    )
    parser.add_argument(
        "--no-prep-cache",
        dest="prep_cache",
        action="store_false",
        help="Compatibility no-op; diagnostics are delta-bank cache-only.",
    )
    parser.set_defaults(prep_cache=True)
    return parser.parse_args()


def _effective_n_val(args, system_name: str) -> int:
    if args.n_val is not None:
        return int(args.n_val)
    return int(heldout_samples_for_system(system_name))


def effective_min_step(system_name: str, min_step: int | None = None) -> int | None:
    system_name = normalize_system_name(system_name)
    if min_step is not None:
        return int(min_step)
    if system_name == "j1j2":
        return int(J1J2_DIAGNOSTIC_STEPS[0])
    if system_name == "tfim":
        return int(TFIM_DIAGNOSTIC_STEPS[0])
    return None


def discover_metric_checkpoints(
    system_name: str,
    input_dir,
    *,
    diag_shift: float | None = None,
    min_step: int | None = None,
    latest_only: bool = False,
    metric: str = "all",
):
    system_name = normalize_system_name(system_name)
    if metric not in ("all", "rval"):
        raise ValueError(f"Unsupported diagnostic metric: {metric}")
    step_stride = None if latest_only else 1
    min_step = effective_min_step(system_name, min_step)

    if model_type_for_system(system_name) == "fnqs":
        return list(
            discover_fnqs_checkpoints(
                input_dir,
                system_name,
                step_stride=step_stride,
                diag_shift=diag_shift,
                min_step=min_step,
            )
        )
    return list(
        discover_vit_checkpoints(
            input_dir,
            system_name,
            step_stride=step_stride,
            diag_shift=diag_shift,
            min_step=min_step,
        )
    )


def _log_checkpoint_count(system_name: str, metric: str, checkpoints) -> None:
    model_name = "FNQS" if model_type_for_system(system_name) == "fnqs" else "ViT"
    print(f"Found {len(checkpoints)} {model_name} checkpoints for {metric}")


def _process_checkpoints(args):
    system_name = normalize_system_name(args.system)
    n_val = _effective_n_val(args, system_name)
    checkpoints = discover_metric_checkpoints(
        system_name,
        args.input_dir,
        diag_shift=args.diag_shift,
        min_step=args.min_step,
        latest_only=args.latest_only,
        metric=args.metric,
    )
    _log_checkpoint_count(system_name, "rval", checkpoints)

    if model_type_for_system(system_name) == "fnqs":
        for checkpoint in checkpoints:
            process_fnqs_checkpoint_metrics(
                checkpoint,
                n_val=n_val,
                validation_round_samples=args.validation_round_samples,
                jvp_chunk=args.jvp_chunk,
                chunk_size_bwd=args.chunk_size_bwd,
                delta_batch_size=args.delta_batch_size,
                m_deltas=args.m_deltas,
                use_prep_cache=args.prep_cache,
                want_rval=True,
                want_twobatch=False,
            )
        return

    for checkpoint in checkpoints:
        process_vit_checkpoint_metrics(
            checkpoint,
            n_val=n_val,
            validation_round_samples=args.validation_round_samples,
            jvp_chunk=args.jvp_chunk,
            conn_chunk=args.conn_chunk,
            eval_chunk=args.eval_chunk,
            chunk_size_bwd=args.chunk_size_bwd,
            delta_batch_size=args.delta_batch_size,
            m_deltas=args.m_deltas,
            use_prep_cache=args.prep_cache,
            want_rval=True,
            want_twobatch=False,
        )


def main():
    args = parse_args()
    _ = args.run_index
    _process_checkpoints(args)


if __name__ == "__main__":
    main()
