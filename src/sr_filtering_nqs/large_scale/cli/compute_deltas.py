"""
Precompute offline multi-delta banks for Section 4 diagnostics.

Examples (run from ``nqs_support_core/``):

    uv run python ../large_scale/cli/compute_deltas.py compute --system j1j2 --m-deltas 100
    uv run python ../large_scale/cli/compute_deltas.py compute --system tfim --diag-shift 1e-3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.cli.diagnostics import discover_metric_checkpoints
from sr_filtering_nqs.large_scale.core.common import ACTIVE_SYSTEM_CHOICES, model_type_for_system, normalize_system_name
from sr_filtering_nqs.large_scale.core.diagnostic_core import (
    precompute_fnqs_checkpoint_delta_bank,
    precompute_vit_checkpoint_delta_bank,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute offline diagnostic delta banks")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compute = subparsers.add_parser("compute", help="Precompute delta banks from checkpoints")
    compute.add_argument("--system", required=True, choices=ACTIVE_SYSTEM_CHOICES)
    compute.add_argument(
        "--metric",
        default="all",
        choices=["all", "rval", "twobatch"],
        help="Compatibility option; all metrics use the same delta bank.",
    )
    compute.add_argument("--input-dir", default=str(ROOT))
    compute.add_argument("--diag-shift", type=float, default=None)
    compute.add_argument("--min-step", type=int, default=None)
    compute.add_argument("--latest-only", action="store_true")
    compute.add_argument("--chunk-size-bwd", type=int, default=None)
    compute.add_argument("--m-deltas", type=int, default=100)

    return parser.parse_args()


def _log_result(result) -> None:
    checkpoint = Path(result["checkpoint"])
    status = "up-to-date" if not result["computed_delta_bank"] else "computed delta_bank"
    chunk_size_bwd = result.get("chunk_size_bwd")
    chunk_suffix = "" if chunk_size_bwd is None else f" (chunk_size_bwd={int(chunk_size_bwd)})"
    print(f"  {checkpoint.name}: {status}{chunk_suffix}")


def compute(args):
    system_name = normalize_system_name(args.system)
    checkpoints = discover_metric_checkpoints(
        system_name,
        args.input_dir,
        diag_shift=args.diag_shift,
        min_step=args.min_step,
        latest_only=args.latest_only,
        metric="all",
    )
    model_name = "FNQS" if model_type_for_system(system_name) == "fnqs" else "ViT"
    print(
        f"Found {len(checkpoints)} {model_name} checkpoints for delta-bank precompute "
        f"(m={int(args.m_deltas)})"
    )

    if model_type_for_system(system_name) == "fnqs":
        for checkpoint in checkpoints:
            result = precompute_fnqs_checkpoint_delta_bank(
                checkpoint,
                m_deltas=args.m_deltas,
                chunk_size_bwd=args.chunk_size_bwd,
            )
            _log_result(result)
        return

    for checkpoint in checkpoints:
        result = precompute_vit_checkpoint_delta_bank(
            checkpoint,
            m_deltas=args.m_deltas,
            chunk_size_bwd=args.chunk_size_bwd,
        )
        _log_result(result)


def main():
    args = parse_args()
    if args.command == "compute":
        compute(args)


if __name__ == "__main__":
    main()
