"""
Shared-checkpoint Figure 2 diagnostics over solve-time lambda sweeps.

Examples (run from ``nqs_support_core/``):

    uv run python ../large_scale/cli/shared_checkpoint_diagnostics.py --system j1j2
    uv run python ../large_scale/cli/shared_checkpoint_diagnostics.py --system tfim --source-steps 1100,1600,2000
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
    heldout_samples_for_system,
    model_type_for_system,
    normalize_system_name,
)
from sr_filtering_nqs.large_scale.core.diagnostic_core import (
    process_fnqs_checkpoint_metrics,
    process_vit_checkpoint_metrics,
    precompute_fnqs_checkpoint_delta_bank,
    precompute_vit_checkpoint_delta_bank,
)
from sr_filtering_nqs.large_scale.core.shared_checkpoint_protocol import (
    DEFAULT_SHARED_FIG2_DIAGNOSTIC_LAMBDAS,
    checkpoint_step_from_path,
    discover_source_checkpoints,
    parse_lambda_values,
    shared_fig2_delta_bank_path,
    shared_fig2_metric_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run shared-checkpoint Figure 2 diagnostics from standard source checkpoints"
    )
    parser.add_argument("--system", required=True, choices=ACTIVE_SYSTEM_CHOICES)
    parser.add_argument("--metric", default="all", choices=["all", "bank", "rval", "twobatch"])
    parser.add_argument("--input-dir", default=str(ROOT))
    parser.add_argument("--source-train-lambda", type=float, default=SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA)
    parser.add_argument(
        "--source-steps",
        type=str,
        default="all",
        help="Comma-separated source checkpoint steps or 'all' for the canonical diagnostic steps.",
    )
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument(
        "--diag-lambdas",
        type=str,
        default=None,
        help="Comma-separated solve-time lambdas; defaults to the shared Figure 2 grid.",
    )
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--protocol-id", type=str, default="shared_fig2_single")
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--validation-round-samples", type=int, default=None)
    parser.add_argument("--jvp-chunk", type=int, default=None)
    parser.add_argument("--conn-chunk", type=int, default=8192)
    parser.add_argument("--eval-chunk", type=int, default=8192)
    parser.add_argument("--chunk-size-bwd", type=int, default=None)
    parser.add_argument("--delta-batch-size", type=int, default=10)
    return parser.parse_args()


def _effective_n_val(args, system_name: str) -> int:
    if args.n_val is not None:
        return int(args.n_val)
    return int(heldout_samples_for_system(system_name))


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _log_delta_bank(result) -> None:
    checkpoint = Path(result["checkpoint"])
    status = "up-to-date" if not result["computed_delta_bank"] else "computed delta bank"
    print(f"    {checkpoint.name}: {status}", flush=True)


def _shared_metadata(
    ckpt_path,
    *,
    protocol_id: str,
    source_training_lambda: float,
    source_step: int,
    diagnostic_lambda: float,
    repeat_count: int,
) -> dict[str, object]:
    return {
        "protocol_id": str(protocol_id),
        "source_training_lambda": float(source_training_lambda),
        "source_step": int(source_step),
        "source_checkpoint": str(Path(ckpt_path).resolve()),
        "diagnostic_lambda": float(diagnostic_lambda),
        "repeat_count": int(repeat_count),
    }


def _ensure_delta_bank(
    ckpt_path,
    *,
    system_name: str,
    repeat_count: int,
    diagnostic_lambda: float,
    chunk_size_bwd: int | None,
    bank_path: Path,
    metadata: dict[str, object],
):
    if model_type_for_system(system_name) == "fnqs":
        result = precompute_fnqs_checkpoint_delta_bank(
            ckpt_path,
            m_deltas=repeat_count,
            chunk_size_bwd=chunk_size_bwd,
            diag_shift_override=diagnostic_lambda,
            output_path=bank_path,
            metadata=metadata,
        )
    else:
        result = precompute_vit_checkpoint_delta_bank(
            ckpt_path,
            m_deltas=repeat_count,
            chunk_size_bwd=chunk_size_bwd,
            diag_shift_override=diagnostic_lambda,
            output_path=bank_path,
            metadata=metadata,
        )
    _log_delta_bank(result)
    return _load_pickle(bank_path)


def main():
    args = parse_args()
    system_name = normalize_system_name(args.system)
    n_val = _effective_n_val(args, system_name)
    diag_lambdas = parse_lambda_values(args.diag_lambdas, default=DEFAULT_SHARED_FIG2_DIAGNOSTIC_LAMBDAS)
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
        "Diagnostic solve lambdas: "
        + ", ".join(f"{lam:.3e}" for lam in diag_lambdas),
        flush=True,
    )

    want_bank = args.metric in ("all", "bank", "rval", "twobatch")
    want_rval = args.metric in ("all", "rval")
    want_twobatch = args.metric in ("all", "twobatch")

    for source_step, ckpt_path in checkpoints:
        source_step = int(source_step)
        if source_step != checkpoint_step_from_path(ckpt_path):
            raise RuntimeError(f"Checkpoint step mismatch for {ckpt_path}")
        print(f"\nCheckpoint step={source_step} path={ckpt_path.name}", flush=True)
        for diagnostic_lambda in diag_lambdas:
            print(f"  solve_lambda={diagnostic_lambda:.3e}", flush=True)
            metadata = _shared_metadata(
                ckpt_path,
                protocol_id=args.protocol_id,
                source_training_lambda=float(args.source_train_lambda),
                source_step=source_step,
                diagnostic_lambda=diagnostic_lambda,
                repeat_count=int(args.repeat_count),
            )
            bank_path = shared_fig2_delta_bank_path(
                ckpt_path,
                protocol_id=args.protocol_id,
                source_training_lambda=float(args.source_train_lambda),
                source_step=source_step,
                diagnostic_lambda=diagnostic_lambda,
                repeat_count=int(args.repeat_count),
            )
            bank_payload = None
            if want_bank:
                bank_payload = _ensure_delta_bank(
                    ckpt_path,
                    system_name=system_name,
                    repeat_count=int(args.repeat_count),
                    diagnostic_lambda=float(diagnostic_lambda),
                    chunk_size_bwd=args.chunk_size_bwd,
                    bank_path=bank_path,
                    metadata=metadata,
                )

            if not want_rval and not want_twobatch:
                continue

            output_paths = {
                "rval": shared_fig2_metric_path(
                    ckpt_path,
                    protocol_id=args.protocol_id,
                    source_training_lambda=float(args.source_train_lambda),
                    source_step=source_step,
                    diagnostic_lambda=diagnostic_lambda,
                    repeat_count=int(args.repeat_count),
                    metric="rval",
                ),
                "twobatch": shared_fig2_metric_path(
                    ckpt_path,
                    protocol_id=args.protocol_id,
                    source_training_lambda=float(args.source_train_lambda),
                    source_step=source_step,
                    diagnostic_lambda=diagnostic_lambda,
                    repeat_count=int(args.repeat_count),
                    metric="twobatch",
                ),
            }

            if model_type_for_system(system_name) == "fnqs":
                process_fnqs_checkpoint_metrics(
                    ckpt_path,
                    n_val=n_val,
                    validation_round_samples=args.validation_round_samples,
                    jvp_chunk=args.jvp_chunk,
                    chunk_size_bwd=args.chunk_size_bwd,
                    delta_batch_size=args.delta_batch_size,
                    m_deltas=args.repeat_count,
                    want_rval=want_rval,
                    want_twobatch=want_twobatch,
                    diag_shift_override=diagnostic_lambda,
                    bank_payload=bank_payload,
                    output_paths=output_paths,
                    result_metadata=metadata,
                )
            else:
                process_vit_checkpoint_metrics(
                    ckpt_path,
                    n_val=n_val,
                    validation_round_samples=args.validation_round_samples,
                    jvp_chunk=args.jvp_chunk,
                    conn_chunk=args.conn_chunk,
                    eval_chunk=args.eval_chunk,
                    chunk_size_bwd=args.chunk_size_bwd,
                    delta_batch_size=args.delta_batch_size,
                    m_deltas=args.repeat_count,
                    want_rval=want_rval,
                    want_twobatch=want_twobatch,
                    diag_shift_override=diagnostic_lambda,
                    bank_payload=bank_payload,
                    output_paths=output_paths,
                    result_metadata=metadata,
                )


if __name__ == "__main__":
    main()
