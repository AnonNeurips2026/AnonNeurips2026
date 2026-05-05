from __future__ import annotations

import argparse
import csv
import shutil
import sys
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core.common import (
    CHECKPOINT_STEPS,
    PHASE_SAMPLES,
    PHASE0_CHECKPOINT_STEPS,
    PHASE_LENGTHS,
    TRAIN_SAMPLES,
    VIT_D_MODEL,
    VIT_DEPTH,
    VIT_EXPANSION,
    VIT_HEADS,
    lambda_grid_for_system,
    normalize_system_name,
    results_dir_for_run,
)
from sr_filtering_nqs.large_scale.cli.train_sweep import run_single


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the branching large-scale J1J2 sweep: Phase 0 once at lambda=1e-3, then sweep lambdas for Phases 1 and 2."
    )
    parser.add_argument("--system", required=True, choices=["j1j2"])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--diag-shift", type=float, help="Run a single phase-1/2 lambda branch")
    group.add_argument("--sweep", action="store_true", help="Run the full lambda grid for phases 1 and 2")
    parser.add_argument("--base-diag-shift", type=float, default=1e-3, help="Phase 0 lambda")
    parser.add_argument("--output-root", type=str, default=str(ROOT / "large_scale"))
    parser.add_argument(
        "--base-output-root",
        type=str,
        default=str(ROOT / "large_scale" / "phase0_base"),
        help="Directory root where the shared Phase 0 run is stored",
    )
    parser.add_argument("--n-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument(
        "--phase-samples",
        type=str,
        default=",".join(str(x) for x in PHASE_SAMPLES),
        help="Comma-separated sample counts for phases 0,1,2.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--latest-checkpoint-every",
        type=int,
        default=None,
        help="Overwrite checkpoints/checkpoint_latest.pkl every N steps. Defaults to 25 for λ<=1e-7, disabled otherwise.",
    )
    parser.add_argument("--depth", type=int, default=VIT_DEPTH)
    parser.add_argument("--d-model", type=int, default=VIT_D_MODEL)
    parser.add_argument("--heads", type=int, default=VIT_HEADS)
    parser.add_argument("--expansion-factor", type=int, default=VIT_EXPANSION)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--chunk-size-bwd",
        type=int,
        default=None,
        help="Backward NTK/Jacobian chunk size forwarded to train_sweep.py.",
    )
    parser.add_argument("--linear-solver", type=str, default="cholesky")
    parser.add_argument("--mode", type=str, default=None, choices=["real", "complex"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--reuse-base",
        action="store_true",
        help="Skip Phase 0 retraining if the shared step-200 checkpoint already exists",
    )
    parser.add_argument(
        "--skip-existing-branches",
        action="store_true",
        help="Skip branch runs whose target directory already contains checkpoints",
    )
    return parser.parse_args()


def format_steps(steps: tuple[int, ...]) -> str:
    return ",".join(str(step) for step in steps)


def make_train_namespace(
    args,
    *,
    diag_shift: float,
    output_root: Path,
    phase_lengths: tuple[int, int, int],
    checkpoint_steps: tuple[int, ...],
    resume: str | None,
) -> Namespace:
    return Namespace(
        system=args.system,
        diag_shift=diag_shift,
        sweep=False,
        output_root=str(output_root),
        resume=resume,
        n_samples=args.n_samples,
        phase_samples=args.phase_samples,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        latest_checkpoint_every=args.latest_checkpoint_every,
        schedule_total_steps=None,
        phase_lengths=format_steps(phase_lengths),
        checkpoint_steps=format_steps(checkpoint_steps),
        depth=args.depth,
        d_model=args.d_model,
        heads=args.heads,
        expansion_factor=args.expansion_factor,
        chunk_size=args.chunk_size,
        chunk_size_bwd=args.chunk_size_bwd,
        linear_solver=args.linear_solver,
        mode=args.mode,
        seed=args.seed,
        fail_fast=args.fail_fast,
    )


def checkpoint_files_for_phase0(base_results_dir: Path) -> list[Path]:
    checkpoints_dir = base_results_dir / "checkpoints"
    return [
        checkpoints_dir / f"checkpoint_step{step:06d}.pkl"
        for step in PHASE0_CHECKPOINT_STEPS
        if (checkpoints_dir / f"checkpoint_step{step:06d}.pkl").exists()
    ]


def checkpoint_step_from_path(path: Path) -> int:
    stem = path.stem
    suffix = stem.removeprefix("checkpoint_step")
    return int(suffix)


def copy_phase0_metrics(base_results_dir: Path, target_results_dir: Path):
    source_csv = base_results_dir / "train_metrics.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing Phase 0 metrics file: {source_csv}")
    target_csv = target_results_dir / "train_metrics.csv"
    target_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(source_csv, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            if int(row["step"]) <= PHASE_LENGTHS[0]:
                rows.append(row)
    if header is None:
        raise RuntimeError(f"Unexpected empty metrics file: {source_csv}")

    with open(target_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def copy_phase0_prefix(base_results_dir: Path, target_results_dir: Path):
    target_results_dir.mkdir(parents=True, exist_ok=True)
    (target_results_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (target_results_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    (target_results_dir / "logs").mkdir(parents=True, exist_ok=True)

    copy_phase0_metrics(base_results_dir, target_results_dir)
    for checkpoint_path in checkpoint_files_for_phase0(base_results_dir):
        shutil.copy2(
            checkpoint_path,
            target_results_dir / "checkpoints" / checkpoint_path.name,
        )


def ensure_phase0_base(args, system_name: str) -> Path:
    base_output_root = Path(args.base_output_root)
    base_results_dir = results_dir_for_run(base_output_root, system_name, args.base_diag_shift)
    step200 = base_results_dir / "checkpoints" / f"checkpoint_step{PHASE_LENGTHS[0]:06d}.pkl"

    if args.reuse_base and step200.exists():
        return step200

    phase0_args = make_train_namespace(
        args,
        diag_shift=args.base_diag_shift,
        output_root=base_output_root,
        phase_lengths=(PHASE_LENGTHS[0], 0, 0),
        checkpoint_steps=PHASE0_CHECKPOINT_STEPS,
        resume=None,
    )
    run_single(phase0_args, system_name, float(args.base_diag_shift))

    if not step200.exists():
        raise FileNotFoundError(f"Expected shared Phase 0 checkpoint not found: {step200}")
    return step200


def run_branch(args, system_name: str, diag_shift: float, base_checkpoint: Path):
    target_results_dir = results_dir_for_run(Path(args.output_root), system_name, diag_shift)
    existing_checkpoints = sorted((target_results_dir / "checkpoints").glob("checkpoint_step*.pkl"))
    if existing_checkpoints:
        existing_steps = [checkpoint_step_from_path(path) for path in existing_checkpoints]
        has_branch_progress = any(step > PHASE_LENGTHS[0] for step in existing_steps)
        if has_branch_progress:
            if args.skip_existing_branches:
                print(f"Skipping existing branch: system={system_name} lambda={diag_shift:.0e}")
                return
            raise RuntimeError(
                f"{target_results_dir} already contains branch checkpoints; clean/archive it or pass --skip-existing-branches"
            )

    copy_phase0_prefix(base_checkpoint.parent.parent, target_results_dir)
    branch_args = make_train_namespace(
        args,
        diag_shift=diag_shift,
        output_root=Path(args.output_root),
        phase_lengths=PHASE_LENGTHS,
        checkpoint_steps=CHECKPOINT_STEPS,
        resume=str(base_checkpoint),
    )
    run_single(branch_args, system_name, float(diag_shift))


def main() -> int:
    args = parse_args()
    system_name = normalize_system_name(args.system)
    lambda_values = lambda_grid_for_system(system_name) if args.sweep else [args.diag_shift]

    base_checkpoint = ensure_phase0_base(args, system_name)
    print(f"Shared Phase 0 checkpoint: {base_checkpoint}")

    failures: list[tuple[float, str]] = []
    for diag_shift in lambda_values:
        try:
            run_branch(args, system_name, float(diag_shift), base_checkpoint)
        except Exception as exc:
            failures.append((float(diag_shift), str(exc)))
            print(
                f"\nFAILED system={system_name} lambda={float(diag_shift):.0e}: {exc}",
                flush=True,
            )
            if args.fail_fast:
                raise

    if failures:
        print("\nSweep completed with failures:", flush=True)
        for diag_shift, message in failures:
            print(f"  lambda={diag_shift:.0e}: {message}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
