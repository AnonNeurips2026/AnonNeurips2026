from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE2_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from sr_filtering_nqs.small_scale.common import (
    build_training_checkpoint_payload,
    make_model,
    make_square_j1j2,
    now,
    save_pickle,
    train_model_checkpoints,
)
from sr_filtering_nqs.matched_state.common import compute_exact_ground_state


DEFAULT_RESULTS_DIR = FIGURE2_DIR / "results"
DEFAULT_CHECKPOINT_DIR = FIGURE2_DIR / "checkpoints_groundstate_rbm_continue"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue RBM training from a saved checkpoint and save the best "
            "ground-state checkpoint reached during the continuation run."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        required=True,
        help="Saved RBM checkpoint used to initialize the continuation run.",
    )
    parser.add_argument(
        "--final-step",
        type=int,
        default=5000,
        help="Absolute training step to stop at if stop-energy is not reached earlier.",
    )
    parser.add_argument(
        "--diag-every",
        type=int,
        default=10,
        help="Exact diagnostic interval during continuation.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=8192,
        help="MCState samples during continuation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-3,
        help="RBM continuation learning rate.",
    )
    parser.add_argument(
        "--train-diag-shift",
        type=float,
        default=1e-3,
        help="RBM continuation SR diagonal shift.",
    )
    parser.add_argument(
        "--stop-energy",
        type=float,
        default=-9.02,
        help="Stop early once a checkpoint reaches this exact energy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base seed for MCState training.",
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
        help="Chunk size for exact forward/JVP evaluations.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory for continuation checkpoints and progress.pkl.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Result pickle path. Defaults to figure2/results/<model>_groundstate_continue.pkl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint-dir/progress.pkl if it already exists.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def main() -> None:
    args = parse_args()
    init_path = Path(args.init_checkpoint)
    init_payload = load_pickle(init_path)
    system_meta = dict(init_payload["system"])

    checkpoint_root = Path(args.checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output is not None
        else DEFAULT_RESULTS_DIR
        / f"{sanitize_name(init_payload['model_name'])}_groundstate_continue.pkl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage_dir = checkpoint_root / f"{sanitize_name(init_payload['model_name'])}_stage"
    best_named_path = checkpoint_root / f"{sanitize_name(init_payload['model_name'])}_groundstate_continue_best.pkl"

    hilbert, hamiltonian, graph, _ = make_square_j1j2(
        L=int(system_meta["L"]),
        J1=float(system_meta["J1"]),
        J2=float(system_meta["J2"]),
    )
    all_states = np.asarray(hilbert.all_states())
    ground = compute_exact_ground_state(hamiltonian)

    model_name = init_payload["model_name"]
    model_desc = init_payload["model_desc"]
    model, _ = make_model(
        model_name,
        hilbert=hilbert,
        graph=graph,
        square_L=int(system_meta["L"]),
        j_couplings=(float(system_meta["J1"]), float(system_meta["J2"])),
    )

    print("Continuing RBM toward the ground state")
    print(f"Initial checkpoint: {init_path}")
    print(f"Initial step: {init_payload['step']}")
    print(f"Initial energy: {init_payload['summary']['energy']:.12f}")
    print(f"Exact ground-state energy: {ground['energy']:.12f}")
    print(
        "Training config: "
        f"train_samples={args.train_samples}, "
        f"train_steps={args.final_step}, "
        f"diag_every={args.diag_every}, "
        f"learning_rate={args.learning_rate}, "
        f"train_diag_shift={args.train_diag_shift}, "
        f"stop_energy={args.stop_energy}"
    )

    training = train_model_checkpoints(
        model=model,
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        all_states=all_states,
        train_samples=args.train_samples,
        train_steps=args.final_step,
        diag_every=args.diag_every,
        learning_rate=args.learning_rate,
        train_diag_shift=args.train_diag_shift,
        seed=args.seed,
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
        store_checkpoint_arrays=False,
        system_meta=system_meta,
        model_name=model_name,
        model_desc=model_desc,
        checkpoint_dir=stage_dir,
        resume=args.resume,
        initial_variables=init_payload["variables"],
        initial_step=int(init_payload["step"]),
        stop_energy=args.stop_energy,
    )

    summaries = training["checkpoint_summaries"]
    runtime = training["checkpoint_runtime"]
    idx = min(range(len(summaries)), key=lambda i: float(summaries[i]["energy"]))
    best_summary = summaries[idx]
    best_runtime = runtime[idx]

    best_payload = build_training_checkpoint_payload(
        system_meta=system_meta,
        model_name=model_name,
        model_desc=model_desc,
        step=int(best_summary["step"]),
        variables=best_runtime["variables"],
        summary=best_summary,
    )
    save_pickle(best_named_path, best_payload)

    result = {
        "format_version": 1,
        "status": "complete",
        "started_from_checkpoint": str(init_path),
        "system": system_meta,
        "model_name": model_name,
        "model_desc": model_desc,
        "ground_energy": float(ground["energy"]),
        "initial_step": int(init_payload["step"]),
        "initial_energy": float(init_payload["summary"]["energy"]),
        "completed_step": int(summaries[-1]["step"]),
        "last_energy": float(summaries[-1]["energy"]),
        "best_step": int(best_summary["step"]),
        "best_energy": float(best_summary["energy"]),
        "energy_error_to_ground": float(best_summary["energy"] - float(ground["energy"])),
        "best_checkpoint_path": str(best_named_path),
        "progress_path": training["progress_path"],
        "checkpoint_dir": training["checkpoint_dir"],
        "n_checkpoints": len(summaries),
        "updated_at_utc": now(),
    }
    save_pickle(output_path, result)

    print("Training complete.")
    print(f"Completed step: {summaries[-1]['step']}")
    print(f"Best step: {best_summary['step']}")
    print(f"Best energy: {best_summary['energy']:.12f}")
    print(f"Energy error to exact ground state: {result['energy_error_to_ground']:.12f}")
    print(f"Best checkpoint: {best_named_path}")
    print(f"Result pickle: {output_path}")


if __name__ == "__main__":
    main()
