from __future__ import annotations

import argparse
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
from sr_filtering_nqs.matched_state.common import (
    build_fullsum_state,
    compute_exact_ground_state,
    evaluate_state_match,
    exact_vector_fidelity,
)


DEFAULT_RESULTS_DIR = FIGURE2_DIR / "results"
DEFAULT_CHECKPOINT_DIR = FIGURE2_DIR / "checkpoints"
DEFAULT_RBM_INIT_CHECKPOINT = DEFAULT_CHECKPOINT_DIR / "rbm_a1_step0100.pkl"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue RBM training from a saved checkpoint, independently train a ViT "
            "toward the Hamiltonian, and save exact ground-state/pair fidelities."
        ),
    )
    parser.add_argument("--L", type=int, default=4, help="Square lattice linear size.")
    parser.add_argument("--J1", type=float, default=1.0, help="Nearest-neighbor coupling.")
    parser.add_argument("--J2", type=float, default=0.5, help="Next-nearest-neighbor coupling.")
    parser.add_argument(
        "--rbm-init-checkpoint",
        type=str,
        default=str(DEFAULT_RBM_INIT_CHECKPOINT),
        help="Saved RBM checkpoint used to resume RBM training.",
    )
    parser.add_argument(
        "--vit-model",
        type=str,
        default="ViT_d2_m24_h4_e4",
        help="ViT model spec to train toward the Hamiltonian.",
    )
    parser.add_argument(
        "--rbm-final-step",
        type=int,
        default=1000,
        help="Absolute RBM training step to stop at if the energy target is not reached earlier.",
    )
    parser.add_argument(
        "--vit-train-steps",
        type=int,
        default=1000,
        help="Maximum ViT VMC training steps toward the Hamiltonian.",
    )
    parser.add_argument(
        "--diag-every",
        type=int,
        default=10,
        help="Exact diagnostic interval used during warm-start training.",
    )
    parser.add_argument(
        "--rbm-train-samples",
        type=int,
        default=4096,
        help="MCState samples during resumed RBM training.",
    )
    parser.add_argument(
        "--vit-train-samples",
        type=int,
        default=4096,
        help="MCState samples during ViT warm-start training.",
    )
    parser.add_argument(
        "--rbm-learning-rate",
        type=float,
        default=0.01,
        help="RBM VMC learning rate.",
    )
    parser.add_argument(
        "--vit-learning-rate",
        type=float,
        default=0.01,
        help="ViT warm-start VMC learning rate.",
    )
    parser.add_argument(
        "--rbm-train-diag-shift",
        type=float,
        default=1e-3,
        help="RBM VMC SR diagonal shift.",
    )
    parser.add_argument(
        "--vit-train-diag-shift",
        type=float,
        default=1e-3,
        help="ViT warm-start VMC SR diagonal shift.",
    )
    parser.add_argument(
        "--rbm-stop-energy",
        type=float,
        default=-9.05,
        help="Stop resumed RBM training early once a checkpoint reaches this energy.",
    )
    parser.add_argument(
        "--vit-stop-energy",
        type=float,
        default=-9.05,
        help="Stop ViT warm-start training early once a checkpoint reaches this energy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed.",
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
        help="Directory for saved warm-start checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Result pickle path. Defaults to figure2/results/groundstate_warmstart_<vit>.pkl",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume warm-start training if stage progress exists under the checkpoint directory.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return __import__("pickle").load(f)


def best_energy_checkpoint(training: dict) -> tuple[int, dict, dict]:
    idx = min(
        range(len(training["checkpoint_summaries"])),
        key=lambda i: float(training["checkpoint_summaries"][i]["energy"]),
    )
    return idx, training["checkpoint_summaries"][idx], training["checkpoint_runtime"][idx]


def named_checkpoint_path(checkpoint_dir: Path, *, model_name: str, suffix: str) -> Path:
    return checkpoint_dir / f"{sanitize_name(model_name)}_{suffix}.pkl"


def save_selected_training_checkpoint(
    *,
    path: Path,
    system_meta: dict,
    model_name: str,
    model_desc: str,
    step: int,
    variables: dict,
    summary: dict,
) -> None:
    payload = build_training_checkpoint_payload(
        system_meta=system_meta,
        model_name=model_name,
        model_desc=model_desc,
        step=step,
        variables=variables,
        summary=summary,
    )
    save_pickle(path, payload)


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output is not None
        else DEFAULT_RESULTS_DIR / f"groundstate_warmstart_{sanitize_name(args.vit_model)}.pkl"
    )

    rbm_init_path = Path(args.rbm_init_checkpoint)
    rbm_init_payload = load_pickle(rbm_init_path)
    system_meta = dict(rbm_init_payload["system"])
    hilbert, hamiltonian, graph, _ = make_square_j1j2(
        L=int(system_meta["L"]),
        J1=float(system_meta["J1"]),
        J2=float(system_meta["J2"]),
    )
    all_states = np.asarray(hilbert.all_states())

    print("Figure 2 ground-state warm-start preparation")
    print(f"System: {system_meta}")
    print(f"Initial RBM checkpoint: {rbm_init_path}")
    print(f"Requested ViT model: {args.vit_model}")
    print(f"Output: {output_path}")

    ground_state = compute_exact_ground_state(hamiltonian)
    print(f"Exact ground-state energy: {ground_state['energy']:.12f}")

    rbm_model_name = rbm_init_payload["model_name"]
    rbm_model_desc = rbm_init_payload["model_desc"]
    rbm_model, _ = make_model(
        rbm_model_name,
        hilbert=hilbert,
        graph=graph,
        square_L=int(system_meta["L"]),
        j_couplings=(float(system_meta["J1"]), float(system_meta["J2"])),
    )

    rbm_stage_dir = checkpoint_dir / f"{sanitize_name(rbm_model_name)}_warmstart_stage"
    rbm_training = train_model_checkpoints(
        model=rbm_model,
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        all_states=all_states,
        train_samples=args.rbm_train_samples,
        train_steps=args.rbm_final_step,
        diag_every=args.diag_every,
        learning_rate=args.rbm_learning_rate,
        train_diag_shift=args.rbm_train_diag_shift,
        seed=args.seed,
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
        store_checkpoint_arrays=False,
        system_meta=system_meta,
        model_name=rbm_model_name,
        model_desc=rbm_model_desc,
        checkpoint_dir=rbm_stage_dir,
        resume=args.resume,
        initial_variables=rbm_init_payload["variables"],
        initial_step=int(rbm_init_payload["step"]),
        stop_energy=args.rbm_stop_energy,
    )
    rbm_best_idx, rbm_best_summary, rbm_best_runtime = best_energy_checkpoint(rbm_training)
    rbm_best_step = int(rbm_best_summary["step"])
    rbm_best_path = named_checkpoint_path(
        checkpoint_dir,
        model_name=rbm_model_name,
        suffix=f"groundstate_best_step{rbm_best_step:04d}",
    )
    save_selected_training_checkpoint(
        path=rbm_best_path,
        system_meta=system_meta,
        model_name=rbm_model_name,
        model_desc=rbm_model_desc,
        step=rbm_best_step,
        variables=rbm_best_runtime["variables"],
        summary=rbm_best_summary,
    )
    print(
        f"RBM best checkpoint: step={rbm_best_step}, "
        f"energy={float(rbm_best_summary['energy']):.12f}, "
        f"saved to {rbm_best_path}"
    )

    vit_model, vit_model_desc = make_model(
        args.vit_model,
        hilbert=hilbert,
        graph=graph,
        square_L=int(system_meta["L"]),
        j_couplings=(float(system_meta["J1"]), float(system_meta["J2"])),
    )
    vit_stage_dir = checkpoint_dir / f"{sanitize_name(args.vit_model)}_warmstart_stage"
    vit_training = train_model_checkpoints(
        model=vit_model,
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        all_states=all_states,
        train_samples=args.vit_train_samples,
        train_steps=args.vit_train_steps,
        diag_every=args.diag_every,
        learning_rate=args.vit_learning_rate,
        train_diag_shift=args.vit_train_diag_shift,
        seed=args.seed + 10_000,
        jac_chunk_size=args.jac_chunk_size,
        apply_chunk_size=args.apply_chunk_size,
        store_checkpoint_arrays=False,
        system_meta=system_meta,
        model_name=args.vit_model,
        model_desc=vit_model_desc,
        checkpoint_dir=vit_stage_dir,
        resume=args.resume,
        stop_energy=args.vit_stop_energy,
    )
    vit_best_idx, vit_best_summary, vit_best_runtime = best_energy_checkpoint(vit_training)
    vit_best_step = int(vit_best_summary["step"])
    vit_best_path = named_checkpoint_path(
        checkpoint_dir,
        model_name=args.vit_model,
        suffix=f"groundstate_best_step{vit_best_step:04d}",
    )
    save_selected_training_checkpoint(
        path=vit_best_path,
        system_meta=system_meta,
        model_name=args.vit_model,
        model_desc=vit_model_desc,
        step=vit_best_step,
        variables=vit_best_runtime["variables"],
        summary=vit_best_summary,
    )
    print(
        f"ViT best checkpoint: step={vit_best_step}, "
        f"energy={float(vit_best_summary['energy']):.12f}, "
        f"saved to {vit_best_path}"
    )

    rbm_state = build_fullsum_state(hilbert, rbm_model, rbm_best_runtime["variables"])
    vit_state = build_fullsum_state(hilbert, vit_model, vit_best_runtime["variables"])
    rbm_ground_metrics = exact_vector_fidelity(ground_state["vector"], rbm_state)
    vit_ground_metrics = exact_vector_fidelity(ground_state["vector"], vit_state)
    pair_metrics = evaluate_state_match(rbm_state, vit_state)

    results = {
        "format_version": 1,
        "created_at_utc": now(),
        "experiment": "figure2_groundstate_warmstart",
        "system": dict(system_meta),
        "ground_state": {
            "energy": float(ground_state["energy"]),
        },
        "rbm_stage": {
            "initial_checkpoint_path": str(rbm_init_path),
            "training_progress_path": None if rbm_training["progress_path"] is None else str(rbm_training["progress_path"]),
            "training_checkpoint_dir": None if rbm_training["checkpoint_dir"] is None else str(rbm_training["checkpoint_dir"]),
            "train_config": {
                "final_step": int(args.rbm_final_step),
                "train_samples": int(args.rbm_train_samples),
                "diag_every": int(args.diag_every),
                "learning_rate": float(args.rbm_learning_rate),
                "train_diag_shift": float(args.rbm_train_diag_shift),
                "stop_energy": float(args.rbm_stop_energy),
            },
            "selected_index": int(rbm_best_idx),
            "selected_step": int(rbm_best_step),
            "selected_summary": dict(rbm_best_summary),
            "selected_checkpoint_path": str(rbm_best_path),
            "checkpoint_summaries": list(rbm_training["checkpoint_summaries"]),
            "ground_state_metrics": rbm_ground_metrics,
        },
        "vit_stage": {
            "training_progress_path": None if vit_training["progress_path"] is None else str(vit_training["progress_path"]),
            "training_checkpoint_dir": None if vit_training["checkpoint_dir"] is None else str(vit_training["checkpoint_dir"]),
            "train_config": {
                "train_steps": int(args.vit_train_steps),
                "train_samples": int(args.vit_train_samples),
                "diag_every": int(args.diag_every),
                "learning_rate": float(args.vit_learning_rate),
                "train_diag_shift": float(args.vit_train_diag_shift),
                "stop_energy": float(args.vit_stop_energy),
            },
            "selected_index": int(vit_best_idx),
            "selected_step": int(vit_best_step),
            "selected_summary": dict(vit_best_summary),
            "selected_checkpoint_path": str(vit_best_path),
            "checkpoint_summaries": list(vit_training["checkpoint_summaries"]),
            "ground_state_metrics": vit_ground_metrics,
        },
        "warmstart_pair_metrics": pair_metrics,
    }
    save_pickle(output_path, results)

    print(
        "Warm-start summary: "
        f"RBM E={float(rbm_best_summary['energy']):.9f}, "
        f"ViT E={float(vit_best_summary['energy']):.9f}, "
        f"RBM-ground F={rbm_ground_metrics['fidelity']:.9f}, "
        f"ViT-ground F={vit_ground_metrics['fidelity']:.9f}, "
        f"RBM-ViT F={pair_metrics['fidelity']:.9f}"
    )
    print(f"Saved warm-start results to {output_path}")


if __name__ == "__main__":
    main()
