from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE2_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import netket as nk
import numpy as np

from advanced_drivers.driver import InfidelityOptimizerNG_FS
from sr_filtering_nqs.small_scale.common import copy_variables, make_model, make_square_j1j2, now, save_pickle
from sr_filtering_nqs.matched_state.common import build_fullsum_state, evaluate_state_match


DEFAULT_OUTPUT_DIR = FIGURE2_DIR / "results"
DEFAULT_CHECKPOINT_DIR = FIGURE2_DIR / "checkpoints"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def default_output_path(rbm_checkpoint: Path, vit_model: str) -> Path:
    stem = sanitize_name(rbm_checkpoint.stem)
    vit_label = sanitize_name(vit_model)
    return DEFAULT_OUTPUT_DIR / f"{stem}_to_{vit_label}_match.pkl"


def default_vit_checkpoint_path(vit_model: str) -> Path:
    vit_label = sanitize_name(vit_model)
    return DEFAULT_CHECKPOINT_DIR / f"{vit_label}_matched_best_from_checkpoint.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact full-sum ViT infidelity minimization against a saved RBM checkpoint.",
    )
    parser.add_argument(
        "--rbm-checkpoint",
        type=str,
        required=True,
        help="Path to the saved RBM checkpoint payload used as the exact target state.",
    )
    parser.add_argument(
        "--vit-model",
        type=str,
        default="ViT_d2_m24_h4_e4",
        help="ViT model spec to match to the saved RBM target.",
    )
    parser.add_argument(
        "--initial-vit-checkpoint",
        type=str,
        default=None,
        help="Optional saved ViT checkpoint payload used to warm-start the infidelity match.",
    )
    parser.add_argument(
        "--match-max-steps",
        type=int,
        default=2000,
        help="Maximum exact infidelity-minimization steps.",
    )
    parser.add_argument(
        "--match-learning-rate",
        type=float,
        default=5e-2,
        help="Learning rate for exact infidelity minimization.",
    )
    parser.add_argument(
        "--match-diag-shift",
        type=float,
        default=1e-4,
        help="Diagonal shift for exact infidelity minimization.",
    )
    parser.add_argument(
        "--match-fidelity-threshold",
        type=float,
        default=0.99,
        help="Stop early once the exact fidelity reaches this threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to initialize the ViT.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save the running result pickle every this many optimization steps.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every this many optimization steps.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to the match-result pickle. Defaults to figure2/results/<rbm>_to_<vit>_match.pkl.",
    )
    parser.add_argument(
        "--vit-checkpoint-output",
        type=str,
        default=None,
        help="Path to the best-so-far ViT checkpoint payload.",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as f:
        return __import__("pickle").load(f)


def build_vit_checkpoint_payload(
    *,
    system_meta: dict,
    vit_model_name: str,
    vit_model_desc: str,
    target_checkpoint_path: Path,
    step: int,
    variables: dict,
    metrics: dict,
) -> dict:
    return {
        "format_version": 1,
        "created_at_utc": now(),
        "system": dict(system_meta),
        "model_name": vit_model_name,
        "model_desc": vit_model_desc,
        "state_origin": "vit_infidelity_checkpoint",
        "target_checkpoint_path": str(target_checkpoint_path),
        "step": int(step),
        "variables": copy_variables(variables),
        "match_metrics": {
            "fidelity": float(metrics["fidelity"]),
            "infidelity_direct": float(metrics["infidelity_direct"]),
            "infidelity_operator": float(metrics["infidelity_operator"]),
            "infidelity_abs_error": float(metrics["infidelity_abs_error"]),
            "overlap_abs": float(metrics["overlap_abs"]),
        },
    }


def build_result_payload(
    *,
    status: str,
    created_at_utc: str,
    system_meta: dict,
    rbm_checkpoint_path: Path,
    rbm_payload: dict,
    vit_model_name: str,
    vit_model_desc: str,
    match_config: dict,
    history: list[dict],
    best_metrics: dict,
    best_step_time_s: float | None,
    best_variables: dict,
    final_metrics: dict,
    final_step: int,
    steps_completed: int,
    stop_reason: str,
    success: bool,
    elapsed_wall_s: float,
    vit_checkpoint_output: Path,
) -> dict:
    return {
        "format_version": 1,
        "created_at_utc": created_at_utc,
        "updated_at_utc": now(),
        "status": status,
        "system": dict(system_meta),
        "rbm_target": {
            "checkpoint_path": str(rbm_checkpoint_path),
            "model_name": rbm_payload["model_name"],
            "model_desc": rbm_payload["model_desc"],
            "state_origin": rbm_payload["state_origin"],
            "step": int(rbm_payload["step"]),
        },
        "match_config": dict(match_config),
        "match_model_name": vit_model_name,
        "match_model_desc": vit_model_desc,
        "history": history,
        "initial_metrics": dict(history[0]),
        "best_metrics": dict(best_metrics),
        "best_step": int(best_metrics["step"]),
        "best_step_time_s": None if best_step_time_s is None else float(best_step_time_s),
        "final_metrics": dict(final_metrics),
        "final_step": int(final_step),
        "steps_completed": int(steps_completed),
        "stop_reason": stop_reason,
        "success": bool(success),
        "elapsed_wall_s": float(elapsed_wall_s),
        "vit_checkpoint_output": str(vit_checkpoint_output),
        "best_variables": copy_variables(best_variables),
    }


def save_running_state(
    *,
    output_path: Path,
    vit_checkpoint_output: Path,
    result_payload: dict,
    checkpoint_payload: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vit_checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(output_path, result_payload)
    save_pickle(vit_checkpoint_output, checkpoint_payload)


def main() -> None:
    args = parse_args()
    rbm_checkpoint_path = Path(args.rbm_checkpoint)
    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path(rbm_checkpoint_path, args.vit_model)
    )
    vit_checkpoint_output = (
        Path(args.vit_checkpoint_output)
        if args.vit_checkpoint_output is not None
        else default_vit_checkpoint_path(args.vit_model)
    )

    rbm_payload = load_pickle(rbm_checkpoint_path)
    system_meta = dict(rbm_payload["system"])
    hilbert, _, graph, _ = make_square_j1j2(
        L=int(system_meta["L"]),
        J1=float(system_meta["J1"]),
        J2=float(system_meta["J2"]),
    )

    rbm_model, rbm_desc = make_model(
        rbm_payload["model_name"],
        hilbert=hilbert,
        graph=graph,
        square_L=int(system_meta["L"]),
        j_couplings=(float(system_meta["J1"]), float(system_meta["J2"])),
    )
    target_state = build_fullsum_state(hilbert, rbm_model, rbm_payload["variables"])

    vit_model, vit_desc = make_model(
        args.vit_model,
        hilbert=hilbert,
        graph=graph,
        square_L=int(system_meta["L"]),
        j_couplings=(float(system_meta["J1"]), float(system_meta["J2"])),
    )
    variational_state = nk.vqs.FullSumState(hilbert=hilbert, model=vit_model, seed=args.seed)
    initial_vit_payload = None
    if args.initial_vit_checkpoint is not None:
        initial_vit_path = Path(args.initial_vit_checkpoint)
        initial_vit_payload = load_pickle(initial_vit_path)
        if initial_vit_payload["model_name"] != args.vit_model:
            raise ValueError(
                f"Initial ViT checkpoint model_name={initial_vit_payload['model_name']} "
                f"does not match requested vit-model={args.vit_model}."
            )
        variational_state.variables = copy_variables(initial_vit_payload["variables"])
    optimizer = nk.optimizer.Sgd(learning_rate=args.match_learning_rate)
    driver = InfidelityOptimizerNG_FS(
        target_state=target_state,
        optimizer=optimizer,
        variational_state=variational_state,
        diag_shift=args.match_diag_shift,
    )

    print("Figure 2 checkpoint-based infidelity match")
    print(f"Target checkpoint: {rbm_checkpoint_path}")
    print(f"Target model: {rbm_payload['model_name']} ({rbm_desc}) @ step {rbm_payload['step']}")
    print(f"Match model: {args.vit_model} ({vit_desc})")
    if args.initial_vit_checkpoint is not None:
        print(f"Initial ViT checkpoint: {args.initial_vit_checkpoint}")
    print(f"Output pickle: {output_path}")
    print(f"Best-checkpoint output: {vit_checkpoint_output}")

    created_at_utc = now()
    start_wall = time.perf_counter()
    history: list[dict] = []
    best_variables = copy_variables(variational_state.variables)
    best_metrics = None
    best_step_time_s = None
    stop_reason = "max_steps_reached"

    match_config = {
        "match_learning_rate": float(args.match_learning_rate),
        "match_diag_shift": float(args.match_diag_shift),
        "match_max_steps": int(args.match_max_steps),
        "match_fidelity_threshold": float(args.match_fidelity_threshold),
        "seed": int(args.seed),
        "save_every": int(args.save_every),
        "log_every": int(args.log_every),
        "initial_vit_checkpoint": None if args.initial_vit_checkpoint is None else str(args.initial_vit_checkpoint),
    }

    for step in range(args.match_max_steps + 1):
        iter_start = time.perf_counter()
        if step > 0:
            driver.run(1, show_progress=False)
        metrics = evaluate_state_match(target_state, variational_state)
        iter_elapsed = time.perf_counter() - iter_start
        entry = {"step": int(step), "step_wall_s": float(iter_elapsed), **metrics}
        history.append(entry)

        if best_metrics is None or metrics["infidelity_direct"] < best_metrics["infidelity_direct"]:
            best_metrics = dict(entry)
            best_variables = copy_variables(variational_state.variables)
            best_step_time_s = float(iter_elapsed)
            checkpoint_payload = build_vit_checkpoint_payload(
                system_meta=system_meta,
                vit_model_name=args.vit_model,
                vit_model_desc=vit_desc,
                target_checkpoint_path=rbm_checkpoint_path,
                step=step,
                variables=best_variables,
                metrics=metrics,
            )
            result_payload = build_result_payload(
                status="running",
                created_at_utc=created_at_utc,
                system_meta=system_meta,
                rbm_checkpoint_path=rbm_checkpoint_path,
                rbm_payload=rbm_payload,
                vit_model_name=args.vit_model,
                vit_model_desc=vit_desc,
                match_config=match_config,
                history=history,
                best_metrics=best_metrics,
                best_step_time_s=best_step_time_s,
                best_variables=best_variables,
                final_metrics=entry,
                final_step=step,
                steps_completed=max(0, step),
                stop_reason="running",
                success=False,
                elapsed_wall_s=time.perf_counter() - start_wall,
                vit_checkpoint_output=vit_checkpoint_output,
            )
            save_running_state(
                output_path=output_path,
                vit_checkpoint_output=vit_checkpoint_output,
                result_payload=result_payload,
                checkpoint_payload=checkpoint_payload,
            )

        if step == 0 or step % args.log_every == 0:
            print(
                f"step={step:04d} "
                f"fidelity={metrics['fidelity']:.9f} "
                f"infidelity={metrics['infidelity_direct']:.9f} "
                f"best_fidelity={best_metrics['fidelity']:.9f} "
                f"step_wall_s={iter_elapsed:.3f}"
            )

        should_save = step > 0 and step % args.save_every == 0
        if should_save:
            checkpoint_payload = build_vit_checkpoint_payload(
                system_meta=system_meta,
                vit_model_name=args.vit_model,
                vit_model_desc=vit_desc,
                target_checkpoint_path=rbm_checkpoint_path,
                step=int(best_metrics["step"]),
                variables=best_variables,
                metrics=best_metrics,
            )
            result_payload = build_result_payload(
                status="running",
                created_at_utc=created_at_utc,
                system_meta=system_meta,
                rbm_checkpoint_path=rbm_checkpoint_path,
                rbm_payload=rbm_payload,
                vit_model_name=args.vit_model,
                vit_model_desc=vit_desc,
                match_config=match_config,
                history=history,
                best_metrics=best_metrics,
                best_step_time_s=best_step_time_s,
                best_variables=best_variables,
                final_metrics=entry,
                final_step=step,
                steps_completed=max(0, step),
                stop_reason="running",
                success=False,
                elapsed_wall_s=time.perf_counter() - start_wall,
                vit_checkpoint_output=vit_checkpoint_output,
            )
            save_running_state(
                output_path=output_path,
                vit_checkpoint_output=vit_checkpoint_output,
                result_payload=result_payload,
                checkpoint_payload=checkpoint_payload,
            )

        if metrics["fidelity"] >= args.match_fidelity_threshold:
            stop_reason = "threshold_reached"
            break

    assert best_metrics is not None
    final_metrics = history[-1]
    success = bool(best_metrics["fidelity"] >= args.match_fidelity_threshold)
    elapsed_wall_s = time.perf_counter() - start_wall

    final_checkpoint_payload = build_vit_checkpoint_payload(
        system_meta=system_meta,
        vit_model_name=args.vit_model,
        vit_model_desc=vit_desc,
        target_checkpoint_path=rbm_checkpoint_path,
        step=int(best_metrics["step"]),
        variables=best_variables,
        metrics=best_metrics,
    )
    final_result_payload = build_result_payload(
        status="completed",
        created_at_utc=created_at_utc,
        system_meta=system_meta,
        rbm_checkpoint_path=rbm_checkpoint_path,
        rbm_payload=rbm_payload,
        vit_model_name=args.vit_model,
        vit_model_desc=vit_desc,
        match_config=match_config,
        history=history,
        best_metrics=best_metrics,
        best_step_time_s=best_step_time_s,
        best_variables=best_variables,
        final_metrics=final_metrics,
        final_step=int(final_metrics["step"]),
        steps_completed=max(0, int(final_metrics["step"])),
        stop_reason=stop_reason,
        success=success,
        elapsed_wall_s=elapsed_wall_s,
        vit_checkpoint_output=vit_checkpoint_output,
    )
    save_running_state(
        output_path=output_path,
        vit_checkpoint_output=vit_checkpoint_output,
        result_payload=final_result_payload,
        checkpoint_payload=final_checkpoint_payload,
    )
    print(
        "Final match summary: "
        f"best_fidelity={best_metrics['fidelity']:.9f}, "
        f"best_step={int(best_metrics['step'])}, "
        f"stop_reason={stop_reason}, "
        f"elapsed_wall_s={elapsed_wall_s:.1f}"
    )


if __name__ == "__main__":
    main()
