from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE2_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import netket as nk
import numpy as np

from advanced_drivers.driver import InfidelityOptimizerNG
from sr_filtering_nqs.small_scale.common import copy_variables, make_model, make_square_j1j2, now, save_pickle
from sr_filtering_nqs.matched_state.common import build_fullsum_state, evaluate_state_match


DEFAULT_OUTPUT_DIR = FIGURE2_DIR / "results"
DEFAULT_CHECKPOINT_DIR = FIGURE2_DIR / "checkpoints"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def load_pickle(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return __import__("pickle").load(f)


def default_output_path(rbm_checkpoint: Path, vit_checkpoint: Path) -> Path:
    rbm_label = sanitize_name(rbm_checkpoint.stem)
    vit_label = sanitize_name(vit_checkpoint.stem)
    return DEFAULT_OUTPUT_DIR / f"alternating_{rbm_label}_and_{vit_label}.pkl"


def stage_checkpoint_path(
    checkpoint_dir: Path,
    *,
    stage_index: int,
    optimize_model_name: str,
    target_model_name: str,
) -> Path:
    optimize_label = sanitize_name(optimize_model_name)
    target_label = sanitize_name(target_model_name)
    return checkpoint_dir / (
        f"alternating_stage{stage_index:02d}_{optimize_label}_to_{target_label}_best.pkl"
    )


def checkpoint_payload(
    *,
    system_meta: dict[str, Any],
    model_name: str,
    model_desc: str,
    variables: dict[str, Any],
    stage_index: int,
    step: int,
    role: str,
    target_model_name: str,
    target_model_desc: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "created_at_utc": now(),
        "system": dict(system_meta),
        "model_name": model_name,
        "model_desc": model_desc,
        "state_origin": "alternating_infidelity_checkpoint",
        "alternating_role": role,
        "stage_index": int(stage_index),
        "step": int(step),
        "target_model_name": target_model_name,
        "target_model_desc": target_model_desc,
        "variables": copy_variables(variables),
        "match_metrics": dict(metrics),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Alternate sampled MCMC infidelity minimization between saved RBM and ViT checkpoints."
        ),
    )
    parser.add_argument(
        "--rbm-checkpoint",
        type=str,
        required=True,
        help="Initial RBM checkpoint payload.",
    )
    parser.add_argument(
        "--vit-checkpoint",
        type=str,
        required=True,
        help="Initial ViT checkpoint payload.",
    )
    parser.add_argument(
        "--start-with",
        choices=("vit", "rbm"),
        default="vit",
        help="Which model to optimize in stage 0.",
    )
    parser.add_argument(
        "--max-stages",
        type=int,
        default=6,
        help="Maximum number of alternating stages.",
    )
    parser.add_argument(
        "--stage-max-steps",
        type=int,
        default=200,
        help="Maximum sampled infidelity steps per stage.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8192,
        help="MC samples drawn from the optimized model per step.",
    )
    parser.add_argument(
        "--target-n-samples",
        type=int,
        default=None,
        help="MC samples drawn from the frozen target model per step. Defaults to --n-samples.",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=16,
        help="Number of Metropolis chains for both models.",
    )
    parser.add_argument(
        "--n-discard-per-chain",
        type=int,
        default=4,
        help="Metropolis burn-in / decorrelation sweeps per chain.",
    )
    parser.add_argument(
        "--vit-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate when optimizing the ViT.",
    )
    parser.add_argument(
        "--rbm-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate when optimizing the RBM.",
    )
    parser.add_argument(
        "--vit-diag-shift",
        type=float,
        default=1e-4,
        help="Diagonal shift when optimizing the ViT.",
    )
    parser.add_argument(
        "--rbm-diag-shift",
        type=float,
        default=1e-4,
        help="Diagonal shift when optimizing the RBM.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=50,
        help="Stop a stage once this many steps pass without significant exact-fidelity gain.",
    )
    parser.add_argument(
        "--plateau-min-fidelity-gain",
        type=float,
        default=1e-4,
        help="Minimum fidelity gain counted as real progress for plateau detection.",
    )
    parser.add_argument(
        "--plateau-warmup-steps",
        type=int,
        default=20,
        help="Do not trigger plateau stopping before this many steps.",
    )
    parser.add_argument(
        "--exact-eval-every",
        type=int,
        default=5,
        help="Evaluate exact full-sum fidelity every this many steps and at step 0.",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="cmc",
        choices=("cmc", "smc"),
        help="Sampled infidelity estimator used by the MCMC driver.",
    )
    parser.add_argument(
        "--use-ntk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the kernel-trick / MinSR linear solve instead of parameter-space SR.",
    )
    parser.add_argument(
        "--chunk-size-bwd",
        type=int,
        default=None,
        help="Optional chunk size for on-the-fly backward passes.",
    )
    parser.add_argument(
        "--resample-fraction",
        type=float,
        default=None,
        help="Optional fraction of MC samples to resample each step.",
    )
    parser.add_argument(
        "--stage-fidelity-threshold",
        type=float,
        default=None,
        help="Optional per-stage early-stop threshold on exact pair fidelity.",
    )
    parser.add_argument(
        "--global-fidelity-threshold",
        type=float,
        default=None,
        help="Optional early-stop threshold on pair fidelity after a stage completes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed for variational state initialization.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print stage progress every this many steps.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Result pickle path.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(DEFAULT_CHECKPOINT_DIR),
        help="Directory for per-stage best checkpoint payloads.",
    )
    return parser.parse_args()


def validate_same_system(
    rbm_payload: dict[str, Any],
    vit_payload: dict[str, Any],
) -> dict[str, Any]:
    rbm_system = dict(rbm_payload["system"])
    vit_system = dict(vit_payload["system"])
    required_keys = ("name", "L", "J1", "J2")
    for key in required_keys:
        if rbm_system.get(key) != vit_system.get(key):
            raise ValueError(
                f"Checkpoint system mismatch on {key}: "
                f"{rbm_system.get(key)} != {vit_system.get(key)}"
            )
    return rbm_system


def make_state_from_variables(
    *,
    model_name: str,
    hilbert: Any,
    graph: Any,
    square_L: int,
    j_couplings: tuple[float, float],
    variables: dict[str, Any],
) -> tuple[Any, str, nk.vqs.FullSumState]:
    model, model_desc = make_model(
        model_name,
        hilbert=hilbert,
        graph=graph,
        square_L=square_L,
        j_couplings=j_couplings,
    )
    state = build_fullsum_state(hilbert, model, variables)
    return model, model_desc, state


def should_exact_eval(step: int, exact_eval_every: int) -> bool:
    return step == 0 or (exact_eval_every > 0 and step % exact_eval_every == 0)


def run_alternating_stage(
    *,
    stage_index: int,
    role: str,
    system_meta: dict[str, Any],
    hilbert: Any,
    graph: Any,
    square_L: int,
    j_couplings: tuple[float, float],
    optimize_model_name: str,
    target_model_name: str,
    optimize_initial_variables: dict[str, Any],
    target_variables: dict[str, Any],
    n_samples: int,
    target_n_samples: int,
    n_chains: int,
    n_discard_per_chain: int,
    learning_rate: float,
    diag_shift: float,
    max_steps: int,
    plateau_patience: int,
    plateau_min_fidelity_gain: float,
    plateau_warmup_steps: int,
    exact_eval_every: int,
    estimator: str,
    use_ntk: bool,
    chunk_size_bwd: int | None,
    resample_fraction: float | None,
    fidelity_threshold: float | None,
    seed: int,
    log_every: int,
    checkpoint_output: Path,
) -> dict[str, Any]:
    target_model, target_model_desc, exact_target_state = make_state_from_variables(
        model_name=target_model_name,
        hilbert=hilbert,
        graph=graph,
        square_L=square_L,
        j_couplings=j_couplings,
        variables=target_variables,
    )
    optimize_model, optimize_model_desc = make_model(
        optimize_model_name,
        hilbert=hilbert,
        graph=graph,
        square_L=square_L,
        j_couplings=j_couplings,
    )
    target_sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)
    variational_sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_chains)
    target_state = nk.vqs.MCState(
        target_sampler,
        model=target_model,
        n_samples=target_n_samples,
        n_discard_per_chain=n_discard_per_chain,
        seed=seed,
    )
    variational_state = nk.vqs.MCState(
        variational_sampler,
        model=optimize_model,
        n_samples=n_samples,
        n_discard_per_chain=n_discard_per_chain,
        seed=seed + 1,
    )
    target_state.variables = copy_variables(target_variables)
    variational_state.variables = copy_variables(optimize_initial_variables)
    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
    driver = InfidelityOptimizerNG(
        target_state=target_state,
        optimizer=optimizer,
        variational_state=variational_state,
        diag_shift=diag_shift,
        estimator=estimator,
        use_ntk=use_ntk,
        chunk_size_bwd=chunk_size_bwd,
        resample_fraction=resample_fraction,
    )

    history: list[dict[str, Any]] = []
    best_variables = copy_variables(variational_state.variables)
    best_metrics: dict[str, Any] | None = None
    best_selection_metric = "exact_fidelity"
    stop_reason = "max_steps_reached"
    significant_best_fidelity = float("-inf")
    last_significant_improve_step = 0

    for step in range(max_steps + 1):
        iter_start = time.perf_counter()
        if step > 0:
            driver.run(1, show_progress=False)
        iter_elapsed = time.perf_counter() - iter_start
        entry = {
            "step": int(step),
            "step_wall_s": float(iter_elapsed),
            "mc_infidelity_mean": float("nan"),
            "mc_infidelity_error": float("nan"),
            "mc_infidelity_variance": float("nan"),
        }
        if step > 0 and driver._loss_stats is not None:
            mc_infidelity = driver._loss_stats
            entry["mc_infidelity_mean"] = float(np.real(mc_infidelity.mean))
            entry["mc_infidelity_error"] = float(np.real(mc_infidelity.error_of_mean))
            entry["mc_infidelity_variance"] = float(np.real(mc_infidelity.variance))

        exact_metrics = None
        if should_exact_eval(step, exact_eval_every):
            exact_start = time.perf_counter()
            exact_variational_state = build_fullsum_state(
                hilbert,
                optimize_model,
                copy_variables(variational_state.variables),
            )
            exact_metrics = evaluate_state_match(exact_target_state, exact_variational_state)
            entry.update(exact_metrics)
            entry["exact_eval_wall_s"] = float(time.perf_counter() - exact_start)
        history.append(entry)

        improved = False
        if exact_metrics is not None:
            if best_metrics is None or exact_metrics["fidelity"] > float(best_metrics.get("fidelity", -1.0)):
                improved = True
                best_selection_metric = "exact_fidelity"

        if improved:
            best_metrics = dict(entry)
            best_variables = copy_variables(variational_state.variables)
            save_pickle(
                checkpoint_output,
                checkpoint_payload(
                    system_meta=system_meta,
                    model_name=optimize_model_name,
                    model_desc=optimize_model_desc,
                    variables=best_variables,
                    stage_index=stage_index,
                    step=step,
                    role=role,
                    target_model_name=target_model_name,
                    target_model_desc=target_model_desc,
                    metrics=best_metrics,
                ),
            )

        if exact_metrics is not None and exact_metrics["fidelity"] > significant_best_fidelity + plateau_min_fidelity_gain:
            significant_best_fidelity = float(exact_metrics["fidelity"])
            last_significant_improve_step = step

        if step == 0 or step % log_every == 0:
            parts = [
                f"stage={stage_index:02d}",
                f"role={role}",
                f"step={step:04d}",
                f"mc_infidelity={entry['mc_infidelity_mean']:.9f}",
                f"step_wall_s={iter_elapsed:.3f}",
            ]
            if exact_metrics is not None:
                parts.append(f"exact_fidelity={exact_metrics['fidelity']:.9f}")
            if best_metrics is not None and "fidelity" in best_metrics:
                parts.append(f"best_exact_fidelity={best_metrics['fidelity']:.9f}")
            print(" ".join(parts))

        if exact_metrics is not None and fidelity_threshold is not None and exact_metrics["fidelity"] >= fidelity_threshold:
            stop_reason = "stage_threshold_reached"
            break
        if step >= plateau_warmup_steps and step - last_significant_improve_step >= plateau_patience:
            stop_reason = "plateau"
            break

    assert best_metrics is not None
    return {
        "stage_index": int(stage_index),
        "role": role,
        "optimize_model_name": optimize_model_name,
        "optimize_model_desc": optimize_model_desc,
        "target_model_name": target_model_name,
        "target_model_desc": target_model_desc,
        "learning_rate": float(learning_rate),
        "diag_shift": float(diag_shift),
        "seed": int(seed),
        "n_samples": int(n_samples),
        "target_n_samples": int(target_n_samples),
        "n_chains": int(n_chains),
        "n_discard_per_chain": int(n_discard_per_chain),
        "exact_eval_every": int(exact_eval_every),
        "estimator": estimator,
        "use_ntk": bool(use_ntk),
        "chunk_size_bwd": None if chunk_size_bwd is None else int(chunk_size_bwd),
        "resample_fraction": None if resample_fraction is None else float(resample_fraction),
        "history": history,
        "best_step": int(best_metrics["step"]),
        "best_metrics": best_metrics,
        "best_selection_metric": best_selection_metric,
        "best_variables": best_variables,
        "final_step": int(history[-1]["step"]),
        "final_metrics": dict(history[-1]),
        "steps_completed": int(len(history) - 1),
        "stop_reason": stop_reason,
        "plateau_patience": int(plateau_patience),
        "plateau_min_fidelity_gain": float(plateau_min_fidelity_gain),
        "plateau_warmup_steps": int(plateau_warmup_steps),
        "last_significant_improve_step": int(last_significant_improve_step),
        "significant_best_fidelity": float(significant_best_fidelity),
        "checkpoint_output": str(checkpoint_output),
    }


def build_result_payload(
    *,
    created_at_utc: str,
    system_meta: dict[str, Any],
    args: argparse.Namespace,
    initial_rbm_checkpoint: Path,
    initial_vit_checkpoint: Path,
    stages: list[dict[str, Any]],
    current_checkpoint_paths: dict[str, str],
    current_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "experiment": "figure2_alternating_infidelity",
        "created_at_utc": created_at_utc,
        "updated_at_utc": now(),
        "status": "completed",
        "system": dict(system_meta),
        "config": {
            "start_with": args.start_with,
            "max_stages": int(args.max_stages),
            "stage_max_steps": int(args.stage_max_steps),
            "n_samples": int(args.n_samples),
            "target_n_samples": (
                int(args.n_samples)
                if args.target_n_samples is None
                else int(args.target_n_samples)
            ),
            "n_chains": int(args.n_chains),
            "n_discard_per_chain": int(args.n_discard_per_chain),
            "vit_learning_rate": float(args.vit_learning_rate),
            "rbm_learning_rate": float(args.rbm_learning_rate),
            "vit_diag_shift": float(args.vit_diag_shift),
            "rbm_diag_shift": float(args.rbm_diag_shift),
            "plateau_patience": int(args.plateau_patience),
            "plateau_min_fidelity_gain": float(args.plateau_min_fidelity_gain),
            "plateau_warmup_steps": int(args.plateau_warmup_steps),
            "exact_eval_every": int(args.exact_eval_every),
            "estimator": args.estimator,
            "use_ntk": bool(args.use_ntk),
            "chunk_size_bwd": (
                None if args.chunk_size_bwd is None else int(args.chunk_size_bwd)
            ),
            "resample_fraction": (
                None if args.resample_fraction is None else float(args.resample_fraction)
            ),
            "stage_fidelity_threshold": (
                None
                if args.stage_fidelity_threshold is None
                else float(args.stage_fidelity_threshold)
            ),
            "global_fidelity_threshold": (
                None
                if args.global_fidelity_threshold is None
                else float(args.global_fidelity_threshold)
            ),
            "seed": int(args.seed),
            "log_every": int(args.log_every),
        },
        "initial_checkpoints": {
            "rbm": str(initial_rbm_checkpoint),
            "vit": str(initial_vit_checkpoint),
        },
        "stages": stages,
        "final_checkpoints": dict(current_checkpoint_paths),
        "final_metrics": current_metrics,
    }


def main() -> None:
    args = parse_args()
    rbm_checkpoint_path = Path(args.rbm_checkpoint)
    vit_checkpoint_path = Path(args.vit_checkpoint)
    output_path = (
        Path(args.output)
        if args.output is not None
        else default_output_path(rbm_checkpoint_path, vit_checkpoint_path)
    )
    checkpoint_dir = Path(args.checkpoint_dir)
    target_n_samples = args.n_samples if args.target_n_samples is None else args.target_n_samples

    rbm_payload = load_pickle(rbm_checkpoint_path)
    vit_payload = load_pickle(vit_checkpoint_path)
    system_meta = validate_same_system(rbm_payload, vit_payload)
    hilbert, _, graph, _ = make_square_j1j2(
        L=int(system_meta["L"]),
        J1=float(system_meta["J1"]),
        J2=float(system_meta["J2"]),
    )
    square_L = int(system_meta["L"])
    j_couplings = (float(system_meta["J1"]), float(system_meta["J2"]))

    _, initial_rbm_desc, initial_rbm_state = make_state_from_variables(
        model_name=rbm_payload["model_name"],
        hilbert=hilbert,
        graph=graph,
        square_L=square_L,
        j_couplings=j_couplings,
        variables=rbm_payload["variables"],
    )
    _, initial_vit_desc, initial_vit_state = make_state_from_variables(
        model_name=vit_payload["model_name"],
        hilbert=hilbert,
        graph=graph,
        square_L=square_L,
        j_couplings=j_couplings,
        variables=vit_payload["variables"],
    )

    current_variables = {
        "rbm": copy_variables(rbm_payload["variables"]),
        "vit": copy_variables(vit_payload["variables"]),
    }
    current_meta = {
        "rbm": {
            "model_name": rbm_payload["model_name"],
            "model_desc": initial_rbm_desc,
        },
        "vit": {
            "model_name": vit_payload["model_name"],
            "model_desc": initial_vit_desc,
        },
    }
    current_checkpoint_paths = {
        "rbm": str(rbm_checkpoint_path),
        "vit": str(vit_checkpoint_path),
    }

    initial_pair_metrics = evaluate_state_match(initial_rbm_state, initial_vit_state)
    print("Figure 2 alternating sampled infidelity match")
    print(f"RBM checkpoint: {rbm_checkpoint_path}")
    print(f"ViT checkpoint: {vit_checkpoint_path}")
    print(
        "Initial pair fidelity: "
        f"{initial_pair_metrics['fidelity']:.9f} "
        f"(RBM={rbm_payload['model_name']}, ViT={vit_payload['model_name']})"
    )
    print(f"Output pickle: {output_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    created_at_utc = now()
    stages: list[dict[str, Any]] = []
    role_order = ["vit", "rbm"] if args.start_with == "vit" else ["rbm", "vit"]

    for stage_index in range(args.max_stages):
        role = role_order[stage_index % 2]
        target_role = "rbm" if role == "vit" else "vit"
        learning_rate = args.vit_learning_rate if role == "vit" else args.rbm_learning_rate
        diag_shift = args.vit_diag_shift if role == "vit" else args.rbm_diag_shift
        checkpoint_output = stage_checkpoint_path(
            checkpoint_dir,
            stage_index=stage_index,
            optimize_model_name=current_meta[role]["model_name"],
            target_model_name=current_meta[target_role]["model_name"],
        )

        target_model, _, target_state = make_state_from_variables(
            model_name=current_meta[target_role]["model_name"],
            hilbert=hilbert,
            graph=graph,
            square_L=square_L,
            j_couplings=j_couplings,
            variables=current_variables[target_role],
        )
        optimize_model, _, optimize_state = make_state_from_variables(
            model_name=current_meta[role]["model_name"],
            hilbert=hilbert,
            graph=graph,
            square_L=square_L,
            j_couplings=j_couplings,
            variables=current_variables[role],
        )
        start_pair_metrics = evaluate_state_match(target_state, optimize_state)
        print(
            f"Starting stage {stage_index} by optimizing {role} "
            f"({current_meta[role]['model_name']}) toward {current_meta[target_role]['model_name']} "
            f"from pair fidelity={start_pair_metrics['fidelity']:.9f}"
        )

        stage = run_alternating_stage(
            stage_index=stage_index,
            role=role,
            system_meta=system_meta,
            hilbert=hilbert,
            graph=graph,
            square_L=square_L,
            j_couplings=j_couplings,
            optimize_model_name=current_meta[role]["model_name"],
            target_model_name=current_meta[target_role]["model_name"],
            optimize_initial_variables=current_variables[role],
            target_variables=current_variables[target_role],
            n_samples=args.n_samples,
            target_n_samples=target_n_samples,
            n_chains=args.n_chains,
            n_discard_per_chain=args.n_discard_per_chain,
            learning_rate=learning_rate,
            diag_shift=diag_shift,
            max_steps=args.stage_max_steps,
            plateau_patience=args.plateau_patience,
            plateau_min_fidelity_gain=args.plateau_min_fidelity_gain,
            plateau_warmup_steps=args.plateau_warmup_steps,
            exact_eval_every=args.exact_eval_every,
            estimator=args.estimator,
            use_ntk=bool(args.use_ntk),
            chunk_size_bwd=args.chunk_size_bwd,
            resample_fraction=args.resample_fraction,
            fidelity_threshold=args.stage_fidelity_threshold,
            seed=args.seed + 10_000 * (stage_index + 1),
            log_every=args.log_every,
            checkpoint_output=checkpoint_output,
        )

        current_variables[role] = copy_variables(stage["best_variables"])
        current_checkpoint_paths[role] = str(checkpoint_output)

        current_rbm_model, _, current_rbm_state = make_state_from_variables(
            model_name=current_meta["rbm"]["model_name"],
            hilbert=hilbert,
            graph=graph,
            square_L=square_L,
            j_couplings=j_couplings,
            variables=current_variables["rbm"],
        )
        current_vit_model, _, current_vit_state = make_state_from_variables(
            model_name=current_meta["vit"]["model_name"],
            hilbert=hilbert,
            graph=graph,
            square_L=square_L,
            j_couplings=j_couplings,
            variables=current_variables["vit"],
        )
        current_pair_metrics = evaluate_state_match(current_rbm_state, current_vit_state)
        rbm_to_initial_rbm = evaluate_state_match(initial_rbm_state, current_rbm_state)
        rbm_to_initial_vit = evaluate_state_match(initial_vit_state, current_rbm_state)
        vit_to_initial_rbm = evaluate_state_match(initial_rbm_state, current_vit_state)
        vit_to_initial_vit = evaluate_state_match(initial_vit_state, current_vit_state)

        stage["start_pair_metrics"] = start_pair_metrics
        stage["post_stage_pair_metrics"] = current_pair_metrics
        stage["post_stage_anchor_metrics"] = {
            "rbm_to_initial_rbm": rbm_to_initial_rbm,
            "rbm_to_initial_vit": rbm_to_initial_vit,
            "vit_to_initial_rbm": vit_to_initial_rbm,
            "vit_to_initial_vit": vit_to_initial_vit,
        }
        stages.append(stage)

        print(
            f"Completed stage {stage_index}: "
            f"best pair fidelity={stage['best_metrics']['fidelity']:.9f}, "
            f"current pair fidelity={current_pair_metrics['fidelity']:.9f}, "
            f"stop_reason={stage['stop_reason']}"
        )
        print(
            "Anchor fidelities: "
            f"RBM->initial_RBM={rbm_to_initial_rbm['fidelity']:.9f}, "
            f"ViT->initial_RBM={vit_to_initial_rbm['fidelity']:.9f}"
        )

        result_payload = build_result_payload(
            created_at_utc=created_at_utc,
            system_meta=system_meta,
            args=args,
            initial_rbm_checkpoint=rbm_checkpoint_path,
            initial_vit_checkpoint=vit_checkpoint_path,
            stages=stages,
            current_checkpoint_paths=current_checkpoint_paths,
            current_metrics={
                "pair": current_pair_metrics,
                "rbm_to_initial_rbm": rbm_to_initial_rbm,
                "rbm_to_initial_vit": rbm_to_initial_vit,
                "vit_to_initial_rbm": vit_to_initial_rbm,
                "vit_to_initial_vit": vit_to_initial_vit,
            },
        )
        save_pickle(output_path, result_payload)

        if (
            args.global_fidelity_threshold is not None
            and current_pair_metrics["fidelity"] >= args.global_fidelity_threshold
        ):
            print(
                "Stopping alternating run because global pair fidelity reached "
                f"{args.global_fidelity_threshold:.6f}."
            )
            break

    print(
        "Final alternating summary: "
        f"stages={len(stages)}, "
        f"pair_fidelity={result_payload['final_metrics']['pair']['fidelity']:.9f}, "
        f"ViT->initial_RBM={result_payload['final_metrics']['vit_to_initial_rbm']['fidelity']:.9f}"
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
