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
import nqs_support_core as nkp
import numpy as np

from advanced_drivers.driver import InfidelityOptimizerNG
from sr_filtering_nqs.small_scale.common import copy_variables, make_model, make_square_j1j2, now, save_pickle


DEFAULT_OUTPUT_DIR = FIGURE2_DIR / "results"
DEFAULT_CHECKPOINT_DIR = FIGURE2_DIR / "checkpoints"


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def default_output_path(rbm_checkpoint: Path, vit_model: str) -> Path:
    stem = sanitize_name(rbm_checkpoint.stem)
    vit_label = sanitize_name(vit_model)
    return DEFAULT_OUTPUT_DIR / f"{stem}_to_{vit_label}_mcmc_match.pkl"


def default_vit_checkpoint_path(vit_model: str) -> Path:
    vit_label = sanitize_name(vit_model)
    return DEFAULT_CHECKPOINT_DIR / f"{vit_label}_mcmc_matched_best_from_checkpoint.pkl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MCMC infidelity minimization against a saved RBM checkpoint.",
    )
    parser.add_argument(
        "--rbm-checkpoint",
        type=str,
        required=True,
        help="Path to the saved RBM checkpoint payload used as the target state.",
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
        help="Optional saved ViT checkpoint payload used to warm-start the match.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=4096,
        help="MC samples drawn from the variational state per step.",
    )
    parser.add_argument(
        "--target-n-samples",
        type=int,
        default=None,
        help="MC samples drawn from the target state per step. Defaults to --n-samples.",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=16,
        help="Number of Metropolis chains for both target and variational states.",
    )
    parser.add_argument(
        "--n-discard-per-chain",
        type=int,
        default=4,
        help="Metropolis burn-in / decorrelation sweeps per chain.",
    )
    parser.add_argument(
        "--match-max-steps",
        type=int,
        default=1000,
        help="Maximum MCMC infidelity-minimization steps.",
    )
    parser.add_argument(
        "--match-learning-rate",
        type=float,
        default=1e-1,
        help="Learning rate for MCMC infidelity minimization.",
    )
    parser.add_argument(
        "--match-diag-shift",
        type=float,
        default=1e-4,
        help="Diagonal shift for MCMC infidelity minimization.",
    )
    parser.add_argument(
        "--match-fidelity-threshold",
        type=float,
        default=0.99,
        help="Stop early once the exact fidelity reaches this threshold.",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="cmc",
        choices=("cmc", "smc"),
        help="Infidelity estimator used by the MCMC driver.",
    )
    parser.add_argument(
        "--use-ntk",
        action="store_true",
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
        "--exact-eval-every",
        type=int,
        default=5,
        help="Evaluate exact full-Hilbert fidelity every this many steps (and at step 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to initialize the ViT sampler/state.",
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
        help="Path to the match-result pickle. Defaults to figure2/results/<rbm>_to_<vit>_mcmc_match.pkl.",
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
    selection_metric: str,
    metrics: dict,
) -> dict:
    payload = {
        "format_version": 1,
        "created_at_utc": now(),
        "system": dict(system_meta),
        "model_name": vit_model_name,
        "model_desc": vit_model_desc,
        "state_origin": "vit_infidelity_mcmc_checkpoint",
        "target_checkpoint_path": str(target_checkpoint_path),
        "step": int(step),
        "selection_metric": selection_metric,
        "variables": copy_variables(variables),
        "match_metrics": dict(metrics),
    }
    return payload


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
    best_selection_metric: str,
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
        "best_selection_metric": best_selection_metric,
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


def normalized_state_vector(vstate: nk.vqs.FullSumState) -> np.ndarray:
    psi = np.asarray(vstate.to_array(normalize=False), dtype=np.complex128)
    norm = np.linalg.norm(psi)
    if norm == 0.0:
        raise ValueError("Encountered zero-norm state while computing exact fidelity.")
    return psi / norm


def exact_overlap_metrics(
    *,
    target_state: nk.vqs.FullSumState,
    variational_state: nk.vqs.FullSumState,
) -> dict[str, float]:
    psi_target = normalized_state_vector(target_state)
    psi_var = normalized_state_vector(variational_state)
    overlap = np.vdot(psi_target, psi_var)
    fidelity = float(np.clip(np.abs(overlap) ** 2, 0.0, 1.0))
    infidelity_op = nkp.infidelity.InfidelityOperator(target_state)
    stats = variational_state.expect(infidelity_op)
    infidelity_operator = float(max(0.0, np.real(stats.mean)))
    return {
        "fidelity": fidelity,
        "infidelity_direct": float(max(0.0, 1.0 - fidelity)),
        "infidelity_operator": infidelity_operator,
        "infidelity_abs_error": float(abs((1.0 - fidelity) - infidelity_operator)),
        "overlap_real": float(np.real(overlap)),
        "overlap_imag": float(np.imag(overlap)),
        "overlap_abs": float(np.abs(overlap)),
    }


def should_exact_eval(step: int, exact_eval_every: int) -> bool:
    return step == 0 or (exact_eval_every > 0 and step % exact_eval_every == 0)


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
    target_n_samples = args.n_samples if args.target_n_samples is None else args.target_n_samples

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
    vit_model, vit_desc = make_model(
        args.vit_model,
        hilbert=hilbert,
        graph=graph,
        square_L=int(system_meta["L"]),
        j_couplings=(float(system_meta["J1"]), float(system_meta["J2"])),
    )

    target_sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=args.n_chains)
    variational_sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=args.n_chains)
    target_state = nk.vqs.MCState(
        target_sampler,
        model=rbm_model,
        n_samples=target_n_samples,
        n_discard_per_chain=args.n_discard_per_chain,
        seed=args.seed,
    )
    variational_state = nk.vqs.MCState(
        variational_sampler,
        model=vit_model,
        n_samples=args.n_samples,
        n_discard_per_chain=args.n_discard_per_chain,
        seed=args.seed + 1,
    )
    target_state.variables = copy_variables(rbm_payload["variables"])

    if args.initial_vit_checkpoint is not None:
        initial_vit_path = Path(args.initial_vit_checkpoint)
        initial_vit_payload = load_pickle(initial_vit_path)
        if initial_vit_payload["model_name"] != args.vit_model:
            raise ValueError(
                f"Initial ViT checkpoint model_name={initial_vit_payload['model_name']} "
                f"does not match requested vit-model={args.vit_model}."
            )
        variational_state.variables = copy_variables(initial_vit_payload["variables"])

    exact_target_state = nk.vqs.FullSumState(
        hilbert=hilbert,
        model=rbm_model,
        variables=copy_variables(rbm_payload["variables"]),
    )

    optimizer = nk.optimizer.Sgd(learning_rate=args.match_learning_rate)
    driver = InfidelityOptimizerNG(
        target_state=target_state,
        optimizer=optimizer,
        variational_state=variational_state,
        diag_shift=args.match_diag_shift,
        estimator=args.estimator,
        use_ntk=bool(args.use_ntk),
        chunk_size_bwd=args.chunk_size_bwd,
        resample_fraction=args.resample_fraction,
    )

    print("Figure 2 checkpoint-based MCMC infidelity match")
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
    best_metrics: dict | None = None
    best_step_time_s = None
    best_selection_metric = "exact_fidelity"
    stop_reason = "max_steps_reached"

    match_config = {
        "match_learning_rate": float(args.match_learning_rate),
        "match_diag_shift": float(args.match_diag_shift),
        "match_max_steps": int(args.match_max_steps),
        "match_fidelity_threshold": float(args.match_fidelity_threshold),
        "n_samples": int(args.n_samples),
        "target_n_samples": int(target_n_samples),
        "n_chains": int(args.n_chains),
        "n_discard_per_chain": int(args.n_discard_per_chain),
        "estimator": str(args.estimator),
        "use_ntk": bool(args.use_ntk),
        "chunk_size_bwd": None if args.chunk_size_bwd is None else int(args.chunk_size_bwd),
        "resample_fraction": None if args.resample_fraction is None else float(args.resample_fraction),
        "exact_eval_every": int(args.exact_eval_every),
        "seed": int(args.seed),
        "save_every": int(args.save_every),
        "log_every": int(args.log_every),
        "initial_vit_checkpoint": None if args.initial_vit_checkpoint is None else str(args.initial_vit_checkpoint),
    }

    def evaluate_exact_metrics() -> dict[str, float]:
        exact_var_state = nk.vqs.FullSumState(
            hilbert=hilbert,
            model=vit_model,
            variables=copy_variables(variational_state.variables),
        )
        return exact_overlap_metrics(
            target_state=exact_target_state,
            variational_state=exact_var_state,
        )

    def maybe_persist(entry: dict, *, status: str) -> None:
        assert best_metrics is not None
        checkpoint_payload = build_vit_checkpoint_payload(
            system_meta=system_meta,
            vit_model_name=args.vit_model,
            vit_model_desc=vit_desc,
            target_checkpoint_path=rbm_checkpoint_path,
            step=int(best_metrics["step"]),
            variables=best_variables,
            selection_metric=best_selection_metric,
            metrics=best_metrics,
        )
        result_payload = build_result_payload(
            status=status,
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
            best_selection_metric=best_selection_metric,
            final_metrics=entry,
            final_step=int(entry["step"]),
            steps_completed=max(0, int(entry["step"])),
            stop_reason="running" if status == "running" else stop_reason,
            success=bool(best_metrics.get("fidelity", 0.0) >= args.match_fidelity_threshold),
            elapsed_wall_s=time.perf_counter() - start_wall,
            vit_checkpoint_output=vit_checkpoint_output,
        )
        save_running_state(
            output_path=output_path,
            vit_checkpoint_output=vit_checkpoint_output,
            result_payload=result_payload,
            checkpoint_payload=checkpoint_payload,
        )

    for step in range(args.match_max_steps + 1):
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
        if should_exact_eval(step, args.exact_eval_every):
            exact_start = time.perf_counter()
            exact_metrics = evaluate_exact_metrics()
            entry.update(exact_metrics)
            entry["exact_eval_wall_s"] = float(time.perf_counter() - exact_start)

        history.append(entry)

        improved = False
        selection_metric = best_selection_metric
        if exact_metrics is not None:
            selection_metric = "exact_fidelity"
            if best_metrics is None or exact_metrics["fidelity"] > float(best_metrics.get("fidelity", -1.0)):
                improved = True
        elif best_metrics is None:
            selection_metric = "mc_infidelity_mean"
            improved = True
        elif "fidelity" not in best_metrics:
            selection_metric = "mc_infidelity_mean"
            if entry["mc_infidelity_mean"] < float(best_metrics.get("mc_infidelity_mean", np.inf)):
                improved = True

        if improved:
            best_metrics = dict(entry)
            best_variables = copy_variables(variational_state.variables)
            best_step_time_s = float(iter_elapsed)
            best_selection_metric = selection_metric
            maybe_persist(entry, status="running")

        if step == 0 or step % args.log_every == 0:
            parts = [
                f"step={step:04d}",
                f"mc_infidelity={entry['mc_infidelity_mean']:.9f}",
                f"best_selection={best_selection_metric}",
                f"step_wall_s={iter_elapsed:.3f}",
            ]
            if "fidelity" in entry:
                parts.append(f"exact_fidelity={entry['fidelity']:.9f}")
            if best_metrics is not None and "fidelity" in best_metrics:
                parts.append(f"best_exact_fidelity={best_metrics['fidelity']:.9f}")
            print(" ".join(parts))

        if step > 0 and step % args.save_every == 0 and best_metrics is not None:
            maybe_persist(entry, status="running")

        if exact_metrics is not None and exact_metrics["fidelity"] >= args.match_fidelity_threshold:
            stop_reason = "threshold_reached"
            break

    assert best_metrics is not None
    final_metrics = history[-1]
    success = bool(float(best_metrics.get("fidelity", 0.0)) >= args.match_fidelity_threshold)
    if stop_reason != "threshold_reached":
        stop_reason = "max_steps_reached"

    final_checkpoint_payload = build_vit_checkpoint_payload(
        system_meta=system_meta,
        vit_model_name=args.vit_model,
        vit_model_desc=vit_desc,
        target_checkpoint_path=rbm_checkpoint_path,
        step=int(best_metrics["step"]),
        variables=best_variables,
        selection_metric=best_selection_metric,
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
        best_selection_metric=best_selection_metric,
        final_metrics=final_metrics,
        final_step=int(final_metrics["step"]),
        steps_completed=max(0, int(final_metrics["step"])),
        stop_reason=stop_reason,
        success=success,
        elapsed_wall_s=time.perf_counter() - start_wall,
        vit_checkpoint_output=vit_checkpoint_output,
    )
    save_running_state(
        output_path=output_path,
        vit_checkpoint_output=vit_checkpoint_output,
        result_payload=final_result_payload,
        checkpoint_payload=final_checkpoint_payload,
    )
    print(
        "Final MCMC match summary: "
        f"best_step={int(best_metrics['step'])}, "
        f"best_selection={best_selection_metric}, "
        f"best_exact_fidelity={float(best_metrics.get('fidelity', float('nan'))):.9f}, "
        f"stop_reason={stop_reason}, "
        f"elapsed_wall_s={time.perf_counter() - start_wall:.1f}"
    )


if __name__ == "__main__":
    main()
