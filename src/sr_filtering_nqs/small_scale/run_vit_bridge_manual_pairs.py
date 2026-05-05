from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import netket as nk

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE1_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sr_filtering_nqs.small_scale.common import (
    compute_exact_regression_diagnostics,
    copy_variables,
    make_model,
    make_square_j1j2,
    now,
    precompute_centered_feature_matrix,
    run_trials_for_checkpoint,
    save_pickle,
)
from sr_filtering_nqs.small_scale.plot_figure1 import plot_lambda_curves, plot_main_figure, plot_width_sweep


DEFAULT_MODELS = ["RBM_a1", "ViT_d2_m24_h4_e4"]
DEFAULT_RESULTS_DIR = FIGURE1_DIR / "results" / "manual_pairs"
DEFAULT_FIGURES_DIR = FIGURE1_DIR / "figures_manual_pairs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exact 4x4 J1-J2 bridge once and render figure outputs for explicit "
            "RBM/ViT checkpoint-step pairs."
        ),
    )
    parser.add_argument("--L", type=int, default=4, help="Square lattice linear size.")
    parser.add_argument("--J1", type=float, default=1.0, help="Nearest-neighbor coupling.")
    parser.add_argument("--J2", type=float, default=0.5, help="Next-nearest-neighbor coupling.")
    parser.add_argument(
        "--models",
        nargs=2,
        default=DEFAULT_MODELS,
        help=f"Exactly two models, default: {DEFAULT_MODELS}",
    )
    parser.add_argument("--ns", type=int, default=2048, help="Fixed exact sample count.")
    parser.add_argument("--train-samples", type=int, default=4096, help="MCState samples during training.")
    parser.add_argument("--train-steps", type=int, default=200, help="Maximum SR training steps.")
    parser.add_argument("--diag-every", type=int, default=10, help="Exact diagnostic interval.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="SR optimizer learning rate.")
    parser.add_argument("--train-diag-shift", type=float, default=1e-3, help="Training regularization.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of exact Born trials.")
    parser.add_argument("--n-lambdas", type=int, default=17, help="Number of lambda grid points.")
    parser.add_argument("--lambda-min-exp", type=float, default=-9.0, help="Minimum lambda exponent.")
    parser.add_argument("--lambda-max-exp", type=float, default=-1.0, help="Maximum lambda exponent.")
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=None,
        help="Optional explicit lambda grid.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--jac-chunk-size", type=int, default=4096, help="Chunk size for Jacobian evaluations.")
    parser.add_argument("--apply-chunk-size", type=int, default=4096, help="Chunk size for forward/JVP evaluations.")
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help=(
            "Checkpoint pair as 'rbm_step:vit_step' or 'label:rbm_step:vit_step'. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for generated result pickles.",
    )
    parser.add_argument(
        "--figures-dir",
        type=str,
        default=str(DEFAULT_FIGURES_DIR),
        help="Directory for generated figure folders.",
    )
    return parser.parse_args()


def parse_pair_spec(spec: str) -> tuple[str, int, int]:
    parts = spec.split(":")
    if len(parts) == 2:
        rbm_step = int(parts[0])
        vit_step = int(parts[1])
        label = f"rbm{rbm_step}_vit{vit_step}"
        return label, rbm_step, vit_step
    if len(parts) == 3:
        label = parts[0]
        rbm_step = int(parts[1])
        vit_step = int(parts[2])
        return label, rbm_step, vit_step
    raise ValueError(f"Invalid --pair spec '{spec}'. Use 'rbm_step:vit_step' or 'label:rbm_step:vit_step'.")


def train_models(
    *,
    model_specs: list[str],
    hilbert,
    hamiltonian,
    graph,
    all_states: np.ndarray,
    square_L: int,
    j_couplings: tuple[float, float],
    train_samples: int,
    train_steps: int,
    diag_every: int,
    learning_rate: float,
    train_diag_shift: float,
    seed: int,
    requested_steps_by_model: dict[str, set[int]],
) -> list[dict]:
    trained_models = []
    for model_idx, model_name in enumerate(model_specs):
        model, model_desc = make_model(
            model_name,
            hilbert=hilbert,
            graph=graph,
            square_L=square_L,
            j_couplings=j_couplings,
        )
        print(f"\n{'=' * 72}")
        print(f"Model {model_name} ({model_desc})")
        print(f"{'=' * 72}")
        sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=16)
        vstate = nk.vqs.MCState(sampler, model=model, n_samples=train_samples, seed=seed + 10_000 * model_idx)
        optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)
        sr = nk.optimizer.SR(
            qgt=nk.optimizer.qgt.QGTJacobianDense,
            solver=nk.optimizer.solver.cholesky,
            diag_shift=train_diag_shift,
        )
        driver = nk.driver.VMC(
            hamiltonian,
            optimizer,
            variational_state=vstate,
            preconditioner=sr,
        )
        requested_steps = requested_steps_by_model[model_name]
        checkpoint_runtime_by_step = {}
        if 0 in requested_steps:
            checkpoint_runtime_by_step[0] = copy_variables(vstate.variables)
        for step in range(diag_every, train_steps + 1, diag_every):
            driver.run(n_iter=diag_every)
            if step in requested_steps:
                checkpoint_runtime_by_step[step] = copy_variables(vstate.variables)

        missing = sorted(requested_steps.difference(checkpoint_runtime_by_step))
        if missing:
            raise ValueError(f"Missing requested checkpoints for {model_name}: {missing}")
        trained_models.append(
            {
                "model_name": model_name,
                "model_desc": model_desc,
                "model": model,
                "checkpoint_runtime_by_step": checkpoint_runtime_by_step,
                "trial_seed": seed + 100_000 * (model_idx + 1),
            }
        )
    return trained_models


def materialize_model_result(
    *,
    item: dict,
    selected_step: int,
    diag_cache: dict[int, dict],
    all_states: np.ndarray,
    hamiltonian,
    lambda_grid: np.ndarray,
    n_samples: int,
    n_trials: int,
    jac_chunk_size: int,
    apply_chunk_size: int,
) -> dict:
    selected = dict(diag_cache[selected_step])
    feature_matrix_centered = precompute_centered_feature_matrix(
        item["model"],
        item["checkpoint_runtime_by_step"][selected_step],
        all_states,
        np.asarray(selected["feature_mean"], dtype=np.float64),
        jac_chunk_size=jac_chunk_size,
    )
    trials, summary, representative = run_trials_for_checkpoint(
        model=item["model"],
        variables=item["checkpoint_runtime_by_step"][selected_step],
        all_states=all_states,
        exact_diag=selected,
        lambda_grid=lambda_grid,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=item["trial_seed"] + 1_000 * selected_step,
        jac_chunk_size=jac_chunk_size,
        apply_chunk_size=apply_chunk_size,
        feature_matrix_centered=feature_matrix_centered,
    )
    return {
        "model_name": item["model_name"],
        "model_desc": item["model_desc"],
        "P_real": int(selected["p_real"]),
        "P_over_Ns": float(selected["p_real"] / n_samples),
        "checkpoint_step": selected_step,
        "checkpoint_energy": float(selected["energy"]),
        "noise_fraction": float(selected["noise_fraction"]),
        "sigma_gap_sq": float(selected["sigma_gap_sq"]),
        "target_var": float(selected["target_var"]),
        "selected_index": int(selected_step),
        "selected_reason": "manual_step",
        "selection_metric": "step",
        "selection_metric_value": float(selected_step),
        "checkpoint_summaries": [
            {
                "step": int(step),
                "energy": float(diag["energy"]),
                "target_var": float(diag["target_var"]),
                "sigma_gap_sq": float(diag["sigma_gap_sq"]),
                "noise_fraction": float(diag["noise_fraction"]),
                "timing_seconds": float(diag["timing_seconds"]),
            }
            for step, diag in sorted(diag_cache.items())
        ],
        "selected_checkpoint": selected,
        "trials": trials,
        "summary": summary,
        "representative": representative,
    }


def main() -> None:
    args = parse_args()
    model_specs = list(args.models)
    lambda_grid = (
        np.asarray(args.lambda_values, dtype=np.float64)
        if args.lambda_values is not None
        else np.logspace(args.lambda_min_exp, args.lambda_max_exp, args.n_lambdas)
    )
    pair_specs = [parse_pair_spec(spec) for spec in args.pair]
    requested_steps_by_model = {
        model_specs[0]: {rbm_step for _, rbm_step, _ in pair_specs},
        model_specs[1]: {vit_step for _, _, vit_step in pair_specs},
    }
    requested_steps_by_model[model_specs[0]].add(0)
    requested_steps_by_model[model_specs[1]].add(0)

    hilbert, hamiltonian, graph, system_meta = make_square_j1j2(L=args.L, J1=args.J1, J2=args.J2)
    all_states = np.asarray(hilbert.all_states())
    trained_models = train_models(
        model_specs=model_specs,
        hilbert=hilbert,
        hamiltonian=hamiltonian,
        graph=graph,
        all_states=all_states,
        square_L=args.L,
        j_couplings=(args.J1, args.J2),
        train_samples=args.train_samples,
        train_steps=args.train_steps,
        diag_every=args.diag_every,
        learning_rate=args.learning_rate,
        train_diag_shift=args.train_diag_shift,
        seed=args.seed,
        requested_steps_by_model=requested_steps_by_model,
    )

    model_lookup = {item["model_name"]: item for item in trained_models}
    rbm_name, vit_name = model_specs
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    diag_cache_by_model: dict[str, dict[int, dict]] = {}
    for model_name, item in model_lookup.items():
        diag_cache = {}
        for step in sorted(requested_steps_by_model[model_name]):
            variables = item["checkpoint_runtime_by_step"][step]
            t0 = time.time()
            diag = compute_exact_regression_diagnostics(
                item["model"],
                variables,
                hamiltonian,
                all_states,
                jac_chunk_size=args.jac_chunk_size,
                apply_chunk_size=args.apply_chunk_size,
                store_arrays=True,
                store_matrix=False,
            )
            diag["timing_seconds"] = float(time.time() - t0)
            diag["step"] = int(step)
            diag["selected_reason"] = "manual_step"
            diag["selection_metric"] = "step"
            diag["selection_metric_value"] = float(step)
            diag_cache[step] = diag
        diag_cache_by_model[model_name] = diag_cache

    for label, rbm_step, vit_step in pair_specs:
        print(f"\nRendering pair {label}: {rbm_name}@{rbm_step} vs {vit_name}@{vit_step}")

        results = {
            "format_version": 1,
            "experiment": "figure1_vit_bridge_manual_pairs",
            "created_at_utc": now(),
            "system": dict(system_meta),
            "ns": int(args.ns),
            "train_config": {
                "train_samples": int(args.train_samples),
                "train_steps": int(args.train_steps),
                "diag_every": int(args.diag_every),
                "learning_rate": float(args.learning_rate),
                "train_diag_shift": float(args.train_diag_shift),
                "seed": int(args.seed),
                "jac_chunk_size": int(args.jac_chunk_size),
                "apply_chunk_size": int(args.apply_chunk_size),
                "store_checkpoint_arrays": False,
            },
            "selection": {
                "mode": "manual_steps",
                "metric": "step",
                "pair_label": label,
                "requested_steps": {
                    rbm_name: int(rbm_step),
                    vit_name: int(vit_step),
                },
            },
            "lambda_grid": np.asarray(lambda_grid, dtype=np.float64),
            "models": [
                materialize_model_result(
                    item=model_lookup[rbm_name],
                    selected_step=rbm_step,
                    diag_cache=diag_cache_by_model[rbm_name],
                    all_states=all_states,
                    hamiltonian=hamiltonian,
                    lambda_grid=lambda_grid,
                    n_samples=args.ns,
                    n_trials=args.n_trials,
                    jac_chunk_size=args.jac_chunk_size,
                    apply_chunk_size=args.apply_chunk_size,
                ),
                materialize_model_result(
                    item=model_lookup[vit_name],
                    selected_step=vit_step,
                    diag_cache=diag_cache_by_model[vit_name],
                    all_states=all_states,
                    hamiltonian=hamiltonian,
                    lambda_grid=lambda_grid,
                    n_samples=args.ns,
                    n_trials=args.n_trials,
                    jac_chunk_size=args.jac_chunk_size,
                    apply_chunk_size=args.apply_chunk_size,
                ),
            ],
        }

        output_pickle = results_dir / f"{label}.pkl"
        output_figures = figures_dir / label
        save_pickle(output_pickle, results)
        plot_main_figure(results, output_figures / "figure1_main.pdf")
        plot_lambda_curves(results, output_figures / "figure1_lambda_curves.pdf")
        plot_width_sweep(results, output_figures / "figure1_width_sweep.pdf")

        print(f"Saved result pickle to {output_pickle}")
        print(f"Saved figures to {output_figures}")
        for model in results["models"]:
            print(
                f"  {model['model_name']} step {model['checkpoint_step']}: "
                f"sigma_gap_sq={model['sigma_gap_sq']:.6f}, "
                f"noise_fraction={model['noise_fraction']:.6f}"
            )


if __name__ == "__main__":
    main()
