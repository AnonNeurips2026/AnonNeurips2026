from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURE2_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURE2_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sr_filtering_nqs.small_scale.plot_figure1 import (
    compute_scatter_limits,
    model_noise_lines,
    representative_lambda_info,
    scatter_panel,
)

MODEL_COLORS = {"rbm": "steelblue", "vit": "crimson"}


def load_results(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def model_pair(data: dict) -> tuple[dict, dict]:
    if len(data["models"]) != 2:
        raise ValueError("Figure 2 plotting expects exactly two models.")
    rbm = next((item for item in data["models"] if item["state_origin"] == "rbm_vmc_checkpoint"), None)
    vit = next((item for item in data["models"] if item["state_origin"] == "vit_infidelity_checkpoint"), None)
    if rbm is None or vit is None:
        raise ValueError("Could not identify RBM and ViT matched-state entries.")
    return rbm, vit


def format_model_label(model: dict) -> str:
    return f"{model['model_desc']}\n$P/N_s={model['P_over_Ns']:.2f}$"


def plot_match_history(ax, data: dict) -> None:
    history = data["matching"]["history"]
    steps = np.asarray([item["step"] for item in history], dtype=np.int64)
    infidelity = np.asarray([item["infidelity_direct"] for item in history], dtype=np.float64)
    best_idx = int(data["matching"]["best_history_index"])

    ax.plot(steps, np.maximum(infidelity, 1e-18), color=MODEL_COLORS["vit"], lw=2)
    ax.scatter(
        [steps[best_idx]],
        [max(infidelity[best_idx], 1e-18)],
        color="black",
        s=28,
        zorder=3,
        label="Best match",
    )
    ax.set_yscale("log")
    ax.set_xlabel("ViT match step")
    ax.set_ylabel("Exact infidelity $1-F$")
    ax.set_title("State matching")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8, loc="upper right")
    ax.text(
        0.02,
        0.98,
        (
            rf"init={data['matching']['initial_infidelity']:.2e}" "\n"
            rf"best={data['matching']['best_infidelity']:.2e}" "\n"
            rf"|direct-op|$_{{\max}}$={data['matching']['consistency_max_abs_error']:.1e}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
    )


def plot_noise_comparison(ax, rbm: dict, vit: dict) -> None:
    models = [rbm, vit]
    labels = [format_model_label(model) for model in models]
    sigma_gap = np.asarray([model["sigma_gap_sq"] for model in models], dtype=np.float64)
    noise_fraction = np.asarray([model["noise_fraction"] for model in models], dtype=np.float64)
    colors = [MODEL_COLORS["rbm"], MODEL_COLORS["vit"]]
    x = np.arange(len(models))

    bars = ax.bar(x, sigma_gap, color=colors, alpha=0.85, width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\sigma_{\mathrm{gap}}^2$")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_title("Exact misspecification at the matched state")
    ax.grid(True, axis="y", which="both", alpha=0.2)

    twin = ax.twinx()
    twin.plot(x, noise_fraction, color="0.2", marker="o", lw=1.6)
    twin.set_ylabel(r"$\sigma_{\mathrm{gap}}^2 / \mathrm{Var}_\pi[H_{\mathrm{loc}}]$")
    twin.set_ylim(0.0, max(0.35, float(np.max(noise_fraction)) * 1.25))

    for idx, (bar, noise_value) in enumerate(zip(bars, noise_fraction)):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() * 1.12,
            rf"{sigma_gap[idx]:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
        twin.text(
            idx,
            noise_value + 0.015,
            rf"{noise_value:.3f}",
            color="0.2",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_generalization(ax, rbm: dict, vit: dict) -> None:
    for model, color in ((rbm, MODEL_COLORS["rbm"]), (vit, MODEL_COLORS["vit"])):
        lambdas = np.asarray(model["summary"]["lambda_grid"], dtype=np.float64)
        mean = np.asarray(model["summary"]["r_pop_ideal_mean"], dtype=np.float64)
        std = np.asarray(model["summary"]["r_pop_ideal_std"], dtype=np.float64)
        ax.plot(lambdas, mean, color=color, lw=2, label=model["model_desc"])
        ax.fill_between(
            lambdas,
            np.maximum(mean - std, 1e-20),
            mean + std,
            color=color,
            alpha=0.18,
        )
        best_lambda = float(model["summary"]["lambda_best"])
        best_value = float(mean[int(model["summary"]["lambda_best_idx"])])
        ax.scatter([best_lambda], [best_value], color=color, s=24, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Diagonal shift $\lambda$")
    ax.set_ylabel(r"One-step excess risk $r_{\mathrm{pop}}^I$")
    ax.set_title("One-step SR generalization")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=8)


def plot_summary_figure(data: dict, output_path: Path) -> None:
    rbm, vit = model_pair(data)
    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.15, 1.45])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    plot_match_history(ax0, data)
    plot_noise_comparison(ax1, rbm, vit)
    plot_generalization(ax2, rbm, vit)

    system = data["system"]
    fig.suptitle(
        (
            "Figure 2: matched-state exact RBM vs ViT"
            "\n"
            rf"{system['name']} $L={system['L']}$, $N_s={data['ns']}$, "
            rf"matched infidelity={data['matching']['matched_infidelity_direct']:.2e}"
        ),
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure1_style_subtitle(
    model: dict,
    *,
    rep: dict,
    summary: dict,
    lambda_idx: int,
    include_fidelity: bool,
    fidelity: float | None,
) -> str:
    lines = [
        rf"$P/N_s$={model['P_over_Ns']:.2f}",
        model_noise_lines(model),
        rf"$r_{{train}}^H$={rep['metrics']['r_train_hloc'][lambda_idx]:.1e}",
        rf"$r_{{pop}}^H$={rep['metrics']['r_pop_hloc'][lambda_idx]:.1e}",
        rf"$r_{{pop}}^I$={rep['metrics']['r_pop_ideal'][lambda_idx]:.1e}",
    ]
    if include_fidelity and fidelity is not None:
        lines.append(rf"$F$={fidelity:.6f}")
    return "\n".join(lines)


def plot_figure1_style_main(data: dict, output_path: Path) -> None:
    rbm, vit = model_pair(data)
    models_to_plot = (rbm, vit)
    model_panels = []

    for model in models_to_plot:
        selected = model["selected_checkpoint"]
        rep = model["representative"]
        summary = model["summary"]
        lam_small, lam_second, second_kind = representative_lambda_info(model)
        second_idx_key = f"lambda_{second_kind}_idx"
        pred_second_key = f"pred_{second_kind}"
        model_panels.append(
            {
                "model": model,
                "selected": selected,
                "rep": rep,
                "summary": summary,
                "lam_small": lam_small,
                "lam_second": lam_second,
                "second_idx_key": second_idx_key,
                "pred_second_key": pred_second_key,
                "target_hloc": np.asarray(selected["target_hloc"], dtype=np.float64),
                "target_ideal": np.asarray(selected["target_ideal"], dtype=np.float64),
                "pi": np.asarray(selected["pi"], dtype=np.float64),
                "train_indices": np.asarray(rep["train_indices"], dtype=np.int64),
                "pred_small": np.asarray(rep["predictions"]["pred_small"], dtype=np.float64),
                "pred_second": np.asarray(rep["predictions"][pred_second_key], dtype=np.float64),
            }
        )

    hloc_limits = compute_scatter_limits(
        [(item["target_hloc"], item["pred_small"], item["pi"]) for item in model_panels]
    )
    ideal_limits = compute_scatter_limits(
        [
            (item["target_ideal"], item["pred_small"], item["pi"])
            for item in model_panels
        ]
        + [
            (item["target_ideal"], item["pred_second"], item["pi"])
            for item in model_panels
        ]
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    matched_fidelity = float(data["matching"]["matched_fidelity"])

    for row, item in enumerate(model_panels):
        model = item["model"]
        rep = item["rep"]
        summary = item["summary"]
        lam_small = item["lam_small"]
        lam_second = item["lam_second"]
        second_idx_key = item["second_idx_key"]
        target_hloc = item["target_hloc"]
        target_ideal = item["target_ideal"]
        pi = item["pi"]
        train_indices = item["train_indices"]
        pred_small = item["pred_small"]
        pred_second = item["pred_second"]

        include_fidelity = model["state_origin"] == "vit_infidelity_checkpoint"
        subtitle_small = figure1_style_subtitle(
            model,
            rep=rep,
            summary=summary,
            lambda_idx=int(summary["lambda_small_idx"]),
            include_fidelity=include_fidelity,
            fidelity=matched_fidelity,
        )
        subtitle_second = figure1_style_subtitle(
            model,
            rep=rep,
            summary=summary,
            lambda_idx=int(summary[second_idx_key]),
            include_fidelity=include_fidelity,
            fidelity=matched_fidelity,
        )

        scatter_panel(
            axes[row, 0],
            target=target_hloc,
            prediction=pred_small,
            train_indices=train_indices,
            pi=pi,
            title=rf"{model['model_desc']}: $H_{{loc,c}}$, $\lambda={lam_small}$",
            subtitle=subtitle_small,
            show_legend=(row == 0),
            limits=hloc_limits,
        )
        scatter_panel(
            axes[row, 1],
            target=target_ideal,
            prediction=pred_small,
            train_indices=train_indices,
            pi=pi,
            title=rf"{model['model_desc']}: $O_c\delta^*$, $\lambda={lam_small}$",
            subtitle=subtitle_small,
            limits=ideal_limits,
        )
        scatter_panel(
            axes[row, 2],
            target=target_ideal,
            prediction=pred_second,
            train_indices=train_indices,
            pi=pi,
            title=rf"{model['model_desc']}: $O_c\delta^*$, $\lambda={lam_second}$",
            subtitle=subtitle_second,
            limits=ideal_limits,
        )
        axes[row, 0].set_ylabel(r"Prediction $O_c(x)^T \hat{\delta}$")

    for col in range(3):
        axes[-1, col].set_xlabel("Target")

    system = data["system"]
    fig.suptitle(
        (
            "Figure 2 main panel: fixed-checkpoint matched-state exact study"
            "\n"
            rf"{system['name']} $L={system['L']}$, $N_s={data['ns']}$, "
            rf"matched fidelity={matched_fidelity:.6f}"
        ),
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_curves(data: dict, output_path: Path) -> None:
    rbm, vit = model_pair(data)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)

    for ax, model, color in (
        (axes[0], rbm, MODEL_COLORS["rbm"]),
        (axes[1], vit, MODEL_COLORS["vit"]),
    ):
        lambdas = np.asarray(model["summary"]["lambda_grid"], dtype=np.float64)
        for key, line_color, label in (
            ("r_train_hloc", "0.35", r"$r_{\mathrm{train}}^H$"),
            ("r_pop_hloc", "black", r"$r_{\mathrm{pop}}^H$"),
            ("r_pop_ideal", color, r"$r_{\mathrm{pop}}^I$"),
        ):
            mean = np.asarray(model["summary"][f"{key}_mean"], dtype=np.float64)
            std = np.asarray(model["summary"][f"{key}_std"], dtype=np.float64)
            ax.plot(lambdas, mean, color=line_color, lw=2, label=label)
            ax.fill_between(
                lambdas,
                np.maximum(mean - std, 1e-20),
                mean + std,
                color=line_color,
                alpha=0.15,
            )

        ax.axvline(float(model["summary"]["lambda_best"]), color="0.55", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Diagonal shift $\lambda$")
        ax.set_title(
            (
                f"{model['model_desc']}\n"
                rf"$\sigma_{{gap}}^2={model['sigma_gap_sq']:.2e}$, "
                rf"$\sigma_{{gap}}^2/\mathrm{{Var}}={model['noise_fraction']:.3f}$"
            ),
            fontsize=10,
        )
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Residual")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 2 matched-state experiment outputs.")
    parser.add_argument("--input", type=str, required=True, help="Input result pickle.")
    parser.add_argument(
        "--style",
        choices=("figure1_main", "summary"),
        default="figure1_main",
        help="Main figure style. `figure1_main` is the paper-run default.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PDFs. Defaults to figure2/figures/<input_stem>/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else FIGURE2_DIR / "figures" / input_path.stem
    )
    data = load_results(input_path)
    if args.style == "figure1_main":
        plot_figure1_style_main(data, output_dir / "figure2_main.pdf")
    else:
        plot_summary_figure(data, output_dir / "figure2_main.pdf")
    plot_lambda_curves(data, output_dir / "figure2_lambda_curves.pdf")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
