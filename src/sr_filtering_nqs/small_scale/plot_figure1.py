from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURE1_DIR = Path(__file__).resolve().parent
EXCESS_RISK_LABEL = r"$\|\hat{\delta}_\lambda - \delta^*\|_S^2$"


def load_results(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def model_by_name(data: dict, name: str) -> dict | None:
    for model in data["models"]:
        if model["model_name"] == name:
            return model
    return None


def select_under_over_models(data: dict) -> tuple[dict, dict]:
    if len(data["models"]) < 2:
        raise ValueError("Need at least two models to plot under/over comparisons.")

    alpha2 = model_by_name(data, "RBM_a2")
    alpha8 = model_by_name(data, "RBM_a8")
    if alpha2 is not None and alpha8 is not None:
        return alpha2, alpha8

    under = min(data["models"], key=lambda item: abs(item["P_over_Ns"] - 0.5))
    over = max(data["models"], key=lambda item: item["P_over_Ns"])
    return under, over


def scatter_panel(
    ax,
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    train_indices: np.ndarray,
    pi: np.ndarray,
    title: str,
    subtitle: str | None = None,
    show_legend: bool = False,
    limits: tuple[float, float] | None = None,
) -> None:
    n_states = len(target)
    train_mask = np.zeros(n_states, dtype=bool)
    train_mask[np.unique(train_indices)] = True
    unseen_mask = ~train_mask

    visible = pi > (np.max(pi) * 1e-6)
    if limits is None:
        lo, hi = compute_scatter_limits([(target, prediction, pi)])
    else:
        lo, hi = limits

    ax.scatter(
        target[visible & unseen_mask],
        prediction[visible & unseen_mask],
        s=0.12,
        c="steelblue",
        alpha=0.08,
        rasterized=True,
        label="Unseen",
        zorder=1,
    )
    ax.scatter(
        target[visible & train_mask],
        prediction[visible & train_mask],
        s=1.8,
        c="crimson",
        alpha=0.55,
        rasterized=True,
        label="Train",
        zorder=2,
    )

    diag = np.linspace(lo, hi, 100)
    ax.plot(diag, diag, ls="--", lw=0.9, color="0.5", alpha=0.8, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    if subtitle:
        ax.text(
            0.02,
            0.98,
            subtitle,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.9},
        )
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, markerscale=10, framealpha=0.95)


def compute_scatter_limits(
    panel_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    visible_values = []
    for target, prediction, pi in panel_arrays:
        visible = pi > (np.max(pi) * 1e-6)
        if np.any(visible):
            visible_values.extend((target[visible], prediction[visible]))

    if not visible_values:
        return -1.0, 1.0

    vals = np.concatenate(visible_values)
    lo = np.percentile(vals, 0.5)
    hi = np.percentile(vals, 99.5)
    pad = 0.10 * max(hi - lo, 1e-6)
    return float(lo - pad), float(hi + pad)


def representative_lambda_info(model: dict) -> tuple[str, str, str]:
    summary = model["summary"]
    if len(summary["lambda_grid"]) == 2:
        return (
            format_lambda(summary["lambda_small"]),
            format_lambda(summary["lambda_last"]),
            "last",
        )
    return (
        format_lambda(summary["lambda_small"]),
        format_lambda(summary["lambda_best"]),
        "best",
    )


def format_lambda(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    mantissa_rounded = int(round(mantissa))
    if np.isclose(mantissa, mantissa_rounded):
        mantissa_str = str(mantissa_rounded)
    else:
        mantissa_str = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    return f"{mantissa_str}e{exponent}"


def model_noise_lines(model: dict) -> str:
    return (
        rf"$\sigma_{{gap}}^2$={model['sigma_gap_sq']:.2e}" "\n"
        rf"$\sigma_{{gap}}^2/\mathrm{{Var}}$={model['noise_fraction']:.2f}"
    )


def plot_main_figure(data: dict, output_path: Path) -> None:
    under_model, over_model = select_under_over_models(data)
    models_to_plot = (under_model, over_model)
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

        subtitle_small = (
            rf"$r_{{train}}^H$={rep['metrics']['r_train_hloc'][summary['lambda_small_idx']]:.1e}" "\n"
            rf"{EXCESS_RISK_LABEL}={rep['metrics']['r_pop_ideal'][summary['lambda_small_idx']]:.1e}"
        )
        subtitle_second = (
            rf"$r_{{train}}^H$={rep['metrics']['r_train_hloc'][summary[second_idx_key]]:.1e}" "\n"
            rf"{EXCESS_RISK_LABEL}={rep['metrics']['r_pop_ideal'][summary[second_idx_key]]:.1e}"
        )

        scatter_panel(
            axes[row, 0],
            target=target_hloc,
            prediction=pred_small,
            train_indices=train_indices,
            pi=pi,
            title=(
                rf"{model['model_desc']} ($P/N_s={model['P_over_Ns']:.2f}$, "
                rf"$\sigma_{{gap}}^2$={model['sigma_gap_sq']:.2e}, "
                rf"$\sigma_{{gap}}^2/\mathrm{{Var}}$={model['noise_fraction']:.2f})"
                "\n"
                rf"$H_{{loc,c}}$, $\lambda={lam_small}$"
            ),
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
            title=rf"$O_c\delta^*$, $\lambda={lam_small}$",
            subtitle=subtitle_small,
            limits=ideal_limits,
        )
        scatter_panel(
            axes[row, 2],
            target=target_ideal,
            prediction=pred_second,
            train_indices=train_indices,
            pi=pi,
            title=rf"$O_c\delta^*$, $\lambda={lam_second}$",
            subtitle=subtitle_second,
            limits=ideal_limits,
        )
        axes[row, 0].set_ylabel(r"Prediction $O_c(x)^T \hat{\delta}$")

    for col in range(3):
        axes[-1, col].set_xlabel("Target")

    system = data["system"]
    fig.suptitle(
        rf"Figure 1 main panel: exact fixed-$N_s$ study"
        "\n"
        rf"{system['name']} $L={system['L']}$, $N_s={data['ns']}$",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_curves(data: dict, output_path: Path) -> None:
    under_model, over_model = select_under_over_models(data)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, model in zip(axes, (under_model, over_model)):
        lambdas = np.asarray(model["summary"]["lambda_grid"], dtype=np.float64)
        for key, color, label in (
            ("r_train_hloc", "crimson", r"$r_{\mathrm{train}}^H$"),
            ("r_pop_hloc", "black", r"$r_{\mathrm{pop}}^H$"),
            ("r_pop_ideal", "steelblue", EXCESS_RISK_LABEL),
        ):
            mean = np.asarray(model["summary"][f"{key}_mean"], dtype=np.float64)
            std = np.asarray(model["summary"][f"{key}_std"], dtype=np.float64)
            ax.plot(lambdas, mean, color=color, lw=2, label=label)
            ax.fill_between(lambdas, np.maximum(mean - std, 1e-20), mean + std, color=color, alpha=0.15)

        ref_lambda = model["summary"]["lambda_last"] if len(model["summary"]["lambda_grid"]) == 2 else model["summary"]["lambda_best"]
        ax.axvline(ref_lambda, color="0.5", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_title(
            rf"{model['model_desc']} ($P/N_s={model['P_over_Ns']:.2f}$)" "\n"
            f"{model_noise_lines(model)}",
            fontsize=11,
        )
        ax.set_xlabel(r"Diagonal shift $\lambda$")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Residual")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_width_sweep(data: dict, output_path: Path) -> None:
    models = sorted(data["models"], key=lambda item: item["P_over_Ns"])
    ratios = np.asarray([m["P_over_Ns"] for m in models], dtype=np.float64)
    labels = [m["model_desc"] for m in models]
    noise = np.asarray([m["noise_fraction"] for m in models], dtype=np.float64)
    small_risk = np.asarray([m["summary"]["r_pop_ideal_mean"][m["summary"]["lambda_small_idx"]] for m in models])
    use_last = len(models[0]["summary"]["lambda_grid"]) == 2
    second_idx_key = "lambda_last_idx" if use_last else "lambda_best_idx"
    second_label = (
        r"$\|\hat{\delta}_{\lambda_{\mathrm{reg}}} - \delta^*\|_S^2$"
        if use_last
        else r"$\|\hat{\delta}_{\lambda_{\mathrm{best}}} - \delta^*\|_S^2$"
    )
    second_risk = np.asarray([m["summary"]["r_pop_ideal_mean"][m["summary"][second_idx_key]] for m in models])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = (
        (noise, r"$\sigma_{\mathrm{gap}}^2 / \mathrm{Var}[H_{\mathrm{loc}}]$"),
        (small_risk, r"$\|\hat{\delta}_{\lambda_{\mathrm{small}}} - \delta^*\|_S^2$"),
        (second_risk, second_label),
    )

    for ax, (values, ylabel) in zip(axes, panels):
        ax.plot(ratios, values, "o-", color="black", lw=1.8, ms=6)
        ax.axvline(1.0, color="0.6", ls="--", lw=1)
        ax.set_xscale("log")
        if np.all(values > 0):
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.2)
        ax.set_xlabel(r"$P/N_s$")
        ax.set_ylabel(ylabel)
        for ratio, value, label in zip(ratios, values, labels):
            ax.annotate(label.replace(" alpha=", "\nα="), (ratio, value), textcoords="offset points", xytext=(4, 4), fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the exact Figure 1 study.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(FIGURE1_DIR / "results" / "rbm_exact_L16_ns2048.pkl"),
        help="Input pickle from run_rbm_exact.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(FIGURE1_DIR / "figures"),
        help="Directory for generated figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_results(Path(args.input))
    output_dir = Path(args.output_dir)

    plot_main_figure(data, output_dir / "figure1_main.pdf")
    plot_lambda_curves(data, output_dir / "figure1_lambda_curves.pdf")
    plot_width_sweep(data, output_dir / "figure1_width_sweep.pdf")

    print(f"Saved figures to {output_dir}")


if __name__ == "__main__":
    main()
