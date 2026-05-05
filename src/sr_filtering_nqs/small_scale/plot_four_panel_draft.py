from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATCHED_NOISE = REPO_ROOT / "artifacts/small_scale/matched_noise/rbm490_vit10.pkl"
DEFAULT_MATCHED_STATE = (
    REPO_ROOT
    / "artifacts/small_scale/matched_state"
    / "alternating_polish_vit_step0100_to_direct_rbm_a2_figure1_style.pkl"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts/reproduced/figure2_four_panel.pdf"
EXCESS_RISK_LABEL = r"$\|\hat{\delta}_{\lambda}-\delta^*\|_S^2$"
DEFAULT_MAX_VAL_POINTS = 30_000
DEFAULT_MAX_TRAIN_POINTS = 1_500
DEFAULT_VISIBILITY_FACTOR = 1e-10
DEFAULT_LIMIT_VISIBILITY_FACTOR = 1e-6
MATCHED_STATE_LIMITS = (-70.0, 70.0)


def load_pickle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def metric_mean_std(
    model: dict,
    metric: str,
    lam_idx: int,
) -> tuple[float, float, int]:
    summary = model.get("summary", {})
    representative = model.get("representative", {})
    rep_metrics = representative.get("metrics", {})
    n_trials = int(summary.get("n_trials", len(model.get("trials", [])) or 1))

    mean_key = f"{metric}_mean"
    std_key = f"{metric}_std"
    if mean_key in summary:
        mean = float(np.asarray(summary[mean_key], dtype=np.float64)[lam_idx])
    else:
        mean = float(np.asarray(rep_metrics[metric], dtype=np.float64)[lam_idx])

    if n_trials > 1 and model.get("trials"):
        values = np.asarray(
            [trial["metrics"][metric][lam_idx] for trial in model["trials"]],
            dtype=np.float64,
        )
        spread = float(np.std(values, ddof=1))
    elif std_key in summary:
        spread = float(np.asarray(summary[std_key], dtype=np.float64)[lam_idx])
    else:
        spread = 0.0
    return mean, spread, n_trials


def format_lambda(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    mantissa_rounded = int(round(mantissa))
    if np.isclose(mantissa, mantissa_rounded):
        mantissa_text = str(mantissa_rounded)
    else:
        mantissa_text = f"{mantissa:.1f}".rstrip("0").rstrip(".")
    return rf"10^{{{exponent}}}" if mantissa_text == "1" else rf"{mantissa_text}\times10^{{{exponent}}}"


def format_sci(value: float, precision: int = 2) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exponent)
    return rf"{mantissa:.{precision}f}\times10^{{{exponent}}}"


def format_decimal(value: float) -> str:
    return f"{value:.3g}"


def format_two_decimals(value: float) -> str:
    return f"{float(value):.2f}"


def format_metric(value: float) -> str:
    value = float(value)
    if value == 0.0 or 1e-3 <= abs(value) < 1e4:
        return format_decimal(value)
    return format_sci(value)


def visible_mask(pi: np.ndarray, visibility_factor: float = DEFAULT_VISIBILITY_FACTOR) -> np.ndarray:
    if visibility_factor <= 0.0:
        return pi > 0.0
    return pi > (np.max(pi) * visibility_factor)


def compute_limits(
    panels: list[dict],
    *,
    visibility_factor: float = DEFAULT_VISIBILITY_FACTOR,
) -> tuple[float, float]:
    values: list[np.ndarray] = []
    for panel in panels:
        if panel is None:
            continue
        mask = visible_mask(panel["pi"], visibility_factor=visibility_factor)
        if np.any(mask):
            values.append(panel["target"][mask])
            values.append(panel["prediction"][mask])
    if not values:
        return -1.0, 1.0

    flat = np.concatenate(values)
    lo, hi = np.percentile(flat, [0.4, 99.6])
    pad = 0.08 * max(float(hi - lo), 1e-6)
    return float(lo - pad), float(hi + pad)


def panel_visible_counts(
    panel: dict,
    *,
    visibility_factor: float = DEFAULT_VISIBILITY_FACTOR,
    limits: tuple[float, float] | None = None,
) -> tuple[int, int]:
    target = panel["target"]
    prediction = panel["prediction"]
    pi = panel["pi"]
    train_indices = panel["train_indices"]
    train_mask = np.zeros(len(target), dtype=bool)
    train_mask[np.unique(train_indices)] = True
    mask = visible_mask(pi, visibility_factor=visibility_factor)
    if limits is not None:
        lo, hi = limits
        mask &= (target >= lo) & (target <= hi) & (prediction >= lo) & (prediction <= hi)
    return int((mask & train_mask).sum()), int((mask & ~train_mask).sum())


def shared_plot_counts(
    panels: list[dict | None],
    *,
    max_train_points: int,
    max_val_points: int,
    visibility_factor: float = DEFAULT_VISIBILITY_FACTOR,
    limits: tuple[float, float] | None = None,
) -> tuple[int, int]:
    counts = [
        panel_visible_counts(panel, visibility_factor=visibility_factor, limits=limits)
        for panel in panels
        if panel is not None
    ]
    if not counts:
        return 0, 0
    train_count = min(max_train_points, min(train for train, _ in counts))
    val_count = min(max_val_points, min(val for _, val in counts))
    return int(train_count), int(val_count)


def deterministic_subsample(indices: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    if len(indices) <= n_points:
        return indices
    rng = np.random.default_rng(seed)
    chosen = rng.choice(indices, size=n_points, replace=False)
    return np.sort(chosen)


def panel_from_model(model: dict) -> dict:
    selected = model["selected_checkpoint"]
    representative = model["representative"]
    summary = model["summary"]
    lam_idx = int(summary["lambda_small_idx"])
    lam = float(summary["lambda_small"])
    excess_risk_mean, excess_risk_std, n_trials = metric_mean_std(
        model,
        "r_pop_ideal",
        lam_idx,
    )

    return {
        "model_desc": str(model["model_desc"]),
        "p_over_ns": float(model["P_over_Ns"]),
        "sigma_gap_sq": float(model["sigma_gap_sq"]),
        "noise_fraction": float(model["noise_fraction"]),
        "lambda": lam,
        "target": np.asarray(selected["target_ideal"], dtype=np.float64),
        "prediction": np.asarray(
            representative["predictions"]["pred_small"], dtype=np.float64
        ),
        "pi": np.asarray(selected["pi"], dtype=np.float64),
        "train_indices": np.asarray(representative["train_indices"], dtype=np.int64),
        "train_residual": float(representative["metrics"]["r_train_hloc"][lam_idx]),
        "excess_risk": excess_risk_mean,
        "excess_risk_std": excess_risk_std,
        "n_trials": n_trials,
        "representative_excess_risk": float(
            representative["metrics"]["r_pop_ideal"][lam_idx]
        ),
    }


def ordered_model_panels(data: dict) -> list[dict]:
    models = sorted(data["models"], key=lambda model: float(model["P_over_Ns"]))
    return [panel_from_model(model) for model in models[:2]]


def draw_scatter_panel(
    ax: plt.Axes,
    panel: dict,
    *,
    limits: tuple[float, float],
    panel_label: str,
    show_legend: bool,
    max_train_points: int,
    max_val_points: int,
    seed: int,
    visibility_factor: float,
) -> None:
    target = panel["target"]
    prediction = panel["prediction"]
    pi = panel["pi"]
    train_indices = panel["train_indices"]

    train_mask = np.zeros(len(target), dtype=bool)
    train_mask[np.unique(train_indices)] = True
    mask = visible_mask(pi, visibility_factor=visibility_factor)
    lo, hi = limits
    mask &= (target >= lo) & (target <= hi) & (prediction >= lo) & (prediction <= hi)
    train_plot_idx = deterministic_subsample(
        np.flatnonzero(mask & train_mask),
        max_train_points,
        seed=10_000 + seed,
    )
    val_plot_idx = deterministic_subsample(
        np.flatnonzero(mask & ~train_mask),
        max_val_points,
        seed=20_000 + seed,
    )

    ax.scatter(
        target[val_plot_idx],
        prediction[val_plot_idx],
        s=0.55,
        c="#1b6fb6",
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        label="Val",
        zorder=1,
    )
    ax.scatter(
        target[train_plot_idx],
        prediction[train_plot_idx],
        s=2.8,
        c="#d91e55",
        alpha=0.68,
        linewidths=0,
        rasterized=True,
        label="Train",
        zorder=2,
    )

    diag = np.linspace(lo, hi, 128)
    ax.plot(diag, diag, color="0.45", lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(axis="both", labelsize=8, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    info = (
        rf"$\mathcal{{E}}_{{\lambda}}\approx"
        rf"{format_two_decimals(panel['excess_risk'])}"
        rf"\pm{format_two_decimals(panel['excess_risk_std'])}$"
    )
    ax.text(
        0.035,
        0.965,
        info,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.6,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "0.2",
            "linewidth": 0.7,
            "alpha": 0.92,
        },
    )
    ax.text(
        0.02,
        1.04,
        panel_label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    if show_legend:
        legend = ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.94,
            fontsize=8.4,
            markerscale=1.0,
            handletextpad=0.35,
            borderpad=0.3,
        )
        for handle, size in zip(legend.legend_handles, (18.0, 36.0)):
            if hasattr(handle, "set_alpha"):
                handle.set_alpha(0.75)
            if hasattr(handle, "set_sizes"):
                handle.set_sizes([size])


def draw_pending_panel(ax: plt.Axes, *, panel_label: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.02,
        1.04,
        panel_label,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.5,
        "matched-state\npending",
        transform=ax.transAxes,
        va="center",
        ha="center",
        fontsize=9,
        color="0.45",
    )


def draw_row_label(
    ax: plt.Axes,
    *,
    heading: str,
    detail: str,
) -> None:
    ax.set_axis_off()
    ax.text(
        0.5,
        0.62,
        heading,
        va="center",
        ha="center",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.40,
        detail,
        va="center",
        ha="center",
        fontsize=9.8,
        linespacing=1.35,
    )


def plot_four_panel(
    *,
    matched_noise: dict,
    matched_state: dict | None,
    output_path: Path,
    panel_output_dir: Path | None = None,
    max_train_points: int = DEFAULT_MAX_TRAIN_POINTS,
    max_val_points: int = DEFAULT_MAX_VAL_POINTS,
    visibility_factor: float = DEFAULT_VISIBILITY_FACTOR,
    limit_visibility_factor: float = DEFAULT_LIMIT_VISIBILITY_FACTOR,
) -> None:
    noise_panels = ordered_model_panels(matched_noise)
    state_panels = ordered_model_panels(matched_state) if matched_state is not None else [None, None]
    state_limits = MATCHED_STATE_LIMITS
    noise_limits = compute_limits(noise_panels, visibility_factor=limit_visibility_factor)
    state_train_points, state_val_points = shared_plot_counts(
        state_panels,
        max_train_points=max_train_points,
        max_val_points=max_val_points,
        visibility_factor=visibility_factor,
        limits=state_limits,
    )
    noise_train_points, noise_val_points = shared_plot_counts(
        noise_panels,
        max_train_points=max_train_points,
        max_val_points=max_val_points,
        visibility_factor=visibility_factor,
        limits=noise_limits,
    )
    common_train_points = noise_train_points
    common_val_points = noise_val_points
    if all(panel is not None for panel in state_panels):
        common_train_points = min(common_train_points, state_train_points)
        common_val_points = min(common_val_points, state_val_points)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(8.2, 5.8), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, width_ratios=[0.42, 1.0, 1.0])
    row_label_axes = [fig.add_subplot(grid[row, 0]) for row in range(2)]
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])],
            [fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])],
        ],
        dtype=object,
    )
    panel_labels = [["a", "b"], ["c", "d"]]

    panel_rows = [
        (noise_panels, common_train_points, common_val_points, noise_limits),
        (state_panels, common_train_points, common_val_points, state_limits),
    ]
    for row, (panels, train_points, val_points, limits) in enumerate(panel_rows):
        for col, panel in enumerate(panels):
            if panel is None:
                draw_pending_panel(axes[row, col], panel_label=panel_labels[row][col])
                continue
            draw_scatter_panel(
                axes[row, col],
                panel,
                limits=limits,
                panel_label=panel_labels[row][col],
                show_legend=(row == 0 and col == 0),
                max_train_points=train_points,
                max_val_points=val_points,
                seed=100 * (row + 1) + col,
                visibility_factor=visibility_factor,
            )

    if all(panel is not None for panel in state_panels):
        matched_fidelity = matched_state.get("matching", {}).get("matched_fidelity")
        fidelity_line = (
            "\n" rf"$F={float(matched_fidelity):.4f}$"
            if matched_fidelity is not None
            else ""
        )
        state_row_detail = (
            rf"$\sigma_{{\mathrm{{RBM}}}}^2={format_decimal(state_panels[0]['sigma_gap_sq'])}$"
            "\n"
            rf"$\sigma_{{\mathrm{{ViT}}}}^2={format_decimal(state_panels[1]['sigma_gap_sq'])}$"
            f"{fidelity_line}"
        )
    else:
        state_row_detail = ""
    noise_sigma_common = 0.5 * (
        noise_panels[0]["sigma_gap_sq"] + noise_panels[1]["sigma_gap_sq"]
    )
    noise_row_detail = rf"$\sigma^2\approx{noise_sigma_common:.1f}$"
    draw_row_label(row_label_axes[0], heading="Matched Noise", detail=noise_row_detail)
    draw_row_label(row_label_axes[1], heading="Matched State", detail=state_row_detail)

    for ax in axes.ravel():
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    rbm_ratio = state_panels[0]["p_over_ns"] if state_panels[0] is not None else noise_panels[0]["p_over_ns"]
    vit_ratio = state_panels[1]["p_over_ns"] if state_panels[1] is not None else noise_panels[1]["p_over_ns"]
    axes[0, 0].set_title(rf"RBM ($P/N_s={rbm_ratio:.2f}$)", pad=13)
    axes[0, 1].set_title(rf"ViT ($P/N_s={vit_ratio:.2f}$)", pad=13)
    for ax in axes.ravel():
        ax.set_xlabel(r"Target $O_c(x)^T\delta^*$", fontsize=8.4, labelpad=2.0)
        ax.set_ylabel(
            r"Pred $O_c(x)^T\hat{\delta}_{\lambda}$",
            fontsize=8.4,
            labelpad=2.0,
        )

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.095, top=0.90, wspace=0.31, hspace=0.42)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if panel_output_dir is not None:
        panel_output_dir.mkdir(parents=True, exist_ok=True)
        renderer = fig.canvas.get_renderer()
        panel_names = [
            ["figure1_panel_a_matched_noise_rbm.pdf", "figure1_panel_b_matched_noise_vit.pdf"],
            ["figure1_panel_c_matched_state_rbm.pdf", "figure1_panel_d_matched_state_vit.pdf"],
        ]
        for row in range(2):
            for col in range(2):
                panel_bbox = axes[row, col].get_tightbbox(renderer)
                panel_bbox = panel_bbox.transformed(fig.dpi_scale_trans.inverted())
                panel_bbox = panel_bbox.expanded(1.05, 1.06)
                fig.savefig(
                    panel_output_dir / panel_names[row][col],
                    dpi=300,
                    bbox_inches=panel_bbox,
                )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the draft four-panel Section 3 diagnostic figure."
    )
    parser.add_argument(
        "--matched-noise",
        type=Path,
        default=DEFAULT_MATCHED_NOISE,
        help=f"Matched-noise result pickle. Default: {DEFAULT_MATCHED_NOISE}",
    )
    parser.add_argument(
        "--matched-state",
        type=Path,
        default=DEFAULT_MATCHED_STATE,
        help=f"Matched-state result pickle. Default: {DEFAULT_MATCHED_STATE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--panel-output-dir",
        type=Path,
        default=None,
        help="Optional directory where the four subplot PDFs are saved separately.",
    )
    parser.add_argument(
        "--max-val-points",
        type=int,
        default=DEFAULT_MAX_VAL_POINTS,
        help=f"Maximum validation points per panel. Default: {DEFAULT_MAX_VAL_POINTS}",
    )
    parser.add_argument(
        "--max-train-points",
        type=int,
        default=DEFAULT_MAX_TRAIN_POINTS,
        help=f"Maximum training points per panel. Default: {DEFAULT_MAX_TRAIN_POINTS}",
    )
    parser.add_argument(
        "--visibility-factor",
        type=float,
        default=DEFAULT_VISIBILITY_FACTOR,
        help=(
            "Plot configurations with pi > max(pi) * visibility_factor. "
            f"Use 0 for all nonzero-probability states. Default: {DEFAULT_VISIBILITY_FACTOR:g}"
        ),
    )
    parser.add_argument(
        "--limit-visibility-factor",
        type=float,
        default=DEFAULT_LIMIT_VISIBILITY_FACTOR,
        help=(
            "Use this higher-probability mask for axis limits, while --visibility-factor "
            "controls which points can be plotted. "
            f"Default: {DEFAULT_LIMIT_VISIBILITY_FACTOR:g}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matched_noise = load_pickle(args.matched_noise)
    matched_state = load_pickle(args.matched_state) if args.matched_state else None
    plot_four_panel(
        matched_noise=matched_noise,
        matched_state=matched_state,
        output_path=args.output,
        panel_output_dir=args.panel_output_dir,
        max_train_points=args.max_train_points,
        max_val_points=args.max_val_points,
        visibility_factor=args.visibility_factor,
        limit_visibility_factor=args.limit_visibility_factor,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
