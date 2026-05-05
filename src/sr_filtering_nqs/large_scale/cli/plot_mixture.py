"""
Plot checkpoint-local multi-lambda SR diagnostics without touching Section 4 plots.

Examples (run from ``nqs_support_core/``):

    uv run python ../large_scale/cli/plot_mixture.py --system j1j2
    uv run python ../large_scale/cli/plot_mixture.py --system tfim --latest-only
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core.common import normalize_system_name
from sr_filtering_nqs.large_scale.core.multilambda_diagnostics import (
    MIXTURE_METHOD_IDS,
    discover_mixture_checkpoints,
    mixture_eval_output_path,
)


METHOD_LABELS = {
    "same_batch_uniform": "same-batch uniform",
    "same_batch_gcv": "same-batch GCV",
    "indep_uniform": "indep-batch uniform",
    "indep_batch_gcv": "indep-batch GCV",
    "indep_stacking": "indep-batch stacking",
}
METHOD_COLORS = {
    "same_batch_uniform": "#1f4e79",
    "same_batch_gcv": "#7a3e00",
    "indep_uniform": "#3a7d44",
    "indep_batch_gcv": "#0e7490",
    "indep_stacking": "#8d2d5a",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot checkpoint-local multi-lambda SR diagnostics")
    parser.add_argument("--system", required=True, choices=["j1j2", "tfim", "ising", "ising1d", "tfim_fnqs"])
    parser.add_argument("--input-dir", default=str(ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "large_scale" / "results" / "mixture"))
    parser.add_argument("--sr-variant", default=None, choices=["same_batch", "indep_batch"])
    parser.add_argument("--sr-weight-scheme", default=None, choices=["uniform", "gcv", "stacking"])
    parser.add_argument("--sr-lambda-grid", type=str, default=None)
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--latest-only", action="store_true")
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(MIXTURE_METHOD_IDS),
        help="Comma-separated subset of mixture method ids to plot.",
    )
    return parser.parse_args()


def _selected_methods(raw: str) -> list[str]:
    methods = [part.strip() for part in raw.split(",") if part.strip()]
    if not methods:
        raise ValueError("At least one method id must be provided")
    unknown = [method for method in methods if method not in MIXTURE_METHOD_IDS]
    if unknown:
        raise ValueError(f"Unknown mixture methods: {unknown}")
    return methods


def _load_eval_payloads(args) -> list[dict]:
    checkpoints = discover_mixture_checkpoints(
        args.system,
        args.input_dir,
        sr_variant=args.sr_variant,
        sr_weight_scheme=args.sr_weight_scheme,
        lambda_grid=args.sr_lambda_grid,
        min_step=args.min_step,
        latest_only=args.latest_only,
    )
    payloads = []
    for _, _, _, step, ckpt_path in checkpoints:
        eval_path = mixture_eval_output_path(ckpt_path, step)
        if not eval_path.exists():
            continue
        with eval_path.open("rb") as f:
            payloads.append(pickle.load(f))
    return payloads


def _aggregate(payloads: list[dict], methods: list[str]) -> dict[str, dict[int, dict[str, float]]]:
    grouped: dict[str, dict[int, dict[str, list[float]]]] = {
        method: defaultdict(lambda: {"r_val_norm": [], "var_hat_norm": []})
        for method in methods
    }
    for payload in payloads:
        step = int(payload["step"])
        for method in methods:
            method_payload = payload["methods"].get(method)
            if method_payload is None:
                continue
            grouped[method][step]["r_val_norm"].append(float(method_payload["r_val_norm"]))
            grouped[method][step]["var_hat_norm"].append(float(method_payload["var_hat_norm"]))

    summary: dict[str, dict[int, dict[str, float]]] = {method: {} for method in methods}
    for method in methods:
        for step, metric_lists in grouped[method].items():
            rval = np.asarray(metric_lists["r_val_norm"], dtype=np.float64)
            variance = np.asarray(metric_lists["var_hat_norm"], dtype=np.float64)
            summary[method][step] = {
                "r_val_norm_mean": float(np.mean(rval)),
                "r_val_norm_std": float(np.std(rval)),
                "var_hat_norm_mean": float(np.mean(variance)),
                "var_hat_norm_std": float(np.std(variance)),
                "count": int(rval.size),
            }
    return summary


def _write_summary_csv(path: Path, summary: dict[str, dict[int, dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method_id",
                "step",
                "count",
                "r_val_norm_mean",
                "r_val_norm_std",
                "var_hat_norm_mean",
                "var_hat_norm_std",
            ],
        )
        writer.writeheader()
        for method, by_step in summary.items():
            for step in sorted(by_step):
                row = dict(by_step[step])
                row["method_id"] = method
                row["step"] = int(step)
                writer.writerow(row)


def _plot_metric(ax: plt.Axes, summary: dict[str, dict[int, dict[str, float]]], *, methods: list[str], mean_key: str, std_key: str, ylabel: str):
    for method in methods:
        by_step = summary.get(method, {})
        if not by_step:
            continue
        steps = sorted(by_step)
        means = np.asarray([by_step[step][mean_key] for step in steps], dtype=np.float64)
        stds = np.asarray([by_step[step][std_key] for step in steps], dtype=np.float64)
        ax.plot(
            steps,
            means,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, None),
            marker="o",
            linewidth=2.0,
        )
        ax.fill_between(
            steps,
            np.maximum(means - stds, 1e-16),
            means + stds,
            color=METHOD_COLORS.get(method, None),
            alpha=0.15,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Checkpoint step")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.2)


def main():
    args = parse_args()
    system_name = normalize_system_name(args.system)
    methods = _selected_methods(args.methods)
    payloads = _load_eval_payloads(args)
    if not payloads:
        raise SystemExit(f"No mixture evaluation payloads found for {system_name}")

    summary = _aggregate(payloads, methods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem_parts = [system_name, "mixture_metrics"]
    if args.sr_variant is not None:
        stem_parts.append(args.sr_variant)
    if args.sr_weight_scheme is not None:
        stem_parts.append(args.sr_weight_scheme)
    if args.latest_only:
        stem_parts.append("latest")
    stem = "_".join(stem_parts)

    _write_summary_csv(output_dir / f"{stem}.csv", summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _plot_metric(
        axes[0],
        summary,
        methods=methods,
        mean_key="r_val_norm_mean",
        std_key="r_val_norm_std",
        ylabel=r"Held-out residual $r_{\mathrm{val}}$",
    )
    _plot_metric(
        axes[1],
        summary,
        methods=methods,
        mean_key="var_hat_norm_mean",
        std_key="var_hat_norm_std",
        ylabel=r"Normalized disagreement variance",
    )
    axes[0].set_title(f"{system_name}: residual")
    axes[1].set_title(f"{system_name}: variance")
    axes[1].legend(frameon=False, loc="best")

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
