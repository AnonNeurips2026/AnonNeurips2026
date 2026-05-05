"""
Plot shared-checkpoint Figure 3 method comparisons from protocol-scoped outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
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

from sr_filtering_nqs.large_scale.core.common import (
    SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA,
    SYSTEM_ORDER,
    SYSTEM_TITLES,
    canonical_shared_fig3_lambda_grid,
    normalize_system_name,
    shared_fig3_default_method_ids,
)
from sr_filtering_nqs.large_scale.core.multilambda_diagnostics import parse_mixture_method_id, validate_mixture_method_ids
from sr_filtering_nqs.large_scale.core.shared_checkpoint_protocol import shared_fig3_eval_files


METHOD_COLORS = {
    "single_lambda": "#1f4e79",
    "indep_single_lambda": "#4b7f52",
    "same_batch_multilambda": "#7a3e00",
    "indep_multilambda": "#8d2d5a",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot shared-checkpoint Figure 3 raw method comparisons")
    parser.add_argument("--input-dir", default=str(ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "large_scale" / "results" / "section4"))
    parser.add_argument("--source-train-lambda", type=float, default=SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA)
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--lambda-grid", type=str, default=None)
    parser.add_argument("--protocol-id", type=str, default=None)
    parser.add_argument("--methods", type=str, default=None)
    return parser.parse_args()


def _selected_methods(raw: str | None, *, k: int, lambda_grid) -> tuple[str, ...]:
    if raw is None:
        methods = shared_fig3_default_method_ids(k)
    else:
        methods = tuple(part.strip() for part in raw.split(",") if part.strip())
        if not methods:
            raise ValueError("At least one Figure 3 method id must be provided")
    return validate_mixture_method_ids(methods, lambda_grid=lambda_grid, single_lambda_count=k)


def _load_eval_payloads(base_root: Path, *, protocol_id: str, source_training_lambda: float, repeat_count: int):
    payloads = []
    for system_name in SYSTEM_ORDER:
        for path in shared_fig3_eval_files(
            base_root,
            system_name,
            protocol_id=protocol_id,
            source_training_lambda=float(source_training_lambda),
            repeat_count=int(repeat_count),
        ):
            with path.open("rb") as f:
                payloads.append(pickle.load(f))
    return payloads


def _filter_payloads(payloads: list[dict], *, lambda_grid) -> list[dict]:
    target = tuple(float(value) for value in lambda_grid)
    return [
        payload
        for payload in payloads
        if tuple(float(value) for value in payload.get("lambda_grid", ())) == target
    ]


def _aggregate(payloads: list[dict], methods: tuple[str, ...]):
    summary = {
        system: {
            method: defaultdict(list)
            for method in methods
        }
        for system in SYSTEM_ORDER
    }
    for payload in payloads:
        system_name = normalize_system_name(payload["system"])
        for method_id in methods:
            method_payload = payload.get("methods", {}).get(method_id)
            if method_payload is None:
                continue
            summary[system_name][method_id]["r_val_raw"].append(float(method_payload["r_val_raw"]))
            summary[system_name][method_id]["var_hat_raw"].append(float(method_payload["var_hat_raw"]))

    aggregated = {system: {} for system in SYSTEM_ORDER}
    for system_name in SYSTEM_ORDER:
        for method_id in methods:
            stats = summary[system_name][method_id]
            if not stats:
                continue
            aggregated[system_name][method_id] = {}
            for metric_name in ("r_val_raw", "var_hat_raw"):
                values = np.asarray(stats[metric_name], dtype=np.float64)
                aggregated[system_name][method_id][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "count": int(values.size),
                }
    return aggregated


def _write_summary_csv(path: Path, *, summary, methods: tuple[str, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "method_id", "metric", "mean", "std", "count"],
        )
        writer.writeheader()
        for system_name in SYSTEM_ORDER:
            for method_id in methods:
                method_stats = summary.get(system_name, {}).get(method_id)
                if not method_stats:
                    continue
                for metric_name in ("r_val_raw", "var_hat_raw"):
                    row = dict(method_stats[metric_name])
                    row.update(
                        {
                            "system": system_name,
                            "method_id": method_id,
                            "metric": metric_name,
                        }
                    )
                    writer.writerow(row)


def _method_label(method_id: str) -> str:
    if method_id == "single_oracle":
        return "single\noracle"
    spec = parse_mixture_method_id(method_id)
    if spec["branch"] == "indep_single_lambda":
        return f"indep\nsingle\nk={spec['k']}"
    if spec["branch"] == "same_batch":
        return f"same-batch\n{spec['weight_scheme'].upper()}\nk={spec['k']}"
    if spec["branch"] == "indep_batch":
        weight = spec["weight_scheme"].upper()
        return f"indep\n{weight}\nk={spec['k']}"
    return method_id


def _method_color(method_id: str) -> str:
    spec = parse_mixture_method_id(method_id)
    return METHOD_COLORS.get(spec["family"], "#1f4e79")


def _render_metric(ax: plt.Axes, *, system_name: str, summary, methods: tuple[str, ...], metric_name: str, ylabel: str):
    values = []
    errors = []
    labels = []
    colors = []
    for method_id in methods:
        method_stats = summary.get(system_name, {}).get(method_id)
        if method_stats is None:
            continue
        values.append(float(method_stats[metric_name]["mean"]))
        errors.append(float(method_stats[metric_name]["std"]))
        labels.append(_method_label(method_id))
        colors.append(_method_color(method_id))

    ax.set_title(SYSTEM_TITLES[system_name])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
    if not values:
        ax.text(0.5, 0.5, "No shared evaluations", ha="center", va="center", transform=ax.transAxes)
        return

    x = np.arange(len(values), dtype=np.float64)
    bars = ax.bar(x, values, color=colors, alpha=0.9)
    ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="black", elinewidth=1.25, capsize=4.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if all(value > 0.0 for value in values):
        ax.set_yscale("log")
        positive_errors = [max(value - error, 1e-30) for value, error in zip(values, errors, strict=True)]
        ymax = max(value + error for value, error in zip(values, errors, strict=True))
        ymin = min(positive_errors)
        if ymin > 0.0 and ymax > ymin:
            ax.set_ylim(ymin * 0.8, ymax * 1.25)
    for bar in bars:
        bar.set_linewidth(0.0)


def main():
    args = parse_args()
    lambda_grid = (
        canonical_shared_fig3_lambda_grid(args.k)
        if args.lambda_grid is None
        else tuple(float(part) for part in args.lambda_grid.split(",") if part.strip())
    )
    if len(tuple(lambda_grid)) != int(args.k):
        raise ValueError(f"Expected a lambda grid of length k={args.k}, got {lambda_grid}")
    protocol_id = args.protocol_id or f"shared_fig3_k{int(args.k)}"
    methods = _selected_methods(args.methods, k=int(args.k), lambda_grid=lambda_grid)
    payloads = _filter_payloads(
        _load_eval_payloads(
            Path(args.input_dir),
            protocol_id=protocol_id,
            source_training_lambda=float(args.source_train_lambda),
            repeat_count=int(args.repeat_count),
        ),
        lambda_grid=lambda_grid,
    )
    summary = _aggregate(payloads, methods)

    output_dir = Path(args.output_dir)
    _write_summary_csv(output_dir / "figure3_shared_checkpoint_raw_summary.csv", summary=summary, methods=methods)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    for row, system_name in enumerate(SYSTEM_ORDER):
        _render_metric(
            axes[row, 0],
            system_name=system_name,
            summary=summary,
            methods=methods,
            metric_name="r_val_raw",
            ylabel=r"Raw $R_{\mathrm{val}}$",
        )
        _render_metric(
            axes[row, 1],
            system_name=system_name,
            summary=summary,
            methods=methods,
            metric_name="var_hat_raw",
            ylabel=r"Raw $V_{\mathrm{mb}}$",
        )
    fig.suptitle("Shared-checkpoint Figure 3 ablations", fontsize=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "figure3_shared_checkpoint_raw.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
