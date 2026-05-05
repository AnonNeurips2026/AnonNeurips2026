from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core.common import normalize_system_name  # noqa: E402
from sr_filtering_nqs.large_scale.core.fixed_data_protocol import FIG3_METHOD_IDS, fig3_result_files, plots_root  # noqa: E402


SYSTEM_ORDER = ("j1j2", "tfim")
SYSTEM_TITLES = {
    "j1j2": r"$J_1$-$J_2$ (8$\times$8, $J_2/J_1=0.5$)",
    "tfim": r"1D TFIM FNQS ($L=100$)",
    "ssm": r"Shastry-Sutherland (8$\times$8, $J'/J=0.8$)",
}
METHOD_LABELS = {
    "single_best_fixed_lambda": "single\nbest fixed",
    "same_batch_multilambda_uniform_k4": "same-batch\nmulti-$\\lambda$\nuniform",
    "same_batch_multilambda_stacking_k4": "same-batch\nmulti-$\\lambda$\nstacking",
    "indep_single_lambda_uniform_k4": "bagged\nsingle-$\\lambda$",
    "indep_multilambda_uniform_k4": "indep\nmulti-$\\lambda$\nuniform",
    "indep_multilambda_stacking_k4": "MS-SR",
}
METHOD_COLORS = {
    "single_best_fixed_lambda": "#1f4e79",
    "same_batch_multilambda_uniform_k4": "#5b8c5a",
    "same_batch_multilambda_stacking_k4": "#89632a",
    "indep_single_lambda_uniform_k4": "#5f7ca1",
    "indep_multilambda_uniform_k4": "#a05b7f",
    "indep_multilambda_stacking_k4": "#8d2d5a",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the 20260422 fixed-data Figure 3 outputs")
    parser.add_argument("--input-dir", default=str(ROOT / "large_scale"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--systems",
        default=",".join(SYSTEM_ORDER),
        help="Comma-separated systems to include, e.g. j1j2,tfim or ssm.",
    )
    return parser.parse_args()


def _parse_systems(raw: str) -> tuple[str, ...]:
    systems = tuple(normalize_system_name(part.strip()) for part in raw.split(",") if part.strip())
    if not systems:
        raise ValueError("Expected at least one system")
    return tuple(dict.fromkeys(systems))


def _load_payloads(base_root: Path, *, systems: tuple[str, ...], k: int) -> dict[str, list[dict]]:
    payloads = {system_name: [] for system_name in systems}
    for system_name in systems:
        for path in fig3_result_files(base_root, system_name, k=int(k)):
            with path.open("rb") as f:
                payload = pickle.load(f)
            payloads[system_name].append(payload)
    return payloads


def _aggregate(payloads: dict[str, list[dict]], systems: tuple[str, ...]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    summary = {
        system_name: {
            method_id: {"r_val_raw": [], "var_hat_raw": []}
            for method_id in FIG3_METHOD_IDS
        }
        for system_name in systems
    }
    for system_name, rows in payloads.items():
        for row in rows:
            for method_id in FIG3_METHOD_IDS:
                method_payload = row.get("methods", {}).get(method_id)
                if method_payload is None:
                    continue
                summary[system_name][method_id]["r_val_raw"].append(float(method_payload["r_val_raw"]))
                summary[system_name][method_id]["var_hat_raw"].append(float(method_payload["var_hat_raw"]))

    reduced = {
        system_name: {} for system_name in systems
    }
    for system_name in systems:
        for method_id in FIG3_METHOD_IDS:
            reduced[system_name][method_id] = {}
            for metric_name in ("r_val_raw", "var_hat_raw"):
                array = np.asarray(summary[system_name][method_id][metric_name], dtype=np.float64)
                reduced[system_name][method_id][metric_name] = {
                    "mean": float(np.mean(array)) if array.size else float("nan"),
                    "std": float(np.std(array)) if array.size else float("nan"),
                    "count": int(array.size),
                }
    return reduced


def _write_summary_csv(output_dir: Path, summary, systems: tuple[str, ...]) -> None:
    path = output_dir / "figure3_fixed_data_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "method_id", "metric", "mean", "std", "count"],
        )
        writer.writeheader()
        for system_name in systems:
            for method_id in FIG3_METHOD_IDS:
                for metric_name in ("r_val_raw", "var_hat_raw"):
                    row = dict(summary[system_name][method_id][metric_name])
                    row.update(
                        {
                            "system": system_name,
                            "method_id": method_id,
                            "metric": metric_name,
                        }
                    )
                    writer.writerow(row)


def _render_metric(ax: plt.Axes, *, system_name: str, summary, metric_name: str, ylabel: str) -> None:
    x = np.arange(len(FIG3_METHOD_IDS), dtype=np.float64)
    means = np.asarray(
        [summary[system_name][method_id][metric_name]["mean"] for method_id in FIG3_METHOD_IDS],
        dtype=np.float64,
    )
    stds = np.asarray(
        [summary[system_name][method_id][metric_name]["std"] for method_id in FIG3_METHOD_IDS],
        dtype=np.float64,
    )
    ax.set_title(SYSTEM_TITLES[system_name])
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)
    ax.bar(
        x,
        means,
        color=[METHOD_COLORS[method_id] for method_id in FIG3_METHOD_IDS],
        alpha=0.92,
    )
    ax.errorbar(x, means, yerr=stds, fmt="none", ecolor="black", elinewidth=1.25, capsize=4.0)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[method_id] for method_id in FIG3_METHOD_IDS])
    finite = means[np.isfinite(means) & (means > 0.0)]
    if finite.size:
        ax.set_yscale("log")


def main() -> None:
    args = parse_args()
    base_root = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else plots_root(base_root)
    systems = _parse_systems(args.systems)

    payloads = _load_payloads(base_root, systems=systems, k=int(args.k))
    summary = _aggregate(payloads, systems)
    _write_summary_csv(output_dir, summary, systems)

    fig, axes = plt.subplots(len(systems), 2, figsize=(13.0, 4.4 * len(systems)), constrained_layout=True, squeeze=False)
    for row, system_name in enumerate(systems):
        _render_metric(
            axes[row, 0],
            system_name=system_name,
            summary=summary,
            metric_name="r_val_raw",
            ylabel=r"Raw $R_{\mathrm{val}}$",
        )
        _render_metric(
            axes[row, 1],
            system_name=system_name,
            summary=summary,
            metric_name="var_hat_raw",
            ylabel=r"Raw $V_{\mathrm{mb}}$",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle("20260422 Fixed-Data Figure 3", fontsize=14)
    fig.savefig(output_dir / "figure3_fixed_data_raw.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
