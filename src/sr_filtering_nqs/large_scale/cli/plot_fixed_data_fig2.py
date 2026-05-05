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
from sr_filtering_nqs.large_scale.core.fixed_data_protocol import fig2_result_files, plots_root  # noqa: E402


SYSTEM_ORDER = ("j1j2", "tfim")
SYSTEM_TITLES = {
    "j1j2": r"$J_1$-$J_2$ (8$\times$8, $J_2/J_1=0.5$)",
    "tfim": r"1D TFIM FNQS ($L=100$)",
    "ssm": r"Shastry-Sutherland (8$\times$8, $J'/J=0.8$)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot the 20260422 fixed-data Figure 2 outputs")
    parser.add_argument("--input-dir", default=str(ROOT / "large_scale"))
    parser.add_argument("--output-dir", default=None)
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


def _load_payloads(base_root: Path, systems: tuple[str, ...]) -> dict[str, list[dict]]:
    payloads = {system_name: [] for system_name in systems}
    for system_name in systems:
        for path in fig2_result_files(base_root, system_name):
            with path.open("rb") as f:
                payload = pickle.load(f)
            payloads[system_name].append(payload)
    return payloads


def _aggregate(payloads: dict[str, list[dict]], systems: tuple[str, ...]) -> dict[str, dict[float, dict[str, dict[str, float]]]]:
    summary: dict[str, dict[float, dict[str, list[float]]]] = {
        system_name: {} for system_name in systems
    }
    for system_name, rows in payloads.items():
        for row in rows:
            lam = float(row["lambda"])
            system_summary = summary[system_name].setdefault(
                lam,
                {"r_val_raw": [], "var_hat_raw": []},
            )
            system_summary["r_val_raw"].append(float(row["r_val_raw"]))
            system_summary["var_hat_raw"].append(float(row["var_hat_raw"]))

    reduced = {system_name: {} for system_name in systems}
    for system_name, per_lambda in summary.items():
        for lam, metrics in per_lambda.items():
            reduced[system_name][lam] = {}
            for metric_name, values in metrics.items():
                array = np.asarray(values, dtype=np.float64)
                reduced[system_name][lam][metric_name] = {
                    "mean": float(np.mean(array)),
                    "std": float(np.std(array)),
                    "count": int(array.size),
                }
    return reduced


def _write_summary_csv(output_dir: Path, summary, systems: tuple[str, ...]) -> None:
    path = output_dir / "figure2_fixed_data_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "lambda", "metric", "mean", "std", "count"],
        )
        writer.writeheader()
        for system_name in systems:
            for lam in sorted(summary.get(system_name, {})):
                for metric_name in ("r_val_raw", "var_hat_raw"):
                    row = dict(summary[system_name][lam][metric_name])
                    row.update(
                        {
                            "system": system_name,
                            "lambda": float(lam),
                            "metric": metric_name,
                        }
                    )
                    writer.writerow(row)


def _render_metric(ax: plt.Axes, *, system_name: str, summary, metric_name: str, ylabel: str) -> None:
    by_lambda = summary.get(system_name, {})
    ax.set_title(SYSTEM_TITLES[system_name])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Solve-time $\lambda$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.2)
    if not by_lambda:
        ax.text(0.5, 0.5, "No results", ha="center", va="center", transform=ax.transAxes)
        return

    lambdas = np.asarray(sorted(by_lambda), dtype=np.float64)
    means = np.asarray([by_lambda[lam][metric_name]["mean"] for lam in lambdas], dtype=np.float64)
    stds = np.asarray([by_lambda[lam][metric_name]["std"] for lam in lambdas], dtype=np.float64)
    ax.plot(lambdas, means, marker="o", linewidth=2.0, color="#1f4e79")
    ax.fill_between(
        lambdas,
        np.maximum(means - stds, 1e-30),
        means + stds,
        color="#1f4e79",
        alpha=0.18,
    )


def main() -> None:
    args = parse_args()
    base_root = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else plots_root(base_root)
    systems = _parse_systems(args.systems)

    payloads = _load_payloads(base_root, systems)
    summary = _aggregate(payloads, systems)
    _write_summary_csv(output_dir, summary, systems)

    fig, axes = plt.subplots(2, len(systems), figsize=(5.8 * len(systems), 8.0), constrained_layout=True, squeeze=False)
    for col, system_name in enumerate(systems):
        _render_metric(
            axes[0, col],
            system_name=system_name,
            summary=summary,
            metric_name="r_val_raw",
            ylabel=r"Raw $R_{\mathrm{val}}$",
        )
        _render_metric(
            axes[1, col],
            system_name=system_name,
            summary=summary,
            metric_name="var_hat_raw",
            ylabel=r"Raw $V_{\mathrm{mb}}$",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle("20260422 Fixed-Data Figure 2", fontsize=14)
    fig.savefig(output_dir / "figure2_fixed_data_raw.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
