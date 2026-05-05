"""
Plot shared-checkpoint Figure 2 diagnostics from protocol-scoped outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import sys
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
    format_lambda_dir,
    results_dir_for_run,
    normalize_system_name,
)
from sr_filtering_nqs.large_scale.core.multidelta_metrics import EPS, center_samples
from sr_filtering_nqs.large_scale.core.shared_checkpoint_protocol import (
    shared_fig2_delta_bank_path,
    shared_fig2_metric_files,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot shared-checkpoint Figure 2 raw metrics")
    parser.add_argument("--input-dir", default=str(ROOT))
    parser.add_argument("--output-dir", default=str(ROOT / "large_scale" / "results" / "section4"))
    parser.add_argument("--protocol-id", default="shared_fig2_single")
    parser.add_argument("--source-train-lambda", type=float, default=SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA)
    parser.add_argument("--repeat-count", type=int, default=20)
    parser.add_argument(
        "--checkpoint-subset",
        choices=("all", "early5", "late5"),
        default="all",
        help="Average over all checkpoints, the earliest 5, or the latest 5 for each system.",
    )
    parser.add_argument(
        "--style",
        choices=("band", "errorbar"),
        default="band",
        help="How to visualize uncertainty around the checkpoint-averaged mean.",
    )
    parser.add_argument(
        "--rval-mode",
        choices=("mean_over_m", "per_delta"),
        default="mean_over_m",
        help=(
            "Use the stored checkpoint-level R_val (already averaged over m), or recompute "
            "held-out R_val separately for each of the m saved deltas."
        ),
    )
    parser.add_argument(
        "--rval-yscale",
        choices=("log", "linear"),
        default="log",
        help="Y-axis scale for the raw R_val panels.",
    )
    parser.add_argument(
        "--twobatch-yscale",
        choices=("log", "linear"),
        default="log",
        help="Y-axis scale for the raw V_mb panels.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory for per-delta R_val recomputation payloads.",
    )
    parser.add_argument(
        "--j1j2-validation-round-samples",
        type=int,
        default=4000,
        help="Validation round size to use when recomputing per-delta J1J2 R_val.",
    )
    parser.add_argument(
        "--j1j2-jvp-chunk",
        type=int,
        default=2000,
        help="JVP chunk to use when recomputing per-delta J1J2 R_val.",
    )
    parser.add_argument(
        "--j1j2-conn-chunk",
        type=int,
        default=4096,
        help="Connected-sample chunk to use when recomputing per-delta J1J2 R_val.",
    )
    parser.add_argument(
        "--j1j2-eval-chunk",
        type=int,
        default=4096,
        help="Model eval chunk to use when recomputing per-delta J1J2 R_val.",
    )
    parser.add_argument(
        "--j1j2-delta-batch-size",
        type=int,
        default=10,
        help="Delta batch size to use when recomputing per-delta J1J2 R_val.",
    )
    parser.add_argument(
        "--tfim-validation-round-samples",
        type=int,
        default=None,
        help="Optional validation round size to use when recomputing per-delta TFIM R_val.",
    )
    parser.add_argument(
        "--tfim-jvp-chunk",
        type=int,
        default=None,
        help="Optional JVP chunk to use when recomputing per-delta TFIM R_val.",
    )
    parser.add_argument(
        "--tfim-delta-batch-size",
        type=int,
        default=10,
        help="Delta batch size to use when recomputing per-delta TFIM R_val.",
    )
    parser.add_argument(
        "--chunk-size-bwd",
        type=int,
        default=None,
        help="Optional backward chunk override for checkpoint reconstruction during per-delta recomputation.",
    )
    return parser.parse_args()


def _load_metric_rows(base_root: Path, *, protocol_id: str, source_training_lambda: float, repeat_count: int):
    rows = {"rval": [], "twobatch": []}
    for system_name in SYSTEM_ORDER:
        for metric in rows:
            for path in shared_fig2_metric_files(
                base_root,
                system_name,
                protocol_id=protocol_id,
                source_training_lambda=float(source_training_lambda),
                metric=metric,
                repeat_count=int(repeat_count),
            ):
                with path.open("rb") as f:
                    payload = dict(pickle.load(f))
                payload["__path__"] = str(path)
                rows[metric].append(payload)
    return rows


def _group_metric_rows(rows: list[dict], *, value_key: str):
    grouped: dict[str, dict[float, list[float]]] = {system: {} for system in SYSTEM_ORDER}
    for row in rows:
        system_name = normalize_system_name(row["system"])
        grouped[system_name].setdefault(float(row["lambda"]), []).append(float(row[value_key]))
    return grouped


def _selected_steps(rows: list[dict], *, checkpoint_subset: str):
    selected = {}
    for system_name in SYSTEM_ORDER:
        system_rows = [row for row in rows if normalize_system_name(row["system"]) == system_name]
        steps = sorted({int(row.get("source_step", row["step"])) for row in system_rows})
        if checkpoint_subset == "early5":
            selected[system_name] = set(steps[:5])
        elif checkpoint_subset == "late5":
            selected[system_name] = set(steps[-5:])
        else:
            selected[system_name] = set(steps)
    return selected


def _filter_rows(rows: list[dict], *, checkpoint_subset: str):
    if checkpoint_subset == "all":
        return rows
    selected = _selected_steps(rows, checkpoint_subset=checkpoint_subset)
    filtered = []
    for row in rows:
        system_name = normalize_system_name(row["system"])
        step = int(row.get("source_step", row["step"]))
        if step in selected[system_name]:
            filtered.append(row)
    return filtered


def _summaries(grouped):
    summary = {}
    for system_name, by_lambda in grouped.items():
        summary[system_name] = {}
        for lam, values in by_lambda.items():
            arr = np.asarray(values, dtype=np.float64)
            summary[system_name][lam] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "count": int(arr.size),
            }
    return summary


def _write_summary_csv(path: Path, *, rval_summary, twobatch_summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["system", "metric", "lambda", "mean", "std", "count"],
        )
        writer.writeheader()
        for metric_name, metric_summary in (
            ("r_val_raw", rval_summary),
            ("v_mb_raw", twobatch_summary),
        ):
            for system_name in SYSTEM_ORDER:
                for lam in sorted(metric_summary.get(system_name, {})):
                    row = dict(metric_summary[system_name][lam])
                    row.update(
                        {
                            "system": system_name,
                            "metric": metric_name,
                            "lambda": float(lam),
                        }
                    )
                    writer.writerow(row)


def _lambda_label(lam: float) -> str:
    exponent = int(round(math.log10(lam)))
    if math.isclose(lam, 10.0**exponent, rel_tol=1e-12, abs_tol=0.0):
        return f"1e{exponent}"
    return format_lambda_dir(lam)


def _metric_value_key(metric_name: str, *, rval_mode: str) -> str:
    if metric_name == "rval":
        return "r_val" if rval_mode == "mean_over_m" else "r_val_per_delta"
    if metric_name == "twobatch":
        return "var_hat_raw"
    raise ValueError(f"Unsupported metric_name={metric_name!r}")


def _per_delta_cache_path(cache_dir: Path, row: dict) -> Path:
    system_name = normalize_system_name(row["system"])
    source_step = int(row.get("source_step", row["step"]))
    diagnostic_lambda = float(row.get("diagnostic_lambda", row["lambda"]))
    source_training_lambda = float(row.get("source_training_lambda", SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA))
    repeat_count = int(row.get("repeat_count", row.get("m_deltas", 0)))
    n_val = int(row["n_val"])
    protocol_id = str(row.get("protocol_id", "shared_fig2_single"))
    return (
        cache_dir
        / protocol_id
        / system_name
        / f"src_{format_lambda_dir(source_training_lambda)}"
        / f"step{source_step:06d}"
        / (
            f"rval_per_delta__diag_{format_lambda_dir(diagnostic_lambda)}"
            f"__m{repeat_count:03d}__nval{n_val}.pkl"
        )
    )


def _checkpoint_path_for_row(base_root: Path, row: dict) -> Path:
    system_name = normalize_system_name(row["system"])
    source_step = int(row.get("source_step", row["step"]))
    source_training_lambda = float(row.get("source_training_lambda", SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA))
    return (
        results_dir_for_run(base_root, system_name, source_training_lambda)
        / "checkpoints"
        / f"checkpoint_step{source_step:06d}.pkl"
    )


def _bank_path_for_row(base_root: Path, row: dict) -> Path:
    ckpt_path = _checkpoint_path_for_row(base_root, row)
    source_step = int(row.get("source_step", row["step"]))
    return shared_fig2_delta_bank_path(
        ckpt_path,
        protocol_id=str(row.get("protocol_id", "shared_fig2_single")),
        source_training_lambda=float(
            row.get("source_training_lambda", SHARED_CHECKPOINT_SOURCE_TRAIN_LAMBDA)
        ),
        source_step=source_step,
        diagnostic_lambda=float(row.get("diagnostic_lambda", row["lambda"])),
        repeat_count=int(row.get("repeat_count", row.get("m_deltas", 0))),
    )


def _system_recompute_settings(args, system_name: str) -> dict[str, int | None]:
    if system_name == "j1j2":
        return {
            "validation_round_samples": args.j1j2_validation_round_samples,
            "jvp_chunk": args.j1j2_jvp_chunk,
            "conn_chunk": args.j1j2_conn_chunk,
            "eval_chunk": args.j1j2_eval_chunk,
            "delta_batch_size": args.j1j2_delta_batch_size,
        }
    if system_name == "tfim":
        return {
            "validation_round_samples": args.tfim_validation_round_samples,
            "jvp_chunk": args.tfim_jvp_chunk,
            "conn_chunk": None,
            "eval_chunk": None,
            "delta_batch_size": args.tfim_delta_batch_size,
        }
    raise ValueError(f"Unsupported system_name={system_name!r}")


def _rval_per_delta_metrics(prediction_bank, energy_channels, *, n_replicas: int | None):
    residual_bank = np.asarray(prediction_bank, dtype=np.float64) - 2.0 * np.asarray(
        energy_channels, dtype=np.float64
    )[None, ...]
    residual_centered, total_count = center_samples(residual_bank, n_replicas=n_replicas)
    residual_axes = tuple(range(1, residual_centered.ndim))
    residual_raw_per_delta = (
        np.square(residual_centered).sum(axis=residual_axes) / float(total_count)
    )

    energy_centered, energy_total_count = center_samples(energy_channels, n_replicas=n_replicas)
    energy_variance = float(np.square(energy_centered).sum() / float(energy_total_count))
    target_norm_sq = 4.0 * energy_variance
    residual_norm_per_delta = residual_raw_per_delta / (target_norm_sq + EPS)
    return {
        "r_val_per_delta": np.asarray(residual_raw_per_delta, dtype=np.float64),
        "r_val_norm_per_delta": np.asarray(residual_norm_per_delta, dtype=np.float64),
        "target_norm_sq": float(target_norm_sq),
        "energy_variance": float(energy_variance),
    }


def _compute_rval_per_delta_for_row(base_root: Path, row: dict, args, *, cache_dir: Path) -> dict:
    from sr_filtering_nqs.large_scale.core.diagnostic_core import (
        _stream_prediction_cache,
        compute_driver_local_energies_round,
        compute_vit_local_energies_round,
        resolve_round_samples,
    )

    system_name = normalize_system_name(row["system"])
    settings = _system_recompute_settings(args, system_name)
    cache_path = _per_delta_cache_path(cache_dir, row)
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        cached_chunk_size_bwd = cached.get("chunk_size_bwd")
        requested_chunk_size_bwd = None if args.chunk_size_bwd is None else int(args.chunk_size_bwd)
        if (
            int(cached.get("n_val", -1)) == int(row["n_val"])
            and dict(cached.get("recompute_settings", {})) == dict(settings)
            and cached_chunk_size_bwd == requested_chunk_size_bwd
        ):
            return cached

    ckpt_path = _checkpoint_path_for_row(base_root, row)
    bank_path = _bank_path_for_row(base_root, row)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing source checkpoint for per-delta R_val: {ckpt_path}")
    if not bank_path.exists():
        raise FileNotFoundError(f"Missing shared delta bank for per-delta R_val: {bank_path}")

    with bank_path.open("rb") as f:
        bank_payload = pickle.load(f)

    n_val = int(row["n_val"])
    print(
        f"Recomputing per-delta R_val: system={system_name} step={int(row.get('source_step', row['step']))} "
        f"lambda={float(row.get('diagnostic_lambda', row['lambda'])):.3e}",
        flush=True,
    )

    if system_name == "j1j2":
        from sr_filtering_nqs.large_scale.core.run_diagnostics import (
            load_checkpoint,
            make_vit_state,
            reconstruct_driver,
            resolve_n_samples_for_config,
        )

        ckpt = load_checkpoint(ckpt_path)
        step = int(ckpt["step"])
        config = ckpt["config"]
        driver, vstate = reconstruct_driver(
            config,
            ckpt["parameters"],
            variables=ckpt.get("variables"),
            step=step,
            chunk_size_bwd_override=args.chunk_size_bwd,
        )
        n_train = resolve_n_samples_for_config(config, step=step)
        round_samples = resolve_round_samples(
            system_name=system_name,
            total_n_val=n_val,
            requested_round_samples=settings["validation_round_samples"],
            default_round_samples=n_train,
        )
        if round_samples == n_train:
            validation_vstate = vstate
        else:
            _, _, validation_vstate = make_vit_state(
                config,
                ckpt["parameters"],
                variables=ckpt.get("variables"),
                step=step,
                n_samples_override=round_samples,
                chunk_size_override=min(int(config.get("chunk_size", 4096)), round_samples),
                seed_override=int(config.get("seed", 0)) + 20_000 + step,
            )

        def compute_eloc_round(samples_round):
            return compute_vit_local_energies_round(
                validation_vstate._apply_fun,
                validation_vstate.variables,
                samples_round,
                driver._ham,
                conn_chunk=int(settings["conn_chunk"]),
                eval_chunk=int(settings["eval_chunk"]),
            )

        n_replicas = None
    else:
        from sr_filtering_nqs.large_scale.core.tfim_fnqs import (
            load_or_create_heldout_h_values,
            make_state_from_config,
            reconstruct_training_driver,
        )

        with ckpt_path.open("rb") as f:
            ckpt = pickle.load(f)
        step = int(ckpt["step"])
        config = ckpt["config"]
        driver, vstate = reconstruct_training_driver(ckpt)
        if args.chunk_size_bwd is not None:
            driver.chunk_size_bwd = int(args.chunk_size_bwd)
        heldout_h = load_or_create_heldout_h_values(base_root, config)
        n_replicas = int(len(heldout_h))
        round_samples = resolve_round_samples(
            system_name=system_name,
            total_n_val=n_val,
            requested_round_samples=settings["validation_round_samples"],
            default_round_samples=int(config["n_samples"]),
            n_replicas=n_replicas,
        )
        validation_vstate = make_state_from_config(
            config,
            parameter_array=heldout_h,
            n_samples=round_samples,
            seed=int(config.get("seed", 0)) + 20_000 + step,
            chunk_size=min(
                int(config.get("validation_chunk_size", config.get("chunk_size", round_samples))),
                round_samples,
            ),
        )
        validation_vstate.variables = vstate.variables

        def compute_eloc_round(samples_round):
            return compute_driver_local_energies_round(driver, validation_vstate, samples_round)

    prediction_bank, energy_channels = _stream_prediction_cache(
        driver=driver,
        validation_vstate=validation_vstate,
        delta_bank=bank_payload["delta_bank"],
        mode=driver.mode,
        n_val=n_val,
        round_samples=int(round_samples),
        jvp_chunk=settings["jvp_chunk"],
        delta_batch_size=int(settings["delta_batch_size"]),
        n_replicas=n_replicas,
        compute_eloc_round=compute_eloc_round,
    )
    per_delta_metrics = _rval_per_delta_metrics(
        prediction_bank,
        energy_channels,
        n_replicas=n_replicas,
    )
    cache_payload = {
        "system": system_name,
        "step": int(step),
        "source_step": int(row.get("source_step", row["step"])),
        "lambda": float(row.get("diagnostic_lambda", row["lambda"])),
        "repeat_count": int(row.get("repeat_count", row.get("m_deltas", 0))),
        "n_val": n_val,
        "recompute_settings": dict(settings),
        "chunk_size_bwd": None if args.chunk_size_bwd is None else int(args.chunk_size_bwd),
        "r_val_per_delta": [float(x) for x in per_delta_metrics["r_val_per_delta"]],
        "r_val_norm_per_delta": [float(x) for x in per_delta_metrics["r_val_norm_per_delta"]],
        "target_norm_sq": float(per_delta_metrics["target_norm_sq"]),
        "energy_variance": float(per_delta_metrics["energy_variance"]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache_payload, f)
    return cache_payload


def _load_rval_per_delta_rows(base_root: Path, rows: list[dict], args, *, output_dir: Path):
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else output_dir / ".rval_per_delta_cache"
    expanded_rows = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        system_name = normalize_system_name(row["system"])
        source_step = int(row.get("source_step", row["step"]))
        diagnostic_lambda = float(row.get("diagnostic_lambda", row["lambda"]))
        print(
            f"[{index}/{total}] Loading per-delta R_val for {system_name} step={source_step} "
            f"lambda={diagnostic_lambda:.3e}",
            flush=True,
        )
        payload = _compute_rval_per_delta_for_row(base_root, row, args, cache_dir=cache_dir)
        for delta_index, value in enumerate(payload["r_val_per_delta"]):
            expanded_rows.append(
                {
                    "system": system_name,
                    "lambda": diagnostic_lambda,
                    "source_step": source_step,
                    "delta_index": int(delta_index),
                    "r_val_per_delta": float(value),
                }
            )
    return expanded_rows


def _render_metric(
    ax: plt.Axes,
    *,
    system_name: str,
    summary,
    ylabel: str,
    style: str,
    yscale: str,
):
    by_lambda = summary.get(system_name, {})
    ax.set_title(SYSTEM_TITLES[system_name])
    ax.set_xscale("log")
    ax.set_yscale(yscale)
    ax.set_xlabel(r"Solve-time $\lambda$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.2)
    if not by_lambda:
        ax.text(0.5, 0.5, "No shared diagnostics", ha="center", va="center", transform=ax.transAxes)
        return

    lambdas = np.asarray(sorted(by_lambda), dtype=np.float64)
    means = np.asarray([by_lambda[lam]["mean"] for lam in lambdas], dtype=np.float64)
    stds = np.asarray([by_lambda[lam]["std"] for lam in lambdas], dtype=np.float64)
    if style == "errorbar":
        ax.errorbar(
            lambdas,
            means,
            yerr=stds,
            fmt="o-",
            linewidth=2.0,
            elinewidth=1.4,
            capsize=4.0,
            color="#1f4e79",
        )
    else:
        ax.plot(lambdas, means, marker="o", linewidth=2.0, color="#1f4e79")
        ax.fill_between(
            lambdas,
            np.maximum(means - stds, 1e-30),
            means + stds,
            color="#1f4e79",
            alpha=0.18,
        )
    ax.set_xticks(lambdas)
    ax.set_xticklabels([_lambda_label(lam) for lam in lambdas], rotation=35, ha="right")


def main():
    args = parse_args()
    base_root = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rows = _load_metric_rows(
        base_root,
        protocol_id=args.protocol_id,
        source_training_lambda=float(args.source_train_lambda),
        repeat_count=int(args.repeat_count),
    )
    filtered_rows = {
        metric: _filter_rows(metric_rows, checkpoint_subset=args.checkpoint_subset)
        for metric, metric_rows in rows.items()
    }
    rval_rows = filtered_rows["rval"]
    if args.rval_mode == "per_delta":
        rval_rows = _load_rval_per_delta_rows(base_root, rval_rows, args, output_dir=output_dir)
    rval_summary = _summaries(
        _group_metric_rows(
            rval_rows,
            value_key=_metric_value_key("rval", rval_mode=args.rval_mode),
        )
    )
    twobatch_summary = _summaries(_group_metric_rows(filtered_rows["twobatch"], value_key="var_hat_raw"))

    subset_suffix = "" if args.checkpoint_subset == "all" else f"_{args.checkpoint_subset}"
    if args.rval_mode == "per_delta":
        summary_csv = output_dir / f"figure2_shared_checkpoint_rval_per_delta_summary{subset_suffix}.csv"
        _write_summary_csv(summary_csv, rval_summary=rval_summary, twobatch_summary={})
    else:
        summary_csv = output_dir / f"figure2_shared_checkpoint_raw_summary{subset_suffix}.csv"
        _write_summary_csv(summary_csv, rval_summary=rval_summary, twobatch_summary=twobatch_summary)

    title_suffix = {
        "all": "",
        "early5": " (Early 5 Checkpoints)",
        "late5": " (Late 5 Checkpoints)",
    }[args.checkpoint_subset]
    if args.rval_mode == "per_delta":
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), constrained_layout=True)
        for col, system_name in enumerate(SYSTEM_ORDER):
            _render_metric(
                axes[col],
                system_name=system_name,
                summary=rval_summary,
                ylabel=r"Raw $R_{\mathrm{val}}$",
                style=args.style,
                yscale=args.rval_yscale,
            )
        fig.suptitle(
            f"Shared-checkpoint Figure 2 raw $R_{{\\mathrm{{val}}}}$ without averaging over $m${title_suffix}",
            fontsize=14,
        )
    else:
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
        for col, system_name in enumerate(SYSTEM_ORDER):
            _render_metric(
                axes[0, col],
                system_name=system_name,
                summary=rval_summary,
                ylabel=r"Raw $R_{\mathrm{val}}$",
                style=args.style,
                yscale=args.rval_yscale,
            )
            _render_metric(
                axes[1, col],
                system_name=system_name,
                summary=twobatch_summary,
                ylabel=r"Raw $V_{\mathrm{mb}}$",
                style=args.style,
                yscale=args.twobatch_yscale,
            )
        fig.suptitle(f"Shared-checkpoint Figure 2 diagnostics{title_suffix}", fontsize=14)

    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_errorbar" if args.style == "errorbar" else ""
    if args.rval_yscale != "log":
        suffix += f"_rval{args.rval_yscale}"
    if args.rval_mode == "mean_over_m" and args.twobatch_yscale != "log":
        suffix += f"_vmb{args.twobatch_yscale}"
    if args.rval_mode == "per_delta":
        out_path = output_dir / f"figure2_shared_checkpoint_rval_per_delta{subset_suffix}{suffix}.pdf"
    else:
        out_path = output_dir / f"figure2_shared_checkpoint_raw{subset_suffix}{suffix}.pdf"
    fig.savefig(out_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
