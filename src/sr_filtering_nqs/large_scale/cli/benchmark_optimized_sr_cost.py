#!/usr/bin/env python3
"""Benchmark fair optimized SR, bagged SR, and paper-faithful MS-SR steps."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
import statistics
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[4]

from sr_filtering_nqs.large_scale.core.optimized_sr_benchmark import (
    DEFAULT_QUANTILES,
    eigh_pinv_solver,
    make_spectrum_quantile_solver,
    run_optimized_step,
)
from sr_filtering_nqs.large_scale.core.tfim_fnqs import make_driver_from_config, restore_training_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--method", choices=("sr", "bagged", "mssr"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--diag-shift", type=float, default=1e-4)
    parser.add_argument(
        "--quantiles",
        default=",".join(str(value) for value in DEFAULT_QUANTILES),
    )
    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument("--apply-update", dest="apply_update", action="store_true")
    update_group.add_argument("--no-apply-update", dest="apply_update", action="store_false")
    parser.set_defaults(apply_update=False)
    return parser.parse_args()


def parse_float_tuple(text: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in text.split(",") if value.strip())


def summarize(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def build_driver(checkpoint: dict, *, seed: int):
    config = dict(checkpoint["config"])
    config["seed"] = int(seed)
    config["linear_solver"] = "pinv"
    state = restore_training_state(
        checkpoint["state_dict"],
        config=config,
        seed=int(seed),
    )
    driver = make_driver_from_config(config, variational_state=state)
    driver.collect_residual_info = False
    return config, driver


def memory_stats():
    stats = jax.devices()[0].memory_stats()
    if stats is None:
        return {}
    return {
        key: int(value)
        for key, value in stats.items()
        if key in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit"}
    }


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    quantiles = parse_float_tuple(args.quantiles)
    if args.method == "mssr" and len(quantiles) != args.k:
        raise ValueError(f"Expected {args.k} quantiles, got {quantiles}")

    checkpoint_path = Path(args.checkpoint).resolve()
    try:
        checkpoint_record = str(checkpoint_path.relative_to(ROOT))
    except ValueError:
        # Keep published benchmark metadata portable and free of machine-local
        # directory prefixes when a checkpoint lives outside the repository.
        checkpoint_record = checkpoint_path.name
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    config, driver = build_driver(checkpoint, seed=args.seed)
    n_samples = int(config["n_samples"])
    spectrum_solver = make_spectrum_quantile_solver(quantiles)

    print(
        f"method={args.method} seed={args.seed} warmup={args.warmup} "
        f"repeats={args.repeats} device={jax.devices()[0]}",
        flush=True,
    )
    for index in range(args.warmup):
        result = run_optimized_step(
            driver,
            method=args.method,
            n_samples=n_samples,
            k=args.k,
            diag_shift=args.diag_shift,
            quantiles=quantiles,
            solver_fn=eigh_pinv_solver,
            spectrum_solver_fn=spectrum_solver,
            chain_prefix=f"fair_cost_seed{args.seed}",
            apply_update=args.apply_update,
        )
        print(f"warmup={index + 1} total_s={result.timings['total_s']:.6f}", flush=True)

    rows = []
    metadata_rows = []
    for index in range(args.repeats):
        result = run_optimized_step(
            driver,
            method=args.method,
            n_samples=n_samples,
            k=args.k,
            diag_shift=args.diag_shift,
            quantiles=quantiles,
            solver_fn=eigh_pinv_solver,
            spectrum_solver_fn=spectrum_solver,
            chain_prefix=f"fair_cost_seed{args.seed}",
            apply_update=args.apply_update,
        )
        row = {"repeat": index + 1, **result.timings}
        rows.append(row)
        metadata_rows.append(result.metadata)
        print(
            f"repeat={index + 1} total_s={row['total_s']:.6f} "
            f"prepare_s={row['prepare_batches_s']:.6f} "
            f"solve_s={row['candidate_solves_s']:.6f} "
            f"cross_s={row['cross_prediction_s']:.6f} "
            f"fit_s={row['simplex_fit_s']:.6f}",
            flush=True,
        )

    timing_keys = [key for key in rows[0] if key != "repeat"]
    payload = {
        "schema_version": 1,
        "checkpoint": checkpoint_record,
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "method": args.method,
        "seed": int(args.seed),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "k": int(args.k),
        "diag_shift": float(args.diag_shift),
        "quantiles": [float(value) for value in quantiles],
        "apply_update": args.apply_update,
        "config": {
            "system": config["system"],
            "length": int(config["length"]),
            "n_samples": n_samples,
            "n_replicas": int(config["n_replicas"]),
            "chunk_size": int(config["chunk_size"]),
            "chunk_size_bwd": int(config["chunk_size_bwd"]),
            "model_config": config["model_config"],
            "benchmark_solver": "eigh_pinv_rtol_1e-12",
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "device": str(jax.devices()[0]),
            "device_kind": jax.devices()[0].device_kind,
            "platform": jax.devices()[0].platform,
            "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        },
        "metadata": metadata_rows[-1],
        "metadata_rows": metadata_rows,
        "rows": rows,
        "summary": {
            key: summarize(row[key] for row in rows)
            for key in timing_keys
        },
        "memory": memory_stats(),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(output_path)
    print(json.dumps({"output": str(output_path), "total_s": payload["summary"]["total_s"]}), flush=True)


if __name__ == "__main__":
    main()
