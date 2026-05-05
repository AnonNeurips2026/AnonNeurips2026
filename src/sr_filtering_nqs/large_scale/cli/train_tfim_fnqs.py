"""
Canonical large-scale TFIM FNQS trainer.

Run from ``nqs_support_core/`` with the project environment, for example:

    uv run python ../large_scale/cli/train_tfim_fnqs.py --diag-shift 1e-3
    uv run python ../large_scale/cli/train_tfim_fnqs.py --sweep

Outputs are written to:

    large_scale/tfim_results/<lambda>/results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from advanced_drivers.callbacks import AbstractCallback
from netket.utils import struct

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core import sr_diagnostics
from sr_filtering_nqs.large_scale.core.common import (
    ACTIVE_SYSTEM_CHOICES,
    DEFAULT_SR_LAMBDA_GRID,
    NGD_METHODS,
    SR_VARIANTS,
    SR_WEIGHT_SCHEMES,
    default_pii_tau,
    normalize_system_name,
    parse_sr_lambda_grid,
    results_dir_for_mixture_run,
    results_dir_for_run,
)
from sr_filtering_nqs.large_scale.core.multilambda_sr import MultiLambdaSRCallback, validate_sr_configuration
from sr_filtering_nqs.large_scale.core.pii_update import PIIUpdateCallback, RGNUpdateCallback
from sr_filtering_nqs.large_scale.core.tfim_fnqs import (
    TFIM_CHECKPOINT_EVERY,
    TFIM_H_HELDOUT_SEED,
    TFIM_H_MAX,
    TFIM_H_MIN,
    TFIM_LENGTH,
    TFIM_LEARNING_RATE,
    TFIM_LINEAR_SOLVER,
    TFIM_MODEL_CONFIG,
    TFIM_ON_THE_FLY,
    TFIM_SYSTEM,
    TFIM_TOTAL_STEPS,
    TFIM_TRAIN_H_COUNT,
    TFIM_TRAIN_SAMPLES,
    TFIM_DIAGNOSTIC_H_COUNT,
    TFIM_DIAGNOSTIC_SAMPLES_PER_H,
    TFIM_USE_NTK,
    build_run_config,
    count_parameters,
    lambda_grid,
    make_driver_from_config,
    make_state_from_config,
    make_training_parameter_array,
    restore_training_state,
    serialize_state,
)


CSV_HEADER = [
    "step",
    "phase_index",
    "phase_name",
    "symmetry_index",
    "energy",
    "energy_per_site",
    "energy_sigma",
    "energy_variance",
    "r_train",
    "r_train_norm",
    "r_train_target_norm_sq",
    "energy_variance_train",
    "learning_rate",
    "acceptance",
    "step_wall_time_s",
    "elapsed_wall_time_s",
]


class TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_stdout(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(log_path, "a", buffering=1) as log_file:
        tee = TeeStream(old_stdout, log_file)
        sys.stdout = tee
        sys.stderr = tee
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def json_ready(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return [json_ready(x) for x in obj]
    if isinstance(obj, list):
        return [json_ready(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(json_ready(payload), f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def atomic_write_pickle(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".pkl.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def read_existing_steps(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {int(row["step"]) for row in csv.DictReader(f)}


def append_metrics_row(csv_path: Path, row: dict):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def learning_rate_at_step(run_config: dict, step: int) -> float:
    init_lr = float(run_config["learning_rate"])
    if run_config.get("lr_schedule", "constant") == "constant":
        return init_lr
    total_steps = max(1, int(run_config["total_steps"]))
    progress = min(max(int(step), 0), total_steps) / float(total_steps)
    return float(init_lr * 0.5 * (1.0 + np.cos(np.pi * progress)))


def _multilambda_checkpoint_info(driver):
    info = getattr(driver, "info", None)
    if not isinstance(info, dict):
        return None
    multilambda = info.get("multilambda")
    if multilambda is None:
        return None
    return json_ready(multilambda)


def _pii_checkpoint_info(driver):
    info = getattr(driver, "info", None)
    if not isinstance(info, dict):
        return None
    pii = info.get("pii")
    if pii is None:
        return None
    return json_ready(pii)


def _rgn_checkpoint_info(driver):
    info = getattr(driver, "info", None)
    if not isinstance(info, dict):
        return None
    rgn = info.get("rgn")
    if rgn is None:
        return None
    return json_ready(rgn)


def _optimizer_checkpoint_info(driver):
    return _rgn_checkpoint_info(driver) or _pii_checkpoint_info(driver) or _multilambda_checkpoint_info(driver)


def _effective_driver_diag_shift(sr_variant: str, diag_shift: float | None, sr_lambda_grid) -> float:
    if sr_variant == "single":
        if diag_shift is None:
            raise ValueError("single SR requires --diag-shift or --sweep")
        return float(diag_shift)
    grid = parse_sr_lambda_grid(sr_lambda_grid)
    return float(max(grid))


def update_summary(
    results_dir: Path,
    run_config: dict,
    *,
    status: str,
    latest_metrics: dict | None,
    checkpoint_steps_saved: list[int],
    error: str | None = None,
):
    summary = {
        "status": status,
        "system": run_config["system"],
        "model_type": run_config["model_type"],
        "diag_shift": run_config["diag_shift"],
        "sr_variant": run_config["sr_variant"],
        "sr_weight_scheme": run_config["sr_weight_scheme"],
        "sr_lambda_grid": run_config["sr_lambda_grid"],
        "sr_lambda_quantiles": run_config["sr_lambda_quantiles"],
        "sr_lambda_ema_rho": run_config["sr_lambda_ema_rho"],
        "sr_weight_ema_rho": run_config["sr_weight_ema_rho"],
        "ngd_method": run_config["ngd_method"],
        "spring_momentum": run_config["spring_momentum"],
        "pii_tau": run_config["pii_tau"],
        "pii_diag_shift": run_config["pii_diag_shift"],
        "init_from": run_config["init_from"],
        "init_from_step": run_config["init_from_step"],
        "total_steps": run_config["total_steps"],
        "checkpoint_steps": run_config["checkpoint_steps"],
        "diagnostic_steps": run_config["diagnostic_steps"],
        "checkpoint_steps_saved": checkpoint_steps_saved,
        "updated_at_epoch_s": time.time(),
    }
    if latest_metrics is not None:
        summary["latest_metrics"] = latest_metrics
    if error is not None:
        summary["error"] = error
    atomic_write_json(results_dir / "summary.json", summary)


def checkpoint_config(run_config: dict, step: int) -> dict:
    config = dict(run_config)
    config["step"] = int(step)
    return config


def save_checkpoint(results_dir: Path, step: int, driver, run_config: dict, metrics: dict):
    checkpoint_path = results_dir / "checkpoints" / f"checkpoint_step{step:06d}.pkl"
    payload = {
        "step": int(step),
        "phase_index": 0,
        "phase_name": "fnqs",
        "symmetry_index": 0,
        "state_dict": serialize_state(driver.state),
        "config": checkpoint_config(run_config, step),
        "train_metrics": metrics,
        "multilambda": _multilambda_checkpoint_info(driver),
        "pii": _pii_checkpoint_info(driver),
        "rgn": _rgn_checkpoint_info(driver),
        "optimizer_update": _optimizer_checkpoint_info(driver),
    }
    atomic_write_pickle(checkpoint_path, payload)
    return checkpoint_path


def save_latest_checkpoint(results_dir: Path, step: int, driver, run_config: dict, metrics: dict):
    checkpoint_path = results_dir / "checkpoints" / "checkpoint_latest.pkl"
    payload = {
        "step": int(step),
        "phase_index": 0,
        "phase_name": "fnqs",
        "symmetry_index": 0,
        "state_dict": serialize_state(driver.state),
        "config": checkpoint_config(run_config, step),
        "train_metrics": metrics,
        "multilambda": _multilambda_checkpoint_info(driver),
        "pii": _pii_checkpoint_info(driver),
        "rgn": _rgn_checkpoint_info(driver),
        "optimizer_update": _optimizer_checkpoint_info(driver),
    }
    atomic_write_pickle(checkpoint_path, payload)
    return checkpoint_path


@dataclass
class ResumeState:
    step: int = 0
    checkpoint: dict | None = None
    source_step: int | None = None
    source_path: str | None = None


class TrainMetricsCallback(AbstractCallback, mutable=True):
    _results_dir: str = struct.field(pytree_node=False)
    _run_config: dict = struct.field(pytree_node=False)
    _checkpoint_steps: tuple = struct.field(pytree_node=False)
    _existing_metric_steps: set = struct.field(pytree_node=False)
    _saved_checkpoint_steps: list = struct.field(pytree_node=False)
    _latest_checkpoint_every: int = struct.field(pytree_node=False)
    _step_offset: int = struct.field(pytree_node=False)
    _step_t0: float = struct.field(pytree_node=False)
    _run_t0: float = struct.field(pytree_node=False)
    _last_metrics: dict = struct.field(pytree_node=False)

    def __init__(
        self,
        *,
        results_dir: Path,
        run_config: dict,
        checkpoint_steps: tuple[int, ...],
        existing_metric_steps: set[int],
        saved_checkpoint_steps: list[int],
        latest_checkpoint_every: int,
        step_offset: int,
        run_start_time: float,
    ):
        self._results_dir = str(results_dir)
        self._run_config = run_config
        self._checkpoint_steps = checkpoint_steps
        self._existing_metric_steps = existing_metric_steps
        self._saved_checkpoint_steps = saved_checkpoint_steps
        self._latest_checkpoint_every = latest_checkpoint_every
        self._step_offset = step_offset
        self._step_t0 = 0.0
        self._run_t0 = run_start_time
        self._last_metrics = {}

    @property
    def latest_metrics(self):
        return self._last_metrics

    def on_step_start(self, step, log_data, driver):
        self._step_t0 = time.time()

    def on_compute_update_end(self, step, log_data, driver):
        loss_stats = driver._loss_stats
        if loss_stats is None:
            return

        energy = float(np.real(complex(loss_stats.Mean)))
        variance = float(np.real(complex(loss_stats.Variance)))
        sigma = float(np.real(complex(loss_stats.Sigma)))
        completed_step = self._step_offset + step + 1

        if np.isnan(energy) or np.isnan(variance):
            driver._stop_run = True
            self._last_metrics = {
                "step": completed_step,
                "phase_index": 0,
                "phase_name": "fnqs",
                "status": "nan",
            }
            return

        optimizer_update = None
        if isinstance(getattr(driver, "info", None), dict):
            optimizer_update = driver.info.get("rgn") or driver.info.get("pii") or driver.info.get("multilambda")

        if isinstance(optimizer_update, dict) and isinstance(optimizer_update.get("train_metrics"), dict):
            train_metrics = {
                key: float(optimizer_update["train_metrics"].get(key, float("nan")))
                for key in ("residual_raw", "residual_norm", "target_norm_sq", "energy_variance")
            }
        elif completed_step in self._checkpoint_steps:
            train_metrics = sr_diagnostics.compute_residual_metrics(
                driver, driver.state, driver.state.samples, driver._dp
            )
        else:
            train_metrics = {
                "residual_raw": float("nan"),
                "residual_norm": float("nan"),
                "target_norm_sq": float("nan"),
                "energy_variance": float("nan"),
            }

        log_data["r_train"] = train_metrics["residual_raw"]
        log_data["r_train_norm"] = train_metrics["residual_norm"]
        log_data["energy_variance_train"] = train_metrics["energy_variance"]

        self._last_metrics = {
            "step": completed_step,
            "phase_index": 0,
            "phase_name": "fnqs",
            "symmetry_index": 0,
            "energy": energy,
            "energy_per_site": energy / self._run_config["length"],
            "energy_sigma": sigma,
            "energy_variance": variance,
            "r_train": train_metrics["residual_raw"],
            "r_train_norm": train_metrics["residual_norm"],
            "r_train_target_norm_sq": train_metrics["target_norm_sq"],
            "energy_variance_train": train_metrics["energy_variance"],
        }

    def on_step_end(self, step, log_data, driver):
        completed_step = self._step_offset + step + 1
        if not self._last_metrics:
            return

        acceptance = log_data.get("acceptance")
        if acceptance is not None:
            acceptance = float(np.asarray(acceptance))

        row = dict(self._last_metrics)
        row.update(
            {
                "learning_rate": learning_rate_at_step(self._run_config, completed_step),
                "acceptance": acceptance,
                "step_wall_time_s": time.time() - self._step_t0,
                "elapsed_wall_time_s": time.time() - self._run_t0,
            }
        )

        if completed_step not in self._existing_metric_steps:
            append_metrics_row(Path(self._results_dir) / "train_metrics.csv", row)
            self._existing_metric_steps.add(completed_step)

        if completed_step % 10 == 0 or completed_step in self._checkpoint_steps:
            print(
                f"  step={completed_step:4d} "
                f"E/N={row['energy_per_site']:.6f} Var={row['energy_variance']:.4f} "
                f"r_train={row['r_train']:.4f} lr={row['learning_rate']:.5f}",
                flush=True,
            )

        if completed_step in self._checkpoint_steps and completed_step not in self._saved_checkpoint_steps:
            checkpoint_path = save_checkpoint(
                Path(self._results_dir),
                completed_step,
                driver,
                self._run_config,
                row,
            )
            self._saved_checkpoint_steps.append(completed_step)
            print(f"  checkpoint saved: {checkpoint_path.name}", flush=True)

        if (
            self._latest_checkpoint_every > 0
            and completed_step % self._latest_checkpoint_every == 0
            and completed_step not in self._checkpoint_steps
        ):
            checkpoint_path = save_latest_checkpoint(
                Path(self._results_dir),
                completed_step,
                driver,
                self._run_config,
                row,
            )
            print(f"  latest checkpoint updated: {checkpoint_path.name}", flush=True)

        if completed_step % 10 == 0 or completed_step in self._checkpoint_steps:
            update_summary(
                Path(self._results_dir),
                self._run_config,
                status="running",
                latest_metrics=row,
                checkpoint_steps_saved=self._saved_checkpoint_steps,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Canonical TFIM FNQS sweep")
    parser.add_argument("--system", default=TFIM_SYSTEM, choices=ACTIVE_SYSTEM_CHOICES)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--diag-shift", type=float, help="Single lambda value")
    group.add_argument("--sweep", action="store_true", help="Run the full lambda grid")
    parser.add_argument("--output-root", type=str, default=str(ROOT / "large_scale"))
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint path")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Initialize state from a checkpoint and start a new branch at local step 0.",
    )
    parser.add_argument("--length", type=int, default=TFIM_LENGTH)
    parser.add_argument("--h-min", type=float, default=TFIM_H_MIN)
    parser.add_argument("--h-max", type=float, default=TFIM_H_MAX)
    parser.add_argument("--train-h-count", type=int, default=TFIM_TRAIN_H_COUNT)
    parser.add_argument("--train-total-samples", type=int, default=TFIM_TRAIN_SAMPLES)
    parser.add_argument("--validation-h-count", type=int, default=TFIM_DIAGNOSTIC_H_COUNT)
    parser.add_argument(
        "--validation-samples-per-h",
        type=int,
        default=TFIM_DIAGNOSTIC_SAMPLES_PER_H,
    )
    parser.add_argument("--learning-rate", type=float, default=TFIM_LEARNING_RATE)
    parser.add_argument("--lr-schedule", type=str, default="constant", choices=["constant", "cosine"])
    parser.add_argument("--linear-solver", type=str, default=TFIM_LINEAR_SOLVER, choices=["cholesky", "pinv_smooth", "pinv", "LU"])
    parser.add_argument("--use-ntk", action="store_true", default=TFIM_USE_NTK)
    parser.add_argument("--on-the-fly", action="store_true", default=TFIM_ON_THE_FLY)
    parser.add_argument("--total-steps", type=int, default=TFIM_TOTAL_STEPS)
    parser.add_argument("--checkpoint-every", type=int, default=TFIM_CHECKPOINT_EVERY)
    parser.add_argument("--num-layers", type=int, default=TFIM_MODEL_CONFIG["num_layers"])
    parser.add_argument("--d-model", type=int, default=TFIM_MODEL_CONFIG["d_model"])
    parser.add_argument("--heads", type=int, default=TFIM_MODEL_CONFIG["heads"])
    parser.add_argument("--patch-size", type=int, default=TFIM_MODEL_CONFIG["b"])
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-size-bwd", type=int, default=None)
    parser.add_argument(
        "--latest-checkpoint-every",
        type=int,
        default=0,
        help=(
            "Overwrite checkpoints/checkpoint_latest.pkl every N completed steps "
            "between permanent checkpoints. Disabled when set to 0."
        ),
    )
    parser.add_argument("--heldout-h-seed", type=int, default=TFIM_H_HELDOUT_SEED)
    parser.add_argument("--sr-variant", type=str, default="single", choices=SR_VARIANTS)
    parser.add_argument("--sr-weight-scheme", type=str, default="uniform", choices=SR_WEIGHT_SCHEMES)
    parser.add_argument(
        "--sr-lambda-grid",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SR_LAMBDA_GRID),
        help="Comma-separated lambda grid for multi-lambda SR variants.",
    )
    parser.add_argument(
        "--sr-weight-fit-samples",
        type=int,
        default=None,
        help="Weight-fit sample count for independent-batch stacking. Defaults to N_s.",
    )
    parser.add_argument(
        "--sr-lambda-quantiles",
        type=str,
        default=None,
        help="Comma-separated NTK spectrum quantiles for a per-step dynamic multi-lambda grid.",
    )
    parser.add_argument(
        "--sr-lambda-ema-rho",
        type=float,
        default=None,
        help="EMA rho for dynamic NTK-quantile lambdas.",
    )
    parser.add_argument(
        "--sr-weight-ema-rho",
        type=float,
        default=None,
        help="EMA rho for multi-lambda mixture weights.",
    )
    parser.add_argument("--ngd-method", type=str, default="sr", choices=NGD_METHODS)
    parser.add_argument("--spring-momentum", type=float, default=0.9)
    parser.add_argument("--spring-proj-reg", type=float, default=None)
    parser.add_argument("--pii-tau", type=float, default=None)
    parser.add_argument("--pii-diag-shift", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_resume_state(path: str, *, reset_step: bool = False) -> ResumeState:
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    source_step = int(checkpoint.get("step", checkpoint.get("config", {}).get("step", 0)))
    return ResumeState(
        step=0 if reset_step else source_step,
        checkpoint=checkpoint,
        source_step=source_step,
        source_path=str(path),
    )


def validate_resume_config(run_config: dict, checkpoint: dict):
    resume_config = checkpoint["config"]
    keys = [
        "system",
        "diag_shift",
        "length",
        "h_min",
        "h_max",
        "train_h_count",
        "n_samples",
        "use_ntk",
        "on_the_fly",
        "linear_solver",
        "total_steps",
        "checkpoint_every",
        "lr_schedule",
        "model_config",
        "sr_variant",
            "sr_weight_scheme",
            "sr_lambda_grid",
            "sr_weight_fit_samples",
            "sr_lambda_quantiles",
            "sr_lambda_ema_rho",
            "sr_weight_ema_rho",
            "ngd_method",
        "spring_momentum",
        "spring_proj_reg",
        "pii_tau",
        "pii_diag_shift",
    ]
    for key in keys:
        if json_ready(run_config[key]) != json_ready(resume_config[key]):
            raise ValueError(
                f"Resume checkpoint mismatch for {key}: "
                f"{resume_config[key]!r} != {run_config[key]!r}"
            )


def run_single(args, system_name: str, diag_shift: float):
    validate_sr_configuration(args.sr_variant, args.sr_weight_scheme)
    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume and --init-from cannot be combined")
    if args.ngd_method != "sr" and args.sr_variant != "single":
        raise ValueError(f"--ngd-method {args.ngd_method} requires --sr-variant single")
    if args.on_the_fly and not args.use_ntk:
        raise ValueError("--on-the-fly requires --use-ntk")
    pii_tau = None
    if args.ngd_method == "pii":
        pii_tau = (
            float(args.pii_tau)
            if args.pii_tau is not None
            else default_pii_tau(system_name, args.length)
        )
    if args.sr_variant == "single":
        results_dir = results_dir_for_run(Path(args.output_root), system_name, diag_shift)
    else:
        results_dir = results_dir_for_mixture_run(
            Path(args.output_root),
            system_name,
            sr_variant=args.sr_variant,
            sr_weight_scheme=args.sr_weight_scheme,
            sr_lambda_grid=parse_sr_lambda_grid(args.sr_lambda_grid),
        )
    checkpoints_dir = results_dir / "checkpoints"
    diagnostics_dir = results_dir / "diagnostics"
    logs_dir = results_dir / "logs"
    for directory in (results_dir, checkpoints_dir, diagnostics_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_config = build_run_config(
        diag_shift=diag_shift,
        output_root=Path(args.output_root),
        seed=args.seed,
        length=args.length,
        h_min=args.h_min,
        h_max=args.h_max,
        train_h_count=args.train_h_count,
        train_total_samples=args.train_total_samples,
        validation_h_count=args.validation_h_count,
        validation_samples_per_h=args.validation_samples_per_h,
        learning_rate=args.learning_rate,
        lr_schedule=args.lr_schedule,
        use_ntk=args.use_ntk,
        on_the_fly=args.on_the_fly,
        linear_solver=args.linear_solver,
        total_steps=args.total_steps,
        checkpoint_every=args.checkpoint_every,
        chunk_size=args.chunk_size,
        chunk_size_bwd=args.chunk_size_bwd,
        heldout_h_seed=args.heldout_h_seed,
        sr_variant=args.sr_variant,
        sr_weight_scheme=args.sr_weight_scheme,
        sr_lambda_grid=parse_sr_lambda_grid(args.sr_lambda_grid),
        sr_weight_fit_samples=args.sr_weight_fit_samples,
        sr_lambda_quantiles=(
            None
            if args.sr_lambda_quantiles is None
            else parse_sr_lambda_grid(args.sr_lambda_quantiles)
        ),
        sr_lambda_ema_rho=args.sr_lambda_ema_rho,
        sr_weight_ema_rho=args.sr_weight_ema_rho,
        ngd_method=args.ngd_method,
        spring_momentum=args.spring_momentum,
        spring_proj_reg=args.spring_proj_reg,
        pii_tau=pii_tau,
        pii_diag_shift=args.pii_diag_shift,
        init_from=args.init_from,
        model_config={
            "num_layers": args.num_layers,
            "d_model": args.d_model,
            "heads": args.heads,
            "b": args.patch_size,
        },
    )
    checkpoint_steps = tuple(int(step) for step in run_config["checkpoint_steps"])

    log_path = logs_dir / "train.log"
    with tee_stdout(log_path):
        existing_ckpts = sorted(checkpoints_dir.glob("checkpoint_step*.pkl"))
        if existing_ckpts and args.resume is None:
            raise RuntimeError(
                f"{results_dir} already contains checkpoints; pass --resume or clean the run directory"
            )

        resume_state = ResumeState()
        if args.resume is not None:
            resume_state = load_resume_state(args.resume)
            validate_resume_config(run_config, resume_state.checkpoint)
        elif args.init_from is not None:
            resume_state = load_resume_state(args.init_from, reset_step=True)
            run_config["init_from_step"] = resume_state.source_step

        atomic_write_json(results_dir / "config.json", run_config)
        saved_checkpoint_steps = sorted(
            int(path.stem.replace("checkpoint_step", ""))
            for path in checkpoints_dir.glob("checkpoint_step*.pkl")
        )
        existing_metric_steps = read_existing_steps(results_dir / "train_metrics.csv")

        update_summary(
            results_dir,
            run_config,
            status="running",
            latest_metrics=None,
            checkpoint_steps_saved=saved_checkpoint_steps,
        )

        if resume_state.checkpoint is not None:
            vstate = restore_training_state(
                resume_state.checkpoint["state_dict"],
                config=run_config,
                seed=args.seed,
            )
        else:
            vstate = make_state_from_config(
                run_config,
                parameter_array=make_training_parameter_array(run_config),
                n_samples=int(run_config["n_samples"]),
                seed=args.seed,
                chunk_size=int(run_config["chunk_size"]),
            )

        driver = make_driver_from_config(run_config, variational_state=vstate)
        driver.collect_residual_info = False
        n_params = count_parameters(vstate.parameters)

        print("=" * 72)
        print(f"System: {system_name} | lambda={diag_shift:.3e}")
        print(f"Output: {results_dir}")
        print(
            f"L={run_config['length']} | h in [{run_config['h_min']}, {run_config['h_max']}] "
            f"| R_train={run_config['train_h_count']} | N_s={run_config['n_samples']}"
        )
        print(
            f"Validation: R={run_config['validation_h_count']} x "
            f"{run_config['validation_samples_per_h']} = {run_config['validation_n_samples']}"
        )
        print(
            f"Model: layers={run_config['model_config']['num_layers']} "
            f"d_model={run_config['model_config']['d_model']} "
            f"heads={run_config['model_config']['heads']} "
            f"patch={run_config['model_config']['b']}"
        )
        print(
            f"Training: steps={run_config['total_steps']} checkpoint_every={run_config['checkpoint_every']} "
            f"lr={run_config['learning_rate']} schedule={run_config['lr_schedule']}"
        )
        if args.latest_checkpoint_every > 0:
            print(
                "Rolling latest checkpoint: "
                f"every {args.latest_checkpoint_every} steps -> checkpoints/checkpoint_latest.pkl"
            )
        print(
            f"SR: use_ntk={run_config['use_ntk']} on_the_fly={run_config['on_the_fly']} "
            f"solver={run_config['linear_solver']} chunk_bwd={run_config['chunk_size_bwd']}"
        )
        print(
            "SR mixing: "
            f"variant={run_config['sr_variant']} "
            f"weight_scheme={run_config['sr_weight_scheme']} "
            f"lambda_grid={run_config['sr_lambda_grid']} "
            f"lambda_quantiles={run_config['sr_lambda_quantiles']} "
            f"lambda_ema_rho={run_config['sr_lambda_ema_rho']} "
            f"weight_ema_rho={run_config['sr_weight_ema_rho']}"
        )
        print(
            "NGD method: "
            f"{run_config['ngd_method']} "
            f"spring_momentum={run_config['spring_momentum']} "
            f"pii_tau={run_config['pii_tau']} "
            f"pii_diag_shift={run_config['pii_diag_shift']}"
        )
        print(f"P={n_params:,} | P/N_s={n_params / run_config['n_samples']:.4f}")
        if args.resume:
            print(f"Resuming from {args.resume} at step {resume_state.step}")
        if args.init_from:
            print(f"Initializing from {args.init_from} source_step={resume_state.source_step}")
        print("=" * 72)

        latest_metrics = None
        run_start_time = time.time()
        remaining_steps = int(run_config["total_steps"]) - resume_state.step
        if remaining_steps <= 0:
            print("Run already complete; nothing to do.", flush=True)
            update_summary(
                results_dir,
                run_config,
                status="completed",
                latest_metrics=latest_metrics,
                checkpoint_steps_saved=saved_checkpoint_steps,
            )
            return

        callback = TrainMetricsCallback(
            results_dir=results_dir,
            run_config=run_config,
            checkpoint_steps=checkpoint_steps,
            existing_metric_steps=existing_metric_steps,
            saved_checkpoint_steps=saved_checkpoint_steps,
            latest_checkpoint_every=max(0, int(args.latest_checkpoint_every)),
            step_offset=resume_state.step,
            run_start_time=run_start_time,
        )
        multilambda_callback = None
        pii_callback = None
        if run_config["sr_variant"] != "single":
            multilambda_callback = MultiLambdaSRCallback(
                sr_variant=run_config["sr_variant"],
                sr_weight_scheme=run_config["sr_weight_scheme"],
                sr_lambda_grid=run_config["sr_lambda_grid"],
                sr_weight_fit_samples=run_config["sr_weight_fit_samples"],
                sr_lambda_quantiles=run_config["sr_lambda_quantiles"],
                sr_lambda_ema_rho=run_config["sr_lambda_ema_rho"],
                sr_weight_ema_rho=run_config["sr_weight_ema_rho"],
            )
        if run_config["ngd_method"] == "pii":
            pii_callback = PIIUpdateCallback(
                tau=run_config["pii_tau"],
                diag_shift=run_config["pii_diag_shift"],
            )
        elif run_config["ngd_method"] == "rgn":
            pii_callback = RGNUpdateCallback(
                diag_shift=run_config["pii_diag_shift"],
            )

        try:
            for local_step in range(remaining_steps):
                step_log_data = {}
                callback.on_step_start(local_step, step_log_data, driver)
                driver.reset_step()
                driver.compute_loss_and_update()
                if pii_callback is not None:
                    pii_callback.on_compute_update_end(local_step, step_log_data, driver)
                elif multilambda_callback is not None:
                    multilambda_callback.on_compute_update_end(local_step, step_log_data, driver)
                callback.on_compute_update_end(local_step, step_log_data, driver)
                if driver._stop_run:
                    break
                driver._log_additional_data(step_log_data)
                driver.update_parameters(driver._dp)
                callback.on_step_end(local_step, step_log_data, driver)
                driver._step_count += 1
        except Exception as exc:
            latest_metrics = callback.latest_metrics or latest_metrics
            update_summary(
                results_dir,
                run_config,
                status="error",
                latest_metrics=latest_metrics,
                checkpoint_steps_saved=saved_checkpoint_steps,
                error=str(exc),
            )
            raise

        latest_metrics = callback.latest_metrics or latest_metrics
        if latest_metrics is not None and latest_metrics.get("status") == "nan":
            update_summary(
                results_dir,
                run_config,
                status="error",
                latest_metrics=latest_metrics,
                checkpoint_steps_saved=saved_checkpoint_steps,
                error="NaN detected during training",
            )
            raise RuntimeError("NaN detected during training")

        update_summary(
            results_dir,
            run_config,
            status="completed",
            latest_metrics=latest_metrics,
            checkpoint_steps_saved=saved_checkpoint_steps,
        )
        if latest_metrics is not None:
            print(
                f"\nCompleted: E/N={latest_metrics['energy_per_site']:.6f} "
                f"Var={latest_metrics['energy_variance']:.4f} "
                f"r_train={latest_metrics['r_train']:.4f}",
                flush=True,
            )


def main():
    args = parse_args()
    system_name = normalize_system_name(args.system)
    if system_name != TFIM_SYSTEM:
        raise ValueError(f"train_tfim_fnqs.py only supports {TFIM_SYSTEM}, got {args.system}")

    validate_sr_configuration(args.sr_variant, args.sr_weight_scheme)
    if args.sr_variant == "single":
        if args.diag_shift is None and not args.sweep:
            raise ValueError("single SR requires --diag-shift or --sweep")
        lambda_values = lambda_grid() if args.sweep else [float(args.diag_shift)]
    else:
        if args.diag_shift is not None or args.sweep:
            raise ValueError("multi-lambda SR runs do not accept --diag-shift or --sweep")
        lambda_values = [
            _effective_driver_diag_shift(
                args.sr_variant,
                args.diag_shift,
                parse_sr_lambda_grid(args.sr_lambda_grid),
            )
        ]
    failures: list[tuple[float, str]] = []
    for diag_shift in lambda_values:
        try:
            run_single(args, system_name, float(diag_shift))
        except Exception as exc:
            failures.append((float(diag_shift), str(exc)))
            print(f"\nFAILED system={system_name} lambda={float(diag_shift):.3e}: {exc}", flush=True)
            if args.fail_fast:
                raise

    if failures:
        print("\nSweep completed with failures:", flush=True)
        for diag_shift, message in failures:
            print(f"  lambda={diag_shift:.3e}: {message}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
