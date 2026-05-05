"""
Canonical symmetry-ramped large-scale ViT trainer.

Run from ``nqs_support_core/`` with the project environment, for example:

    uv run python ../large_scale/cli/train_sweep.py --system j1j2 --diag-shift 1e-3
    uv run python ../large_scale/cli/train_sweep.py --system j1j2 --sweep

Outputs are written to:

    large_scale/j1j2_results/<lambda>/results/
    This trainer is now the symmetry-ramped J1J2 path only.
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
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax

import advanced_drivers as advd
from advanced_drivers.callbacks import AbstractCallback
from netket.utils import struct
from nqs_nets.net.wrappers import ViTNd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core import sr_diagnostics
from sr_filtering_nqs.large_scale.core.common import (
    CHECKPOINT_STEPS,
    DEFAULT_SR_LAMBDA_GRID,
    LAMBDA_GRID,
    LATTICE_SIZE,
    LEARNING_RATE,
    NGD_METHODS,
    PHASE_SAMPLES,
    SR_VARIANTS,
    SR_WEIGHT_SCHEMES,
    TOTAL_STEPS,
    TRAIN_SAMPLES,
    VIT_D_MODEL,
    VIT_DEPTH,
    VIT_EXPANSION,
    VIT_HEADS,
    WARMUP_STEPS,
    default_pii_tau,
    make_system,
    normalize_system_name,
    phase_specs_for_system,
    parse_sr_lambda_grid,
    results_dir_for_mixture_run,
    results_dir_for_run,
)
from sr_filtering_nqs.large_scale.core.multilambda_sr import MultiLambdaSRCallback, validate_sr_configuration
from sr_filtering_nqs.large_scale.core.pii_update import PIIUpdateCallback, RGNUpdateCallback


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


def parse_int_list(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


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


def count_parameters(params) -> int:
    return int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params)))


def get_bare_params(params):
    if hasattr(params, "keys") and set(params.keys()) == {"module"}:
        return get_bare_params(params["module"])
    return params


def adapt_params_to_model(bare_params, template_params):
    if hasattr(template_params, "keys") and set(template_params.keys()) == {"module"}:
        return {"module": adapt_params_to_model(bare_params, template_params["module"])}
    return bare_params


def make_lr_schedule(init_lr: float, total_steps: int, warmup_steps: int, *, schedule: str = "cosine"):
    if schedule == "constant":
        return lambda count: float(init_lr)
    if schedule != "cosine":
        raise ValueError(f"Unsupported lr schedule: {schedule}")
    if warmup_steps > 0:
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=init_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=0.0,
        )
    return optax.cosine_decay_schedule(init_value=init_lr, decay_steps=total_steps)


def offset_schedule(schedule_fn, start_step: int):
    def wrapped(count):
        return schedule_fn(count + start_step)

    return wrapped


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


def checkpoint_config(run_config: dict, phase_spec: dict, step: int) -> dict:
    config = dict(run_config)
    config.update(
        {
            "step": step,
            "phase_index": phase_spec["phase_index"],
            "phase_name": phase_spec["name"],
            "symmetry_index": phase_spec["symmetry_index"],
            "n_samples": phase_spec["n_samples"],
        }
    )
    return config


def save_checkpoint(results_dir: Path, step: int, phase_spec: dict, driver, run_config: dict, metrics: dict):
    checkpoint_path = results_dir / "checkpoints" / f"checkpoint_step{step:06d}.pkl"
    payload = {
        "step": step,
        "phase_index": phase_spec["phase_index"],
        "phase_name": phase_spec["name"],
        "symmetry_index": phase_spec["symmetry_index"],
        "parameters": jax.tree_util.tree_map(np.asarray, driver.state.parameters),
        "variables": jax.tree_util.tree_map(np.asarray, driver.state.variables),
        "config": checkpoint_config(run_config, phase_spec, step),
        "train_metrics": metrics,
        "multilambda": _multilambda_checkpoint_info(driver),
        "pii": _pii_checkpoint_info(driver),
        "rgn": _rgn_checkpoint_info(driver),
        "optimizer_update": _optimizer_checkpoint_info(driver),
    }
    atomic_write_pickle(checkpoint_path, payload)
    return checkpoint_path


def save_latest_checkpoint(results_dir: Path, step: int, phase_spec: dict, driver, run_config: dict, metrics: dict):
    checkpoint_path = results_dir / "checkpoints" / "checkpoint_latest.pkl"
    payload = {
        "step": step,
        "phase_index": phase_spec["phase_index"],
        "phase_name": phase_spec["name"],
        "symmetry_index": phase_spec["symmetry_index"],
        "parameters": jax.tree_util.tree_map(np.asarray, driver.state.parameters),
        "variables": jax.tree_util.tree_map(np.asarray, driver.state.variables),
        "config": checkpoint_config(run_config, phase_spec, step),
        "train_metrics": metrics,
        "multilambda": _multilambda_checkpoint_info(driver),
        "pii": _pii_checkpoint_info(driver),
        "rgn": _rgn_checkpoint_info(driver),
        "optimizer_update": _optimizer_checkpoint_info(driver),
    }
    atomic_write_pickle(checkpoint_path, payload)
    return checkpoint_path


def update_summary(results_dir: Path, run_config: dict, *, status: str, latest_metrics: dict | None, checkpoint_steps_saved: list[int], error: str | None = None):
    summary = {
        "status": status,
        "system": run_config["system"],
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
        "checkpoint_steps_saved": checkpoint_steps_saved,
        "updated_at_epoch_s": time.time(),
    }
    if latest_metrics is not None:
        summary["latest_metrics"] = latest_metrics
    if error is not None:
        summary["error"] = error
    atomic_write_json(results_dir / "summary.json", summary)


@dataclass
class ResumeState:
    step: int = 0
    phase_index: int = 0
    parameters: object | None = None
    variables: object | None = None
    source_step: int | None = None
    source_path: str | None = None


class TrainMetricsCallback(AbstractCallback, mutable=True):
    _results_dir: str = struct.field(pytree_node=False)
    _run_config: dict = struct.field(pytree_node=False)
    _phase_spec: dict = struct.field(pytree_node=False)
    _step_base: int = struct.field(pytree_node=False)
    _global_schedule: object = struct.field(pytree_node=False)
    _checkpoint_steps: tuple = struct.field(pytree_node=False)
    _existing_metric_steps: set = struct.field(pytree_node=False)
    _saved_checkpoint_steps: list = struct.field(pytree_node=False)
    _latest_checkpoint_every: int = struct.field(pytree_node=False)
    _step_t0: float = struct.field(pytree_node=False)
    _run_t0: float = struct.field(pytree_node=False)
    _last_metrics: dict = struct.field(pytree_node=False)

    def __init__(
        self,
        *,
        results_dir: Path,
        run_config: dict,
        phase_spec: dict,
        step_base: int,
        global_schedule,
        checkpoint_steps: tuple[int, ...],
        existing_metric_steps: set[int],
        saved_checkpoint_steps: list[int],
        latest_checkpoint_every: int,
        run_start_time: float,
    ):
        self._results_dir = str(results_dir)
        self._run_config = run_config
        self._phase_spec = phase_spec
        self._step_base = int(step_base)
        self._global_schedule = global_schedule
        self._checkpoint_steps = checkpoint_steps
        self._existing_metric_steps = existing_metric_steps
        self._saved_checkpoint_steps = saved_checkpoint_steps
        self._latest_checkpoint_every = latest_checkpoint_every
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
        if np.isnan(energy) or np.isnan(variance):
            driver._stop_run = True
            self._last_metrics = {
                "step": self._step_base + step + 1,
                "phase_index": self._phase_spec["phase_index"],
                "phase_name": self._phase_spec["name"],
                "status": "nan",
            }
            return

        completed_step = self._step_base + step + 1
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
            "phase_index": self._phase_spec["phase_index"],
            "phase_name": self._phase_spec["name"],
            "symmetry_index": self._phase_spec["symmetry_index"],
            "energy": energy,
            "energy_per_site": energy / self._run_config["n_sites"],
            "energy_sigma": sigma,
            "energy_variance": variance,
            "r_train": train_metrics["residual_raw"],
            "r_train_norm": train_metrics["residual_norm"],
            "r_train_target_norm_sq": train_metrics["target_norm_sq"],
            "energy_variance_train": train_metrics["energy_variance"],
        }

    def on_step_end(self, step, log_data, driver):
        completed_step = self._step_base + step + 1
        if not self._last_metrics:
            return

        lr = float(self._global_schedule(completed_step))
        acceptance = log_data.get("acceptance")
        if acceptance is not None:
            acceptance = float(np.asarray(acceptance))

        row = dict(self._last_metrics)
        row.update(
            {
                "learning_rate": lr,
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
                f"  step={completed_step:4d} phase={self._phase_spec['phase_index']} "
                f"E/N={row['energy_per_site']:.6f} Var={row['energy_variance']:.4f} "
                f"r_train={row['r_train']:.4f} lr={row['learning_rate']:.5f}",
                flush=True,
            )

        if completed_step in self._checkpoint_steps and completed_step not in self._saved_checkpoint_steps:
            checkpoint_path = save_checkpoint(
                Path(self._results_dir),
                completed_step,
                self._phase_spec,
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
                self._phase_spec,
                driver,
                self._run_config,
                row,
            )
            print(f"  latest checkpoint updated: {checkpoint_path.name}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Canonical large-scale ViT trainer for J1J2/SSM")
    parser.add_argument("--system", required=True, choices=["j1j2", "ssm", "shastry"])
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--diag-shift", type=float, help="Single lambda value")
    group.add_argument("--sweep", action="store_true", help="Run the full lambda grid")
    parser.add_argument("--output-root", type=str, default=str(ROOT / "large_scale"))
    parser.add_argument("--resume", type=str, default=None, help="Resume from a checkpoint path")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Initialize parameters from a checkpoint and start a new branch at local step 0.",
    )
    parser.add_argument("--n-samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument(
        "--phase-samples",
        type=str,
        default=",".join(str(x) for x in PHASE_SAMPLES),
        help="Comma-separated sample counts for phases 0,1,2. Defaults to canonical phase-specific counts.",
    )
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--lr-schedule", type=str, default="cosine", choices=["constant", "cosine"])
    parser.add_argument("--warmup-steps", type=int, default=WARMUP_STEPS)
    parser.add_argument(
        "--schedule-total-steps",
        type=int,
        default=None,
        help="Total step horizon for the global warmup+cosine schedule. Defaults to the run total.",
    )
    parser.add_argument("--phase-lengths", type=str, default=",".join(str(x) for x in (200, 3500, 3500)))
    parser.add_argument("--checkpoint-steps", type=str, default=",".join(str(x) for x in CHECKPOINT_STEPS))
    parser.add_argument(
        "--latest-checkpoint-every",
        type=int,
        default=None,
        help="Overwrite checkpoints/checkpoint_latest.pkl every N steps. Defaults to 25 for λ<=1e-7, disabled otherwise.",
    )
    parser.add_argument("--depth", type=int, default=VIT_DEPTH)
    parser.add_argument("--d-model", type=int, default=VIT_D_MODEL)
    parser.add_argument("--heads", type=int, default=VIT_HEADS)
    parser.add_argument("--expansion-factor", type=int, default=VIT_EXPANSION)
    parser.add_argument(
        "--lattice-size",
        type=int,
        default=LATTICE_SIZE,
        help="Linear lattice size for J1J2/SSM ViT systems. Defaults to the canonical large-scale size.",
    )
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--chunk-size-bwd",
        type=int,
        default=None,
        help="Backward NTK/Jacobian chunk size. Lower values reduce SR memory use at extra runtime cost.",
    )
    parser.add_argument("--linear-solver", type=str, default="cholesky", choices=["cholesky", "pinv_smooth", "pinv", "LU"])
    parser.add_argument("--mode", type=str, default=None, choices=["real", "complex"])
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


def build_run_config(
    args,
    *,
    system_name: str,
    diag_shift: float,
    phase_specs: tuple[dict, ...],
    checkpoint_steps: tuple[int, ...],
    n_sites: int,
    pii_tau: float | None = None,
    init_from_step: int | None = None,
):
    run_total_steps = int(sum(spec["length"] for spec in phase_specs))
    latest_checkpoint_every = args.latest_checkpoint_every
    if latest_checkpoint_every is None:
        latest_checkpoint_every = 25 if diag_shift <= 1e-7 else 0
    sr_lambda_grid = [float(x) for x in parse_sr_lambda_grid(args.sr_lambda_grid)]
    return {
        "system": system_name,
        "diag_shift": diag_shift,
        "sr_variant": args.sr_variant,
        "sr_weight_scheme": args.sr_weight_scheme,
        "sr_lambda_grid": sr_lambda_grid,
        "sr_weight_fit_samples": (
            None if args.sr_weight_fit_samples is None else int(args.sr_weight_fit_samples)
        ),
        "sr_lambda_quantiles": (
            None
            if args.sr_lambda_quantiles is None
            else [float(x) for x in parse_sr_lambda_grid(args.sr_lambda_quantiles)]
        ),
        "sr_lambda_ema_rho": (
            None if args.sr_lambda_ema_rho is None else float(args.sr_lambda_ema_rho)
        ),
        "sr_weight_ema_rho": (
            None if args.sr_weight_ema_rho is None else float(args.sr_weight_ema_rho)
        ),
        "ngd_method": args.ngd_method,
        "spring_momentum": float(args.spring_momentum),
        "spring_proj_reg": (
            None if args.spring_proj_reg is None else float(args.spring_proj_reg)
        ),
        "pii_tau": None if pii_tau is None else float(pii_tau),
        "pii_diag_shift": float(args.pii_diag_shift),
        "init_from": args.init_from,
        "init_from_step": init_from_step,
        "lattice_size": int(args.lattice_size),
        "n_sites": n_sites,
        "n_samples": phase_specs[0]["n_samples"],
        "phase_samples": [spec["n_samples"] for spec in phase_specs],
        "learning_rate": args.learning_rate,
        "lr_schedule": args.lr_schedule,
        "warmup_steps": args.warmup_steps,
        "total_steps": run_total_steps,
        "schedule_total_steps": (
            int(args.schedule_total_steps)
            if args.schedule_total_steps is not None
            else run_total_steps
        ),
        "phase_lengths": [spec["length"] for spec in phase_specs],
        "phase_specs": [dict(spec) for spec in phase_specs],
        "checkpoint_steps": list(checkpoint_steps),
        "latest_checkpoint_every": int(latest_checkpoint_every),
        "vit_depth": args.depth,
        "vit_d_model": args.d_model,
        "vit_heads": args.heads,
        "vit_expansion": args.expansion_factor,
        "chunk_size": args.chunk_size,
        "chunk_size_bwd": args.chunk_size_bwd,
        "linear_solver": args.linear_solver,
        "mode": args.mode,
        "output_head": "Vanilla",
        "seed": args.seed,
    }


def load_resume_state(path: str, *, reset_step: bool = False) -> ResumeState:
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    checkpoint_config_payload = checkpoint.get("config", {})
    source_step = int(checkpoint.get("step", checkpoint_config_payload.get("step", 0)))
    return ResumeState(
        step=0 if reset_step else source_step,
        phase_index=int(checkpoint.get("phase_index", checkpoint_config_payload.get("phase_index", 0))),
        parameters=checkpoint.get("parameters"),
        variables=checkpoint.get("variables"),
        source_step=source_step,
        source_path=str(path),
    )


def resolve_solver(name: str):
    return {
        "cholesky": nk.optimizer.solver.cholesky,
        "pinv_smooth": nk.optimizer.solver.pinv_smooth,
        "pinv": nk.optimizer.solver.pinv,
        "LU": nk.optimizer.solver.LU,
    }[name]


def run_single(args, system_name: str, diag_shift: float):
    validate_sr_configuration(args.sr_variant, args.sr_weight_scheme)
    if args.resume is not None and args.init_from is not None:
        raise ValueError("--resume and --init-from cannot be combined")
    if args.ngd_method != "sr" and args.sr_variant != "single":
        raise ValueError(f"--ngd-method {args.ngd_method} requires --sr-variant single")
    system = make_system(system_name, L=int(args.lattice_size))
    phase_lengths = parse_int_list(args.phase_lengths)
    phase_samples = parse_int_list(args.phase_samples)
    base_phase_specs = phase_specs_for_system(system_name)
    if len(phase_lengths) not in (1, len(base_phase_specs)):
        raise ValueError(
            f"Expected either one identity-only phase length or {len(base_phase_specs)} phase lengths"
        )
    if len(phase_samples) not in (1, len(phase_lengths)):
        raise ValueError("Expected either one sample count or one per phase length")
    if len(phase_lengths) == 1:
        base_phase_specs = base_phase_specs[:1]
    if len(phase_samples) == 1 and len(phase_lengths) > 1:
        phase_samples = tuple(int(phase_samples[0]) for _ in phase_lengths)
    checkpoint_steps = parse_int_list(args.checkpoint_steps)

    phase_specs = []
    step_start = 0
    for base_spec, length, n_samples in zip(base_phase_specs, phase_lengths, phase_samples):
        spec = dict(base_spec)
        spec["length"] = length
        spec["n_samples"] = n_samples
        spec["step_start"] = step_start
        spec["step_end"] = step_start + length
        phase_specs.append(spec)
        step_start = spec["step_end"]
    phase_specs = tuple(phase_specs)

    if checkpoint_steps[-1] != phase_specs[-1]["step_end"]:
        print(
            f"warning: last checkpoint step {checkpoint_steps[-1]} does not match total steps {phase_specs[-1]['step_end']}",
            flush=True,
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
        elif args.init_from is not None:
            resume_state = load_resume_state(args.init_from, reset_step=True)

        base_network = ViTNd(
            depth=args.depth,
            d_model=args.d_model,
            heads=args.heads,
            expansion_factor=args.expansion_factor,
            output_head="Vanilla",
            system=system,
        )
        base_model = base_network.network
        pii_tau = None
        if args.ngd_method == "pii":
            pii_tau = (
                float(args.pii_tau)
                if args.pii_tau is not None
                else default_pii_tau(system_name, system.graph.n_nodes)
            )
        run_config = build_run_config(
            args,
            system_name=system_name,
            diag_shift=diag_shift,
            phase_specs=phase_specs,
            checkpoint_steps=checkpoint_steps,
            n_sites=system.graph.n_nodes,
            pii_tau=pii_tau,
            init_from_step=resume_state.source_step if args.init_from is not None else None,
        )
        global_schedule = make_lr_schedule(
            args.learning_rate,
            run_config["schedule_total_steps"],
            args.warmup_steps,
            schedule=args.lr_schedule,
        )
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

        print("=" * 72)
        print(f"System: {system_name} | lambda={diag_shift:.0e}")
        print(f"Output: {results_dir}")
        print(f"Lattice: L={run_config['lattice_size']} | N={run_config['n_sites']}")
        print(
            f"Phase samples: {[spec['n_samples'] for spec in phase_specs]} "
            f"| lr={args.learning_rate} | schedule={args.lr_schedule} | warmup={args.warmup_steps}"
        )
        print(f"ViT: depth={args.depth} d_model={args.d_model} heads={args.heads} expansion={args.expansion_factor}")
        print(f"Chunking: forward={args.chunk_size} backward={args.chunk_size_bwd}")
        print(f"Phase lengths: {[spec['length'] for spec in phase_specs]}")
        print(f"Checkpoint steps: {list(checkpoint_steps)}")
        print(
            "SR config: "
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
        if run_config["latest_checkpoint_every"] > 0:
            print(
                "Rolling latest checkpoint: "
                f"every {run_config['latest_checkpoint_every']} steps -> checkpoints/checkpoint_latest.pkl"
            )
        if args.resume:
            print(f"Resuming from {args.resume} at step {resume_state.step}")
        if args.init_from:
            print(f"Initializing from {args.init_from} source_step={resume_state.source_step}")
        print("=" * 72)

        prev_params = resume_state.parameters
        prev_variables = resume_state.variables
        latest_metrics = None
        run_start_time = time.time()
        for phase_spec in phase_specs:
            if phase_spec["step_end"] <= resume_state.step:
                continue

            phase_resume_step = max(phase_spec["step_start"], resume_state.step)
            remaining_steps = phase_spec["step_end"] - phase_resume_step
            if remaining_steps <= 0:
                continue

            phase_n_samples = int(phase_spec["n_samples"])
            sampler = nk.sampler.MetropolisExchange(
                system.hilbert_space,
                graph=system.graph,
                d_max=2,
                n_chains=phase_n_samples,
            )
            model = system.symmetrizing_functions[phase_spec["symmetry_index"]](base_model)
            vstate = nk.vqs.MCState(
                sampler,
                model=model,
                n_samples=phase_n_samples,
                chunk_size=args.chunk_size,
            )
            if prev_params is not None:
                bare_params = get_bare_params(prev_params)
                vstate.parameters = adapt_params_to_model(bare_params, vstate.parameters)
            elif prev_variables is not None:
                try:
                    vstate.variables = prev_variables
                except Exception:
                    pass

            n_params = count_parameters(vstate.parameters)
            print(
                f"\nPhase {phase_spec['phase_index']}: {phase_spec['name']} "
                f"({phase_resume_step} -> {phase_spec['step_end']}, remaining={remaining_steps})"
            )
            print(
                f"  symmetry_index={phase_spec['symmetry_index']} | "
                f"N_s={phase_n_samples} | P={n_params:,} | P/N_s={n_params / phase_n_samples:.2f}"
            )

            optimizer = nk.optimizer.Sgd(
                learning_rate=offset_schedule(global_schedule, phase_resume_step)
            )
            driver = advd.driver.VMC_NG(
                hamiltonian=system.hamiltonian,
                optimizer=optimizer,
                diag_shift=diag_shift,
                proj_reg=(
                    run_config["spring_proj_reg"]
                    if run_config["ngd_method"] == "spring"
                    else None
                ),
                momentum=(
                    run_config["spring_momentum"]
                    if run_config["ngd_method"] == "spring"
                    else None
                ),
                variational_state=vstate,
                chunk_size_bwd=args.chunk_size_bwd,
                use_ntk=True,
                linear_solver_fn=resolve_solver(args.linear_solver),
                mode=args.mode,
            )
            # Exact r_train is computed only at checkpoint steps. Keeping
            # collect_residual_info disabled avoids pushing the full SR residual
            # info/NTK objects through RuntimeLog every iteration.
            driver.collect_residual_info = False

            callback = TrainMetricsCallback(
                results_dir=results_dir,
                run_config=run_config,
                phase_spec=phase_spec,
                step_base=phase_resume_step,
                global_schedule=global_schedule,
                checkpoint_steps=checkpoint_steps,
                existing_metric_steps=existing_metric_steps,
                saved_checkpoint_steps=saved_checkpoint_steps,
                latest_checkpoint_every=run_config["latest_checkpoint_every"],
                run_start_time=run_start_time,
            )
            callbacks = [callback]
            if run_config["ngd_method"] == "pii":
                callbacks.insert(
                    0,
                    PIIUpdateCallback(
                        tau=run_config["pii_tau"],
                        diag_shift=run_config["pii_diag_shift"],
                    ),
                )
            elif run_config["ngd_method"] == "rgn":
                callbacks.insert(
                    0,
                    RGNUpdateCallback(
                        diag_shift=run_config["pii_diag_shift"],
                    ),
                )
            elif run_config["sr_variant"] != "single":
                callbacks.insert(
                    0,
                    MultiLambdaSRCallback(
                        sr_variant=run_config["sr_variant"],
                        sr_weight_scheme=run_config["sr_weight_scheme"],
                        sr_lambda_grid=run_config["sr_lambda_grid"],
                        sr_weight_fit_samples=run_config["sr_weight_fit_samples"],
                        sr_lambda_quantiles=run_config["sr_lambda_quantiles"],
                        sr_lambda_ema_rho=run_config["sr_lambda_ema_rho"],
                        sr_weight_ema_rho=run_config["sr_weight_ema_rho"],
                    ),
                )

            log = nk.logging.RuntimeLog()
            try:
                driver.run(n_iter=remaining_steps, out=log, callback=callbacks)
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
                status="running",
                latest_metrics=latest_metrics,
                checkpoint_steps_saved=saved_checkpoint_steps,
            )
            prev_params = get_bare_params(driver.state.parameters)
            prev_variables = driver.state.variables
            resume_state = ResumeState(
                step=phase_spec["step_end"],
                phase_index=phase_spec["phase_index"],
                parameters=driver.state.parameters,
                variables=driver.state.variables,
            )

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
    validate_sr_configuration(args.sr_variant, args.sr_weight_scheme)
    if args.sr_variant == "single":
        if args.diag_shift is None and not args.sweep:
            raise ValueError("single SR requires --diag-shift or --sweep")
        lambda_values = LAMBDA_GRID if args.sweep else [args.diag_shift]
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

    failures = []
    for diag_shift in lambda_values:
        try:
            run_single(args, system_name, float(diag_shift))
        except Exception as exc:
            failures.append((diag_shift, str(exc)))
            print(f"\nFAILED system={system_name} lambda={diag_shift:.0e}: {exc}", flush=True)
            if args.fail_fast:
                raise

    if failures:
        raise SystemExit(
            "\n".join(
                [f"{len(failures)} runs failed:"] + [f"  lambda={lam:.0e}: {msg}" for lam, msg in failures]
            )
        )


if __name__ == "__main__":
    main()
