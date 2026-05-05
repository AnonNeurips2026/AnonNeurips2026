from __future__ import annotations

from pathlib import Path

from sr_filtering_nqs.large_scale.core.common import (
    SHARED_FIG2_DIAGNOSTIC_LAMBDAS,
    default_source_steps_for_system,
    format_lambda_dir,
    lambda_grid_tag,
    normalize_system_name,
    results_dir_for_run,
)


def checkpoint_step_from_path(ckpt_path) -> int:
    ckpt_path = Path(ckpt_path)
    stem = ckpt_path.stem
    if not stem.startswith("checkpoint_step"):
        raise ValueError(f"Unsupported checkpoint filename: {ckpt_path.name}")
    return int(stem.replace("checkpoint_step", ""))


def checkpoint_diagnostics_dir(ckpt_path) -> Path:
    ckpt_path = Path(ckpt_path)
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent / "diagnostics"
    return ckpt_path.parent


def parse_source_steps(raw, system_name: str) -> tuple[int, ...]:
    system_name = normalize_system_name(system_name)
    if raw is None:
        return tuple(int(step) for step in default_source_steps_for_system(system_name))
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ("", "all"):
            return tuple(int(step) for step in default_source_steps_for_system(system_name))
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    steps = sorted({int(value) for value in values})
    if not steps:
        raise ValueError("source_steps must contain at least one checkpoint step")
    return tuple(steps)


def parse_lambda_values(raw, *, default) -> tuple[float, ...]:
    if raw is None:
        return tuple(float(value) for value in default)
    if isinstance(raw, str):
        values = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        values = list(raw)
    if not values:
        raise ValueError("Expected at least one positive lambda value")
    lambdas = tuple(float(value) for value in values)
    if any(value <= 0.0 for value in lambdas):
        raise ValueError(f"All lambda values must be positive, got {lambdas}")
    return lambdas


def discover_source_checkpoints(
    input_dir,
    system_name: str,
    *,
    source_training_lambda: float,
    source_steps=None,
    latest_only: bool = False,
) -> list[tuple[int, Path]]:
    system_name = normalize_system_name(system_name)
    ckpt_dir = results_dir_for_run(input_dir, system_name, source_training_lambda) / "checkpoints"
    if not ckpt_dir.exists():
        return []

    rows = sorted(
        (
            checkpoint_step_from_path(path),
            path,
        )
        for path in ckpt_dir.glob("checkpoint_step*.pkl")
    )
    if not rows:
        return []

    by_step = {step: path for step, path in rows}
    requested_steps = parse_source_steps(source_steps, system_name)
    missing_steps = [step for step in requested_steps if step not in by_step]
    if missing_steps and not (
        source_steps is None
        or (isinstance(source_steps, str) and source_steps.strip().lower() in ("", "all"))
    ):
        raise FileNotFoundError(
            f"Missing requested source checkpoints for {system_name}: {missing_steps}"
        )
    selected = [(step, by_step[step]) for step in requested_steps if step in by_step]
    if latest_only:
        return [] if not selected else [selected[-1]]
    return selected


def _protocol_root(
    ckpt_path,
    *,
    protocol_id: str,
    source_training_lambda: float,
) -> Path:
    source_token = format_lambda_dir(float(source_training_lambda))
    return checkpoint_diagnostics_dir(ckpt_path) / protocol_id / f"src_{source_token}"


def shared_protocol_step_dir(
    ckpt_path,
    *,
    protocol_id: str,
    source_training_lambda: float,
    source_step: int | None = None,
) -> Path:
    step = checkpoint_step_from_path(ckpt_path) if source_step is None else int(source_step)
    return _protocol_root(
        ckpt_path,
        protocol_id=protocol_id,
        source_training_lambda=float(source_training_lambda),
    ) / f"step{step:06d}"


def repeat_count_tag(repeat_count: int) -> str:
    return f"m{int(repeat_count):03d}"


def shared_fig2_delta_bank_path(
    ckpt_path,
    *,
    protocol_id: str,
    source_training_lambda: float,
    source_step: int,
    diagnostic_lambda: float,
    repeat_count: int,
) -> Path:
    lam_token = format_lambda_dir(float(diagnostic_lambda))
    return shared_protocol_step_dir(
        ckpt_path,
        protocol_id=protocol_id,
        source_training_lambda=float(source_training_lambda),
        source_step=int(source_step),
    ) / f"delta_bank__diag_{lam_token}__{repeat_count_tag(repeat_count)}.pkl"


def shared_fig2_metric_path(
    ckpt_path,
    *,
    protocol_id: str,
    source_training_lambda: float,
    source_step: int,
    diagnostic_lambda: float,
    repeat_count: int,
    metric: str,
) -> Path:
    if metric not in ("rval", "twobatch"):
        raise ValueError(f"Unsupported shared Figure 2 metric={metric!r}")
    lam_token = format_lambda_dir(float(diagnostic_lambda))
    return shared_protocol_step_dir(
        ckpt_path,
        protocol_id=protocol_id,
        source_training_lambda=float(source_training_lambda),
        source_step=int(source_step),
    ) / f"{metric}__diag_{lam_token}__{repeat_count_tag(repeat_count)}.pkl"


def shared_fig3_bank_path(
    ckpt_path,
    *,
    protocol_id: str,
    source_training_lambda: float,
    source_step: int,
    lambda_grid,
    repeat_count: int,
    oracle_lambda: float,
) -> Path:
    grid = tuple(float(value) for value in lambda_grid)
    grid_token = lambda_grid_tag(grid)
    oracle_token = format_lambda_dir(float(oracle_lambda))
    return shared_protocol_step_dir(
        ckpt_path,
        protocol_id=protocol_id,
        source_training_lambda=float(source_training_lambda),
        source_step=int(source_step),
    ) / (
        f"mixture_bank__k{len(grid)}__grid_{grid_token}"
        f"__oracle_{oracle_token}__{repeat_count_tag(repeat_count)}.pkl"
    )


def shared_fig3_eval_path(
    ckpt_path,
    *,
    protocol_id: str,
    source_training_lambda: float,
    source_step: int,
    lambda_grid,
    repeat_count: int,
    oracle_lambda: float,
) -> Path:
    grid = tuple(float(value) for value in lambda_grid)
    grid_token = lambda_grid_tag(grid)
    oracle_token = format_lambda_dir(float(oracle_lambda))
    return shared_protocol_step_dir(
        ckpt_path,
        protocol_id=protocol_id,
        source_training_lambda=float(source_training_lambda),
        source_step=int(source_step),
    ) / (
        f"mixture_eval__k{len(grid)}__grid_{grid_token}"
        f"__oracle_{oracle_token}__{repeat_count_tag(repeat_count)}.pkl"
    )


def shared_fig2_metric_files(
    input_dir,
    system_name: str,
    *,
    protocol_id: str,
    source_training_lambda: float,
    metric: str,
    repeat_count: int,
) -> list[Path]:
    system_name = normalize_system_name(system_name)
    if metric not in ("rval", "twobatch"):
        raise ValueError(f"Unsupported shared Figure 2 metric={metric!r}")
    source_root = results_dir_for_run(input_dir, system_name, source_training_lambda)
    protocol_root = (
        source_root
        / "diagnostics"
        / protocol_id
        / f"src_{format_lambda_dir(float(source_training_lambda))}"
    )
    if not protocol_root.exists():
        return []
    pattern = f"step*/{metric}__diag_*__{repeat_count_tag(repeat_count)}.pkl"
    return sorted(protocol_root.glob(pattern))


def shared_fig3_eval_files(
    input_dir,
    system_name: str,
    *,
    protocol_id: str,
    source_training_lambda: float,
    repeat_count: int,
) -> list[Path]:
    system_name = normalize_system_name(system_name)
    source_root = results_dir_for_run(input_dir, system_name, source_training_lambda)
    protocol_root = (
        source_root
        / "diagnostics"
        / protocol_id
        / f"src_{format_lambda_dir(float(source_training_lambda))}"
    )
    if not protocol_root.exists():
        return []
    pattern = f"step*/mixture_eval__*__{repeat_count_tag(repeat_count)}.pkl"
    return sorted(protocol_root.glob(pattern))


DEFAULT_SHARED_FIG2_DIAGNOSTIC_LAMBDAS = SHARED_FIG2_DIAGNOSTIC_LAMBDAS
