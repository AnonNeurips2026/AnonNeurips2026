from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ("j1j2", "tfim")

sys.path.insert(0, str(ROOT))

from sr_filtering_nqs.large_scale.core.common import (  # noqa: E402
    CHECKPOINT_STEPS,
    canonical_results_root_candidates,
    format_lambda_dir,
    J1J2_DIAGNOSTIC_STEPS,
    lambda_grid_for_system,
)


J1J2_ACTIVE_CHECKPOINT_COUNT = len([step for step in CHECKPOINT_STEPS if step >= 200])
J1J2_ACTIVE_DIAGNOSTIC_COUNT = len(J1J2_DIAGNOSTIC_STEPS)
TFIM_ACTIVE_CHECKPOINT_COUNT = 20
TFIM_ACTIVE_DIAGNOSTIC_COUNT = 10
J1J2_ACTIVE_STEPS = tuple(step for step in CHECKPOINT_STEPS if step >= 200)
TFIM_ACTIVE_CHECKPOINT_STEPS = tuple(range(100, 2001, 100))
TFIM_ACTIVE_DIAGNOSTIC_STEPS = tuple(range(1100, 2001, 100))
DELTA_BANK_PATTERN = "delta_bank_step*.pkl"
RVAL_PATTERN = "rval_step*.pkl"
TWOBATCH_PATTERN = "twobatch_step*.pkl"


def _results_dir(system: str, lambda_token: str) -> Path:
    fallback = canonical_results_root_candidates(ROOT, system)[0] / lambda_token / "results"
    for base in canonical_results_root_candidates(ROOT, system):
        candidate = base / lambda_token / "results"
        if candidate.exists():
            return candidate
    return fallback


def _load_summary(results_dir: Path) -> dict | None:
    summary_path = results_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text())
    except json.JSONDecodeError:
        return {"status": "error", "error": f"invalid json: {summary_path}"}


def _extract_steps(directory: Path, pattern: str) -> list[int]:
    if not directory.exists():
        return []
    steps = []
    for path in directory.glob(pattern):
        match = re.search(r"step(\d+)", path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def _expected_steps(
    system: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if system == "j1j2":
        return (
            J1J2_ACTIVE_STEPS,
            J1J2_DIAGNOSTIC_STEPS,
            J1J2_DIAGNOSTIC_STEPS,
            J1J2_DIAGNOSTIC_STEPS,
        )
    return (
        TFIM_ACTIVE_CHECKPOINT_STEPS,
        TFIM_ACTIVE_DIAGNOSTIC_STEPS,
        TFIM_ACTIVE_DIAGNOSTIC_STEPS,
        TFIM_ACTIVE_DIAGNOSTIC_STEPS,
    )


def _step_summary(steps: list[int]) -> str:
    if not steps:
        return "-"
    if len(steps) == 1:
        return str(steps[0])
    diffs = [b - a for a, b in zip(steps[:-1], steps[1:])]
    if len(set(diffs)) == 1:
        return f"{steps[0]}-{steps[-1]} x{diffs[0]}"
    return ",".join(str(step) for step in steps)


def summarize_lambda(system: str, diag_shift: float) -> dict:
    lambda_token = format_lambda_dir(diag_shift)
    results_dir = _results_dir(system, lambda_token)
    summary = _load_summary(results_dir)

    (
        expected_checkpoint_steps,
        expected_delta_bank_steps,
        expected_rval_steps,
        expected_twobatch_steps,
    ) = _expected_steps(system)
    checkpoint_steps = _extract_steps(results_dir / "checkpoints", "checkpoint_step*.pkl")
    delta_bank_steps = _extract_steps(results_dir / "diagnostics", DELTA_BANK_PATTERN)
    rval_steps = _extract_steps(results_dir / "diagnostics", RVAL_PATTERN)
    twobatch_steps = _extract_steps(results_dir / "diagnostics", TWOBATCH_PATTERN)

    checkpoint_done = [step for step in expected_checkpoint_steps if step in checkpoint_steps]
    delta_bank_done = [step for step in expected_delta_bank_steps if step in delta_bank_steps]
    rval_done = [step for step in expected_rval_steps if step in rval_steps]
    twobatch_done = [step for step in expected_twobatch_steps if step in twobatch_steps]
    checkpoint_missing = [step for step in expected_checkpoint_steps if step not in checkpoint_steps]
    delta_bank_missing = [step for step in expected_delta_bank_steps if step not in delta_bank_steps]
    rval_missing = [step for step in expected_rval_steps if step not in rval_steps]
    twobatch_missing = [step for step in expected_twobatch_steps if step not in twobatch_steps]

    expected_checkpoints = None
    expected_rval = None
    expected_twobatch = None
    status = "pending"

    if summary is not None:
        status = summary.get("status", "unknown")
        expected_checkpoints = len(summary.get("checkpoint_steps", [])) or None
        expected_rval = len(summary.get("diagnostic_steps", [])) or None
        expected_twobatch = expected_rval

    if system == "j1j2":
        expected_checkpoints = J1J2_ACTIVE_CHECKPOINT_COUNT
        if expected_rval is None:
            expected_rval = J1J2_ACTIVE_DIAGNOSTIC_COUNT
        if expected_twobatch is None:
            expected_twobatch = J1J2_ACTIVE_DIAGNOSTIC_COUNT
    elif system == "tfim":
        if expected_checkpoints is None:
            expected_checkpoints = TFIM_ACTIVE_CHECKPOINT_COUNT
        if expected_rval is None:
            expected_rval = TFIM_ACTIVE_DIAGNOSTIC_COUNT
        if expected_twobatch is None:
            expected_twobatch = TFIM_ACTIVE_DIAGNOSTIC_COUNT

    if summary is None and (checkpoint_steps or delta_bank_steps or rval_steps or twobatch_steps):
        status = "unknown"

    return {
        "lambda_token": lambda_token,
        "status": status,
        "checkpoints": len(checkpoint_done),
        "expected_checkpoints": expected_checkpoints,
        "delta_bank": len(delta_bank_done),
        "expected_delta_bank": len(expected_delta_bank_steps),
        "rval": len(rval_done),
        "expected_rval": expected_rval,
        "twobatch": len(twobatch_done),
        "expected_twobatch": expected_twobatch,
        "checkpoint_missing": checkpoint_missing,
        "delta_bank_missing": delta_bank_missing,
        "rval_missing": rval_missing,
        "twobatch_missing": twobatch_missing,
        "results_dir": results_dir,
    }


def _fmt_count(value: int, expected: int | None) -> str:
    if expected is None:
        return str(value)
    return f"{value}/{expected}"


def summarize_system(system: str) -> list[dict]:
    return [summarize_lambda(system, float(diag_shift)) for diag_shift in lambda_grid_for_system(system)]


def _render_table(rows: list[dict]) -> str:
    headers = [
        "lambda",
        "status",
        "train",
        "train_todo",
        "delta_bank",
        "delta_bank_todo",
        "rval",
        "rval_todo",
        "twobatch",
        "twobatch_todo",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "lambda": row["lambda_token"],
                "status": row["status"],
                "train": _fmt_count(row["checkpoints"], row["expected_checkpoints"]),
                "train_todo": _step_summary(row["checkpoint_missing"]),
                "delta_bank": _fmt_count(row["delta_bank"], row["expected_delta_bank"]),
                "delta_bank_todo": _step_summary(row["delta_bank_missing"]),
                "rval": _fmt_count(row["rval"], row["expected_rval"]),
                "rval_todo": _step_summary(row["rval_missing"]),
                "twobatch": _fmt_count(row["twobatch"], row["expected_twobatch"]),
                "twobatch_todo": _step_summary(row["twobatch_missing"]),
            }
        )

    widths = {header: len(header) for header in headers}
    for row in table_rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    def render_line(parts: list[str]) -> str:
        return "| " + " | ".join(parts) + " |"

    lines = [
        render_line([header.ljust(widths[header]) for header in headers]),
        render_line(["-" * widths[header] for header in headers]),
    ]
    for row in table_rows:
        lines.append(render_line([row[header].ljust(widths[header]) for header in headers]))
    return "\n".join(lines)


def main() -> int:
    for system in SYSTEMS:
        rows = summarize_system(system)
        completed = sum(1 for row in rows if row["status"] == "completed")
        running = sum(1 for row in rows if row["status"] == "running")
        error = sum(1 for row in rows if row["status"] == "error")
        pending = sum(1 for row in rows if row["status"] == "pending")

        print(f"[{system}] completed={completed} running={running} error={error} pending={pending}")
        print(_render_table(rows))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
