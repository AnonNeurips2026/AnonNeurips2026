from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACCUMULATOR_VERSION = 1
EPS = 1e-30


class IncrementalDiagnosticsError(RuntimeError):
    pass


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".pkl.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@dataclass
class GlobalCenteredAccumulator:
    n_channels: int

    def __post_init__(self):
        self.count = 0
        self.sums = np.zeros(self.n_channels, dtype=np.float64)
        self.sumsq = np.zeros(self.n_channels, dtype=np.float64)

    def add(self, *channels):
        if len(channels) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(channels)}")
        n = None
        for idx, values in enumerate(channels):
            array = np.asarray(values, dtype=np.float64).reshape(-1)
            if n is None:
                n = array.size
            elif array.size != n:
                raise ValueError("All channels must have the same length")
            self.sums[idx] += float(array.sum())
            self.sumsq[idx] += float(np.square(array).sum())
        self.count += int(n or 0)

    def mean_square(self) -> float:
        if self.count <= 0:
            return float("nan")
        centered = self.sumsq - np.square(self.sums) / float(self.count)
        return float(centered.sum() / float(self.count))

    def total_count(self) -> int:
        return int(self.count)

    def to_state(self) -> dict:
        return {
            "kind": "global",
            "n_channels": int(self.n_channels),
            "count": int(self.count),
            "sums": np.asarray(self.sums, dtype=np.float64),
            "sumsq": np.asarray(self.sumsq, dtype=np.float64),
        }

    @classmethod
    def from_state(cls, state: dict) -> "GlobalCenteredAccumulator":
        acc = cls(n_channels=int(state["n_channels"]))
        acc.count = int(state["count"])
        acc.sums = np.asarray(state["sums"], dtype=np.float64).copy()
        acc.sumsq = np.asarray(state["sumsq"], dtype=np.float64).copy()
        return acc


@dataclass
class ReplicaCenteredAccumulator:
    n_replicas: int
    n_channels: int

    def __post_init__(self):
        self.counts = np.zeros(self.n_replicas, dtype=np.int64)
        self.sums = np.zeros((self.n_channels, self.n_replicas), dtype=np.float64)
        self.sumsq = np.zeros(self.n_channels, dtype=np.float64)

    def add(self, *channels):
        if len(channels) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {len(channels)}")
        m = None
        for idx, values in enumerate(channels):
            array = np.asarray(values, dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != self.n_replicas:
                raise ValueError(
                    f"Expected channel shape ({self.n_replicas}, m), got {array.shape}"
                )
            if m is None:
                m = array.shape[1]
            elif array.shape[1] != m:
                raise ValueError("All channels must have the same per-replica width")
            self.sums[idx] += array.sum(axis=1)
            self.sumsq[idx] += float(np.square(array).sum())
        self.counts += int(m or 0)

    def mean_square(self) -> float:
        total = int(self.counts.sum())
        if total <= 0:
            return float("nan")
        centered = self.sumsq.copy()
        valid = self.counts > 0
        for idx in range(self.n_channels):
            centered[idx] -= float(
                np.sum(
                    np.where(
                        valid,
                        np.square(self.sums[idx]) / self.counts.clip(min=1),
                        0.0,
                    )
                )
            )
        return float(centered.sum() / float(total))

    def total_count(self) -> int:
        return int(self.counts.sum())

    def to_state(self) -> dict:
        return {
            "kind": "replica",
            "n_replicas": int(self.n_replicas),
            "n_channels": int(self.n_channels),
            "counts": np.asarray(self.counts, dtype=np.int64),
            "sums": np.asarray(self.sums, dtype=np.float64),
            "sumsq": np.asarray(self.sumsq, dtype=np.float64),
        }

    @classmethod
    def from_state(cls, state: dict) -> "ReplicaCenteredAccumulator":
        acc = cls(
            n_replicas=int(state["n_replicas"]),
            n_channels=int(state["n_channels"]),
        )
        acc.counts = np.asarray(state["counts"], dtype=np.int64).copy()
        acc.sums = np.asarray(state["sums"], dtype=np.float64).copy()
        acc.sumsq = np.asarray(state["sumsq"], dtype=np.float64).copy()
        return acc


def accumulator_from_state(state: dict):
    kind = state["kind"]
    if kind == "global":
        return GlobalCenteredAccumulator.from_state(state)
    if kind == "replica":
        return ReplicaCenteredAccumulator.from_state(state)
    raise ValueError(f"Unknown accumulator kind: {kind}")


def _result_meta(result: dict) -> dict:
    return {
        "version": ACCUMULATOR_VERSION,
        "result_n_val": int(result.get("n_val", 0)),
    }


def attach_rval_state(result: dict, acc_pair) -> dict:
    payload = dict(result)
    residual_acc, energy_acc = acc_pair
    payload["_diagnostic_state"] = {
        "metric": "rval",
        "meta": _result_meta(result),
        "residual": residual_acc.to_state(),
        "energy": energy_acc.to_state(),
    }
    return payload


def attach_twobatch_state(result: dict, acc) -> dict:
    payload = dict(result)
    payload["_diagnostic_state"] = {
        "metric": "twobatch",
        "meta": _result_meta(result),
        "prediction": acc.to_state(),
    }
    return payload


def _load_incremental_state(result: dict, *, metric: str):
    state = result.get("_diagnostic_state")
    if not isinstance(state, dict):
        raise IncrementalDiagnosticsError(
            f"Existing {metric} result does not contain accumulator state."
        )
    if state.get("metric") != metric:
        raise IncrementalDiagnosticsError(
            f"Accumulator state metric mismatch: expected {metric}, got {state.get('metric')}"
        )
    meta = state.get("meta") or {}
    if int(meta.get("version", 0)) != ACCUMULATOR_VERSION:
        raise IncrementalDiagnosticsError(
            f"Unsupported accumulator state version for {metric}: {meta.get('version')}"
        )
    result_n_val = int(result.get("n_val", 0))
    meta_n_val = int(meta.get("result_n_val", result_n_val))
    if result_n_val != meta_n_val:
        raise IncrementalDiagnosticsError(
            f"Existing {metric} result reports n_val={result_n_val}, but stored accumulator metadata "
            f"reports n_val={meta_n_val}."
        )
    return state


def load_existing_rval_accumulators(result: dict):
    state = _load_incremental_state(result, metric="rval")
    residual_acc = accumulator_from_state(state["residual"])
    energy_acc = accumulator_from_state(state["energy"])
    if residual_acc.total_count() != energy_acc.total_count():
        raise IncrementalDiagnosticsError(
            "Stored rval accumulator counts disagree between residual and energy statistics."
        )
    if residual_acc.total_count() != int(result.get("n_val", 0)):
        raise IncrementalDiagnosticsError(
            "Stored rval accumulator count does not match the result n_val."
        )
    return residual_acc, energy_acc


def load_existing_twobatch_accumulator(result: dict):
    state = _load_incremental_state(result, metric="twobatch")
    acc = accumulator_from_state(state["prediction"])
    if acc.total_count() != int(result.get("n_val", 0)):
        raise IncrementalDiagnosticsError(
            "Stored twobatch accumulator count does not match the result n_val."
        )
    return acc


def finalize_rval_metrics(acc_pair):
    residual_acc, energy_acc = acc_pair
    residual_raw = float(residual_acc.mean_square())
    energy_variance = float(energy_acc.mean_square())
    target_norm_sq = 4.0 * energy_variance
    residual_norm = residual_raw / (target_norm_sq + EPS)
    return {
        "residual_raw": residual_raw,
        "residual_norm": residual_norm,
        "target_norm_sq": target_norm_sq,
        "energy_variance": energy_variance,
    }


def finalize_twobatch_metrics(acc, *, train_target_norm_sq: float):
    snorm_diff_sq = float(acc.mean_square())
    var_hat_raw = snorm_diff_sq / 2.0
    var_hat_norm = var_hat_raw / (float(train_target_norm_sq) + EPS)
    return {
        "snorm_diff_sq": snorm_diff_sq,
        "var_hat_raw": var_hat_raw,
        "var_hat_norm": var_hat_norm,
    }
