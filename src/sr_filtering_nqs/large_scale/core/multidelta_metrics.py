from __future__ import annotations

import numpy as np


EPS = 1e-30


def reshape_for_centering(values, *, n_replicas: int | None):
    array = np.asarray(values, dtype=np.float64)
    total_count = int(array.shape[-1])
    if n_replicas is None:
        return array, total_count
    if total_count % n_replicas != 0:
        raise ValueError(
            f"Flat sample axis {total_count} is not divisible by n_replicas={n_replicas}"
        )
    return array.reshape(array.shape[:-1] + (n_replicas, total_count // n_replicas)), total_count


def center_samples(values, *, n_replicas: int | None):
    reshaped, total_count = reshape_for_centering(values, n_replicas=n_replicas)
    centered = reshaped - np.mean(reshaped, axis=-1, keepdims=True)
    return centered, total_count


def finalize_multidelta_rval(prediction_bank, energy_channels, *, n_replicas: int | None):
    residual_bank = np.asarray(prediction_bank, dtype=np.float64) - 2.0 * np.asarray(
        energy_channels, dtype=np.float64
    )[None, ...]
    residual_centered, total_count = center_samples(residual_bank, n_replicas=n_replicas)
    residual_axes = tuple(range(1, residual_centered.ndim))
    residual_per_delta = np.square(residual_centered).sum(axis=residual_axes) / float(total_count)
    residual_raw = float(np.mean(residual_per_delta))

    energy_centered, energy_total_count = center_samples(
        energy_channels, n_replicas=n_replicas
    )
    energy_variance = float(np.square(energy_centered).sum() / float(energy_total_count))
    target_norm_sq = 4.0 * energy_variance
    residual_norm = residual_raw / (target_norm_sq + EPS)

    return {
        "residual_raw": residual_raw,
        "residual_norm": residual_norm,
        "target_norm_sq": target_norm_sq,
        "energy_variance": energy_variance,
    }


def finalize_multidelta_twobatch(
    prediction_bank,
    *,
    n_replicas: int | None,
    train_target_norm_sq: float,
):
    centered_bank, total_count = center_samples(prediction_bank, n_replicas=n_replicas)
    m_deltas = int(centered_bank.shape[0])
    if m_deltas < 2:
        raise ValueError("twobatch requires at least two deltas")
    centered_mean = np.mean(centered_bank, axis=0, keepdims=True)
    centered_diff = centered_bank - centered_mean
    var_hat_raw = float(
        np.square(centered_diff).sum() / (float(m_deltas - 1) * float(total_count))
    )
    snorm_diff_sq = 2.0 * var_hat_raw
    var_hat_norm = var_hat_raw / (float(train_target_norm_sq) + EPS)
    return {
        "snorm_diff_sq": snorm_diff_sq,
        "var_hat_raw": var_hat_raw,
        "var_hat_norm": var_hat_norm,
    }
