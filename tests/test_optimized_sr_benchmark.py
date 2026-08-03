import jax.numpy as jnp
import numpy as np

from sr_filtering_nqs.large_scale.core.fixed_data_protocol import (
    exact_loo_stacking_weights,
    prediction_channels_from_bank_values,
    prediction_vectors_from_bank,
    target_vector_from_energy_channels,
)
from sr_filtering_nqs.large_scale.core.optimized_sr_benchmark import (
    _prediction_sufficient_statistics_on_device,
    _positive_spectrum_quantiles,
    eigh_pinv_solver,
    exact_loo_weights_from_sufficient_statistics,
    spectrum_quantile_eigh_pinv_solver,
)


def test_eigh_pinv_solver_matches_dense_solve():
    matrix = jnp.asarray([[3.0, 0.5], [0.5, 2.0]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)
    actual, info = eigh_pinv_solver(matrix, rhs)
    expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs))
    assert info is None
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_spectrum_quantile_solver_reuses_unshifted_factors():
    matrix = jnp.diag(jnp.asarray([0.0, 1.0, 4.0, 9.0], dtype=jnp.float64))
    rhs = jnp.asarray([0.0, 1.0, 2.0, 3.0], dtype=jnp.float64)
    quantiles = (0.9, 0.5, 0.1)
    actual, info = spectrum_quantile_eigh_pinv_solver(
        matrix,
        rhs,
        quantiles=quantiles,
    )
    expected_grid = np.quantile(np.asarray([1.0, 4.0, 9.0]), quantiles)
    expected = np.asarray(rhs) / (np.diag(np.asarray(matrix)) + expected_grid[0])
    np.testing.assert_allclose(info["lambda_grid"], expected_grid, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_positive_spectrum_quantiles_drop_nonpositive_values():
    values = jnp.asarray([-1.0, 0.0, 1.0, 3.0, 8.0], dtype=jnp.float64)
    actual = _positive_spectrum_quantiles(
        values,
        quantiles=(0.25, 0.75),
        floor_rel=1e-12,
    )
    expected = np.quantile(np.asarray([1.0, 3.0, 8.0]), [0.25, 0.75])
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_loo_sufficient_statistics_match_dense_objective_weights():
    rng = np.random.default_rng(12)
    count = 4
    dimension = 17
    targets = rng.normal(size=(count, dimension))
    cross_predictions = rng.normal(size=(count, count, dimension))
    for heldout in range(count):
        cross_predictions[heldout, heldout] = 0.0

    dense_weights = exact_loo_stacking_weights(targets, cross_predictions)
    grams = []
    projections = []
    norms = []
    for heldout in range(count):
        predictions = np.delete(cross_predictions[heldout], heldout, axis=0)
        grams.append(predictions @ predictions.T)
        projections.append(predictions @ targets[heldout])
        norms.append(np.dot(targets[heldout], targets[heldout]))
    sufficient_weights = exact_loo_weights_from_sufficient_statistics(
        np.asarray(grams),
        np.asarray(projections),
        np.asarray(norms),
    )
    np.testing.assert_allclose(sufficient_weights, dense_weights, rtol=1e-6, atol=1e-7)


def test_on_device_sufficient_statistics_match_dense_reference():
    rng = np.random.default_rng(23)
    candidate_count = 3
    sample_count = 12
    n_replicas = 3

    for mode in ("real", "complex"):
        value_width = sample_count if mode == "real" else 2 * sample_count
        values = rng.normal(size=(candidate_count, value_width))
        local_loss = rng.normal(size=sample_count)
        if mode == "complex":
            local_loss = local_loss + 1j * rng.normal(size=sample_count)

        actual_gram, actual_projection, actual_target_norm = (
            _prediction_sufficient_statistics_on_device(
                jnp.asarray(values),
                jnp.asarray(local_loss),
                mode=mode,
                sample_count=sample_count,
                n_replicas=n_replicas,
            )
        )

        prediction_bank = prediction_channels_from_bank_values(
            values,
            mode=mode,
            sample_count=sample_count,
        )
        if mode == "complex":
            energy_channels = np.stack(
                [np.real(local_loss), np.imag(local_loss)],
                axis=0,
            )
        else:
            energy_channels = np.real(local_loss)[None, :]
        expected_predictions = prediction_vectors_from_bank(
            prediction_bank,
            mode=mode,
            n_replicas=n_replicas,
        )
        expected_target, _ = target_vector_from_energy_channels(
            energy_channels,
            mode=mode,
            n_replicas=n_replicas,
        )

        np.testing.assert_allclose(
            actual_gram,
            expected_predictions @ expected_predictions.T,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            actual_projection,
            expected_predictions @ expected_target,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            actual_target_norm,
            expected_target @ expected_target,
            rtol=1e-10,
            atol=1e-10,
        )
