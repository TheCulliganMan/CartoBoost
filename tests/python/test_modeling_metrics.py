import cartoboost.metrics as metrics_module
import numpy as np
import pytest
from cartoboost import (
    brier_score,
    calibrated_intervals,
    conformal_residual_quantile,
    ece_calibration_error,
    interval_coverage,
    jitter_volatility,
    logloss,
    mean_average_precision,
    mean_interval_width,
    mean_reciprocal_rank,
    ndcg_at_k,
    pinball_loss,
    pr_auc,
    residual_morans_i,
    roc_auc,
    spatial_cv_gap,
)
from cartoboost.accelerators import available_backends


def test_conformal_residual_quantile_builds_calibrated_intervals():
    y_true = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    y_pred = np.array([9.0, 13.5, 13.0, 15.5, 21.0])

    quantile = conformal_residual_quantile(y_true, y_pred, alpha=0.5)
    lower, upper = calibrated_intervals([11.0, 20.0], quantile)

    assert quantile == pytest.approx(1.5)
    assert lower == pytest.approx([9.5, 18.5])
    assert upper == pytest.approx([12.5, 21.5])


def test_calibrated_intervals_can_compute_quantile_from_calibration_data():
    lower, upper = calibrated_intervals(
        [5.0],
        y_calibration=[0.0, 2.0, 4.0],
        calibration_predictions=[0.0, 1.0, 7.0],
        alpha=0.5,
    )

    assert lower == pytest.approx([2.0])
    assert upper == pytest.approx([8.0])


def test_pinball_interval_coverage_and_width_metrics():
    y_true = np.array([0.0, 2.0, 4.0, 8.0])
    y_pred = np.array([1.0, 1.0, 5.0, 6.0])
    lower = np.array([-1.0, 1.0, 4.5, 5.5])
    upper = np.array([1.0, 3.0, 7.5, 7.0])

    assert pinball_loss(y_true, y_pred, quantile=0.8) == pytest.approx(0.7)
    assert interval_coverage(y_true, lower, upper) == pytest.approx(0.5)
    assert mean_interval_width(lower, upper) == pytest.approx(2.125)


@pytest.mark.parametrize("backend", available_backends("affine"))
def test_pinball_loss_accepts_every_affine_backend(backend):
    assert pinball_loss(
        [1.0, 3.0],
        [0.5, 4.0],
        quantile=0.8,
        backend=backend,
    ) == pytest.approx(0.3)


@pytest.mark.parametrize("backend", available_backends("affine"))
def test_weighted_pinball_loss_accepts_every_affine_backend(backend):
    assert pinball_loss(
        [1.0, 3.0],
        [0.5, 4.0],
        quantile=0.8,
        sample_weight=[1.0, 3.0],
        backend=backend,
    ) == pytest.approx(0.25)


def test_jitter_volatility_uses_per_sample_instability():
    predictions = np.array(
        [
            [10.0, 20.0],
            [12.0, 18.0],
            [14.0, 22.0],
        ]
    )

    assert jitter_volatility(predictions) == pytest.approx(np.mean(np.std(predictions, axis=0)))
    assert jitter_volatility(predictions, baseline=[12.0, 20.0]) == pytest.approx(
        np.mean([np.sqrt(8.0 / 3.0), np.sqrt(8.0 / 3.0)])
    )


def test_classification_metrics_cover_probability_quality():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    assert logloss(y_true, y_proba) < 0.2
    assert brier_score(y_true, y_proba) == pytest.approx(0.025)
    assert roc_auc(y_true, y_proba) == pytest.approx(1.0)
    assert pr_auc(y_true, y_proba) == pytest.approx(1.0)
    assert ece_calibration_error(y_true, y_proba, n_bins=2) == pytest.approx(0.15)


def test_classification_metrics_accept_mixed_hashable_labels():
    y_true = np.array(["airport", "airport", 1, 1], dtype=object)
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    assert logloss(y_true, y_proba) < 0.2
    assert brier_score(y_true, y_proba) == pytest.approx(0.025)
    assert roc_auc(y_true, y_proba) == pytest.approx(1.0)
    assert pr_auc(y_true, y_proba) == pytest.approx(1.0)
    assert ece_calibration_error(y_true, y_proba, n_bins=2) == pytest.approx(0.15)


def test_classification_metrics_accept_tuple_labels():
    y_true = [("zone", "airport"), ("zone", "airport"), ("zone", "midtown"), ("zone", "midtown")]
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    assert logloss(y_true, y_proba) < 0.2
    assert brier_score(y_true, y_proba) == pytest.approx(0.025)
    assert roc_auc(y_true, y_proba) == pytest.approx(1.0)
    assert pr_auc(y_true, y_proba) == pytest.approx(1.0)
    assert ece_calibration_error(y_true, y_proba, n_bins=2) == pytest.approx(0.15)


def test_multiclass_logloss_respects_explicit_label_order():
    y_true = np.array(["airport", "street", "dispatch"], dtype=object)
    y_proba = np.array(
        [
            [0.85, 0.10, 0.05],
            [0.15, 0.80, 0.05],
            [0.20, 0.10, 0.70],
        ]
    )

    loss = logloss(
        y_true,
        y_proba,
        labels=["airport", "street", "dispatch"],
    )

    assert loss == pytest.approx(-np.mean(np.log([0.85, 0.80, 0.70])))


def test_grouped_ranking_metrics_cover_ndcg_map_mrr():
    relevance = np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    good_scores = np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    bad_scores = -good_scores
    groups = [3, 3]

    assert ndcg_at_k(relevance, good_scores, groups=groups, k=3) == pytest.approx(1.0)
    assert mean_average_precision(relevance, good_scores, groups=groups) == pytest.approx(1.0)
    assert mean_reciprocal_rank(relevance, good_scores, groups=groups) == pytest.approx(1.0)
    assert ndcg_at_k(relevance, good_scores, groups=groups) > ndcg_at_k(
        relevance,
        bad_scores,
        groups=groups,
    )


def test_ranking_metrics_accept_contiguous_query_ids():
    relevance = np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    scores = np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    query_ids = ["pickup-a", "pickup-a", "pickup-a", "pickup-b", "pickup-b", "pickup-b"]

    assert ndcg_at_k(relevance, scores, groups=query_ids) == pytest.approx(1.0)
    assert mean_average_precision(relevance, scores, groups=query_ids) == pytest.approx(1.0)
    assert mean_reciprocal_rank(relevance, scores, groups=query_ids) == pytest.approx(1.0)


def test_ranking_metrics_prefer_explicit_group_sizes_when_ambiguous():
    relevance = [1.0, 0.0]
    scores = [0.0, 1.0]

    assert ndcg_at_k(relevance, scores, groups=[1, 1]) == pytest.approx(0.5)
    assert mean_average_precision(relevance, scores, groups=[1, 1]) == pytest.approx(0.5)
    assert mean_reciprocal_rank(relevance, scores, groups=[1, 1]) == pytest.approx(0.5)


def test_ranking_metrics_accept_numeric_query_ids_when_not_size_vector():
    relevance = np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    scores = np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    query_ids = [1, 1, 1, 2, 2, 2]

    assert ndcg_at_k(relevance, scores, groups=query_ids) == pytest.approx(1.0)
    assert mean_average_precision(relevance, scores, groups=query_ids) == pytest.approx(1.0)
    assert mean_reciprocal_rank(relevance, scores, groups=query_ids) == pytest.approx(1.0)


def test_ranking_metrics_accept_tuple_query_ids():
    relevance = np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    scores = np.array([0.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    query_ids = [
        ("pickup", "a"),
        ("pickup", "a"),
        ("pickup", "a"),
        ("pickup", "b"),
        ("pickup", "b"),
        ("pickup", "b"),
    ]

    assert ndcg_at_k(relevance, scores, groups=query_ids) == pytest.approx(1.0)
    assert mean_average_precision(relevance, scores, groups=query_ids) == pytest.approx(1.0)
    assert mean_reciprocal_rank(relevance, scores, groups=query_ids) == pytest.approx(1.0)


def test_ranking_ndcg_uses_top_k_ideal_and_clamps_negative_relevance():
    assert ndcg_at_k([0.0, 3.0, 2.0], [0.0, 3.0, 2.0], k=1) == pytest.approx(1.0)
    assert ndcg_at_k([-2.0, 1.0], [0.0, 1.0], k=1) == pytest.approx(1.0)
    assert ndcg_at_k([-2.0, 1.0], [1.0, 0.0], k=1) == pytest.approx(0.0)


def test_spatial_cv_gap_is_random_minus_spatial_score():
    assert spatial_cv_gap(0.91, 0.73) == pytest.approx(0.18)


def test_residual_morans_i_supports_inverse_distance_and_radius_weights():
    coordinates = np.array([[0.0], [1.0], [2.0], [3.0]])
    clustered_residuals = np.array([1.0, 1.0, -1.0, -1.0])

    inverse_i = residual_morans_i(
        coordinates,
        clustered_residuals,
        weights="inverse_distance",
    )
    radius_i = residual_morans_i(
        coordinates,
        clustered_residuals,
        weights="radius",
        radius=1.1,
    )

    assert inverse_i == pytest.approx(-1.0 / 13.0)
    assert radius_i == pytest.approx(1.0 / 3.0)


def test_residual_morans_i_dispatches_distance_and_contraction(monkeypatch):
    calls: list[tuple[str, str]] = []

    def decision(backend, operation, workload_size, minimum_accelerated_size):
        calls.append(("decision", operation))
        return {"executed": "cpu" if backend == "cpu" else "cuda"}

    def distances(left, right=None, backend=None):
        calls.append(("distance", str(backend)))
        left_array = np.asarray(left, dtype=float)
        right_array = left_array if right is None else np.asarray(right, dtype=float)
        delta = left_array[:, None, :] - right_array[None, :, :]
        return np.sum(delta * delta, axis=2).astype(np.float32)

    def dense(features, weights, biases, backend=None):
        calls.append(("dense", str(backend)))
        return (
            np.asarray(features, dtype=np.float32)
            @ np.asarray(weights, dtype=np.float32)
            + np.asarray(biases, dtype=np.float32)
        )

    monkeypatch.setattr(metrics_module, "workload_decision", decision)
    monkeypatch.setattr(metrics_module, "pairwise_squared_distances", distances)
    monkeypatch.setattr(metrics_module, "dense_layer", dense)
    coordinates = np.arange(256, dtype=float).reshape(-1, 1)
    residuals = np.sin(coordinates[:, 0] * 0.1)

    accelerated = residual_morans_i(coordinates, residuals, backend="cuda")
    expected = residual_morans_i(coordinates, residuals, backend="cpu")

    assert accelerated == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert ("decision", "pairwise_distance") in calls
    assert ("decision", "dense") in calls
    assert ("distance", "cuda") in calls
    assert ("dense", "cuda") in calls


def test_metric_validation_rejects_bad_shapes_and_weights():
    with pytest.raises(ValueError, match="same shape"):
        pinball_loss([1.0], [1.0, 2.0], quantile=0.5)
    with pytest.raises(ValueError, match="lower bounds"):
        interval_coverage([1.0], [2.0], [0.0])
    with pytest.raises(ValueError, match="at least two jitter repeats"):
        jitter_volatility([[1.0, 2.0]])
    with pytest.raises(ValueError, match="positive value"):
        residual_morans_i([[0.0], [1.0]], [1.0, -1.0], weights="radius")
