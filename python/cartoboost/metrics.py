"""Metrics and diagnostics for CartoBoost regression, classification, and ranking."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import numpy as np

from . import _native
from .accelerators import dense_layer, pairwise_squared_distances, workload_decision
from .config import Backend

__path__ = [str(Path(__file__).with_suffix(""))]

__all__ = [
    "brier_score",
    "calibrated_intervals",
    "conformal_residual_quantile",
    "ece_calibration_error",
    "interval_coverage",
    "jitter_volatility",
    "logloss",
    "mean_average_precision",
    "mean_interval_width",
    "mean_reciprocal_rank",
    "competition_forecast_metrics",
    "aggregate_equal_level_wrmsse",
    "ndcg_at_k",
    "pinball_loss",
    "pr_auc",
    "residual_morans_i",
    "roc_auc",
    "rmsse_scale",
    "spatial_cv_gap",
    "wrmsse",
]


def conformal_residual_quantile(
    y_true: object,
    y_pred: object,
    *,
    alpha: float = 0.1,
) -> float:
    """Return the finite-sample split-conformal absolute residual quantile.

    ``alpha`` is the tolerated miscoverage rate, so ``alpha=0.1`` targets 90%
    marginal coverage. The returned quantile can be added to and subtracted
    from point predictions to form symmetric calibrated intervals.

    Example:
        >>> conformal_residual_quantile([10.0, 12.0], [9.0, 13.5], alpha=0.5)
        1.5
    """

    _validate_alpha(alpha)
    y_true_arr, y_pred_arr = _paired_vectors(y_true, y_pred, "y_true", "y_pred")
    scores = np.abs(y_true_arr - y_pred_arr)
    if scores.size == 0:
        raise ValueError("calibration data must contain at least one row")

    quantile_level = min(1.0, np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size)
    return float(np.quantile(scores, quantile_level, method="higher"))


def calibrated_intervals(
    y_pred: object,
    residual_quantile: float | None = None,
    *,
    y_calibration: object | None = None,
    calibration_predictions: object | None = None,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build symmetric conformal intervals around predictions.

    Pass either a precomputed ``residual_quantile`` or calibration truth and
    predictions. The helper returns ``(lower, upper)`` arrays matching
    ``y_pred``.

    Example:
        >>> lower, upper = calibrated_intervals([10.0, 12.0], residual_quantile=1.0)
        >>> lower.tolist(), upper.tolist()
        ([9.0, 11.0], [11.0, 13.0])
    """

    predictions = _as_float_vector(y_pred, "y_pred")
    if residual_quantile is None:
        if y_calibration is None or calibration_predictions is None:
            raise ValueError(
                "provide residual_quantile or both y_calibration and calibration_predictions"
            )
        residual_quantile = conformal_residual_quantile(
            y_calibration,
            calibration_predictions,
            alpha=alpha,
        )
    q = float(residual_quantile)
    if not np.isfinite(q) or q < 0.0:
        raise ValueError("residual_quantile must be a finite non-negative value")

    return predictions - q, predictions + q


def pinball_loss(
    y_true: object,
    y_pred: object,
    *,
    quantile: float,
    sample_weight: object | None = None,
    backend: Backend | str = Backend.CPU,
) -> float:
    """Return mean pinball loss for a quantile prediction.

    Example:
        >>> pinball_loss([1.0, 3.0], [2.0, 2.0], quantile=0.5)
        0.5
    """

    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    y_true_arr, y_pred_arr = _paired_vectors(y_true, y_pred, "y_true", "y_pred")
    weights = None if sample_weight is None else _as_float_vector(sample_weight, "sample_weight")
    if weights is not None and weights.shape != y_true_arr.shape:
        raise ValueError("sample_weight must have the same length as y_true")
    return float(
        _native.prob_pinball_loss_value(
            y_true_arr.tolist(),
            y_pred_arr.tolist(),
            float(quantile),
            str(backend),
            None if weights is None else weights.tolist(),
        )
    )


def interval_coverage(
    y_true: object,
    lower: object,
    upper: object,
    *,
    sample_weight: object | None = None,
    backend: Backend | str = Backend.CPU,
) -> float:
    """Return the share of targets inside inclusive prediction intervals.

    Example:
        >>> interval_coverage([1.0, 3.0], [0.0, 2.0], [2.0, 2.5])
        0.5
    """

    y_true_arr, lower_arr = _paired_vectors(y_true, lower, "y_true", "lower")
    _, upper_arr = _paired_vectors(y_true_arr, upper, "y_true", "upper")
    if np.any(lower_arr > upper_arr):
        raise ValueError("lower bounds must be less than or equal to upper bounds")
    weights = None if sample_weight is None else _as_float_vector(sample_weight, "sample_weight")
    if weights is not None and weights.shape != y_true_arr.shape:
        raise ValueError("sample_weight must have the same length as y_true")
    return float(
        _native.prob_interval_coverage_value(
            y_true_arr.tolist(),
            lower_arr.tolist(),
            upper_arr.tolist(),
            str(backend),
            None if weights is None else weights.tolist(),
        )
    )


def mean_interval_width(
    lower: object,
    upper: object,
    *,
    sample_weight: object | None = None,
    backend: Backend | str = Backend.CPU,
) -> float:
    """Return the mean width of prediction intervals.

    Example:
        >>> mean_interval_width([0.0, 2.0], [2.0, 5.0])
        2.5
    """

    lower_arr, upper_arr = _paired_vectors(lower, upper, "lower", "upper")
    widths = upper_arr - lower_arr
    if np.any(widths < 0.0):
        raise ValueError("lower bounds must be less than or equal to upper bounds")
    weights = None if sample_weight is None else _as_float_vector(sample_weight, "sample_weight")
    if weights is not None and weights.shape != lower_arr.shape:
        raise ValueError("sample_weight must have the same length as lower")
    return float(
        _native.prob_mean_interval_width_value(
            lower_arr.tolist(),
            upper_arr.tolist(),
            str(backend),
            None if weights is None else weights.tolist(),
        )
    )


def logloss(
    y_true: object,
    y_proba: object,
    *,
    labels: object | None = None,
    eps: float = 1e-15,
    sample_weight: object | None = None,
) -> float:
    """Return binary or multiclass negative log likelihood.

    Example:
        >>> round(logloss([0, 1], [0.1, 0.9]), 3)
        0.105
    """

    truth, probabilities, class_labels = _classification_arrays(y_true, y_proba, labels)
    clipped = np.clip(probabilities, float(eps), 1.0 - float(eps))
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    label_to_index = {label: idx for idx, label in enumerate(class_labels.tolist())}
    target_indices = np.asarray([label_to_index[label] for label in truth.tolist()], dtype=int)
    losses = -np.log(clipped[np.arange(truth.shape[0]), target_indices])
    return _weighted_mean(losses, sample_weight)


def brier_score(
    y_true: object,
    y_proba: object,
    *,
    positive_label: object | None = None,
    sample_weight: object | None = None,
    backend: Backend | str = Backend.CPU,
) -> float:
    """Return the binary Brier score for positive-class probabilities.

    Example:
        >>> brier_score([0, 1], [0.25, 0.75])
        0.0625
    """

    truth = _as_object_vector(y_true, "y_true")
    proba = _positive_probability(y_proba)
    if truth.ndim != 1 or truth.shape[0] != proba.shape[0]:
        raise ValueError("y_true and y_proba must have matching row counts")
    positive = _resolve_positive_label(truth, positive_label)
    target = np.asarray([value == positive for value in truth.tolist()], dtype=float)
    weights = None if sample_weight is None else _as_float_vector(sample_weight, "sample_weight")
    if weights is not None and weights.shape != target.shape:
        raise ValueError("sample_weight must have the same length as y_true")
    return float(
        _native.prob_brier_score_value(
            target.tolist(),
            proba.tolist(),
            str(backend),
            None if weights is None else weights.tolist(),
        )
    )


def roc_auc(
    y_true: object,
    y_score: object,
    *,
    positive_label: object | None = None,
    sample_weight: object | None = None,
) -> float:
    """Return binary ROC-AUC using deterministic rank statistics.

    Example:
        >>> roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        1.0
    """

    truth = _as_object_vector(y_true, "y_true")
    scores = _as_float_vector(y_score, "y_score")
    if truth.ndim != 1 or truth.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have matching row counts")
    positive = _resolve_positive_label(truth, positive_label)
    binary = np.asarray([value == positive for value in truth.tolist()], dtype=int)
    if np.unique(binary).size != 2:
        raise ValueError("ROC-AUC requires both positive and negative labels")
    if sample_weight is not None:
        return _weighted_binary_auc(binary, scores, sample_weight)
    ranks = _average_ranks(scores)
    pos_ranks = ranks[binary == 1]
    n_pos = float(pos_ranks.size)
    n_neg = float(scores.size - pos_ranks.size)
    return float((np.sum(pos_ranks) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def pr_auc(
    y_true: object,
    y_score: object,
    *,
    positive_label: object | None = None,
) -> float:
    """Return binary precision-recall AUC with step-wise interpolation.

    Example:
        >>> pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        1.0
    """

    truth = _as_object_vector(y_true, "y_true")
    scores = _as_float_vector(y_score, "y_score")
    if truth.ndim != 1 or truth.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have matching row counts")
    positive = _resolve_positive_label(truth, positive_label)
    binary = np.asarray([value == positive for value in truth.tolist()], dtype=int)
    positives = int(np.sum(binary))
    if positives == 0 or positives == binary.shape[0]:
        raise ValueError("PR-AUC requires both positive and negative labels")
    order = np.argsort(-scores, kind="mergesort")
    sorted_binary = binary[order]
    tp = np.cumsum(sorted_binary)
    fp = np.cumsum(1 - sorted_binary)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / positives
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def ece_calibration_error(
    y_true: object,
    y_proba: object,
    *,
    positive_label: object | None = None,
    n_bins: int = 10,
) -> float:
    """Return expected calibration error for binary probabilities.

    Example:
        >>> round(ece_calibration_error([0, 1], [0.25, 0.75], n_bins=2), 3)
        0.25
    """

    if int(n_bins) <= 0:
        raise ValueError("n_bins must be positive")
    truth = _as_object_vector(y_true, "y_true")
    proba = _positive_probability(y_proba)
    if truth.ndim != 1 or truth.shape[0] != proba.shape[0]:
        raise ValueError("y_true and y_proba must have matching row counts")
    positive = _resolve_positive_label(truth, positive_label)
    target = np.asarray([value == positive for value in truth.tolist()], dtype=float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    error = 0.0
    for idx in range(int(n_bins)):
        if idx == int(n_bins) - 1:
            mask = (proba >= edges[idx]) & (proba <= edges[idx + 1])
        else:
            mask = (proba >= edges[idx]) & (proba < edges[idx + 1])
        if not np.any(mask):
            continue
        confidence = float(np.mean(proba[mask]))
        accuracy = float(np.mean(target[mask]))
        error += float(np.mean(mask)) * abs(confidence - accuracy)
    return error


def ndcg_at_k(
    relevance: object,
    scores: object,
    *,
    groups: object | None = None,
    k: int | None = None,
) -> float:
    """Return mean NDCG@k, optionally averaged over query groups.

    Example:
        >>> ndcg_at_k([0.0, 3.0, 1.0], [0.1, 0.9, 0.3], k=2)
        1.0
    """

    rel, pred, group_sizes = _ranking_inputs(relevance, scores, groups)
    values = [_ndcg_one(rel[start:stop], pred[start:stop], k) for start, stop in group_sizes]
    return float(np.mean(values))


def mean_average_precision(
    relevance: object,
    scores: object,
    *,
    groups: object | None = None,
    k: int | None = None,
) -> float:
    """Return mean average precision over binary-positive relevance.

    Example:
        >>> mean_average_precision([0.0, 1.0, 1.0], [0.1, 0.9, 0.8])
        1.0
    """

    rel, pred, group_sizes = _ranking_inputs(relevance, scores, groups)
    values = [
        _average_precision_one(rel[start:stop], pred[start:stop], k) for start, stop in group_sizes
    ]
    return float(np.mean(values))


def mean_reciprocal_rank(
    relevance: object,
    scores: object,
    *,
    groups: object | None = None,
    k: int | None = None,
) -> float:
    """Return mean reciprocal rank over binary-positive relevance.

    Example:
        >>> mean_reciprocal_rank([0.0, 1.0, 0.0], [0.2, 0.9, 0.8])
        1.0
    """

    rel, pred, group_sizes = _ranking_inputs(relevance, scores, groups)
    values = [
        _reciprocal_rank_one(rel[start:stop], pred[start:stop], k) for start, stop in group_sizes
    ]
    return float(np.mean(values))


def spatial_cv_gap(random_cv_score: float, spatial_cv_score: float) -> float:
    """Return random-CV score minus spatial-CV score.

    Example:
        >>> spatial_cv_gap(0.91, 0.73)
        0.18000000000000005
    """

    random_score = float(random_cv_score)
    spatial_score = float(spatial_cv_score)
    if not np.isfinite(random_score) or not np.isfinite(spatial_score):
        raise ValueError("CV scores must be finite")
    return random_score - spatial_score


def rmsse_scale(y_train: object, *, seasonal_period: int = 1) -> float:
    """Return the squared seasonal-naive scale used by RMSSE and WRMSSE.

    Example:
        >>> rmsse_scale([1.0, 2.0, 4.0], seasonal_period=1)
        2.5
    """

    if seasonal_period <= 0:
        raise ValueError("seasonal_period must be positive")
    train = _as_float_vector(y_train, "y_train")
    return float(_native.rmsse_scale_value(train.tolist(), int(seasonal_period)))


def wrmsse(
    y_train: object,
    y_true: object,
    y_pred: object,
    weights: object,
    *,
    seasonal_period: int = 1,
    series_ids: object | None = None,
    return_breakdown: bool = False,
) -> float | dict[str, object]:
    """Return official-style weighted RMSSE for hierarchical forecasting.

    Inputs are shaped ``(n_series, n_time)`` for training history and
    ``(n_series, horizon)`` for actual and predicted forecast windows. Weights
    are non-negative value weights and are normalized inside the metric.

    Example:
        >>> round(wrmsse([[1, 2, 3]], [[4]], [[3]], [1.0]), 6)
        1.0
    """

    train = _as_float_matrix(y_train, "y_train")
    truth = _as_float_matrix(y_true, "y_true")
    pred = _as_float_matrix(y_pred, "y_pred")
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if train.shape[0] != truth.shape[0]:
        raise ValueError("y_train, y_true, and y_pred must have the same number of series")
    if truth.shape[1] == 0:
        raise ValueError("y_true and y_pred must contain at least one horizon")

    weight_arr = _as_float_vector(weights, "weights")
    if weight_arr.shape[0] != truth.shape[0]:
        raise ValueError("weights length must match the number of series")
    if series_ids is None:
        ids = [str(idx) for idx in range(truth.shape[0])]
    else:
        ids_arr = np.asarray(series_ids, dtype=object)
        if ids_arr.ndim != 1 or ids_arr.shape[0] != truth.shape[0]:
            raise ValueError("series_ids must be one-dimensional and match the number of series")
        ids = [str(value) for value in ids_arr]
        if any(not value for value in ids):
            raise ValueError("series_ids must be non-empty")

    series = [
        (
            ids[idx],
            train[idx].tolist(),
            truth[idx].tolist(),
            pred[idx].tolist(),
            float(weight_arr[idx]),
        )
        for idx in range(truth.shape[0])
    ]
    result = json.loads(_native.wrmsse_value(series, int(seasonal_period)))
    score = float(result["wrmsse"])

    if not return_breakdown:
        return score

    return result


def aggregate_equal_level_wrmsse(
    level_scores: Iterable[dict[str, object] | tuple[object, object]],
    *,
    return_breakdown: bool = False,
) -> float | dict[str, object]:
    """Aggregate hierarchy-level WRMSSE values with equal level weight.

    Example:
        >>> aggregate_equal_level_wrmsse([("state", 1.0), ("store", 3.0)])
        2.0
    """

    rows: list[tuple[str, float]] = []
    for row in level_scores:
        if isinstance(row, dict):
            level = str(row["level"])
            score = float(row["wrmsse"])
        else:
            level, score = row
            level = str(level)
            score = float(score)
        rows.append((level, score))
    result = json.loads(_native.aggregate_equal_level_wrmsse_value(rows))
    score = float(result["wrmsse"])
    if not return_breakdown:
        return score
    return result


def competition_forecast_metrics(
    training_series: object,
    y_true: object,
    y_pred: object,
    *,
    seasonality: int,
    baseline_smape: float | None = None,
    baseline_mase: float | None = None,
) -> dict[str, float | None]:
    """Return generic sMAPE, MASE, and optional baseline-relative metrics.

    Example:
        >>> result = competition_forecast_metrics([[1, 2, 3]], [4, 5], [4, 6], seasonality=1)
        >>> sorted(result)
        ['mase', 'owa', 'smape']
    """

    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    train_rows = [
        _as_float_vector(row, "training_series row").tolist()
        for row in np.asarray(training_series, dtype=object)
    ]
    truth, pred = _paired_vectors(y_true, y_pred, "y_true", "y_pred")
    payload = _native.competition_forecast_metrics_value(
        train_rows,
        truth.tolist(),
        pred.tolist(),
        int(seasonality),
        baseline_smape,
        baseline_mase,
    )
    return json.loads(payload)


def jitter_volatility(
    predictions: object,
    *,
    baseline: object | None = None,
    sample_weight: object | None = None,
) -> float:
    """Measure prediction instability under repeated coordinate jitter.

    ``predictions`` is expected to have shape ``(n_repeats, n_samples)``. Without
    a baseline, the metric is the weighted mean per-sample standard deviation
    across repeats. With a baseline, it is the weighted mean root-mean-square
    deviation from that baseline across repeats.

    Example:
        >>> round(jitter_volatility([[1.0, 2.0], [1.5, 3.0]]), 3)
        0.375
    """

    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2:
        raise ValueError("predictions must have shape (n_repeats, n_samples)")
    if values.shape[0] < 2:
        raise ValueError("predictions must contain at least two jitter repeats")
    if values.shape[1] == 0:
        raise ValueError("predictions must contain at least one sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("predictions must contain only finite values")

    if baseline is None:
        per_sample = np.std(values, axis=0)
    else:
        baseline_arr = _as_float_vector(baseline, "baseline")
        if baseline_arr.shape[0] != values.shape[1]:
            raise ValueError("baseline length must match the number of samples")
        per_sample = np.sqrt(np.mean((values - baseline_arr[None, :]) ** 2, axis=0))
    return _weighted_mean(per_sample, sample_weight)


def residual_morans_i(
    coordinates: object,
    residuals: object,
    *,
    weights: Literal["inverse_distance", "radius"] = "inverse_distance",
    radius: float | None = None,
    distance_epsilon: float = 1e-12,
    backend: Backend | str = Backend.CPU,
) -> float:
    """Return Moran's I for residual spatial autocorrelation.

    The implementation uses dense pairwise weights and is intended for
    evaluation samples, not very large production batches.

    Example:
        >>> round(residual_morans_i([[0.0], [1.0], [2.0]], [1.0, 0.0, -1.0]), 3)
        -0.3
    """

    coords = _as_coordinates(coordinates)
    residual_arr = _as_float_vector(residuals, "residuals")
    if coords.shape[0] != residual_arr.shape[0]:
        raise ValueError("coordinates and residuals must contain the same number of rows")
    if coords.shape[0] < 2:
        raise ValueError("at least two rows are required")
    if not np.isfinite(distance_epsilon) or distance_epsilon <= 0.0:
        raise ValueError("distance_epsilon must be a finite positive value")

    centered = residual_arr - np.mean(residual_arr)
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("residuals must have non-zero variance")

    distance_work = int(coords.shape[0] * coords.shape[0] * coords.shape[1])
    dispatch = workload_decision(str(backend), "pairwise_distance", distance_work, 16_384)
    if dispatch["executed"] == "cpu":
        distances = _pairwise_distances(coords)
    else:
        distances = np.sqrt(
            np.maximum(
                pairwise_squared_distances(
                    coords,
                    backend=str(dispatch["executed"]),
                ).astype(float),
                0.0,
            )
        )
    if weights == "inverse_distance":
        weight_matrix = 1.0 / np.maximum(distances, distance_epsilon)
        np.fill_diagonal(weight_matrix, 0.0)
    elif weights == "radius":
        if radius is None or not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be a finite positive value when weights='radius'")
        weight_matrix = ((distances <= radius) & (distances > 0.0)).astype(float)
    else:
        raise ValueError("weights must be 'inverse_distance' or 'radius'")

    weight_sum = float(np.sum(weight_matrix))
    if weight_sum == 0.0:
        raise ValueError("spatial weights contain no neighbor pairs")

    contraction_work = int(weight_matrix.shape[0] * weight_matrix.shape[1])
    contraction_dispatch = workload_decision(
        str(backend),
        "dense",
        contraction_work,
        16_384,
    )
    if contraction_dispatch["executed"] == "cpu":
        weighted_residuals = weight_matrix @ centered
    else:
        weighted_residuals = dense_layer(
            weight_matrix,
            centered.reshape(-1, 1),
            np.zeros(1, dtype=np.float32),
            backend=str(contraction_dispatch["executed"]),
        ).astype(float)[:, 0]
    numerator = float(np.dot(centered, weighted_residuals))
    return coords.shape[0] / weight_sum * numerator / denominator


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")


def _classification_arrays(
    y_true: object,
    y_proba: object,
    labels: object | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = _as_object_vector(y_true, "y_true")
    if truth.shape[0] == 0:
        raise ValueError("y_true must be a non-empty one-dimensional array")
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])
    if probabilities.ndim != 2 or probabilities.shape[0] != truth.shape[0]:
        raise ValueError("y_proba must have shape (n_samples,) or (n_samples, n_classes)")
    if probabilities.shape[1] < 2:
        raise ValueError("y_proba must contain at least two classes")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("y_proba must contain only finite values")
    if np.any(probabilities < 0.0):
        raise ValueError("y_proba must be non-negative")
    row_sums = probabilities.sum(axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError("each y_proba row must have positive total probability")
    if labels is None:
        class_labels = _as_object_vector(_unique_in_order(truth.tolist()), "labels")
    else:
        class_labels = _as_object_vector(labels, "labels")
    if class_labels.shape[0] != probabilities.shape[1]:
        raise ValueError("labels must match the number of probability columns")
    if not set(truth.tolist()).issubset(set(class_labels.tolist())):
        raise ValueError("labels must contain every y_true value")
    return truth, probabilities, class_labels


def _positive_probability(y_proba: object) -> np.ndarray:
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.ndim == 2:
        if probabilities.shape[1] != 2:
            raise ValueError("binary metric probabilities must have exactly two columns")
        probabilities = probabilities[:, 1]
    if probabilities.ndim != 1:
        raise ValueError("y_proba must be one-dimensional or a two-column matrix")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("y_proba must contain only finite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("y_proba must be in [0, 1]")
    return probabilities


def _resolve_positive_label(truth: np.ndarray, positive_label: object | None) -> object:
    labels = _unique_in_order(truth.tolist())
    if len(labels) != 2:
        raise ValueError("binary metrics require exactly two labels")
    if positive_label is None:
        return labels[1]
    if positive_label not in labels:
        raise ValueError("positive_label must be present in y_true")
    return positive_label


def _unique_in_order(values: list[object]) -> list[object]:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    start = 0
    while start < values.shape[0]:
        stop = start + 1
        while stop < values.shape[0] and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _weighted_binary_auc(binary: np.ndarray, scores: np.ndarray, sample_weight: object) -> float:
    weights = _as_float_vector(sample_weight, "sample_weight")
    if weights.shape != scores.shape:
        raise ValueError("sample_weight must match y_score")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be non-negative")
    pos_mask = binary == 1
    neg_mask = ~pos_mask
    pos_weight = float(np.sum(weights[pos_mask]))
    neg_weight = float(np.sum(weights[neg_mask]))
    if pos_weight <= 0.0 or neg_weight <= 0.0:
        raise ValueError("weighted ROC-AUC requires positive weight for both labels")
    total = 0.0
    for pos_score, pos_w in zip(scores[pos_mask], weights[pos_mask], strict=True):
        wins = np.sum(weights[neg_mask] * (pos_score > scores[neg_mask]))
        ties = 0.5 * np.sum(weights[neg_mask] * (pos_score == scores[neg_mask]))
        total += float(pos_w) * float(wins + ties)
    return total / (pos_weight * neg_weight)


def _ranking_inputs(
    relevance: object,
    scores: object,
    groups: object | None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    rel, pred = _paired_vectors(relevance, scores, "relevance", "scores")
    if groups is None:
        return rel, pred, [(0, rel.shape[0])]
    group_values = _as_object_vector(groups, "groups")
    size_values = _try_group_sizes(group_values)
    if size_values is not None and sum(size_values) == rel.shape[0]:
        stops = np.cumsum(size_values)
        starts = np.concatenate([[0], stops[:-1]])
        return (
            rel,
            pred,
            [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)],
        )
    if group_values.shape[0] != rel.shape[0]:
        sizes = [int(value) for value in group_values.tolist()]
        if any(size <= 0 for size in sizes) or sum(sizes) != rel.shape[0]:
            raise ValueError("group sizes must be positive and sum to row count")
        stops = np.cumsum(sizes)
        starts = np.concatenate([[0], stops[:-1]])
        return (
            rel,
            pred,
            [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)],
        )
    spans: list[tuple[int, int]] = []
    start = 0
    seen = set()
    current = group_values[0]
    for idx, value in enumerate(group_values.tolist()):
        if value == current:
            continue
        seen.add(current)
        if value in seen:
            raise ValueError("query id groups must be contiguous")
        spans.append((start, idx))
        start = idx
        current = value
    spans.append((start, group_values.shape[0]))
    return rel, pred, spans


def _try_group_sizes(values: np.ndarray) -> list[int] | None:
    sizes = []
    for value in values.tolist():
        if isinstance(value, bool):
            return None
        try:
            size = int(value)
        except (TypeError, ValueError):
            return None
        if size != value or size <= 0:
            return None
        sizes.append(size)
    return sizes


def _ndcg_one(relevance: np.ndarray, scores: np.ndarray, k: int | None) -> float:
    limit = _rank_limit(relevance.shape[0], k)
    order = np.argsort(-scores, kind="mergesort")[:limit]
    ideal = np.argsort(-relevance, kind="mergesort")[:limit]
    dcg = _dcg(relevance[order])
    idcg = _dcg(relevance[ideal])
    return 0.0 if idcg == 0.0 else dcg / idcg


def _average_precision_one(relevance: np.ndarray, scores: np.ndarray, k: int | None) -> float:
    limit = _rank_limit(relevance.shape[0], k)
    positives = relevance > 0.0
    positive_count = int(np.sum(positives))
    if positive_count == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")[:limit]
    hits = positives[order].astype(float)
    cumulative_hits = np.cumsum(hits)
    precision = cumulative_hits / np.arange(1, hits.shape[0] + 1, dtype=float)
    return float(np.sum(precision * hits) / min(positive_count, limit))


def _reciprocal_rank_one(relevance: np.ndarray, scores: np.ndarray, k: int | None) -> float:
    limit = _rank_limit(relevance.shape[0], k)
    order = np.argsort(-scores, kind="mergesort")[:limit]
    hits = np.nonzero(relevance[order] > 0.0)[0]
    if hits.size == 0:
        return 0.0
    return 1.0 / float(hits[0] + 1)


def _rank_limit(row_count: int, k: int | None) -> int:
    if k is None:
        return row_count
    if int(k) <= 0:
        raise ValueError("k must be positive")
    return min(int(k), row_count)


def _dcg(gains: np.ndarray) -> float:
    gains = np.maximum(gains, 0.0)
    discounts = 1.0 / np.log2(np.arange(2, gains.shape[0] + 2, dtype=float))
    return float(np.sum((np.power(2.0, gains) - 1.0) * discounts))


def _as_float_vector(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_object_vector(values: object, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        arr = np.empty(values.shape[0], dtype=object)
        arr[:] = [
            values[index].item() if hasattr(values[index], "item") else values[index]
            for index in range(values.shape[0])
        ]
        return arr
    items = list(values)  # type: ignore[arg-type]
    arr = np.empty(len(items), dtype=object)
    arr[:] = items
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return arr


def _as_float_matrix(values: object, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one column")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _paired_vectors(
    left: object,
    right: object,
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[np.ndarray, np.ndarray]:
    left_arr = _as_float_vector(left, left_name)
    right_arr = _as_float_vector(right, right_name)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")
    if left_arr.size == 0:
        raise ValueError(f"{left_name} and {right_name} must contain at least one value")
    return left_arr, right_arr


def _weighted_mean(values: np.ndarray, sample_weight: object | None) -> float:
    if values.size == 0:
        raise ValueError("values must contain at least one row")
    if sample_weight is None:
        return float(np.mean(values))
    weights = _as_float_vector(sample_weight, "sample_weight")
    if weights.shape != values.shape:
        raise ValueError("sample_weight must match the metric value shape")
    if np.any(weights < 0.0):
        raise ValueError("sample_weight must be non-negative")
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0:
        raise ValueError("sample_weight must have positive total weight")
    return float(np.average(values, weights=weights))


def _as_coordinates(coordinates: object) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    if coords.ndim != 2 or coords.shape[1] == 0:
        raise ValueError("coordinates must have shape (n_samples, n_dimensions)")
    if not np.all(np.isfinite(coords)):
        raise ValueError("coordinates must contain only finite values")
    return coords


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))
