"""Evaluation helpers for link ranking and graph-neighborhood tasks."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .. import _native


def _safe_float_array(values: Sequence[float] | Sequence[int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("metric input must be 1D")
    return array


def binary_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute ROC-AUC without external dependencies."""
    y = _safe_float_array(labels)
    s = _safe_float_array(scores)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have equal length")
    return float(_native.binary_roc_auc_value(y.tolist(), s.tolist()))


def binary_average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Compute area under the precision-recall staircase."""
    y = _safe_float_array(labels)
    s = _safe_float_array(scores)
    if y.shape != s.shape:
        raise ValueError("labels and scores must have equal length")

    order = np.argsort(-s)
    sorted_labels = y[order]
    positive_count = float(np.sum(sorted_labels > 0.5))
    if positive_count == 0.0:
        return 0.0

    tp = 0.0
    precision_acc = 0.0
    for index, label in enumerate(sorted_labels, start=1):
        if label > 0.5:
            tp += 1.0
            precision_acc += tp / float(index)
    return precision_acc / positive_count


def top_k_metrics(
    labels: Sequence[int],
    scores: Sequence[float],
    query_ids: Sequence[Any],
    k: int,
) -> dict[str, float]:
    """Group-wise top-k recall-style metrics."""
    y = _safe_float_array(labels)
    s = _safe_float_array(scores)
    query_values = list(query_ids)
    if not (y.shape == s.shape == (len(query_values),)):
        raise ValueError("labels, scores, and query_ids must align")
    if k <= 0:
        raise ValueError("k must be positive")

    query_to_indices: dict[Any, list[int]] = defaultdict(list)
    for index, key in enumerate(query_values):
        query_to_indices[key].append(index)

    recalls: list[float] = []
    hit_rates: list[float] = []
    for indices in query_to_indices.values():
        if not indices:
            continue
        ordered = sorted(indices, key=lambda index: s[index], reverse=True)
        topk = ordered[:k]
        positives = [i for i in indices if y[i] > 0.5]
        positives_in_topk = [i for i in topk if y[i] > 0.5]
        if positives:
            recalls.append(len(positives_in_topk) / float(len(positives)))
            hit_rates.append(1.0 if positives_in_topk else 0.0)

    if not recalls:
        return {"recall_at_k": 0.0, "hit_rate_at_k": 0.0}
    return {
        "recall_at_k": float(np.mean(recalls)),
        "hit_rate_at_k": float(np.mean(hit_rates)),
    }


def mean_reciprocal_rank(
    labels: Sequence[int],
    scores: Sequence[float],
    query_ids: Sequence[Any],
) -> float:
    """Mean reciprocal rank over grouped candidate sets."""
    y = _safe_float_array(labels)
    s = _safe_float_array(scores)
    query_values = list(query_ids)
    if not (y.shape == s.shape == (len(query_values),)):
        raise ValueError("labels, scores, and query_ids must align")

    group_lookup: dict[Any, int] = {}
    group_ids = [group_lookup.setdefault(key, len(group_lookup)) for key in query_values]
    return float(_native.graph_mean_reciprocal_rank_value(y.tolist(), s.tolist(), group_ids))


def link_prediction_report(
    labels: Sequence[int],
    scores: Sequence[float],
    query_ids: Sequence[Any] | None = None,
    *,
    k: int = 10,
) -> dict[str, float]:
    """Return a compact link-prediction and link-ranking metric bundle."""
    report = {
        "auc": binary_auc(labels, scores),
        "average_precision": binary_average_precision(labels, scores),
    }
    if query_ids is None:
        return report
    report.update(top_k_metrics(labels, scores, query_ids=query_ids, k=k))
    report["mean_reciprocal_rank"] = mean_reciprocal_rank(labels, scores, query_ids)
    return report
