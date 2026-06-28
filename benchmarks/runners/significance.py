"""Small statistical helpers for benchmark reports."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def normal_mean_ci(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float, float]:
    if not values:
        raise ValueError("values must not be empty")
    center = mean(values)
    if len(values) == 1:
        return center, center, center
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    z = 1.96 if confidence == 0.95 else 1.0
    half_width = z * math.sqrt(variance / len(values))
    return center, center - half_width, center + half_width


def paired_bootstrap_ci(
    challenger: Sequence[float],
    baseline: Sequence[float],
    *,
    iterations: int = 2000,
    seed: int = 11,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    if len(challenger) != len(baseline):
        raise ValueError("paired samples must have the same length")
    if not challenger:
        raise ValueError("paired samples must not be empty")

    deltas = [left - right for left, right in zip(challenger, baseline, strict=True)]
    observed = mean(deltas)
    rng = random.Random(seed)
    boot = []
    n = len(deltas)
    for _ in range(iterations):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boot.append(mean(sample))
    boot.sort()
    alpha = (1.0 - confidence) / 2.0
    low_idx = max(0, min(len(boot) - 1, int(alpha * len(boot))))
    high_idx = max(0, min(len(boot) - 1, int((1.0 - alpha) * len(boot)) - 1))
    return observed, boot[low_idx], boot[high_idx]


def average_ranks(rows: Sequence[dict[str, float]], *, lower_is_better: bool) -> dict[str, float]:
    if not rows:
        raise ValueError("rows must not be empty")
    totals: dict[str, float] = {}
    for row in rows:
        ordered = sorted(row.items(), key=lambda item: item[1], reverse=not lower_is_better)
        for rank, (model, _value) in enumerate(ordered, start=1):
            totals[model] = totals.get(model, 0.0) + rank
    return {model: total / len(rows) for model, total in sorted(totals.items())}
