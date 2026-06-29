from __future__ import annotations

import numpy as np


def expected_calibration_error(probability, actual, *, bins: int = 10) -> float:
    probs = np.asarray(probability, dtype=float)
    y = np.asarray(actual, dtype=float)
    if probs.shape != y.shape:
        raise ValueError("probability and actual must have the same shape")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if mask.any():
            total += mask.mean() * abs(float(probs[mask].mean() - y[mask].mean()))
    return float(total)
