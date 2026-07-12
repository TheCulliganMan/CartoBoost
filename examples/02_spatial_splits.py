#!/usr/bin/env python3
"""Train CartoBoost structured spatial candidates on deterministic datasets."""

from __future__ import annotations

import math

from cartoboost import CartoBoostRegressor
from cartoboost.schema import FeatureSchema, NumericSpec, SpatialPairSpec


def diagonal_dataset() -> tuple[list[list[float]], list[float]]:
    points: list[list[float]] = []
    targets: list[float] = []
    for x in [-2.0, -1.0, 1.0, 2.0]:
        for y in [-2.0, -1.0, 1.0, 2.0]:
            points.append([x, y])
            targets.append(10.0 if x + y > 0.0 else -10.0)
    return points, targets


def radial_dataset() -> tuple[list[list[float]], list[float]]:
    points: list[list[float]] = []
    targets: list[float] = []
    for x in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        for y in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            points.append([x, y])
            inside = math.hypot(x, y) <= 1.5
            targets.append(8.0 if inside else -8.0)
    return points, targets


def fit_and_report(
    name: str,
    data: tuple[list[list[float]], list[float]],
) -> None:
    x, y = data
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        split_policy="structured",
    )
    schema = FeatureSchema.from_specs(
        [SpatialPairSpec("x", "y"), NumericSpec("y")]
    )
    model.fit(x, y, feature_schema=schema)
    pred = model.predict(x)
    mae = sum(abs(actual - prediction) for actual, prediction in zip(y, pred, strict=True)) / len(y)
    print(f"{name}: mae={mae:.6f}")


def main() -> None:
    fit_and_report("diagonal_2d", diagonal_dataset())
    fit_and_report("gaussian_2d", radial_dataset())


if __name__ == "__main__":
    main()
