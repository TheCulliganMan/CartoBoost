"""Five-minute NumPy-only CartoBoost workflow.

This example intentionally uses only the base package dependency set.  It
creates a small taxi-shaped duration problem, performs a native out-of-time split,
compares an axis baseline with the schema-driven structured policy, and verifies artifact
round-trip predictions.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from cartoboost import CartoBoostRegressor
from cartoboost.config import SplitPolicy
from cartoboost.geo import PanelIndex, TimeIndex
from cartoboost.schema import FeatureSchema, NumericSpec, PeriodicSpec, SpatialPairSpec
from cartoboost.validation import native_out_of_time_split


def main() -> None:
    rng = np.random.default_rng(7)
    rows = 240
    hour = np.arange(rows, dtype=float) % 24.0
    pickup_x = rng.uniform(-1.0, 1.0, rows)
    pickup_y = rng.uniform(-1.0, 1.0, rows)
    distance = rng.uniform(1.0, 8.0, rows)
    features = np.column_stack([distance, hour, pickup_x, pickup_y])
    target = (
        8.0
        + 1.8 * distance
        + 2.0 * np.sin(2.0 * np.pi * hour / 24.0)
        + 0.8 * pickup_x
        - 0.6 * pickup_y
        + rng.normal(0.0, 0.15, rows)
    )

    timestamps = [
        (datetime(2026, 1, 1) + timedelta(hours=index)).isoformat() for index in range(rows)
    ]
    panel = PanelIndex(["nyc_taxi"] * rows, time=TimeIndex(timestamps, frequency="h"))
    holdout = 48
    split = native_out_of_time_split(
        panel,
        min_train_size=rows - holdout,
        horizon=holdout,
        step=holdout,
        dataset_fingerprint="sha256:quickstart",
        coordinate_crs_note="not_applicable",
        model_version="0.3.0",
        dependency_versions={"cartoboost": "0.3.0"},
    )
    _, train_idx, validation_idx = split.folds()[0]
    schema = FeatureSchema.from_specs(
        [
            NumericSpec("distance"),
            PeriodicSpec("hour", 24),
            SpatialPairSpec("pickup_x", "pickup_y"),
            NumericSpec("pickup_y"),
        ]
    )
    baseline = CartoBoostRegressor(
        n_estimators=40,
        learning_rate=0.06,
        max_depth=4,
        min_samples_leaf=8,
        split_policy=SplitPolicy.AXIS_ONLY,
        random_state=7,
    )
    structured = CartoBoostRegressor(
        n_estimators=40,
        learning_rate=0.06,
        max_depth=4,
        min_samples_leaf=8,
        split_policy=SplitPolicy.STRUCTURED,
        random_state=7,
    )
    baseline.fit(features[train_idx], target[train_idx], feature_schema=schema)
    structured.fit(features[train_idx], target[train_idx], feature_schema=schema)
    baseline_predictions = baseline.predict(features[validation_idx])
    structured_predictions = structured.predict(features[validation_idx])

    with TemporaryDirectory() as directory:
        artifact = Path(directory) / "structured.cartoboost.json"
        structured.save(artifact)
        restored = CartoBoostRegressor.load(artifact)
        restored_predictions = restored.predict(features[validation_idx])

    payload = {
        "rows": rows,
        "train_rows": len(train_idx),
        "validation_rows": len(validation_idx),
        "baseline_rmse": float(
            np.sqrt(np.mean((baseline_predictions - target[validation_idx]) ** 2))
        ),
        "structured_rmse": float(
            np.sqrt(np.mean((structured_predictions - target[validation_idx]) ** 2))
        ),
        "roundtrip_max_abs_diff": float(
            np.max(np.abs(structured_predictions - restored_predictions))
        ),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
