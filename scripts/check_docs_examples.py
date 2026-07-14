#!/usr/bin/env python3
"""Run executable examples for the maintained geo-system docs."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
for path in (ROOT, PYTHON):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DOC_REFERENCES = {
    "model_choice": ROOT / "docs" / "user-guide" / "model-types.md",
    "geo_evaluation": ROOT / "docs" / "user-guide" / "geo-evaluation-standard.md",
    "probabilistic_conformal": ROOT
    / "docs"
    / "user-guide"
    / "forecasting-models"
    / "probabilistic-conformal.md",
}


def main() -> int:
    checks = {
        "docs_reference_contract": check_docs_reference_contract(),
        "model_choice": run_model_choice_example(),
        "geo_evaluation": run_geo_evaluation_example(),
        "probabilistic_conformal": run_probabilistic_conformal_example(),
    }
    failed = [name for name, result in checks.items() if not result["passed"]]
    report = {"artifact_version": 1, "checks": checks, "passed": not failed}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit("docs example checks failed: " + ", ".join(failed))
    return 0


def check_docs_reference_contract() -> dict[str, Any]:
    required = {
        "model_choice": [
            "ModelRegistry.defaults()",
            "Automatic geo model selection is intentionally not shipped",
            "scripts/check_docs_examples.py",
        ],
        "geo_evaluation": [
            "spatial_block_cv_manifest",
            "CoordinateMatrix",
            "scripts/check_docs_examples.py",
        ],
        "probabilistic_conformal": [
            "ConformalIntervalRegressor",
            "SpatialConformalRegressor",
            "interval_coverage",
            "scripts/check_docs_examples.py",
        ],
    }
    missing: dict[str, list[str]] = {}
    for name, markers in required.items():
        text = DOC_REFERENCES[name].read_text(encoding="utf-8")
        absent = [marker for marker in markers if marker not in text]
        if absent:
            missing[name] = absent
    return {"passed": not missing, "missing": missing}


def run_model_choice_example() -> dict[str, Any]:
    from cartoboost import CartoBoostRegressor
    from cartoboost.models import ModelRegistry, model_card

    rng = np.random.default_rng(42)
    coords = rng.uniform(0.0, 1.0, size=(48, 2))
    X = np.column_stack(
        [
            coords[:, 0],
            coords[:, 1],
            np.sin(coords[:, 0] * np.pi),
        ]
    )
    y = 2.0 * coords[:, 0] - 0.4 * coords[:, 1] + 0.15 * rng.normal(size=48)
    registry = ModelRegistry.defaults()
    names = registry.names(namespace="geo")
    model = CartoBoostRegressor(n_estimators=12, max_depth=2, random_state=7)
    model.fit(X[:36], y[:36])
    pred = model.predict(X[36:])
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "cartoboost-regressor.json"
        model.save(artifact)
        loaded = CartoBoostRegressor.load(artifact)
        loaded_pred = loaded.predict(X[36:])
    card = model_card(model)
    return {
        "passed": bool(names) and np.allclose(pred, loaded_pred),
        "geo_registry_entries": names,
        "selected_family": "models.cartoboost_regressor",
        "prediction_rows": int(pred.shape[0]),
        "model_card_keys": sorted(card),
    }


def run_geo_evaluation_example() -> dict[str, Any]:
    from cartoboost.geo import CoordinateMatrix, spatial_block_cv_manifest

    coords = CoordinateMatrix(
        x=[0.0, 0.2, 0.8, 1.0, 2.0, 2.1, 3.0, 3.2],
        y=[0.0, 0.1, 0.8, 1.1, 1.8, 2.2, 3.1, 3.3],
        crs="EPSG:2263",
    )
    manifest = spatial_block_cv_manifest(
        coords,
        n_folds=4,
        dataset_fingerprint="sha256:" + "a" * 64,
        coordinate_crs_note="Synthetic taxi-zone centroids projected to EPSG:2263.",
        model_version="docs-example",
        dependency_versions={"cartoboost": "docs-example"},
        random_seed=42,
        split_id="docs_spatial_block_cv",
    )
    folds = list(manifest.folds())
    fold_sizes = [(len(train), len(test)) for _, train, test in folds]
    return {
        "passed": len(folds) == 4 and all(train and test for train, test in fold_sizes),
        "manifest_hash": manifest.hash(),
        "fold_sizes": fold_sizes,
    }


def run_probabilistic_conformal_example() -> dict[str, Any]:
    from cartoboost import CartoBoostRegressor
    from cartoboost.forecasting import (
        ConformalIntervalRegressor,
        SpatialConformalRegressor,
        interval_coverage,
        mean_interval_width,
    )

    x_train = np.arange(24, dtype=float).reshape(-1, 1)
    y_train = 0.5 * x_train[:, 0] + np.sin(x_train[:, 0] / 3.0)
    x_cal = np.arange(24, 36, dtype=float).reshape(-1, 1)
    y_cal = 0.5 * x_cal[:, 0] + np.sin(x_cal[:, 0] / 3.0)
    x_holdout = np.arange(36, 42, dtype=float).reshape(-1, 1)
    y_holdout = 0.5 * x_holdout[:, 0] + np.sin(x_holdout[:, 0] / 3.0)

    base = CartoBoostRegressor(n_estimators=24, learning_rate=0.08, max_depth=2)
    model = ConformalIntervalRegressor(base, alpha=0.2)
    model.fit(
        x_train,
        y_train,
        x_cal,
        y_cal,
        train_end_exclusive=24,
        calibration_start=24,
        calibration_end_exclusive=36,
        test_start=36,
    )
    interval = model.predict_interval(x_holdout, test_start=36)

    spatial = SpatialConformalRegressor(
        CartoBoostRegressor(n_estimators=24, learning_rate=0.08, max_depth=2),
        alpha=0.2,
    )
    groups = np.array(["pickup_core" if i % 2 == 0 else "pickup_edge" for i in range(len(x_cal))])
    spatial.fit(
        x_train,
        y_train,
        x_cal,
        y_cal,
        groups=groups,
        train_end_exclusive=24,
        calibration_start=24,
        calibration_end_exclusive=36,
        test_start=36,
    )
    spatial_interval = spatial.predict_interval(
        x_holdout,
        test_start=36,
        groups=["pickup_core", "pickup_edge", "pickup_core", "pickup_edge", "new_zone", "new_zone"],
    )
    coverage = interval_coverage(y_holdout, interval.lower, interval.upper)
    width = mean_interval_width(interval.lower, interval.upper)
    return {
        "passed": np.isfinite(coverage)
        and 0.0 <= coverage <= 1.0
        and width > 0.0
        and spatial_interval.lower.shape == interval.lower.shape,
        "coverage": float(coverage),
        "mean_interval_width": float(width),
        "spatial_method": spatial_interval.metadata["method"],
        "spatial_group_count": len(spatial.group_residual_quantiles_),
    }


if __name__ == "__main__":
    raise SystemExit(main())
