#!/usr/bin/env python3
"""Compare CartoBoost against small sklearn baselines on deterministic fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = ROOT / "python"
if str(PYTHON_SOURCE) not in sys.path:
    sys.path.insert(0, str(PYTHON_SOURCE))

from cartoboost import CartoBoostRegressor  # noqa: E402

DEFAULT_OUTPUT = ROOT / "target" / "validation" / "baseline_comparison.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    train_x = rng.uniform(-2.0, 2.0, size=(256, 3))
    test_x = rng.uniform(-2.0, 2.0, size=(128, 3))

    def target(x: np.ndarray) -> np.ndarray:
        return 2.0 * x[:, 0] - 0.75 * x[:, 1] + np.where(x[:, 2] > 0.0, 1.5, -1.5)

    return train_x, target(train_x), test_x, target(test_x)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(mean_squared_error(actual, predicted) ** 0.5)


def collect() -> dict[str, Any]:
    train_x, train_y, test_x, test_y = fixture()
    cartoboost = CartoBoostRegressor(
        n_estimators=25,
        learning_rate=0.1,
        max_depth=2,
        min_samples_leaf=4,
        min_gain=0.0,
        split_policy="axis_only",
    ).fit(train_x, train_y)
    sklearn_gbr = GradientBoostingRegressor(
        n_estimators=25,
        learning_rate=0.1,
        max_depth=2,
        min_samples_leaf=4,
        random_state=0,
    ).fit(train_x, train_y)

    return {
        "artifact_version": 1,
        "fixture": "deterministic_axis_regression",
        "models": {
            "cartoboost": {
                "test_rmse": rmse(test_y, cartoboost.predict(test_x)),
                "backend": cartoboost._backend_used,
            },
            "sklearn_gradient_boosting": {
                "test_rmse": rmse(test_y, sklearn_gbr.predict(test_x)),
            },
        },
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
