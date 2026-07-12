from __future__ import annotations

import json
import math

import cartoboost.metrics as metrics_module
import numpy as np
import pytest
from cartoboost import _native
from cartoboost.metrics.wrmsse import aggregate_equal_level_wrmsse, rmsse_scale, wrmsse


def test_wrmsse_matches_manual_m5_style_reference() -> None:
    y_train = np.array(
        [
            [10.0, 12.0, 14.0, 16.0],
            [20.0, 21.0, 23.0, 26.0],
        ]
    )
    y_true = np.array(
        [
            [18.0, 20.0],
            [30.0, 34.0],
        ]
    )
    y_pred = np.array(
        [
            [17.0, 23.0],
            [32.0, 31.0],
        ]
    )
    weights = np.array([2.0, 1.0])

    result = wrmsse(
        y_train,
        y_true,
        y_pred,
        weights,
        seasonal_period=1,
        series_ids=["PULocationID=1", "PULocationID=2"],
        return_breakdown=True,
    )

    first_rmsse = math.sqrt((1.0 + 9.0) / 2.0 / 4.0)
    second_rmsse = math.sqrt((4.0 + 9.0) / 2.0 / (14.0 / 3.0))
    expected = (2.0 / 3.0) * first_rmsse + (1.0 / 3.0) * second_rmsse
    assert result["wrmsse"] == pytest.approx(expected)
    assert result["wrmsse"] == pytest.approx(1.1387538887346516)
    assert result["series"][0]["series_id"] == "PULocationID=1"
    assert result["series"][0]["scale"] == pytest.approx(4.0)
    assert result["series"][1]["scale"] == pytest.approx(14.0 / 3.0)


def test_wrmsse_returns_scalar_by_default() -> None:
    score = wrmsse(
        [[1.0, 2.0, 3.0], [3.0, 5.0, 7.0]],
        [[4.0], [9.0]],
        [[5.0], [8.0]],
        [1.0, 3.0],
    )

    assert isinstance(score, float)
    assert score > 0.0


def test_aggregate_equal_level_wrmsse_delegates_to_native(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeNative:
        def aggregate_equal_level_wrmsse_value(self, level_scores):
            calls["level_scores"] = level_scores
            return json.dumps(
                {
                    "wrmsse": 0.9,
                    "levels": [
                        {
                            "level": "total",
                            "wrmsse": 0.6,
                            "level_weight": 0.5,
                            "contribution": 0.3,
                        },
                        {
                            "level": "item_store",
                            "wrmsse": 1.2,
                            "level_weight": 0.5,
                            "contribution": 0.6,
                        },
                    ],
                }
            )

    monkeypatch.setattr(metrics_module, "_native", FakeNative())

    result = aggregate_equal_level_wrmsse(
        [{"level": "total", "wrmsse": 0.6}, ("item_store", 1.2)],
        return_breakdown=True,
    )

    assert result["wrmsse"] == pytest.approx(0.9)
    assert result["levels"][0]["contribution"] == pytest.approx(0.3)
    assert calls["level_scores"] == [("total", 0.6), ("item_store", 1.2)]


def test_ordered_nonnegative_weights_value_uses_native_nonnegative_fallback() -> None:
    weights = _native.ordered_nonnegative_weights_value(
        ["a", "b", "c"],
        [("a", -2.0), ("b", 4.0)],
    )

    assert weights == {"a": 0.0, "b": 4.0, "c": 0.0}
    assert _native.ordered_nonnegative_weights_value(["a", "b"], [("a", -1.0)]) == {
        "a": 1.0,
        "b": 1.0,
    }


def test_wrmsse_delegates_scoring_to_native(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeNative:
        def wrmsse_value(self, series, seasonal_period):
            calls["series"] = series
            calls["seasonal_period"] = seasonal_period
            return json.dumps(
                {
                    "wrmsse": 0.25,
                    "series": [
                        {
                            "series_id": "lane-a",
                            "weight": 2.0,
                            "normalized_weight": 1.0,
                            "scale": 4.0,
                            "rmsse": 0.25,
                            "contribution": 0.25,
                        }
                    ],
                }
            )

        def rmsse_scale_value(self, train, seasonal_period):
            calls["scale"] = (train, seasonal_period)
            return 4.0

    monkeypatch.setattr(metrics_module, "_native", FakeNative())

    result = wrmsse(
        [[1.0, 3.0, 5.0]],
        [[7.0]],
        [[8.0]],
        [2.0],
        seasonal_period=2,
        series_ids=["lane-a"],
        return_breakdown=True,
    )

    assert result["wrmsse"] == pytest.approx(0.25)
    assert calls["series"] == [("lane-a", [1.0, 3.0, 5.0], [7.0], [8.0], 2.0)]
    assert calls["seasonal_period"] == 2
    assert rmsse_scale([1.0, 3.0, 5.0], seasonal_period=2) == pytest.approx(4.0)
    assert calls["scale"] == ([1.0, 3.0, 5.0], 2)


def test_rmsse_scale_rejects_constant_or_short_history() -> None:
    with pytest.raises(ValueError, match="longer than seasonal_period"):
        rmsse_scale([1.0], seasonal_period=1)

    with pytest.raises(ValueError, match="scale is zero"):
        rmsse_scale([2.0, 2.0, 2.0], seasonal_period=1)


def test_wrmsse_rejects_invalid_shapes_and_weights() -> None:
    with pytest.raises(ValueError, match="same shape"):
        wrmsse([[1.0, 2.0]], [[3.0, 4.0]], [[3.0]], [1.0])

    with pytest.raises(ValueError, match="same number of series"):
        wrmsse([[1.0, 2.0]], [[3.0], [4.0]], [[3.0], [4.0]], [1.0, 1.0])

    with pytest.raises(ValueError, match="non-negative"):
        wrmsse([[1.0, 2.0]], [[3.0]], [[4.0]], [-1.0])

    with pytest.raises(ValueError, match="positive value"):
        wrmsse([[1.0, 2.0]], [[3.0]], [[4.0]], [0.0])
