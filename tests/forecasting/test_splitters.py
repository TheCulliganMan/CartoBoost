from __future__ import annotations

import pandas as pd
import pytest
from cartoboost.forecasting import ExpandingWindowSplitter, SlidingWindowSplitter


def taxi_panel() -> pd.DataFrame:
    rows = []
    for zone in ["pickup_1", "pickup_2"]:
        for hour in range(8):
            rows.append(
                {
                    "series_id": zone,
                    "timestamp": hour,
                    "fare": float(hour),
                    "trip_distance": 1.0 + hour,
                }
            )
    return pd.DataFrame(rows)


def test_expanding_window_folds_are_leakage_safe_and_deterministic() -> None:
    data = taxi_panel()
    splitter = ExpandingWindowSplitter(
        horizon=2,
        step=2,
        min_train_size=3,
        timestamp_col="timestamp",
        series_id_col="series_id",
    )

    folds = list(splitter.split(data))

    assert [fold.fold_id for fold in folds] == ["fold_0000", "fold_0001"]
    for fold in folds:
        assert (
            data.loc[fold.train_indices, "timestamp"].max()
            < data.loc[fold.validation_indices, "timestamp"].min()
        )
        assert fold.metadata["series_count"] == 2
        assert fold.metadata["validation_timestamp_count"] == 2


def test_sliding_window_respects_max_train_size() -> None:
    splitter = SlidingWindowSplitter(
        horizon=1,
        step=1,
        min_train_size=2,
        max_train_size=3,
        timestamp_col="timestamp",
        series_id_col="series_id",
    )

    folds = list(splitter.split(taxi_panel()))

    assert folds[-1].metadata["train_timestamp_count"] == 3
    assert folds[-1].train_start == 4
    assert folds[-1].train_end == 6


def test_no_random_cv_or_overlapping_validation_boundary() -> None:
    data = taxi_panel()
    data.loc[0, "timestamp"] = 4
    splitter = ExpandingWindowSplitter(
        horizon=1,
        min_train_size=3,
        timestamp_col="timestamp",
        series_id_col="series_id",
    )

    folds = list(splitter.split(data))

    assert folds
    for fold in folds:
        assert (
            data.loc[fold.train_indices, "timestamp"].max()
            < data.loc[fold.validation_indices, "timestamp"].min()
        )


def test_splitters_preserve_strict_boundary_with_unsorted_panel_rows() -> None:
    data = taxi_panel().sample(frac=1.0, random_state=7).reset_index(drop=True)
    splitter = ExpandingWindowSplitter(
        horizon=2,
        step=1,
        min_train_size=2,
        timestamp_col="timestamp",
        series_id_col="series_id",
    )

    folds = list(splitter.split(data))

    assert folds
    for fold in folds:
        train_max = data.loc[fold.train_indices, "timestamp"].max()
        validation_min = data.loc[fold.validation_indices, "timestamp"].min()
        assert train_max < validation_min


def test_sliding_requires_max_train_size() -> None:
    with pytest.raises(ValueError, match="max_train_size"):
        SlidingWindowSplitter(horizon=1, min_train_size=2, timestamp_col="timestamp")
