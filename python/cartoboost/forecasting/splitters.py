"""Leakage-safe rolling-origin splitters for forecasting workflows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class ForecastFold:
    """One deterministic forecasting fold."""

    fold_id: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_start: Any
    train_end: Any
    validation_start: Any
    validation_end: Any
    horizon: int
    step: int
    metadata: dict[str, Any]


class RollingOriginSplitter:
    """Rolling-origin splitter with expanding or sliding train windows."""

    def __init__(
        self,
        *,
        horizon: int,
        step: int = 1,
        min_train_size: int = 1,
        max_train_size: int | None = None,
        n_splits: int | None = None,
        timestamp_col: str | None = None,
        series_id_col: str | None = None,
        window: Literal["expanding", "sliding"] = "expanding",
    ) -> None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if step <= 0:
            raise ValueError("step must be positive")
        if min_train_size <= 0:
            raise ValueError("min_train_size must be positive")
        if max_train_size is not None and max_train_size < min_train_size:
            raise ValueError("max_train_size must be greater than or equal to min_train_size")
        if n_splits is not None and n_splits <= 0:
            raise ValueError("n_splits must be positive when provided")
        if window not in {"expanding", "sliding"}:
            raise ValueError("window must be 'expanding' or 'sliding'")
        if window == "sliding" and max_train_size is None:
            raise ValueError("sliding windows require max_train_size")

        self.horizon = int(horizon)
        self.step = int(step)
        self.min_train_size = int(min_train_size)
        self.max_train_size = None if max_train_size is None else int(max_train_size)
        self.n_splits = None if n_splits is None else int(n_splits)
        self.timestamp_col = timestamp_col
        self.series_id_col = series_id_col
        self.window = window

    def split(self, data: Any) -> Iterator[ForecastFold]:
        timestamps = _timestamps(data, self.timestamp_col)
        series_ids = _series_ids(data, self.series_id_col, len(timestamps))
        order = np.lexsort((np.arange(len(timestamps)), timestamps))
        sorted_times = timestamps[order]
        unique_times = np.unique(sorted_times)
        if unique_times.size <= self.horizon:
            return

        cutoffs = range(
            self.min_train_size - 1,
            unique_times.size - self.horizon,
            self.step,
        )
        if self.n_splits is not None:
            cutoffs = cutoffs[-self.n_splits :]
        candidates: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
        for cutoff_pos in cutoffs:
            train_times = unique_times[: cutoff_pos + 1]
            if self.max_train_size is not None:
                train_times = train_times[-self.max_train_size :]
            validation_times = unique_times[cutoff_pos + 1 : cutoff_pos + 1 + self.horizon]

            train_mask = np.isin(timestamps, train_times)
            validation_mask = np.isin(timestamps, validation_times)
            train_indices = np.flatnonzero(train_mask)
            validation_indices = np.flatnonzero(validation_mask)
            if train_indices.size < self.min_train_size:
                continue
            if validation_indices.size == 0:
                continue

            train_max = np.max(timestamps[train_indices])
            validation_min = np.min(timestamps[validation_indices])
            if not train_max < validation_min:
                raise ValueError("leakage detected: max(train timestamp) must be < min(validation)")

            candidates.append(
                (
                    train_indices,
                    validation_indices,
                    {
                        "series_id_col": self.series_id_col,
                        "series_count": int(np.unique(series_ids[validation_indices]).size),
                        "timestamp_col": self.timestamp_col,
                        "train_size": int(train_indices.size),
                        "validation_size": int(validation_indices.size),
                        "train_timestamp_count": int(train_times.size),
                        "validation_timestamp_count": int(validation_times.size),
                        "origin_timestamp": _scalar(unique_times[cutoff_pos]),
                    },
                )
            )

        for i, (train_indices, validation_indices, metadata) in enumerate(candidates):
            yield ForecastFold(
                fold_id=f"fold_{i:04d}",
                train_indices=train_indices,
                validation_indices=validation_indices,
                train_start=_scalar(np.min(timestamps[train_indices])),
                train_end=_scalar(np.max(timestamps[train_indices])),
                validation_start=_scalar(np.min(timestamps[validation_indices])),
                validation_end=_scalar(np.max(timestamps[validation_indices])),
                horizon=self.horizon,
                step=self.step,
                metadata=metadata,
            )

    def get_n_splits(self, data: Any) -> int:
        return sum(1 for _ in self.split(data))


class ExpandingWindowSplitter(RollingOriginSplitter):
    """Rolling-origin splitter whose train window grows over time."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(window="expanding", **kwargs)


class SlidingWindowSplitter(RollingOriginSplitter):
    """Rolling-origin splitter with a fixed-size rolling train window."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(window="sliding", **kwargs)


def _timestamps(data: Any, timestamp_col: str | None) -> np.ndarray:
    if timestamp_col is not None:
        values = _column(data, timestamp_col)
    else:
        values = getattr(data, "index", None)
        if values is None:
            values = np.arange(len(data))
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError("timestamps must be one-dimensional")
    if arr.size == 0:
        raise ValueError("data must contain at least one row")
    return arr


def _series_ids(data: Any, series_id_col: str | None, size: int) -> np.ndarray:
    if series_id_col is None:
        return np.zeros(size, dtype=object)
    arr = np.asarray(_column(data, series_id_col), dtype=object)
    if arr.shape != (size,):
        raise ValueError("series_id_col must be one-dimensional and match data length")
    return arr


def _column(data: Any, name: str) -> Any:
    try:
        return data[name]
    except Exception as exc:
        raise ValueError(f"data must contain column {name!r}") from exc


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value
