from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .frequency import (
    infer_frequency,
    normalize_frequency,
    require_pandas,
    validate_regular_frequency,
)


def _as_list(values: Sequence[str] | None, *, name: str) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of column names, not a string")
    result = [str(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate column names")
    return result


@dataclass(frozen=True)
class ForecastFrame:
    """Validated time-series data for CartoBoost forecasting APIs."""

    data: Any
    timestamp_col: str
    target_col: str
    series_id_col: str | None
    freq: str | None
    static_covariates: tuple[str, ...] = ()
    known_future_covariates: tuple[str, ...] = ()
    historical_covariates: tuple[str, ...] = ()
    allow_irregular: bool = False
    allow_missing_targets: bool = False
    allow_missing_covariates: bool = False
    sample_weight_col: str | None = None
    _native_frame: Any | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_rows == 0:
            raise ValueError("forecast data must contain at least one row")
        if self.series_id_col is not None and self.data[self.series_id_col].isna().any():
            raise ValueError(
                f"series id column {self.series_id_col!r} must not contain null values"
            )

    @classmethod
    def from_pandas(
        cls,
        frame: Any,
        *,
        timestamp_col: str,
        target_col: str,
        series_id_col: str | None = None,
        freq: str | None = None,
        static_covariates: Sequence[str] | None = None,
        known_future_covariates: Sequence[str] | None = None,
        historical_covariates: Sequence[str] | None = None,
        allow_irregular: bool = False,
        allow_missing_targets: bool = False,
        allow_missing_covariates: bool = False,
        sample_weight_col: str | None = None,
    ) -> ForecastFrame:
        pd = require_pandas()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("ForecastFrame.from_pandas requires a pandas DataFrame")
        static = _as_list(static_covariates, name="static_covariates")
        known_future = _as_list(known_future_covariates, name="known_future_covariates")
        historical = _as_list(historical_covariates, name="historical_covariates")
        if sample_weight_col is not None:
            sample_weight_col = str(sample_weight_col)
            if sample_weight_col in {timestamp_col, target_col, series_id_col}:
                raise ValueError("sample_weight_col cannot be a reserved forecast column")
            if sample_weight_col in static or sample_weight_col in known_future:
                raise ValueError("sample_weight_col must not be a static or known-future covariate")
            if sample_weight_col not in historical:
                historical.append(sample_weight_col)
        required = [timestamp_col, target_col, *static, *known_future, *historical]
        if series_id_col is not None:
            required.append(series_id_col)
        _validate_columns(frame, required)
        _validate_covariate_metadata(
            timestamp_col=timestamp_col,
            target_col=target_col,
            series_id_col=series_id_col,
            static_covariates=static,
            known_future_covariates=known_future,
            historical_covariates=historical,
        )

        data = frame.copy()
        try:
            data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"timestamp column {timestamp_col!r} contains unparseable values"
            ) from exc
        if data[timestamp_col].isna().any():
            raise ValueError(f"timestamp column {timestamp_col!r} must not contain null values")

        try:
            target_values = data[target_col].to_numpy(dtype=float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"target column {target_col!r} must contain numeric values") from exc
        if np.isinf(target_values).any() or (
            not allow_missing_targets and np.isnan(target_values).any()
        ):
            raise ValueError(
                f"target column {target_col!r} must contain only finite values "
                "unless allow_missing_targets=True"
            )
        for covariate_col in [*static, *known_future, *historical]:
            try:
                covariate_values = data[covariate_col].to_numpy(dtype=float, copy=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"forecast covariate column {covariate_col!r} must contain numeric values"
                ) from exc
            if np.isinf(covariate_values).any() or (
                not allow_missing_covariates and np.isnan(covariate_values).any()
            ):
                raise ValueError(
                    f"forecast covariate column {covariate_col!r} must contain only finite values "
                    "unless allow_missing_covariates=True"
                )
        if sample_weight_col is not None:
            sample_weights = data[sample_weight_col].to_numpy(dtype=float, copy=False)
            if not np.isfinite(sample_weights).all() or (sample_weights <= 0.0).any():
                raise ValueError(
                    f"sample weight column {sample_weight_col!r} must contain "
                    "positive finite values"
                )

        sort_cols = [timestamp_col] if series_id_col is None else [series_id_col, timestamp_col]
        data = data.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        duplicate_cols = (
            [timestamp_col] if series_id_col is None else [series_id_col, timestamp_col]
        )
        native_source_data = data
        if sample_weight_col is not None:
            data = _aggregate_duplicate_timestamps(
                data,
                group_cols=duplicate_cols,
                target_col=target_col,
                value_cols=[*static, *known_future, *historical],
                sample_weight_col=sample_weight_col,
            )
        duplicates = data.duplicated(subset=duplicate_cols, keep=False)
        if duplicates.any():
            raise ValueError("forecast data contains duplicate timestamp rows within a series")

        normalized_freq = normalize_frequency(freq)
        if series_id_col is None:
            normalized_freq = _resolve_frequency(
                data[timestamp_col],
                normalized_freq,
                allow_irregular=allow_irregular,
                label="series",
            )
        else:
            normalized_freq = _resolve_panel_frequency(
                data,
                timestamp_col=timestamp_col,
                series_id_col=series_id_col,
                freq=normalized_freq,
                allow_irregular=allow_irregular,
            )

        result = cls(
            data=data,
            timestamp_col=timestamp_col,
            target_col=target_col,
            series_id_col=series_id_col,
            freq=normalized_freq,
            static_covariates=tuple(static),
            known_future_covariates=tuple(known_future),
            historical_covariates=tuple(historical),
            allow_irregular=allow_irregular,
            allow_missing_targets=allow_missing_targets,
            allow_missing_covariates=allow_missing_covariates,
            sample_weight_col=sample_weight_col,
        )
        native_frame = _build_native_frame(result, data=native_source_data)
        return cls(
            data=result.data,
            timestamp_col=result.timestamp_col,
            target_col=result.target_col,
            series_id_col=result.series_id_col,
            freq=result.freq,
            static_covariates=result.static_covariates,
            known_future_covariates=result.known_future_covariates,
            historical_covariates=result.historical_covariates,
            allow_irregular=result.allow_irregular,
            allow_missing_targets=result.allow_missing_targets,
            allow_missing_covariates=result.allow_missing_covariates,
            sample_weight_col=result.sample_weight_col,
            _native_frame=native_frame,
        )

    @property
    def is_panel(self) -> bool:
        return self.series_id_col is not None

    @property
    def n_rows(self) -> int:
        return int(len(self.data))

    @property
    def series_ids(self) -> list[Any]:
        if self.series_id_col is None:
            return []
        return list(self.data[self.series_id_col].drop_duplicates())

    def to_pandas(self) -> Any:
        return self.data.copy()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "timestamp_col": self.timestamp_col,
            "target_col": self.target_col,
            "series_id_col": self.series_id_col,
            "freq": self.freq,
            "is_panel": self.is_panel,
            "n_rows": self.n_rows,
            "series_ids": [str(value) for value in self.series_ids],
            "static_covariates": list(self.static_covariates),
            "known_future_covariates": list(self.known_future_covariates),
            "historical_covariates": list(self.historical_covariates),
            "allow_irregular": self.allow_irregular,
            "allow_missing_targets": self.allow_missing_targets,
            "allow_missing_covariates": self.allow_missing_covariates,
            "sample_weight_col": self.sample_weight_col,
        }


@dataclass(frozen=True)
class PredictionInterval:
    level: float
    lower: Sequence[float]
    upper: Sequence[float]

    def __post_init__(self) -> None:
        level = float(self.level)
        if not math.isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("prediction interval level must be between 0 and 1")
        if len(self.lower) != len(self.upper):
            raise ValueError("prediction interval lower and upper lengths must match")
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("prediction interval bounds must be finite")
        if (lower > upper).any():
            raise ValueError("prediction interval lower bounds must not exceed upper bounds")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "lower", tuple(float(value) for value in lower))
        object.__setattr__(self, "upper", tuple(float(value) for value in upper))

    @property
    def suffix(self) -> str:
        percentage = self.level * 100.0
        if percentage.is_integer():
            return str(int(percentage))
        return f"{percentage:g}".replace(".", "_")


@dataclass(frozen=True)
class ForecastResult:
    """Deterministic forecast output table with JSON roundtrip support."""

    data: Any
    timestamp_col: str = "timestamp"
    prediction_col: str = "prediction"
    series_id_col: str | None = None

    def __post_init__(self) -> None:
        validated = _validate_result_data(
            self.data,
            timestamp_col=self.timestamp_col,
            prediction_col=self.prediction_col,
            series_id_col=self.series_id_col,
        )
        object.__setattr__(self, "data", validated)

    @classmethod
    def from_predictions(
        cls,
        *,
        timestamps: Sequence[Any],
        predictions: Sequence[float],
        series_id: Any | Sequence[Any] | None = None,
        intervals: Sequence[PredictionInterval] | None = None,
        timestamp_col: str = "timestamp",
        prediction_col: str = "prediction",
        series_id_col: str = "series_id",
    ) -> ForecastResult:
        pd = require_pandas()
        if len(timestamps) != len(predictions):
            raise ValueError("timestamps and predictions lengths must match")
        rows = len(predictions)
        if rows == 0:
            raise ValueError("predictions must contain at least one row")
        prediction_values = np.asarray(predictions, dtype=float)
        if not np.isfinite(prediction_values).all():
            raise ValueError("predictions must be finite")
        payload: dict[str, Any] = {
            timestamp_col: pd.to_datetime(list(timestamps), errors="raise"),
            prediction_col: prediction_values,
        }
        result_series_col = None
        if series_id is not None:
            result_series_col = series_id_col
            if isinstance(series_id, str) or not isinstance(series_id, Iterable):
                if pd.isna(series_id):
                    raise ValueError("series_id values must not contain null values")
                payload[series_id_col] = [series_id] * rows
            else:
                series_values = list(series_id)
                if len(series_values) != rows:
                    raise ValueError("series_id length must match predictions")
                if pd.Series(series_values).isna().any():
                    raise ValueError("series_id values must not contain null values")
                payload[series_id_col] = series_values
        interval_levels: set[float] = set()
        for interval in sorted(intervals or [], key=lambda item: item.level):
            if interval.level in interval_levels:
                raise ValueError("prediction interval levels must be unique")
            interval_levels.add(interval.level)
            if len(interval.lower) != rows:
                raise ValueError("prediction interval length must match predictions")
            payload[f"prediction_lower_{interval.suffix}"] = np.asarray(interval.lower, dtype=float)
            payload[f"prediction_upper_{interval.suffix}"] = np.asarray(interval.upper, dtype=float)

        columns = [timestamp_col, prediction_col]
        if result_series_col is not None:
            columns = [result_series_col, *columns]
        for interval in sorted(intervals or [], key=lambda item: item.level):
            columns.extend(
                [f"prediction_lower_{interval.suffix}", f"prediction_upper_{interval.suffix}"]
            )
        data = pd.DataFrame(payload, columns=columns)
        sort_cols = (
            [timestamp_col] if result_series_col is None else [result_series_col, timestamp_col]
        )
        data = data.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        return cls(
            data=data,
            timestamp_col=timestamp_col,
            prediction_col=prediction_col,
            series_id_col=result_series_col,
        )

    def to_pandas(self) -> Any:
        return self.data.copy()

    def to_json(self) -> str:
        records = []
        for row in self.data.to_dict(orient="records"):
            encoded = dict(row)
            encoded[self.timestamp_col] = row[self.timestamp_col].isoformat()
            records.append(encoded)
        payload = {
            "timestamp_col": self.timestamp_col,
            "prediction_col": self.prediction_col,
            "series_id_col": self.series_id_col,
            "columns": list(self.data.columns),
            "records": records,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> ForecastResult:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_json(cls, payload: str) -> ForecastResult:
        pd = require_pandas()
        decoded = json.loads(payload)
        required_keys = {"timestamp_col", "prediction_col", "series_id_col", "columns", "records"}
        missing = sorted(required_keys - set(decoded))
        if missing:
            raise ValueError(f"forecast result JSON is missing required keys: {missing}")
        data = pd.DataFrame(decoded["records"], columns=decoded["columns"])
        timestamp_col = decoded["timestamp_col"]
        return cls(
            data=data,
            timestamp_col=timestamp_col,
            prediction_col=decoded["prediction_col"],
            series_id_col=decoded["series_id_col"],
        )


def _validate_result_data(
    data: Any,
    *,
    timestamp_col: str,
    prediction_col: str,
    series_id_col: str | None,
) -> Any:
    pd = require_pandas()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("ForecastResult data must be a pandas DataFrame")
    required = [timestamp_col, prediction_col]
    if series_id_col is not None:
        required.append(series_id_col)
    _validate_columns(data, required)
    if len(data) == 0:
        raise ValueError("forecast result data must contain at least one row")

    result = data.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors="raise")
    if result[timestamp_col].isna().any():
        raise ValueError(f"timestamp column {timestamp_col!r} must not contain null values")
    try:
        predictions = result[prediction_col].to_numpy(dtype=float, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"prediction column {prediction_col!r} must contain numeric values"
        ) from exc
    if not np.isfinite(predictions).all():
        raise ValueError(f"prediction column {prediction_col!r} must contain only finite values")
    result[prediction_col] = predictions

    if series_id_col is not None and result[series_id_col].isna().any():
        raise ValueError(f"series id column {series_id_col!r} must not contain null values")

    for lower_col in [
        column for column in result.columns if column.startswith("prediction_lower_")
    ]:
        suffix = lower_col.removeprefix("prediction_lower_")
        upper_col = f"prediction_upper_{suffix}"
        if upper_col not in result.columns:
            raise ValueError(f"missing prediction interval upper column {upper_col!r}")
        lower = result[lower_col].to_numpy(dtype=float, copy=False)
        upper = result[upper_col].to_numpy(dtype=float, copy=False)
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("prediction interval columns must contain only finite values")
        if (lower > upper).any():
            raise ValueError("prediction interval lower bounds must not exceed upper bounds")
        result[lower_col] = lower
        result[upper_col] = upper

    for upper_col in [
        column for column in result.columns if column.startswith("prediction_upper_")
    ]:
        suffix = upper_col.removeprefix("prediction_upper_")
        lower_col = f"prediction_lower_{suffix}"
        if lower_col not in result.columns:
            raise ValueError(f"missing prediction interval lower column {lower_col!r}")

    sort_cols = [timestamp_col] if series_id_col is None else [series_id_col, timestamp_col]
    return result.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _validate_columns(frame: Any, required: Sequence[str]) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"forecast data is missing required columns: {missing}")


def _validate_covariate_metadata(
    *,
    timestamp_col: str,
    target_col: str,
    series_id_col: str | None,
    static_covariates: Sequence[str],
    known_future_covariates: Sequence[str],
    historical_covariates: Sequence[str],
) -> None:
    reserved = {timestamp_col, target_col}
    if series_id_col is not None:
        reserved.add(series_id_col)
    for name, values in {
        "static_covariates": static_covariates,
        "known_future_covariates": known_future_covariates,
        "historical_covariates": historical_covariates,
    }.items():
        overlap = sorted(set(values) & reserved)
        if overlap:
            raise ValueError(f"{name} cannot include reserved columns: {overlap}")
    all_covariates = [*static_covariates, *known_future_covariates, *historical_covariates]
    if len(set(all_covariates)) != len(all_covariates):
        raise ValueError("covariate columns must belong to only one forecasting role")


def _aggregate_duplicate_timestamps(
    data: Any,
    *,
    group_cols: Sequence[str],
    target_col: str,
    value_cols: Sequence[str],
    sample_weight_col: str,
) -> Any:
    working = data.copy()
    weight = working[sample_weight_col].to_numpy(dtype=float, copy=False)
    weighted_cols = [target_col, *[col for col in value_cols if col != sample_weight_col]]
    for column in weighted_cols:
        working[f"__cartoboost_weighted_{column}"] = (
            working[column].to_numpy(dtype=float, copy=False) * weight
        )

    aggregations = {sample_weight_col: "sum"}
    for column in weighted_cols:
        aggregations[f"__cartoboost_weighted_{column}"] = "sum"

    grouped = (
        working.groupby(list(group_cols), sort=False, dropna=False).agg(aggregations).reset_index()
    )
    total_weight = grouped[sample_weight_col].to_numpy(dtype=float, copy=False)
    if (total_weight <= 0.0).any():
        raise ValueError(f"sample weight column {sample_weight_col!r} must sum to positive values")
    for column in weighted_cols:
        grouped[column] = grouped[f"__cartoboost_weighted_{column}"] / total_weight
        grouped = grouped.drop(columns=[f"__cartoboost_weighted_{column}"])

    ordered_columns = [column for column in data.columns if column in grouped.columns]
    return (
        grouped[ordered_columns]
        .sort_values(list(group_cols), kind="mergesort")
        .reset_index(drop=True)
    )


def _resolve_frequency(
    timestamps: Any,
    freq: str | None,
    *,
    allow_irregular: bool,
    label: str,
) -> str | None:
    if freq is not None:
        if allow_irregular:
            return freq
        return validate_regular_frequency(timestamps, freq, label=label)
    inferred = infer_frequency(timestamps)
    if inferred is None and not allow_irregular:
        raise ValueError("could not infer a regular frequency; pass freq or allow_irregular=True")
    return inferred


def _resolve_panel_frequency(
    data: Any,
    *,
    timestamp_col: str,
    series_id_col: str,
    freq: str | None,
    allow_irregular: bool,
) -> str | None:
    resolved: set[str] = set()
    for series_id, group in data.groupby(series_id_col, sort=False):
        series_freq = _resolve_frequency(
            group[timestamp_col],
            freq,
            allow_irregular=allow_irregular,
            label=f"series {series_id!r}",
        )
        if series_freq is not None:
            resolved.add(series_freq)
    if freq is not None:
        return freq
    if len(resolved) > 1:
        raise ValueError("panel series must share one inferred frequency")
    if not resolved:
        return None
    return next(iter(resolved))


def _build_native_frame(frame: ForecastFrame, *, data: Any | None = None) -> Any:
    if frame.freq is None:
        return None
    try:
        from cartoboost import _native
    except ImportError as exc:
        raise RuntimeError("cartoboost._native is required for ForecastFrame validation") from exc
    rows = []
    row_covariates = []
    covariate_cols = [
        *frame.static_covariates,
        *frame.known_future_covariates,
        *frame.historical_covariates,
    ]
    data = frame.data if data is None else data
    records = data.to_dict(orient="records")
    sample_weights = None
    if frame.sample_weight_col is not None:
        sample_weights = [float(value) for value in data[frame.sample_weight_col].tolist()]
    for values in records:
        series_id = (
            "__single__" if frame.series_id_col is None else str(values[frame.series_id_col])
        )
        timestamp = values[frame.timestamp_col].isoformat()
        target = float(values[frame.target_col])
        rows.append((series_id, timestamp, target))
        row_covariates.append({name: float(values[name]) for name in covariate_cols})
    try:
        return _native.ForecastFrame(
            rows,
            frame.freq,
            timestamp_col=frame.timestamp_col,
            target_col=frame.target_col,
            series_id_col=frame.series_id_col,
            static_covariates=list(frame.static_covariates),
            known_future_covariates=list(frame.known_future_covariates),
            historical_covariates=list(frame.historical_covariates),
            row_covariates=row_covariates if covariate_cols else None,
            sample_weights=sample_weights,
            sample_weight_col=frame.sample_weight_col,
            allow_irregular=frame.allow_irregular,
            allow_missing_targets=frame.allow_missing_targets,
            allow_missing_covariates=frame.allow_missing_covariates,
        )
    except TypeError:
        return _native.ForecastFrame(rows, frame.freq)
