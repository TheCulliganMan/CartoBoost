from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any


def _require_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs.
        raise ImportError(
            "CartoBoost forecasting lag features require pandas. Install pandas to use "
            "cartoboost.forecasting."
        ) from exc
    return pd


@dataclass(frozen=True)
class LagFeatureConfig:
    """Target lag configuration for leakage-safe forecasting features."""

    lags: Sequence[int] = (1,)
    difference_lags: Sequence[int] = ()
    rolling_trend_windows: Sequence[int] = ()
    partial_rolling_mean_windows: Sequence[int] = ()


# Stable naming used by the Rust-backed forecasting boundary.  The legacy
# implementation name remains internal to this module; callers should use
# ``LagConfig`` from ``cartoboost.forecasting``.
LagConfig = LagFeatureConfig


@dataclass(frozen=True)
class RollingFeatureConfig:
    """Shifted rolling and expanding target summary configuration."""

    windows: Sequence[int] = (3,)
    aggregations: Sequence[str] = ("mean",)
    min_periods: int | None = None
    include_expanding: bool = False
    expanding_aggregations: Sequence[str] = ("mean",)


@dataclass(frozen=True)
class CalendarFeatureConfig:
    """Calendar fields derived from the timestamp column."""

    features: Sequence[str] = ("hour", "dayofweek")


@dataclass
class LagFeatureBuilder:
    """Build leakage-safe lag features for panel forecasting.

    Target-derived features are shifted before aggregation, so every row at timestamp
    ``t`` only sees observations with timestamps strictly before ``t`` in the same panel.
    """

    time_col: str
    target_col: str
    panel_cols: Sequence[str] = field(default_factory=list)
    lag_config: LagFeatureConfig = field(default_factory=LagFeatureConfig)
    rolling_config: RollingFeatureConfig | None = None
    calendar_config: CalendarFeatureConfig | None = None
    static_cols: Sequence[str] = field(default_factory=list)
    known_future_cols: Sequence[str] = field(default_factory=list)
    holiday_fn: Callable[[Any], bool] | None = None

    def __post_init__(self) -> None:
        self.panel_cols = list(self.panel_cols)
        self.static_cols = list(self.static_cols)
        self.known_future_cols = list(self.known_future_cols)
        self._validate_config()
        self.feature_names_: list[str] = self._target_feature_names() + self._calendar_names()
        self.feature_names_.extend(self.static_cols)
        self.feature_names_.extend(self.known_future_cols)

    @property
    def feature_names(self) -> list[str]:
        return list(self.feature_names_)

    def fit(self, frame: Any) -> LagFeatureBuilder:
        data = self._as_frame(frame)
        self._validate_required_columns(data, include_target=True)
        data = self._sorted(data)
        self.history_ = data.copy()
        self.cutoffs_ = {
            key: group[self.time_col].max()
            for key, group in data.groupby(self._panel_group_keys(), dropna=False, sort=False)
        }
        if not self.panel_cols:
            self.cutoffs_ = {(): data[self.time_col].max()}
        self.is_fitted_ = True
        return self

    def fit_transform(self, frame: Any, *, drop_missing: bool = True) -> Any:
        data = self._as_frame(frame)
        self.fit(data)
        return self.transform_history(data, drop_missing=drop_missing)

    def transform_history(self, frame: Any, *, drop_missing: bool = True) -> Any:
        data = self._as_frame(frame)
        self._validate_required_columns(data, include_target=True)
        transformed = self._build_history_features(data)
        if drop_missing:
            target_features = self._target_feature_names()
            if target_features:
                transformed = transformed.dropna(subset=target_features)
        return transformed.reset_index(drop=True)

    def transform_future_row(self, history: Any, future_row: Any) -> Any:
        pd = _require_pandas()
        history_frame = self._as_frame(history)
        if isinstance(future_row, pd.Series):
            row = future_row.copy()
        else:
            row = pd.Series(dict(future_row))
        self._validate_required_columns(history_frame, include_target=True)
        self._validate_future_row(row)
        panel_history = self._panel_history_before(history_frame, row)

        shifted_target = panel_history[self.target_col]
        values: dict[str, float] = self._target_values_from_prior(shifted_target)
        values.update(self._calendar_values(row[self.time_col]))
        for col in self.static_cols:
            values[col] = self._row_or_history_value(row, panel_history, col)
        for col in self.known_future_cols:
            values[col] = row[col]
        return pd.Series({name: values.get(name) for name in self.feature_names_})

    def _build_history_features(self, frame: Any) -> Any:
        pd = _require_pandas()
        data = self._sorted(frame)
        target_features = self._target_feature_names()
        for name in target_features:
            data[name] = float("nan")
        for _, group in data.groupby(self._panel_group_keys(), dropna=False, sort=False):
            prior_targets = []
            for _timestamp, timestamp_group in group.groupby(self.time_col, sort=False):
                values = self._target_values_from_prior(pd.Series(prior_targets))
                for name, value in values.items():
                    data.loc[timestamp_group.index, name] = value
                prior_targets.extend(timestamp_group[self.target_col].tolist())
        for name, values in self._calendar_frame(data[self.time_col]).items():
            data[name] = values
        if not self.panel_cols:
            data = data.drop(columns=[self._single_panel_col()], errors="ignore")
        return data

    def _validate_config(self) -> None:
        for lag in self.lag_config.lags:
            if int(lag) != lag or lag < 1:
                raise ValueError("lags must be positive integers")
        for lag in self.lag_config.difference_lags:
            if int(lag) != lag or lag < 1:
                raise ValueError("difference_lags must be positive integers")
        for window in self.lag_config.rolling_trend_windows:
            if int(window) != window or window < 2:
                raise ValueError("rolling_trend_windows must be integers >= 2")
        if self.rolling_config is not None:
            for window in self.rolling_config.windows:
                if int(window) != window or window < 1:
                    raise ValueError("rolling windows must be positive integers")
            if self.rolling_config.min_periods is not None and (
                int(self.rolling_config.min_periods) != self.rolling_config.min_periods
                or self.rolling_config.min_periods < 1
            ):
                raise ValueError("rolling min_periods must be a positive integer or None")
            allowed = {"mean", "sum", "min", "max", "std"}
            for agg in list(self.rolling_config.aggregations) + list(
                self.rolling_config.expanding_aggregations
            ):
                if agg not in allowed:
                    raise ValueError(f"unsupported aggregation {agg!r}")
        if self.calendar_config is not None:
            allowed_calendar = {
                "hour",
                "dayofweek",
                "day",
                "month",
                "quarter",
                "year",
                "is_month_end",
                "is_weekend",
                "is_holiday",
            }
            for feature in self.calendar_config.features:
                if feature == "is_holiday" and self.holiday_fn is None:
                    raise ValueError("calendar feature 'is_holiday' requires holiday_fn")
                if feature not in allowed_calendar:
                    raise ValueError(f"unsupported calendar feature {feature!r}")

    def _validate_required_columns(self, frame: Any, *, include_target: bool) -> None:
        required = [self.time_col, *self.panel_cols, *self.static_cols, *self.known_future_cols]
        if include_target:
            required.append(self.target_col)
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")

    def _validate_future_row(self, row: Any) -> None:
        required = [self.time_col, *self.panel_cols, *self.known_future_cols]
        missing = [col for col in required if col not in row.index]
        if missing:
            raise ValueError(f"missing required future columns: {missing}")

    def _target_feature_names(self) -> list[str]:
        names = [f"{self.target_col}_lag_{lag}" for lag in self.lag_config.lags]
        if self.rolling_config is not None:
            for window in self.rolling_config.windows:
                for agg in self.rolling_config.aggregations:
                    names.append(f"{self.target_col}_roll_{window}_{agg}")
            if self.rolling_config.include_expanding:
                for agg in self.rolling_config.expanding_aggregations:
                    names.append(f"{self.target_col}_expand_{agg}")
        names.extend(
            f"{self.target_col}_delta_lag_{lag}" for lag in self.lag_config.difference_lags
        )
        names.extend(
            f"{self.target_col}_roll_trend_{window}"
            for window in self.lag_config.rolling_trend_windows
        )
        return names

    def _calendar_names(self) -> list[str]:
        if self.calendar_config is None:
            return []
        return [f"{self.time_col}_{feature}" for feature in self.calendar_config.features]

    def _calendar_frame(self, values: Any) -> dict[str, Any]:
        if self.calendar_config is None:
            return {}
        return {
            name: values.map(lambda ts, feature=feature: self._calendar_value(ts, feature))
            for name, feature in zip(
                self._calendar_names(),
                self.calendar_config.features,
                strict=True,
            )
        }

    def _calendar_values(self, timestamp: Any) -> dict[str, float]:
        if self.calendar_config is None:
            return {}
        return {
            name: self._calendar_value(timestamp, feature)
            for name, feature in zip(
                self._calendar_names(),
                self.calendar_config.features,
                strict=True,
            )
        }

    def _calendar_value(self, timestamp: Any, feature: str) -> float:
        pd = _require_pandas()
        ts = pd.Timestamp(timestamp)
        if feature == "hour":
            return float(ts.hour)
        if feature == "dayofweek":
            return float(ts.dayofweek)
        if feature == "day":
            return float(ts.day)
        if feature == "month":
            return float(ts.month)
        if feature == "quarter":
            return float(ts.quarter)
        if feature == "year":
            return float(ts.year)
        if feature == "is_month_end":
            return float(ts.is_month_end)
        if feature == "is_weekend":
            return float(ts.dayofweek >= 5)
        if feature == "is_holiday":
            holiday_fn = self.holiday_fn
            if holiday_fn is None:
                raise ValueError("calendar feature 'is_holiday' requires holiday_fn")
            return float(bool(holiday_fn(ts)))
        raise ValueError(f"unsupported calendar feature {feature!r}")

    def _panel_history_before(self, history: Any, row: Any) -> Any:
        panel = self._sorted(history)
        for col in self.panel_cols:
            panel = panel[panel[col] == row[col]]
        return panel[panel[self.time_col] < row[self.time_col]]

    def _row_or_history_value(self, row: Any, history: Any, col: str) -> Any:
        if col in row.index:
            return row[col]
        if col in history.columns and len(history) > 0:
            return history[col].iloc[-1]
        raise ValueError(f"missing static column {col!r} for future row")

    def _panel_group_keys(self) -> str | list[str]:
        return self.panel_cols if self.panel_cols else self._single_panel_col()

    def _sorted(self, frame: Any) -> Any:
        data = frame.copy()
        if not self.panel_cols:
            data[self._single_panel_col()] = 0
        return data.sort_values([*self.panel_cols, self.time_col], kind="mergesort")

    @staticmethod
    def _single_panel_col() -> str:
        return "__cartoboost_single_panel__"

    def _target_values_from_prior(self, prior_targets: Any) -> dict[str, float]:
        values: dict[str, float] = {}
        for lag in self.lag_config.lags:
            name = f"{self.target_col}_lag_{lag}"
            values[name] = self._series_value(prior_targets, -lag)
        if self.rolling_config is not None:
            for window in self.rolling_config.windows:
                window_values = prior_targets.tail(window)
                min_periods = self.rolling_config.min_periods or window
                for agg in self.rolling_config.aggregations:
                    name = f"{self.target_col}_roll_{window}_{agg}"
                    values[name] = (
                        self._aggregate(window_values, agg)
                        if len(window_values.dropna()) >= min_periods
                        else float("nan")
                    )
            if self.rolling_config.include_expanding:
                for agg in self.rolling_config.expanding_aggregations:
                    name = f"{self.target_col}_expand_{agg}"
                    values[name] = self._aggregate(prior_targets.dropna(), agg)
        for lag in self.lag_config.difference_lags:
            name = f"{self.target_col}_delta_lag_{lag}"
            if len(prior_targets) <= lag:
                values[name] = float("nan")
            else:
                values[name] = float(prior_targets.iloc[-1] - prior_targets.iloc[-1 - lag])
        for window in self.lag_config.rolling_trend_windows:
            name = f"{self.target_col}_roll_trend_{window}"
            window_values = prior_targets.tail(window).dropna()
            if len(window_values) < window:
                values[name] = float("nan")
            else:
                values[name] = float(
                    (window_values.iloc[-1] - window_values.iloc[0]) / (window - 1)
                )
        return values

    def _as_frame(self, frame: Any) -> Any:
        pd = _require_pandas()
        if isinstance(frame, pd.DataFrame):
            return frame.copy()
        return pd.DataFrame(frame)

    @staticmethod
    def _series_value(values: Any, position: int) -> float:
        if len(values) < abs(position):
            return float("nan")
        return values.iloc[position]

    @staticmethod
    def _rolling_agg(rolling: Any, agg: str) -> Any:
        return getattr(rolling, agg)()

    @staticmethod
    def _expanding_agg(expanding: Any, agg: str) -> Any:
        return getattr(expanding, agg)()

    @staticmethod
    def _aggregate(values: Any, agg: str) -> float:
        clean = values.dropna()
        if len(clean) == 0:
            return float("nan")
        return float(getattr(clean, agg)())
