from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..config import Backend, ChoiceStrEnum
from ._native_wrappers import NativeForecastWrapper
from .frequency import require_pandas
from .local.piecewise_linear import (
    _country_holiday_event_tuples,
    _country_holiday_years,
    _flatten_component_columns,
    _holiday_event_tuples,
)
from .schema import ForecastFrame


class Trend(ChoiceStrEnum):
    PIECEWISE_LINEAR = "piecewise_linear"
    FLAT = "flat"
    LINEAR = "linear"


class SeasonalityMode(ChoiceStrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class ComponentMode(ChoiceStrEnum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"


class PanelMode(ChoiceStrEnum):
    GLOBAL = "global"
    LOCAL = "local"


class Loss(ChoiceStrEnum):
    SMOOTH_L1 = "smooth_l1"
    L2 = "l2"
    HUBER = "huber"


class NBeatsForecaster(NativeForecastWrapper):
    """Thin Python wrapper for the Rust N-BEATS style forecasting expert."""

    native_class_name = "NBeatsForecaster"

    def __init__(
        self,
        *,
        input_size: int = 8,
        hidden_size: int = 16,
        epochs: int = 80,
        learning_rate: float = 0.01,
        backend: Backend = Backend.CPU,
        **metadata: Any,
    ) -> None:
        _validate_common(input_size, hidden_size, epochs, learning_rate)
        super().__init__(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            backend=_choice_value(backend),
            metadata=dict(metadata),
        )

    def _new_native_model(self) -> Any:
        try:
            from cartoboost import _native
        except ImportError as exc:
            raise NotImplementedError(
                "Rust binding for NBeatsForecaster is not available."
            ) from exc
        native_class = getattr(_native, self.native_class_name, None)
        if native_class is None:
            raise NotImplementedError("Rust binding for NBeatsForecaster is not available.")
        return native_class(
            input_size=self._params["input_size"],
            hidden_size=self._params["hidden_size"],
            epochs=self._params["epochs"],
            learning_rate=self._params["learning_rate"],
            backend=self._params["backend"],
        )


class NHiTSForecaster(NativeForecastWrapper):
    """Thin Python wrapper for the Rust N-HiTS style forecasting expert."""

    native_class_name = "NHiTSForecaster"

    def __init__(
        self,
        *,
        input_size: int = 12,
        hidden_size: int = 16,
        epochs: int = 80,
        learning_rate: float = 0.01,
        pooling_size: int = 2,
        backend: Backend = Backend.CPU,
        **metadata: Any,
    ) -> None:
        _validate_common(input_size, hidden_size, epochs, learning_rate)
        if pooling_size < 1 or pooling_size > input_size:
            raise ValueError("pooling_size must be between 1 and input_size")
        super().__init__(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            pooling_size=int(pooling_size),
            backend=_choice_value(backend),
            metadata=dict(metadata),
        )

    def _new_native_model(self) -> Any:
        try:
            from cartoboost import _native
        except ImportError as exc:
            raise NotImplementedError("Rust binding for NHiTSForecaster is not available.") from exc
        native_class = getattr(_native, self.native_class_name, None)
        if native_class is None:
            raise NotImplementedError("Rust binding for NHiTSForecaster is not available.")
        return native_class(
            input_size=self._params["input_size"],
            hidden_size=self._params["hidden_size"],
            epochs=self._params["epochs"],
            learning_rate=self._params["learning_rate"],
            pooling_size=self._params["pooling_size"],
            backend=self._params["backend"],
        )


class NeuralPanelForecaster(NativeForecastWrapper):
    """Thin Python wrapper for the Rust neural panel forecaster."""

    native_class_name = "NeuralPanelForecaster"

    def __init__(
        self,
        *,
        n_lags: int = 8,
        n_forecasts: int = 1,
        quantiles: tuple[float, ...] | list[float] | None = None,
        trend: Trend = Trend.PIECEWISE_LINEAR,
        n_changepoints: int = 10,
        changepoints_range: float = 0.8,
        daily_fourier_order: int = 0,
        weekly_fourier_order: int = 0,
        yearly_fourier_order: int = 0,
        custom_seasonalities: (
            list[tuple[str, float, int]] | list[tuple[str, float, int, str | None]] | None
        ) = None,
        seasonality_mode: SeasonalityMode = SeasonalityMode.ADDITIVE,
        events: dict[str, list[int]] | None = None,
        holidays: Any | None = None,
        holidays_mode: SeasonalityMode | None = None,
        country_holidays: str | None = None,
        country_holiday_years: list[int] | tuple[int, ...] | None = None,
        country_holiday_subdivision: str | None = None,
        event_mode: ComponentMode = ComponentMode.ADDITIVE,
        future_regressors: dict[str, ComponentMode] | None = None,
        lagged_regressors: dict[str, int] | None = None,
        ar_layers: list[int] | None = None,
        lagged_reg_layers: list[int] | None = None,
        trend_mode: PanelMode = PanelMode.GLOBAL,
        seasonality_global_local: PanelMode = PanelMode.GLOBAL,
        event_global_local: PanelMode = PanelMode.GLOBAL,
        regressor_global_local: PanelMode = PanelMode.GLOBAL,
        local_l2: float = 0.0,
        seed: int = 0,
        loss: Loss = Loss.SMOOTH_L1,
        epochs: int = 80,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        newer_sample_weight: bool = False,
        backend: Backend = Backend.CPU,
        **metadata: Any,
    ) -> None:
        _validate_panel(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            quantiles=quantiles,
            changepoints_range=changepoints_range,
            local_l2=local_l2,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        if holidays_mode is not None:
            if event_mode is not None and _choice_value(event_mode) != _choice_value(holidays_mode):
                raise ValueError("event_mode and holidays_mode must agree when both are set")
            event_mode = holidays_mode
        holiday_events, _ = _holiday_event_tuples(holidays, None)
        if country_holidays is not None:
            holiday_events.extend(
                _country_holiday_event_tuples(
                    country_holidays,
                    _country_holiday_years(country_holiday_years),
                    country_holiday_subdivision,
                )
            )
        future_regressors = _merge_future_regressors(
            future_regressors,
            _holiday_future_regressors(holiday_events, event_mode),
        )
        super().__init__(
            n_lags=int(n_lags),
            n_forecasts=int(n_forecasts),
            quantiles=list(quantiles or (0.5,)),
            trend=_choice_value(trend),
            n_changepoints=int(n_changepoints),
            changepoints_range=float(changepoints_range),
            daily_fourier_order=int(daily_fourier_order),
            weekly_fourier_order=int(weekly_fourier_order),
            yearly_fourier_order=int(yearly_fourier_order),
            custom_seasonalities=_normalize_custom_seasonalities(custom_seasonalities),
            seasonality_mode=_choice_value(seasonality_mode),
            events=dict(events or {}),
            event_mode=_choice_value(event_mode) if event_mode is not None else None,
            future_regressors=future_regressors or {},
            lagged_regressors=dict(lagged_regressors or {}),
            ar_layers=list(ar_layers or ()),
            lagged_reg_layers=list(lagged_reg_layers or ()),
            trend_mode=_choice_value(trend_mode),
            seasonality_global_local=_choice_value(seasonality_global_local),
            event_global_local=_choice_value(event_global_local),
            regressor_global_local=_choice_value(regressor_global_local),
            local_l2=float(local_l2),
            seed=int(seed),
            loss=_choice_value(loss),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            newer_sample_weight=bool(newer_sample_weight),
            backend=_choice_value(backend),
            metadata=dict(metadata),
        )
        self._holiday_events = holiday_events

    def _new_native_model(self) -> Any:
        params = dict(self._params)
        params.pop("metadata", None)
        native_class = _native_forecaster_class(self.native_class_name)
        return native_class(**params)

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        if self._holiday_events and args:
            args = self._coerce_holiday_fit_args(args)
        return super().fit(*args, **kwargs)

    def add_future_regressor(
        self,
        name: str,
        mode: ComponentMode = ComponentMode.ADDITIVE,
    ) -> NeuralPanelForecaster:
        self._ensure_unfitted()
        normalized_mode = _choice_value(mode)
        self._params["future_regressors"][str(name)] = normalized_mode
        return self

    def add_lagged_regressor(self, name: str, n_lags: int) -> NeuralPanelForecaster:
        self._ensure_unfitted()
        if n_lags < 1:
            raise ValueError("n_lags must be a positive integer")
        self._params["lagged_regressors"][str(name)] = int(n_lags)
        return self

    def add_events(
        self,
        name: str,
        lower_window: int = 0,
        upper_window: int = 0,
    ) -> NeuralPanelForecaster:
        self._ensure_unfitted()
        if not name:
            raise ValueError("event name must not be empty")
        if lower_window > upper_window:
            raise ValueError("lower_window must not exceed upper_window")
        self._params["events"][str(name)] = list(range(int(lower_window), int(upper_window) + 1))
        return self

    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        *,
        condition_name: str | None = None,
    ) -> NeuralPanelForecaster:
        self._ensure_unfitted()
        if not name:
            raise ValueError("seasonality name must not be empty")
        if period <= 0.0:
            raise ValueError("period must be positive")
        if fourier_order < 1:
            raise ValueError("fourier_order must be a positive integer")
        custom_seasonalities = self._params["custom_seasonalities"]
        custom_seasonalities.append(
            (
                str(name),
                float(period),
                int(fourier_order),
                None if condition_name is None else str(condition_name),
            )
        )
        return self

    def add_country_holidays(
        self,
        country_name: str,
        *,
        years: list[int] | tuple[int, ...] | None = None,
        subdivision: str | None = None,
    ) -> NeuralPanelForecaster:
        self._ensure_unfitted()
        holiday_events = _country_holiday_event_tuples(
            country_name,
            _country_holiday_years(years),
            subdivision,
        )
        self._holiday_events.extend(holiday_events)
        self._params["future_regressors"] = _merge_future_regressors(
            self._params["future_regressors"],
            _holiday_future_regressors(holiday_events, self._params["event_mode"]),
        )
        return self

    def predict(self, horizon: int, *, known_future: Any | None = None) -> Any:
        if known_future is None:
            return super().predict(int(horizon))
        if self._holiday_events:
            known_future = self._coerce_holiday_frame(known_future)
        self._check_is_fitted()
        native_frame = _native_frame_from_forecast_frame(known_future)
        method = getattr(self._native_model, "predict_with_known_future", None)
        if method is None:
            raise NotImplementedError(
                "Rust binding for NeuralPanelForecaster known-future prediction is not available."
            )
        return method(int(horizon), native_frame)

    def components_json(self, horizon: int, *, known_future: Any | None = None) -> Any:
        self._check_is_fitted()
        if known_future is None:
            method = getattr(self._native_model, "components_json", None)
            if method is None:
                raise NotImplementedError(
                    "Rust binding for NeuralPanelForecaster components output is not available."
                )
            return method(int(horizon))
        if self._holiday_events:
            known_future = self._coerce_holiday_frame(known_future)
        native_frame = _native_frame_from_forecast_frame(known_future)
        method = getattr(self._native_model, "components_json", None)
        if method is None:
            raise NotImplementedError(
                "Rust binding for NeuralPanelForecaster components output is not available."
            )
        return method(int(horizon), native_frame)

    def components(self, horizon: int, *, known_future: Any | None = None) -> dict[str, Any]:
        return dict(json.loads(self.components_json(horizon, known_future=known_future)))

    def history_components_json(self) -> str:
        self._check_is_fitted()
        method = getattr(self._native_model, "history_components_json", None)
        if method is None:
            raise NotImplementedError(
                "Rust binding for NeuralPanelForecaster history output is not available."
            )
        return method()

    def history_components(self) -> dict[str, Any]:
        return dict(json.loads(self.history_components_json()))

    def history_components_frame(self) -> Any:
        pd = require_pandas()
        payload = self.history_components()
        rows: list[dict[str, Any]] = []
        for record in payload.get("records", []):
            row = dict(record)
            components = row.pop("feature_contributions", row.pop("components", {}))
            if isinstance(components, Mapping):
                _flatten_component_columns(row, "feature_contributions", components)
            rows.append(row)
        return pd.DataFrame(rows)

    def make_future_dataframe(
        self,
        frame: ForecastFrame,
        periods: int,
        *,
        include_history: bool = False,
        future_covariates: Mapping[str, Any] | None = None,
    ) -> ForecastFrame:
        if periods < 1:
            raise ValueError("periods must be a positive integer")
        if not isinstance(frame, ForecastFrame):
            raise TypeError("frame must be a cartoboost.forecasting.ForecastFrame")
        if frame.freq is None:
            raise ValueError("frame must have a regular frequency to build a future dataframe")
        if frame.sample_weight_col is not None:
            raise ValueError("future dataframe generation does not support sample weights")
        pd = require_pandas()
        data = frame.to_pandas()
        future_rows: list[dict[str, Any]] = []
        series_ids = frame.series_ids if frame.is_panel else [None]
        offset = pd.tseries.frequencies.to_offset(frame.freq)
        future_covariates = dict(future_covariates or {})

        for series_id in series_ids:
            if frame.is_panel:
                series_data = data[data[frame.series_id_col] == series_id]
                series_key = str(series_id)
            else:
                series_data = data
                series_key = "__single__"
            if series_data.empty:
                raise ValueError("frame must contain at least one row per series")
            last_row = series_data.iloc[-1]
            last_timestamp = pd.to_datetime(last_row[frame.timestamp_col], errors="raise")
            future_timestamps = pd.date_range(
                start=last_timestamp + offset,
                periods=periods,
                freq=frame.freq,
            )
            static_values = {name: float(last_row[name]) for name in frame.static_covariates}
            for step, timestamp in enumerate(future_timestamps):
                row: dict[str, Any] = {
                    frame.timestamp_col: timestamp,
                    frame.target_col: 0.0,
                }
                if frame.series_id_col is not None:
                    row[frame.series_id_col] = series_id
                row.update(static_values)
                for name in frame.known_future_covariates + frame.historical_covariates:
                    row[name] = _future_covariate_value(
                        future_covariates,
                        name,
                        series_key,
                        step,
                        periods,
                    )
                future_rows.append(row)

        future_data = pd.DataFrame(future_rows)
        if include_history:
            combined = pd.concat([data, future_data], ignore_index=True, sort=False)
        else:
            combined = future_data
        future_frame = ForecastFrame.from_pandas(
            combined,
            timestamp_col=frame.timestamp_col,
            target_col=frame.target_col,
            series_id_col=frame.series_id_col,
            freq=frame.freq,
            static_covariates=frame.static_covariates,
            known_future_covariates=list(frame.known_future_covariates),
            historical_covariates=frame.historical_covariates,
            allow_irregular=frame.allow_irregular,
            sample_weight_col=frame.sample_weight_col,
        )
        if self._holiday_events:
            future_frame = _forecast_frame_with_holidays(future_frame, self._holiday_events)
        return future_frame

    def _coerce_holiday_fit_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        first = args[0]
        if not isinstance(first, ForecastFrame):
            raise ValueError(
                "holiday and country_holiday support requires ForecastFrame inputs with timestamps"
            )
        return (self._coerce_holiday_frame(first), *args[1:])

    def _coerce_holiday_frame(self, frame: ForecastFrame) -> ForecastFrame:
        if not self._holiday_events:
            return frame
        return _forecast_frame_with_holidays(frame, self._holiday_events)

    def _ensure_unfitted(self) -> None:
        if self.is_fitted_:
            raise RuntimeError(f"{self.__class__.__name__} cannot be modified after fitting")


class LaneNeuralPanelForecaster(NeuralPanelForecaster):
    """Taxi lane neural panel wrapper with origin/destination/lane metadata."""

    native_class_name = "LaneNeuralPanelForecaster"

    def __init__(self, *, embedding_dim: int = 8, **kwargs: Any) -> None:
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be a positive integer")
        super().__init__(**kwargs)
        self._params["embedding_dim"] = int(embedding_dim)

    def predict_for_lanes(self, horizon: int, series_ids: list[str] | tuple[str, ...]) -> Any:
        self._check_is_fitted()
        method = getattr(self._native_model, "predict_for_lanes", None)
        if method is None:
            raise NotImplementedError(
                "Rust binding for LaneNeuralPanelForecaster does not expose predict_for_lanes()."
            )
        return method(int(horizon), [str(series_id) for series_id in series_ids])


NHITSForecaster = NHiTSForecaster
NBEATSForecaster = NBeatsForecaster

__all__ = [
    "Backend",
    "ChoiceStrEnum",
    "ComponentMode",
    "Loss",
    "LaneNeuralPanelForecaster",
    "NBeatsForecaster",
    "NBEATSForecaster",
    "NeuralPanelForecaster",
    "NHiTSForecaster",
    "PanelMode",
    "SeasonalityMode",
    "NHITSForecaster",
    "Trend",
]


def _normalize_custom_seasonalities(
    seasonalities: (list[tuple[str, float, int]] | list[tuple[str, float, int, str | None]] | None),
) -> list[tuple[str, float, int, str | None]]:
    if seasonalities is None:
        return []
    normalized: list[tuple[str, float, int, str | None]] = []
    for seasonality in seasonalities:
        if len(seasonality) == 3:
            name, period_days, fourier_order = seasonality
            normalized.append((str(name), float(period_days), int(fourier_order), None))
        elif len(seasonality) == 4:
            name, period_days, fourier_order, condition_name = seasonality
            normalized.append(
                (
                    str(name),
                    float(period_days),
                    int(fourier_order),
                    None if condition_name is None else str(condition_name),
                )
            )
        else:
            raise ValueError("custom_seasonalities entries must be 3-tuples or 4-tuples")
    return normalized


def _choice_value(value: str | ChoiceStrEnum) -> str:
    if isinstance(value, ChoiceStrEnum):
        return value.value
    return str(value)


def _merge_future_regressors(
    base: dict[str, ComponentMode] | None,
    holiday_regressors: dict[str, ComponentMode],
) -> dict[str, ComponentMode]:
    merged = dict(base or {})
    for name, mode in holiday_regressors.items():
        existing = merged.get(name)
        if existing is not None and existing != mode:
            raise ValueError(f"future_regressors already defines {name!r} with a different mode")
        merged[name] = mode
    return merged


def _holiday_future_regressors(
    holidays: list[tuple[str, str, int | None, int | None]],
    event_mode: ComponentMode | None,
) -> dict[str, ComponentMode]:
    if event_mode is None:
        return {}
    return {name: event_mode for name, *_ in holidays}


def _forecast_frame_with_holidays(
    frame: ForecastFrame,
    holiday_specs: list[tuple[str, str, int | None, int | None]],
) -> ForecastFrame:
    if not holiday_specs:
        return frame
    pd = __import__("pandas")
    data = frame.to_pandas()
    timestamp_values = pd.to_datetime(data[frame.timestamp_col], errors="raise").dt.normalize()
    holiday_names: list[str] = []
    for name, timestamp, lower_window, upper_window in holiday_specs:
        if not name:
            raise ValueError("holiday names must not be empty")
        if name in {frame.timestamp_col, frame.target_col, frame.series_id_col}:
            raise ValueError(f"holiday name {name!r} conflicts with a reserved forecast column")
        holiday_date = pd.to_datetime(timestamp, errors="raise").normalize()
        lower = 0 if lower_window is None else int(lower_window)
        upper = 0 if upper_window is None else int(upper_window)
        if lower > upper:
            raise ValueError(f"holiday {name!r} has invalid window bounds")
        offsets = (timestamp_values - holiday_date).dt.days
        generated = ((offsets >= lower) & (offsets <= upper)).astype(float)
        if name in data.columns:
            existing = data[name].to_numpy(dtype=float, copy=False)
            if not np.isfinite(existing).all():
                raise ValueError(f"holiday column {name!r} must contain only finite values")
            data[name] = np.maximum(existing, generated.to_numpy(dtype=float, copy=False))
        else:
            data[name] = generated
        holiday_names.append(name)
    known_future = list(frame.known_future_covariates)
    for name in holiday_names:
        if name not in known_future:
            known_future.append(name)
    return ForecastFrame.from_pandas(
        data,
        timestamp_col=frame.timestamp_col,
        target_col=frame.target_col,
        series_id_col=frame.series_id_col,
        freq=frame.freq,
        static_covariates=frame.static_covariates,
        known_future_covariates=known_future,
        historical_covariates=frame.historical_covariates,
        allow_irregular=frame.allow_irregular,
        sample_weight_col=frame.sample_weight_col,
    )


def _future_covariate_value(
    future_covariates: Mapping[str, Any],
    name: str,
    series_key: str,
    step: int,
    periods: int,
) -> float:
    if name not in future_covariates:
        raise ValueError(f"future covariate {name!r} is required to build the future dataframe")
    values = future_covariates[name]
    if isinstance(values, Mapping):
        if series_key not in values:
            raise ValueError(
                f"future covariate {name!r} is missing values for series {series_key!r}"
            )
        values = values[series_key]
    sequence = list(values)
    if len(sequence) != periods:
        raise ValueError(
            f"future covariate {name!r} must contain exactly {periods} values for each series"
        )
    return float(sequence[step])


def _validate_common(
    input_size: int,
    hidden_size: int,
    epochs: int,
    learning_rate: float,
) -> None:
    if input_size < 1:
        raise ValueError("input_size must be a positive integer")
    if hidden_size < 1:
        raise ValueError("hidden_size must be a positive integer")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")


def _validate_panel(
    *,
    n_lags: int,
    n_forecasts: int,
    quantiles: tuple[float, ...] | list[float] | None,
    changepoints_range: float,
    local_l2: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> None:
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative")
    if n_forecasts < 1:
        raise ValueError("n_forecasts must be a positive integer")
    if changepoints_range <= 0.0 or changepoints_range > 1.0:
        raise ValueError("changepoints_range must be in (0, 1]")
    if local_l2 < 0.0:
        raise ValueError("local_l2 must be non-negative")
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")
    for quantile in quantiles or (0.5,):
        if quantile <= 0.0 or quantile >= 1.0:
            raise ValueError("quantiles must be in (0, 1)")


def _native_forecaster_class(name: str) -> Any:
    try:
        from cartoboost import _native
    except ImportError as exc:
        raise NotImplementedError(f"Rust binding for {name} is not available.") from exc
    native_class = getattr(_native, name, None)
    if native_class is None:
        raise NotImplementedError(f"Rust binding for {name} is not available.")
    return native_class


def _native_frame_from_forecast_frame(value: Any) -> Any:
    native_frame = getattr(value, "_native_frame", None)
    if native_frame is not None:
        return native_frame
    if value.__class__.__name__ == "ForecastFrame" and value.__class__.__module__.endswith(
        "._native"
    ):
        return value
    raise TypeError(
        "known_future must be a cartoboost.forecasting.ForecastFrame or native ForecastFrame"
    )
