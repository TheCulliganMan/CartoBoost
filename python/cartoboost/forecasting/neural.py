from __future__ import annotations

from typing import Any

from ._native_wrappers import NativeForecastWrapper


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
        **metadata: Any,
    ) -> None:
        _validate_common(input_size, hidden_size, epochs, learning_rate)
        super().__init__(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            epochs=int(epochs),
            learning_rate=float(learning_rate),
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
        trend: str = "piecewise_linear",
        n_changepoints: int = 10,
        changepoints_range: float = 0.8,
        daily_fourier_order: int = 0,
        weekly_fourier_order: int = 0,
        yearly_fourier_order: int = 0,
        custom_seasonalities: list[tuple[str, float, int]] | None = None,
        seasonality_mode: str = "additive",
        events: dict[str, list[int]] | None = None,
        event_mode: str = "additive",
        future_regressors: dict[str, str] | None = None,
        lagged_regressors: dict[str, int] | None = None,
        ar_layers: list[int] | None = None,
        lagged_reg_layers: list[int] | None = None,
        trend_mode: str = "global",
        seasonality_global_local: str = "global",
        local_l2: float = 0.0,
        seed: int = 0,
        **metadata: Any,
    ) -> None:
        _validate_panel(
            n_lags=n_lags,
            n_forecasts=n_forecasts,
            quantiles=quantiles,
            changepoints_range=changepoints_range,
            local_l2=local_l2,
        )
        super().__init__(
            n_lags=int(n_lags),
            n_forecasts=int(n_forecasts),
            quantiles=list(quantiles or (0.5,)),
            trend=str(trend),
            n_changepoints=int(n_changepoints),
            changepoints_range=float(changepoints_range),
            daily_fourier_order=int(daily_fourier_order),
            weekly_fourier_order=int(weekly_fourier_order),
            yearly_fourier_order=int(yearly_fourier_order),
            custom_seasonalities=list(custom_seasonalities or ()),
            seasonality_mode=str(seasonality_mode),
            events=dict(events or {}),
            event_mode=str(event_mode),
            future_regressors=dict(future_regressors or {}),
            lagged_regressors=dict(lagged_regressors or {}),
            ar_layers=list(ar_layers or ()),
            lagged_reg_layers=list(lagged_reg_layers or ()),
            trend_mode=str(trend_mode),
            seasonality_global_local=str(seasonality_global_local),
            local_l2=float(local_l2),
            seed=int(seed),
            metadata=dict(metadata),
        )

    def _new_native_model(self) -> Any:
        params = dict(self._params)
        params.pop("metadata", None)
        native_class = _native_forecaster_class(self.native_class_name)
        return native_class(**params)


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
    "LaneNeuralPanelForecaster",
    "NBeatsForecaster",
    "NBEATSForecaster",
    "NeuralPanelForecaster",
    "NHiTSForecaster",
    "NHITSForecaster",
]


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
) -> None:
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative")
    if n_forecasts < 1:
        raise ValueError("n_forecasts must be a positive integer")
    if changepoints_range <= 0.0 or changepoints_range > 1.0:
        raise ValueError("changepoints_range must be in (0, 1]")
    if local_l2 < 0.0:
        raise ValueError("local_l2 must be non-negative")
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
