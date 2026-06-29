from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..config import Objective
from ._native_wrappers import _native_class
from .base import BaseForecaster
from .schema import ForecastFrame


@dataclass(frozen=True)
class AutoForecasterConfig:
    """Deterministic routing configuration for CartoBoost forecasting."""

    seed: int = 42
    season_length: int | None = None
    quantiles: tuple[float, ...] = ()
    n_threads: int | None = None
    no_hyperopt: bool = True
    objective: Objective = Objective.RMSE_WAPE
    validation_window: int | None = None
    validation_origin_count: int = 2
    baseline_displacement_gain: float = 0.03
    hard_winner_relative_gain: float = 0.05
    min_blend_weight: float = 0.15
    max_blend_weight: float = 0.85
    max_direct_horizon: int = 28
    max_candidate_count: int | None = None
    covariate_features: tuple[str, ...] | None = None
    covariate_calendar_interactions: bool = False
    rich_calendar_features: bool = False
    elapsed_calendar_features: bool = False
    elapsed_calendar_periods: tuple[int, ...] = ()
    ewm_alpha_percents: tuple[int, ...] = ()
    partial_rolling_mean_windows: tuple[int, ...] = ()


class AutoForecaster(BaseForecaster):
    """Deterministic out-of-the-box CartoBoost forecaster.

    The Python class is intentionally thin: model behavior is delegated to the
    Rust-backed forecaster selected by the fixed routing rules. This facade does
    not run hyperparameter search or benchmark-specific tuning.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        season_length: int | None = None,
        quantiles: Sequence[float] | None = None,
        n_threads: int | None = None,
        objective: Objective = Objective.RMSE_WAPE,
        validation_window: int | None = None,
        validation_origin_count: int = 2,
        baseline_displacement_gain: float = 0.03,
        hard_winner_relative_gain: float = 0.05,
        min_blend_weight: float = 0.15,
        max_blend_weight: float = 0.85,
        max_direct_horizon: int = 28,
        max_candidate_count: int | None = None,
        covariate_features: Sequence[str] | None = None,
        covariate_calendar_interactions: bool = False,
        rich_calendar_features: bool = False,
        elapsed_calendar_features: bool = False,
        elapsed_calendar_periods: Sequence[int] | None = None,
        ewm_alpha_percents: Sequence[int] | None = None,
        partial_rolling_mean_windows: Sequence[int] | None = None,
        **cartoboost_params: Any,
    ) -> None:
        self.config = AutoForecasterConfig(
            seed=int(seed),
            season_length=season_length,
            quantiles=tuple(float(q) for q in (quantiles or ())),
            n_threads=n_threads,
            objective=str(objective),
            validation_window=validation_window,
            validation_origin_count=int(validation_origin_count),
            baseline_displacement_gain=float(baseline_displacement_gain),
            hard_winner_relative_gain=float(hard_winner_relative_gain),
            min_blend_weight=float(min_blend_weight),
            max_blend_weight=float(max_blend_weight),
            max_direct_horizon=int(max_direct_horizon),
            max_candidate_count=None if max_candidate_count is None else int(max_candidate_count),
            covariate_features=_normalize_optional_feature_names(
                covariate_features,
                name="covariate_features",
            ),
            covariate_calendar_interactions=bool(covariate_calendar_interactions),
            rich_calendar_features=bool(rich_calendar_features),
            elapsed_calendar_features=bool(elapsed_calendar_features),
            elapsed_calendar_periods=_normalize_elapsed_calendar_periods(
                elapsed_calendar_periods,
            ),
            ewm_alpha_percents=_normalize_ewm_alpha_percents(ewm_alpha_percents),
            partial_rolling_mean_windows=_normalize_positive_ints(
                partial_rolling_mean_windows,
                name="partial_rolling_mean_windows",
            ),
        )
        if self.config.validation_origin_count <= 0:
            raise ValueError("validation_origin_count must be positive")
        if self.config.max_candidate_count is not None and self.config.max_candidate_count <= 0:
            raise ValueError("max_candidate_count must be positive when provided")
        self.cartoboost_params = dict(cartoboost_params)
        self._model: Any | None = None
        self._effective_covariate_features: list[str] | None = None

    def fit(self, frame: ForecastFrame, *_args: Any, **_kwargs: Any) -> AutoForecaster:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("AutoForecaster.fit requires a ForecastFrame")
        native_class = _native_class("AutoForecastModel")
        if native_class is None:
            raise NotImplementedError("Rust binding for AutoForecastModel is not available.")
        effective_covariates = self._covariate_features_for_frame(frame)
        params = {
            "lags": self._default_lags(),
            "rolling_windows": self._default_windows(),
            "partial_rolling_mean_windows": list(self.config.partial_rolling_mean_windows),
            "rolling_std_windows": self._default_windows(),
            "rolling_min_windows": self._default_windows(),
            "rolling_max_windows": self._default_windows(),
            "ewm_alpha_percents": list(self.config.ewm_alpha_percents),
            "calendar_features": True,
            "rich_calendar_features": self.config.rich_calendar_features,
            "elapsed_calendar_features": self.config.elapsed_calendar_features,
            "elapsed_calendar_periods": list(self.config.elapsed_calendar_periods),
            "covariate_features": effective_covariates,
            "covariate_calendar_interactions": self.config.covariate_calendar_interactions,
            "season_length": self.config.season_length or 7,
            "validation_window": self.config.validation_window,
            "validation_origin_count": self.config.validation_origin_count,
            "objective": self.config.objective,
            "baseline_displacement_gain": self.config.baseline_displacement_gain,
            "hard_winner_relative_gain": self.config.hard_winner_relative_gain,
            "min_blend_weight": self.config.min_blend_weight,
            "max_blend_weight": self.config.max_blend_weight,
            "max_direct_horizon": self.config.max_direct_horizon,
            "max_candidate_count": self.config.max_candidate_count,
            **self.cartoboost_params,
        }
        model = native_class(**params)
        model.fit(frame._native_frame)
        self._model = model
        self._effective_covariate_features = list(effective_covariates)
        self._mark_fitted()
        return self

    def predict(self, horizon: int, *_args: Any, **_kwargs: Any) -> Any:
        self._check_is_fitted()
        horizon = self.validate_horizon(horizon)
        return self._model.predict(horizon)

    def get_metadata(self) -> dict[str, Any]:
        self._check_is_fitted()
        metadata_json = getattr(self._model, "metadata_json", None)
        metadata = {} if metadata_json is None else json.loads(metadata_json())
        metadata["auto_forecaster"] = {
            "seed": self.config.seed,
            "season_length": self.config.season_length,
            "quantiles": list(self.config.quantiles),
            "no_hyperopt": self.config.no_hyperopt,
            "objective": self.config.objective,
            "validation_window": self.config.validation_window,
            "validation_origin_count": self.config.validation_origin_count,
            "max_direct_horizon": self.config.max_direct_horizon,
            "max_candidate_count": self.config.max_candidate_count,
            "covariate_features": (
                None
                if self.config.covariate_features is None
                else list(self.config.covariate_features)
            ),
            "covariate_calendar_interactions": self.config.covariate_calendar_interactions,
            "rich_calendar_features": self.config.rich_calendar_features,
            "elapsed_calendar_features": self.config.elapsed_calendar_features,
            "elapsed_calendar_periods": list(self.config.elapsed_calendar_periods),
            "ewm_alpha_percents": list(self.config.ewm_alpha_percents),
            "partial_rolling_mean_windows": list(self.config.partial_rolling_mean_windows),
            "effective_covariate_features": list(self._effective_covariate_features or []),
            "selected_model": "AutoForecastModel",
        }
        return metadata

    @property
    def metadata_(self) -> dict[str, Any]:
        return self.get_metadata()

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "seed": self.config.seed,
            "season_length": self.config.season_length,
            "quantiles": self.config.quantiles,
            "n_threads": self.config.n_threads,
            "objective": self.config.objective,
            "validation_window": self.config.validation_window,
            "validation_origin_count": self.config.validation_origin_count,
            "baseline_displacement_gain": self.config.baseline_displacement_gain,
            "hard_winner_relative_gain": self.config.hard_winner_relative_gain,
            "min_blend_weight": self.config.min_blend_weight,
            "max_blend_weight": self.config.max_blend_weight,
            "max_direct_horizon": self.config.max_direct_horizon,
            "max_candidate_count": self.config.max_candidate_count,
            "covariate_features": self.config.covariate_features,
            "covariate_calendar_interactions": self.config.covariate_calendar_interactions,
            "rich_calendar_features": self.config.rich_calendar_features,
            "elapsed_calendar_features": self.config.elapsed_calendar_features,
            "elapsed_calendar_periods": self.config.elapsed_calendar_periods,
            "ewm_alpha_percents": self.config.ewm_alpha_percents,
            "partial_rolling_mean_windows": self.config.partial_rolling_mean_windows,
            **self.cartoboost_params,
        }

    def set_params(self, **params: Any) -> AutoForecaster:
        updated = {**self.get_params(), **params}
        self.__init__(**updated)
        return self

    def _default_lags(self) -> list[int]:
        season = self.config.season_length
        lags = [1, 2, 3, 7, 14, 28]
        if season is not None and season > 0:
            lags.append(int(season))
        return sorted(set(lags))

    def _default_windows(self) -> list[int]:
        windows = [7, 14, 28]
        season = self.config.season_length
        if season is not None and season > 1:
            windows.append(int(season))
        return sorted(set(windows))

    def _covariate_features_for_frame(self, frame: ForecastFrame) -> list[str]:
        if self.config.covariate_features is not None:
            return list(self.config.covariate_features)
        return list(frame.static_covariates)


def _normalize_optional_feature_names(
    values: Sequence[str] | None,
    *,
    name: str,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of column names, not a string")
    result = tuple(str(value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate column names")
    return result


def _normalize_ewm_alpha_percents(values: Sequence[int] | None) -> tuple[int, ...]:
    raw_values = () if values is None else tuple(values)
    result = tuple(int(value) for value in raw_values)
    if any(value < 1 or value > 100 for value in result):
        raise ValueError("ewm_alpha_percents must contain integers in 1..=100")
    if len(set(result)) != len(result):
        raise ValueError("ewm_alpha_percents must not contain duplicate values")
    return result


def _normalize_positive_ints(values: Sequence[int] | None, *, name: str) -> tuple[int, ...]:
    raw_values = () if values is None else tuple(values)
    result = tuple(int(value) for value in raw_values)
    if any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def _normalize_minimum_ints(
    values: Sequence[int] | None,
    *,
    name: str,
    minimum: int,
) -> tuple[int, ...]:
    raw_values = () if values is None else tuple(values)
    result = tuple(int(value) for value in raw_values)
    if any(value < minimum for value in result):
        raise ValueError(f"{name} must contain integers greater than or equal to {minimum}")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate values")
    return result


def _normalize_elapsed_calendar_periods(values: Sequence[int] | None) -> tuple[int, ...]:
    result = _normalize_minimum_ints(
        values,
        name="elapsed_calendar_periods",
        minimum=2,
    )
    if len(result) > 1:
        raise ValueError("elapsed_calendar_periods supports at most one value")
    return result
