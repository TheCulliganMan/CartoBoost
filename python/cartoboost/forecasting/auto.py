from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._artifacts import (
    decode_stable_forecast_artifact,
    library_version,
    stable_forecast_artifact_payload,
)
from ..config import BoosterConfig, Objective
from ._native_wrappers import (
    _forecast_frame_from_artifact,
    _forecast_frame_to_artifact,
    _native_class,
)
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
    booster_config: BoosterConfig = BoosterConfig()
    n_estimators: int | None = None


@dataclass(frozen=True)
class AutoForecastConfig(AutoForecasterConfig):
    """Stable name for the typed Rust-backed automatic forecast configuration."""


class AutoForecaster(BaseForecaster):
    """Deterministic out-of-the-box CartoBoost forecaster.

    The Python class is intentionally thin: model behavior is delegated to the
    Rust-backed forecaster selected by the fixed routing rules. This facade does
    not run hyperparameter search or benchmark-specific tuning.
    """

    def __init__(
        self,
        *,
        config: AutoForecasterConfig | None = None,
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
        booster_config: BoosterConfig | None = None,
        n_estimators: int | None = None,
    ) -> None:
        if config is not None:
            if not isinstance(config, AutoForecasterConfig):
                raise TypeError("config must be an AutoForecastConfig")
            seed = config.seed
            season_length = config.season_length
            quantiles = config.quantiles
            n_threads = config.n_threads
            objective = config.objective
            validation_window = config.validation_window
            validation_origin_count = config.validation_origin_count
            baseline_displacement_gain = config.baseline_displacement_gain
            hard_winner_relative_gain = config.hard_winner_relative_gain
            min_blend_weight = config.min_blend_weight
            max_blend_weight = config.max_blend_weight
            max_direct_horizon = config.max_direct_horizon
            max_candidate_count = config.max_candidate_count
            covariate_features = config.covariate_features
            covariate_calendar_interactions = config.covariate_calendar_interactions
            rich_calendar_features = config.rich_calendar_features
            elapsed_calendar_features = config.elapsed_calendar_features
            elapsed_calendar_periods = config.elapsed_calendar_periods
            ewm_alpha_percents = config.ewm_alpha_percents
            partial_rolling_mean_windows = config.partial_rolling_mean_windows
            booster_config = config.booster_config
            n_estimators = config.n_estimators
        if n_estimators is not None:
            if int(n_estimators) <= 0:
                raise ValueError("n_estimators must be positive when provided")
            base_booster = booster_config or BoosterConfig()
            booster_config = BoosterConfig(
                n_estimators=int(n_estimators),
                learning_rate=base_booster.learning_rate,
                max_depth=base_booster.max_depth,
                min_samples_leaf=base_booster.min_samples_leaf,
                min_gain=base_booster.min_gain,
                split_policy=base_booster.split_policy,
                n_threads=base_booster.n_threads,
            )
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
            booster_config=booster_config or BoosterConfig(),
            n_estimators=None if n_estimators is None else int(n_estimators),
        )
        if self.config.validation_origin_count <= 0:
            raise ValueError("validation_origin_count must be positive")
        if self.config.max_candidate_count is not None and self.config.max_candidate_count <= 0:
            raise ValueError("max_candidate_count must be positive when provided")
        self._model: Any | None = None
        self._effective_covariate_features: list[str] | None = None
        self._training_frame: ForecastFrame | None = None

    def fit(self, frame: ForecastFrame, *_args: Any, **_kwargs: Any) -> AutoForecaster:
        if not isinstance(frame, ForecastFrame):
            raise TypeError("AutoForecaster.fit requires a ForecastFrame")
        native_class = _native_class("AutoForecastModel")
        if native_class is None:
            raise NotImplementedError("Rust binding for AutoForecastModel is not available.")
        effective_covariates = self._covariate_features_for_frame(frame)
        model = native_class(**self._native_params(effective_covariates))
        model.fit(frame._native_frame)
        self._model = model
        self._effective_covariate_features = list(effective_covariates)
        self._training_frame = frame
        self._mark_fitted()
        return self

    def _new_native_model(self) -> Any:
        """Create an unfitted Rust selector for native rolling backtests."""

        native_class = _native_class("AutoForecastModel")
        if native_class is None:
            raise NotImplementedError("Rust binding for AutoForecastModel is not available.")
        effective_covariates = list(self.config.covariate_features or ())
        return native_class(**self._native_params(effective_covariates))

    def _native_params(self, effective_covariates: Sequence[str]) -> dict[str, Any]:
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
        }
        if self.config.n_estimators is not None:
            params["n_estimators"] = self.config.n_estimators
        return params

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        if self._training_frame is None:
            raise RuntimeError("AutoForecaster has no training frame to serialize")
        config_payload = _auto_config_payload(self.config)
        payload = {
            "training_frame": _forecast_frame_to_artifact(self._training_frame),
            "params": config_payload,
        }
        artifact = stable_forecast_artifact_payload(
            "auto_forecaster",
            payload=payload,
            library_version=library_version(),
            training_config=config_payload,
        )
        Path(path).write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> AutoForecaster:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = decode_stable_forecast_artifact(raw, "auto_forecaster")
        payload = artifact["payload"]
        config_payload = artifact["training_config"]
        if not isinstance(config_payload, dict):
            raise ValueError("auto_forecaster training_config must be an object")
        frame = _forecast_frame_from_artifact(payload.get("training_frame"))
        obj = cls(config=_auto_config_from_payload(config_payload))
        obj.fit(frame)
        return obj

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
            "booster_config": self.config.booster_config.to_dict(),
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
            "booster_config": self.config.booster_config,
            "n_estimators": self.config.n_estimators,
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


def _auto_config_payload(config: AutoForecasterConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "season_length": config.season_length,
        "quantiles": list(config.quantiles),
        "n_threads": config.n_threads,
        "no_hyperopt": config.no_hyperopt,
        "objective": str(config.objective),
        "validation_window": config.validation_window,
        "validation_origin_count": config.validation_origin_count,
        "baseline_displacement_gain": config.baseline_displacement_gain,
        "hard_winner_relative_gain": config.hard_winner_relative_gain,
        "min_blend_weight": config.min_blend_weight,
        "max_blend_weight": config.max_blend_weight,
        "max_direct_horizon": config.max_direct_horizon,
        "max_candidate_count": config.max_candidate_count,
        "covariate_features": (
            None if config.covariate_features is None else list(config.covariate_features)
        ),
        "covariate_calendar_interactions": config.covariate_calendar_interactions,
        "rich_calendar_features": config.rich_calendar_features,
        "elapsed_calendar_features": config.elapsed_calendar_features,
        "elapsed_calendar_periods": list(config.elapsed_calendar_periods),
        "ewm_alpha_percents": list(config.ewm_alpha_percents),
        "partial_rolling_mean_windows": list(config.partial_rolling_mean_windows),
        "booster_config": config.booster_config.to_dict(),
        "n_estimators": config.n_estimators,
    }


def _auto_config_from_payload(payload: dict[str, Any]) -> AutoForecastConfig:
    booster_payload = dict(payload.get("booster_config") or {})
    from ..config import SplitPolicy

    booster = BoosterConfig(
        n_estimators=int(booster_payload.get("n_estimators", 100)),
        learning_rate=float(booster_payload.get("learning_rate", 0.05)),
        max_depth=int(booster_payload.get("max_depth", 4)),
        min_samples_leaf=int(booster_payload.get("min_samples_leaf", 20)),
        min_gain=float(booster_payload.get("min_gain", 1e-8)),
        split_policy=SplitPolicy(booster_payload.get("split_policy", "auto")),
        n_threads=booster_payload.get("n_threads"),
    )
    return AutoForecastConfig(
        seed=int(payload.get("seed", 42)),
        season_length=payload.get("season_length"),
        quantiles=tuple(payload.get("quantiles", ())),
        n_threads=payload.get("n_threads"),
        no_hyperopt=bool(payload.get("no_hyperopt", True)),
        objective=Objective(payload.get("objective", Objective.RMSE_WAPE.value)),
        validation_window=payload.get("validation_window"),
        validation_origin_count=int(payload.get("validation_origin_count", 2)),
        baseline_displacement_gain=float(payload.get("baseline_displacement_gain", 0.03)),
        hard_winner_relative_gain=float(payload.get("hard_winner_relative_gain", 0.05)),
        min_blend_weight=float(payload.get("min_blend_weight", 0.15)),
        max_blend_weight=float(payload.get("max_blend_weight", 0.85)),
        max_direct_horizon=int(payload.get("max_direct_horizon", 28)),
        max_candidate_count=payload.get("max_candidate_count"),
        covariate_features=(
            None
            if payload.get("covariate_features") is None
            else tuple(payload["covariate_features"])
        ),
        covariate_calendar_interactions=bool(payload.get("covariate_calendar_interactions", False)),
        rich_calendar_features=bool(payload.get("rich_calendar_features", False)),
        elapsed_calendar_features=bool(payload.get("elapsed_calendar_features", False)),
        elapsed_calendar_periods=tuple(payload.get("elapsed_calendar_periods", ())),
        ewm_alpha_percents=tuple(payload.get("ewm_alpha_percents", ())),
        partial_rolling_mean_windows=tuple(payload.get("partial_rolling_mean_windows", ())),
        booster_config=booster,
        n_estimators=payload.get("n_estimators"),
    )
