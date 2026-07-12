from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..._artifacts import (
    decode_stable_forecast_artifact,
    library_version,
    stable_forecast_artifact_payload,
)
from ...config import BoosterConfig, SplitPolicy
from .._native_wrappers import (
    NativeForecastWrapper,
    _forecast_frame_from_artifact,
    _forecast_frame_to_artifact,
    _native_class,
)
from ..lag_features import CalendarFeatureConfig, LagConfig, RollingFeatureConfig
from ..schema import ForecastFrame


@dataclass(frozen=True)
class ForecastResult:
    """Thin result container for native CartoBoost lag forecast outputs."""

    frame: Any
    predictions: np.ndarray
    feature_names: list[str]
    regressor_metadata: dict[str, Any]


class CartoBoostLagForecaster(NativeForecastWrapper):
    """Thin wrapper for the Rust CartoBoost lag forecasting binding."""

    native_class_name = "CartoBoostLagForecaster"

    def __init__(
        self,
        *,
        time_col: str | None = None,
        target_col: str | None = None,
        panel_cols: Sequence[str] = (),
        frequency: str = "D",
        freq: str | None = None,
        allow_irregular: bool = False,
        booster_config: BoosterConfig | None = None,
        split_policy: SplitPolicy = SplitPolicy.AUTO,
        lag_config: LagConfig | None = None,
        rolling_config: RollingFeatureConfig | None = None,
        calendar_config: CalendarFeatureConfig | None = None,
        lags: Sequence[int] | None = None,
        rolling_windows: Sequence[int] | None = None,
        partial_rolling_mean_windows: Sequence[int] | None = None,
        rolling_std_windows: Sequence[int] | None = None,
        rolling_min_windows: Sequence[int] | None = None,
        rolling_max_windows: Sequence[int] | None = None,
        ewm_alpha_percents: Sequence[int] | None = None,
        covariate_features: Sequence[str] | None = None,
        covariate_indicator_values: Any = None,
        covariate_calendar_interactions: bool | None = None,
        difference_lags: Sequence[int] | None = None,
        rolling_trend_windows: Sequence[int] | None = None,
        calendar_features: bool | None = None,
        rich_calendar_features: bool | None = None,
        elapsed_calendar_features: bool | None = None,
        elapsed_calendar_periods: Sequence[int] | None = None,
        recursive: bool | None = None,
        prediction_interval_levels: Sequence[float] | None = None,
        trend_features: bool | None = None,
        target_mode: str | None = None,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        min_samples_leaf: int | None = None,
        min_gain: float | None = None,
        n_threads: int | None = None,
        regressor_params: dict[str, Any] | None = None,
    ) -> None:
        params = {
            key: value
            for key, value in {
                "booster_config": booster_config,
                "split_policy": split_policy,
                "allow_irregular": allow_irregular,
                "regressor_params": regressor_params,
                "lag_config": lag_config,
                "rolling_config": rolling_config,
                "calendar_config": calendar_config,
                "lags": lags,
                "rolling_windows": rolling_windows,
                "partial_rolling_mean_windows": partial_rolling_mean_windows,
                "rolling_std_windows": rolling_std_windows,
                "rolling_min_windows": rolling_min_windows,
                "rolling_max_windows": rolling_max_windows,
                "ewm_alpha_percents": ewm_alpha_percents,
                "covariate_features": covariate_features,
                "covariate_indicator_values": covariate_indicator_values,
                "covariate_calendar_interactions": covariate_calendar_interactions,
                "difference_lags": difference_lags,
                "rolling_trend_windows": rolling_trend_windows,
                "calendar_features": calendar_features,
                "rich_calendar_features": rich_calendar_features,
                "elapsed_calendar_features": elapsed_calendar_features,
                "elapsed_calendar_periods": elapsed_calendar_periods,
                "recursive": recursive,
                "prediction_interval_levels": prediction_interval_levels,
                "trend_features": trend_features,
                "target_mode": target_mode,
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "min_gain": min_gain,
                "n_threads": n_threads,
            }.items()
            if value is not None
        }
        self.time_col = time_col
        self.target_col = target_col
        self.panel_cols = list(panel_cols)
        self.frequency = freq or frequency
        self.allow_irregular = bool(allow_irregular)
        self.split_policy = SplitPolicy(split_policy)
        self.covariate_features = list(covariate_features or [])
        native_params = self._native_params(params)
        super().__init__(**native_params)

    def get_params(self) -> dict[str, Any]:
        params = dict(self._params)
        params.pop("splitters", None)
        params["split_policy"] = self.split_policy
        params["allow_irregular"] = self.allow_irregular
        return params

    def set_params(self, **params: Any) -> CartoBoostLagForecaster:
        if "splitters" in params:
            raise ValueError("splitters was removed in CartoBoost 0.3; use split_policy")
        if "split_policy" in params:
            self.split_policy = SplitPolicy(params.pop("split_policy"))
            self._params["splitters"] = _native_splitters_for_policy(self.split_policy)
        super().set_params(**params)
        return self

    def fit(self, *args: Any, **kwargs: Any) -> CartoBoostLagForecaster:
        if args and isinstance(args[0], ForecastFrame):
            self._training_frame = args[0]
        elif args and self.time_col is not None and self.target_col is not None:
            try:
                import pandas as pd
            except ImportError:
                pd = None
            if pd is not None and isinstance(args[0], pd.DataFrame):
                frame_data = args[0]
                series_id_col = self.panel_cols[0] if len(self.panel_cols) == 1 else None
                if len(self.panel_cols) > 1:
                    frame_data = frame_data.copy()
                    series_id_col = "__cartoboost_series_id__"
                    frame_data[series_id_col] = (
                        frame_data[self.panel_cols]
                        .astype(str)
                        .agg(
                            "|".join,
                            axis=1,
                        )
                    )
                try:
                    self._training_frame = ForecastFrame.from_pandas(
                        frame_data,
                        timestamp_col=self.time_col,
                        target_col=self.target_col,
                        series_id_col=series_id_col,
                        freq=self.frequency,
                        known_future_covariates=self.covariate_features,
                        allow_irregular=self.allow_irregular,
                    )
                except ValueError as exc:
                    if "duplicate timestamp rows" in str(exc):
                        raise ValueError(
                            "CartoBoostLagForecaster requires unique timestamps within each panel"
                        ) from exc
                    raise
        return super().fit(*args, **kwargs)

    def save(self, path: str | Path) -> None:
        self._check_is_fitted()
        training_frame = getattr(self, "_training_frame", None)
        if training_frame is None:
            raise RuntimeError("CartoBoostLagForecaster has no training frame to serialize")
        init_payload = {
            "time_col": self.time_col,
            "target_col": self.target_col,
            "panel_cols": list(self.panel_cols),
            "frequency": self.frequency,
            **self.get_params(),
        }
        payload = {
            "training_frame": _forecast_frame_to_artifact(training_frame),
            "init": init_payload,
        }
        artifact = stable_forecast_artifact_payload(
            "cartoboost_lag",
            payload=payload,
            library_version=library_version(),
            training_config=init_payload,
        )
        Path(path).write_text(json.dumps(artifact, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CartoBoostLagForecaster:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        artifact = decode_stable_forecast_artifact(raw, "cartoboost_lag")
        payload = artifact["payload"]
        init_payload = payload.get("init")
        if not isinstance(init_payload, dict):
            raise ValueError("cartoboost_lag artifact payload is missing init parameters")
        frame = _forecast_frame_from_artifact(payload.get("training_frame"))
        obj = cls(**init_payload)
        obj.fit(frame)
        return obj

    def _native_params(self, params: dict[str, Any]) -> dict[str, Any]:
        booster_config = params.pop("booster_config", None)
        params.pop("allow_irregular", None)
        split_policy = SplitPolicy(params.pop("split_policy", SplitPolicy.AUTO))
        if booster_config is not None:
            if not isinstance(booster_config, BoosterConfig):
                raise TypeError("booster_config must be a cartoboost.BoosterConfig")
            params.setdefault("n_estimators", booster_config.n_estimators)
            params.setdefault("learning_rate", booster_config.learning_rate)
            params.setdefault("max_depth", booster_config.max_depth)
            params.setdefault("min_samples_leaf", booster_config.min_samples_leaf)
            params.setdefault("min_gain", booster_config.min_gain)
            split_policy = booster_config.split_policy
        native_splitters = _native_splitters_for_policy(split_policy)
        if native_splitters is not None:
            params.setdefault("splitters", native_splitters)

        regressor_params = dict(params.pop("regressor_params", {}) or {})
        unsupported_regressor = sorted(
            set(regressor_params)
            - {
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_samples_leaf",
                "min_gain",
                "n_threads",
            }
        )
        if unsupported_regressor:
            raise ValueError(
                f"unsupported CartoBoostLagForecaster regressor_params: {unsupported_regressor}"
            )
        params.update(regressor_params)

        lag_config = params.pop("lag_config", None)
        if lag_config is not None:
            if not isinstance(lag_config, LagConfig):
                raise TypeError("lag_config must be a LagConfig")
            params.setdefault("lags", list(lag_config.lags))
            params.setdefault("difference_lags", list(lag_config.difference_lags))
            params.setdefault("rolling_trend_windows", list(lag_config.rolling_trend_windows))
            params.setdefault(
                "partial_rolling_mean_windows",
                list(lag_config.partial_rolling_mean_windows),
            )

        rolling_config = params.pop("rolling_config", None)
        if rolling_config is not None:
            if not isinstance(rolling_config, RollingFeatureConfig):
                raise TypeError("rolling_config must be a RollingFeatureConfig")
            unsupported_aggs = sorted(
                set(rolling_config.aggregations) - {"mean", "std", "min", "max"}
            )
            if unsupported_aggs:
                raise ValueError(
                    "native CartoBoostLagForecaster supports rolling mean/std/min/max only; "
                    f"unsupported: {unsupported_aggs}"
                )
            if rolling_config.include_expanding:
                raise ValueError(
                    "native CartoBoostLagForecaster does not support expanding features"
                )
            if rolling_config.min_periods not in {None, 1}:
                raise ValueError(
                    "native CartoBoostLagForecaster supports rolling_config.min_periods=None or 1"
                )
            if "mean" in rolling_config.aggregations:
                if rolling_config.min_periods == 1:
                    params.setdefault(
                        "partial_rolling_mean_windows",
                        list(rolling_config.windows),
                    )
                else:
                    params.setdefault("rolling_windows", list(rolling_config.windows))
            if "std" in rolling_config.aggregations:
                params.setdefault("rolling_std_windows", list(rolling_config.windows))
            if "min" in rolling_config.aggregations:
                params.setdefault("rolling_min_windows", list(rolling_config.windows))
            if "max" in rolling_config.aggregations:
                params.setdefault("rolling_max_windows", list(rolling_config.windows))

        calendar_config = params.pop("calendar_config", None)
        if calendar_config is not None:
            if not isinstance(calendar_config, CalendarFeatureConfig):
                raise TypeError("calendar_config must be a CalendarFeatureConfig")
            supported = {"dayofweek", "month", "day"}
            requested = set(calendar_config.features)
            unsupported = sorted(requested - supported)
            if unsupported:
                raise ValueError(
                    "native CartoBoostLagForecaster supports calendar features "
                    f"{sorted(supported)}; unsupported: {unsupported}"
                )
            params.setdefault("calendar_features", bool(requested))

        unsupported = sorted(
            set(params)
            - {
                "lags",
                "rolling_windows",
                "partial_rolling_mean_windows",
                "rolling_std_windows",
                "rolling_min_windows",
                "rolling_max_windows",
                "ewm_alpha_percents",
                "covariate_features",
                "covariate_indicator_values",
                "covariate_calendar_interactions",
                "difference_lags",
                "rolling_trend_windows",
                "calendar_features",
                "rich_calendar_features",
                "elapsed_calendar_features",
                "elapsed_calendar_periods",
                "recursive",
                "prediction_interval_levels",
                "trend_features",
                "target_mode",
                "n_estimators",
                "learning_rate",
                "max_depth",
                "min_samples_leaf",
                "min_gain",
                "splitters",
            }
        )
        if unsupported:
            raise ValueError(f"unsupported CartoBoostLagForecaster parameters: {unsupported}")
        return params

    def _coerce_fit_args(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if args and self.time_col is not None and self.target_col is not None:
            frame = self._native_frame_from_dataframe(args[0])
            if frame is not None:
                return (frame, *args[1:])
        return super()._coerce_fit_args(args)

    def predict(self, horizon: int, *, known_future: Any | None = None) -> Any:
        if known_future is None:
            return super().predict(horizon)
        self._check_is_fitted()
        native_frame = self._native_future_frame_from_dataframe(known_future)
        predict = getattr(self._native_model, "predict_with_known_future", None)
        if predict is None:
            raise NotImplementedError(
                "Rust binding for CartoBoostLagForecaster known-future prediction is not available."
            )
        return predict(int(horizon), native_frame)

    def _native_frame_from_dataframe(self, value: Any) -> Any | None:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised only in minimal installs.
            raise ImportError(
                "CartoBoostLagForecaster DataFrame input requires pandas. Install pandas to use "
                "time_col/target_col ergonomics."
            ) from exc
        if not isinstance(value, pd.DataFrame):
            return None
        required = [self.time_col, self.target_col, *self.panel_cols, *self.covariate_features]
        missing = [col for col in required if col not in value.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        data = value.sort_values([*self.panel_cols, self.time_col], kind="mergesort")
        if self.panel_cols:
            duplicate_mask = data.duplicated([*self.panel_cols, self.time_col], keep=False)
        else:
            duplicate_mask = data.duplicated([self.time_col], keep=False)
        if duplicate_mask.any():
            raise ValueError(
                "CartoBoostLagForecaster requires unique timestamps within each panel when "
                "coercing a DataFrame to the native ForecastFrame"
            )
        rows = []
        row_covariates = []
        for row in data.itertuples(index=False):
            row_values = dict(zip(data.columns, row, strict=True))
            if self.panel_cols:
                series_id = "|".join(str(row_values[col]) for col in self.panel_cols)
            else:
                series_id = "__single__"
            timestamp = pd.Timestamp(row_values[self.time_col]).strftime("%Y-%m-%dT%H:%M:%S")
            rows.append((series_id, timestamp, float(row_values[self.target_col])))
            row_covariates.append(
                {name: float(row_values[name]) for name in self.covariate_features}
            )
        native_frame_class = _native_class("ForecastFrame")
        if native_frame_class is None:
            raise NotImplementedError("Rust binding for ForecastFrame is not available.")
        return native_frame_class(
            rows,
            self.frequency,
            row_covariates=row_covariates if self.covariate_features else None,
            allow_irregular=self.allow_irregular,
        )

    def _native_future_frame_from_dataframe(self, value: Any) -> Any:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised only in minimal installs.
            raise ImportError(
                "CartoBoostLagForecaster known_future DataFrame input requires pandas."
            ) from exc
        if not isinstance(value, pd.DataFrame):
            raise TypeError("known_future must be a pandas DataFrame")
        required = [self.time_col, *self.panel_cols, *self.covariate_features]
        missing = [col for col in required if col not in value.columns]
        if missing:
            raise ValueError(f"missing required known_future columns: {missing}")
        data = value.sort_values([*self.panel_cols, self.time_col], kind="mergesort")
        rows = []
        row_covariates = []
        for row in data.itertuples(index=False):
            row_values = dict(zip(data.columns, row, strict=True))
            if self.panel_cols:
                series_id = "|".join(str(row_values[col]) for col in self.panel_cols)
            else:
                series_id = "__single__"
            timestamp = pd.Timestamp(row_values[self.time_col]).strftime("%Y-%m-%dT%H:%M:%S")
            rows.append((series_id, timestamp, 0.0))
            row_covariates.append(
                {name: float(row_values[name]) for name in self.covariate_features}
            )
        native_frame_class = _native_class("ForecastFrame")
        if native_frame_class is None:
            raise NotImplementedError("Rust binding for ForecastFrame is not available.")
        return native_frame_class(
            rows,
            self.frequency,
            row_covariates=row_covariates if self.covariate_features else None,
            allow_irregular=self.allow_irregular,
        )


def _native_splitters_for_policy(policy: SplitPolicy) -> list[str] | None:
    if policy is SplitPolicy.AXIS_ONLY:
        return ["axis"]
    if policy is SplitPolicy.STRUCTURED:
        # Lag forecasters expose generated calendar features rather than a
        # user-supplied FeatureSchema. Keep the structured policy explicit at
        # this native boundary: daily panels may use both the within-day
        # cycle and the weekly cycle, while sparse memberships remain opt-in.
        return ["axis", "periodic_time", "periodic:7", "sparse_set"]
    return None


__all__ = ["CartoBoostLagForecaster", "ForecastResult"]
