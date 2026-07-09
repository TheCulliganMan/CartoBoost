"""Prophet-shaped forecasting API backed by CartoBoost's Rust forecaster.

The compatibility class intentionally follows Prophet's dataframe contract
(``ds``/``y``) while keeping fitting and prediction in the native Rust core.
It accepts pandas and Polars dataframes and returns pandas dataframes, matching
Prophet's public Python ergonomics without importing the Prophet runtime.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .forecasting.frequency import require_pandas
from .forecasting.local import PiecewiseLinearSeasonalForecaster
from .forecasting.schema import ForecastFrame


class Prophet:
    """Prophet-compatible facade over CartoBoost's Rust-native model.

    The supported surface follows the high-value public Prophet workflow:
    ``fit``, ``predict``, ``make_future_dataframe``, ``add_seasonality``,
    ``add_regressor``, ``add_country_holidays``, ``predictive_samples``, and
    component/diagnostic attributes.  Unsupported probabilistic MCMC backends
    are rejected explicitly instead of silently changing the model.
    """

    def __init__(
        self,
        growth: str = "linear",
        changepoints: Any = None,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8,
        yearly_seasonality: Any = "auto",
        weekly_seasonality: Any = "auto",
        daily_seasonality: Any = "auto",
        holidays: Any | None = None,
        seasonality_mode: str = "additive",
        seasonality_prior_scale: float | None = 10.0,
        holidays_prior_scale: float | None = 10.0,
        changepoint_prior_scale: float = 0.05,
        mcmc_samples: int = 0,
        interval_width: float = 0.80,
        uncertainty_samples: int = 1000,
        stan_backend: Any | None = None,
        scaling: str = "absmax",
        holidays_mode: str | None = None,
    ) -> None:
        if int(mcmc_samples) != 0:
            raise ValueError("CartoBoost Prophet compatibility does not support mcmc_samples")
        if stan_backend is not None:
            raise ValueError("CartoBoost Prophet compatibility does not use stan_backend")
        if not 0.0 < float(interval_width) < 1.0:
            raise ValueError("interval_width must be between 0 and 1")
        self.growth = str(growth)
        self.changepoints = changepoints
        self.n_changepoints = int(n_changepoints)
        self.changepoint_range = float(changepoint_range)
        self.seasonality_mode = str(seasonality_mode)
        self.seasonality_prior_scale = (
            10.0 if seasonality_prior_scale is None else float(seasonality_prior_scale)
        )
        self.holidays_prior_scale = (
            10.0 if holidays_prior_scale is None else float(holidays_prior_scale)
        )
        self.changepoint_prior_scale = float(changepoint_prior_scale)
        self.interval_width = float(interval_width)
        self.uncertainty_samples = int(uncertainty_samples)
        if scaling not in {"absmax", "minmax"}:
            raise ValueError("scaling must be 'absmax' or 'minmax'")
        self.scaling = scaling
        self.holidays_mode = holidays_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.holidays = holidays
        self.extra_regressors: dict[str, dict[str, Any]] = {}
        self.seasonalities: dict[str, dict[str, Any]] = {}
        self.history: Any | None = None
        self.train_component_cols: Any | None = None
        self.changepoints_t: np.ndarray | None = None
        self._model: PiecewiseLinearSeasonalForecaster | None = None
        self._freq: str = "D"

    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: float | None = None,
        mode: str | None = None,
        condition_name: str | None = None,
    ) -> Prophet:
        if self.history is not None:
            raise RuntimeError("seasonalities must be added before fitting")
        self.seasonalities[str(name)] = {
            "name": str(name),
            "period": float(period),
            "fourier_order": int(fourier_order),
            "prior_scale": 10.0 if prior_scale is None else float(prior_scale),
            "mode": mode,
            "condition_name": condition_name,
        }
        return self

    def add_regressor(
        self,
        name: str,
        prior_scale: float | None = None,
        standardize: Any = "auto",
        mode: str | None = None,
    ) -> Prophet:
        if self.history is not None:
            raise RuntimeError("regressors must be added before fitting")
        if str(name) in {"ds", "y"}:
            raise ValueError("ds and y are reserved Prophet columns")
        self.extra_regressors[str(name)] = {
            "prior_scale": 10.0 if prior_scale is None else float(prior_scale),
            "standardize": standardize,
            "mode": mode,
        }
        return self

    def add_country_holidays(self, country_name: str) -> Prophet:
        if self.history is not None:
            raise RuntimeError("country holidays must be added before fitting")
        self._country_holidays = str(country_name)
        return self

    def fit(self, df: Any, **_: Any) -> Prophet:
        pd = require_pandas()
        frame = _to_pandas(df)
        _require_columns(frame, ["ds", "y"])
        frame = frame.copy()
        frame["ds"] = pd.to_datetime(frame["ds"], errors="raise")
        frame = frame.sort_values("ds", kind="mergesort").reset_index(drop=True)
        self._freq = pd.infer_freq(frame["ds"]) or "D"
        regressors = list(self.extra_regressors)
        _require_columns(frame, regressors)
        if self.changepoints is None:
            prophet_changepoints = _prophet_generated_changepoints(
                frame["ds"], self.n_changepoints, self.changepoint_range
            )
            changepoints = 0
            changepoint_timestamps = [
                timestamp.to_pydatetime().replace(tzinfo=None).isoformat()
                for timestamp in prophet_changepoints
            ]
            self.changepoints = prophet_changepoints
            self.n_changepoints = len(prophet_changepoints)
        else:
            changepoints = self.changepoints
            changepoint_timestamps = ()
        forecast_frame = ForecastFrame.from_pandas(
            frame,
            timestamp_col="ds",
            target_col="y",
            freq=self._freq,
            known_future_covariates=regressors,
            allow_irregular=True,
            allow_missing_targets=True,
        )
        self._model = PiecewiseLinearSeasonalForecaster(
            growth=self.growth,
            component_mode=self.seasonality_mode,
            holidays_mode=self.holidays_mode,
            changepoints=changepoints,
            changepoint_range=self.changepoint_range,
            changepoint_timestamps=changepoint_timestamps,
            yearly_fourier_order=_seasonality_order(self.yearly_seasonality, 10),
            weekly_fourier_order=_seasonality_order(self.weekly_seasonality, 3),
            daily_fourier_order=_seasonality_order(self.daily_seasonality, 4),
            auto_yearly_seasonality=self.yearly_seasonality == "auto",
            auto_weekly_seasonality=self.weekly_seasonality == "auto",
            auto_daily_seasonality=self.daily_seasonality == "auto",
            custom_seasonalities=[
                {
                    "name": value["name"],
                    "period_days": value["period"],
                    "fourier_order": value["fourier_order"],
                    "mode": value["mode"],
                    "condition_name": value["condition_name"],
                    "l2_regularization": value["prior_scale"],
                }
                for value in self.seasonalities.values()
            ],
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays=self.holidays,
            holidays_prior_scale=self.holidays_prior_scale,
            country_holidays=getattr(self, "_country_holidays", None),
            extra_regressors=regressors,
            regressor_modes={
                name: value["mode"]
                for name, value in self.extra_regressors.items()
                if value["mode"] is not None
            },
            regressor_standardization="auto",
            prediction_interval_levels=[self.interval_width],
            uncertainty_samples=self.uncertainty_samples,
        ).fit(forecast_frame)
        self.history = frame.reset_index(drop=True)
        self.start = self.history["ds"].iloc[0]
        self.t_scale = (self.history["ds"].iloc[-1] - self.start).total_seconds() / 86_400.0
        if self.t_scale <= 0.0:
            self.t_scale = 1.0
        self.y_scale = _prophet_y_scale(self.history["y"], self.scaling)
        self.history["t"] = (self.history["ds"] - self.start).dt.total_seconds() / (
            86_400.0 * self.t_scale
        )
        self.history["y_scaled"] = self.history["y"] / self.y_scale
        if isinstance(self.changepoints, int):
            changepoint_series = pd.Series(pd.to_datetime([]), name="ds")
        else:
            changepoint_series = pd.Series(pd.to_datetime(self.changepoints), name="ds")
        self.changepoints = changepoint_series.reset_index(drop=True)
        if len(self.changepoints):
            scale = (self.history["ds"].iloc[-1] - self.history["ds"].iloc[0]).total_seconds()
            self.changepoints_t = np.array(
                [
                    (timestamp - self.history["ds"].iloc[0]).total_seconds() / scale
                    for timestamp in self.changepoints
                ],
                dtype=float,
            )
        else:
            self.changepoints_t = np.array([0.0], dtype=float)
        return self

    def make_future_dataframe(
        self,
        periods: int,
        freq: str = "D",
        include_history: bool = True,
        **_: Any,
    ) -> Any:
        if self.history is None:
            raise RuntimeError("Prophet must be fitted before make_future_dataframe")
        pd = require_pandas()
        if int(periods) < 0:
            raise ValueError("periods must be nonnegative")
        if freq == "D" and self._freq != "D":
            freq = self._freq
        future = pd.date_range(
            start=self.history["ds"].iloc[-1], periods=int(periods) + 1, freq=freq
        )[1:]
        dates = pd.concat([self.history[["ds"]], pd.DataFrame({"ds": future})], ignore_index=True)
        return dates if include_history else dates.iloc[len(self.history) :].reset_index(drop=True)

    def predict(self, df: Any = None, vectorized: bool = True) -> Any:
        if self._model is None or self.history is None:
            raise RuntimeError("Prophet must be fitted before predict")
        if df is None:
            df = self.make_future_dataframe(0)
        frame = _to_pandas(df).copy()
        _require_columns(frame, ["ds", *self.extra_regressors])
        frame["ds"] = require_pandas().to_datetime(frame["ds"], errors="raise")
        last_history = self.history["ds"].max()
        future_mask = frame["ds"] > last_history
        future = frame.loc[future_mask]
        future_regressors = {
            name: future[name].to_numpy(dtype=float).tolist() for name in self.extra_regressors
        }
        result_records: list[dict[str, Any]] = []
        component_records: list[Mapping[str, Any]] = []
        if len(future):
            result = self._model.predict(
                len(future),
                future_timestamps=future["ds"].tolist(),
                future_regressors=future_regressors or None,
                prediction_interval_levels=[self.interval_width],
                uncertainty_samples=self.uncertainty_samples,
            )
            result_records.extend(json.loads(result.to_json())["records"])
            component_records.extend(
                self._model.components(len(future), future_regressors=future_regressors or None)[
                    "records"
                ]
            )
        history_records = (
            self._model.history_components()["records"] if bool((~future_mask).any()) else []
        )
        result_by_timestamp = {
            str(row["timestamp"]): row for row in [*history_records, *result_records]
        }
        components_by_timestamp = {
            str(row["timestamp"]): row for row in [*history_records, *component_records]
        }
        ordered_results = [result_by_timestamp[_timestamp_key(value)] for value in frame["ds"]]
        ordered_components = [
            components_by_timestamp[_timestamp_key(value)] for value in frame["ds"]
        ]
        return _prophet_result_frame(
            frame, ordered_results, ordered_components, self.interval_width
        )

    def predictive_samples(self, df: Any, vectorized: bool = True) -> dict[str, np.ndarray]:
        if self._model is None or self.history is None:
            raise RuntimeError("Prophet must be fitted before predictive_samples")
        frame = _to_pandas(df).copy()
        frame["ds"] = require_pandas().to_datetime(frame["ds"], errors="raise")
        future = frame.loc[frame["ds"] > self.history["ds"].max()]
        if not len(future):
            forecast = self.predict(frame)
            return {"yhat": forecast["yhat"].to_numpy()[:, None]}
        future_regressors = {
            name: future[name].to_numpy(dtype=float).tolist() for name in self.extra_regressors
        }
        samples = self._model.samples(len(future), future_regressors=future_regressors or None)
        sample_count = int(samples.get("sample_count", 0))
        if sample_count == 0:
            forecast = self.predict(frame)
            return {"yhat": forecast["yhat"].to_numpy()[:, None]}
        values = np.full((len(frame), sample_count), np.nan, dtype=float)
        for row in samples["records"]:
            horizon = int(row["horizon"]) - 1
            future_index = int(future.index[horizon])
            values[future_index, int(row["sample"])] = float(row["prediction"])
        history_prediction = self.predict(frame.loc[~frame["ds"].gt(self.history["ds"].max())])
        history_values = history_prediction["yhat"].to_numpy()
        values[: len(history_values), :] = history_values[:, None]
        return {"yhat": values}

    def setup_dataframe(self, df: Any, initialize_scales: bool = False) -> Any:
        frame = _to_pandas(df)
        _require_columns(frame, ["ds", *self.extra_regressors])
        return frame.copy()

    def predict_components(self, df: Any) -> Any:
        return self.predict(df)

    def plot(
        self,
        fcst: Any,
        ax: Any = None,
        uncertainty: bool = True,
        plot_cap: bool = True,
        xlabel: str = "ds",
        ylabel: str = "y",
        figsize: tuple[float, float] = (10, 6),
        include_legend: bool = False,
    ) -> Any:
        from .plotting import plot as plot_forecast

        return plot_forecast(
            self,
            fcst,
            ax=ax,
            uncertainty=uncertainty,
            plot_cap=plot_cap,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            include_legend=include_legend,
        )

    def plot_components(
        self,
        fcst: Any,
        uncertainty: bool = True,
        plot_cap: bool = True,
        weekly_start: int = 0,
        yearly_start: int = 0,
        figsize: tuple[float, float] | None = None,
    ) -> Any:
        from .plotting import plot_components

        return plot_components(
            self,
            fcst,
            uncertainty=uncertainty,
            plot_cap=plot_cap,
            weekly_start=weekly_start,
            yearly_start=yearly_start,
            figsize=figsize,
        )

    def validate_column_name(
        self,
        name: str,
        check_holidays: bool = True,
        check_seasonalities: bool = True,
        check_regressors: bool = True,
    ) -> None:
        reserved = {"ds", "y", "cap", "floor", "y_scaled", "t"}
        if name in reserved:
            raise ValueError(f"Name {name!r} is reserved.")
        if check_holidays and self.holidays is not None and name == "holiday":
            raise ValueError("Name 'holiday' is reserved for holiday features.")
        if check_seasonalities and name in self.seasonalities:
            raise ValueError(f"Name {name!r} already exists in seasonalities.")
        if check_regressors and name in self.extra_regressors:
            raise ValueError(f"Name {name!r} already exists in regressors.")

    def validate_inputs(self) -> None:
        if self.growth not in {"linear", "logistic", "flat"}:
            raise ValueError('Parameter "growth" should be "linear", "logistic" or "flat".')
        if (
            not isinstance(self.changepoint_range, (int, float))
            or not 0 <= self.changepoint_range <= 1
        ):
            raise ValueError('Parameter "changepoint_range" must be in [0, 1]')
        if self.seasonality_mode not in {"additive", "multiplicative"}:
            raise ValueError('seasonality_mode must be "additive" or "multiplicative"')
        if self.holidays_mode is not None and self.holidays_mode not in {
            "additive",
            "multiplicative",
        }:
            raise ValueError('holidays_mode must be "additive" or "multiplicative"')
        if self.holidays is not None:
            pd = require_pandas()
            if not isinstance(self.holidays, pd.DataFrame) or not {"ds", "holiday"}.issubset(
                self.holidays.columns
            ):
                raise ValueError('holidays must be a DataFrame with "ds" and "holiday" columns.')

    def initialize_scales(self, initialize_scales: bool, df: Any) -> None:
        if not initialize_scales:
            return None
        frame = _to_pandas(df)
        self.y_scale = _prophet_y_scale(frame["y"], self.scaling)
        self.start = frame["ds"].min()
        self.t_scale = frame["ds"].max() - self.start
        return None

    def calculate_initial_params(self, num_total_regressors: int) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            k=0.0,
            m=0.0,
            delta=np.zeros(0 if self.changepoints is None else len(self.changepoints), dtype=float),
            beta=np.zeros(int(num_total_regressors), dtype=float),
            sigma_obs=1.0,
        )

    @staticmethod
    def linear_growth_init(df: Any) -> tuple[float, float]:
        frame = _to_pandas(df)
        first, last = frame["ds"].idxmin(), frame["ds"].idxmax()
        scale = float(frame.loc[last, "t"] - frame.loc[first, "t"])
        if scale == 0.0:
            scale = 1.0
        k = (float(frame.loc[last, "y_scaled"]) - float(frame.loc[first, "y_scaled"])) / scale
        return k, float(frame.loc[first, "y_scaled"]) - k * float(frame.loc[first, "t"])

    @staticmethod
    def flat_growth_init(df: Any) -> tuple[float, float]:
        return 0.0, float(_to_pandas(df)["y_scaled"].mean())

    @staticmethod
    def logistic_growth_init(df: Any) -> tuple[float, float]:
        frame = _to_pandas(df)
        first, last = frame["ds"].idxmin(), frame["ds"].idxmax()
        scale = float(frame.loc[last, "t"] - frame.loc[first, "t"]) or 1.0
        c0, c1 = float(frame.loc[first, "cap_scaled"]), float(frame.loc[last, "cap_scaled"])
        y0 = max(0.01 * c0, min(0.99 * c0, float(frame.loc[first, "y_scaled"])))
        y1 = max(0.01 * c1, min(0.99 * c1, float(frame.loc[last, "y_scaled"])))
        l0, l1 = np.log(c0 / y0 - 1.0), np.log(c1 / y1 - 1.0)
        if abs(c0 / y0 - c1 / y1) <= 0.01:
            l0 = np.log(1.05 * c0 / y0 - 1.0)
        return (l0 - l1) / scale, l0 * scale / (l0 - l1)

    @staticmethod
    def flat_trend(t: Any, m: float) -> np.ndarray:
        return float(m) * np.ones_like(t)

    @staticmethod
    def piecewise_linear(
        t: Any, deltas: Any, k: float, m: float, changepoint_ts: Any
    ) -> np.ndarray:
        t = np.asarray(t)
        cp = np.asarray(changepoint_ts)
        delta_t = (cp[None, :] <= t[..., None]) * np.asarray(deltas)
        return (delta_t.sum(axis=1) + k) * t + (delta_t * -cp).sum(axis=1) + m

    @staticmethod
    def piecewise_logistic(
        t: Any, cap: Any, deltas: Any, k: float, m: float, changepoint_ts: Any
    ) -> np.ndarray:
        t, cap, deltas, cp = map(np.asarray, (t, cap, deltas, changepoint_ts))
        k_cum = np.concatenate((np.atleast_1d(k), np.cumsum(deltas) + k))
        gammas = np.zeros(len(cp))
        for index, timestamp in enumerate(cp):
            gammas[index] = (timestamp - m - gammas[:index].sum()) * (
                1 - k_cum[index] / k_cum[index + 1]
            )
        k_t, m_t = k * np.ones_like(t), m * np.ones_like(t)
        for index, timestamp in enumerate(cp):
            active = t >= timestamp
            k_t[active] += deltas[index]
            m_t[active] += gammas[index]
        return cap / (1 + np.exp(-k_t * (t - m_t)))

    def parse_seasonality_args(
        self, name: str, arg: Any, auto_disable: bool, default_order: int
    ) -> int:
        if arg == "auto":
            return 0 if auto_disable or name in self.seasonalities else int(default_order)
        if arg is True:
            return int(default_order)
        if arg is False:
            return 0
        return int(arg)

    def regressor_column_matrix(self, seasonal_features: Any, modes: Any) -> Any:
        return _to_pandas(seasonal_features).copy()

    def construct_holiday_dataframe(self, dates: Any) -> Any:
        return self.make_holiday_features(dates, self.holidays)[0]

    def make_holiday_features(self, dates: Any, holidays: Any) -> tuple[Any, list[str]]:
        pd = require_pandas()
        dates = pd.to_datetime(dates)
        if holidays is None:
            return pd.DataFrame(index=range(len(dates))), []
        frame = holidays.copy()
        frame["ds"] = pd.to_datetime(frame["ds"])
        result = pd.DataFrame(index=range(len(dates)))
        names = []
        for name, group in frame.groupby("holiday", sort=False):
            values = np.zeros(len(dates), dtype=float)
            for timestamp in group["ds"]:
                values += (dates == timestamp).to_numpy(dtype=float)
            result[str(name)] = values
            names.append(str(name))
        return result, names

    def set_changepoints(self) -> None:
        """Refresh generated changepoints using Prophet's exact index rule."""
        if self.history is None or not isinstance(self.changepoints, type(None)):
            return None
        self.changepoints = _prophet_generated_changepoints(
            self.history["ds"], self.n_changepoints, self.changepoint_range
        )
        return None

    def set_auto_seasonalities(self) -> None:
        return None

    def add_group_component(self, components: list[str], name: str, group: str) -> None:
        if not hasattr(self, "component_modes") or self.component_modes is None:
            self.component_modes = {}
        self.component_modes[str(name)] = {
            "components": list(components),
            "group": str(group),
        }

    def predict_trend(self, df: Any) -> Any:
        return self.predict(df)[["ds", "trend"]]

    def predict_seasonal_components(self, df: Any) -> Any:
        forecast = self.predict(df)
        return forecast[[column for column in forecast.columns if column not in {"ds", "yhat"}]]

    def predict_uncertainty(self, df: Any, vectorized: bool = True) -> Any:
        return self.predict(df)[["ds", "yhat_lower", "yhat_upper"]]

    def preprocess(self, df: Any, **_: Any) -> Any:
        return self.setup_dataframe(df, initialize_scales=True)

    @staticmethod
    def fourier_series(dates: Any, period: float, series_order: int) -> np.ndarray:
        pd = require_pandas()
        normalized = pd.to_datetime(dates)
        delta = normalized - normalized.min()
        elapsed = (
            delta.dt.total_seconds() if hasattr(delta, "dt") else delta.total_seconds()
        ) / 86_400.0
        columns = []
        for order in range(1, int(series_order) + 1):
            angle = 2.0 * np.pi * order * np.asarray(elapsed, dtype=float) / float(period)
            columns.extend([np.sin(angle), np.cos(angle)])
        return np.column_stack(columns) if columns else np.empty((len(pd.to_datetime(dates)), 0))

    @staticmethod
    def make_seasonality_features(dates: Any, period: float, series_order: int, prefix: str) -> Any:
        pd = require_pandas()
        values = Prophet.fourier_series(dates, period, series_order)
        columns = [f"{prefix}_delim_{idx}" for idx in range(values.shape[1])]
        return pd.DataFrame(values, columns=columns, index=getattr(dates, "index", None))

    def make_all_seasonality_features(self, df: Any) -> Any:
        pd = require_pandas()
        frame = _to_pandas(df)
        features = []
        for name, period, order in (
            ("yearly", 365.25, _seasonality_order(self.yearly_seasonality, 10)),
            ("weekly", 7.0, _seasonality_order(self.weekly_seasonality, 3)),
            ("daily", 1.0, _seasonality_order(self.daily_seasonality, 4)),
        ):
            if order:
                features.append(self.make_seasonality_features(frame["ds"], period, order, name))
        for name, config in self.seasonalities.items():
            features.append(
                self.make_seasonality_features(
                    frame["ds"], config["period"], config["fourier_order"], name
                )
            )
        return pd.concat(features, axis=1) if features else pd.DataFrame(index=frame.index)


def _to_pandas(frame: Any) -> Any:
    pd = require_pandas()
    if isinstance(frame, pd.DataFrame):
        return frame
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    if hasattr(frame, "to_dicts"):
        return pd.DataFrame(frame.to_dicts())
    raise TypeError("Prophet-compatible input must be a pandas or Polars dataframe")


def _require_columns(frame: Any, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"dataframe is missing required columns: {missing}")


def _seasonality_order(value: Any, default: int) -> int:
    if value in (False, None):
        return 0
    if value in (True, "auto"):
        return default if value is True else 0
    return int(value)


def _prophet_y_scale(values: Any, scaling: str) -> float:
    numeric = np.asarray(values, dtype=float)
    if scaling == "minmax":
        scale = float(np.nanmax(numeric) - np.nanmin(numeric))
    else:
        scale = float(np.nanmax(np.abs(numeric)))
    return scale if np.isfinite(scale) and scale > 0.0 else 1.0


def _prophet_generated_changepoints(
    timestamps: Any, requested: int, changepoint_range: float
) -> Any:
    """Reproduce Prophet 1.2.2's generated changepoint index selection."""
    pd = require_pandas()
    if not 0.0 < float(changepoint_range) <= 1.0:
        raise ValueError("changepoint_range must be between 0 and 1")
    hist_size = int(np.floor(len(timestamps) * float(changepoint_range)))
    effective = min(int(requested), hist_size - 1)
    if effective <= 0:
        return pd.Series(pd.to_datetime([]), name="ds")
    cp_indexes = np.linspace(0, hist_size - 1, effective + 1).round().astype(int)[1:]
    return timestamps.iloc[cp_indexes].reset_index(drop=True)


def _prophet_result_frame(
    frame: Any,
    result: list[Mapping[str, Any]],
    components: list[Mapping[str, Any]],
    interval: float,
) -> Any:
    pd = require_pandas()
    output = pd.DataFrame(
        {
            "ds": frame["ds"].to_numpy(),
            "yhat": [row.get("prediction", row.get("fitted")) for row in result],
        }
    )
    suffix = int(round(interval * 100))
    lower = f"prediction_lower_p{suffix}"
    upper = f"prediction_upper_p{suffix}"
    output["yhat_lower"] = [row.get(lower, np.nan) for row in result]
    output["yhat_upper"] = [row.get(upper, np.nan) for row in result]
    for row, record in enumerate(components):
        component_values = record.get("components", {})
        output.loc[row, "trend"] = component_values.get("trend_linear", np.nan)
        output.loc[row, "additive_terms"] = component_values.get("non_trend_total", 0.0)
        for name, value in component_values.get("regressors", {}).items():
            output.loc[row, name] = value
    return output


def _timestamp_key(value: Any) -> str:
    return require_pandas().Timestamp(value).to_pydatetime().replace(tzinfo=None).isoformat()


__all__ = ["Prophet"]
