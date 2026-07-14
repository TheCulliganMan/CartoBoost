"""Unified Rust-backed forecasting API for CartoBoost."""

from .. import _forecasting_catalog as _catalog
from .auto import AutoForecastConfig, AutoForecaster
from .backtesting import BacktestFoldResult, BacktestResult, RollingOriginBacktester
from .global_models import CartoBoostLagForecaster
from .lag_features import LagConfig
from .local import NaiveForecaster, SeasonalNaiveForecaster
from .metrics import ForecastMetricSet
from .schema import ForecastFrame, ForecastResult
from .splitters import ForecastFold, RollingOriginSplitter

__all__ = [
    "AutoForecastConfig",
    "AutoForecaster",
    "BacktestFoldResult",
    "BacktestResult",
    "CartoBoostLagForecaster",
    "ForecastFold",
    "ForecastFrame",
    "ForecastMetricSet",
    "ForecastResult",
    "LagConfig",
    "NaiveForecaster",
    "RollingOriginBacktester",
    "RollingOriginSplitter",
    "SeasonalNaiveForecaster",
    *_catalog.__all__,
]


def __getattr__(name: str):
    return getattr(_catalog, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
