"""Stable Rust-backed forecasting API for CartoBoost 0.3.

Advanced local, neural, graph, spatial, probabilistic, and compatibility
forecasters remain available through the lazy ``cartoboost.preview.forecasting``
namespace. They are intentionally absent from this module's stable surface.
"""

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
]
