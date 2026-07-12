"""Lazy implementation map for :mod:`cartoboost.preview.forecasting`.

This module is private; users reach it through the preview proxy so importing
the stable forecasting package never imports optional model families.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    # Stable forecasting objects are also visible from the preview namespace.
    "AutoForecastConfig": ("cartoboost.forecasting.auto", "AutoForecastConfig"),
    "AutoForecaster": ("cartoboost.forecasting.auto", "AutoForecaster"),
    "CartoBoostLagForecaster": (
        "cartoboost.forecasting.global_models",
        "CartoBoostLagForecaster",
    ),
    "ForecastFrame": ("cartoboost.forecasting.schema", "ForecastFrame"),
    "ForecastResult": ("cartoboost.forecasting.schema", "ForecastResult"),
    "ForecastMetricSet": ("cartoboost.forecasting.metrics", "ForecastMetricSet"),
    "LagConfig": ("cartoboost.forecasting.lag_features", "LagConfig"),
    "NaiveForecaster": ("cartoboost.forecasting.local", "NaiveForecaster"),
    "RollingOriginBacktester": (
        "cartoboost.forecasting.backtesting",
        "RollingOriginBacktester",
    ),
    "RollingOriginSplitter": ("cartoboost.forecasting.splitters", "RollingOriginSplitter"),
    "SeasonalNaiveForecaster": (
        "cartoboost.forecasting.local",
        "SeasonalNaiveForecaster",
    ),
    # Forecasting artifacts and shared utilities.
    "ForecastArtifact": ("cartoboost.forecasting.artifacts", "ForecastArtifact"),
    "ForecastArtifactManifest": (
        "cartoboost.forecasting.artifacts",
        "ForecastArtifactManifest",
    ),
    "BacktestFoldResult": ("cartoboost.forecasting.backtesting", "BacktestFoldResult"),
    "BacktestResult": ("cartoboost.forecasting.backtesting", "BacktestResult"),
    "BaseForecaster": ("cartoboost.forecasting.base", "BaseForecaster"),
    "PanelForecasterMixin": ("cartoboost.forecasting.base", "PanelForecasterMixin"),
    "SingleSeriesForecasterMixin": (
        "cartoboost.forecasting.base",
        "SingleSeriesForecasterMixin",
    ),
    "ForecastingConfig": ("cartoboost.forecasting.config", "ForecastingConfig"),
    "ForecastFold": ("cartoboost.forecasting.splitters", "ForecastFold"),
    "ExpandingWindowSplitter": (
        "cartoboost.forecasting.splitters",
        "ExpandingWindowSplitter",
    ),
    "SlidingWindowSplitter": ("cartoboost.forecasting.splitters", "SlidingWindowSplitter"),
    "WeightedEnsembleForecaster": (
        "cartoboost.forecasting.ensemble",
        "WeightedEnsembleForecaster",
    ),
    "infer_frequency": ("cartoboost.forecasting.frequency", "infer_frequency"),
    "next_timestamps": ("cartoboost.forecasting.frequency", "next_timestamps"),
    "normalize_frequency": ("cartoboost.forecasting.frequency", "normalize_frequency"),
    "validate_horizon": ("cartoboost.forecasting.frequency", "validate_horizon"),
    "validate_regular_frequency": (
        "cartoboost.forecasting.frequency",
        "validate_regular_frequency",
    ),
    "CalendarFeatureConfig": ("cartoboost.forecasting.lag_features", "CalendarFeatureConfig"),
    "LagFeatureBuilder": ("cartoboost.forecasting.lag_features", "LagFeatureBuilder"),
    "LagFeatureConfig": ("cartoboost.forecasting.lag_features", "LagFeatureConfig"),
    "RollingFeatureConfig": ("cartoboost.forecasting.lag_features", "RollingFeatureConfig"),
    "PredictionInterval": ("cartoboost.forecasting.schema", "PredictionInterval"),
    # Local and panel preview models.
    "AutoARIMAForecaster": ("cartoboost.forecasting.local", "AutoARIMAForecaster"),
    "ArimaForecaster": ("cartoboost.forecasting.local", "ArimaForecaster"),
    "AutoKalmanForecaster": ("cartoboost.forecasting.local", "AutoKalmanForecaster"),
    "AutoLocalLevelKalmanForecaster": (
        "cartoboost.forecasting.local",
        "AutoLocalLevelKalmanForecaster",
    ),
    "AutoStatsBank": ("cartoboost.forecasting.local", "AutoStatsBank"),
    "CrostonForecaster": ("cartoboost.forecasting.local", "CrostonForecaster"),
    "ETSForecaster": ("cartoboost.forecasting.local", "ETSForecaster"),
    "KalmanForecaster": ("cartoboost.forecasting.local", "KalmanForecaster"),
    "KrigingForecaster": ("cartoboost.forecasting.local", "KrigingForecaster"),
    "LocalLevelKalmanForecaster": (
        "cartoboost.forecasting.local",
        "LocalLevelKalmanForecaster",
    ),
    "OptimizedThetaForecaster": (
        "cartoboost.forecasting.local",
        "OptimizedThetaForecaster",
    ),
    "PiecewiseLinearSeasonalForecaster": (
        "cartoboost.forecasting.local",
        "PiecewiseLinearSeasonalForecaster",
    ),
    "SbaForecaster": ("cartoboost.forecasting.local", "SbaForecaster"),
    "SpatialPiecewiseKrigingForecaster": (
        "cartoboost.forecasting.local",
        "SpatialPiecewiseKrigingForecaster",
    ),
    "ThetaForecaster": ("cartoboost.forecasting.local", "ThetaForecaster"),
    "TsbForecaster": ("cartoboost.forecasting.local", "TsbForecaster"),
    "DCRNNForecaster": ("cartoboost.forecasting.graph_st", "DCRNNForecaster"),
    "GraphTemporalFrame": ("cartoboost.forecasting.graph_st", "GraphTemporalFrame"),
    "GraphWaveNetForecaster": ("cartoboost.forecasting.graph_st", "GraphWaveNetForecaster"),
    "STAEformerForecaster": ("cartoboost.forecasting.graph_st", "STAEformerForecaster"),
    "available_graph_st_backends": (
        "cartoboost.forecasting.graph_st",
        "available_graph_st_backends",
    ),
    "LaneNeuralPanelForecaster": (
        "cartoboost.forecasting.neural",
        "LaneNeuralPanelForecaster",
    ),
    "NBEATSForecaster": ("cartoboost.forecasting.neural", "NBEATSForecaster"),
    "NBeatsForecaster": ("cartoboost.forecasting.neural", "NBeatsForecaster"),
    "NeuralPanelForecaster": ("cartoboost.forecasting.neural", "NeuralPanelForecaster"),
    "NHITSForecaster": ("cartoboost.forecasting.neural", "NHITSForecaster"),
    "NHiTSForecaster": ("cartoboost.forecasting.neural", "NHiTSForecaster"),
    # Preview probabilistic, registry, and sequence surfaces.
    "ConformalCalibrator": ("cartoboost.forecasting.probabilistic", "ConformalCalibrator"),
    "ConformalInterval": ("cartoboost.forecasting.probabilistic", "ConformalInterval"),
    "ConformalIntervalRegressor": (
        "cartoboost.forecasting.probabilistic",
        "ConformalIntervalRegressor",
    ),
    "DistributionalForecastResult": (
        "cartoboost.forecasting.probabilistic",
        "DistributionalForecastResult",
    ),
    "ForecastConformalCalibrator": (
        "cartoboost.forecasting.probabilistic",
        "ForecastConformalCalibrator",
    ),
    "QuantileCartoBoostRegressor": (
        "cartoboost.forecasting.probabilistic",
        "QuantileCartoBoostRegressor",
    ),
    "QuantileForecast": ("cartoboost.forecasting.probabilistic", "QuantileForecast"),
    "SpatialConformalRegressor": (
        "cartoboost.forecasting.probabilistic",
        "SpatialConformalRegressor",
    ),
    "interval_coverage": ("cartoboost.forecasting.probabilistic", "interval_coverage"),
    "mean_interval_width": (
        "cartoboost.forecasting.probabilistic",
        "mean_interval_width",
    ),
    "ForecastModelSpec": ("cartoboost.forecasting.registry", "ForecastModelSpec"),
    "ForecastRegistry": ("cartoboost.forecasting.registry", "ForecastRegistry"),
    "ReferencePathConfig": ("cartoboost.forecasting.sequence", "ReferencePathConfig"),
    "ReferenceSignal": ("cartoboost.forecasting.sequence", "ReferenceSignal"),
    "SequenceRow": ("cartoboost.forecasting.sequence", "SequenceRow"),
    "SequenceSeries": ("cartoboost.forecasting.sequence", "SequenceSeries"),
    "SequenceStateSpaceConfig": (
        "cartoboost.forecasting.sequence",
        "SequenceStateSpaceConfig",
    ),
    "forward_ekf": ("cartoboost.forecasting.sequence", "forward_ekf"),
    "generate_group_oof_candidate_rows": (
        "cartoboost.forecasting.sequence",
        "generate_group_oof_candidate_rows",
    ),
    "missing_target_continuation": (
        "cartoboost.forecasting.sequence",
        "missing_target_continuation",
    ),
    "per_group_error_summary": ("cartoboost.forecasting.sequence", "per_group_error_summary"),
    "reference_path_posterior_mean": (
        "cartoboost.forecasting.sequence",
        "reference_path_posterior_mean",
    ),
    "reference_path_viterbi": ("cartoboost.forecasting.sequence", "reference_path_viterbi"),
    "rts_smoother": ("cartoboost.forecasting.sequence", "rts_smoother"),
    "sequence_blend": ("cartoboost.forecasting.sequence", "sequence_blend"),
    "ukf_reference": ("cartoboost.forecasting.sequence", "ukf_reference"),
    "validate_oof_meta_training": (
        "cartoboost.forecasting.sequence",
        "validate_oof_meta_training",
    ),
    "validate_sequence_frame": (
        "cartoboost.forecasting.sequence",
        "validate_sequence_frame",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
