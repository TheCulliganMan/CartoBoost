"""General Rust-backed numerical utilities."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .config import Drift, Kernel


def series_forecast(
    model: str, values: Sequence[float], horizon: int, **params: Any
) -> list[float]:
    """Forecast a numeric sequence with a Rust-backed single-series method.

    This is a general utility independent of `cartoboost.forecasting` frames.
    Supported native model names include `naive`, `seasonal_naive`, `theta`,
    `optimized_theta`, `ets`, `arima`, `auto_arima`, `local_level_kalman`, and
    `local_linear_trend_kalman`.
    """

    native = _native()
    return list(
        native.utility_series_forecast(
            str(model),
            [float(value) for value in values],
            int(horizon),
            json.dumps(params) if params else None,
        )
    )


def naive_forecast(values: Sequence[float], horizon: int) -> list[float]:
    """Forecast by repeating the final observed value."""

    return series_forecast("naive", values, horizon)


def seasonal_naive_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    season_length: int,
) -> list[float]:
    """Forecast by repeating the last observed seasonal cycle."""

    return series_forecast(
        "seasonal_naive",
        values,
        horizon,
        season_length=int(season_length),
    )


def theta_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    theta: float = 2.0,
    alpha: float = 0.5,
) -> list[float]:
    """Forecast with the Rust theta model."""

    return series_forecast("theta", values, horizon, theta=float(theta), alpha=float(alpha))


def optimized_theta_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    theta_grid: Sequence[float] = (1.0, 2.0),
    alpha_grid: Sequence[float] = (0.2, 0.5, 0.8),
) -> list[float]:
    """Forecast with deterministic Rust optimized-theta grid selection."""

    return series_forecast(
        "optimized_theta",
        values,
        horizon,
        theta_grid=[float(value) for value in theta_grid],
        alpha_grid=[float(value) for value in alpha_grid],
    )


def ets_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    alpha: float = 0.5,
    beta: float = 0.1,
    gamma: float | None = None,
    season_length: int | None = None,
) -> list[float]:
    """Forecast with the Rust additive ETS utility."""

    return series_forecast(
        "ets",
        values,
        horizon,
        alpha=float(alpha),
        beta=float(beta),
        gamma=None if gamma is None else float(gamma),
        season_length=None if season_length is None else int(season_length),
    )


def arima_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    p: int = 1,
    d: int = 0,
    q: int = 0,
) -> list[float]:
    """Forecast with the Rust ARIMA(p,d,q) utility."""

    return series_forecast("arima", values, horizon, p=int(p), d=int(d), q=int(q))


def auto_arima_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    max_p: int = 3,
    max_d: int = 1,
    max_q: int = 2,
) -> list[float]:
    """Forecast with deterministic Rust AutoARIMA candidate selection."""

    return series_forecast(
        "auto_arima",
        values,
        horizon,
        max_p=int(max_p),
        max_d=int(max_d),
        max_q=int(max_q),
    )


def local_linear_trend_kalman_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    level_process_variance: float = 0.05,
    trend_process_variance: float = 0.005,
    observation_variance: float = 1.0,
) -> list[float]:
    """Forecast with the Rust local-linear-trend Kalman state-space utility."""

    return series_forecast(
        "local_linear_trend_kalman",
        values,
        horizon,
        level_process_variance=float(level_process_variance),
        trend_process_variance=float(trend_process_variance),
        observation_variance=float(observation_variance),
    )


def local_level_kalman_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    level_process_variance: float = 0.05,
    observation_variance: float = 1.0,
) -> list[float]:
    """Forecast with the Rust local-level Kalman utility."""

    return series_forecast(
        "local_level_kalman",
        values,
        horizon,
        level_process_variance=float(level_process_variance),
        observation_variance=float(observation_variance),
    )


def local_level_kalman_filter(
    values: Sequence[float],
    *,
    level_process_variance: float = 0.05,
    observation_variance: float = 1.0,
    horizon: int = 0,
    interval_z: float = 1.959963984540054,
) -> dict[str, Any]:
    """Run the Rust local-level Kalman filter on a numeric series.

    When `horizon` is positive, the result includes `forecast_distribution`
    rows with mean, variance, and normal-approximation bounds using
    `interval_z`.
    """

    native = _native()
    payload = native.utility_local_level_kalman_filter(
        [float(value) for value in values],
        float(level_process_variance),
        float(observation_variance),
        int(horizon),
        float(interval_z),
    )
    return json.loads(payload)


def intermittent_demand_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    method: str = "croston",
    alpha: float = 0.1,
    beta: float = 0.1,
) -> list[float]:
    """Forecast non-negative intermittent demand with Croston, SBA, or TSB."""

    native = _native()
    return list(
        native.utility_intermittent_demand_forecast(
            [float(value) for value in values],
            int(horizon),
            str(method),
            float(alpha),
            float(beta),
        )
    )


def croston_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    alpha: float = 0.1,
) -> list[float]:
    """Forecast intermittent demand with Croston's method."""

    return intermittent_demand_forecast(values, horizon, method="croston", alpha=alpha)


def sba_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    alpha: float = 0.1,
) -> list[float]:
    """Forecast intermittent demand with the Syntetos-Boylan approximation."""

    return intermittent_demand_forecast(values, horizon, method="sba", alpha=alpha)


def tsb_forecast(
    values: Sequence[float],
    horizon: int,
    *,
    alpha: float = 0.1,
    beta: float = 0.1,
) -> list[float]:
    """Forecast intermittent demand with the Teunter-Syntetos-Babai method."""

    return intermittent_demand_forecast(
        values,
        horizon,
        method="tsb",
        alpha=alpha,
        beta=beta,
    )


def kalman_filter(
    values: Sequence[float],
    *,
    level_process_variance: float = 0.05,
    trend_process_variance: float = 0.005,
    observation_variance: float = 1.0,
    horizon: int = 0,
    interval_z: float = 1.959963984540054,
) -> dict[str, Any]:
    """Run the Rust local-linear Kalman filter on a numeric series.

    This is a general utility, independent of `cartoboost.forecasting`.
    When `horizon` is positive, the result includes `forecast_distribution`
    rows with mean, variance, and normal-approximation bounds using
    `interval_z`.
    """

    native = _native()
    payload = native.utility_kalman_filter(
        [float(value) for value in values],
        float(level_process_variance),
        float(trend_process_variance),
        float(observation_variance),
        int(horizon),
        float(interval_z),
    )
    return json.loads(payload)


def ordinary_kriging_predict(
    observations: Sequence[tuple[float, float, float]],
    targets: Sequence[tuple[float, float]],
    *,
    range: float = 1.0,
    nugget: float = 1.0e-6,
    sill: float = 1.0,
    variogram_model: Kernel = Kernel.EXPONENTIAL,
    drift: Drift = Drift.ORDINARY,
    anisotropy_angle_degrees: float = 0.0,
    anisotropy_scaling: float = 1.0,
    max_neighbors: int | None = None,
    min_neighbors: int = 1,
    max_distance: float | None = None,
    detailed: bool = False,
) -> list[dict[str, Any]]:
    """Predict target coordinates with Rust ordinary kriging.

    `observations` are `(x, y, value)` triples. `targets` are `(x, y)` pairs.
    This utility is independent of `cartoboost.forecasting`.
    """

    native = _native()
    obs = [(float(x), float(y), float(value)) for x, y, value in observations]
    target_rows = [(float(x), float(y)) for x, y in targets]
    uses_advanced_config = (
        float(sill) != 1.0
        or variogram_model != Kernel.EXPONENTIAL
        or drift != Drift.ORDINARY
        or float(anisotropy_angle_degrees) != 0.0
        or float(anisotropy_scaling) != 1.0
        or max_neighbors is not None
        or int(min_neighbors) != 1
        or max_distance is not None
    )
    if detailed or uses_advanced_config:
        rows = native.utility_ordinary_kriging_predict_detailed(
            obs,
            target_rows,
            float(range),
            float(nugget),
            float(sill),
            str(variogram_model),
            str(drift),
            float(anisotropy_angle_degrees),
            float(anisotropy_scaling),
            None if max_neighbors is None else int(max_neighbors),
            int(min_neighbors),
            None if max_distance is None else float(max_distance),
        )
        detailed_rows = [
            {
                "x": x,
                "y": y,
                "mean": mean,
                "variance": variance,
                "weights": list(weights),
                "neighbor_indices": list(neighbor_indices),
            }
            for x, y, mean, variance, weights, neighbor_indices in rows
        ]
        if detailed:
            return detailed_rows
        return [
            {"x": row["x"], "y": row["y"], "mean": row["mean"], "weights": row["weights"]}
            for row in detailed_rows
        ]
    rows = native.utility_ordinary_kriging_predict(
        obs,
        target_rows,
        float(range),
        float(nugget),
    )
    return [
        {"x": x, "y": y, "mean": mean, "weights": list(weights)} for x, y, mean, weights in rows
    ]


def ordinary_kriging_leave_one_out(
    observations: Sequence[tuple[float, float, float]],
    *,
    range: float = 1.0,
    nugget: float = 1.0e-6,
    sill: float = 1.0,
    variogram_model: Kernel = Kernel.EXPONENTIAL,
    drift: Drift = Drift.ORDINARY,
    anisotropy_angle_degrees: float = 0.0,
    anisotropy_scaling: float = 1.0,
    max_neighbors: int | None = None,
    min_neighbors: int = 1,
    max_distance: float | None = None,
) -> list[dict[str, Any]]:
    """Run Rust leave-one-out kriging diagnostics for observed coordinates."""

    native = _native()
    rows = native.utility_ordinary_kriging_leave_one_out(
        [(float(x), float(y), float(value)) for x, y, value in observations],
        float(range),
        float(nugget),
        float(sill),
        str(variogram_model),
        str(drift),
        float(anisotropy_angle_degrees),
        float(anisotropy_scaling),
        None if max_neighbors is None else int(max_neighbors),
        int(min_neighbors),
        None if max_distance is None else float(max_distance),
    )
    return [
        {
            "x": x,
            "y": y,
            "mean": mean,
            "variance": variance,
            "weights": list(weights),
            "neighbor_indices": list(neighbor_indices),
        }
        for x, y, mean, variance, weights, neighbor_indices in rows
    ]


def empirical_variogram(
    observations: Sequence[tuple[float, float, float]],
    *,
    bin_count: int = 10,
    max_distance: float | None = None,
    anisotropy_angle_degrees: float = 0.0,
    anisotropy_scaling: float = 1.0,
) -> list[dict[str, Any]]:
    """Compute a Rust binned empirical semivariogram."""

    native = _native()
    payload = native.utility_empirical_variogram(
        [(float(x), float(y), float(value)) for x, y, value in observations],
        int(bin_count),
        None if max_distance is None else float(max_distance),
        float(anisotropy_angle_degrees),
        float(anisotropy_scaling),
    )
    return list(json.loads(payload)["bins"])


def fit_ordinary_kriging_variogram(
    observations: Sequence[tuple[float, float, float]],
    *,
    variogram_models: Sequence[str] | None = None,
    range_candidates: Sequence[float] | None = None,
    nugget_candidates: Sequence[float] | None = None,
    sill_candidates: Sequence[float] | None = None,
    bin_count: int = 10,
    anisotropy_angle_degrees: float = 0.0,
    anisotropy_scaling: float = 1.0,
) -> dict[str, Any]:
    """Fit a kriging variogram by weighted least squares over candidate grids."""

    native = _native()
    payload = native.utility_fit_ordinary_kriging_variogram(
        [(float(x), float(y), float(value)) for x, y, value in observations],
        None if variogram_models is None else [str(model) for model in variogram_models],
        None if range_candidates is None else [float(value) for value in range_candidates],
        None if nugget_candidates is None else [float(value) for value in nugget_candidates],
        None if sill_candidates is None else [float(value) for value in sill_candidates],
        int(bin_count),
        float(anisotropy_angle_degrees),
        float(anisotropy_scaling),
    )
    return dict(json.loads(payload))


def ordinary_kriging_leave_one_out_diagnostics(
    observations: Sequence[tuple[float, float, float]],
    *,
    range: float = 1.0,
    nugget: float = 1.0e-6,
    sill: float = 1.0,
    variogram_model: Kernel = Kernel.EXPONENTIAL,
    drift: Drift = Drift.ORDINARY,
    anisotropy_angle_degrees: float = 0.0,
    anisotropy_scaling: float = 1.0,
    max_neighbors: int | None = None,
    min_neighbors: int = 1,
    max_distance: float | None = None,
) -> dict[str, Any]:
    """Run leave-one-out kriging and return predictions plus residual diagnostics."""

    native = _native()
    payload = native.utility_ordinary_kriging_leave_one_out_diagnostics(
        [(float(x), float(y), float(value)) for x, y, value in observations],
        float(range),
        float(nugget),
        float(sill),
        str(variogram_model),
        str(drift),
        float(anisotropy_angle_degrees),
        float(anisotropy_scaling),
        None if max_neighbors is None else int(max_neighbors),
        int(min_neighbors),
        None if max_distance is None else float(max_distance),
    )
    return dict(json.loads(payload))


def _native() -> Any:
    try:
        from cartoboost import _native as native
    except ImportError as exc:
        raise NotImplementedError("cartoboost._native is required for utilities") from exc
    return native


__all__ = [
    "arima_forecast",
    "auto_arima_forecast",
    "empirical_variogram",
    "ets_forecast",
    "fit_ordinary_kriging_variogram",
    "kalman_filter",
    "croston_forecast",
    "local_level_kalman_forecast",
    "local_level_kalman_filter",
    "local_linear_trend_kalman_forecast",
    "intermittent_demand_forecast",
    "naive_forecast",
    "optimized_theta_forecast",
    "ordinary_kriging_predict",
    "ordinary_kriging_leave_one_out",
    "ordinary_kriging_leave_one_out_diagnostics",
    "seasonal_naive_forecast",
    "sba_forecast",
    "series_forecast",
    "theta_forecast",
    "tsb_forecast",
]
