"""Plotting helpers for CartoBoost predictions, metrics, and forecast artifacts.

The functions in this module render already-computed model outputs. They do
not fit models, score benchmarks, or change forecasting behavior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PALETTE = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#be123c",
    "#4b5563",
)


def plot_predicted_actual(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    title: str = "Predicted vs actual",
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    ax: Any | None = None,
    point_color: str = DEFAULT_PALETTE[0],
    reference_color: str = "#111827",
) -> Any:
    """Render a predicted-vs-actual scatter plot with a parity reference line."""

    pyplot = _require_pyplot()
    actual, predicted = _paired_numeric_arrays(y_true, y_pred, "y_true", "y_pred")
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(6.6, 5.2))

    axis.scatter(actual, predicted, s=34, alpha=0.78, color=point_color, edgecolor="white")
    lo = float(min(np.min(actual), np.min(predicted)))
    hi = float(max(np.max(actual), np.max(predicted)))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    axis.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color=reference_color)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.22)
    _tight_layout(figure)
    return figure


def plot_residual_diagnostics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    title: str = "Residual diagnostics",
    residual_label: str = "Residual",
    fitted_label: str = "Predicted",
    figure_size: tuple[float, float] = (10.0, 4.6),
) -> Any:
    """Render residual-vs-fitted and residual distribution diagnostics."""

    pyplot = _require_pyplot()
    actual, predicted = _paired_numeric_arrays(y_true, y_pred, "y_true", "y_pred")
    residuals = actual - predicted

    figure, axes = pyplot.subplots(1, 2, figsize=figure_size)
    axes[0].scatter(predicted, residuals, s=34, alpha=0.78, color=DEFAULT_PALETTE[1])
    axes[0].axhline(0.0, color="#111827", linewidth=1.1, linestyle="--")
    axes[0].set_xlabel(fitted_label)
    axes[0].set_ylabel(residual_label)
    axes[0].set_title("Residuals by prediction")
    axes[0].grid(alpha=0.22)

    bins = min(24, max(6, int(np.sqrt(residuals.size))))
    axes[1].hist(residuals, bins=bins, color=DEFAULT_PALETTE[2], alpha=0.82, edgecolor="white")
    axes[1].axvline(0.0, color="#111827", linewidth=1.1, linestyle="--")
    axes[1].set_xlabel(residual_label)
    axes[1].set_ylabel("Rows")
    axes[1].set_title("Residual distribution")
    axes[1].grid(axis="y", alpha=0.22)

    figure.suptitle(title)
    _tight_layout(figure)
    return figure


def plot_metric_comparison(
    rows: Any,
    *,
    metric: str = "rmse",
    label_col: str = "model",
    title: str | None = None,
    ylabel: str | None = None,
    sort: bool = True,
    ax: Any | None = None,
) -> Any:
    """Render a sorted bar chart from metric rows or a dataframe-like object."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    values: list[tuple[str, float]] = []
    for row in parsed:
        if label_col not in row or metric not in row:
            raise ValueError(f"each row must contain '{label_col}' and '{metric}'")
        values.append((str(row[label_col]), _finite_float(row[metric], metric)))
    if not values:
        raise ValueError("rows must contain at least one metric row")
    if sort:
        values.sort(key=lambda item: item[1])

    labels = [label for label, _ in values]
    metric_values = [value for _, value in values]
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(max(7.0, len(labels) * 0.76), 4.8))
    colors = [DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)] for index in range(len(labels))]
    axis.bar(labels, metric_values, color=colors)
    axis.set_title(title or f"{metric.upper()} by model")
    axis.set_ylabel(ylabel or metric.upper())
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.22)
    _tight_layout(figure)
    return figure


def plot_horizon_metrics(
    rows: Any,
    *,
    horizon_col: str = "horizon",
    metric_col: str = "rmse",
    model_col: str = "model",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Render metric trajectories by forecast horizon for one or more models."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    series: dict[str, list[tuple[float, float]]] = {}
    for row in parsed:
        missing = {horizon_col, metric_col, model_col}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"each row must contain column(s): {names}")
        model = str(row[model_col])
        horizon = _finite_float(row[horizon_col], horizon_col)
        metric_value = _finite_float(row[metric_col], metric_col)
        series.setdefault(model, []).append((horizon, metric_value))
    if not series:
        raise ValueError("rows must contain at least one horizon metric row")

    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.6, 4.8))
    for index, (model, points) in enumerate(sorted(series.items())):
        points.sort(key=lambda item: item[0])
        axis.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            linewidth=1.9,
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
            label=model,
        )
    axis.set_title(title or f"{metric_col.upper()} by forecast horizon")
    axis.set_xlabel("Horizon")
    axis.set_ylabel(metric_col.upper())
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _tight_layout(figure)
    return figure


def plot_backtest_metrics(
    rows: Any,
    *,
    fold_col: str = "fold",
    metric_col: str = "rmse",
    model_col: str = "model",
    title: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Render model metric trajectories over rolling-origin or blocked folds."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    series: dict[str, list[tuple[float, float]]] = {}
    for row in parsed:
        missing = {fold_col, metric_col, model_col}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"each row must contain column(s): {names}")
        model = str(row[model_col])
        fold = _finite_float(row[fold_col], fold_col)
        metric_value = _finite_float(row[metric_col], metric_col)
        series.setdefault(model, []).append((fold, metric_value))
    if not series:
        raise ValueError("rows must contain at least one backtest metric row")

    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.8, 4.8))
    for index, (model, points) in enumerate(sorted(series.items())):
        points.sort(key=lambda item: item[0])
        axis.plot(
            [item[0] for item in points],
            [item[1] for item in points],
            marker="o",
            linewidth=1.9,
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
            label=model,
        )
    axis.set_title(title or f"{metric_col.upper()} by validation fold")
    axis.set_xlabel(fold_col)
    axis.set_ylabel(metric_col.upper())
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _tight_layout(figure)
    return figure


def plot_interval_calibration(
    rows: Any,
    *,
    nominal_col: str = "nominal_coverage",
    observed_col: str = "coverage",
    width_col: str | None = "mean_width",
    title: str = "Prediction interval calibration",
) -> Any:
    """Render nominal-vs-observed interval coverage and optional mean width."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one interval metric row")
    points: list[tuple[float, float, float | None]] = []
    for row in parsed:
        if nominal_col not in row or observed_col not in row:
            raise ValueError(f"each row must contain '{nominal_col}' and '{observed_col}'")
        nominal = _finite_float(row[nominal_col], nominal_col)
        observed = _finite_float(row[observed_col], observed_col)
        if not 0.0 <= nominal <= 1.0 or not 0.0 <= observed <= 1.0:
            raise ValueError("coverage values must be between 0 and 1")
        width = _finite_float(row[width_col], width_col) if width_col and width_col in row else None
        points.append((nominal, observed, width))
    points.sort(key=lambda item: item[0])

    has_width = any(width is not None for _, _, width in points)
    figure, axes = pyplot.subplots(
        1, 2 if has_width else 1, figsize=(10.0 if has_width else 5.6, 4.8)
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    calibration_axis = axes[0]
    nominal_values = [item[0] for item in points]
    observed_values = [item[1] for item in points]
    calibration_axis.plot(
        nominal_values,
        observed_values,
        marker="o",
        linewidth=1.9,
        color=DEFAULT_PALETTE[0],
        label="Observed",
    )
    calibration_axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.1, color="#111827")
    calibration_axis.set_xlim(0.0, 1.0)
    calibration_axis.set_ylim(0.0, 1.0)
    calibration_axis.set_xlabel("Nominal coverage")
    calibration_axis.set_ylabel("Observed coverage")
    calibration_axis.set_title("Coverage calibration")
    calibration_axis.grid(alpha=0.22)

    if has_width:
        width_axis = axes[1]
        width_axis.bar(
            [f"{value:.0%}" for value in nominal_values],
            [0.0 if width is None else width for _, _, width in points],
            color=DEFAULT_PALETTE[2],
        )
        width_axis.set_xlabel("Nominal coverage")
        width_axis.set_ylabel(width_col or "Mean width")
        width_axis.set_title("Interval width")
        width_axis.grid(axis="y", alpha=0.22)

    figure.suptitle(title)
    _tight_layout(figure)
    return figure


def plot_seasonality_curve(
    rows: Any,
    *,
    x_col: str = "phase",
    value_col: str = "value",
    lower_col: str | None = None,
    upper_col: str | None = None,
    label_col: str | None = None,
    title: str = "Seasonality curve",
    xlabel: str | None = None,
    ylabel: str | None = None,
    ax: Any | None = None,
) -> Any:
    """Render one or more seasonal/component curves with optional uncertainty bands."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one seasonality row")
    for row in parsed:
        missing = {x_col, value_col}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"seasonality rows must contain column(s): {names}")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in parsed:
        label = str(row[label_col]) if label_col and label_col in row else value_col
        groups.setdefault(label, []).append(row)

    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.8, 4.8))
    for index, (label, group_rows) in enumerate(sorted(groups.items())):
        sorted_rows = sorted(group_rows, key=lambda row: _finite_float(row[x_col], x_col))
        x_values = [_finite_float(row[x_col], x_col) for row in sorted_rows]
        values = [_finite_float(row[value_col], value_col) for row in sorted_rows]
        color = DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)]
        axis.plot(x_values, values, linewidth=1.9, color=color, label=label)
        if (
            lower_col
            and upper_col
            and all(lower_col in row and upper_col in row for row in sorted_rows)
        ):
            lower = [_finite_float(row[lower_col], lower_col) for row in sorted_rows]
            upper = [_finite_float(row[upper_col], upper_col) for row in sorted_rows]
            if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
                raise ValueError("seasonality lower values must be <= upper values")
            axis.fill_between(x_values, lower, upper, color=color, alpha=0.14)

    axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
    axis.set_title(title)
    axis.set_xlabel(xlabel or x_col)
    axis.set_ylabel(ylabel or value_col)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _tight_layout(figure)
    return figure


def plot_changepoint_effects(
    rows: Any,
    *,
    time_col: str = "timestamp",
    delta_col: str = "delta",
    label_col: str | None = None,
    title: str = "Changepoint effects",
    ax: Any | None = None,
) -> Any:
    """Render signed changepoint effect magnitudes."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one changepoint row")
    for row in parsed:
        missing = {time_col, delta_col}.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"changepoint rows must contain column(s): {names}")
    parsed = sorted(parsed, key=lambda row: row[time_col])

    labels = [
        str(row[label_col]) if label_col and label_col in row else str(row[time_col])
        for row in parsed
    ]
    deltas = [_finite_float(row[delta_col], delta_col) for row in parsed]
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(max(7.0, len(labels) * 0.64), 4.8))
    colors = [DEFAULT_PALETTE[2] if value >= 0.0 else DEFAULT_PALETTE[1] for value in deltas]
    axis.bar(labels, deltas, color=colors)
    axis.axhline(0.0, color="#111827", linewidth=1.0)
    axis.set_title(title)
    axis.set_xlabel(time_col if label_col is None else label_col)
    axis.set_ylabel(delta_col)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.22)
    _tight_layout(figure)
    return figure


def plot_cutoff_predictions(
    rows: Any,
    *,
    time_col: str = "timestamp",
    cutoff_col: str = "cutoff",
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    title: str = "Predictions by cutoff",
    ax: Any | None = None,
) -> Any:
    """Render cross-validation predictions grouped by cutoff date or fold."""

    pyplot = _require_pyplot()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one cutoff prediction row")
    required = {time_col, cutoff_col, actual_col, prediction_col}
    for row in parsed:
        missing = required.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"cutoff prediction rows must contain column(s): {names}")

    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(9.0, 4.9))
    actual_by_time: dict[Any, float] = {}
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in parsed:
        actual_by_time[row[time_col]] = _finite_float(row[actual_col], actual_col)
        groups.setdefault(str(row[cutoff_col]), []).append(row)

    actual_points = sorted(actual_by_time.items(), key=lambda item: item[0])
    axis.plot(
        [item[0] for item in actual_points],
        [item[1] for item in actual_points],
        color="#111827",
        linewidth=1.8,
        label="Actual",
    )
    for index, (cutoff, group_rows) in enumerate(sorted(groups.items())):
        sorted_rows = sorted(group_rows, key=lambda row: row[time_col])
        axis.plot(
            [row[time_col] for row in sorted_rows],
            [_finite_float(row[prediction_col], prediction_col) for row in sorted_rows],
            marker="o",
            linewidth=1.4,
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
            label=f"Cutoff {cutoff}",
            alpha=0.86,
        )

    axis.set_title(title)
    axis.set_xlabel(time_col)
    axis.set_ylabel(prediction_col)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    _tight_layout(figure)
    return figure


def plot_spatial_points(
    rows: Any,
    *,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    value_col: str | None = None,
    title: str = "Spatial points",
    ax: Any | None = None,
    cmap: str = "viridis",
    point_size: float = 34.0,
) -> Any:
    """Render static latitude/longitude points with optional visualization packages."""

    pyplot = _require_pyplot()
    geopandas, point_cls, _ = _require_static_map_stack()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one spatial point row")
    geometries = []
    values = []
    for row in parsed:
        if latitude_col not in row or longitude_col not in row:
            raise ValueError(f"point rows must contain '{latitude_col}' and '{longitude_col}'")
        latitude = _finite_float(row[latitude_col], latitude_col)
        longitude = _finite_float(row[longitude_col], longitude_col)
        geometries.append(point_cls(longitude, latitude))
        if value_col is not None:
            if value_col not in row:
                raise ValueError(f"point rows must contain '{value_col}'")
            values.append(_finite_float(row[value_col], value_col))

    frame = geopandas.GeoDataFrame(parsed, geometry=geometries, crs="EPSG:4326")
    if value_col is not None:
        frame[value_col] = values
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.2, 6.0))
    frame.plot(
        ax=axis,
        column=value_col,
        cmap=cmap if value_col else None,
        markersize=point_size,
        legend=value_col is not None,
        color=None if value_col else DEFAULT_PALETTE[0],
        alpha=0.82,
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.18)
    _tight_layout(figure)
    return figure


def plot_route_segments(
    rows: Any,
    *,
    pickup_latitude_col: str = "pickup_latitude",
    pickup_longitude_col: str = "pickup_longitude",
    dropoff_latitude_col: str = "dropoff_latitude",
    dropoff_longitude_col: str = "dropoff_longitude",
    value_col: str | None = None,
    title: str = "Route segments",
    ax: Any | None = None,
    cmap: str = "viridis",
    linewidth: float = 1.4,
) -> Any:
    """Render static pickup/dropoff route segments with optional visualization packages."""

    pyplot = _require_pyplot()
    geopandas, point_cls, line_cls = _require_static_map_stack()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one route row")
    required = {
        pickup_latitude_col,
        pickup_longitude_col,
        dropoff_latitude_col,
        dropoff_longitude_col,
    }
    geometries = []
    values = []
    for row in parsed:
        missing = required.difference(row)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"route rows must contain column(s): {names}")
        pickup = point_cls(
            _finite_float(row[pickup_longitude_col], pickup_longitude_col),
            _finite_float(row[pickup_latitude_col], pickup_latitude_col),
        )
        dropoff = point_cls(
            _finite_float(row[dropoff_longitude_col], dropoff_longitude_col),
            _finite_float(row[dropoff_latitude_col], dropoff_latitude_col),
        )
        geometries.append(line_cls([pickup, dropoff]))
        if value_col is not None:
            if value_col not in row:
                raise ValueError(f"route rows must contain '{value_col}'")
            values.append(_finite_float(row[value_col], value_col))

    frame = geopandas.GeoDataFrame(parsed, geometry=geometries, crs="EPSG:4326")
    if value_col is not None:
        frame[value_col] = values
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(7.2, 6.0))
    frame.plot(
        ax=axis,
        column=value_col,
        cmap=cmap if value_col else None,
        linewidth=linewidth,
        legend=value_col is not None,
        color=None if value_col else DEFAULT_PALETTE[0],
        alpha=0.72,
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.grid(alpha=0.18)
    _tight_layout(figure)
    return figure


def write_pydeck_point_map(
    rows: Any,
    output_html: str | Path,
    *,
    latitude_col: str = "latitude",
    longitude_col: str = "longitude",
    value_col: str | None = None,
    tooltip_cols: Sequence[str] = (),
    title: str = "CartoBoost point map",
    radius_scale: float = 35.0,
) -> Path:
    """Write an interactive PyDeck point map and return the HTML path."""

    pydeck = _require_pydeck_stack()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one spatial point row")
    _validate_coordinate_rows(parsed, latitude_col, longitude_col)
    latitude, longitude = _mean_lat_lon(parsed, latitude_col, longitude_col)
    layer = pydeck.Layer(
        "ScatterplotLayer",
        data=parsed,
        get_position=[longitude_col, latitude_col],
        get_radius=_pydeck_radius_expression(value_col, radius_scale),
        get_fill_color=_pydeck_color_expression(value_col),
        pickable=True,
        opacity=0.82,
    )
    deck = pydeck.Deck(
        layers=[layer],
        initial_view_state=pydeck.ViewState(latitude=latitude, longitude=longitude, zoom=10),
        tooltip=_pydeck_tooltip(tooltip_cols),
        map_style=None,
        description=title,
    )
    return _write_pydeck_html(deck, output_html)


def write_pydeck_route_map(
    rows: Any,
    output_html: str | Path,
    *,
    pickup_latitude_col: str = "pickup_latitude",
    pickup_longitude_col: str = "pickup_longitude",
    dropoff_latitude_col: str = "dropoff_latitude",
    dropoff_longitude_col: str = "dropoff_longitude",
    value_col: str | None = None,
    tooltip_cols: Sequence[str] = (),
    title: str = "CartoBoost route map",
) -> Path:
    """Write an interactive PyDeck pickup/dropoff arc map and return the HTML path."""

    pydeck = _require_pydeck_stack()
    parsed = _rows_from_table(rows)
    if not parsed:
        raise ValueError("rows must contain at least one route row")
    _validate_coordinate_rows(parsed, pickup_latitude_col, pickup_longitude_col)
    _validate_coordinate_rows(parsed, dropoff_latitude_col, dropoff_longitude_col)
    latitude, longitude = _mean_lat_lon(parsed, pickup_latitude_col, pickup_longitude_col)
    layer = pydeck.Layer(
        "ArcLayer",
        data=parsed,
        get_source_position=[pickup_longitude_col, pickup_latitude_col],
        get_target_position=[dropoff_longitude_col, dropoff_latitude_col],
        get_source_color=_pydeck_color_expression(value_col, positive=(37, 99, 235)),
        get_target_color=_pydeck_color_expression(value_col, positive=(220, 38, 38)),
        get_width=2,
        pickable=True,
        auto_highlight=True,
    )
    deck = pydeck.Deck(
        layers=[layer],
        initial_view_state=pydeck.ViewState(latitude=latitude, longitude=longitude, zoom=10),
        tooltip=_pydeck_tooltip(tooltip_cols),
        map_style=None,
        description=title,
    )
    return _write_pydeck_html(deck, output_html)


def plot(
    m: Any,
    fcst: Any,
    ax: Any | None = None,
    uncertainty: bool = True,
    plot_cap: bool = True,
    xlabel: str = "ds",
    ylabel: str = "y",
    figsize: tuple[float, float] = (10, 6),
    include_legend: bool = False,
) -> Any:
    """Plot a Prophet-shaped forecast with the same public signature as Prophet."""

    pyplot = _require_pyplot()
    _require_prophet_forecast_columns(fcst, ["ds", "yhat"])
    user_provided_ax = ax is not None
    if ax is None:
        figure = pyplot.figure(facecolor="w", figsize=figsize)
        ax = figure.add_subplot(111)
    else:
        figure = ax.get_figure()

    history = m.history
    ax.plot(history["ds"], history["y"], "k.", label="Observed data points")
    ax.plot(fcst["ds"], fcst["yhat"], ls="-", c="#0072B2", label="Forecast")
    if plot_cap and "cap" in fcst:
        ax.plot(fcst["ds"], fcst["cap"], ls="--", c="k", label="Maximum capacity")
    if m.logistic_floor and "floor" in fcst and plot_cap:
        ax.plot(fcst["ds"], fcst["floor"], ls="--", c="k", label="Minimum capacity")
    if uncertainty and m.uncertainty_samples:
        ax.fill_between(
            fcst["ds"],
            fcst["yhat_lower"],
            fcst["yhat_upper"],
            color="#0072B2",
            alpha=0.2,
            label="Uncertainty interval",
        )
    _format_prophet_date_axis(ax)
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if include_legend:
        ax.legend()
    if not user_provided_ax:
        figure.tight_layout()
    return figure


def plot_components(
    m: Any,
    fcst: Any,
    uncertainty: bool = True,
    plot_cap: bool = True,
    weekly_start: int = 0,
    yearly_start: int = 0,
    figsize: tuple[float, float] | None = None,
) -> Any:
    """Plot Prophet-style trend, holiday, seasonal, and regressor components."""

    pyplot = _require_pyplot()
    components = ["trend"]
    if m.train_holiday_names is not None and "holidays" in fcst:
        components.append("holidays")
    seasonalities = m.seasonalities
    if "weekly" in seasonalities and "weekly" in fcst:
        components.append("weekly")
    if "yearly" in seasonalities and "yearly" in fcst:
        components.append("yearly")
    components.extend(
        name for name in sorted(seasonalities) if name in fcst and name not in {"weekly", "yearly"}
    )
    extra_regressors = m.extra_regressors
    modes = {"additive": False, "multiplicative": False}
    for props in extra_regressors.values():
        modes[props["mode"]] = True
    for mode in ["additive", "multiplicative"]:
        column = f"extra_regressors_{mode}"
        if modes[mode] and column in fcst:
            components.append(column)

    figure_size = figsize or (9, 3 * len(components))
    figure, axes = pyplot.subplots(len(components), 1, facecolor="w", figsize=figure_size)
    if len(components) == 1:
        axes = [axes]

    multiplicative_axes = []
    pandas = _require_pandas()
    dt = pandas.to_datetime(m.history["ds"]).diff()
    min_dt = dt.iloc[dt.values.nonzero()[0]].min()

    for axis, name in zip(axes, components, strict=True):
        if name == "trend":
            plot_forecast_component(
                m, fcst, "trend", ax=axis, uncertainty=uncertainty, plot_cap=plot_cap
            )
        elif name in seasonalities:
            period = seasonalities[name]["period"]
            if (name == "weekly" or period == 7) and min_dt == pandas.Timedelta(days=1):
                plot_weekly(
                    m, ax=axis, uncertainty=uncertainty, weekly_start=weekly_start, name=name
                )
            elif name == "yearly" or period == 365.25:
                plot_yearly(
                    m, ax=axis, uncertainty=uncertainty, yearly_start=yearly_start, name=name
                )
            else:
                plot_seasonality(m, name, ax=axis, uncertainty=uncertainty)
        else:
            plot_forecast_component(m, fcst, name, ax=axis, uncertainty=uncertainty, plot_cap=False)
        if name in m.component_modes["multiplicative"]:
            multiplicative_axes.append(axis)

    figure.tight_layout()
    for axis in multiplicative_axes:
        set_y_as_percent(axis)
    return figure


def plot_forecast_component(
    m: Any,
    fcst: Any,
    name: str,
    ax: Any | None = None,
    uncertainty: bool = True,
    plot_cap: bool = False,
    figsize: tuple[float, float] = (10, 6),
) -> list[Any]:
    """Plot one Prophet-style forecast component and return Matplotlib artists."""

    pyplot = _require_pyplot()
    if ax is None:
        figure = pyplot.figure(facecolor="w", figsize=figsize)
        ax = figure.add_subplot(111)
    _require_prophet_forecast_columns(fcst, ["ds", name])
    artists = []
    artists += ax.plot(fcst["ds"], fcst[name], ls="-", c="#0072B2")
    if plot_cap and "cap" in fcst:
        artists += ax.plot(fcst["ds"], fcst["cap"], ls="--", c="k")
    if m.logistic_floor and "floor" in fcst and plot_cap:
        ax.plot(fcst["ds"], fcst["floor"], ls="--", c="k")
    if uncertainty and m.uncertainty_samples:
        artists.append(
            ax.fill_between(
                fcst["ds"], fcst[f"{name}_lower"], fcst[f"{name}_upper"], color="#0072B2", alpha=0.2
            )
        )
    _format_prophet_date_axis(ax)
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xlabel("ds")
    ax.set_ylabel(name)
    if name in m.component_modes["multiplicative"]:
        set_y_as_percent(ax)
    return artists


def seasonality_plot_df(m: Any, ds: Any) -> Any:
    """Prepare the Prophet-shaped dataframe used for seasonality plots."""

    pandas = _require_pandas()
    data: dict[str, Any] = {"ds": ds, "cap": 1.0, "floor": 0.0}
    for name in m.extra_regressors:
        data[name] = 0.0
    for props in m.seasonalities.values():
        if props["condition_name"] is not None:
            data[props["condition_name"]] = True
    frame = pandas.DataFrame(data)
    return m.setup_dataframe(frame)


def plot_weekly(
    m: Any,
    ax: Any | None = None,
    uncertainty: bool = True,
    weekly_start: int = 0,
    figsize: tuple[float, float] = (10, 6),
    name: str = "weekly",
) -> list[Any]:
    """Plot a weekly Prophet-style seasonal component."""

    pyplot = _require_pyplot()
    pandas = _require_pandas()
    if ax is None:
        figure = pyplot.figure(facecolor="w", figsize=figsize)
        ax = figure.add_subplot(111)
    days = pandas.date_range(start="2017-01-01", periods=7) + pandas.Timedelta(days=weekly_start)
    seas = m.predict_seasonal_components(seasonality_plot_df(m, days))
    labels = days.day_name()
    artists = ax.plot(range(len(labels)), seas[name], ls="-", c="#0072B2")
    if uncertainty and m.uncertainty_samples:
        artists.append(
            ax.fill_between(
                range(len(labels)),
                seas[f"{name}_lower"],
                seas[f"{name}_upper"],
                color="#0072B2",
                alpha=0.2,
            )
        )
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Day of week")
    ax.set_ylabel(name)
    if m.seasonalities[name]["mode"] == "multiplicative":
        set_y_as_percent(ax)
    return artists


def plot_yearly(
    m: Any,
    ax: Any | None = None,
    uncertainty: bool = True,
    yearly_start: int = 0,
    figsize: tuple[float, float] = (10, 6),
    name: str = "yearly",
) -> list[Any]:
    """Plot a yearly Prophet-style seasonal component."""

    pyplot = _require_pyplot()
    pandas = _require_pandas()
    if ax is None:
        figure = pyplot.figure(facecolor="w", figsize=figsize)
        ax = figure.add_subplot(111)
    days = pandas.date_range(start="2017-01-01", periods=365) + pandas.Timedelta(days=yearly_start)
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    artists = ax.plot(df_y["ds"], seas[name], ls="-", c="#0072B2")
    if uncertainty and m.uncertainty_samples:
        artists.append(
            ax.fill_between(
                df_y["ds"], seas[f"{name}_lower"], seas[f"{name}_upper"], color="#0072B2", alpha=0.2
            )
        )
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    _format_month_axis(ax)
    ax.set_xlabel("Day of year")
    ax.set_ylabel(name)
    if m.seasonalities[name]["mode"] == "multiplicative":
        set_y_as_percent(ax)
    return artists


def plot_seasonality(
    m: Any,
    name: str,
    ax: Any | None = None,
    uncertainty: bool = True,
    figsize: tuple[float, float] = (10, 6),
) -> list[Any]:
    """Plot a custom Prophet-style seasonal component."""

    pyplot = _require_pyplot()
    pandas = _require_pandas()
    if ax is None:
        figure = pyplot.figure(facecolor="w", figsize=figsize)
        ax = figure.add_subplot(111)
    start = pandas.to_datetime("2017-01-01 0000")
    period = m.seasonalities[name]["period"]
    end = start + pandas.Timedelta(days=period)
    days = pandas.to_datetime(np.linspace(start.value, end.value, 200))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    artists = ax.plot(df_y["ds"], seas[name], ls="-", c="#0072B2")
    if uncertainty and m.uncertainty_samples:
        artists.append(
            ax.fill_between(
                df_y["ds"], seas[f"{name}_lower"], seas[f"{name}_upper"], color="#0072B2", alpha=0.2
            )
        )
    ax.grid(True, which="major", c="gray", ls="-", lw=1, alpha=0.2)
    _format_seasonality_axis(ax, name, period, start, end)
    ax.set_ylabel(name)
    if m.seasonalities[name]["mode"] == "multiplicative":
        set_y_as_percent(ax)
    return artists


def set_y_as_percent(ax: Any) -> Any:
    """Format an axis as Prophet does for multiplicative components."""

    ticks = ax.get_yticks()
    ax.set_yticks(ticks.tolist())
    ax.set_yticklabels([f"{100 * tick:.4g}%" for tick in ticks])
    return ax


def add_changepoints_to_plot(
    ax: Any,
    m: Any,
    fcst: Any,
    threshold: float = 0.01,
    cp_color: str = "r",
    cp_linestyle: str = "--",
    trend: bool = True,
) -> list[Any]:
    """Add significant Prophet-style changepoint markers to an existing plot."""

    artists = []
    if trend:
        artists.append(ax.plot(fcst["ds"], fcst["trend"], c=cp_color))
    if len(m.changepoints) > 0:
        significant = m.changepoints[np.abs(np.nanmean(m.params["delta"], axis=0)) >= threshold]
    else:
        significant = []
    for changepoint in significant:
        artists.append(ax.axvline(x=changepoint, c=cp_color, ls=cp_linestyle))
    return artists


def plot_cross_validation_metric(
    df_cv: Any,
    metric: str,
    rolling_window: float = 0.1,
    ax: Any | None = None,
    figsize: tuple[float, float] = (10, 6),
    color: str = "b",
    point_color: str = "gray",
) -> Any:
    """Plot a Prophet-style cross-validation metric against forecast horizon."""

    pyplot = _require_pyplot()
    metrics = _prophet_performance_metrics(df_cv, metric, rolling_window)
    point_metrics = _prophet_performance_metrics(df_cv, metric, -1)
    if ax is None:
        figure = pyplot.figure(facecolor="w", figsize=figsize)
        ax = figure.add_subplot(111)
    else:
        figure = ax.get_figure()
    x_points, unit_name = _timedelta_plot_values(point_metrics["horizon"])
    x_smooth, _ = _timedelta_plot_values(metrics["horizon"], unit_name=unit_name)
    ax.plot(x_points, point_metrics[metric], ".", alpha=0.1, c=point_color)
    ax.plot(x_smooth, metrics[metric], "-", c=color)
    ax.grid(True)
    ax.set_xlabel(f"Horizon ({unit_name})")
    ax.set_ylabel(metric)
    return figure


def plot_plotly(
    m: Any,
    fcst: Any,
    uncertainty: bool = True,
    plot_cap: bool = True,
    trend: bool = False,
    changepoints: bool = False,
    changepoints_threshold: float = 0.01,
    xlabel: str = "ds",
    ylabel: str = "y",
    figsize: tuple[int, int] = (900, 600),
) -> Any:
    """Plot a Prophet-shaped forecast with Plotly using Prophet's public signature."""

    go, _ = _require_plotly()
    data = [
        go.Scatter(
            name="Actual",
            x=m.history["ds"],
            y=m.history["y"],
            marker={"color": "black", "size": 4},
            mode="markers",
        )
    ]
    if uncertainty and m.uncertainty_samples:
        data.append(
            go.Scatter(
                x=fcst["ds"],
                y=fcst["yhat_lower"],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
            )
        )
    data.append(
        go.Scatter(
            name="Predicted",
            x=fcst["ds"],
            y=fcst["yhat"],
            mode="lines",
            line={"color": "#0072B2", "width": 2},
            fillcolor="rgba(0, 114, 178, 0.2)",
            fill="tonexty" if uncertainty and m.uncertainty_samples else "none",
        )
    )
    if uncertainty and m.uncertainty_samples:
        data.append(
            go.Scatter(
                x=fcst["ds"],
                y=fcst["yhat_upper"],
                mode="lines",
                line={"width": 0},
                fillcolor="rgba(0, 114, 178, 0.2)",
                fill="tonexty",
                hoverinfo="skip",
            )
        )
    if plot_cap and "cap" in fcst:
        data.append(
            go.Scatter(
                name="Cap",
                x=fcst["ds"],
                y=fcst["cap"],
                mode="lines",
                line={"color": "black", "dash": "dash", "width": 2},
            )
        )
    if m.logistic_floor and "floor" in fcst and plot_cap:
        data.append(
            go.Scatter(
                name="Floor",
                x=fcst["ds"],
                y=fcst["floor"],
                mode="lines",
                line={"color": "black", "dash": "dash", "width": 2},
            )
        )
    if trend:
        data.append(
            go.Scatter(
                name="Trend",
                x=fcst["ds"],
                y=fcst["trend"],
                mode="lines",
                line={"color": "#B23B00", "width": 2},
            )
        )
    if changepoints and len(m.changepoints) > 0:
        significant = m.changepoints[
            np.abs(np.nanmean(m.params["delta"], axis=0)) >= changepoints_threshold
        ]
        data.append(
            go.Scatter(
                x=significant,
                y=fcst.loc[fcst["ds"].isin(significant), "trend"],
                marker={
                    "size": 50,
                    "symbol": "line-ns-open",
                    "color": "#B23B00",
                    "line": {"width": 2},
                },
                mode="markers",
                hoverinfo="skip",
            )
        )
    layout = {
        "showlegend": False,
        "width": figsize[0],
        "height": figsize[1],
        "yaxis": {"title": ylabel},
        "xaxis": {
            "title": xlabel,
            "type": "date",
            "rangeselector": {
                "buttons": [
                    {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
                    {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                    {"step": "all"},
                ]
            },
            "rangeslider": {"visible": True},
        },
    }
    return go.Figure(data=data, layout=layout)


def plot_components_plotly(
    m: Any,
    fcst: Any,
    uncertainty: bool = True,
    plot_cap: bool = True,
    figsize: tuple[int, int] = (900, 200),
) -> Any:
    """Plot Prophet-style forecast components with Plotly."""

    go, make_subplots = _require_plotly()
    components: dict[str, Mapping[str, Any]] = {
        "trend": get_forecast_component_plotly_props(m, fcst, "trend", uncertainty, plot_cap)
    }
    if m.train_holiday_names is not None and "holidays" in fcst:
        components["holidays"] = get_forecast_component_plotly_props(
            m, fcst, "holidays", uncertainty
        )
    modes = {"additive": False, "multiplicative": False}
    for props in m.extra_regressors.values():
        modes[props["mode"]] = True
    for mode in ["additive", "multiplicative"]:
        column = f"extra_regressors_{mode}"
        if modes[mode] and column in fcst:
            components[column] = get_forecast_component_plotly_props(m, fcst, column)
    for name in m.seasonalities:
        components[name] = get_seasonality_plotly_props(m, name)

    figure = make_subplots(rows=len(components), cols=1, print_grid=False)
    figure["layout"].update(
        go.Layout(showlegend=False, width=figsize[0], height=figsize[1] * len(components))
    )
    for index, props in enumerate(components.values(), start=1):
        xaxis = figure["layout"]["xaxis" if index == 1 else f"xaxis{index}"]
        yaxis = figure["layout"]["yaxis" if index == 1 else f"yaxis{index}"]
        xaxis.update(props["xaxis"])
        yaxis.update(props["yaxis"])
        for trace in props["traces"]:
            figure.append_trace(trace, index, 1)
    return figure


def plot_forecast_component_plotly(
    m: Any,
    fcst: Any,
    name: str,
    uncertainty: bool = True,
    plot_cap: bool = False,
    figsize: tuple[int, int] = (900, 300),
) -> Any:
    """Plot one Prophet-style forecast component with Plotly."""

    go, _ = _require_plotly()
    props = get_forecast_component_plotly_props(m, fcst, name, uncertainty, plot_cap)
    return go.Figure(
        data=props["traces"],
        layout=go.Layout(
            width=figsize[0],
            height=figsize[1],
            showlegend=False,
            xaxis=props["xaxis"],
            yaxis=props["yaxis"],
        ),
    )


def plot_seasonality_plotly(
    m: Any,
    name: str,
    uncertainty: bool = True,
    figsize: tuple[int, int] = (900, 300),
) -> Any:
    """Plot one Prophet-style seasonal component with Plotly."""

    go, _ = _require_plotly()
    props = get_seasonality_plotly_props(m, name, uncertainty)
    return go.Figure(
        data=props["traces"],
        layout=go.Layout(
            width=figsize[0],
            height=figsize[1],
            showlegend=False,
            xaxis=props["xaxis"],
            yaxis=props["yaxis"],
        ),
    )


def get_forecast_component_plotly_props(
    m: Any,
    fcst: Any,
    name: str,
    uncertainty: bool = True,
    plot_cap: bool = False,
) -> dict[str, Any]:
    """Prepare Plotly traces and axes for one Prophet-style forecast component."""

    go, _ = _require_plotly()
    pandas = _require_pandas()
    range_margin = (fcst["ds"].max() - fcst["ds"].min()) * 0.05
    range_x = [fcst["ds"].min() - range_margin, fcst["ds"].max() + range_margin]
    text = None
    mode = "lines"
    if name == "holidays":
        holidays = m.construct_holiday_dataframe(fcst["ds"])
        holiday_features, _, _ = m.make_holiday_features(fcst["ds"], holidays)
        holiday_features.columns = holiday_features.columns.str.replace("_delim_", "", regex=False)
        holiday_features.columns = holiday_features.columns.str.replace("+0", "", regex=False)
        text = pandas.Series(data="", index=holiday_features.index)
        for holiday_feature, indices in holiday_features.items():
            text[indices.astype(bool) & (text != "")] += "<br>"
            text[indices.astype(bool)] += holiday_feature
    traces = [
        go.Scatter(
            name=name,
            x=fcst["ds"],
            y=fcst[name],
            mode=mode,
            line=go.scatter.Line(color="#0072B2", width=2),
            text=text,
        )
    ]
    if (
        uncertainty
        and m.uncertainty_samples
        and (fcst[f"{name}_upper"] != fcst[f"{name}_lower"]).any()
    ):
        traces.append(
            go.Scatter(
                name=f"{name}_upper",
                x=fcst["ds"],
                y=fcst[f"{name}_upper"],
                mode=mode,
                line=go.scatter.Line(width=0, color="rgba(0, 114, 178, 0.2)"),
            )
        )
        traces.append(
            go.Scatter(
                name=f"{name}_lower",
                x=fcst["ds"],
                y=fcst[f"{name}_lower"],
                mode=mode,
                line=go.scatter.Line(width=0, color="rgba(0, 114, 178, 0.2)"),
                fillcolor="rgba(0, 114, 178, 0.2)",
                fill="tonexty",
            )
        )
    if plot_cap and "cap" in fcst:
        traces.append(
            go.Scatter(
                name="Cap",
                x=fcst["ds"],
                y=fcst["cap"],
                mode="lines",
                line=go.scatter.Line(color="black", dash="dash", width=2),
            )
        )
    if m.logistic_floor and "floor" in fcst and plot_cap:
        traces.append(
            go.Scatter(
                name="Floor",
                x=fcst["ds"],
                y=fcst["floor"],
                mode="lines",
                line=go.scatter.Line(color="black", dash="dash", width=2),
            )
        )
    yaxis = go.layout.YAxis(
        rangemode="normal" if name == "trend" else "tozero",
        title=go.layout.yaxis.Title(text=name),
        zerolinecolor="#AAA",
    )
    if name in m.component_modes["multiplicative"]:
        yaxis.update(tickformat="%", hoverformat=".2%")
    return {"traces": traces, "xaxis": go.layout.XAxis(type="date", range=range_x), "yaxis": yaxis}


def get_seasonality_plotly_props(m: Any, name: str, uncertainty: bool = True) -> dict[str, Any]:
    """Prepare Plotly traces and axes for one Prophet-style seasonality."""

    go, _ = _require_plotly()
    pandas = _require_pandas()
    start = pandas.to_datetime("2017-01-01 0000")
    period = m.seasonalities[name]["period"]
    end = start + pandas.Timedelta(days=period)
    if (m.history["ds"].dt.hour == 0).all():
        points = int(np.floor(period))
    elif (m.history["ds"].dt.minute == 0).all():
        points = int(np.floor(period * 24))
    else:
        points = int(np.floor(period * 24 * 60))
    days = pandas.to_datetime(np.linspace(start.value, end.value, points, endpoint=False))
    df_y = seasonality_plot_df(m, days)
    seas = m.predict_seasonal_components(df_y)
    traces = [
        go.Scatter(
            name=name,
            x=df_y["ds"],
            y=seas[name],
            mode="lines",
            line=go.scatter.Line(color="#0072B2", width=2),
        )
    ]
    if (
        uncertainty
        and m.uncertainty_samples
        and (seas[f"{name}_upper"] != seas[f"{name}_lower"]).any()
    ):
        traces.append(
            go.Scatter(
                name=f"{name}_upper",
                x=df_y["ds"],
                y=seas[f"{name}_upper"],
                mode="lines",
                line=go.scatter.Line(width=0, color="rgba(0, 114, 178, 0.2)"),
            )
        )
        traces.append(
            go.Scatter(
                name=f"{name}_lower",
                x=df_y["ds"],
                y=seas[f"{name}_lower"],
                mode="lines",
                line=go.scatter.Line(width=0, color="rgba(0, 114, 178, 0.2)"),
                fillcolor="rgba(0, 114, 178, 0.2)",
                fill="tonexty",
            )
        )
    if period <= 2:
        tickformat = "%H:%M"
    elif period < 7:
        tickformat = "%A %H:%M"
    elif period < 14:
        tickformat = "%A"
    else:
        tickformat = "%B %e"
    range_margin = (df_y["ds"].max() - df_y["ds"].min()) * 0.05
    yaxis = go.layout.YAxis(title=go.layout.yaxis.Title(text=name), zerolinecolor="#AAA")
    if m.seasonalities[name]["mode"] == "multiplicative":
        yaxis.update(tickformat="%", hoverformat=".2%")
    return {
        "traces": traces,
        "xaxis": go.layout.XAxis(
            tickformat=tickformat,
            type="date",
            range=[df_y["ds"].min() - range_margin, df_y["ds"].max() + range_margin],
        ),
        "yaxis": yaxis,
    }


def plot_forecast(
    forecast: Any,
    *,
    history: Any | None = None,
    time_col: str = "timestamp",
    actual_col: str = "actual",
    prediction_col: str = "prediction",
    lower_col: str | None = "lower",
    upper_col: str | None = "upper",
    series_id: str | None = None,
    series_id_col: str = "series_id",
    changepoints: Sequence[Any] | None = None,
    title: str = "Forecast",
    ax: Any | None = None,
) -> Any:
    """Render history, forecasts, actual holdout values, and optional intervals."""

    pyplot = _require_pyplot()
    forecast_rows = _filter_series(_rows_from_table(forecast), series_id, series_id_col)
    history_rows = (
        _filter_series(_rows_from_table(history), series_id, series_id_col) if history else []
    )
    if not forecast_rows:
        raise ValueError("forecast must contain at least one row")
    if any(time_col not in row for row in forecast_rows):
        raise ValueError(f"forecast rows must contain '{time_col}'")
    if any(prediction_col not in row for row in forecast_rows):
        raise ValueError(f"forecast rows must contain '{prediction_col}'")

    forecast_rows = sorted(forecast_rows, key=lambda row: row[time_col])
    history_rows = sorted(history_rows, key=lambda row: row[time_col]) if history_rows else []
    figure, axis = _figure_axis(pyplot, ax=ax, figsize=(9.0, 4.9))

    if history_rows:
        if any(time_col not in row or actual_col not in row for row in history_rows):
            raise ValueError(f"history rows must contain '{time_col}' and '{actual_col}'")
        axis.plot(
            [row[time_col] for row in history_rows],
            [_finite_float(row[actual_col], actual_col) for row in history_rows],
            color="#111827",
            linewidth=1.8,
            label="History",
        )

    forecast_times = [row[time_col] for row in forecast_rows]
    forecast_values = [_finite_float(row[prediction_col], prediction_col) for row in forecast_rows]
    axis.plot(
        forecast_times,
        forecast_values,
        marker="o",
        color=DEFAULT_PALETTE[0],
        linewidth=1.9,
        label="Forecast",
    )

    if any(actual_col in row and row[actual_col] is not None for row in forecast_rows):
        actual_rows = [
            row for row in forecast_rows if actual_col in row and row[actual_col] is not None
        ]
        axis.scatter(
            [row[time_col] for row in actual_rows],
            [_finite_float(row[actual_col], actual_col) for row in actual_rows],
            color=DEFAULT_PALETTE[1],
            s=34,
            label="Actual",
            zorder=3,
        )

    if (
        lower_col
        and upper_col
        and all(lower_col in row and upper_col in row for row in forecast_rows)
    ):
        lower = [_finite_float(row[lower_col], lower_col) for row in forecast_rows]
        upper = [_finite_float(row[upper_col], upper_col) for row in forecast_rows]
        if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("interval lower values must be <= upper values")
        axis.fill_between(
            forecast_times, lower, upper, color=DEFAULT_PALETTE[0], alpha=0.16, label="Interval"
        )

    if changepoints:
        for index, changepoint in enumerate(changepoints):
            axis.axvline(
                changepoint,
                color="#6b7280",
                linewidth=0.9,
                linestyle=":",
                alpha=0.72,
                label="Changepoint" if index == 0 else None,
            )

    axis.set_title(title if series_id is None else f"{title}: {series_id}")
    axis.set_xlabel(time_col)
    axis.set_ylabel(prediction_col)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    axis.tick_params(axis="x", rotation=25)
    _tight_layout(figure)
    return figure


def plot_forecast_components(
    rows: Any,
    *,
    time_col: str = "timestamp",
    component_cols: Sequence[str] | None = None,
    series_id: str | None = None,
    series_id_col: str = "series_id",
    changepoints: Sequence[Any] | None = None,
    title: str = "Forecast components",
    figure_size: tuple[float, float] | None = None,
) -> Any:
    """Render trend, seasonal, event, or other forecast component columns."""

    pyplot = _require_pyplot()
    parsed_rows = _filter_series(_rows_from_table(rows), series_id, series_id_col)
    if not parsed_rows:
        raise ValueError("rows must contain at least one component row")
    if any(time_col not in row for row in parsed_rows):
        raise ValueError(f"rows must contain '{time_col}'")
    parsed_rows = sorted(parsed_rows, key=lambda row: row[time_col])

    selected_components = list(component_cols or _infer_component_columns(parsed_rows, time_col))
    if not selected_components:
        raise ValueError("component_cols must contain at least one numeric component column")
    for row in parsed_rows:
        missing = [column for column in selected_components if column not in row]
        if missing:
            names = ", ".join(missing)
            raise ValueError(f"component rows must contain column(s): {names}")

    height = max(2.8, 2.35 * len(selected_components))
    figure, axes = pyplot.subplots(
        len(selected_components),
        1,
        figsize=figure_size or (9.0, height),
        sharex=True,
        squeeze=False,
    )
    times = [row[time_col] for row in parsed_rows]
    for index, component in enumerate(selected_components):
        axis = axes[index][0]
        values = [_finite_float(row[component], component) for row in parsed_rows]
        axis.plot(
            times,
            values,
            color=DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
            linewidth=1.9,
        )
        axis.axhline(0.0, color="#9ca3af", linewidth=0.8, linestyle="--")
        if changepoints and index == 0:
            for changepoint in changepoints:
                axis.axvline(changepoint, color="#6b7280", linewidth=0.9, linestyle=":", alpha=0.72)
        axis.set_ylabel(component)
        axis.grid(alpha=0.22)

    axes[0][0].set_title(title if series_id is None else f"{title}: {series_id}")
    axes[-1][0].set_xlabel(time_col)
    _tight_layout(figure)
    return figure


def save_figure(figure: Any, path: str | Path, *, dpi: int = 160, close: bool = False) -> Path:
    """Save a Matplotlib figure and return the written path."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if close:
        _require_pyplot().close(figure)
    return output


def write_plot_report(
    output_dir: str | Path,
    *,
    predicted_actual: tuple[Sequence[float], Sequence[float]] | None = None,
    forecast: Any | None = None,
    history: Any | None = None,
    components: Any | None = None,
    metric_rows: Any | None = None,
    horizon_rows: Any | None = None,
    backtest_rows: Any | None = None,
    interval_rows: Any | None = None,
    seasonality_rows: Any | None = None,
    changepoint_rows: Any | None = None,
    cutoff_rows: Any | None = None,
    prefix: str = "cartoboost",
    dpi: int = 160,
) -> dict[str, str]:
    """Write a standard diagnostic plot bundle and return paths by plot name."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def write(name: str, figure: Any) -> None:
        path = save_figure(figure, output / f"{prefix}_{name}.png", dpi=dpi, close=True)
        written[name] = str(path)

    if predicted_actual is not None:
        y_true, y_pred = predicted_actual
        write("predicted_actual", plot_predicted_actual(y_true, y_pred))
        write("residual_diagnostics", plot_residual_diagnostics(y_true, y_pred))
    if forecast is not None:
        write("forecast", plot_forecast(forecast, history=history))
    if components is not None:
        write("forecast_components", plot_forecast_components(components))
    if metric_rows is not None:
        write("metric_comparison", plot_metric_comparison(metric_rows))
    if horizon_rows is not None:
        write("horizon_metrics", plot_horizon_metrics(horizon_rows))
    if backtest_rows is not None:
        write("backtest_metrics", plot_backtest_metrics(backtest_rows))
    if interval_rows is not None:
        write("interval_calibration", plot_interval_calibration(interval_rows))
    if seasonality_rows is not None:
        write("seasonality_curve", plot_seasonality_curve(seasonality_rows))
    if changepoint_rows is not None:
        write("changepoint_effects", plot_changepoint_effects(changepoint_rows))
    if cutoff_rows is not None:
        write("cutoff_predictions", plot_cutoff_predictions(cutoff_rows))
    if not written:
        raise ValueError("at least one plot input must be provided")
    return written


def _require_pyplot() -> Any:
    try:
        import matplotlib.pyplot as pyplot
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "CartoBoost visualization requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from exc
    return pyplot


def _require_pandas() -> Any:
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Prophet-compatible plotting requires pandas. Install it with "
            "`uv sync --group dev`, the benchmark dependency group, or a "
            "pandas-enabled environment."
        ) from exc
    return pandas


def _require_plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "Prophet-compatible interactive plotting requires plotly. Install plotly to use "
            "`plot_plotly`, `plot_components_plotly`, and related helpers."
        ) from exc
    return go, make_subplots


def _require_static_map_stack() -> tuple[Any, Any, Any]:
    try:
        import geopandas
        from shapely.geometry import LineString, Point
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "CartoBoost map visualization requires geopandas and shapely. "
            "Install them with `pip install geopandas shapely`."
        ) from exc
    return geopandas, Point, LineString


def _require_pydeck_stack() -> Any:
    try:
        import pydeck
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "CartoBoost interactive map visualization requires pydeck. "
            "Install it with `pip install pydeck`."
        ) from exc
    return pydeck


def _figure_axis(pyplot: Any, *, ax: Any | None, figsize: tuple[float, float]) -> tuple[Any, Any]:
    if ax is not None:
        return ax.figure, ax
    return pyplot.subplots(figsize=figsize)


def _format_prophet_date_axis(ax: Any) -> None:
    try:
        from matplotlib.dates import AutoDateFormatter, AutoDateLocator
    except ImportError:  # pragma: no cover - matplotlib already required by caller
        return
    locator = AutoDateLocator(interval_multiples=False)
    formatter = AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _format_month_axis(ax: Any) -> None:
    try:
        from matplotlib.dates import MonthLocator, num2date
        from matplotlib.ticker import FuncFormatter
    except ImportError:  # pragma: no cover - matplotlib already required by caller
        return
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, pos=None: f"{num2date(x):%B} {num2date(x).day}")
    )
    ax.xaxis.set_major_locator(MonthLocator(range(1, 13), bymonthday=1, interval=2))


def _format_seasonality_axis(
    ax: Any,
    name: str,
    period: float,
    start: Any,
    end: Any,
) -> None:
    pandas = _require_pandas()
    try:
        from matplotlib.dates import num2date
        from matplotlib.ticker import FuncFormatter
    except ImportError:  # pragma: no cover - matplotlib already required by caller
        return
    n_ticks = 8
    ticks = pandas.to_datetime(np.linspace(start.value, end.value, n_ticks)).to_pydatetime()
    ax.set_xticks(ticks)
    if name == "yearly":
        formatter = FuncFormatter(lambda x, pos=None: f"{num2date(x):%B} {num2date(x).day}")
        ax.set_xlabel("Day of year")
    elif name == "weekly":
        formatter = FuncFormatter(lambda x, pos=None: f"{num2date(x):%A}")
        ax.set_xlabel("Day of Week")
    elif name == "daily" or period <= 2:
        formatter = FuncFormatter(lambda x, pos=None: f"{num2date(x):%T}")
        ax.set_xlabel("Hour of day" if name == "daily" else "Hours")
    else:
        formatter = FuncFormatter(lambda x, pos=None: f"{pos * period / (n_ticks - 1):.0f}")
        ax.set_xlabel("Days")
    ax.xaxis.set_major_formatter(formatter)


def _tight_layout(figure: Any) -> None:
    try:
        figure.tight_layout()
    except RuntimeError:
        pass


def _paired_numeric_arrays(
    left: Sequence[float],
    right: Sequence[float],
    left_name: str,
    right_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError(f"{left_name} and {right_name} must have the same shape")
    if left_array.ndim != 1:
        raise ValueError(f"{left_name} and {right_name} must be one-dimensional")
    if left_array.size == 0:
        raise ValueError(f"{left_name} and {right_name} must contain at least one value")
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError(f"{left_name} and {right_name} must contain only finite values")
    return left_array, right_array


def _rows_from_table(table: Any) -> list[dict[str, Any]]:
    if table is None:
        return []
    if hasattr(table, "to_dict"):
        try:
            records = table.to_dict(orient="records")
        except TypeError:
            records = table.to_dict()
        if isinstance(records, Mapping):
            keys = list(records)
            if not keys:
                return []
            length = len(records[keys[0]])
            return [{key: records[key][index] for key in keys} for index in range(length)]
        return [dict(row) for row in records]
    if isinstance(table, Mapping):
        keys = list(table)
        if not keys:
            return []
        length = len(table[keys[0]])
        return [{key: table[key][index] for key in keys} for index in range(length)]
    return [dict(row) for row in table]


def _require_prophet_forecast_columns(frame: Any, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"forecast rows must contain Prophet column(s): {names}")


def _prophet_performance_metrics(df_cv: Any, metric: str, rolling_window: float) -> Any:
    pandas = _require_pandas()
    valid = {"mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"}
    if metric not in valid:
        raise ValueError(f"metric must be one of {sorted(valid)}")
    frame = df_cv.copy() if hasattr(df_cv, "copy") else pandas.DataFrame(df_cv)
    required = {"ds", "cutoff", "y", "yhat"}
    missing = required.difference(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"cross-validation rows must contain column(s): {names}")
    frame["ds"] = pandas.to_datetime(frame["ds"])
    frame["cutoff"] = pandas.to_datetime(frame["cutoff"])
    frame["horizon"] = frame["ds"] - frame["cutoff"]
    frame = frame.sort_values("horizon")
    if metric == "coverage" and not {"yhat_lower", "yhat_upper"}.issubset(frame.columns):
        raise ValueError("coverage metric requires 'yhat_lower' and 'yhat_upper'")
    if metric == "mape" and frame["y"].abs().min() < 1e-8:
        raise ValueError("mape metric is undefined when y is close to 0")
    window = int(rolling_window * frame.shape[0])
    if window >= 0:
        window = max(window, 1)
        window = min(window, frame.shape[0])

    error = frame["y"] - frame["yhat"]
    abs_error = error.abs()
    squared_error = error**2
    if metric == "mse":
        values = squared_error
        result = (
            pandas.DataFrame({"horizon": frame["horizon"], metric: values})
            if window < 0
            else _rolling_mean_by_h(values.values, frame["horizon"].values, window, metric)
        )
    elif metric == "rmse":
        result = (
            pandas.DataFrame({"horizon": frame["horizon"], "rmse": np.sqrt(squared_error)})
            if window < 0
            else _rolling_mean_by_h(squared_error.values, frame["horizon"].values, window, "rmse")
        )
        if window >= 0:
            result["rmse"] = np.sqrt(result["rmse"])
    elif metric == "mae":
        result = (
            pandas.DataFrame({"horizon": frame["horizon"], metric: abs_error})
            if window < 0
            else _rolling_mean_by_h(abs_error.values, frame["horizon"].values, window, metric)
        )
    elif metric == "mape":
        values = np.abs((frame["y"] - frame["yhat"]) / frame["y"])
        result = (
            pandas.DataFrame({"horizon": frame["horizon"], metric: values})
            if window < 0
            else _rolling_mean_by_h(values.values, frame["horizon"].values, window, metric)
        )
    elif metric == "mdape":
        values = np.abs((frame["y"] - frame["yhat"]) / frame["y"])
        result = (
            pandas.DataFrame({"horizon": frame["horizon"], metric: values})
            if window < 0
            else _rolling_median_by_h(values.values, frame["horizon"], window, metric)
        )
    elif metric == "smape":
        values = abs_error / ((frame["y"].abs() + frame["yhat"].abs()) / 2)
        values = values.fillna(0)
        result = (
            pandas.DataFrame({"horizon": frame["horizon"], metric: values})
            if window < 0
            else _rolling_mean_by_h(values.values, frame["horizon"].values, window, metric)
        )
    else:
        values = (frame["y"] >= frame["yhat_lower"]) & (frame["y"] <= frame["yhat_upper"])
        result = (
            pandas.DataFrame({"horizon": frame["horizon"], metric: values})
            if window < 0
            else _rolling_mean_by_h(values.values, frame["horizon"].values, window, metric)
        )
    return result


def _timedelta_plot_values(values: Any, *, unit_name: str | None = None) -> tuple[np.ndarray, str]:
    pandas = _require_pandas()
    timedeltas = pandas.to_timedelta(values)
    unit_names: list[tuple[str, float, str]] = [
        ("days", 24 * 60 * 60 * 10**9, "D"),
        ("hours", 60 * 60 * 10**9, "h"),
        ("minutes", 60 * 10**9, "m"),
        ("seconds", 10**9, "s"),
        ("milliseconds", 10**6, "ms"),
        ("microseconds", 10**3, "us"),
        ("nanoseconds", 1.0, "ns"),
    ]
    if unit_name is None:
        tick_width = float(max(timedeltas.astype("timedelta64[ns]").astype(np.int64))) / 10.0
        unit_name = "nanoseconds"
        for candidate, divisor, _unit in unit_names:
            if divisor < tick_width:
                unit_name = candidate
                break
    divisor = {name: divisor for name, divisor, _ in unit_names}[unit_name]
    return timedeltas.astype("timedelta64[ns]").astype(np.int64) / float(divisor), unit_name


def _rolling_mean_by_h(x: Any, h: Any, w: int, name: str) -> Any:
    pandas = _require_pandas()
    grouped = (
        pandas.DataFrame({"x": x, "h": h})
        .groupby("h")
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("h")
    )
    sums = grouped["x"]["sum"].values
    counts = grouped["x"]["count"].values
    horizons = grouped.h.values
    trailing_i = len(grouped) - 1
    x_sum = 0.0
    n_sum = 0
    res_x = np.empty(len(grouped))
    for i in range(len(grouped) - 1, -1, -1):
        x_sum += sums[i]
        n_sum += counts[i]
        while n_sum >= w:
            excess_n = n_sum - w
            excess_x = excess_n * sums[i] / counts[i]
            res_x[trailing_i] = (x_sum - excess_x) / w
            x_sum -= sums[trailing_i]
            n_sum -= counts[trailing_i]
            trailing_i -= 1
    return pandas.DataFrame(
        {"horizon": horizons[(trailing_i + 1) :], name: res_x[(trailing_i + 1) :]}
    )


def _rolling_median_by_h(x: Any, h: Any, w: int, name: str) -> Any:
    pandas = _require_pandas()
    frame = pandas.DataFrame({"x": x, "h": h})
    grouped = frame.groupby("h")
    horizons = grouped.size().reset_index().sort_values("h")["h"]
    res_h = []
    res_x = []
    i = len(horizons) - 1
    while i >= 0:
        horizon = horizons[i]
        values = grouped.get_group(horizon).x.tolist()
        next_idx_to_add = np.array(h == horizon).argmax() - 1
        while len(values) < w and next_idx_to_add >= 0:
            values.append(x[next_idx_to_add])
            next_idx_to_add -= 1
        if len(values) < w:
            break
        res_h.append(horizons[i])
        res_x.append(np.median(values))
        i -= 1
    res_h.reverse()
    res_x.reverse()
    return pandas.DataFrame({"horizon": res_h, name: res_x})


def _infer_component_columns(rows: Sequence[Mapping[str, Any]], time_col: str) -> list[str]:
    reserved = {
        time_col,
        "timestamp",
        "series_id",
        "horizon",
        "actual",
        "prediction",
        "mean",
        "lower",
        "upper",
    }
    columns: list[str] = []
    for column in rows[0]:
        if column in reserved or column.startswith("lower") or column.startswith("upper"):
            continue
        try:
            [_finite_float(row[column], column) for row in rows]
        except ValueError:
            continue
        columns.append(str(column))
    return columns


def _filter_series(
    rows: Sequence[Mapping[str, Any]],
    series_id: str | None,
    series_id_col: str,
) -> list[dict[str, Any]]:
    if series_id is None:
        return [dict(row) for row in rows]
    return [dict(row) for row in rows if str(row.get(series_id_col)) == series_id]


def _finite_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} values must be numeric") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} values must be finite")
    return parsed


def _validate_coordinate_rows(
    rows: Sequence[Mapping[str, Any]],
    latitude_col: str,
    longitude_col: str,
) -> None:
    for row in rows:
        if latitude_col not in row or longitude_col not in row:
            raise ValueError(f"rows must contain '{latitude_col}' and '{longitude_col}'")
        _finite_float(row[latitude_col], latitude_col)
        _finite_float(row[longitude_col], longitude_col)


def _mean_lat_lon(
    rows: Sequence[Mapping[str, Any]],
    latitude_col: str,
    longitude_col: str,
) -> tuple[float, float]:
    latitudes = [_finite_float(row[latitude_col], latitude_col) for row in rows]
    longitudes = [_finite_float(row[longitude_col], longitude_col) for row in rows]
    return float(np.mean(latitudes)), float(np.mean(longitudes))


def _pydeck_color_expression(
    value_col: str | None,
    *,
    positive: tuple[int, int, int] = (37, 99, 235),
) -> list[int] | str:
    if value_col is None:
        return [positive[0], positive[1], positive[2], 190]
    return (
        f"{value_col} >= 0 ? [{positive[0]}, {positive[1]}, {positive[2]}, 210] "
        f": [220, 38, 38, 210]"
    )


def _pydeck_radius_expression(value_col: str | None, radius_scale: float) -> float | str:
    if value_col is None:
        return radius_scale
    return f"Math.max(1, Math.abs({value_col})) * {radius_scale}"


def _pydeck_tooltip(columns: Sequence[str]) -> Mapping[str, str] | None:
    if not columns:
        return None
    lines = [f"{column}: {{{column}}}" for column in columns]
    return {"text": "\n".join(lines)}


def _write_pydeck_html(deck: Any, output_html: str | Path) -> Path:
    output = Path(output_html)
    output.parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(str(output), open_browser=False)
    return output


__all__ = [
    "DEFAULT_PALETTE",
    "add_changepoints_to_plot",
    "get_forecast_component_plotly_props",
    "get_seasonality_plotly_props",
    "plot",
    "plot_backtest_metrics",
    "plot_changepoint_effects",
    "plot_components",
    "plot_components_plotly",
    "plot_cutoff_predictions",
    "plot_forecast",
    "plot_forecast_component",
    "plot_forecast_component_plotly",
    "plot_forecast_components",
    "plot_horizon_metrics",
    "plot_interval_calibration",
    "plot_metric_comparison",
    "plot_plotly",
    "plot_predicted_actual",
    "plot_residual_diagnostics",
    "plot_route_segments",
    "plot_seasonality",
    "plot_seasonality_plotly",
    "plot_seasonality_curve",
    "plot_spatial_points",
    "plot_cross_validation_metric",
    "plot_weekly",
    "plot_yearly",
    "save_figure",
    "seasonality_plot_df",
    "set_y_as_percent",
    "write_pydeck_point_map",
    "write_pydeck_route_map",
    "write_plot_report",
]
