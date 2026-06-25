from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from cartoboost.plotting import (
    add_changepoints_to_plot,
    get_forecast_component_plotly_props,
    get_seasonality_plotly_props,
    plot,
    plot_backtest_metrics,
    plot_changepoint_effects,
    plot_components,
    plot_components_plotly,
    plot_cross_validation_metric,
    plot_cutoff_predictions,
    plot_forecast,
    plot_forecast_component,
    plot_forecast_component_plotly,
    plot_forecast_components,
    plot_horizon_metrics,
    plot_interval_calibration,
    plot_metric_comparison,
    plot_plotly,
    plot_predicted_actual,
    plot_residual_diagnostics,
    plot_route_segments,
    plot_seasonality,
    plot_seasonality_curve,
    plot_seasonality_plotly,
    plot_spatial_points,
    plot_weekly,
    plot_yearly,
    save_figure,
    seasonality_plot_df,
    set_y_as_percent,
    write_plot_report,
    write_pydeck_point_map,
)

PROPHET_PLOT_122_PUBLIC_UTILITIES = {
    "plot",
    "plot_components",
    "plot_forecast_component",
    "seasonality_plot_df",
    "plot_weekly",
    "plot_yearly",
    "plot_seasonality",
    "set_y_as_percent",
    "add_changepoints_to_plot",
    "plot_cross_validation_metric",
    "plot_plotly",
    "plot_components_plotly",
    "plot_forecast_component_plotly",
    "plot_seasonality_plotly",
    "get_forecast_component_plotly_props",
    "get_seasonality_plotly_props",
}


PROPHET_PLOT_122_SIGNATURES = {
    "plot": (
        "m",
        "fcst",
        "ax=None",
        "uncertainty=True",
        "plot_cap=True",
        "xlabel='ds'",
        "ylabel='y'",
        "figsize=(10, 6)",
        "include_legend=False",
    ),
    "plot_components": (
        "m",
        "fcst",
        "uncertainty=True",
        "plot_cap=True",
        "weekly_start=0",
        "yearly_start=0",
        "figsize=None",
    ),
    "plot_forecast_component": (
        "m",
        "fcst",
        "name",
        "ax=None",
        "uncertainty=True",
        "plot_cap=False",
        "figsize=(10, 6)",
    ),
    "seasonality_plot_df": ("m", "ds"),
    "plot_weekly": (
        "m",
        "ax=None",
        "uncertainty=True",
        "weekly_start=0",
        "figsize=(10, 6)",
        "name='weekly'",
    ),
    "plot_yearly": (
        "m",
        "ax=None",
        "uncertainty=True",
        "yearly_start=0",
        "figsize=(10, 6)",
        "name='yearly'",
    ),
    "plot_seasonality": ("m", "name", "ax=None", "uncertainty=True", "figsize=(10, 6)"),
    "set_y_as_percent": ("ax",),
    "add_changepoints_to_plot": (
        "ax",
        "m",
        "fcst",
        "threshold=0.01",
        "cp_color='r'",
        "cp_linestyle='--'",
        "trend=True",
    ),
    "plot_cross_validation_metric": (
        "df_cv",
        "metric",
        "rolling_window=0.1",
        "ax=None",
        "figsize=(10, 6)",
        "color='b'",
        "point_color='gray'",
    ),
    "plot_plotly": (
        "m",
        "fcst",
        "uncertainty=True",
        "plot_cap=True",
        "trend=False",
        "changepoints=False",
        "changepoints_threshold=0.01",
        "xlabel='ds'",
        "ylabel='y'",
        "figsize=(900, 600)",
    ),
    "plot_components_plotly": (
        "m",
        "fcst",
        "uncertainty=True",
        "plot_cap=True",
        "figsize=(900, 200)",
    ),
    "plot_forecast_component_plotly": (
        "m",
        "fcst",
        "name",
        "uncertainty=True",
        "plot_cap=False",
        "figsize=(900, 300)",
    ),
    "plot_seasonality_plotly": (
        "m",
        "name",
        "uncertainty=True",
        "figsize=(900, 300)",
    ),
    "get_forecast_component_plotly_props": (
        "m",
        "fcst",
        "name",
        "uncertainty=True",
        "plot_cap=False",
    ),
    "get_seasonality_plotly_props": ("m", "name", "uncertainty=True"),
}


class ProphetLikeModel(SimpleNamespace):
    def setup_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["ds"] = pd.to_datetime(frame["ds"])
        return frame

    def predict_seasonal_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        phase = np.arange(len(frame), dtype=float)
        values = {
            "weekly": np.sin(phase / max(1.0, len(frame) - 1.0) * 2.0 * np.pi),
            "yearly": np.cos(phase / max(1.0, len(frame) - 1.0) * 2.0 * np.pi),
            "daily": np.sin(phase / max(1.0, len(frame) - 1.0) * 4.0 * np.pi),
        }
        output = {}
        for name, series in values.items():
            output[name] = series
            output[f"{name}_lower"] = series - 0.1
            output[f"{name}_upper"] = series + 0.1
        return pd.DataFrame(output)


def prophet_like_inputs() -> tuple[ProphetLikeModel, pd.DataFrame]:
    ds = pd.date_range("2026-01-01", periods=8, freq="D")
    history = pd.DataFrame({"ds": ds[:5], "y": [10.0, 11.0, 13.0, 14.0, 15.0]})
    fcst = pd.DataFrame(
        {
            "ds": ds,
            "yhat": np.linspace(10.0, 18.0, len(ds)),
            "yhat_lower": np.linspace(9.0, 17.0, len(ds)),
            "yhat_upper": np.linspace(11.0, 19.0, len(ds)),
            "trend": np.linspace(10.0, 16.0, len(ds)),
            "trend_lower": np.linspace(9.5, 15.5, len(ds)),
            "trend_upper": np.linspace(10.5, 16.5, len(ds)),
            "weekly": np.sin(np.arange(len(ds))),
            "weekly_lower": np.sin(np.arange(len(ds))) - 0.1,
            "weekly_upper": np.sin(np.arange(len(ds))) + 0.1,
            "yearly": np.cos(np.arange(len(ds))),
            "yearly_lower": np.cos(np.arange(len(ds))) - 0.1,
            "yearly_upper": np.cos(np.arange(len(ds))) + 0.1,
            "daily": np.sin(np.arange(len(ds)) / 2.0),
            "daily_lower": np.sin(np.arange(len(ds)) / 2.0) - 0.1,
            "daily_upper": np.sin(np.arange(len(ds)) / 2.0) + 0.1,
            "extra_regressors_additive": np.linspace(0.0, 1.0, len(ds)),
            "extra_regressors_additive_lower": np.linspace(-0.1, 0.9, len(ds)),
            "extra_regressors_additive_upper": np.linspace(0.1, 1.1, len(ds)),
        }
    )
    model = ProphetLikeModel(
        history=history,
        uncertainty_samples=100,
        logistic_floor=False,
        train_holiday_names=None,
        seasonalities={
            "weekly": {"period": 7, "mode": "additive", "condition_name": None},
            "yearly": {"period": 365.25, "mode": "additive", "condition_name": None},
            "daily": {"period": 1, "mode": "additive", "condition_name": None},
        },
        extra_regressors={"weather": {"mode": "additive"}},
        component_modes={"multiplicative": []},
        changepoints=pd.Series([pd.Timestamp("2026-01-03"), pd.Timestamp("2026-01-06")]),
        params={"delta": np.asarray([[0.02, 0.0]])},
    )
    return model, fcst


def assert_writes_png(tmp_path: Path, figure, name: str) -> None:
    path = save_figure(figure, tmp_path / name, close=True)
    assert path.exists()
    assert path.stat().st_size > 0


def normalized_signature(callable_obj) -> tuple[str, ...]:
    import inspect

    output = []
    for parameter in inspect.signature(callable_obj).parameters.values():
        item = parameter.name
        if parameter.default is not inspect.Parameter.empty:
            item = f"{item}={parameter.default!r}"
        output.append(item)
    return tuple(output)


def test_predicted_actual_and_residual_diagnostics_write_figures(tmp_path: Path) -> None:
    actual = [12.0, 16.0, 22.0, 27.0, 33.0]
    predicted = [11.0, 17.0, 20.0, 29.0, 31.0]

    assert_writes_png(tmp_path, plot_predicted_actual(actual, predicted), "predicted_actual.png")
    assert_writes_png(tmp_path, plot_residual_diagnostics(actual, predicted), "residuals.png")


def test_metric_and_horizon_comparison_write_figures(tmp_path: Path) -> None:
    metric_rows = [
        {"model": "cartoboost", "rmse": 4.1},
        {"model": "lightgbm", "rmse": 4.8},
        {"model": "mean", "rmse": 9.4},
    ]
    horizon_rows = [
        {"model": "cartoboost", "horizon": 1, "rmse": 2.1},
        {"model": "cartoboost", "horizon": 2, "rmse": 2.8},
        {"model": "lightgbm", "horizon": 1, "rmse": 2.3},
        {"model": "lightgbm", "horizon": 2, "rmse": 3.0},
    ]

    assert_writes_png(tmp_path, plot_metric_comparison(metric_rows), "metrics.png")
    assert_writes_png(tmp_path, plot_horizon_metrics(horizon_rows), "horizon.png")


def test_backtest_and_interval_calibration_plots_write_figures(tmp_path: Path) -> None:
    backtest_rows = [
        {"model": "cartoboost", "fold": 1, "rmse": 2.4},
        {"model": "cartoboost", "fold": 2, "rmse": 2.7},
        {"model": "seasonal_naive", "fold": 1, "rmse": 3.2},
        {"model": "seasonal_naive", "fold": 2, "rmse": 3.5},
    ]
    interval_rows = [
        {"nominal_coverage": 0.5, "coverage": 0.48, "mean_width": 4.2},
        {"nominal_coverage": 0.8, "coverage": 0.77, "mean_width": 7.9},
        {"nominal_coverage": 0.9, "coverage": 0.88, "mean_width": 9.4},
    ]

    assert_writes_png(tmp_path, plot_backtest_metrics(backtest_rows), "backtest.png")
    assert_writes_png(tmp_path, plot_interval_calibration(interval_rows), "intervals.png")


def test_component_diagnostics_write_figures(tmp_path: Path) -> None:
    seasonality_rows = [
        {"component": "weekly", "phase": 0, "value": -2.0, "lower": -3.0, "upper": -1.0},
        {"component": "weekly", "phase": 1, "value": 1.5, "lower": 0.5, "upper": 2.5},
        {"component": "weekly", "phase": 2, "value": 2.0, "lower": 1.0, "upper": 3.0},
    ]
    changepoint_rows = [
        {"timestamp": "2026-01-03", "delta": 2.5},
        {"timestamp": "2026-01-10", "delta": -1.2},
    ]
    cutoff_rows = [
        {"cutoff": "fold_1", "timestamp": "2026-01-03", "actual": 45.0, "prediction": 44.0},
        {"cutoff": "fold_1", "timestamp": "2026-01-04", "actual": 48.0, "prediction": 47.0},
        {"cutoff": "fold_2", "timestamp": "2026-01-04", "actual": 48.0, "prediction": 49.0},
        {"cutoff": "fold_2", "timestamp": "2026-01-05", "actual": 51.0, "prediction": 50.0},
    ]

    assert_writes_png(
        tmp_path,
        plot_seasonality_curve(
            seasonality_rows,
            lower_col="lower",
            upper_col="upper",
            label_col="component",
        ),
        "seasonality.png",
    )
    assert_writes_png(tmp_path, plot_changepoint_effects(changepoint_rows), "changepoints.png")
    assert_writes_png(tmp_path, plot_cutoff_predictions(cutoff_rows), "cutoffs.png")


def test_plot_report_writes_named_diagnostic_bundle(tmp_path: Path) -> None:
    written = write_plot_report(
        tmp_path / "report",
        predicted_actual=([12.0, 15.0, 18.0], [11.5, 15.8, 17.4]),
        metric_rows=[
            {"model": "cartoboost", "rmse": 2.2},
            {"model": "mean", "rmse": 6.8},
        ],
        seasonality_rows=[
            {"phase": 0, "value": -1.0},
            {"phase": 1, "value": 1.0},
        ],
        prefix="taxi",
    )

    assert set(written) == {
        "metric_comparison",
        "predicted_actual",
        "residual_diagnostics",
        "seasonality_curve",
    }
    for path in written.values():
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_map_visualization_helpers_have_clear_optional_extra_errors(tmp_path: Path) -> None:
    with pytest.raises(ImportError, match=r"cartoboost\[visualization\]"):
        plot_spatial_points([{"latitude": 40.7, "longitude": -73.9, "value": 12.0}])

    with pytest.raises(ImportError, match=r"cartoboost\[visualization\]"):
        plot_route_segments(
            [
                {
                    "pickup_latitude": 40.7,
                    "pickup_longitude": -73.9,
                    "dropoff_latitude": 40.75,
                    "dropoff_longitude": -73.98,
                }
            ]
        )

    with pytest.raises(ImportError, match=r"cartoboost\[visualization\]"):
        write_pydeck_point_map(
            [{"latitude": 40.7, "longitude": -73.9}],
            tmp_path / "points.html",
        )


def test_forecast_plot_supports_history_actuals_and_intervals(tmp_path: Path) -> None:
    history = [
        {"series_id": "pickup_1", "timestamp": "2026-01-01", "actual": 41.0},
        {"series_id": "pickup_1", "timestamp": "2026-01-02", "actual": 45.0},
    ]
    forecast = [
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-03",
            "prediction": 47.0,
            "actual": 48.0,
            "lower": 43.0,
            "upper": 51.0,
        },
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-04",
            "prediction": 49.0,
            "actual": 50.0,
            "lower": 44.0,
            "upper": 54.0,
        },
    ]

    figure = plot_forecast(
        forecast,
        history=history,
        series_id="pickup_1",
        changepoints=["2026-01-03"],
    )
    assert_writes_png(tmp_path, figure, "forecast.png")


def test_component_plot_supports_trend_seasonality_and_changepoints(tmp_path: Path) -> None:
    rows = [
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-01",
            "trend": 40.0,
            "weekly": -2.0,
            "event": 0.0,
        },
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-02",
            "trend": 42.0,
            "weekly": 1.5,
            "event": 3.0,
        },
        {
            "series_id": "pickup_1",
            "timestamp": "2026-01-03",
            "trend": 44.0,
            "weekly": 2.0,
            "event": 0.0,
        },
    ]

    figure = plot_forecast_components(
        rows,
        component_cols=["trend", "weekly", "event"],
        series_id="pickup_1",
        changepoints=["2026-01-02"],
    )
    assert_writes_png(tmp_path, figure, "components.png")


def test_prophet_plotting_public_surface_matches_upstream_122() -> None:
    import cartoboost.plotting as plotting

    assert PROPHET_PLOT_122_PUBLIC_UTILITIES.issubset(set(plotting.__all__))
    for name in PROPHET_PLOT_122_PUBLIC_UTILITIES:
        assert callable(getattr(plotting, name))
        assert normalized_signature(getattr(plotting, name)) == PROPHET_PLOT_122_SIGNATURES[name]


def test_prophet_forecast_plot_matches_upstream_artist_contract() -> None:
    model, fcst = prophet_like_inputs()

    figure = plot(model, fcst)
    axis = figure.axes[0]

    observed, forecast = axis.lines[:2]
    assert pd.to_datetime(observed.get_xdata()).tolist() == model.history["ds"].tolist()
    assert observed.get_ydata().tolist() == model.history["y"].tolist()
    assert observed.get_marker() == "."
    assert observed.get_color() == "k"
    assert pd.to_datetime(forecast.get_xdata()).tolist() == fcst["ds"].tolist()
    assert forecast.get_ydata().tolist() == fcst["yhat"].tolist()
    assert forecast.get_color() == "#0072B2"
    assert forecast.get_linestyle() == "-"
    assert axis.get_xlabel() == "ds"
    assert axis.get_ylabel() == "y"
    assert axis.get_legend() is None
    assert axis.collections


def test_prophet_forecast_component_floor_display_and_return_match_upstream() -> None:
    model, fcst = prophet_like_inputs()
    model.logistic_floor = True
    fcst = fcst.assign(cap=np.linspace(20.0, 24.0, len(fcst)), floor=0.0)

    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots()
    artists = plot_forecast_component(model, fcst, "trend", ax=axis, plot_cap=True)

    assert len(axis.lines) == 3
    assert [line.get_ydata().tolist() for line in axis.lines] == [
        fcst["trend"].tolist(),
        fcst["cap"].tolist(),
        fcst["floor"].tolist(),
    ]
    assert len(artists) == 3
    assert artists[0] is axis.lines[0]
    assert artists[1] is axis.lines[1]
    assert artists[2] in axis.collections
    pyplot.close(figure)


def test_prophet_compatible_helpers_require_prophet_shaped_model_attributes() -> None:
    model, fcst = prophet_like_inputs()
    incomplete_model = SimpleNamespace(history=model.history)

    with pytest.raises(AttributeError):
        plot(incomplete_model, fcst)
    with pytest.raises(AttributeError):
        plot_forecast_component(incomplete_model, fcst, "trend")
    with pytest.raises(AttributeError):
        seasonality_plot_df(SimpleNamespace(extra_regressors={}, seasonalities={}), fcst["ds"])


def test_prophet_compatible_matplotlib_helpers_write_figures(tmp_path: Path) -> None:
    model, fcst = prophet_like_inputs()

    assert_writes_png(tmp_path, plot(model, fcst, include_legend=True), "prophet_plot.png")
    assert_writes_png(tmp_path, plot_components(model, fcst), "prophet_components.png")

    import matplotlib.pyplot as pyplot

    component_fig, component_ax = pyplot.subplots()
    artists = plot_forecast_component(model, fcst, "trend", ax=component_ax)
    assert artists
    assert_writes_png(tmp_path, component_fig, "prophet_component.png")

    weekly_fig, weekly_ax = pyplot.subplots()
    assert plot_weekly(model, ax=weekly_ax)
    assert_writes_png(tmp_path, weekly_fig, "prophet_weekly.png")

    yearly_fig, yearly_ax = pyplot.subplots()
    assert plot_yearly(model, ax=yearly_ax)
    assert_writes_png(tmp_path, yearly_fig, "prophet_yearly.png")

    seasonality_fig, seasonality_ax = pyplot.subplots()
    assert plot_seasonality(model, "daily", ax=seasonality_ax)
    assert_writes_png(tmp_path, seasonality_fig, "prophet_daily.png")

    changepoint_fig, changepoint_ax = pyplot.subplots()
    plot(model, fcst, ax=changepoint_ax)
    artists = add_changepoints_to_plot(changepoint_ax, model, fcst, threshold=0.01)
    assert len(artists) == 2
    assert_writes_png(tmp_path, changepoint_fig, "prophet_changepoints.png")

    percent_fig, percent_ax = pyplot.subplots()
    percent_ax.plot([0, 1], [0.1, 0.2])
    assert set_y_as_percent(percent_ax) is percent_ax
    assert_writes_png(tmp_path, percent_fig, "prophet_percent.png")


def test_prophet_compatible_seasonality_dataframe_and_cv_metric(tmp_path: Path) -> None:
    model, _ = prophet_like_inputs()
    frame = seasonality_plot_df(model, pd.date_range("2026-02-01", periods=3, freq="D"))

    assert list(frame.columns) == ["ds", "cap", "floor", "weather"]
    assert frame["weather"].tolist() == [0.0, 0.0, 0.0]

    cv_rows = pd.DataFrame(
        {
            "ds": pd.to_datetime(["2026-01-08", "2026-01-09", "2026-01-10"]),
            "cutoff": pd.to_datetime(["2026-01-05", "2026-01-05", "2026-01-05"]),
            "y": [12.0, 14.0, 15.0],
            "yhat": [11.0, 13.0, 16.0],
            "yhat_lower": [10.0, 12.0, 14.0],
            "yhat_upper": [13.0, 15.0, 17.0],
        }
    )
    figure = plot_cross_validation_metric(cv_rows, "rmse", rolling_window=0.5)
    assert_writes_png(tmp_path, figure, "prophet_cv_rmse.png")


def test_prophet_cross_validation_metric_uses_upstream_horizon_grouped_rolling() -> None:
    cv_rows = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-09",
                    "2026-01-10",
                ]
            ),
            "cutoff": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-06",
                ]
            ),
            "y": [10.0, 13.0, 15.0, 20.0, 23.0, 27.0],
            "yhat": [9.0, 11.0, 14.0, 16.0, 20.0, 22.0],
        }
    )

    figure = plot_cross_validation_metric(cv_rows, "mse", rolling_window=0.5)
    axis = figure.axes[0]

    assert axis.get_xlabel() == "Horizon (hours)"
    assert axis.lines[1].get_xdata().tolist() == [72.0, 96.0]
    assert axis.lines[1].get_ydata().tolist() == pytest.approx([14.0 / 3.0, 47.5 / 3.0])


def test_prophet_cross_validation_metric_internals_cover_all_upstream_metrics() -> None:
    import cartoboost.plotting as plotting

    cv_rows = pd.DataFrame(
        {
            "ds": pd.to_datetime(
                [
                    "2026-01-07",
                    "2026-01-08",
                    "2026-01-08",
                    "2026-01-09",
                    "2026-01-09",
                    "2026-01-10",
                ]
            ),
            "cutoff": pd.to_datetime(
                [
                    "2026-01-05",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-05",
                    "2026-01-06",
                    "2026-01-06",
                ]
            ),
            "y": [10.0, 13.0, 15.0, 20.0, 23.0, 27.0],
            "yhat": [9.0, 11.0, 14.0, 16.0, 20.0, 22.0],
            "yhat_lower": [8.0, 10.0, 13.0, 18.0, 21.0, 20.0],
            "yhat_upper": [11.0, 12.0, 16.0, 21.0, 24.0, 25.0],
        }
    )

    expected = {
        "mse": [14.0 / 3.0, 47.5 / 3.0],
        "rmse": [np.sqrt(14.0 / 3.0), np.sqrt(47.5 / 3.0)],
        "mae": [2.0, 23.0 / 6.0],
        "mape": [0.1225380899293943, 0.17577521780420333],
        "mdape": [0.13043478260869565, 0.18518518518518517],
        "smape": [0.13110529598521833, 0.19313487668969395],
        "coverage": [2.0 / 3.0, 0.5],
    }
    for metric, values in expected.items():
        result = plotting._prophet_performance_metrics(cv_rows, metric, 0.5)
        assert result["horizon"].tolist() == [pd.Timedelta(days=3), pd.Timedelta(days=4)]
        assert result[metric].tolist() == pytest.approx(values)

    point_result = plotting._prophet_performance_metrics(cv_rows, "coverage", -1)
    assert point_result["coverage"].tolist() == [True, True, False, True, True, False]


def test_prophet_plotly_forecast_trace_contract_matches_upstream() -> None:
    pytest.importorskip("plotly")
    model, fcst = prophet_like_inputs()
    model.logistic_floor = True
    fcst = fcst.assign(cap=np.linspace(20.0, 24.0, len(fcst)), floor=0.0)

    figure = plot_plotly(model, fcst, trend=True, changepoints=True)

    assert [trace.name for trace in figure.data] == [
        "Actual",
        None,
        "Predicted",
        None,
        "Cap",
        "Floor",
        "Trend",
        None,
    ]
    assert figure.data[0].mode == "markers"
    assert figure.data[0].marker.color == "black"
    assert figure.data[0].marker.size == 4
    assert figure.data[2].line.color == "#0072B2"
    assert figure.data[2].fill == "tonexty"
    assert figure.data[4].line.dash == "dash"
    assert figure.data[5].line.dash == "dash"
    assert figure.data[6].line.color == "#B23B00"
    assert figure.layout.showlegend is False
    assert figure.layout.xaxis.rangeslider.visible is True


def test_prophet_compatible_plotly_helpers_match_expected_shape() -> None:
    pytest.importorskip("plotly")
    model, fcst = prophet_like_inputs()

    forecast_fig = plot_plotly(model, fcst, trend=True, changepoints=True)
    assert len(forecast_fig.data) >= 5

    components_fig = plot_components_plotly(model, fcst)
    assert len(components_fig.data) >= 4

    component_fig = plot_forecast_component_plotly(model, fcst, "trend")
    assert component_fig.layout.yaxis.title.text == "trend"

    seasonality_fig = plot_seasonality_plotly(model, "weekly")
    assert seasonality_fig.layout.yaxis.title.text == "weekly"

    component_props = get_forecast_component_plotly_props(model, fcst, "trend")
    seasonality_props = get_seasonality_plotly_props(model, "weekly")
    assert set(component_props) == {"traces", "xaxis", "yaxis"}
    assert set(seasonality_props) == {"traces", "xaxis", "yaxis"}


def test_plotting_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        plot_predicted_actual([1.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="lower values"):
        plot_forecast([{"timestamp": "2026-01-01", "prediction": 1.0, "lower": 2.0, "upper": 1.0}])

    with pytest.raises(ValueError, match="model"):
        plot_metric_comparison([{"name": "cartoboost", "rmse": 1.0}])
