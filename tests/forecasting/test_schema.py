import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from cartoboost.preview.forecasting import (  # noqa: E402
    BaseForecaster,
    ForecastFrame,
    ForecastResult,
    PanelForecasterMixin,
    PredictionInterval,
    SingleSeriesForecasterMixin,
)


def test_forecast_frame_from_pandas_sorts_and_exports_metadata_for_single_series():
    raw = pd.DataFrame(
        {
            "pickup_hour": ["2025-01-03", "2025-01-01", "2025-01-02"],
            "fare": [12.0, 10.0, 11.0],
            "day_of_week": [5, 3, 4],
            "weather_code": [1, 1, 2],
        }
    )

    frame = ForecastFrame.from_pandas(
        raw,
        timestamp_col="pickup_hour",
        target_col="fare",
        known_future_covariates=["day_of_week"],
        historical_covariates=["weather_code"],
    )

    sorted_data = frame.to_pandas()
    assert list(sorted_data["fare"]) == [10.0, 11.0, 12.0]
    assert frame.freq == "D"
    assert frame.to_metadata() == {
        "timestamp_col": "pickup_hour",
        "target_col": "fare",
        "series_id_col": None,
        "freq": "D",
        "is_panel": False,
        "n_rows": 3,
        "series_ids": [],
        "static_covariates": [],
        "known_future_covariates": ["day_of_week"],
        "historical_covariates": ["weather_code"],
        "allow_irregular": False,
        "allow_missing_targets": False,
        "allow_missing_covariates": False,
        "sample_weight_col": None,
    }


def test_forecast_frame_rejects_duplicate_timestamps_within_single_series():
    raw = pd.DataFrame(
        {
            "pickup_hour": ["2025-01-01", "2025-01-01"],
            "fare": [10.0, 11.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate timestamp"):
        ForecastFrame.from_pandas(raw, timestamp_col="pickup_hour", target_col="fare", freq="D")


def test_forecast_frame_aggregates_duplicate_timestamps_with_sample_weights():
    raw = pd.DataFrame(
        {
            "lane": ["A-B", "A-B", "A-B", "A-B"],
            "pickup_hour": [
                "2025-01-01 08:00",
                "2025-01-01 08:00",
                "2025-01-01 09:00",
                "2025-01-01 09:00",
            ],
            "fare": [10.0, 20.0, 30.0, 40.0],
            "trip_distance": [1.0, 3.0, 5.0, 7.0],
            "trip_count": [1.0, 3.0, 2.0, 2.0],
        }
    )

    frame = ForecastFrame.from_pandas(
        raw,
        timestamp_col="pickup_hour",
        target_col="fare",
        series_id_col="lane",
        freq="h",
        historical_covariates=["trip_distance"],
        sample_weight_col="trip_count",
    )

    data = frame.to_pandas()
    assert len(data) == 2
    assert list(data["fare"]) == [17.5, 35.0]
    assert list(data["trip_distance"]) == [2.5, 6.0]
    assert list(data["trip_count"]) == [4.0, 4.0]
    assert frame._native_frame.row_count() == 2
    assert frame.historical_covariates == ("trip_distance", "trip_count")
    assert frame.to_metadata()["sample_weight_col"] == "trip_count"


def test_forecast_frame_rejects_invalid_sample_weights():
    raw = pd.DataFrame(
        {
            "pickup_hour": ["2025-01-01", "2025-01-01"],
            "fare": [10.0, 11.0],
            "trip_count": [1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="positive finite"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            freq="D",
            sample_weight_col="trip_count",
        )


def test_forecast_frame_rejects_unparseable_timestamps():
    raw = pd.DataFrame({"pickup_hour": ["not-a-date"], "fare": [10.0]})

    with pytest.raises(ValueError, match="unparseable"):
        ForecastFrame.from_pandas(raw, timestamp_col="pickup_hour", target_col="fare", freq="D")


def test_forecast_frame_rejects_empty_data():
    raw = pd.DataFrame({"pickup_hour": [], "fare": []})

    with pytest.raises(ValueError, match="at least one row"):
        ForecastFrame.from_pandas(raw, timestamp_col="pickup_hour", target_col="fare", freq="D")


@pytest.mark.parametrize("bad_target", [np.nan, np.inf, -np.inf])
def test_forecast_frame_rejects_non_finite_targets(bad_target):
    raw = pd.DataFrame({"pickup_hour": ["2025-01-01"], "fare": [bad_target]})

    with pytest.raises(ValueError, match="finite"):
        ForecastFrame.from_pandas(raw, timestamp_col="pickup_hour", target_col="fare", freq="D")


def test_forecast_frame_rejects_irregular_data_unless_allowed():
    raw = pd.DataFrame(
        {
            "pickup_hour": ["2025-01-01", "2025-01-02", "2025-01-04"],
            "fare": [10.0, 11.0, 14.0],
        }
    )

    with pytest.raises(ValueError, match="regular frequency|irregular"):
        ForecastFrame.from_pandas(raw, timestamp_col="pickup_hour", target_col="fare")

    frame = ForecastFrame.from_pandas(
        raw,
        timestamp_col="pickup_hour",
        target_col="fare",
        allow_irregular=True,
    )
    assert frame.freq is None
    assert frame.allow_irregular is True


def test_forecast_frame_builds_native_irregular_frame_with_frequency_hint(install_fake_native):
    native = install_fake_native("NaiveForecaster")
    raw = pd.DataFrame(
        {
            "pickup_hour": ["2025-01-01", "2025-01-08", "2025-01-09"],
            "fare": [10.0, 17.0, 18.0],
        }
    )

    frame = ForecastFrame.from_pandas(
        raw,
        timestamp_col="pickup_hour",
        target_col="fare",
        freq="D",
        allow_irregular=True,
    )

    assert frame.freq == "D"
    assert frame._native_frame is not None
    assert frame._native_frame.freq == "D"
    assert frame._native_frame.kwargs["allow_irregular"] is True
    assert native.frame_class is frame._native_frame.__class__


def test_forecast_frame_rejects_native_boundary_signature_mismatch(
    install_fake_native,
    monkeypatch,
):
    install_fake_native("NaiveForecaster")
    import cartoboost

    def reject_legacy_boundary(*args, **kwargs):
        del args, kwargs
        raise TypeError("legacy ForecastFrame signature")

    monkeypatch.setattr(cartoboost._native, "ForecastFrame", reject_legacy_boundary)
    raw = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01", periods=3, freq="D"),
            "fare": [10.0, 11.0, 12.0],
        }
    )
    with pytest.raises(RuntimeError, match="typed ForecastFrame payload"):
        ForecastFrame.from_pandas(raw, timestamp_col="pickup_hour", target_col="fare", freq="D")


def test_forecast_frame_allows_missing_targets_only_when_explicit(install_fake_native):
    native = install_fake_native("NaiveForecaster")
    raw = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01", periods=3, freq="D"),
            "fare": [10.0, np.nan, 12.0],
        }
    )

    with pytest.raises(ValueError, match="allow_missing_targets"):
        ForecastFrame.from_pandas(raw, timestamp_col="pickup_hour", target_col="fare", freq="D")

    frame = ForecastFrame.from_pandas(
        raw,
        timestamp_col="pickup_hour",
        target_col="fare",
        freq="D",
        allow_missing_targets=True,
    )

    assert frame.allow_missing_targets is True
    assert frame._native_frame is not None
    assert frame._native_frame.kwargs["allow_missing_targets"] is True
    assert native.frame_class is frame._native_frame.__class__


def test_forecast_frame_rejects_infinite_targets_even_when_missing_allowed():
    raw = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01", periods=2, freq="D"),
            "fare": [10.0, np.inf],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            freq="D",
            allow_missing_targets=True,
        )


def test_forecast_frame_allows_missing_covariates_only_when_explicit(install_fake_native):
    native = install_fake_native("NaiveForecaster")
    raw = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01", periods=3, freq="D"),
            "fare": [10.0, 11.0, 12.0],
            "QUOTE_MILES": [4.2, np.nan, 5.1],
        }
    )

    with pytest.raises(ValueError, match="allow_missing_covariates"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            freq="D",
            historical_covariates=["QUOTE_MILES"],
        )

    frame = ForecastFrame.from_pandas(
        raw,
        timestamp_col="pickup_hour",
        target_col="fare",
        freq="D",
        historical_covariates=["QUOTE_MILES"],
        allow_missing_covariates=True,
    )

    assert frame.allow_missing_covariates is True
    assert frame._native_frame is not None
    assert frame._native_frame.kwargs["allow_missing_covariates"] is True
    assert native.frame_class is frame._native_frame.__class__


def test_forecast_frame_rejects_infinite_covariates_even_when_missing_allowed():
    raw = pd.DataFrame(
        {
            "pickup_hour": pd.date_range("2025-01-01", periods=2, freq="D"),
            "fare": [10.0, 11.0],
            "QUOTE_MILES": [4.2, np.inf],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            freq="D",
            historical_covariates=["QUOTE_MILES"],
            allow_missing_covariates=True,
        )


def test_panel_forecast_frame_detects_duplicates_per_series_and_keeps_series_isolated():
    raw = pd.DataFrame(
        {
            "zone_pair": ["B", "A", "B", "A"],
            "pickup_hour": ["2025-01-02", "2025-01-01", "2025-01-01", "2025-01-02"],
            "fare": [22.0, 10.0, 21.0, 11.0],
            "trip_distance": [3.0, 1.0, 2.5, 1.2],
        }
    )

    frame = ForecastFrame.from_pandas(
        raw,
        timestamp_col="pickup_hour",
        target_col="fare",
        series_id_col="zone_pair",
        freq="D",
        static_covariates=["trip_distance"],
    )

    sorted_data = frame.to_pandas()
    assert list(sorted_data["zone_pair"]) == ["A", "A", "B", "B"]
    assert list(sorted_data["fare"]) == [10.0, 11.0, 21.0, 22.0]
    assert frame.series_ids == ["A", "B"]
    assert frame.to_metadata()["static_covariates"] == ["trip_distance"]

    duplicate = raw.copy()
    duplicate.loc[len(duplicate)] = ["A", "2025-01-01", 12.0, 1.1]
    with pytest.raises(ValueError, match="duplicate timestamp"):
        ForecastFrame.from_pandas(
            duplicate,
            timestamp_col="pickup_hour",
            target_col="fare",
            series_id_col="zone_pair",
            freq="D",
        )


def test_panel_forecast_frame_rejects_null_series_ids():
    raw = pd.DataFrame(
        {
            "zone_pair": ["A", None],
            "pickup_hour": ["2025-01-01", "2025-01-01"],
            "fare": [10.0, 11.0],
        }
    )

    with pytest.raises(ValueError, match="series id column"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            series_id_col="zone_pair",
            freq="D",
        )


def test_panel_forecast_frame_rejects_mixed_inferred_frequency():
    raw = pd.DataFrame(
        {
            "zone_pair": ["A", "A", "A", "B", "B", "B"],
            "pickup_hour": [
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-01",
                "2025-01-03",
                "2025-01-05",
            ],
            "fare": [10.0, 11.0, 12.0, 20.0, 22.0, 24.0],
        }
    )

    with pytest.raises(ValueError, match="share one inferred frequency"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            series_id_col="zone_pair",
        )


def test_covariate_metadata_rejects_missing_reserved_and_overlapping_columns():
    raw = pd.DataFrame({"pickup_hour": ["2025-01-01"], "fare": [10.0], "hour": [8]})

    with pytest.raises(ValueError, match="missing required columns"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            known_future_covariates=["day_of_week"],
            freq="D",
        )
    with pytest.raises(ValueError, match="reserved"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            known_future_covariates=["fare"],
            freq="D",
        )
    with pytest.raises(ValueError, match="only one forecasting role"):
        ForecastFrame.from_pandas(
            raw,
            timestamp_col="pickup_hour",
            target_col="fare",
            known_future_covariates=["hour"],
            historical_covariates=["hour"],
            freq="D",
        )


def test_forecast_result_has_deterministic_columns_and_json_roundtrip():
    result = ForecastResult.from_predictions(
        series_id=["B", "A", "A", "B"],
        timestamps=[
            "2025-01-02",
            "2025-01-01",
            "2025-01-02",
            "2025-01-01",
        ],
        predictions=[22.0, 10.0, 11.0, 21.0],
        intervals=[
            PredictionInterval(
                level=0.9, lower=[20.0, 8.0, 9.0, 19.0], upper=[24.0, 12.0, 13.0, 23.0]
            ),
            PredictionInterval(
                level=0.5, lower=[21.0, 9.0, 10.0, 20.0], upper=[23.0, 11.0, 12.0, 22.0]
            ),
        ],
    )

    data = result.to_pandas()
    assert list(data.columns) == [
        "series_id",
        "timestamp",
        "prediction",
        "prediction_lower_50",
        "prediction_upper_50",
        "prediction_lower_90",
        "prediction_upper_90",
    ]
    assert list(data["series_id"]) == ["A", "A", "B", "B"]
    assert json.loads(result.to_json())["columns"] == list(data.columns)

    restored = ForecastResult.from_json(result.to_json())

    pd.testing.assert_frame_equal(restored.to_pandas(), data)


def test_forecast_result_direct_construction_validates_and_sorts():
    result = ForecastResult(
        pd.DataFrame(
            {
                "series_id": ["B", "A"],
                "timestamp": ["2025-01-02", "2025-01-01"],
                "prediction": [12, 10],
                "prediction_lower_90": [11, 9],
                "prediction_upper_90": [13, 11],
            }
        ),
        series_id_col="series_id",
    )

    data = result.to_pandas()
    assert list(data["series_id"]) == ["A", "B"]
    assert list(data["prediction"]) == [10.0, 12.0]


def test_forecast_result_rejects_empty_predictions_duplicate_intervals_and_null_series():
    with pytest.raises(ValueError, match="at least one row"):
        ForecastResult.from_predictions(timestamps=[], predictions=[])

    interval = PredictionInterval(level=0.9, lower=[1.0], upper=[2.0])
    with pytest.raises(ValueError, match="unique"):
        ForecastResult.from_predictions(
            timestamps=["2025-01-01"],
            predictions=[1.5],
            intervals=[interval, interval],
        )

    with pytest.raises(ValueError, match="series_id values"):
        ForecastResult.from_predictions(
            timestamps=["2025-01-01"],
            predictions=[1.5],
            series_id=[None],
        )


def test_prediction_interval_validates_bounds_and_level():
    with pytest.raises(ValueError, match="between 0 and 1"):
        PredictionInterval(level=1.0, lower=[1.0], upper=[2.0])
    with pytest.raises(ValueError, match="must not exceed"):
        PredictionInterval(level=0.8, lower=[3.0], upper=[2.0])


def test_base_forecaster_and_mixins_validate_inputs():
    single = ForecastFrame.from_pandas(
        pd.DataFrame({"pickup_hour": ["2025-01-01", "2025-01-02"], "fare": [10.0, 11.0]}),
        timestamp_col="pickup_hour",
        target_col="fare",
    )
    panel = ForecastFrame.from_pandas(
        pd.DataFrame(
            {
                "zone_pair": ["A", "A"],
                "pickup_hour": ["2025-01-01", "2025-01-02"],
                "fare": [10.0, 11.0],
            }
        ),
        timestamp_col="pickup_hour",
        target_col="fare",
        series_id_col="zone_pair",
    )

    forecaster = BaseForecaster()
    with pytest.raises(ValueError, match="not fitted"):
        forecaster.predict(1)
    assert forecaster.fit(single).predict(2) == 2

    with pytest.raises(ValueError, match="without series_id_col"):
        SingleSeriesForecasterMixin()._validate_single_series_frame(panel)
    with pytest.raises(ValueError, match="require series_id_col"):
        PanelForecasterMixin()._validate_panel_frame(single)
    assert PanelForecasterMixin()._validate_panel_frame(panel) is panel
