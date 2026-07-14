import pickle
from pathlib import Path

import pytest
from cartoboost.forecasting.local import SeasonalNaiveForecaster


def test_seasonal_naive_validates_season_length():
    with pytest.raises(ValueError, match="positive integer"):
        SeasonalNaiveForecaster(season_length=0)
    with pytest.raises(ValueError, match="prediction_interval_levels"):
        SeasonalNaiveForecaster(season_length=2, prediction_interval_levels=[0.0])


def test_seasonal_naive_converts_panel_and_delegates_to_native(install_fake_native):
    native = install_fake_native("SeasonalNaiveForecaster")

    result = (
        SeasonalNaiveForecaster(season_length=3)
        .fit({"pickup_1": [10.0, 20.0], "pickup_2": [30.0, 40.0]})
        .predict(4)
    )

    assert result == {"args": (4,), "kwargs": {}}
    assert native.calls[0] == (
        "init",
        {"season_length": 3, "prediction_interval_levels": ()},
    )
    assert native.calls[1][1].rows == [
        ("pickup_1", "1970-01-01T00:00:00", 10.0),
        ("pickup_1", "1970-01-02T00:00:00", 20.0),
        ("pickup_2", "1970-01-01T00:00:00", 30.0),
        ("pickup_2", "1970-01-02T00:00:00", 40.0),
    ]
    assert native.calls[2] == ("predict", (4,), {})


def test_seasonal_naive_native_metadata_round_trips_after_fit():
    metadata = SeasonalNaiveForecaster(season_length=3).fit([1.0, 2.0, 3.0]).get_metadata()

    assert metadata == {"model": "seasonal_naive", "season_length": 3}


def test_seasonal_naive_pickle_and_weights_roundtrip(tmp_path: Path):
    model = SeasonalNaiveForecaster(season_length=2).fit([1.0, 2.0, 3.0, 4.0])
    expected = model.predict(3).predictions()
    path = tmp_path / "seasonal-naive.weights.json"

    model.save_weights(path)

    assert SeasonalNaiveForecaster.load_weights(path).predict(3).predictions() == expected
    assert pickle.loads(pickle.dumps(model)).predict(3).predictions() == expected
