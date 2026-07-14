import pickle
from pathlib import Path

import pytest
from cartoboost.forecasting.local import ArimaForecaster, NaiveForecaster, ThetaForecaster


def test_naive_requires_fit_before_predict():
    with pytest.raises(RuntimeError, match="fitted"):
        NaiveForecaster().predict(2)


def test_naive_validates_prediction_interval_levels():
    with pytest.raises(ValueError, match="prediction_interval_levels"):
        NaiveForecaster(prediction_interval_levels=[1.0])


def test_naive_converts_series_and_delegates_to_native(install_fake_native):
    native = install_fake_native("NaiveForecaster")

    result = NaiveForecaster(prediction_interval_levels=[0.8]).fit([1.0, 2.0, 4.0]).predict(2)

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == ("init", {"prediction_interval_levels": (0.8,)})
    assert native.calls[1][0] == "fit"
    assert isinstance(native.calls[1][1], native.frame_class)
    assert native.calls[1][1].rows == [
        ("__single__", "1970-01-01T00:00:00", 1.0),
        ("__single__", "1970-01-02T00:00:00", 2.0),
        ("__single__", "1970-01-03T00:00:00", 4.0),
    ]
    assert native.calls[2] == ("predict", (2,), {})


def test_naive_native_metadata_round_trips_after_fit():
    metadata = NaiveForecaster().fit([1.0, 2.0, 4.0]).get_metadata()

    assert metadata == {"model": "naive"}


def test_naive_pickle_and_weights_roundtrip(tmp_path: Path):
    model = NaiveForecaster().fit([1.0, 2.0, 4.0])
    expected = model.predict(3).predictions()
    path = tmp_path / "naive.weights.json"

    model.save_weights(path)

    assert NaiveForecaster.load_weights(path).predict(3).predictions() == expected
    assert pickle.loads(pickle.dumps(model)).predict(3).predictions() == expected


@pytest.mark.parametrize(
    "model",
    [
        ThetaForecaster().fit([1.0, 2.0, 3.0, 4.0, 5.0]),
        ArimaForecaster(p=1, d=0, q=0).fit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ],
)
def test_native_forecast_refit_artifact_roundtrip(model, tmp_path: Path):
    expected = model.predict(2).predictions()
    path = tmp_path / f"{type(model).__name__}.weights.json"

    model.save_weights(path)

    assert type(model).load_weights(path).predict(2).predictions() == expected
    assert pickle.loads(pickle.dumps(model)).predict(2).predictions() == expected
