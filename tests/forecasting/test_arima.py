from pathlib import Path

import pytest
from cartoboost.forecasting.local import ArimaForecaster
from cartoboost.forecasting.local.arima import AutoARIMAForecaster


def test_arima_validates_order_parameters():
    with pytest.raises(ValueError, match="nonnegative"):
        ArimaForecaster(p=-1)
    with pytest.raises(ValueError, match="d must be <= 2"):
        ArimaForecaster(d=3)


def test_arima_converts_series_and_delegates_to_native(install_fake_native):
    native = install_fake_native("ArimaForecaster")

    result = ArimaForecaster(p=2, d=1, q=1).fit([10.0, 11.0, 13.0, 16.0]).predict(2)

    assert result == {"args": (2,), "kwargs": {}}
    assert native.calls[0] == ("init", {"p": 2, "d": 1, "q": 1})
    assert native.calls[1][1].rows[-1] == ("__single__", "1970-01-04T00:00:00", 16.0)
    assert native.calls[2] == ("predict", (2,), {})


def test_auto_arima_rejects_unknown_parameters():
    with pytest.raises(ValueError, match="unknown AutoARIMAForecaster parameters"):
        AutoARIMAForecaster(error_policy="fallback")
    with pytest.raises(ValueError, match="unknown AutoARIMAForecaster parameters"):
        AutoARIMAForecaster(seasonal=True)
    with pytest.raises(ValueError, match="max_q"):
        AutoARIMAForecaster(max_q=9)


def test_auto_arima_fit_predict_uses_rust_binding():
    model = AutoARIMAForecaster(max_p=2, max_d=1, max_q=1)

    model.fit([10.0, 11.0, 13.0, 16.0, 20.0])
    result = model.predict(2)

    assert [row[3] for row in result.predictions()] == ["auto_arima", "auto_arima"]
    assert [row[2] for row in result.predictions()] == [1, 2]
    assert model.get_metadata()["selected_order"] is not None


def test_arima_exposes_native_metadata():
    model = ArimaForecaster(p=2, d=1, q=1)

    model.fit([10.0, 11.0, 13.0, 16.0, 20.0])

    assert model.get_metadata() == {"model": "arima", "p": 2, "d": 1, "q": 1}


def test_arima_native_paths_release_gil_for_fit_predict_and_utility():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "crates" / "cartoboost-py" / "src" / "lib.rs").read_text(encoding="utf-8")

    assert "fn fit_forecaster_py<M: Forecaster>" in source
    assert "py.detach(|| model.fit(&frame.frame))" in source
    assert "fn predict_forecaster_py<M: Forecaster>" in source
    assert "forecast_to_py(py.detach(|| model.predict(horizon)))" in source
    assert "fn utility_series_forecast(" in source
    assert "forecaster.fit(&frame)?;" in source
    assert "forecaster.predict(horizon)" in source


def test_native_module_declares_free_threaded_python_support():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "crates" / "cartoboost-py" / "src" / "lib.rs").read_text(encoding="utf-8")

    assert "#[pymodule(gil_used = false)]" in source
