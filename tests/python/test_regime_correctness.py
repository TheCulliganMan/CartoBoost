"""Independent oracles for regime fitting, batch invariance, and artifact migration."""

import json
import pickle

import numpy as np
import pytest
from cartoboost import _native
from cartoboost.deep import RegimeMoEForecaster


def fixture():
    rng = np.random.default_rng(211)
    x = rng.normal(size=(96, 4))
    residual = np.sin(np.arange(96))
    y = 2 + x[:, 0] + np.maximum(x[:, 0], 0) * 3 + residual * 0.4
    return x, y, {"recent_residuals": residual, "recent_volatility": np.abs(residual)}


@pytest.mark.parametrize("layout", ["c", "fortran", "reverse", "stride", "pandas", "polars"])
def test_regime_predictions_do_not_depend_on_batch_or_layout(layout, tmp_path):
    x, y, router = fixture()
    model = RegimeMoEForecaster().fit(x, y, **router)
    if layout == "fortran":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1]
        router = {key: value[::-1] for key, value in router.items()}
    elif layout == "stride":
        padded = np.zeros((len(x), 8))
        padded[:, ::2] = x
        x = padded[:, ::2]
    expected = model.predict(x, **router)
    single = np.concatenate(
        [
            model.predict(x[i : i + 1], **{key: value[i : i + 1] for key, value in router.items()})
            for i in range(len(x))
        ]
    )
    np.testing.assert_allclose(single, expected, atol=1e-12, rtol=0)
    permutation = np.random.default_rng(33).permutation(len(x))
    np.testing.assert_allclose(
        model.predict(x[permutation], **{k: v[permutation] for k, v in router.items()}),
        expected[permutation],
        atol=1e-12,
        rtol=0,
    )
    if layout in {"pandas", "polars"}:
        package = pytest.importorskip(layout)
        table = package.DataFrame(x)
        np.testing.assert_allclose(model.predict(table, **router), expected, atol=1e-12, rtol=0)
    model.save(tmp_path / "regime.json")
    for restored in [
        RegimeMoEForecaster.load(tmp_path / "regime.json"),
        pickle.loads(pickle.dumps(model)),
    ]:
        np.testing.assert_allclose(restored.predict(x, **router), expected, atol=0, rtol=0)


def test_regime_training_metrics_and_solver_match_available_inference_inputs():
    x, y, router = fixture()
    model = RegimeMoEForecaster(ridge=0.01).fit(x, y, **router)
    parts = model.predict_components(x, **router)
    predicted = parts["combined_prediction"]
    assert model.train_rmse_ == pytest.approx(np.sqrt(np.mean((y - predicted) ** 2)), abs=1e-12)
    assert model.score(x, y, **router) == pytest.approx(model.train_rmse_, abs=1e-12)
    # Both ridge stages use the same features that are available at inference.
    for design, coefficient in [
        (x, model.global_coef_),
        (parts["expert_predictions"], model.mixer_coef_),
    ]:
        design = np.column_stack([np.ones(len(y)), design])
        penalty = np.eye(design.shape[1]) * model.ridge
        penalty[0, 0] = 0
        oracle = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        np.testing.assert_allclose(coefficient, oracle, atol=1e-8, rtol=1e-8)
    base = parts["expert_predictions"][:, 0]
    np.testing.assert_allclose(
        parts["expert_predictions"][:, 3],
        base + router["recent_residuals"] * parts["expert_weights"][:, 3],
    )


def test_legacy_regime_artifact_keeps_predictions_and_refit_upgrades(tmp_path):
    x, y, router = fixture()
    model = RegimeMoEForecaster().fit(x, y, **router)
    path = model.save(tmp_path / "legacy.json")
    artifact = json.loads(path.read_text())
    for key in [
        "artifact_version",
        "base_mean",
        "first_feature_median",
        "router_widths",
        "selection",
    ]:
        artifact.pop(key)
    path.write_text(json.dumps(artifact))
    with pytest.warns(UserWarning, match="batch-dependent"):
        legacy = RegimeMoEForecaster.load(path)
    base = np.column_stack([np.ones(len(x)), x]) @ model.global_coef_
    experts = np.column_stack(
        [
            base,
            base.mean() + 0.35 * (base - base.mean()),
            base + np.maximum(0, x[:, 0] - np.median(x[:, 0])),
            base,
            base + 0.5 * x[:, -1],
            np.full(len(x), base.mean()),
        ]
    )
    oracle = np.column_stack([np.ones(len(x)), experts]) @ model.mixer_coef_ + model.residual_mean_
    np.testing.assert_allclose(legacy.predict(x, **router), oracle, atol=1e-10, rtol=1e-10)
    legacy.fit(x, y, **router)
    legacy.save(path)
    assert json.loads(path.read_text())["artifact_version"] == 2


def test_native_regime_logical_layout_and_validation(tmp_path):
    x, y, router = fixture()
    model = RegimeMoEForecaster().fit(x, y, **router)
    r, widths = model._router_matrix(len(x), **router)
    native = _native.RegimeMoEModel.fit_arrays(
        np.asfortranarray(x), y[::-1][::-1], np.asfortranarray(r), widths, 1e-6, "cpu"
    )
    np.testing.assert_allclose(
        native.predict_arrays(x, r, widths), model.predict(x, **router), atol=0, rtol=0
    )
    assert model.predict(np.empty((0, 4))).shape == (0,)
    assert model.predict_components(np.empty((0, 4)))["expert_weights"].shape == (0, 6)
    for bad in [np.full_like(x, np.nan), x[:, :2]]:
        with pytest.raises(ValueError):
            model.predict(bad)
    for bad_ridge in [-1, np.nan, np.inf]:
        with pytest.raises(ValueError):
            RegimeMoEForecaster(ridge=bad_ridge).fit(x, y)
    with pytest.raises(ValueError):
        RegimeMoEForecaster().fit(x, y, pair_ids=["short"])
    with pytest.raises(ValueError):
        RegimeMoEForecaster().fit(x, y[:-1])
    with pytest.raises(ValueError):
        model.predict(x, time_features=np.zeros((len(x), 2)))
    path = model.save(tmp_path / "model.json")
    payload = json.loads(path.read_text())
    payload["base_mean"] = None
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="training references"):
        RegimeMoEForecaster.load(path)


def test_unfitted_regime_and_multi_column_router_contract():
    x, y, _ = fixture()
    with pytest.raises(RuntimeError, match="fit"):
        RegimeMoEForecaster().predict(x)
    time = np.column_stack([np.arange(len(x)), np.ones(len(x))])
    recent = np.sin(np.arange(len(x)))
    model = RegimeMoEForecaster().fit(x, y, time_features=time, recent_residuals=recent)
    parts = model.predict_components(x, time_features=time, recent_residuals=recent)
    np.testing.assert_allclose(
        parts["expert_predictions"][:, 3],
        parts["expert_predictions"][:, 0] + recent * parts["expert_weights"][:, 3],
    )


def test_legacy_regime_pickle_state_and_empty_router_columns():
    x, y, _ = fixture()
    model = RegimeMoEForecaster().fit(x, y)
    state = dict(model.__dict__)
    state.pop("_native_model")
    restored = RegimeMoEForecaster.__new__(RegimeMoEForecaster)
    with pytest.warns(UserWarning, match="Legacy RegimeMoE pickle"):
        restored.__setstate__(state)
    assert np.isfinite(restored.predict(x)).all()
    empty = {
        name: np.empty((len(x), 0))
        for name in [
            "time_features",
            "recent_volatility",
            "recent_residuals",
            "graph_centrality",
            "historical_sparsity",
            "candidate_value",
        ]
    }
    fitted = RegimeMoEForecaster().fit(x, y, **empty)
    np.testing.assert_allclose(fitted.predict(x, **empty), model.predict(x), atol=1e-10, rtol=1e-10)
    copy = pickle.loads(pickle.dumps(RegimeMoEForecaster()))
    assert copy.is_fitted_ is False


def test_regime_pickle_preserves_fit_metadata():
    x, y, _ = fixture()
    entities = [f"entity-{i}" for i in range(len(x))]
    pairs = [f"pair-{i % 3}" for i in range(len(x))]
    model = RegimeMoEForecaster().fit(x, y, entity_ids=entities, pair_ids=pairs)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.entity_ids_ == entities
    assert restored.pair_ids_ == pairs
    assert restored.shared_regime_router_ is None
    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
