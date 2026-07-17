import pytest
from cartoboost import Backend, CartoBoostRegressor, FeatureKind
from cartoboost import regressor as regressor_module


def test_backend_enum_covers_native_accelerator_contract():
    assert {backend.value for backend in Backend} >= {
        "auto",
        "cpu",
        "cuda",
        "directml",
        "rocm",
        "metal",
        "webgpu",
    }


def test_unknown_splitter_is_rejected_before_native_training():
    model = CartoBoostRegressor(splitters=["axis", "not_a_splitter"])

    with pytest.raises(ValueError, match="unknown splitter"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


def test_splitters_must_be_a_sequence_of_names():
    model = CartoBoostRegressor(splitters="axis")

    with pytest.raises(ValueError, match="splitters must be a list"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


def test_feature_schema_metadata_is_retained(tmp_path):
    schema = {"distance": {"role": FeatureKind.NUMERIC}}
    model = CartoBoostRegressor(max_depth=0)

    model.fit([[0.0], [1.0]], [2.0, 4.0], feature_schema=schema)
    model_path = tmp_path / "schema.json"
    model.save(model_path)
    restored = CartoBoostRegressor.load(model_path)

    assert model.feature_schema_ == {"names": ["distance"], "kinds": ["Numeric"]}
    assert restored.feature_schema_ == {"names": ["distance"], "kinds": ["Numeric"]}


def test_periodic_custom_period_splitter_is_validated():
    model = CartoBoostRegressor(splitters=["periodic:168"], max_depth=0)

    model.fit([[1.0], [2.0]], [3.0, 4.0])

    assert model.splitters == ["periodic:168"]


def test_native_load_restores_public_estimator_params(monkeypatch, tmp_path):
    class FakeNativeModel:
        n_estimators = 7
        learning_rate = 0.25
        max_depth = 2
        min_samples_leaf = 3
        min_gain = 0.125
        splitters = ["gaussian_2d", "periodic:168"]
        leaf_predictor = "linear"
        linear_leaf_features = [0, 2]
        l2_regularization = 0.75
        fuzzy = True
        fuzzy_bandwidth = 1.5
        fuzzy_kernel = "gaussian"
        feature_count = 3
        feature_schema_json = (
            '{"names":["x","y","hour"],"kinds":["Numeric","Numeric",{"Periodic":{"period":168}}]}'
        )
        metadata_json = '{"library_name":"cartoboost-core","library_version":"0.1.0"}'

        @classmethod
        def load(cls, path):
            assert path.name == "native.json"
            return cls()

        def predict(self, rows):
            return [0.0 for _ in rows]

    monkeypatch.setattr(regressor_module, "_NativeRegressorModel", FakeNativeModel)

    loaded = CartoBoostRegressor.load(tmp_path / "native.json")

    assert loaded.get_params() == {
        "n_estimators": 7,
        "learning_rate": 0.25,
        "max_depth": 2,
        "min_samples_leaf": 3,
        "min_gain": 0.125,
        "loss": "l2",
        "quantile_alpha": 0.5,
        "huber_delta": 1.0,
        "log_offset": 1.0,
        "loss_params": None,
        "splitters": ["gaussian_2d", "periodic:168"],
        "leaf_predictor": "linear",
        "linear_leaf_features": ["0", "2"],
        "fuzzy": True,
        "fuzzy_bandwidth": 1.5,
        "fuzzy_kernel": "gaussian",
        "l2_regularization": 0.75,
        "constant_l2_regularization": 0.0,
        "random_state": None,
        "n_threads": None,
        "monotonic_constraints": None,
        "tensorboard_log_dir": None,
        "tensorboard_run_name": None,
    }
    assert loaded.n_features_in_ == 3
    assert loaded.feature_schema_["names"] == ["x", "y", "hour"]
    assert loaded.metadata_["library_name"] == "cartoboost-core"


def test_fuzzy_kernel_validation_rejects_unknown_name():
    model = CartoBoostRegressor(fuzzy=True, fuzzy_kernel="unknown")

    with pytest.raises(ValueError, match="fuzzy_kernel"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])


def test_n_threads_must_be_positive():
    model = CartoBoostRegressor(n_threads=0)

    with pytest.raises(ValueError, match="n_threads must be positive"):
        model.fit([[0.0], [1.0]], [0.0, 1.0])
