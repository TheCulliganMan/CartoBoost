from __future__ import annotations

import json
import math
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised when the optional sklearn extra is installed.
    from sklearn.base import BaseEstimator, ClassifierMixin
except ImportError:  # pragma: no cover - lightweight fallback for core installs.

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class ClassifierMixin:  # type: ignore[no-redef]
        pass


from ._artifacts import decode_stable_model_artifact, library_version, stable_model_artifact_payload
from ._native import CartoBoostClassifier as _NativeClassifierModel
from .config import FuzzyKernel, LeafPredictor, Objective, SplitPolicy
from .regressor import (
    _as_sample_weight_array,
    _encode_sparse_columns,
    _encoded_feature_schema,
    _feature_schema_metadata,
    _fit_transform_categorical_features,
    _is_empty_sparse_sets,
    _json_attr,
    _normalize_sparse_sets,
    _resolve_linear_leaf_features,
    _rust_feature_schema_json,
    _sparse_names_from_feature_schema,
    _split_policy_from_native,
    _transform_categorical_features,
)
from .regressor import (
    _resolve_splitters as _resolve_regressor_splitters,
)
from .tensorboard import write_training_history


class CartoBoostClassifier(ClassifierMixin, BaseEstimator):
    """Sklearn-style CartoBoost classifier backed by native Rust logloss objectives.

    Parameters mirror :class:`CartoBoostRegressor` where they share tree-building
    behavior. Targets may use any hashable Python label values; labels
    are encoded for native training and decoded for predictions.

    Example:
        >>> clf = CartoBoostClassifier(n_estimators=8, max_depth=1, split_policy="axis_only")
        >>> clf.fit([[0.0], [1.0], [2.0], [3.0]], ["low", "low", "high", "high"])
        CartoBoostClassifier(...)
        >>> clf.predict_proba([[2.5]]).shape
        (1, 2)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        min_samples_leaf: int = 20,
        min_gain: float = 1e-8,
        objective: Objective = Objective.AUTO,
        class_weight: dict[Any, float] | str | None = None,
        split_policy: SplitPolicy = SplitPolicy.AUTO,
        leaf_predictor: LeafPredictor = LeafPredictor.CONSTANT,
        linear_leaf_features: list[str] | None = None,
        fuzzy: bool = False,
        fuzzy_bandwidth: float = 0.0,
        fuzzy_kernel: FuzzyKernel = FuzzyKernel.LINEAR,
        l2_regularization: float = 1.0,
        constant_l2_regularization: float = 0.0,
        random_state: int | None = None,
        n_threads: int | None = None,
        tensorboard_log_dir: str | Path | None = None,
        tensorboard_run_name: str | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.objective = objective
        self.class_weight = class_weight
        self.split_policy = SplitPolicy(split_policy)
        self.leaf_predictor = leaf_predictor
        self.linear_leaf_features = linear_leaf_features
        self.fuzzy = fuzzy
        self.fuzzy_bandwidth = fuzzy_bandwidth
        self.fuzzy_kernel = fuzzy_kernel
        self.l2_regularization = l2_regularization
        self.constant_l2_regularization = constant_l2_regularization
        self.random_state = random_state
        self.n_threads = n_threads
        self.tensorboard_log_dir = tensorboard_log_dir
        self.tensorboard_run_name = tensorboard_run_name
        self._model: Any | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return sklearn-compatible constructor parameters.

        Example:
            >>> CartoBoostClassifier(n_estimators=3).get_params()["n_estimators"]
            3
        """
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_gain": self.min_gain,
            "objective": self.objective,
            "class_weight": self.class_weight,
            "split_policy": self.split_policy,
            "leaf_predictor": self.leaf_predictor,
            "linear_leaf_features": self.linear_leaf_features,
            "fuzzy": self.fuzzy,
            "fuzzy_bandwidth": self.fuzzy_bandwidth,
            "fuzzy_kernel": self.fuzzy_kernel,
            "l2_regularization": self.l2_regularization,
            "constant_l2_regularization": self.constant_l2_regularization,
            "random_state": self.random_state,
            "n_threads": self.n_threads,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "tensorboard_run_name": self.tensorboard_run_name,
        }

    def set_params(self, **params: Any) -> CartoBoostClassifier:
        """Set sklearn-compatible constructor parameters and clear fitted state.

        Example:
            >>> CartoBoostClassifier().set_params(n_estimators=3).n_estimators
            3
        """
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(f"unknown parameter {key!r}")
            setattr(self, key, value)
        self._validate_params()
        self._model = None
        return self

    def fit(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[Any],
        sample_weight: Iterable[float] | None = None,
        feature_schema: Any | None = None,
        sparse_sets: Any | None = None,
    ) -> CartoBoostClassifier:
        """Fit native binary or multiclass logloss trees and return ``self``.

        Example:
            >>> CartoBoostClassifier(n_estimators=2).fit([[0.0], [1.0]], [0, 1])
            CartoBoostClassifier(...)
        """
        self._validate_params()
        labels = _as_label_array(y)
        if labels.shape[0] == 0:
            raise ValueError("y must not be empty")
        classes, encoded = _encode_labels(labels)
        if classes.shape[0] < 2:
            raise ValueError("CartoBoostClassifier requires at least two classes")
        weights_array = _as_sample_weight_array(sample_weight, labels.shape[0])
        class_weights = _class_weight_vector(self.class_weight, classes, encoded)
        dense_array, categorical_encoder, feature_names = _fit_transform_categorical_features(
            X,
            encoded,
            feature_schema,
            sample_weight=sample_weight,
        )
        if dense_array.shape[0] != labels.shape[0]:
            raise ValueError("X and y must contain the same number of rows")
        sparse_columns, sparse_names = _normalize_sparse_sets(sparse_sets, labels.shape[0])
        sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        encoded_feature_schema = _encoded_feature_schema(
            feature_schema,
            categorical_encoder,
            dense_array.shape[1],
        )
        schema_json = _rust_feature_schema_json(
            encoded_feature_schema,
            dense_array.shape[1],
            sparse_names,
        )
        schema_metadata = _feature_schema_metadata(feature_schema)

        self.n_features_in_ = (
            int(categorical_encoder["original_feature_count"])
            if categorical_encoder
            else dense_array.shape[1]
        )
        self.encoded_n_features_in_ = dense_array.shape[1]
        self.n_sparse_sets_in_ = len(sparse_columns)
        self.sparse_set_names_ = sparse_names
        self.classes_ = classes
        self.n_classes_ = int(classes.shape[0])
        self.feature_schema_ = schema_metadata
        self.categorical_encoder_ = categorical_encoder
        if feature_names is not None:
            self.feature_names_in_ = np.asarray(feature_names, dtype=object)

        model = _NativeClassifierModel(
            n_estimators=int(self.n_estimators),
            learning_rate=float(self.learning_rate),
            max_depth=int(self.max_depth),
            min_samples_leaf=int(self.min_samples_leaf),
            min_gain=float(self.min_gain),
            objective=_resolved_objective(str(self.objective), self.n_classes_),
            class_count=self.n_classes_,
            class_weights=class_weights,
            splitters=_resolve_splitters(
                self.split_policy,
                feature_schema,
                n_rows=dense_array.shape[0],
            ),
            leaf_predictor=str(self.leaf_predictor),
            linear_leaf_features=_resolve_linear_leaf_features(
                self.linear_leaf_features,
                dense_array.shape[1],
            ),
            l2_regularization=float(self.l2_regularization),
            constant_l2_regularization=float(self.constant_l2_regularization),
            fuzzy=bool(self.fuzzy),
            fuzzy_bandwidth=float(self.fuzzy_bandwidth),
            fuzzy_kernel=str(self.fuzzy_kernel),
            n_threads=None if self.n_threads is None else int(self.n_threads),
        )
        model.fit_arrays(
            dense_array,
            np.ascontiguousarray(encoded, dtype=np.float64),
            None
            if weights_array is None
            else np.ascontiguousarray(weights_array, dtype=np.float64),
            sparse_offsets,
            sparse_ids,
            schema_json,
        )
        self._model = model
        self.feature_schema_ = (
            json.loads(schema_json) if schema_json is not None else schema_metadata
        )
        self.metadata_ = _json_attr(model, "metadata_json")
        self.training_config_ = _json_attr(model, "training_config_json")
        self.training_history_ = _json_attr(model, "training_history_json") or []
        write_training_history(
            model,
            self.tensorboard_log_dir,
            run_name=self.tensorboard_run_name,
        )
        self.requires_sparse_sets_ = bool(
            getattr(model, "requires_sparse_sets", bool(sparse_columns))
        )
        self.is_fitted_ = True
        return self

    def predict(self, X: Iterable[Iterable[float]], sparse_sets: Any | None = None) -> np.ndarray:
        """Return predicted class labels for rows in ``X``.

        Example:
            >>> clf = CartoBoostClassifier(n_estimators=2, split_policy="axis_only")
            >>> clf.fit([[0.0], [1.0], [2.0], [3.0]], ["no", "no", "yes", "yes"])
            CartoBoostClassifier(...)
            >>> clf.predict([[2.5]]).tolist()
            ['yes']
        """
        if self._model is None:
            raise RuntimeError("CartoBoostClassifier is not fitted")
        dense_array, sparse_offsets, sparse_ids = self._prediction_inputs(X, sparse_sets)
        encoded = np.asarray(
            self._model.predict_arrays(dense_array, sparse_offsets, sparse_ids),
            dtype=np.int64,
        )
        return self.classes_[encoded]

    def predict_proba(
        self,
        X: Iterable[Iterable[float]],
        sparse_sets: Any | None = None,
    ) -> np.ndarray:
        """Return class probabilities with columns ordered like ``classes_``.

        Example:
            >>> clf = CartoBoostClassifier(n_estimators=2, split_policy="axis_only")
            >>> clf.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
            CartoBoostClassifier(...)
            >>> clf.predict_proba([[2.5]]).shape
            (1, 2)
        """
        if self._model is None:
            raise RuntimeError("CartoBoostClassifier is not fitted")
        dense_array, sparse_offsets, sparse_ids = self._prediction_inputs(X, sparse_sets)
        return np.asarray(
            self._model.predict_proba_arrays(dense_array, sparse_offsets, sparse_ids),
            dtype=float,
        )

    def decision_function(
        self,
        X: Iterable[Iterable[float]],
        sparse_sets: Any | None = None,
    ) -> np.ndarray:
        """Return raw native margins before the probability transform.

        Example:
            >>> clf = CartoBoostClassifier(n_estimators=2, split_policy="axis_only")
            >>> clf.fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
            CartoBoostClassifier(...)
            >>> clf.decision_function([[2.5]]).shape
            (1,)
        """
        if self._model is None:
            raise RuntimeError("CartoBoostClassifier is not fitted")
        dense_array, sparse_offsets, sparse_ids = self._prediction_inputs(X, sparse_sets)
        # Native decision_function accepts list rows today; keep this path until
        # a dedicated array binding is added.
        if sparse_offsets or sparse_ids:
            margins = self._model.decision_function(
                dense_array.tolist(),
                _decode_sparse_offsets(sparse_offsets, sparse_ids, dense_array.shape[0]),
            )
        else:
            margins = self._model.decision_function(dense_array.tolist())
        margins_array = np.asarray(margins, dtype=float)
        if self.n_classes_ == 2 and margins_array.ndim == 2 and margins_array.shape[1] == 1:
            return margins_array[:, 0]
        return margins_array

    def score(
        self,
        X: Iterable[Iterable[float]],
        y: Iterable[Any],
        sparse_sets: Any | None = None,
    ) -> float:
        """Return classification accuracy.

        Example:
            >>> clf = CartoBoostClassifier(n_estimators=2, split_policy="axis_only")
            >>> clf.fit([[0.0], [1.0]], [0, 1])
            CartoBoostClassifier(...)
            >>> clf.score([[0.0], [1.0]], [0, 1]) >= 0.0
            True
        """

        pred = np.asarray(self.predict(X, sparse_sets=sparse_sets), dtype=object)
        truth = np.asarray(list(y), dtype=object)
        if pred.shape[0] != truth.shape[0]:
            raise ValueError("X predictions and y must have the same number of rows")
        return float(np.mean(pred == truth))

    def save(self, path: str | Path) -> None:
        """Write a classifier artifact, including class labels and encoders.

        Example:
            >>> clf = CartoBoostClassifier(n_estimators=2, split_policy="axis_only")
            >>> clf.fit([[0.0], [1.0]], [0, 1])
            CartoBoostClassifier(...)
            >>> clf.save("airport-trip-classifier.json")
        """
        if self._model is None:
            raise RuntimeError("CartoBoostClassifier is not fitted")
        path = Path(path)
        with tempfile.TemporaryDirectory() as temp_dir:
            native_path = Path(temp_dir) / "native-classifier.json"
            self._model.save(native_path)
            native_payload = json.loads(native_path.read_text(encoding="utf-8"))
        payload = stable_model_artifact_payload(
            "classifier",
            library_version=library_version(),
            training_config=native_payload.get("training_config", {}),
            payload={
                "classes": _jsonable_classes(self.classes_),
                "categorical_encoder": getattr(self, "categorical_encoder_", None),
                "native_model": native_payload,
            },
        )
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def save_weights(self, path: str | Path, *, format: str = "auto") -> None:
        """Fail loudly for unsupported classifier weight/export artifacts.

        Example:
            >>> clf = CartoBoostClassifier(n_estimators=2, split_policy="axis_only")
            >>> clf.fit([[0.0], [1.0]], [0, 1])
            CartoBoostClassifier(...)
            >>> clf.save_weights("classifier.onnx", format="onnx")
            Traceback (most recent call last):
            ...
            NotImplementedError: CartoBoostClassifier does not support save_weights...
        """
        if self._model is None:
            raise RuntimeError("CartoBoostClassifier is not fitted")
        raise NotImplementedError(
            "CartoBoostClassifier does not support save_weights or ONNX export; "
            "use save(path) for the supported classifier artifact"
        )

    @classmethod
    def load(cls, path: str | Path) -> CartoBoostClassifier:
        """Load a classifier artifact written by ``save``.

        Example:
            >>> restored = CartoBoostClassifier.load("airport-trip-classifier.json")
            >>> restored.classes_.tolist()
            [0, 1]
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        envelope = decode_stable_model_artifact(payload, "classifier")
        inner = envelope["payload"]
        native_payload = inner.get("native_model")
        if not isinstance(native_payload, dict):
            raise ValueError("stable classifier artifact payload is missing native_model")
        with tempfile.TemporaryDirectory() as temp_dir:
            native_path = Path(temp_dir) / "native-classifier.json"
            native_path.write_text(json.dumps(native_payload, sort_keys=True), encoding="utf-8")
            native_model = _NativeClassifierModel.load(native_path)
        estimator = cls._from_native_model(native_model)
        if "classes" not in inner:
            raise ValueError("stable classifier artifact payload is missing classes")
        estimator.classes_ = _object_array_1d(
            [_decode_class_label(label) for label in inner["classes"]],
        )
        estimator.n_classes_ = int(estimator.classes_.shape[0])
        estimator.categorical_encoder_ = inner.get("categorical_encoder")
        if estimator.categorical_encoder_:
            estimator.n_features_in_ = int(estimator.categorical_encoder_["original_feature_count"])
            estimator.encoded_n_features_in_ = native_model.feature_count
        return estimator

    @classmethod
    def _from_native_model(cls, native_model: Any) -> CartoBoostClassifier:
        estimator = cls(
            n_estimators=native_model.n_estimators,
            learning_rate=native_model.learning_rate,
            max_depth=native_model.max_depth,
            min_samples_leaf=native_model.min_samples_leaf,
            min_gain=native_model.min_gain,
            objective=str(native_model.objective),
            split_policy=_split_policy_from_native(native_model.splitters),
            class_weight=None,
        )
        estimator._model = native_model
        estimator.n_features_in_ = native_model.feature_count
        estimator.encoded_n_features_in_ = native_model.feature_count
        estimator.categorical_encoder_ = None
        class_values = np.asarray(native_model.class_values, dtype=np.int64)
        estimator.classes_ = class_values
        estimator.n_classes_ = int(class_values.shape[0])
        estimator.feature_schema_ = _json_attr(native_model, "feature_schema_json")
        estimator.sparse_set_names_ = _sparse_names_from_feature_schema(estimator.feature_schema_)
        estimator.n_sparse_sets_in_ = len(estimator.sparse_set_names_)
        estimator.metadata_ = _json_attr(native_model, "metadata_json")
        estimator.training_config_ = _json_attr(native_model, "training_config_json")
        estimator.training_history_ = _json_attr(native_model, "training_history_json") or []
        estimator.requires_sparse_sets_ = bool(getattr(native_model, "requires_sparse_sets", False))
        estimator.is_fitted_ = True
        return estimator

    def _prediction_inputs(
        self,
        X: Iterable[Iterable[float]],
        sparse_sets: Any | None,
    ) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
        expected_sparse_count = getattr(self, "n_sparse_sets_in_", 0)
        dense_array = _transform_categorical_features(
            X,
            getattr(self, "categorical_encoder_", None),
        )
        expected_dense = getattr(self, "encoded_n_features_in_", self.n_features_in_)
        if hasattr(self, "encoded_n_features_in_") and dense_array.shape[1] != expected_dense:
            raise ValueError(
                f"encoded X has {dense_array.shape[1]} features, but CartoBoostClassifier was "
                f"fitted with {expected_dense} encoded features"
            )
        if not getattr(self, "requires_sparse_sets_", False):
            sparse_columns: list[list[list[int]]] = []
            sparse_names: list[str] = []
            sparse_offsets: list[list[int]] = []
            sparse_ids: list[list[int]] = []
        elif _is_empty_sparse_sets(sparse_sets) and expected_sparse_count == 0:
            sparse_columns = []
            sparse_names = []
            sparse_offsets = []
            sparse_ids = []
        else:
            sparse_columns, sparse_names = _normalize_sparse_sets(
                sparse_sets,
                dense_array.shape[0],
                getattr(self, "sparse_set_names_", None),
            )
            sparse_offsets, sparse_ids = _encode_sparse_columns(sparse_columns)
        if sparse_columns and len(sparse_columns) != expected_sparse_count:
            raise ValueError(
                f"sparse_sets has {len(sparse_columns)} columns, but CartoBoostClassifier was "
                f"fitted with {expected_sparse_count}"
            )
        if not sparse_columns and getattr(self, "requires_sparse_sets_", False):
            raise ValueError("sparse_sets are required for prediction with this sparse-list model")
        return dense_array, sparse_offsets, sparse_ids

    def _validate_params(self) -> None:
        if int(self.n_estimators) <= 0:
            raise ValueError("n_estimators must be positive")
        learning_rate = float(self.learning_rate)
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be non-negative")
        if int(self.min_samples_leaf) <= 0:
            raise ValueError("min_samples_leaf must be positive")
        min_gain = float(self.min_gain)
        if not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain must be finite and non-negative")
        if str(self.objective) not in {
            "auto",
            "binary",
            "binary_logloss",
            "logloss",
            "multiclass",
            "multiclass_logloss",
            "multi_logloss",
        }:
            raise ValueError("objective must be 'auto', 'binary_logloss', or 'multiclass_logloss'")
        if (
            self.class_weight is not None
            and self.class_weight != "balanced"
            and not isinstance(
                self.class_weight,
                Mapping,
            )
        ):
            raise ValueError("class_weight must be None, 'balanced', or a label-to-weight mapping")
        if self.leaf_predictor not in {LeafPredictor.CONSTANT, LeafPredictor.LINEAR}:
            raise ValueError("leaf_predictor must be 'constant' or 'linear'")
        if float(self.l2_regularization) < 0 or not math.isfinite(float(self.l2_regularization)):
            raise ValueError("l2_regularization must be finite and non-negative")
        constant_l2 = float(self.constant_l2_regularization)
        if constant_l2 < 0 or not math.isfinite(constant_l2):
            raise ValueError("constant_l2_regularization must be finite and non-negative")
        if float(self.fuzzy_bandwidth) < 0 or not math.isfinite(float(self.fuzzy_bandwidth)):
            raise ValueError("fuzzy_bandwidth must be finite and non-negative")
        if str(self.fuzzy_kernel) not in {
            "linear",
            "triangular",
            "gaussian",
            "exponential",
            "bisquare",
            "epanechnikov",
            "tricube",
        }:
            raise ValueError(
                "fuzzy_kernel must be 'linear', 'gaussian', 'exponential', "
                "'bisquare', 'epanechnikov', or 'tricube'"
            )
        if self.n_threads is not None and int(self.n_threads) <= 0:
            raise ValueError("n_threads must be positive")


def _resolve_splitters(
    policy: SplitPolicy | str,
    schema: Any | None,
    *,
    n_rows: int | None = None,
) -> list[str]:
    return _resolve_regressor_splitters(policy, schema, n_rows=n_rows)


def _encode_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_to_index: dict[Any, int] = {}
    classes_list: list[Any] = []
    encoded_values = []
    for label in labels.tolist():
        try:
            class_index = class_to_index.get(label)
        except TypeError as exc:
            raise TypeError("classifier class labels must be hashable") from exc
        if class_index is None:
            class_index = len(classes_list)
            class_to_index[label] = class_index
            classes_list.append(label)
        encoded_values.append(class_index)
    classes = _object_array_1d(classes_list)
    encoded = np.asarray(encoded_values, dtype=np.float64)
    return classes, encoded


def _as_label_array(values: Iterable[Any]) -> np.ndarray:
    ndim = getattr(values, "ndim", None)
    if ndim is not None and int(ndim) != 1:
        raise ValueError("y must be a 1D array of class labels")
    return _object_array_1d(list(values))


def _object_array_1d(labels: list[Any]) -> np.ndarray:
    result = np.empty(len(labels), dtype=object)
    result[:] = labels
    if result.ndim != 1:
        raise ValueError("y must be a 1D array of class labels")
    return result


def _class_weight_vector(
    class_weight: dict[Any, float] | str | None,
    classes: np.ndarray,
    encoded: np.ndarray,
) -> list[float]:
    if class_weight is None:
        return []
    if class_weight == "balanced":
        counts = Counter(int(value) for value in encoded.tolist())
        total = float(encoded.shape[0])
        class_count = float(classes.shape[0])
        return [total / (class_count * float(counts[idx])) for idx in range(classes.shape[0])]
    if not isinstance(class_weight, Mapping):
        raise ValueError("class_weight must be None, 'balanced', or a label-to-weight mapping")
    weights = []
    for label in classes.tolist():
        value = float(class_weight.get(label, 1.0))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("class_weight values must be finite and non-negative")
        weights.append(value)
    return weights


def _resolved_objective(objective: str, class_count: int) -> str:
    if objective == "auto":
        return "binary_logloss" if class_count == 2 else "multiclass_logloss"
    return objective


def _jsonable_classes(classes: np.ndarray) -> list[Any]:
    result = []
    for label in classes.tolist():
        if isinstance(label, np.generic):
            label = label.item()
        result.append(_encode_class_label(label))
    return result


def _encode_class_label(label: Any) -> Any:
    if isinstance(label, np.generic):
        label = label.item()
    if label is None or isinstance(label, str | int | float | bool):
        try:
            json.dumps(label)
        except TypeError as exc:
            raise TypeError("classifier class labels must be JSON-serializable to save") from exc
        return label
    if isinstance(label, tuple):
        return {
            "__cartoboost_label_type__": "tuple",
            "items": [_encode_class_label(item) for item in label],
        }
    try:
        json.dumps(label)
    except TypeError as exc:
        raise TypeError("classifier class labels must be JSON-serializable to save") from exc
    return label


def _decode_class_label(payload: Any) -> Any:
    if (
        isinstance(payload, dict)
        and payload.get("__cartoboost_label_type__") == "tuple"
        and isinstance(payload.get("items"), list)
    ):
        return tuple(_decode_class_label(item) for item in payload["items"])
    return payload


def _decode_sparse_offsets(
    sparse_offsets: list[list[int]],
    sparse_ids: list[list[int]],
    row_count: int,
) -> list[list[list[int]]]:
    columns = []
    for offsets, ids in zip(sparse_offsets, sparse_ids, strict=True):
        if len(offsets) != row_count + 1:
            raise ValueError("sparse_offsets column must have rows + 1 entries")
        columns.append([ids[offsets[row] : offsets[row + 1]] for row in range(row_count)])
    return columns
