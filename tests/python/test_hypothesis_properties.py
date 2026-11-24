from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from cartoboost import CartoBoostRegressor
from hypothesis import given, settings
from hypothesis import strategies as st


@st.composite
def small_regression_cases(draw):
    rows = draw(st.integers(min_value=2, max_value=8))
    cols = draw(st.integers(min_value=1, max_value=3))
    value = st.floats(
        min_value=-10.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    )
    flat_x = draw(st.lists(value, min_size=rows * cols, max_size=rows * cols))
    y = draw(st.lists(value, min_size=rows, max_size=rows))
    x = [flat_x[row * cols : (row + 1) * cols] for row in range(rows)]
    return x, y


def _fit_or_skip(model, *args, **kwargs):
    try:
        model.fit(*args, **kwargs)
    except ImportError as exc:
        pytest.skip(str(exc))
    except NotImplementedError as exc:
        pytest.skip(str(exc))
    return model


@settings(max_examples=12, deadline=None)
@given(small_regression_cases())
def test_native_backend_predictions_are_finite(case):
    x, y = case
    model = CartoBoostRegressor(
        n_estimators=3,
        learning_rate=0.25,
        max_depth=2,
        min_samples_leaf=1,
        min_gain=0.0,
    )

    model.fit(x, y)
    predictions = model.predict(x)

    assert len(predictions) == len(x)
    assert all(math.isfinite(float(prediction)) for prediction in predictions)


@settings(max_examples=8, deadline=None)
@given(small_regression_cases())
def test_native_backend_save_load_preserves_predictions(case):
    x, y = case
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
    )

    model.fit(x, y)
    before = model.predict(x)
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "model.json"
        model.save(path)

        restored = CartoBoostRegressor.load(path)

    assert restored.predict(x) == pytest.approx(before)


@settings(max_examples=8, deadline=None)
@given(small_regression_cases())
def test_native_backend_batch_prediction_equals_row_by_row(case):
    x, y = case
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.4,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
    )

    model.fit(x, y)
    batch = model.predict(x)
    row_by_row = [model.predict([row])[0] for row in x]

    assert batch == pytest.approx(row_by_row)


@settings(max_examples=8, deadline=None)
@given(
    st.lists(
        st.floats(
            min_value=-5.0,
            max_value=5.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=2,
        max_size=8,
    )
)
def test_zero_weight_rows_do_not_affect_weighted_initial_prediction(targets):
    x = [[float(idx)] for idx in range(len(targets))]
    weights = [1.0] + [0.0 for _ in targets[1:]]
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=0,
        min_samples_leaf=1,
        min_gain=0.0,
    )

    model.fit(x, targets, sample_weight=weights)

    assert model.predict(x) == pytest.approx([targets[0] for _ in targets])


def test_native_fuzzy_midpoint_preserves_branch_mass():
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="axis_only",
        fuzzy=True,
        fuzzy_bandwidth=1.0,
    )

    _fit_or_skip(model, [[0.0], [1.0], [2.0], [3.0]], [0.0, 0.0, 10.0, 10.0])

    midpoint = float(model.predict([[1.5]])[0])
    assert midpoint == pytest.approx(5.0, abs=1e-12)


def test_native_periodic_prediction_invariant_under_period_shift():
    x = [[float(hour)] for hour in range(24)]
    y = [8.0 if hour >= 22 or hour <= 2 else -2.0 for hour in range(24)]
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )

    _fit_or_skip(
        model,
        x,
        y,
        feature_schema={"dense": [{"name": "hour", "kind": "periodic", "period": 24}]},
    )

    base = model.predict([[23.0], [1.0], [12.0]])
    shifted = model.predict([[47.0], [25.0], [36.0]])
    assert shifted == pytest.approx(base, abs=1e-12)


def test_native_duplicate_sparse_ids_do_not_change_predictions():
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [6.0, 6.0, -1.0, -1.0]
    sparse_sets = {"route_cells": [[7, 11], [11], [3], []]}
    duplicate_sparse_sets = {"route_cells": [[11, 7, 7], [11, 11], [3, 3], []]}
    model = CartoBoostRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )

    _fit_or_skip(
        model,
        x,
        y,
        sparse_sets=sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("route_cells", "sparse_set")],
        },
    )

    assert model.predict(x, sparse_sets=duplicate_sparse_sets) == pytest.approx(
        model.predict(x, sparse_sets=sparse_sets),
        abs=1e-12,
    )


def test_native_sparse_save_load_preserves_predictions(tmp_path: Path):
    x = [[0.0], [0.0], [0.0], [0.0]]
    y = [9.0, 9.0, -3.0, -3.0]
    sparse_sets = {"route_cells": [[5, 7], [7], [2], []]}
    model = CartoBoostRegressor(
        n_estimators=2,
        learning_rate=0.5,
        max_depth=1,
        min_samples_leaf=1,
        min_gain=0.0,
        split_policy="structured",
    )
    _fit_or_skip(
        model,
        x,
        y,
        sparse_sets=sparse_sets,
        feature_schema={
            "dense": [("feature_0", "numeric")],
            "sparse_sets": [("route_cells", "sparse_set")],
        },
    )
    before = np.asarray(model.predict(x, sparse_sets=sparse_sets), dtype=float)
    path = tmp_path / "sparse.json"

    model.save(path)
    restored = CartoBoostRegressor.load(path)

    after = np.asarray(restored.predict(x, sparse_sets=sparse_sets), dtype=float)
    assert after == pytest.approx(before, abs=1e-12)
