import numpy as np
import pytest
from cartoboost import (
    environmental_blocked_cv,
    grouped_blocked_cv,
    out_of_time_split,
    spatial_blocked_cv,
    spatial_buffered_cv,
    spatial_grouped_cv,
    temporal_blocked_cv,
)
from cartoboost.evaluation import _apply_spatial_buffer


def test_grouped_blocked_cv_holds_out_complete_groups():
    groups = np.array(["a", "a", "b", "b", "c", "d"])

    splits = list(grouped_blocked_cv(groups, n_splits=3))

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        assert len(test_idx) > 0


def test_grouped_blocked_cv_preserves_tuple_group_ids():
    groups = [("pickup", 1), ("pickup", 1), ("dropoff", 2), ("dropoff", 2)]

    splits = list(grouped_blocked_cv(groups, n_splits=2))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        train_groups = {groups[index] for index in train_idx}
        test_groups = {groups[index] for index in test_idx}
        assert train_groups.isdisjoint(test_groups)


def test_temporal_blocked_cv_uses_contiguous_sorted_test_blocks_with_gap():
    times = np.array([5, 1, 4, 2, 3, 6])

    splits = list(temporal_blocked_cv(times, n_splits=3, gap=1))
    first_train, first_test = splits[0]

    assert len(splits) == 3
    assert set(first_test) == {1, 3}
    assert set(first_train) == {0, 2, 5}
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))


def test_out_of_time_split_uses_latest_fraction_as_validation():
    times = np.array([5, 1, 4, 2, 3, 6])

    train_idx, validation_idx = out_of_time_split(times, validation_fraction=0.33)

    assert set(train_idx) == {1, 2, 3, 4}
    assert set(validation_idx) == {0, 5}
    assert times[train_idx].max() < times[validation_idx].min()


def test_out_of_time_split_supports_size_cutoff_and_gap():
    times = np.array(
        [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
        ],
        dtype="datetime64[D]",
    )

    size_train_idx, size_validation_idx = out_of_time_split(
        times,
        validation_size=2,
        validation_fraction=None,
        gap=1,
    )
    cutoff_train_idx, cutoff_validation_idx = out_of_time_split(
        times,
        validation_fraction=None,
        cutoff="2025-01-03",
        gap=1,
    )

    assert size_train_idx.tolist() == [0, 1]
    assert size_validation_idx.tolist() == [3, 4]
    assert cutoff_train_idx.tolist() == [0, 1]
    assert cutoff_validation_idx.tolist() == [3, 4]


def test_spatial_blocked_cv_holds_out_grid_blocks():
    coordinates = np.array(
        [
            [0.0, 0.0],
            [0.2, 0.1],
            [0.0, 10.0],
            [0.2, 10.2],
            [10.0, 0.0],
            [10.2, 0.2],
            [10.0, 10.0],
            [10.2, 10.2],
        ]
    )

    splits = list(spatial_blocked_cv(coordinates, n_splits=4, grid_shape=(2, 2)))

    assert len(splits) == 4
    assert sorted(np.concatenate([test_idx for _, test_idx in splits]).tolist()) == list(range(8))
    for train_idx, test_idx in splits:
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(test_idx) == 2


def test_spatial_blocked_cv_supports_one_dimensional_coordinates():
    splits = list(spatial_blocked_cv([3.0, 1.0, 2.0, 4.0], n_splits=2))

    assert [test_idx.tolist() for _, test_idx in splits] == [[1, 2], [0, 3]]


def test_spatial_buffered_cv_removes_nearby_training_rows():
    coordinates = np.array([[0.0], [1.0], [2.0], [10.0], [11.0], [12.0]])

    splits = list(spatial_buffered_cv(coordinates, n_splits=2, buffer_radius=1.1))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        distances = np.abs(coordinates[train_idx] - coordinates[test_idx].T)
        assert np.min(distances) > 1.1


def test_spatial_buffered_cv_backend_matches_cpu(monkeypatch):
    coordinates = np.column_stack(
        (
            np.linspace(0.0, 100.0, 256),
            np.sin(np.linspace(0.0, 12.0, 256)),
        )
    )
    expected = list(
        spatial_buffered_cv(
            coordinates,
            n_splits=4,
            buffer_radius=1.0,
            backend="cpu",
        )
    )
    calls: list[str] = []

    def accelerated_distances(left, right, backend=None):
        calls.append(str(backend))
        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        deltas = left_array[:, None, :] - right_array[None, :, :]
        return np.sum(deltas * deltas, axis=2).astype(np.float32)

    monkeypatch.setattr(
        "cartoboost.evaluation.workload_decision",
        lambda backend, operation, workload_size, minimum_accelerated_size: {"executed": "cuda"},
    )
    monkeypatch.setattr(
        "cartoboost.evaluation.pairwise_squared_distances",
        accelerated_distances,
    )
    actual = list(
        spatial_buffered_cv(
            coordinates,
            n_splits=4,
            buffer_radius=1.0,
            backend="cuda",
        )
    )

    assert calls == ["cuda"] * 4
    for (expected_train, expected_test), (actual_train, actual_test) in zip(
        expected, actual, strict=True
    ):
        assert np.array_equal(actual_train, expected_train)
        assert np.array_equal(actual_test, expected_test)


def test_spatial_buffered_cv_supports_coordinate_columns_and_random_state():
    frame = {
        "pickup_x": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
        "pickup_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    first = list(
        spatial_buffered_cv(
            frame,
            coordinate_cols=("pickup_x", "pickup_y"),
            n_splits=2,
            buffer_radius=1.1,
            random_state=7,
        )
    )
    second = list(
        spatial_buffered_cv(
            frame,
            coordinate_cols=("pickup_x", "pickup_y"),
            n_splits=2,
            buffer_radius=1.1,
            random_state=7,
        )
    )

    assert [(train_idx.tolist(), test_idx.tolist()) for train_idx, test_idx in first] == [
        (train_idx.tolist(), test_idx.tolist()) for train_idx, test_idx in second
    ]
    coords = np.column_stack([frame["pickup_x"], frame["pickup_y"]])
    for train_idx, test_idx in first:
        distances = np.sqrt(np.sum((coords[train_idx, None, :] - coords[test_idx]) ** 2, axis=2))
        assert np.min(distances) > 1.1


def test_spatial_buffered_cv_allows_degree_buffers_only_when_explicit():
    splits = list(
        spatial_buffered_cv(
            [[40.7, -73.9], [40.8, -74.0]],
            n_splits=2,
            buffer_radius=0.0,
            coordinate_units="degrees",
        )
    )
    assert len(splits) == 2

    allowed = list(
        spatial_buffered_cv(
            [[40.7, -73.9], [40.8, -74.0], [41.5, -75.0]],
            n_splits=2,
            buffer_radius=0.01,
            coordinate_units="degrees",
            allow_degree_buffer=True,
        )
    )
    assert len(allowed) == 2


def test_spatial_buffer_helper_rejects_overlapping_fold_indices():
    with pytest.raises(ValueError, match="overlapping train and test"):
        _apply_spatial_buffer(
            np.asarray([[0.0], [1.0], [2.0]]),
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            0.0,
        )


def test_spatial_grouped_cv_holds_out_groups_and_buffer():
    coordinates = np.array([[0.0], [0.2], [3.0], [3.2], [10.0], [10.2]])
    groups = np.array(["a", "a", "b", "b", "c", "c"])

    splits = list(
        spatial_grouped_cv(
            coordinates,
            groups,
            n_splits=3,
            buffer_radius=1.0,
        )
    )

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert set(groups[train_idx]).isdisjoint(set(groups[test_idx]))
        distances = np.abs(coordinates[train_idx] - coordinates[test_idx].T)
        assert np.min(distances) > 1.0


def test_spatial_grouped_cv_preserves_tuple_group_ids():
    coordinates = np.array([[0.0], [0.2], [10.0], [10.2]])
    groups = [("pickup", 1), ("pickup", 1), ("dropoff", 2), ("dropoff", 2)]

    splits = list(spatial_grouped_cv(coordinates, groups, n_splits=2))

    assert len(splits) == 2
    for train_idx, test_idx in splits:
        train_groups = {groups[index] for index in train_idx}
        test_groups = {groups[index] for index in test_idx}
        assert train_groups.isdisjoint(test_groups)


def test_spatial_grouped_cv_supports_pandas_coordinate_columns():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "pickup_x": [0.0, 0.2, 3.0, 3.2, 10.0, 10.2],
            "pickup_y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "route_group": ["a", "a", "b", "b", "c", "c"],
        }
    )

    splits = list(
        spatial_grouped_cv(
            frame,
            frame["route_group"],
            n_splits=3,
            buffer_radius=1.0,
            coordinate_cols=("pickup_x", "pickup_y"),
        )
    )

    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert set(frame.loc[train_idx, "route_group"]).isdisjoint(
            set(frame.loc[test_idx, "route_group"])
        )


def test_spatial_grouped_cv_rejects_degree_buffer_without_opt_in():
    with pytest.raises(ValueError, match="linear CRS"):
        list(
            spatial_grouped_cv(
                [[40.7, -73.9], [40.8, -74.0], [41.5, -75.0]],
                ["a", "b", "c"],
                n_splits=2,
                buffer_radius=0.1,
                coordinate_units="degrees",
            )
        )


def test_environmental_blocked_cv_supports_deterministic_non_sklearn_blocks():
    features = {"temp": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0], "humidity": [4, 4, 4, 9, 9, 9]}

    splits = list(
        environmental_blocked_cv(
            features,
            feature_cols=["temp", "humidity"],
            n_splits=2,
            use_sklearn=False,
        )
    )

    assert len(splits) == 2
    assert sorted(np.concatenate([test_idx for _, test_idx in splits]).tolist()) == list(range(6))


def test_environmental_blocked_cv_supports_pandas_feature_columns():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        {
            "temperature": [0.0, 1.0, 2.0, 10.0, 11.0, 12.0],
            "rain_mm": [4, 4, 4, 9, 9, 9],
            "ignored": ["a", "b", "c", "d", "e", "f"],
        }
    )

    splits = list(
        environmental_blocked_cv(
            frame,
            feature_cols=["temperature", "rain_mm"],
            n_splits=2,
            use_sklearn=False,
        )
    )

    assert len(splits) == 2
    assert sorted(np.concatenate([test_idx for _, test_idx in splits]).tolist()) == list(range(6))


def test_blocked_cv_column_selector_errors_are_explicit():
    with pytest.raises(ValueError, match="coordinate_cols"):
        list(spatial_buffered_cv({"x": [0.0], "y": [1.0]}, n_splits=2, buffer_radius=0.0))
    with pytest.raises(ValueError, match="feature_cols"):
        list(environmental_blocked_cv({"temp": [0.0], "humidity": [1.0]}, n_splits=2))


def test_blocked_cv_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        list(grouped_blocked_cv([1, 2, 3], n_splits=1))
    with pytest.raises(ValueError, match="unique groups"):
        list(grouped_blocked_cv(["a", "a", "b"], n_splits=3))
    with pytest.raises(ValueError, match="gap must be non-negative"):
        list(temporal_blocked_cv([1, 2, 3], n_splits=2, gap=-1))
    with pytest.raises(ValueError, match="grid_shape"):
        list(spatial_blocked_cv([[0.0, 0.0], [1.0, 1.0]], n_splits=2, grid_shape=(1, 1)))
    with pytest.raises(ValueError, match="linear CRS"):
        list(
            spatial_buffered_cv(
                [[40.7, -73.9], [40.8, -74.0]],
                n_splits=2,
                buffer_radius=0.1,
                coordinate_units="degrees",
            )
        )


def test_out_of_time_split_validation_errors_are_explicit():
    with pytest.raises(ValueError, match="validation_fraction"):
        out_of_time_split([1, 2, 3], validation_fraction=1.0)
    with pytest.raises(ValueError, match="pass only one"):
        out_of_time_split([1, 2, 3], validation_size=1, validation_fraction=0.2)
    with pytest.raises(ValueError, match="leaves no validation"):
        out_of_time_split([1, 2, 3], validation_fraction=None, cutoff=3)
    with pytest.raises(ValueError, match="leaves no training"):
        out_of_time_split([1, 2, 3], validation_fraction=None, cutoff=1, gap=1)
