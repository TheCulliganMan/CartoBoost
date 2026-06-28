from __future__ import annotations

import json

import cartoboost.h3 as h3_helpers
import cartoboost.s2 as s2_helpers
import pytest
from cartoboost.geo import (
    CoordinateMatrix,
    GeoSpatialWeights,
    PanelIndex,
    TimeIndex,
    buffered_spatial_cv_manifest,
    clockwise_bearing_unit_vector,
    clockwise_bearing_unit_vectors,
    group_spatial_cv_manifest,
    initial_bearing_unit_vector_latlng,
    initial_bearing_unit_vectors_latlng,
    local_frame_features,
    radial_anchor_distances,
    rbf_anchor_features,
    rolling_origin_panel_split_manifest,
    route_feature_vector,
    route_latlng_points,
    spatial_block_cv_manifest,
    spatial_temporal_blocked_split_manifest,
)


def _metadata() -> dict[str, object]:
    return {
        "dataset_fingerprint": "sha256:test",
        "coordinate_crs_note": "EPSG:2263 projected taxi zone centroids",
        "model_version": "0.2.32",
        "dependency_versions": {"cartoboost": "0.2.32"},
        "random_seed": 42,
    }


def test_geo_containers_round_trip() -> None:
    coords = CoordinateMatrix([0.0, 1.0], [2.0, 3.0], crs="EPSG:4326")
    assert coords.len() == 2
    assert CoordinateMatrix.from_json(coords.to_json()).len() == 2

    time = TimeIndex(["2024-01-01", "2024-01-02"], frequency="D")
    assert time.timestamps()[0].startswith("2024-01-01T00:00:00")

    panel = PanelIndex(["zone_a", "zone_a"], time=time)
    assert PanelIndex.from_json(panel.to_json()).len() == 2


def test_bearing_unit_vectors_preserve_clockwise_wraparound() -> None:
    assert clockwise_bearing_unit_vector((0.0, 0.0), (0.0, 4.0)) == (0.0, 1.0)
    assert clockwise_bearing_unit_vector((0.0, 0.0), (4.0, 0.0)) == (1.0, 0.0)
    northwest = clockwise_bearing_unit_vector((0.0, 0.0), (-3.0, 3.0))
    assert northwest is not None
    assert northwest[0] < 0.0
    assert northwest[1] > 0.0
    assert clockwise_bearing_unit_vector((1.0, 1.0), (1.0, 1.0)) is None
    assert clockwise_bearing_unit_vectors([(0.0, 0.0)], [(0.0, 1.0)]) == [(0.0, 1.0)]


def test_latlng_initial_bearing_unit_vectors_are_native_backed() -> None:
    north = initial_bearing_unit_vector_latlng((40.0, -73.0), (41.0, -73.0))
    assert north is not None
    assert abs(north[0]) < 1.0e-12
    assert abs(north[1] - 1.0) < 1.0e-12
    northwest = initial_bearing_unit_vector_latlng((40.0, -73.0), (41.0, -74.0))
    assert northwest is not None
    assert northwest[0] < 0.0
    assert northwest[1] > 0.0
    assert initial_bearing_unit_vectors_latlng([(40.0, -73.0)], [(41.0, -73.0)])[0] == north


def test_route_radial_rbf_and_local_frame_features_are_native_backed() -> None:
    assert route_feature_vector((0.0, 0.0), (3.0, 4.0)) == (1.5, 2.0, 5.0, 0.6, 0.8)
    assert radial_anchor_distances((3.0, 4.0), [(0.0, 0.0), (3.0, 0.0)]) == [5.0, 4.0]
    rbf = rbf_anchor_features((0.0, 0.0), [(0.0, 0.0), (1.0, 0.0)], length_scale=1.0)
    assert rbf[0] == 1.0
    assert abs(rbf[1] - 0.6065306597) < 1.0e-9
    assert local_frame_features((2.0, 3.0), (1.0, 1.0), (0.0, 1.0)) == (2.0, -1.0)


def test_route_latlng_points_accepts_decoded_osrm_and_valhalla_shapes() -> None:
    osrm_route = {
        "geometry": {
            "type": "LineString",
            "coordinates": [[-73.9855, 40.7580], [-73.9570, 40.7804]],
        }
    }
    assert route_latlng_points(osrm_route) == [(40.7580, -73.9855), (40.7804, -73.9570)]
    assert route_latlng_points({"shape": [{"lat": 40.0, "lon": -73.0}]}) == [(40.0, -73.0)]
    with pytest.raises(ValueError, match="encoded polyline"):
        route_latlng_points({"geometry": "}_p~F~ps|U_ulLnnqC"})


def test_h3_route_sparse_sets_use_native_route_row_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        h3_helpers,
        "latlng_to_h3_id",
        lambda latitude, longitude, *, resolution: int(round(latitude * 10)),
    )
    monkeypatch.setattr(
        h3_helpers,
        "h3_parent_id",
        lambda cell, *, parent_resolution: int(cell) // 10 + parent_resolution,
    )

    sparse_sets = h3_helpers.build_h3_route_sparse_sets(
        [
            [(40.0, -73.0), (40.0, -73.0), (41.0, -73.5)],
            {"geometry": {"type": "LineString", "coordinates": [[-74.0, 42.0]]}},
        ],
        resolution=9,
        parent_resolutions=[5],
    )

    assert sparse_sets == {"route_h3": [[45, 46, 400, 410], [47, 420]]}


def test_s2_route_sparse_sets_use_native_route_row_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        s2_helpers,
        "latlng_to_s2_id",
        lambda latitude, longitude, *, level: int(round(abs(longitude) * 10)),
    )
    monkeypatch.setattr(
        s2_helpers,
        "s2_parent_id",
        lambda cell, *, parent_level: int(cell) // 10 + parent_level,
    )

    sparse_sets = s2_helpers.build_s2_route_sparse_sets(
        [[(40.0, -73.0), (40.5, -73.0), (41.0, -73.2)]],
        level=12,
        parent_levels=[8],
    )

    assert sparse_sets == {"route_s2": [[81, 730, 732]]}


def test_spatial_weights_csr_round_trip_and_checks() -> None:
    weights = GeoSpatialWeights.from_edges(3, [(0, 1, 2.0), (1, 0, 2.0)])
    assert weights.is_symmetric(0.0)
    assert weights.isolated_nodes() == [2]
    normalized = weights.row_normalize()
    payload = json.loads(normalized.to_json())
    assert payload["row_normalized"] is True
    assert GeoSpatialWeights.from_json(normalized.to_json()).isolated_nodes() == [2]


def test_spatial_split_manifests_are_deterministic() -> None:
    coords = CoordinateMatrix([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 0.0])
    first = spatial_block_cv_manifest(
        coords, n_folds=2, split_id="pickup_zone_block", **_metadata()
    )
    second = spatial_block_cv_manifest(
        coords, n_folds=2, split_id="pickup_zone_block", **_metadata()
    )

    assert first.folds() == second.folds()
    assert first.hash() == second.hash()
    assert first.hash().startswith("sha256:")


def test_geo_splitter_variants_produce_manifest_hashes() -> None:
    coords = CoordinateMatrix([0.0, 10.0, 1.0, 11.0], [0.0, 0.0, 0.0, 0.0])
    time = TimeIndex(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"])
    panel = PanelIndex(["a", "a", "b", "b"], time=time)

    manifests = [
        buffered_spatial_cv_manifest(coords, n_folds=2, buffer_distance=0.25, **_metadata()),
        group_spatial_cv_manifest(["a", "a", "b", "b"], n_folds=2, **_metadata()),
        rolling_origin_panel_split_manifest(
            panel,
            min_train_size=1,
            horizon=1,
            step=1,
            **_metadata(),
        ),
        spatial_temporal_blocked_split_manifest(
            coords,
            time,
            n_spatial_folds=2,
            min_train_time=2,
            horizon=2,
            **_metadata(),
        ),
    ]

    for manifest in manifests:
        assert manifest.hash().startswith("sha256:")
        assert manifest.folds()
