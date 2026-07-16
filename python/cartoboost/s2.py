from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import cache
from typing import Any

from . import _native
from .geo import route_latlng_points

__all__ = [
    "build_s2_route_sparse_sets",
    "build_s2_sparse_sets",
    "encode_s2_cells",
    "encode_s2_route_cells",
    "latlng_to_s2_id",
    "normalize_s2_id",
    "s2_parent_id",
]


def normalize_s2_id(value: Any) -> int:
    """Normalize an S2 cell ID into a non-negative integer."""

    if isinstance(value, bool):
        raise ValueError("S2 IDs must be non-negative integers or integer strings")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("S2 IDs must be non-negative")
        return value
    if isinstance(value, str):
        return int(_native.s2_normalize_id_text(value))
    raise ValueError("S2 IDs must be non-negative integers or integer strings")


def latlng_to_s2_id(latitude: Any, longitude: Any, *, level: int) -> int:
    """Encode a latitude/longitude pair with the optional ``s2sphere`` package."""

    s2sphere = _load_s2sphere()
    lvl = _normalize_level(level, "level")
    lat = _normalize_coordinate(latitude, "latitude")
    lng = _normalize_coordinate(longitude, "longitude")
    lat_lng = s2sphere.LatLng.from_degrees(lat, lng)
    cell = s2sphere.CellId.from_lat_lng(lat_lng).parent(lvl)
    return normalize_s2_id(cell.id())


def s2_parent_id(cell: Any, *, parent_level: int) -> int:
    """Return an S2 parent cell ID with the optional ``s2sphere`` package."""

    s2sphere = _load_s2sphere()
    parent = _normalize_level(parent_level, "parent_level")
    cell_id = s2sphere.CellId(normalize_s2_id(cell))
    if parent > cell_id.level():
        raise ValueError("parent_level must be less than or equal to the cell level")
    return normalize_s2_id(cell_id.parent(parent).id())


def encode_s2_cells(
    latitudes: Iterable[Any],
    longitudes: Iterable[Any],
    *,
    level: int,
) -> list[int]:
    """Encode latitude/longitude rows into S2 integer cell IDs."""

    lat_values = list(latitudes)
    lng_values = list(longitudes)
    if len(lat_values) != len(lng_values):
        raise ValueError("latitudes and longitudes must have the same number of rows")
    if not lat_values:
        raise ValueError("latitudes and longitudes must contain at least one row")
    return [
        latlng_to_s2_id(latitude, longitude, level=level)
        for latitude, longitude in zip(lat_values, lng_values, strict=True)
    ]


def encode_s2_route_cells(
    route: Any,
    *,
    level: int,
    parent_levels: Iterable[int] = (),
) -> list[int]:
    """Encode one decoded OSRM/Valhalla route into sorted S2 sparse IDs."""

    child_level = _normalize_level(level, "level")
    parents = [_normalize_level(value, "parent_levels") for value in parent_levels]
    _native.s2_validate_parent_levels_value(child_level, parents)
    cells: list[int] = []
    for latitude, longitude in route_latlng_points(route):
        child = latlng_to_s2_id(latitude, longitude, level=child_level)
        cells.append(child)
        cells.extend(s2_parent_id(child, parent_level=parent) for parent in parents)
    return [int(value) for value in _native.geo_assemble_route_sparse_rows_value([cells])[0]]


def build_s2_route_sparse_sets(
    routes: Iterable[Any],
    *,
    name: str = "route_s2",
    level: int,
    parent_levels: Iterable[int] = (),
) -> dict[str, list[list[int]]]:
    """Build one S2 sparse-set column from decoded route geometries."""

    if not name:
        raise ValueError("name must be non-empty")
    route_values = list(routes)
    if not route_values:
        raise ValueError("routes must contain at least one row")
    parents = tuple(parent_levels)
    rows = [
        encode_s2_route_cells(
            route,
            level=level,
            parent_levels=parents,
        )
        for route in route_values
    ]
    return {name: [list(row) for row in _native.geo_assemble_route_sparse_rows_value(rows)]}


def build_s2_sparse_sets(
    coordinates: dict[str, tuple[Iterable[Any], Iterable[Any]]],
    *,
    level: int,
    parent_levels: Iterable[int] = (),
) -> dict[str, list[list[int]]]:
    """Build sparse-set columns from named latitude/longitude coordinate pairs.

    ``coordinates`` maps output sparse-set names to ``(latitudes, longitudes)``.
    Each row contains the child S2 cell plus any requested S2 parent cells.
    """

    if not coordinates:
        raise ValueError("coordinates cannot be empty")
    child_level = _normalize_level(level, "level")
    parents = [_normalize_level(value, "parent_levels") for value in parent_levels]
    _native.s2_validate_parent_levels_value(child_level, parents)

    sparse_sets: dict[str, list[list[int]]] = {}
    expected_rows: int | None = None
    for name, pair in coordinates.items():
        if not name:
            raise ValueError("coordinate names must be non-empty")
        try:
            latitudes, longitudes = pair
        except (TypeError, ValueError) as exc:
            raise ValueError("coordinate values must be (latitudes, longitudes) pairs") from exc
        cells = encode_s2_cells(latitudes, longitudes, level=child_level)
        if expected_rows is None:
            expected_rows = len(cells)
        else:
            _native.geo_validate_equal_row_count_value(name, len(cells), expected_rows)
        parent_columns = [
            [s2_parent_id(cell, parent_level=parent) for cell in cells] for parent in parents
        ]
        sparse_sets[name] = [
            [int(value) for value in row]
            for row in _native.geo_assemble_sparse_column_value(cells, parent_columns)
        ]
    return sparse_sets


@cache
def _load_s2sphere() -> Any:
    try:
        return importlib.import_module("s2sphere")
    except ImportError as exc:
        raise ImportError(
            "S2 auto-encoding requires the optional 's2sphere' package. "
            "Install it with `pip install s2sphere`."
        ) from exc


def _normalize_coordinate(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite coordinate")
    try:
        coordinate = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite coordinate") from exc
    return float(_native.geo_normalize_coordinate_value(coordinate, field_name))


def _normalize_level(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer S2 level")
    try:
        level = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer S2 level") from exc
    if level != value:
        raise ValueError(f"{field_name} must be an integer S2 level")
    return int(_native.s2_normalize_level_value(level, field_name))
