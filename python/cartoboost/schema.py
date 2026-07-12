from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any


class FeatureKind(str, Enum):
    NUMERIC = "Numeric"
    SPATIAL = "Spatial"
    CATEGORICAL = "Categorical"
    ORDINAL = "Ordinal"
    SPARSE_SET = "SparseSet"
    H3_SPARSE_SET = "H3SparseSet"
    PERIODIC = "Periodic"


@dataclass(frozen=True)
class NumericSpec:
    name: str
    kind: FeatureKind = FeatureKind.NUMERIC


@dataclass(frozen=True)
class CategoricalSpec:
    name: str
    kind: FeatureKind = FeatureKind.CATEGORICAL


@dataclass(frozen=True)
class OrdinalSpec:
    name: str
    kind: FeatureKind = FeatureKind.ORDINAL


@dataclass(frozen=True)
class PeriodicSpec:
    name: str
    period: int
    kind: FeatureKind = FeatureKind.PERIODIC

    def __post_init__(self) -> None:
        if int(self.period) != self.period or self.period <= 0:
            raise ValueError("periodic feature specs require a positive integer period")


@dataclass(frozen=True)
class SpatialPairSpec:
    name: str
    other: str
    kind: FeatureKind = FeatureKind.SPATIAL


@dataclass(frozen=True)
class SparseSetSpec:
    name: str
    kind: FeatureKind = FeatureKind.SPARSE_SET


FeatureSpec = (
    NumericSpec | CategoricalSpec | OrdinalSpec | PeriodicSpec | SpatialPairSpec | SparseSetSpec
)


_NUMERIC_KIND_ALIASES = frozenset({FeatureKind.NUMERIC, "numeric"})
_CATEGORICAL_KIND_ALIASES = frozenset(
    {FeatureKind.CATEGORICAL, "categorical", "category", "nominal", "string", "object"}
)
_ORDINAL_KIND_ALIASES = frozenset({FeatureKind.ORDINAL, "ordinal", "ordered"})
_SPATIAL_KIND_ALIASES = frozenset(
    {
        FeatureKind.SPATIAL,
        "spatial",
        "geo",
        "geometry",
        "coordinate",
        "coordinates",
        "centroid",
        "Spatial",
    }
)
_SPARSE_SET_KIND_ALIASES = frozenset(
    {
        FeatureKind.SPARSE_SET,
        "sparse_set",
        "sparse",
        "zip_sparse_set",
        "zip3_sparse_set",
        "zone_sparse_set",
        "region_sparse_set",
        "ZipSparseSet",
        "ZoneSparseSet",
        "RegionSparseSet",
        "GeoSparseSet",
        "GeoAbstractSparseSet",
    }
)
_H3_SPARSE_SET_KIND_ALIASES = frozenset(
    {FeatureKind.H3_SPARSE_SET, "h3_sparse_set", "h3_sparse", "H3SparseSet"}
)
_PERIODIC_KIND_ALIASES = frozenset({FeatureKind.PERIODIC, "periodic"})


@dataclass(frozen=True)
class FeatureSchema:
    """Python helper for CartoBoost dense and sparse feature metadata."""

    dense: list[Any] | tuple[Any, ...]
    sparse_sets: list[Any] | tuple[Any, ...] | None = None

    @classmethod
    def from_specs(
        cls,
        dense: list[FeatureSpec] | tuple[FeatureSpec, ...],
        sparse_sets: list[SparseSetSpec] | tuple[SparseSetSpec, ...] = (),
    ) -> FeatureSchema:
        return cls(dense=dense, sparse_sets=sparse_sets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dense": list(self.dense),
            "sparse_sets": list(self.sparse_sets or []),
        }

    def to_rust_payload(self, dense_width: int, sparse_names: list[str]) -> dict[str, Any]:
        dense_entries = list(self.dense)
        sparse_entries = list(self.sparse_sets or [])
        names = [_entry_name(entry, idx, "feature") for idx, entry in enumerate(dense_entries)]
        kinds = [_entry_kind(entry, FeatureKind.NUMERIC) for entry in dense_entries]
        spatial_names = {
            str(other)
            for entry in dense_entries
            if (other := _spatial_pair_other(entry)) is not None
        }
        # A spatial pair declares both coordinates as spatial features at the
        # native boundary.  This keeps Rust's candidate search restricted to
        # explicitly declared coordinate columns even when the second member
        # was supplied as a plain NumericSpec for ergonomic convenience.
        kinds = [
            FeatureKind.SPATIAL if name in spatial_names else kind
            for name, kind in zip(names, kinds, strict=True)
        ]
        names.extend(
            _entry_name(entry, idx, "sparse_set") for idx, entry in enumerate(sparse_entries)
        )
        kinds.extend(_entry_kind(entry, FeatureKind.SPARSE_SET) for entry in sparse_entries)
        _validate_length(names, kinds, dense_width, sparse_names)
        return {"names": names, "kinds": kinds}

    def to_json(self, dense_width: int, sparse_names: list[str]) -> str:
        payload = self.to_rust_payload(dense_width, sparse_names)
        self.validate(dense_width, sparse_names, payload=payload)
        return json.dumps(payload)

    def validate(
        self,
        dense_width: int,
        sparse_names: list[str],
        *,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """Validate this schema with the Rust-owned schema contract.

        The Python spec classes provide construction ergonomics only.  Final
        validation is deliberately delegated to the native extension so a
        schema cannot be accepted by Python and rejected later by training,
        artifact loading, or another runtime.
        """

        normalized = (
            payload if payload is not None else self.to_rust_payload(dense_width, sparse_names)
        )
        try:
            from . import _native
        except ImportError as exc:  # pragma: no cover - exercised in source-only installs
            raise RuntimeError(
                "cartoboost._native is required to validate FeatureSchema; "
                "install the built CartoBoost wheel or run maturin develop"
            ) from exc
        validator = getattr(_native, "validate_feature_schema_json", None)
        if validator is None:  # pragma: no cover - protects mismatched native wheels
            raise RuntimeError(
                "cartoboost._native.validate_feature_schema_json is unavailable; "
                "reinstall the CartoBoost native extension"
            )
        validator(json.dumps(normalized))


def normalize_feature_kind(kind: Any, entry: dict[str, Any] | None = None) -> Any:
    match kind:
        case dict():
            return _normalize_mapping_feature_kind(kind)
        case _ if kind in _NUMERIC_KIND_ALIASES:
            return FeatureKind.NUMERIC
        case _ if kind in _CATEGORICAL_KIND_ALIASES:
            return FeatureKind.CATEGORICAL
        case _ if kind in _ORDINAL_KIND_ALIASES:
            return FeatureKind.ORDINAL
        case _ if kind in _SPATIAL_KIND_ALIASES:
            return FeatureKind.SPATIAL
        case _ if kind in _SPARSE_SET_KIND_ALIASES:
            return FeatureKind.SPARSE_SET
        case _ if kind in _H3_SPARSE_SET_KIND_ALIASES:
            if entry is not None:
                _validate_h3_sparse_set_entry(entry)
            return FeatureKind.SPARSE_SET
        case _ if kind in _PERIODIC_KIND_ALIASES:
            period = 24 if entry is None else entry.get("period", 24)
            return _periodic_kind_payload(period)
        case _:
            raise ValueError(f"unknown feature kind {kind!r}")


def _normalize_mapping_feature_kind(kind: dict[str, Any]) -> Any:
    match kind:
        case {FeatureKind.PERIODIC: {"period": period}}:
            return _periodic_kind_payload(period)
        case {"periodic": {"period": period}}:
            return _periodic_kind_payload(period)
        case {"periodic": period}:
            return _periodic_kind_payload(period)
        case _:
            raise ValueError(f"unknown feature kind {kind!r}")


def _periodic_kind_payload(period: Any) -> dict[str, dict[str, int]]:
    return {FeatureKind.PERIODIC: {"period": _positive_period(period)}}


def _entry_name(entry: Any, idx: int, prefix: str) -> str:
    if isinstance(entry, dict):
        return str(entry.get("name", f"{prefix}_{idx}"))
    if isinstance(entry, tuple) and len(entry) == 2:
        return str(entry[0])
    if hasattr(entry, "name"):
        return str(entry.name)
    return str(entry)


def _entry_kind(entry: Any, default: FeatureKind) -> Any:
    match entry:
        case dict():
            return normalize_feature_kind(entry.get("kind", entry.get("role", default)), entry)
        case tuple() if len(entry) == 2:
            return normalize_feature_kind(entry[1])
        case _ if hasattr(entry, "kind"):
            return normalize_feature_kind(entry.kind, entry.__dict__)
        case _ if default is FeatureKind.SPARSE_SET:
            return FeatureKind.SPARSE_SET
        case _:
            return FeatureKind.NUMERIC


def _spatial_pair_other(entry: Any) -> Any | None:
    if isinstance(entry, dict):
        kind = entry.get("kind", entry.get("role", ""))
        other = entry.get("other")
    else:
        kind = getattr(entry, "kind", "")
        other = getattr(entry, "other", None)
    if getattr(kind, "value", kind) in _SPATIAL_KIND_ALIASES and other is not None:
        return other
    return None


def _positive_period(period: Any) -> int:
    try:
        value = float(period)
    except (TypeError, ValueError) as exc:
        raise ValueError("periodic feature_schema entries require a positive period") from exc
    if not math.isfinite(value) or value <= 0 or not value.is_integer():
        raise ValueError("periodic feature_schema entries require a positive integer period")
    return int(value)


def _h3_resolution(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"h3_sparse_set feature_schema entries require integer {field_name}")
    try:
        resolution = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"h3_sparse_set feature_schema entries require integer {field_name}"
        ) from exc
    if resolution != value and not (
        isinstance(value, float) and math.isfinite(value) and value.is_integer()
    ):
        raise ValueError(f"h3_sparse_set feature_schema entries require integer {field_name}")
    if resolution < 0 or resolution > 15:
        raise ValueError(f"h3_sparse_set {field_name} must be between 0 and 15")
    return resolution


def _validate_h3_sparse_set_entry(entry: dict[str, Any]) -> None:
    if "resolution" not in entry:
        raise ValueError("h3_sparse_set feature_schema entries require resolution")
    resolution = _h3_resolution(entry["resolution"], "resolution")
    parent_resolutions = entry.get("parent_resolutions", [])
    if parent_resolutions is None:
        parent_resolutions = []
    if isinstance(parent_resolutions, str) or not isinstance(
        parent_resolutions,
        list | tuple,
    ):
        raise ValueError("h3_sparse_set parent_resolutions must be a list of integer resolutions")
    for parent_resolution in parent_resolutions:
        parent = _h3_resolution(parent_resolution, "parent_resolutions")
        if parent >= resolution:
            raise ValueError("h3_sparse_set parent_resolutions must be less than resolution")


def _validate_length(
    names: list[str],
    kinds: list[Any],
    dense_width: int,
    sparse_names: list[str],
) -> None:
    expected = dense_width + len(sparse_names)
    if len(names) != len(kinds):
        raise ValueError("feature_schema names length must match kinds length")
    if len(names) != expected:
        raise ValueError(
            f"feature_schema length {len(names)} does not match dataset feature count {expected}"
        )
