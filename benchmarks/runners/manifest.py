"""Load and validate public benchmark manifest files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACKS_DIR = ROOT / "tracks"
CONFIGS_DIR = ROOT / "configs"

OFFICIAL_GEO_FAMILIES: dict[str, dict[str, Any]] = {
    "nyc_tlc_zone_lane_demand": {
        "track": "spatial",
        "datasets": {"nyc_tlc_yellow_taxi_q1_2024_v1"},
        "baselines": {"lightgbm", "xgboost", "catboost", "hist_gradient_boosting"},
        "diagnostics": {
            "residual_morans_i",
            "spatial_block_error",
            "calibration_by_region",
            "interval_width_by_region",
        },
    },
    "metr_la_pems_graph_forecasting": {
        "track": "graph",
        "datasets": {"metr_la_traffic_speed_v1", "pems_bay_traffic_speed_v1"},
        "baselines": {"pytorch_geometric_temporal_baseline", "dcrnn_baseline"},
        "diagnostics": {"horizon_error", "graph_distance_residual_decay"},
    },
    "epa_air_quality_interpolation": {
        "track": "spatial",
        "datasets": {"epa_air_quality_daily_pm25_v1"},
        "baselines": {"pysal_spatial_regression", "pykrige", "gstools"},
        "diagnostics": {
            "residual_morans_i",
            "variogram_plot",
            "spatial_block_error",
            "calibration_by_region",
            "interval_width_by_region",
        },
    },
    "california_housing_sanity": {
        "track": "tabular",
        "datasets": {"sklearn_california_housing_regression_seed42_5000_v1"},
        "baselines": {"lightgbm", "xgboost", "catboost", "hist_gradient_boosting"},
        "diagnostics": {"spatial_block_error"},
    },
    "synthetic_spatial_fields": {
        "track": "spatial",
        "datasets": {"synthetic_spatial_fields_v1"},
        "baselines": {"pysal_spatial_regression", "pykrige", "gstools"},
        "diagnostics": {"residual_morans_i", "variogram_plot", "spatial_block_error"},
    },
    "synthetic_graph_diffusion": {
        "track": "graph",
        "datasets": {"synthetic_graph_diffusion_v1"},
        "baselines": {"pytorch_geometric_temporal_baseline", "graphsage_baseline"},
        "diagnostics": {"graph_distance_residual_decay", "horizon_error"},
    },
    "synthetic_geo_causal_lift_panels": {
        "track": "spatial",
        "datasets": {"synthetic_geo_causal_lift_panels_v1"},
        "baselines": {"pysal_spatial_regression", "mean"},
        "diagnostics": {"causal_placebo_summary", "spatial_block_error"},
    },
}

REQUIRED_CLAIM_SCORECARD_OUTPUTS = {"leaderboard_json", "leaderboard_markdown"}
REQUIRED_CLAIM_RESOURCE_METRICS = {
    "fit_wallclock_seconds",
    "predict_wallclock_seconds",
    "peak_memory_mb",
}


class ManifestError(ValueError):
    """Raised when a benchmark manifest is incomplete or inconsistent."""


@dataclass(frozen=True)
class TrackSpec:
    """Resolved manifest files for a benchmark track."""

    name: str
    datasets: dict[str, Any]
    tasks: dict[str, Any]
    metrics: dict[str, Any]
    splits: dict[str, Any]
    search_spaces: dict[str, Any]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestError(f"{path} must contain a JSON object")
    return value


def load_track(track: str, root: Path = TRACKS_DIR) -> TrackSpec:
    track_dir = root / track
    if not track_dir.exists():
        raise ManifestError(f"unknown benchmark track: {track}")

    loaded = {}
    for key, filename in {
        "datasets": "datasets.json",
        "tasks": "tasks.json",
        "metrics": "metrics.json",
        "splits": "splits.json",
        "search_spaces": "search_spaces.json",
    }.items():
        path = track_dir / filename
        if not path.exists():
            raise ManifestError(f"{track} is missing {filename}")
        loaded[key] = load_json(path)

    spec = TrackSpec(name=track, **loaded)
    validate_track(spec)
    return spec


def validate_track(spec: TrackSpec) -> None:
    dataset_ids = {item["id"] for item in _items(spec.datasets, "datasets", spec.name)}
    split_ids = {item["id"] for item in _items(spec.splits, "splits", spec.name)}
    metric_ids = {item["id"] for item in _items(spec.metrics, "metrics", spec.name)}
    search_ids = {
        item["model_family"] for item in _items(spec.search_spaces, "search_spaces", spec.name)
    }

    if "cartoboost" not in search_ids:
        raise ManifestError(f"{spec.name} must define a cartoboost search space")

    if spec.name in {"tabular", "spatial", "graph"}:
        validate_frozen_dataset_identities(spec)
    if spec.name == "spatial":
        validate_spatial_geo_splits(spec)

    for task in _items(spec.tasks, "tasks", spec.name):
        _require(
            task, ["id", "dataset_id", "split_id", "primary_metric", "required_model_families"]
        )
        if task["dataset_id"] not in dataset_ids:
            raise ManifestError(f"{spec.name}/{task['id']} references unknown dataset_id")
        if task["split_id"] not in split_ids:
            raise ManifestError(f"{spec.name}/{task['id']} references unknown split_id")
        if task["primary_metric"] not in metric_ids:
            raise ManifestError(f"{spec.name}/{task['id']} references unknown primary_metric")
        missing_models = set(task["required_model_families"]) - search_ids
        if missing_models:
            missing = ", ".join(sorted(missing_models))
            raise ManifestError(f"{spec.name}/{task['id']} missing search spaces: {missing}")


def validate_frozen_dataset_identities(spec: TrackSpec) -> None:
    for dataset in _items(spec.datasets, "datasets", spec.name):
        _require(dataset, ["id", "source", "source_url", "source_identity", "hash"])
        value = str(dataset["hash"])
        if value == "to_be_frozen" or not value.startswith("sha256:"):
            raise ManifestError(f"{spec.name}/{dataset['id']} must define a sha256 dataset hash")


def validate_spatial_geo_splits(spec: TrackSpec) -> None:
    geo_split_kinds = {
        "spatial_block_cv",
        "buffered_spatial_cv",
        "group_spatial_cv",
        "rolling_origin_panel_split",
        "spatial_temporal_blocked_split",
    }
    for split in _items(spec.splits, "splits", spec.name):
        _require(split, ["id", "kind", "coordinate_crs_note", "split_manifest_hash"])
        kind = str(split["kind"])
        if kind not in geo_split_kinds:
            raise ManifestError(f"spatial/{split['id']} must use a leakage-safe geo split kind")
        if "random" in split["id"] or "random" in kind:
            raise ManifestError(
                f"spatial/{split['id']} cannot use random row split as a geographic claim"
            )
        if split.get("random_row_split_allowed") is not False:
            raise ManifestError(
                f"spatial/{split['id']} must explicitly disallow random-row geographic claims"
            )
        if not str(split["split_manifest_hash"]).startswith("sha256:"):
            raise ManifestError(f"spatial/{split['id']} must include split manifest hash")


def load_config(name: str, root: Path = CONFIGS_DIR) -> dict[str, Any]:
    return load_json(root / f"{name}.json")


def validate_configs(root: Path = CONFIGS_DIR) -> None:
    seeds = load_config("seeds", root)
    _require(seeds, ["benchmark_seeds", "hpo_seeds"])
    if not seeds["benchmark_seeds"] or not seeds["hpo_seeds"]:
        raise ManifestError("seed lists must not be empty")

    budgets = load_config("budgets", root)
    for budget in _items(budgets, "budgets", "configs"):
        _require(budget, ["id", "max_trials", "max_wallclock_minutes", "early_stop_rounds"])
        if budget["max_trials"] <= 0 or budget["max_wallclock_minutes"] <= 0:
            raise ManifestError(f"invalid budget: {budget['id']}")

    baselines = load_config("required_baselines", root)
    for track in ["tabular", "spatial", "graph", "forecasting"]:
        if track not in baselines:
            raise ManifestError(f"required_baselines missing {track}")
    validate_required_baselines(baselines)
    validate_official_geo_benchmark_suite(load_all_tracks())


def validate_required_baselines(
    baselines: dict[str, Any],
    tracks_root: Path = TRACKS_DIR,
) -> None:
    for track, required in baselines.items():
        if not isinstance(required, list) or not required:
            raise ManifestError(f"required_baselines.{track} must be a non-empty list")
        spec = load_track(track, tracks_root)
        search_ids = {
            item["model_family"] for item in _items(spec.search_spaces, "search_spaces", track)
        }
        missing = set(required) - search_ids
        if missing:
            names = ", ".join(sorted(missing))
            raise ManifestError(f"required_baselines.{track} missing search spaces: {names}")


def load_all_tracks(root: Path = TRACKS_DIR) -> list[TrackSpec]:
    return [load_track(path.name, root) for path in sorted(root.iterdir()) if path.is_dir()]


def validate_official_geo_benchmark_suite(specs: list[TrackSpec]) -> None:
    """Validate benchmark manifests that back geographic modeling claims."""

    specs_by_name = {spec.name: spec for spec in specs}
    tasks_by_claim: dict[str, list[dict[str, Any]]] = {}
    datasets_by_track = {
        spec.name: {item["id"] for item in _items(spec.datasets, "datasets", spec.name)}
        for spec in specs
    }

    for spec in specs:
        for task in _items(spec.tasks, "tasks", spec.name):
            claim_family = task.get("claim_family")
            if claim_family is None:
                continue
            if claim_family not in OFFICIAL_GEO_FAMILIES:
                raise ManifestError(f"{spec.name}/{task['id']} uses unknown claim_family")
            tasks_by_claim.setdefault(str(claim_family), []).append(task)
            _require(
                task,
                [
                    "claim_family",
                    "required_diagnostics",
                    "scorecard_outputs",
                    "resource_metrics",
                ],
            )
            _require_set(
                task,
                "scorecard_outputs",
                REQUIRED_CLAIM_SCORECARD_OUTPUTS,
                f"{spec.name}/{task['id']}",
            )
            _require_set(
                task,
                "resource_metrics",
                REQUIRED_CLAIM_RESOURCE_METRICS,
                f"{spec.name}/{task['id']}",
            )

    for claim_family, requirement in OFFICIAL_GEO_FAMILIES.items():
        track = requirement["track"]
        spec = specs_by_name.get(track)
        if spec is None:
            raise ManifestError(f"{claim_family} references missing track: {track}")

        if not requirement["datasets"] <= datasets_by_track[track]:
            missing = ", ".join(sorted(requirement["datasets"] - datasets_by_track[track]))
            raise ManifestError(f"{claim_family} missing datasets: {missing}")

        claim_tasks = tasks_by_claim.get(claim_family, [])
        if not claim_tasks:
            raise ManifestError(f"{claim_family} must have at least one benchmark task")

        task_dataset_ids = {task["dataset_id"] for task in claim_tasks}
        if not requirement["datasets"] <= task_dataset_ids:
            missing = ", ".join(sorted(requirement["datasets"] - task_dataset_ids))
            raise ManifestError(f"{claim_family} missing task coverage: {missing}")

        task_baselines = set().union(
            *(set(task["required_model_families"]) for task in claim_tasks)
        )
        missing_baselines = requirement["baselines"] - task_baselines
        if missing_baselines:
            names = ", ".join(sorted(missing_baselines))
            raise ManifestError(f"{claim_family} missing required baselines: {names}")

        task_diagnostics = set().union(*(set(task["required_diagnostics"]) for task in claim_tasks))
        missing_diagnostics = requirement["diagnostics"] - task_diagnostics
        if missing_diagnostics:
            names = ", ".join(sorted(missing_diagnostics))
            raise ManifestError(f"{claim_family} missing required diagnostics: {names}")


def _items(mapping: dict[str, Any], key: str, owner: str) -> list[dict[str, Any]]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{owner} must define a non-empty {key} list")
    for item in value:
        if not isinstance(item, dict):
            raise ManifestError(f"{owner}.{key} entries must be objects")
    return value


def _require(mapping: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ManifestError(f"missing required keys: {', '.join(missing)}")


def _require_set(
    mapping: dict[str, Any],
    key: str,
    required: set[str],
    owner: str,
) -> None:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{owner} must define a non-empty {key} list")
    missing = required - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise ManifestError(f"{owner} missing {key}: {names}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", action="append", help="Track to validate. Defaults to all.")
    args = parser.parse_args(argv)

    validate_configs()
    tracks = args.track or [path.name for path in sorted(TRACKS_DIR.iterdir()) if path.is_dir()]
    for track in tracks:
        spec = load_track(track)
        print(f"validated {spec.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
