#!/usr/bin/env python3
"""Check NYC Taxi Path C claim gate artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "docs" / "assets" / "nyc_taxi_benchmarks" / "path_c_claims.json"

REQUIRED_FIELDS = {
    "claim_id",
    "task",
    "split_kind",
    "train_index_sha256",
    "test_index_sha256",
    "dataset_hash",
    "model",
    "architecture",
    "capability_tier",
    "falsifier_baseline",
    "primary_metric",
    "improvement_threshold",
    "rmse",
    "mae",
    "wape",
    "r2",
    "fit_seconds",
    "predict_seconds",
    "peak_memory_mb",
    "save_load_max_abs_diff",
    "feature_access_policy",
    "target_encoding_train_only",
    "selection_uses_outer_test_labels",
}
SPATIAL_TRANSFER = "spatial_transfer"
TEMPORAL_CLAIMS = {"temporal_structure", "known_future_sensitivity"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_INPUT)
    return parser.parse_args()


def load_artifact(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Path C artifact is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_artifact(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    rows = data.get("claims")
    if not isinstance(rows, list) or not rows:
        return ["Path C artifact must contain a non-empty claims list"]

    integrity = data.get("benchmark_integrity", {})
    dataset = data.get("dataset", {})
    if integrity.get("synthetic_smoke") is True or dataset.get("source") == "synthetic_smoke":
        errors.append("synthetic benchmarks cannot be counted as Path C evidence")
    if dataset.get("source") != "nyc_tlc_trip_records":
        errors.append("Path C evidence must use NYC TLC trip records")
    if dataset.get("taxi_type") != "yellow" or int(dataset.get("year", 0)) != 2024:
        errors.append("Path C evidence must use 2024 yellow taxi data")

    claims_seen: set[str] = set()
    falsifiers_by_claim: dict[str, set[str]] = {}
    for index, row in enumerate(rows):
        label = f"claims[{index}]"
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"{label} missing required fields: {', '.join(missing)}")
            continue
        claim_id = str(row["claim_id"])
        claims_seen.add(claim_id)
        falsifier = str(row.get("falsifier_baseline") or "")
        falsifiers_by_claim.setdefault(claim_id, set()).add(falsifier)
        if not falsifier:
            errors.append(f"{label} lacks a falsifier baseline")
        if str(row.get("train_index_sha256", "")).strip() in {"", "sha256:"}:
            errors.append(f"{label} missing train_index_sha256")
        if str(row.get("test_index_sha256", "")).strip() in {"", "sha256:"}:
            errors.append(f"{label} missing test_index_sha256")
        if not bool(row.get("target_encoding_train_only")):
            errors.append(f"{label} target encoding is not train-only")
        if bool(row.get("selection_uses_outer_test_labels")):
            errors.append(f"{label} uses outer test labels for selection")
        if row.get("save_load_max_abs_diff") is None:
            errors.append(f"{label} save/load parity is missing")
        if claim_id in TEMPORAL_CLAIMS and row.get("split_kind") == "random":
            errors.append(f"{label} uses random split for temporal claim")
        if claim_id == SPATIAL_TRANSFER and row.get("split_kind") == "random":
            errors.append(f"{label} uses random split for spatial-transfer claim")
        if (
            claim_id == "directional_structure"
            and row.get("directionality_tested") != "A_to_B_vs_B_to_A"
        ):
            errors.append(f"{label} ordered-pair claim does not test A->B vs B->A")
        if claim_id == "known_future_sensitivity" and "known_future_ablation_delta_rmse" not in row:
            errors.append(f"{label} known-future ablation delta is missing")
        if row.get("gate_required", True) and row.get("passed") is not True:
            errors.append(f"{label} did not pass its claim gate")

    required_claims = {
        "directional_structure",
        "temporal_structure",
        "known_future_sensitivity",
        "spatial_transfer",
        "residual_correction",
    }
    missing_claims = sorted(required_claims - claims_seen)
    if missing_claims:
        errors.append(f"missing Path C claims: {', '.join(missing_claims)}")

    directional_falsifiers = falsifiers_by_claim.get("directional_structure", set())
    for required in {"unordered_pair_baseline", "source_target_additive_baseline"}:
        if required not in directional_falsifiers:
            errors.append(f"directional_structure missing falsifier: {required}")
    temporal_falsifiers = falsifiers_by_claim.get("temporal_structure", set())
    for required in {"trailing_mean", "seasonal_naive", "pooled_ridge"}:
        if required not in temporal_falsifiers:
            errors.append(f"temporal_structure missing falsifier: {required}")
    spatial_falsifiers = falsifiers_by_claim.get("spatial_transfer", set())
    for required in {"target_encoded_zone_only_baseline", "mean_baseline"}:
        if required not in spatial_falsifiers:
            errors.append(f"spatial_transfer missing falsifier: {required}")
    residual_falsifiers = falsifiers_by_claim.get("residual_correction", set())
    for required in {"raw_baseline", "global_residual_mean", "linear_residual_model"}:
        if required not in residual_falsifiers:
            errors.append(f"residual_correction missing falsifier: {required}")
    if "future_known_covariates_ablated" not in falsifiers_by_claim.get(
        "known_future_sensitivity",
        set(),
    ):
        errors.append("known_future_sensitivity ablation is missing")
    return errors


def main() -> None:
    args = parse_args()
    errors = check_artifact(load_artifact(args.artifact))
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Path C gates passed: {args.artifact}")


if __name__ == "__main__":
    main()
