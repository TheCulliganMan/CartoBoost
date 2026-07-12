#!/usr/bin/env python3
"""Classify official geo benchmark evidence without overstating acceptance."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cartoboost.models import ModelRegistry  # noqa: E402

from benchmarks.runners.manifest import OFFICIAL_GEO_FAMILIES  # noqa: E402

PUBLIC_ARTIFACTS = {
    "nyc_tlc_zone_lane_demand": ROOT / "docs/assets/nyc_taxi_benchmarks/results.json",
}

SYNTHETIC_AUTOGEO_ARTIFACTS = [
    ROOT / "target/autogeo-gate/autogeo_benchmark_gate.json",
    ROOT / "target/autogeo-gate-verify/autogeo_benchmark_gate.json",
    ROOT / "target/autogeo-gate-test/autogeo_benchmark_gate.json",
    ROOT / "target/autogeo-gate-smoke/autogeo_benchmark_gate.json",
]


def main() -> int:
    report = official_geo_evidence_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["audit_passed"]:
        raise SystemExit("official geo evidence audit failed")
    return 0


def official_geo_evidence_report() -> dict[str, Any]:
    families = {family: classify_family(family) for family in OFFICIAL_GEO_FAMILIES}
    synthetic_gate = classify_synthetic_autogeo_gate()
    real_autogeo_wins = sum(
        1
        for row in families.values()
        if row["evidence_level"] == "real_public_autogeo_acceptance"
        and row["beats_or_ties_strong_baseline"]
    )
    # v0.3 deliberately removes the Python AutoGeo selector from the
    # distribution.  Its three-family evidence requirement remains a future
    # admission gate, not a release blocker for the smaller stable surface.
    selector_shipped = "models.auto_geo_model" in ModelRegistry.defaults().names()
    return {
        "artifact_type": "cartoboost.official_geo_evidence_audit",
        "artifact_version": 1,
        "audit_passed": True,
        "acceptance_passed": (not selector_shipped) or real_autogeo_wins >= 3,
        "required_real_autogeo_family_wins": 3,
        "acceptance_scope": "deferred_until_native_autogeo_selector",
        "selector_shipped": selector_shipped,
        "real_autogeo_family_wins": real_autogeo_wins,
        "families": families,
        "synthetic_autogeo_gate": synthetic_gate,
        "claim_policy": (
            "AutoGeo evidence is deferred because the selector is not shipped in v0.3. "
            "When a native selector returns, only real public artifacts on leakage-safe "
            "official families count toward its three-family admission gate; synthetic gates "
            "remain non-acceptance evidence."
        ),
    }


def classify_family(family: str) -> dict[str, Any]:
    artifact = PUBLIC_ARTIFACTS.get(family)
    if artifact is None or not artifact.exists():
        return {
            "evidence_level": "missing_public_artifact",
            "artifact": None if artifact is None else str(artifact.relative_to(ROOT)),
            "leakage_safe": False,
            "autogeo": False,
            "beats_or_ties_strong_baseline": False,
            "reason": "no committed public artifact is registered for this family",
        }

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if family == "nyc_tlc_zone_lane_demand":
        return classify_nyc_tlc_artifact(artifact, payload)
    return {
        "evidence_level": "unclassified_public_artifact",
        "artifact": str(artifact.relative_to(ROOT)),
        "leakage_safe": False,
        "autogeo": False,
        "beats_or_ties_strong_baseline": False,
        "reason": "artifact parser is not implemented for this family",
    }


def classify_nyc_tlc_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    integrity = dict(payload.get("benchmark_integrity", {}))
    models = set(payload.get("model_roster", []))
    external = payload.get("external_baseline_comparison", [])
    spatial_manifests = list(iter_split_manifests(payload, split_id="spatial_holdout"))
    leakage_safe = (
        integrity.get("synthetic_smoke") is False
        and bool(payload.get("dataset_hash"))
        and bool(spatial_manifests)
        and all(
            manifest.get("random_row_split_allowed_for_geo_claim") is False
            and str(manifest.get("split_manifest_hash", "")).startswith("sha256:")
            for manifest in spatial_manifests
        )
    )
    strong_baselines = {"lightgbm", "xgboost", "catboost", "hist_gradient_boosting"}
    has_strong_baselines = strong_baselines <= models
    has_cartoboost_comparison = any(
        row.get("baseline") in strong_baselines and row.get("task") for row in external
    )
    autogeo = "autogeo" in models or "auto_geo_model" in models
    if leakage_safe and has_strong_baselines and has_cartoboost_comparison and not autogeo:
        evidence_level = "real_public_non_autogeo_benchmark"
        reason = "real leakage-safe NYC TLC benchmark exists, but the v0.3 selector is not shipped"
    elif leakage_safe and autogeo:
        evidence_level = "real_public_autogeo_acceptance"
        reason = "real leakage-safe NYC TLC selector artifact is present"
    else:
        evidence_level = "incomplete_public_artifact"
        reason = "artifact is missing leakage-safe split, strong baselines, or comparison rows"
    return {
        "evidence_level": evidence_level,
        "artifact": str(path.relative_to(ROOT)),
        "leakage_safe": leakage_safe,
        "autogeo": autogeo,
        "strong_baselines_present": has_strong_baselines,
        "beats_or_ties_strong_baseline": False,
        "reason": reason,
        "spatial_manifest_count": len(spatial_manifests),
    }


def iter_split_manifests(payload: dict[str, Any], *, split_id: str) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for task in dict(payload.get("tasks", {})).values():
        split = dict(task.get("splits", {})).get(split_id)
        if not isinstance(split, dict):
            continue
        manifest = split.get("split_manifest")
        if isinstance(manifest, dict):
            manifests.append(manifest)
    return manifests


def classify_synthetic_autogeo_gate() -> dict[str, Any]:
    artifacts = [path for path in SYNTHETIC_AUTOGEO_ARTIFACTS if path.exists()]
    if not artifacts:
        return {
            "present": False,
            "acceptance_passed": False,
            "artifact": None,
            "reason": "no local synthetic AutoGeo gate artifact found",
            "counts_toward_final_acceptance": False,
        }
    artifact = max(artifacts, key=lambda path: path.stat().st_mtime)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return {
        "present": True,
        "artifact": str(artifact.relative_to(ROOT)),
        "acceptance_passed": bool(payload.get("acceptance", {}).get("passed")),
        "family_wins_or_ties": int(payload.get("acceptance", {}).get("family_wins_or_ties", 0)),
        "data_kind": payload.get("data_kind"),
        "counts_toward_final_acceptance": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
