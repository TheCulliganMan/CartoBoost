#!/usr/bin/env python3
"""Check release-engineering gates that should fail before packaging."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.runners.manifest import (  # noqa: E402
    OFFICIAL_GEO_FAMILIES,
    load_all_tracks,
    validate_configs,
    validate_official_geo_benchmark_suite,
)

PYPROJECT = ROOT / "pyproject.toml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish-pypi.yml"

BASELINE_DEPENDENCIES = {
    "catboost": {"catboost"},
    "darts_baseline": {"darts"},
    "gstools": {"gstools"},
    "lightgbm": {"lightgbm"},
    "neuralforecast_baseline": {"neuralforecast"},
    "pykrige": {"pykrige"},
    "pysal_spatial_regression": {"esda", "libpysal", "spreg"},
    "xgboost": {"xgboost"},
}

EXTERNAL_INSTALL_METADATA_BASELINES = {
    "dcrnn_baseline",
    "pytorch_geometric_temporal_baseline",
}

REQUIRED_CI_COMMANDS = {
    "uv run pre-commit run --all-files",
    "uv run pytest --ignore=tests/python/test_full_validation_artifacts.py",
    "uv run python scripts/check_release_gates.py",
    "uv run python scripts/check_public_api_contract.py",
    "uv run python scripts/check_artifact_compatibility.py",
    "uv run python scripts/check_docs_examples.py",
    "uv run python scripts/check_official_geo_evidence.py",
    "uv run python scripts/check_performance_thresholds.py",
    "uv run python scripts/run_autogeo_benchmark_gate.py",
    "uv run python scripts/run_full_validation.py",
    "uv run python scripts/run_v1_validation.py",
    "cargo bench --workspace --no-run",
}

REQUIRED_RELEASE_MARKERS = {
    'manylinux: "2014"',
    "macos-15",
    "windows-latest",
    "actions/attest-build-provenance",
    "id-token: write",
    "attestations: write",
    "twine check dist/*",
}


def main() -> int:
    checks = {
        "manifest_contract": check_manifest_contract(),
        "benchmark_dependencies": check_benchmark_dependencies(),
        "external_baseline_install_metadata": check_external_baseline_install_metadata(),
        "ci_release_gates": check_ci_release_gates(),
        "publish_artifact_attestation": check_publish_artifact_attestation(),
    }
    failed = [name for name, result in checks.items() if not result["passed"]]
    report = {"artifact_version": 1, "checks": checks, "passed": not failed}
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit("release gate checks failed: " + ", ".join(failed))
    return 0


def check_manifest_contract() -> dict[str, Any]:
    validate_configs()
    specs = load_all_tracks()
    validate_official_geo_benchmark_suite(specs)
    claim_tasks = [
        task
        for spec in specs
        for task in spec.tasks["tasks"]
        if task.get("claim_family") in OFFICIAL_GEO_FAMILIES
    ]
    return {
        "passed": len({task["claim_family"] for task in claim_tasks}) == len(OFFICIAL_GEO_FAMILIES),
        "claim_family_count": len({task["claim_family"] for task in claim_tasks}),
        "claim_task_count": len(claim_tasks),
    }


def check_benchmark_dependencies() -> dict[str, Any]:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    bench_block = _dependency_group_block(pyproject, "bench")
    declared = set(re.findall(r'"([^">=<!~\[]+)', bench_block))
    specs = load_all_tracks()
    required_families = {
        family
        for spec in specs
        for task in spec.tasks["tasks"]
        for family in task.get("required_model_families", [])
    }
    expected = set().union(
        *(
            BASELINE_DEPENDENCIES[family]
            for family in required_families & set(BASELINE_DEPENDENCIES)
        )
    )
    missing = sorted(expected - declared)
    return {
        "passed": not missing,
        "expected": sorted(expected),
        "missing": missing,
    }


def check_ci_release_gates() -> dict[str, Any]:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    missing = sorted(command for command in REQUIRED_CI_COMMANDS if command not in text)
    return {
        "passed": not missing,
        "missing": missing,
    }


def check_external_baseline_install_metadata() -> dict[str, Any]:
    specs = load_all_tracks()
    search_spaces = {
        item["model_family"]: item for spec in specs for item in spec.search_spaces["search_spaces"]
    }
    missing = [
        family
        for family in sorted(EXTERNAL_INSTALL_METADATA_BASELINES)
        if not search_spaces.get(family, {}).get("external_dependencies")
        or not search_spaces.get(family, {}).get("install_hint")
    ]
    return {
        "passed": not missing,
        "missing": missing,
    }


def check_publish_artifact_attestation() -> dict[str, Any]:
    text = PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    missing = sorted(marker for marker in REQUIRED_RELEASE_MARKERS if marker not in text)
    return {
        "passed": not missing,
        "missing": missing,
    }


def _dependency_group_block(pyproject: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^\[dependency-groups\].*?^{name} = \[(.*?)^\]",
        pyproject,
    )
    if match is None:
        raise ValueError(f"dependency group {name!r} not found")
    return match.group(1)


if __name__ == "__main__":
    raise SystemExit(main())
