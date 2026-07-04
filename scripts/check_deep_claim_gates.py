from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "assets" / "deep_claim_benchmarks" / "results.json"
RESULTS_JSONL = ROOT / "docs" / "assets" / "deep_claim_benchmarks" / "results.jsonl"
RESULTS_MD = ROOT / "docs" / "assets" / "deep_claim_benchmarks" / "results.md"
SAVE_LOAD_TOLERANCE = 1.0e-9

REQUIRED_FIELDS = {
    "claim_id",
    "architecture",
    "capability_tier",
    "implementation_backend",
    "falsifier_baseline",
    "dataset_hash",
    "split_hash",
    "seed",
    "primary_metric",
    "improvement_threshold",
    "percent_improvement",
    "fit_seconds",
    "predict_seconds",
    "peak_memory_mb",
    "save_load_max_abs_diff",
    "leakage_policy",
    "experimental_status",
}

REQUIRED_CLAIMS = {
    "pair_embedding_mlp_nonlinear_directional_pair",
    "selective_ssm_lite_long_memory_panel",
    "inverted_transformer_cross_entity_panel",
    "delay_aware_graph_transformer_directional_delay",
    "regime_moe_mixed_regime_data",
    "retrieval_augmented_forecaster_rare_pattern",
    "conditional_flow_distribution_head_calibration_sharpness",
    "choice_set_utility_softmax_candidate_competition",
}


def load_payload() -> dict[str, Any]:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def main() -> int:
    payload = load_payload()
    rows = list(payload.get("rows", []))
    errors: list[str] = []
    if not RESULTS_JSONL.exists():
        errors.append("missing results.jsonl")
    if not RESULTS_MD.exists():
        errors.append("missing results.md")
    if RESULTS_JSONL.exists():
        jsonl_rows = [
            json.loads(line)
            for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(jsonl_rows) != len(rows):
            errors.append("results.jsonl row count does not match results.json")
    present = {str(row.get("claim_id", "")) for row in rows}
    for claim_id in sorted(REQUIRED_CLAIMS - present):
        errors.append(f"missing deep claim result: {claim_id}")
    for row in rows:
        claim_id = str(row.get("claim_id", "<missing>"))
        missing = REQUIRED_FIELDS - set(row)
        for field in sorted(missing):
            errors.append(f"{claim_id} missing {field}")
        if row.get("passed") is not True:
            errors.append(f"deep claim gate failed: {claim_id}")
        if float(row.get("percent_improvement", -1.0)) < float(
            row.get("improvement_threshold", 0.0)
        ):
            errors.append(f"{claim_id} below improvement threshold")
        if float(row.get("save_load_max_abs_diff", 0.0)) > SAVE_LOAD_TOLERANCE:
            errors.append(f"{claim_id} save/load drift exceeds tolerance")
        status = str(row.get("experimental_status", ""))
        if status == "experimental":
            errors.append(f"{claim_id} experimental class counted as primary evidence")
        if "temporal" in claim_id or "ssm" in claim_id or "inverted" in claim_id:
            policy = str(row.get("leakage_policy", "")).lower()
            if "random" in policy and "forbidden" not in policy:
                errors.append(f"{claim_id} temporal claim uses random split")
        if "graph" in claim_id:
            baseline = str(row.get("falsifier_baseline", "")).lower()
            if "reversed" not in baseline:
                errors.append(f"{claim_id} graph claim lacks reversed-edge falsifier")
            if "no_delay" not in baseline and "no-graph" not in baseline and "no_delay" not in row:
                errors.append(f"{claim_id} graph claim lacks no-delay/no-graph falsifier")
    if not payload.get("all_passed"):
        errors.append("all_passed is false")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
