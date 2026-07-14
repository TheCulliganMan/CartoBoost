from __future__ import annotations

import json
from typing import Any

REQUIRED_CAPABILITY_FIELDS = (
    "class_name",
    "architecture",
    "capability_tier",
    "implementation_backend",
    "trainable_parameters",
    "uses_native_core",
    "benchmark_evidence",
    "experimental_status",
)

VALID_EXPERIMENTAL_STATUS = {
    "stable_native",
    "supported",
    "deterministic_python",
    "shallow_neural",
    "native_deep",
    "experimental",
}

# Canonical v0.3 manifest fields.  The historical capability labels remain in
# each row for report consumers, while these fields provide one auditable
# stable/supported/experimental contract.
CANONICAL_MANIFEST_FIELDS = (
    "key",
    "model_key",
    "tier",
    "backend",
    "task",
    "artifact_version",
    "dependencies",
    "evidence_level",
    "stable",
)
VALID_TIERS = {"stable", "supported", "experimental"}
_STABLE_MODEL_KEYS = {
    "CartoBoostRegressor": "models.cartoboost_regressor",
    "CartoBoostClassifier": "models.cartoboost_classifier",
    "CartoBoostRanker": "models.cartoboost_ranker",
    "AutoForecaster": "forecasting.auto_forecaster",
    "CartoBoostLagForecaster": "forecasting.cartoboost_lag",
}


def _canonical_manifest_fields(row: dict[str, Any]) -> None:
    class_name = str(row.get("class_name", ""))
    legacy_status = str(row.get("experimental_status", ""))
    evidence = str(row.get("benchmark_evidence", ""))
    model_key = _STABLE_MODEL_KEYS.get(class_name)
    canonical = _rust_manifest_by_key().get(model_key) if model_key else None
    if canonical is not None:
        tier = str(canonical["tier"])
        backend = str(canonical["backend"])
        task = str(canonical["task"])
        artifact_version = int(canonical["artifact_version"])
        dependencies = list(canonical.get("dependencies", []))
        evidence_level = str(canonical["evidence_level"])
    else:
        tier = (
            "experimental"
            if legacy_status == "experimental" or evidence == "experimental only"
            else "supported"
        )
        backend = str(row.get("implementation_backend", "unknown"))
        task = "forecasting" if "forecast" in class_name.lower() else "generic"
        if "embedding" in class_name.lower() or "representation" in class_name.lower():
            task = "representation"
        elif "graph" in class_name.lower():
            task = "graph"
        elif "causal" in class_name.lower() or "did" in class_name.lower():
            task = "causal"
        elif "interval" in class_name.lower() or "conformal" in class_name.lower():
            task = "probabilistic"
        artifact_version = 2
        dependencies = []
        evidence_level = {
            "real-data evidence": "real_data",
            "synthetic claim evidence": "synthetic",
            "experimental only": "experimental_only",
            "incomplete evidence": "api_only",
            "API contract only": "api_only",
        }.get(evidence, "api_only")
    row.update(
        {
            "key": f"capabilities.{class_name}",
            "model_key": model_key,
            "tier": tier,
            "backend": backend,
            "task": task,
            "artifact_version": artifact_version,
            "dependencies": dependencies,
            "evidence_level": evidence_level,
            "stable": tier == "stable",
        }
    )


def _rust_manifest_by_key() -> dict[str, dict[str, Any]]:
    from .models import native_model_manifest

    return {
        str(row["key"]): dict(row)
        for row in native_model_manifest()
        if isinstance(row, dict) and row.get("key")
    }


def _rust_stable_model_keys() -> set[str]:
    return {key for key, row in _rust_manifest_by_key().items() if row.get("tier") == "stable"}


CAPABILITY_TABLE: tuple[dict[str, Any], ...] = (
    {
        "class_name": "CartoBoostRegressor",
        "architecture": "gradient_boosted_oblivious_trees",
        "capability_tier": "stable_native",
        "implementation_backend": "rust_native",
        "trainable_parameters": "tree_splits_leaf_values",
        "uses_native_core": True,
        "benchmark_evidence": "real-data evidence",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "CartoBoostClassifier",
        "architecture": "gradient_boosted_oblivious_trees_classifier",
        "capability_tier": "stable_native",
        "implementation_backend": "rust_native",
        "trainable_parameters": "tree_splits_leaf_values",
        "uses_native_core": True,
        "benchmark_evidence": "real-data evidence",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "CartoBoostRanker",
        "architecture": "gradient_boosted_pairwise_ranker",
        "capability_tier": "stable_native",
        "implementation_backend": "rust_native",
        "trainable_parameters": "tree_splits_leaf_values",
        "uses_native_core": True,
        "benchmark_evidence": "real-data evidence",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "AutoForecaster",
        "architecture": "native_auto_forecaster",
        "capability_tier": "stable_native",
        "implementation_backend": "rust_native",
        "trainable_parameters": "lagged_booster_candidates",
        "uses_native_core": True,
        "benchmark_evidence": "real-data evidence",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "CartoBoostLagForecaster",
        "architecture": "native_lag_forecaster",
        "capability_tier": "stable_native",
        "implementation_backend": "rust_native",
        "trainable_parameters": "lagged_booster_weights",
        "uses_native_core": True,
        "benchmark_evidence": "real-data evidence",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "DirectionalPairForecaster",
        "architecture": "pair_embedding_mlp",
        "capability_tier": "native_deep",
        "implementation_backend": "rust_native",
        "trainable_parameters": "pair_embeddings_mlp_weights",
        "uses_native_core": True,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "InvertedTemporalTransformer",
        "architecture": "inverted_transformer",
        "capability_tier": "shallow_neural",
        "implementation_backend": "python_numpy",
        "trainable_parameters": "entity_token_projection",
        "uses_native_core": False,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "shallow_neural",
    },
    {
        "class_name": "PropagationDelayGraphForecaster",
        "architecture": "delay_aware_graph_transformer",
        "capability_tier": "native_deep",
        "implementation_backend": "rust_native_with_python_facade",
        "trainable_parameters": "graph_delay_weights",
        "uses_native_core": True,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "ConditionalFlowDistributionHead",
        "architecture": "conditional_residual_sampler",
        "capability_tier": "native_deep",
        "implementation_backend": "rust_native",
        "trainable_parameters": "location_scale_ridge_weights",
        "uses_native_core": True,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "ChoiceSetTransformer",
        "architecture": "choice_set_utility_softmax",
        "capability_tier": "native_deep",
        "implementation_backend": "rust_native",
        "trainable_parameters": "utility_softmax_temperature",
        "uses_native_core": True,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "GeoTemporalDiffusionScenarioModel",
        "architecture": "conditional_residual_diffusion",
        "capability_tier": "experimental",
        "implementation_backend": "rust_native",
        "trainable_parameters": "scenario_noise_graph_diffusion",
        "uses_native_core": True,
        "benchmark_evidence": "experimental only",
        "experimental_status": "experimental",
    },
    {
        "class_name": "GraphNeuralOperator",
        "architecture": "graph_neural_operator",
        "capability_tier": "experimental",
        "implementation_backend": "rust_native",
        "trainable_parameters": "field_smoothing_operator_weights",
        "uses_native_core": True,
        "benchmark_evidence": "experimental only",
        "experimental_status": "experimental",
    },
    {
        "class_name": "RegimeMoEForecaster",
        "architecture": "regime_moe",
        "capability_tier": "shallow_neural",
        "implementation_backend": "python_numpy",
        "trainable_parameters": "expert_mixer_weights",
        "uses_native_core": False,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "shallow_neural",
    },
    {
        "class_name": "ResponseCurveModel",
        "architecture": "native_response_curve_utility",
        "capability_tier": "native_deep",
        "implementation_backend": "rust_native",
        "trainable_parameters": "utility_response_weights",
        "uses_native_core": True,
        "benchmark_evidence": "API contract only",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "EventOutcomeModel",
        "architecture": "native_event_outcome_head",
        "capability_tier": "native_deep",
        "implementation_backend": "rust_native",
        "trainable_parameters": "event_probability_weights",
        "uses_native_core": True,
        "benchmark_evidence": "API contract only",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "ServiceTimeResidualModel",
        "architecture": "native_service_residual_head",
        "capability_tier": "native_deep",
        "implementation_backend": "rust_native",
        "trainable_parameters": "residual_correction_weights",
        "uses_native_core": True,
        "benchmark_evidence": "API contract only",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "ConstrainedDecisionOptimizer",
        "architecture": "native_constrained_candidate_optimizer",
        "capability_tier": "native_deep",
        "implementation_backend": "rust_native",
        "trainable_parameters": "not_applicable_optimizer",
        "uses_native_core": True,
        "benchmark_evidence": "API contract only",
        "experimental_status": "native_deep",
    },
)

_ALIAS_CAPABILITIES: tuple[tuple[str, str], ...] = (
    ("TemporalEntityTransformer", "InvertedTemporalTransformer"),
    ("InvertedEntityTransformer", "InvertedTemporalTransformer"),
    ("SpatioTemporalGraphForecaster", "PropagationDelayGraphForecaster"),
    ("DelayAwareGraphTransformer", "PropagationDelayGraphForecaster"),
    ("DynamicAdjacencyTransformer", "PropagationDelayGraphForecaster"),
    ("JointHorizonFlowHead", "ConditionalFlowDistributionHead"),
    ("ResidualFlowCalibrator", "ConditionalFlowDistributionHead"),
    ("UtilityNet", "ChoiceSetTransformer"),
    ("NestedChoiceHead", "ChoiceSetTransformer"),
    ("CounterfactualCandidateScorer", "ChoiceSetTransformer"),
    ("GeoTemporalMixtureOfExperts", "RegimeMoEForecaster"),
    ("PairRegimeRouter", "RegimeMoEForecaster"),
    ("EntityRegimeRouter", "RegimeMoEForecaster"),
    ("FlowScenarioGenerator", "GeoTemporalDiffusionScenarioModel"),
    ("ConditionalResidualDiffusion", "GeoTemporalDiffusionScenarioModel"),
    ("FourierGeoOperator", "GraphNeuralOperator"),
    ("SpatioTemporalOperator", "GraphNeuralOperator"),
)

_UTILITY_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "class_name": "EntityPanelFrame",
        "architecture": "entity_panel_frame_schema",
        "capability_tier": "deterministic_python",
        "implementation_backend": "python_validation_schema",
        "trainable_parameters": "not_applicable",
        "uses_native_core": False,
        "benchmark_evidence": "API contract only",
        "experimental_status": "deterministic_python",
    },
    {
        "class_name": "DirectionalPairFrame",
        "architecture": "directional_pair_frame_schema",
        "capability_tier": "deterministic_python",
        "implementation_backend": "python_validation_schema",
        "trainable_parameters": "not_applicable",
        "uses_native_core": False,
        "benchmark_evidence": "API contract only",
        "experimental_status": "deterministic_python",
    },
    {
        "class_name": "GraphTemporalFrame",
        "architecture": "graph_temporal_frame_schema",
        "capability_tier": "deterministic_python",
        "implementation_backend": "python_validation_schema",
        "trainable_parameters": "not_applicable",
        "uses_native_core": False,
        "benchmark_evidence": "API contract only",
        "experimental_status": "deterministic_python",
    },
    {
        "class_name": "ResponseCurveFrame",
        "architecture": "response_curve_frame_schema",
        "capability_tier": "deterministic_python",
        "implementation_backend": "python_validation_schema",
        "trainable_parameters": "not_applicable",
        "uses_native_core": False,
        "benchmark_evidence": "API contract only",
        "experimental_status": "deterministic_python",
    },
)


def capability_table() -> list[dict[str, Any]]:
    rows = [dict(row) for row in CAPABILITY_TABLE]
    by_name = {str(row["class_name"]): row for row in rows}
    for alias, source in _ALIAS_CAPABILITIES:
        source_row = dict(by_name[source])
        source_row["class_name"] = alias
        source_row["alias_of"] = source
        rows.append(source_row)
    rows.extend(dict(row) for row in _UTILITY_CAPABILITIES)
    for row in rows:
        _canonical_manifest_fields(row)
    return rows


def validate_capability_table() -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for row in capability_table():
        class_name = str(row.get("class_name", ""))
        if class_name in seen:
            errors.append(f"duplicate capability row for {class_name}")
        seen.add(class_name)
        for field in REQUIRED_CAPABILITY_FIELDS:
            if field not in row:
                errors.append(f"{class_name} missing {field}")
        for field in CANONICAL_MANIFEST_FIELDS:
            if field not in row:
                errors.append(f"{class_name} missing canonical manifest field {field}")
        if row.get("tier") not in VALID_TIERS:
            errors.append(f"{class_name} has invalid canonical tier {row.get('tier')!r}")
        if bool(row.get("stable")) != (row.get("tier") == "stable"):
            errors.append(f"{class_name} stable flag must agree with canonical tier")
        status = row.get("experimental_status")
        if status not in VALID_EXPERIMENTAL_STATUS:
            errors.append(f"{class_name} has invalid experimental_status {status!r}")
        if row.get("capability_tier") != status:
            errors.append(f"{class_name} capability_tier must match experimental_status")
        evidence = str(row.get("benchmark_evidence", ""))
        if status == "stable_native" and evidence in {"", "API contract only", "experimental only"}:
            errors.append(f"{class_name} stable_native lacks benchmark evidence")
        if status == "experimental" and evidence != "experimental only":
            errors.append(f"{class_name} experimental class counted as primary evidence")
    stable_keys = {
        str(row["model_key"])
        for row in capability_table()
        if row.get("stable") and row.get("model_key")
    }
    expected_stable_keys = _rust_stable_model_keys()
    if stable_keys != expected_stable_keys:
        errors.append(
            "stable capability manifest does not match the stable model registry: "
            f"expected {sorted(expected_stable_keys)}, got {sorted(stable_keys)}"
        )
    try:
        from .models import native_model_manifest

        native_stable_keys = {
            str(row.get("key")) for row in native_model_manifest() if row.get("tier") == "stable"
        }
    except (ImportError, ValueError, json.JSONDecodeError) as exc:
        errors.append(f"Rust model manifest unavailable: {exc}")
    else:
        if native_stable_keys != expected_stable_keys:
            errors.append(
                "Rust model manifest disagrees with the Python capability contract: "
                f"expected {sorted(expected_stable_keys)}, got {sorted(native_stable_keys)}"
            )
    return errors


def stable_capability_manifest() -> list[dict[str, Any]]:
    """Return only rows corresponding to the v0.3 stable model contract."""

    return [row for row in capability_table() if row.get("stable") and row.get("model_key")]


__all__ = [
    "CAPABILITY_TABLE",
    "CANONICAL_MANIFEST_FIELDS",
    "REQUIRED_CAPABILITY_FIELDS",
    "VALID_EXPERIMENTAL_STATUS",
    "VALID_TIERS",
    "capability_table",
    "stable_capability_manifest",
    "validate_capability_table",
]
