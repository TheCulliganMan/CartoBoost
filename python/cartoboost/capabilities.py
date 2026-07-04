from __future__ import annotations

from typing import Any

REQUIRED_CAPABILITY_FIELDS = (
    "class_name",
    "architecture",
    "implementation_backend",
    "trainable_parameters",
    "uses_native_core",
    "benchmark_evidence",
    "experimental_status",
)

VALID_EXPERIMENTAL_STATUS = {
    "stable_native",
    "deterministic_python",
    "shallow_neural",
    "native_deep",
    "experimental",
}

CAPABILITY_TABLE: tuple[dict[str, Any], ...] = (
    {
        "class_name": "CartoBoostRegressor",
        "architecture": "gradient_boosted_oblivious_trees",
        "implementation_backend": "rust_native",
        "trainable_parameters": "tree_splits_leaf_values",
        "uses_native_core": True,
        "benchmark_evidence": "real-data evidence",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "CartoBoostClassifier",
        "architecture": "gradient_boosted_oblivious_trees_classifier",
        "implementation_backend": "rust_native",
        "trainable_parameters": "tree_splits_leaf_values",
        "uses_native_core": True,
        "benchmark_evidence": "real-data evidence",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "CartoBoostRanker",
        "architecture": "gradient_boosted_pairwise_ranker",
        "implementation_backend": "rust_native",
        "trainable_parameters": "tree_splits_leaf_values",
        "uses_native_core": True,
        "benchmark_evidence": "API contract only",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "AutoGeoModel",
        "architecture": "model_selection_stack",
        "implementation_backend": "python_orchestration_rust_models",
        "trainable_parameters": "selected_candidate_models",
        "uses_native_core": True,
        "benchmark_evidence": "real-data evidence",
        "experimental_status": "stable_native",
    },
    {
        "class_name": "EntityEmbedding",
        "architecture": "entity_embedding",
        "implementation_backend": "deterministic_python_numpy",
        "trainable_parameters": "deterministic_hash_projection",
        "uses_native_core": False,
        "benchmark_evidence": "API contract only",
        "experimental_status": "deterministic_python",
    },
    {
        "class_name": "PairEmbedding",
        "architecture": "pair_embedding",
        "implementation_backend": "deterministic_python_numpy",
        "trainable_parameters": "deterministic_hash_projection",
        "uses_native_core": False,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "deterministic_python",
    },
    {
        "class_name": "SpatioTemporalAdaptiveEmbedding",
        "architecture": "spatiotemporal_adaptive_embedding",
        "implementation_backend": "deterministic_python_numpy",
        "trainable_parameters": "deterministic_hash_projection",
        "uses_native_core": False,
        "benchmark_evidence": "API contract only",
        "experimental_status": "deterministic_python",
    },
    {
        "class_name": "HistoricalAnalogRetriever",
        "architecture": "exact_knn_memory",
        "implementation_backend": "deterministic_python_numpy",
        "trainable_parameters": "stored_normalized_memory",
        "uses_native_core": False,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "deterministic_python",
    },
    {
        "class_name": "MultiViewSpatialAttention",
        "architecture": "multi_view_spatial_attention",
        "implementation_backend": "deterministic_python_numpy",
        "trainable_parameters": "view_weights",
        "uses_native_core": False,
        "benchmark_evidence": "API contract only",
        "experimental_status": "shallow_neural",
    },
    {
        "class_name": "DirectionalPairForecaster",
        "architecture": "pair_embedding_mlp",
        "implementation_backend": "rust_native",
        "trainable_parameters": "pair_embeddings_mlp_weights",
        "uses_native_core": True,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "TemporalSSMForecaster",
        "architecture": "selective_ssm_lite",
        "implementation_backend": "python_numpy_decoder",
        "trainable_parameters": "horizon_specific_ridge_decoder",
        "uses_native_core": False,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "shallow_neural",
    },
    {
        "class_name": "SelectiveStateSpaceBlock",
        "architecture": "selective_ssm_lite_encoder",
        "implementation_backend": "python_numpy_recurrence",
        "trainable_parameters": "deterministic_projection_matrices",
        "uses_native_core": False,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "shallow_neural",
    },
    {
        "class_name": "InvertedTemporalTransformer",
        "architecture": "inverted_transformer",
        "implementation_backend": "python_numpy",
        "trainable_parameters": "entity_token_projection",
        "uses_native_core": False,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "shallow_neural",
    },
    {
        "class_name": "PropagationDelayGraphForecaster",
        "architecture": "delay_aware_graph_transformer",
        "implementation_backend": "rust_native_with_python_facade",
        "trainable_parameters": "graph_delay_weights",
        "uses_native_core": True,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "ConditionalFlowDistributionHead",
        "architecture": "conditional_residual_sampler",
        "implementation_backend": "rust_native",
        "trainable_parameters": "location_scale_ridge_weights",
        "uses_native_core": True,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "ChoiceSetTransformer",
        "architecture": "choice_set_utility_softmax",
        "implementation_backend": "rust_native",
        "trainable_parameters": "utility_softmax_temperature",
        "uses_native_core": True,
        "benchmark_evidence": "synthetic claim evidence",
        "experimental_status": "native_deep",
    },
    {
        "class_name": "GeoTemporalDiffusionScenarioModel",
        "architecture": "conditional_residual_diffusion",
        "implementation_backend": "rust_native",
        "trainable_parameters": "scenario_noise_graph_diffusion",
        "uses_native_core": True,
        "benchmark_evidence": "experimental only",
        "experimental_status": "experimental",
    },
    {
        "class_name": "GraphNeuralOperator",
        "architecture": "graph_neural_operator",
        "implementation_backend": "rust_native",
        "trainable_parameters": "field_smoothing_operator_weights",
        "uses_native_core": True,
        "benchmark_evidence": "experimental only",
        "experimental_status": "experimental",
    },
)


def capability_table() -> list[dict[str, Any]]:
    return [dict(row) for row in CAPABILITY_TABLE]


def validate_capability_table() -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for row in CAPABILITY_TABLE:
        class_name = str(row.get("class_name", ""))
        if class_name in seen:
            errors.append(f"duplicate capability row for {class_name}")
        seen.add(class_name)
        for field in REQUIRED_CAPABILITY_FIELDS:
            if field not in row:
                errors.append(f"{class_name} missing {field}")
        status = row.get("experimental_status")
        if status not in VALID_EXPERIMENTAL_STATUS:
            errors.append(f"{class_name} has invalid experimental_status {status!r}")
    return errors


__all__ = [
    "CAPABILITY_TABLE",
    "REQUIRED_CAPABILITY_FIELDS",
    "VALID_EXPERIMENTAL_STATUS",
    "capability_table",
    "validate_capability_table",
]
