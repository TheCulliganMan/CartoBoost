use cartoboost_representation::{
    EntityEmbedding, FuturePatchReconstruction, GraphEdgeDenoising, HistoricalAnalogRetriever,
    MaskedEntityTimeModeling, MaskedPairTimeModeling, MultiViewSpatialAttention, PairEmbedding,
    RegimeRouter, SelfSupervisedPretrainer, SpatialNeighborContrastiveLoss,
    SpatioTemporalAdaptiveEmbedding, TemporalOrderContrastiveLoss, GRAPH_EDGE_DENOISING,
    MASKED_ENTITY_TIME_MODELING, MASKED_PAIR_TIME_MODELING, REPRESENTATION_ARTIFACT_VERSION,
    SPATIAL_NEIGHBOR_CONTRASTIVE_LOSS, TEMPORAL_ORDER_CONTRASTIVE_LOSS,
};
use std::collections::BTreeMap;

#[test]
fn entity_embedding_roundtrips_with_unknown_hash_fallback() {
    let mut model = EntityEmbedding::new(4, 8, 3).unwrap();
    model
        .fit(["region_a", "region_b"], Some("2024-01-01".to_string()))
        .unwrap();

    let known = model.transform(["region_a"]).unwrap();
    let unknown = model.transform(["never_seen"]).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    model.save_json(file.path()).unwrap();
    let loaded = EntityEmbedding::load_json(file.path()).unwrap();

    assert_eq!(loaded.transform(["region_a"]).unwrap(), known);
    assert_eq!(loaded.transform(["never_seen"]).unwrap(), unknown);
    let artifact = loaded.artifact().unwrap();
    assert_eq!(artifact.model_class, "EntityEmbedding");
    assert_eq!(artifact.artifact_version, REPRESENTATION_ARTIFACT_VERSION);
    assert_eq!(artifact.id_maps["entity"]["__unknown__"], 0);
    assert!(artifact.save_load_parity_checked);
    assert_eq!(artifact.backend.selected, "cpu");
    assert!(artifact.backend.accelerator_ready["cuda"]);
    assert!(artifact.backend.accelerator_ready["rocm"]);
    assert!(artifact.backend.accelerator_ready["mlx"]);
}

#[test]
fn pair_embedding_preserves_direction_and_unseen_pair_fallback() {
    let mut model = PairEmbedding::new(3, 8, 12, 5).unwrap();
    model.fit(&["A", "B"], &["B", "A"], None).unwrap();

    let forward = model.transform(&["A"], &["B"]).unwrap();
    let reverse = model.transform(&["B"], &["A"]).unwrap();
    let unseen = model.transform(&["A"], &["Z"]).unwrap();

    assert_ne!(forward, reverse);
    assert_eq!(model.transform(&["A"], &["Z"]).unwrap(), unseen);
    let artifact = model.artifact().unwrap();
    assert_ne!(
        artifact.id_maps["pair"]["A\0B"],
        artifact.id_maps["pair"]["B\0A"]
    );
    assert!(artifact.save_load_parity_checked);
}

#[test]
fn adaptive_embedding_changes_with_time_and_context() {
    let mut model = SpatioTemporalAdaptiveEmbedding::new(4, 8, 9).unwrap();
    model.fit(["node_1"]).unwrap();

    let early = model
        .transform(&["node_1"], &[vec![0.0, 0.0]], Some(&[vec![1.0, 0.0]]))
        .unwrap();
    let late = model
        .transform(&["node_1"], &[vec![1.0, 1.0]], Some(&[vec![0.0, 1.0]]))
        .unwrap();

    assert_ne!(early, late);
    let file = tempfile::NamedTempFile::new().unwrap();
    model.save_json(file.path()).unwrap();
    let loaded = SpatioTemporalAdaptiveEmbedding::load_json(file.path()).unwrap();
    assert_eq!(
        loaded
            .transform(&["node_1"], &[vec![0.0, 0.0]], Some(&[vec![1.0, 0.0]]))
            .unwrap(),
        early
    );
}

#[test]
fn multi_view_attention_reports_weights_ablation_and_missing_views() {
    let mut views = BTreeMap::new();
    views.insert(
        "physical_distance".to_string(),
        vec![vec![1.0, 0.2], vec![0.8, 0.1], vec![0.1, 1.0]],
    );
    views.insert(
        "observed_flow".to_string(),
        vec![vec![0.0, 2.0], vec![0.2, 1.8], vec![1.5, 0.3]],
    );
    views.insert(
        "hub_centrality".to_string(),
        vec![vec![3.0], vec![1.0], vec![0.5]],
    );
    let mut model = MultiViewSpatialAttention::new(4, 23).unwrap();
    model.fit(&["pickup", "dropoff", "hub"], &views).unwrap();

    let output = model.transform(&views).unwrap();
    assert_eq!(output.embedding.len(), 3);
    assert_eq!(output.view_weights[0].len(), 3);
    for row in output.view_weights.iter() {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    let mut missing = views.clone();
    missing.remove("hub_centrality");
    let missing_output = model.transform(&missing).unwrap();
    assert_eq!(
        missing_output.missing_views,
        vec!["hub_centrality".to_string()]
    );
    let missing_idx = model
        .learned_view_weights()
        .keys()
        .position(|name| name == "hub_centrality")
        .unwrap();
    assert!(missing_output
        .view_weights
        .iter()
        .all(|row| row[missing_idx].abs() < 1e-12));

    let report = model.view_ablation_report(&views).unwrap();
    assert!(report.full_beats_best_single_view);
    assert_eq!(report.single_view_proxy_scores.len(), 3);
    assert_eq!(model.learned_view_weights().len(), 3);
    let artifact = model.artifact().unwrap();
    assert_eq!(artifact.model_class, "MultiViewSpatialAttention");
    assert_eq!(artifact.architecture, "multi_view_spatial_attention");
    assert_eq!(artifact.hash_bucket_config["view_count"], 3);
    assert!(artifact.feature_roles.contains_key("learned_view_weights"));
    assert!(artifact.save_load_parity_checked);
}

#[test]
fn regime_router_reports_entropy_usage_and_artifact_metadata() {
    let mut model = RegimeRouter::new(3, 4, 8, 11).unwrap();
    model
        .fit(
            &["entity_a", "entity_b", "entity_c", "entity_a"],
            Some(&[
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 1.0],
            ]),
            Some("2024-03-01".to_string()),
        )
        .unwrap();

    let route = model
        .route(
            &["entity_a", "entity_z"],
            Some(&[vec![0.5, 0.0], vec![0.0, 0.5]]),
        )
        .unwrap();

    assert_eq!(route.expert_weights.len(), 2);
    assert_eq!(route.expert_weights[0].len(), 3);
    for row in route.expert_weights.iter() {
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
    assert_eq!(route.router_entropy.len(), 2);
    assert!((model.expert_usage().values().sum::<f64>() - 1.0).abs() < 1e-12);
    let artifact = model.artifact().unwrap();
    assert_eq!(artifact.model_class, "RegimeRouter");
    assert_eq!(artifact.architecture, "regime_router");
    assert_eq!(artifact.hash_bucket_config["expert_count"], 3);
    assert_eq!(artifact.training_cutoff.as_deref(), Some("2024-03-01"));
    assert!(artifact
        .training_metrics
        .contains_key("mean_router_entropy"));
    assert!(artifact.save_load_parity_checked);
}

#[test]
fn historical_analog_retriever_returns_cutoff_safe_neighbors() {
    let mut retriever = HistoricalAnalogRetriever::new(true, 0).unwrap();
    retriever
        .fit(
            &["early_a", "early_b", "future_c"],
            &[vec![0.0, 0.0], vec![1.0, 1.0], vec![0.1, 0.0]],
            Some(&["2024-01-01", "2024-01-02", "2024-02-01"]),
            Some("2024-02-15".to_string()),
        )
        .unwrap();

    let unrestricted = retriever.query(&[vec![0.09, 0.0]], 2, None).unwrap();
    let cutoff = retriever
        .query(&[vec![0.09, 0.0]], 2, Some("2024-01-15"))
        .unwrap();

    assert_eq!(unrestricted[0].analog_ids[0], "future_c");
    assert!(!cutoff[0].analog_ids.contains(&"future_c".to_string()));
    assert_eq!(cutoff[0].analog_ids[0], "early_a");
    assert!(cutoff[0].distances[0] <= cutoff[0].distances[1]);
    let artifact = retriever.artifact().unwrap();
    assert_eq!(artifact.model_class, "HistoricalAnalogRetriever");
    assert_eq!(artifact.architecture, "exact_knn_memory");
    assert_eq!(artifact.hash_bucket_config["memory_size"], 3);
    assert_eq!(artifact.training_metrics["memory_size"], 3.0);
    assert!(artifact.save_load_parity_checked);
}

#[test]
fn self_supervised_pretrainer_respects_cutoff_and_emits_reusable_embeddings() {
    let tasks = vec![
        MASKED_ENTITY_TIME_MODELING.to_string(),
        MASKED_PAIR_TIME_MODELING.to_string(),
        GRAPH_EDGE_DENOISING.to_string(),
        TEMPORAL_ORDER_CONTRASTIVE_LOSS.to_string(),
        SPATIAL_NEIGHBOR_CONTRASTIVE_LOSS.to_string(),
        cartoboost_representation::FUTURE_PATCH_RECONSTRUCTION.to_string(),
    ];
    let mut model = SelfSupervisedPretrainer::new(3, 8, 17, tasks.clone()).unwrap();
    model
        .fit(
            &["entity_a", "entity_b", "entity_a", "future_entity"],
            &[
                vec![1.0, 0.0],
                vec![0.0, 1.0],
                vec![1.5, 0.0],
                vec![100.0, 100.0],
            ],
            Some(&["2024-01-01", "2024-01-02", "2024-01-03", "2024-03-01"]),
            "2024-02-01".to_string(),
        )
        .unwrap();

    let values = model.transform(&["entity_a", "future_entity"]).unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(model.pretrained_pair_embeddings().unwrap().len(), 2);
    assert_eq!(model.pretrained_node_embeddings().unwrap()[0].len(), 3);
    assert_eq!(model.pretrained_temporal_encoder().unwrap().len(), 1);
    let artifact = model.artifact().unwrap();
    assert_eq!(artifact.model_class, "SelfSupervisedPretrainer");
    assert_eq!(
        artifact.architecture,
        "deterministic_self_supervised_pretrainer"
    );
    assert_eq!(artifact.training_cutoff.as_deref(), Some("2024-02-01"));
    assert_eq!(artifact.training_metrics["pretraining_rows"], 3.0);
    assert!(artifact
        .training_metrics
        .contains_key("masked_pair_proxy_rmse"));
    assert!(artifact
        .training_metrics
        .contains_key("graph_edge_denoising_proxy_auc"));
    assert!(artifact
        .training_metrics
        .contains_key("temporal_order_contrastive_margin"));
    assert!(artifact
        .training_metrics
        .contains_key("spatial_neighbor_contrastive_margin"));
    assert!(artifact
        .training_metrics
        .contains_key("future_patch_reconstruction_proxy_rmse"));
    assert!(artifact.save_load_parity_checked);
    assert_eq!(model.tasks(), tasks.as_slice());
    assert!(!artifact.id_maps["entity"].contains_key("future_entity"));
    assert_eq!(
        std::any::type_name::<MaskedEntityTimeModeling>(),
        "cartoboost_representation::MaskedEntityTimeModeling"
    );
    assert_eq!(
        std::any::type_name::<MaskedPairTimeModeling>(),
        "cartoboost_representation::MaskedPairTimeModeling"
    );
    assert_eq!(
        std::any::type_name::<GraphEdgeDenoising>(),
        "cartoboost_representation::GraphEdgeDenoising"
    );
    assert_eq!(
        std::any::type_name::<TemporalOrderContrastiveLoss>(),
        "cartoboost_representation::TemporalOrderContrastiveLoss"
    );
    assert_eq!(
        std::any::type_name::<SpatialNeighborContrastiveLoss>(),
        "cartoboost_representation::SpatialNeighborContrastiveLoss"
    );
    assert_eq!(
        std::any::type_name::<FuturePatchReconstruction>(),
        "cartoboost_representation::FuturePatchReconstruction"
    );
}
