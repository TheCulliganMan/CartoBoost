use cartoboost_neural::{
    select_backend, GraphSageConfig, GraphSageEncoder, GraphSageModelArtifact, HeteroGraph,
    HeteroGraphSageConfig, HeteroGraphSageEncoder, HeteroTypedEdge, HinSageConfig, HinSageEncoder,
    HinSageGraph, HomogeneousGraph,
};
use tempfile::tempdir;

#[test]
fn fits_graphsage_homogeneous_graph_and_tracks_loss_curve() {
    let graph = HomogeneousGraph::from_directed_edges(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])
        .expect("graph should build");
    let features = vec![
        vec![0.1_f32, 0.0_f32],
        vec![0.2_f32, 0.4_f32],
        vec![0.3_f32, 0.2_f32],
        vec![0.6_f32, 0.9_f32],
        vec![1.2_f32, 0.5_f32],
    ];

    let config = GraphSageConfig {
        hidden_dims: vec![3],
        epochs: 6,
        learning_rate: 0.05,
        negative_samples: 2,
        seed: 777,
        add_self_loop: true,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };

    let mut model = GraphSageEncoder::new(config.clone(), 2).expect("encoder should build");
    let embedding = model.fit(&graph, &features).expect("fit should complete");
    let loss_curve = model.loss_curve();

    assert_eq!(embedding.node_count(), 5);
    assert_eq!(embedding.dim(), 3);
    assert_eq!(loss_curve.values().len(), config.epochs);
    assert!(loss_curve.values().iter().all(|loss| loss.is_finite()));
    assert!(embedding
        .vectors()
        .iter()
        .all(|row| row.iter().all(|value| value.is_finite())));

    let mut repeat_model = GraphSageEncoder::new(config, 2).expect("encoder should build");
    let repeat_embedding = repeat_model
        .fit(&graph, &features)
        .expect("second fit should complete");

    assert_eq!(embedding.vectors(), repeat_embedding.vectors());
}

#[test]
fn graphsage_encode_reuses_fit_topology_for_same_graph() {
    let graph = HomogeneousGraph::from_directed_edges(3, &[(0, 1), (0, 2), (1, 2)])
        .expect("graph should build");
    let features = vec![
        vec![1.0_f32, 0.0_f32],
        vec![0.0_f32, 1.0_f32],
        vec![1.0_f32, 1.0_f32],
    ];
    let config = GraphSageConfig {
        hidden_dims: vec![2],
        epochs: 2,
        learning_rate: 0.01,
        negative_samples: 1,
        seed: 91,
        add_self_loop: true,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };
    let mut model = GraphSageEncoder::new(config, 2).expect("encoder should build");

    let fitted = model.fit(&graph, &features).expect("fit should complete");
    let encoded = model
        .encode(&features)
        .expect("encode should reuse fitted topology");
    let encoded_with_graph = model
        .encode_graph(&graph, &features)
        .expect("graph encode should complete");

    assert_eq!(encoded.vectors(), fitted.vectors());
    assert_eq!(encoded_with_graph.vectors(), fitted.vectors());
}

#[test]
fn graphsage_dense_source_negative_sampling_completes() {
    let graph = HomogeneousGraph::from_directed_edges(3, &[(0, 0), (0, 1), (0, 2)])
        .expect("graph should build");
    let features = vec![
        vec![1.0_f32, 0.0_f32],
        vec![0.0_f32, 1.0_f32],
        vec![1.0_f32, 1.0_f32],
    ];
    let config = GraphSageConfig {
        hidden_dims: vec![2],
        epochs: 2,
        learning_rate: 0.01,
        negative_samples: 4,
        seed: 92,
        add_self_loop: false,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };
    let mut model = GraphSageEncoder::new(config, 2).expect("encoder should build");

    let embedding = model
        .fit(&graph, &features)
        .expect("dense-source fit should complete");

    assert_eq!(embedding.node_count(), 3);
    assert!(model
        .loss_curve()
        .values()
        .iter()
        .all(|loss| loss.is_finite()));
}

#[test]
fn invalid_graph_construction_is_rejected() {
    assert!(
        HomogeneousGraph::from_directed_edges(0, &[(0, 0)]).is_err(),
        "homogeneous graph requires positive node_count",
    );

    let invalid_edges = vec![HeteroTypedEdge {
        source: 0,
        target: 0,
        relation: 0,
    }];
    assert!(
        HeteroGraph::from_typed_edges(2, 0, &invalid_edges).is_err(),
        "heterogeneous graph requires positive relation_count",
    );
    assert!(
        HeteroGraph::from_typed_edges(
            2,
            1,
            &[HeteroTypedEdge {
                source: 0,
                target: 1,
                relation: 2,
            }],
        )
        .is_err(),
        "heterogeneous graph rejects out-of-range relation ids",
    );
}

#[test]
fn graphsage_without_layers_returns_raw_features() {
    let graph =
        HomogeneousGraph::from_directed_edges(3, &[(0, 1), (1, 2)]).expect("graph should build");
    let features = vec![
        vec![2.0_f32, -1.0],
        vec![0.0_f32, 3.0],
        vec![1.5_f32, 0.5_f32],
    ];

    let config = GraphSageConfig {
        hidden_dims: vec![],
        epochs: 3,
        learning_rate: 0.05,
        negative_samples: 1,
        seed: 123,
        add_self_loop: true,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };

    let mut model = GraphSageEncoder::new(config, 2).expect("encoder should build");
    let embedding = model.fit(&graph, &features).expect("fit should complete");
    assert_eq!(embedding.vectors(), &features);
}

#[test]
fn fits_hinsage_heterogeneous_graph_with_relation_conditioning() {
    let edges = vec![
        HeteroTypedEdge {
            source: 0,
            target: 1,
            relation: 0,
        },
        HeteroTypedEdge {
            source: 1,
            target: 2,
            relation: 1,
        },
        HeteroTypedEdge {
            source: 2,
            target: 3,
            relation: 0,
        },
        HeteroTypedEdge {
            source: 3,
            target: 0,
            relation: 1,
        },
    ];
    let graph = HeteroGraph::from_typed_edges(4, 2, &edges).expect("hetero graph should build");
    let features = vec![
        vec![1.0_f32, 0.5_f32],
        vec![0.0_f32, 2.0_f32],
        vec![3.0_f32, -1.0_f32],
        vec![2.5_f32, 0.7_f32],
    ];

    let config = HeteroGraphSageConfig {
        hidden_dims: vec![2],
        epochs: 4,
        learning_rate: 0.03,
        negative_samples: 2,
        seed: 44,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };

    let mut encoder =
        HeteroGraphSageEncoder::new(config.clone(), 2, 2).expect("hinsage should build");
    let embedding = encoder.fit(&graph, &features).expect("fit should complete");

    assert_eq!(embedding.node_count(), 4);
    assert_eq!(embedding.dim(), 2);
    assert!(embedding
        .vectors()
        .iter()
        .all(|row| row.iter().all(|value| value.is_finite())));

    let mut repeat = HeteroGraphSageEncoder::new(config, 2, 2).expect("hinsage should build");
    let repeat_embedding = repeat
        .fit(&graph, &features)
        .expect("second fit should complete");
    assert_eq!(embedding.vectors(), repeat_embedding.vectors());

    let curve = encoder.loss_curve();
    assert_eq!(curve.values().len(), 4);
    assert!(curve.values().iter().all(|value| value.is_finite()));

    assert_ne!(encoder.relation_count(), 0);
}

#[test]
fn hetero_graphsage_encode_reuses_fit_topology_for_same_graph() {
    let edges = vec![
        HeteroTypedEdge {
            source: 0,
            target: 1,
            relation: 0,
        },
        HeteroTypedEdge {
            source: 0,
            target: 2,
            relation: 1,
        },
        HeteroTypedEdge {
            source: 1,
            target: 2,
            relation: 0,
        },
    ];
    let graph = HeteroGraph::from_typed_edges(3, 2, &edges).expect("hetero graph should build");
    let features = vec![
        vec![1.0_f32, 0.0_f32],
        vec![0.0_f32, 1.0_f32],
        vec![1.0_f32, 1.0_f32],
    ];
    let config = HeteroGraphSageConfig {
        hidden_dims: vec![2],
        epochs: 2,
        learning_rate: 0.01,
        negative_samples: 1,
        seed: 93,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };
    let mut model = HeteroGraphSageEncoder::new(config, 2, 2).expect("encoder should build");

    let fitted = model.fit(&graph, &features).expect("fit should complete");
    let encoded = model
        .encode(&features)
        .expect("encode should reuse fitted topology");
    let encoded_with_graph = model
        .encode_graph(&graph, &features)
        .expect("graph encode should complete");

    assert_eq!(encoded.vectors(), fitted.vectors());
    assert_eq!(encoded_with_graph.vectors(), fitted.vectors());
}

#[test]
fn native_hinsage_validates_node_types_samples_neighbors_and_builds_link_embeddings() {
    let node_types = vec![0, 1, 2, 2, 2];
    let edge_type_triples = vec![(0, 0, 2), (1, 1, 2), (2, 2, 0)];
    let edges = vec![
        HeteroTypedEdge {
            source: 0,
            target: 2,
            relation: 0,
        },
        HeteroTypedEdge {
            source: 0,
            target: 3,
            relation: 0,
        },
        HeteroTypedEdge {
            source: 0,
            target: 4,
            relation: 0,
        },
        HeteroTypedEdge {
            source: 1,
            target: 2,
            relation: 1,
        },
        HeteroTypedEdge {
            source: 2,
            target: 0,
            relation: 2,
        },
    ];
    let graph = HinSageGraph::from_typed_schema(node_types, 3, 3, edge_type_triples.clone(), edges)
        .expect("typed graph should build");
    let features = vec![
        vec![1.0_f32, 0.0],
        vec![0.0_f32, 1.0],
        vec![0.5_f32, 0.2],
        vec![0.7_f32, 0.4],
        vec![0.9_f32, 0.6],
    ];
    let config = HinSageConfig {
        hidden_dims: vec![3],
        epochs: 3,
        learning_rate: 0.02,
        negative_samples: 1,
        seed: 99,
        l2_regularization: 0.0,
        neighbor_samples: vec![2, 0, 0],
        backend: select_backend(Some("cpu")).unwrap(),
    };

    let mut encoder =
        HinSageEncoder::new(config, 2, 3, edge_type_triples).expect("encoder should build");
    let embedding = encoder.fit(&graph, &features).expect("fit should complete");
    let link_features = encoder
        .link_embeddings(embedding.vectors(), &[(0, 2), (1, 2)])
        .expect("link embeddings should build");

    assert_eq!(embedding.node_count(), 5);
    assert_eq!(embedding.dim(), 3);
    assert_eq!(link_features.len(), 2);
    assert_eq!(link_features[0].len(), 12);
    assert_eq!(encoder.loss_curve().values().len(), 3);

    let artifact = encoder.to_artifact();
    assert!(matches!(artifact.model, GraphSageModelArtifact::HinSage(_)));
    let loaded = HinSageEncoder::from_artifact(artifact).expect("artifact should load");
    assert_eq!(loaded.node_type_count(), 3);
    assert_eq!(loaded.relation_count(), 3);
}

#[test]
fn native_hinsage_rejects_edges_that_violate_type_schema() {
    let bad = HinSageGraph::from_typed_schema(
        vec![0, 1],
        2,
        1,
        vec![(0, 0, 1)],
        vec![HeteroTypedEdge {
            source: 1,
            target: 0,
            relation: 0,
        }],
    );
    assert!(bad.is_err());
}

#[test]
fn validates_feature_shape_failures_on_fit_and_encode() {
    let graph =
        HomogeneousGraph::from_directed_edges(3, &[(0, 1), (1, 2)]).expect("graph should build");
    let config = GraphSageConfig {
        hidden_dims: vec![2],
        epochs: 1,
        learning_rate: 0.01,
        negative_samples: 1,
        seed: 123,
        add_self_loop: false,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };
    let mut encoder = GraphSageEncoder::new(config, 2).expect("encoder should build");

    let short_rows = vec![vec![1.0_f32], vec![2.0_f32]];
    assert!(encoder.fit(&graph, &short_rows).is_err());

    let wrong_count = vec![vec![1.0_f32, 1.5_f32], vec![2.0_f32, 2.5_f32]];
    assert!(encoder.fit(&graph, &wrong_count).is_err());

    let wrong_width = vec![vec![1.0_f32, 1.5_f32, 0.2_f32]];
    assert!(encoder.encode(&wrong_width).is_err());
}

#[test]
fn single_node_negative_sampler_case_is_stable() {
    let graph = HomogeneousGraph::from_directed_edges(1, &[]).expect("graph should build");
    let config = GraphSageConfig {
        hidden_dims: vec![2],
        epochs: 2,
        learning_rate: 0.05,
        negative_samples: 4,
        seed: 8,
        add_self_loop: true,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };
    let mut encoder = GraphSageEncoder::new(config, 2).expect("encoder should build");
    let embedding = encoder
        .fit(&graph, &[vec![1.0_f32, 2.0_f32]])
        .expect("single-node fit should not panic");

    assert_eq!(embedding.node_count(), 1);
    assert_eq!(embedding.dim(), 2);
    assert!(embedding.vectors()[0].iter().all(|value| value.is_finite()));
}

#[test]
fn graphsage_artifact_roundtrip_preserves_state() {
    let graph = HomogeneousGraph::from_directed_edges(4, &[(0, 1), (1, 2), (2, 3)]).unwrap();
    let features = vec![
        vec![1.0_f32, 0.0_f32],
        vec![0.2_f32, 1.1_f32],
        vec![0.5_f32, 0.5_f32],
        vec![1.0_f32, 1.0_f32],
    ];

    let config = GraphSageConfig {
        hidden_dims: vec![2],
        epochs: 2,
        learning_rate: 0.04,
        negative_samples: 1,
        seed: 33,
        add_self_loop: true,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };
    let mut model = GraphSageEncoder::new(config.clone(), 2).unwrap();
    let _fitted = model.fit(&graph, &features).unwrap();

    let artifact = model.to_artifact();
    assert!(matches!(
        artifact.model,
        GraphSageModelArtifact::Homogeneous(_)
    ));

    let reloaded = GraphSageEncoder::from_artifact(artifact.clone()).unwrap();
    assert_eq!(reloaded.to_artifact(), artifact);

    let dir = tempdir().unwrap();
    let path = dir.path().join("graphsage.json");
    model.save_artifact_json(&path).unwrap();
    let loaded = GraphSageEncoder::load_artifact_json(&path).unwrap();
    assert_eq!(loaded.to_artifact().model, model.to_artifact().model);
}

#[test]
fn hinsage_artifact_roundtrip_preserves_state() {
    let edges = vec![
        HeteroTypedEdge {
            source: 0,
            target: 1,
            relation: 0,
        },
        HeteroTypedEdge {
            source: 1,
            target: 2,
            relation: 1,
        },
        HeteroTypedEdge {
            source: 2,
            target: 0,
            relation: 0,
        },
    ];
    let graph = HeteroGraph::from_typed_edges(3, 2, &edges).unwrap();
    let features = vec![
        vec![0.2_f32, 1.1_f32],
        vec![0.4_f32, 0.3_f32],
        vec![0.6_f32, 0.8_f32],
    ];

    let config = HeteroGraphSageConfig {
        hidden_dims: vec![2],
        epochs: 2,
        learning_rate: 0.03,
        negative_samples: 1,
        seed: 12,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };

    let mut model = HeteroGraphSageEncoder::new(config, 2, 2).unwrap();
    let _fitted = model.fit(&graph, &features).unwrap();

    assert!(matches!(
        model.to_artifact().model,
        GraphSageModelArtifact::Hetero(_)
    ));

    let artifact = model.to_artifact();
    let reloaded = HeteroGraphSageEncoder::from_artifact(artifact).unwrap();
    assert_eq!(reloaded.to_artifact(), model.to_artifact());
}

#[test]
fn encoder_outputs_match_graphsage_loss_type_usage() {
    let graph = HomogeneousGraph::from_directed_edges(2, &[(0, 1)]).expect("graph should build");
    let features = vec![vec![1.0_f32, 2.0_f32], vec![3.0_f32, 4.0_f32]];

    let config = GraphSageConfig {
        hidden_dims: vec![2],
        epochs: 1,
        learning_rate: 0.02,
        negative_samples: 0,
        seed: 55,
        add_self_loop: false,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };

    let mut model = GraphSageEncoder::new(config, 2).expect("encoder should build");
    let _ = model.fit(&graph, &features).expect("fit should complete");

    let loss = model.loss_curve().final_loss();
    assert!(loss.is_some_and(|value| value.is_finite()));
    let output = model
        .encode(&features)
        .expect("encode should run after training");
    assert_eq!(output.dim(), 2);
    assert!(output
        .vectors()
        .iter()
        .all(|row| row.iter().all(|value| value.is_finite())));
}

#[cfg(all(
    feature = "metal",
    any(
        target_os = "macos",
        target_os = "ios",
        target_os = "tvos",
        target_os = "visionos"
    )
))]
#[test]
fn metal_graphsage_encode_graph_matches_cpu_backend() {
    if !cartoboost_neural::available_backends()
        .iter()
        .any(|backend| backend == "metal")
    {
        return;
    }
    let graph =
        HomogeneousGraph::from_directed_edges(4, &[(0, 1), (1, 2), (2, 3), (3, 0)]).unwrap();
    let features = vec![
        vec![0.1_f32, 0.2_f32, 0.3_f32],
        vec![0.4_f32, 0.5_f32, 0.6_f32],
        vec![0.7_f32, 0.8_f32, 0.9_f32],
        vec![1.0_f32, 1.1_f32, 1.2_f32],
    ];
    let cpu_config = GraphSageConfig {
        hidden_dims: vec![5, 3],
        epochs: 0,
        learning_rate: 0.01,
        negative_samples: 0,
        seed: 808,
        add_self_loop: true,
        l2_regularization: 0.0,
        backend: select_backend(Some("cpu")).unwrap(),
    };
    let mut metal_config = cpu_config.clone();
    metal_config.backend = select_backend(Some("metal")).unwrap();

    let cpu = GraphSageEncoder::new(cpu_config, 3).unwrap();
    let metal = GraphSageEncoder::new(metal_config, 3).unwrap();
    let cpu_embedding = cpu.encode_graph(&graph, &features).unwrap();
    let metal_embedding = metal.encode_graph(&graph, &features).unwrap();

    for (cpu_row, metal_row) in cpu_embedding
        .vectors()
        .iter()
        .zip(metal_embedding.vectors())
    {
        for (left, right) in cpu_row.iter().zip(metal_row) {
            assert!((left - right).abs() < 1.0e-5);
        }
    }
}
