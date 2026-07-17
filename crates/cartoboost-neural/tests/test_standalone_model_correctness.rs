use cartoboost_neural::{
    select_backend, ArtifactFallbackKind, GraphSageConfig, GraphSageLinkPredictor,
    GraphSageRegressor, HeteroGraphSageConfig, HeteroGraphSageLinkPredictor,
    HeteroGraphSageRegressor, HinSageConfig, HinSageLinkPredictor, HinSageRegressor,
    NeuralEmbeddingRegressor, Node2VecConfig, Node2VecLinkPredictor, Node2VecRegressor,
    StandaloneBoosterConfig,
};
use tempfile::tempdir;

fn booster_config() -> StandaloneBoosterConfig {
    StandaloneBoosterConfig {
        n_estimators: 12,
        learning_rate: 0.3,
        max_depth: 2,
        min_samples_leaf: 1,
        min_gain: 0.0,
        backend: "cpu".to_string(),
    }
}

fn node2vec_config(seed: u64) -> Node2VecConfig {
    Node2VecConfig {
        dim: 4,
        walk_length: 5,
        walks_per_node: 2,
        window_size: 2,
        epochs: 1,
        learning_rate: 0.025,
        min_learning_rate: 0.0001,
        negative_samples: 1,
        p: 1.0,
        q: 1.0,
        seed,
        l2_regularization: 0.0,
        normalize: true,
    }
}

fn graphsage_config(seed: u64) -> GraphSageConfig {
    GraphSageConfig {
        hidden_dims: vec![4],
        epochs: 2,
        learning_rate: 0.02,
        negative_samples: 1,
        seed,
        add_self_loop: true,
        l2_regularization: 1.0e-5,
        backend: select_backend(Some("cpu")).unwrap(),
    }
}

fn hetero_graphsage_config(seed: u64) -> HeteroGraphSageConfig {
    HeteroGraphSageConfig {
        hidden_dims: vec![4],
        epochs: 2,
        learning_rate: 0.02,
        negative_samples: 1,
        seed,
        l2_regularization: 1.0e-5,
        backend: select_backend(Some("cpu")).unwrap(),
    }
}

fn hinsage_config(seed: u64) -> HinSageConfig {
    HinSageConfig {
        hidden_dims: vec![4],
        epochs: 2,
        learning_rate: 0.02,
        negative_samples: 1,
        seed,
        l2_regularization: 1.0e-5,
        neighbor_samples: vec![2, 2],
        backend: select_backend(Some("cpu")).unwrap(),
    }
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-9,
            "expected {actual} to be within 1e-9 of {expected}"
        );
    }
}

struct TypedGraphFixture {
    node_features: Vec<Vec<f32>>,
    typed_edges: Vec<(usize, usize, usize)>,
    node_types: Vec<usize>,
    sources: Vec<usize>,
    destinations: Vec<usize>,
    target: Vec<f64>,
}

fn typed_graph_fixture() -> TypedGraphFixture {
    TypedGraphFixture {
        node_features: vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.25],
        ],
        typed_edges: vec![(0, 1, 0), (1, 2, 1), (2, 3, 0), (3, 0, 1)],
        node_types: vec![0, 1, 0, 1],
        sources: vec![0, 1, 2, 3],
        destinations: vec![1, 2, 3, 0],
        target: vec![1.2, 1.7, 2.1, 0.5],
    }
}

#[test]
fn neural_embedding_regressor_predicts_reloads_and_validates_dense_width() {
    let ids = vec![1, 2, 1, 3, 2, 3];
    let dense = vec![
        vec![0.1],
        vec![0.2],
        vec![0.1],
        vec![0.3],
        vec![0.2],
        vec![0.3],
    ];
    let target = vec![1.0, 2.0, 1.1, 3.0, 2.1, 3.2];
    let mut model = NeuralEmbeddingRegressor::new(
        3,
        ArtifactFallbackKind::GlobalMeanVector,
        Some(7),
        1.0,
        booster_config(),
    )
    .expect("model");

    model.fit(&ids, &target, Some(&dense)).expect("fit");
    let predictions = model.predict(&ids, Some(&dense)).expect("predict");

    assert_eq!(predictions.len(), target.len());
    assert!(predictions.iter().all(|value| value.is_finite()));
    assert!(model
        .predict(&ids, Some(&vec![vec![0.1, 0.2]; ids.len()]))
        .is_err());

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("neural_embedding.json");
    model.save_artifact_json(&path).expect("save");
    let restored = NeuralEmbeddingRegressor::load_artifact_json(&path).expect("load");
    let restored_predictions = restored
        .predict(&ids, Some(&dense))
        .expect("predict restored");

    assert_eq!(restored_predictions, predictions);
}

#[test]
fn node2vec_regressor_handles_node_and_pair_modes_with_reload() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
    let row_nodes = vec![0, 1, 2, 3];
    let row_targets = vec![1, 2, 3, 0];
    let dense = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let target = vec![1.0, 1.5, 2.0, 0.7];

    let mut pair_model =
        Node2VecRegressor::new(node2vec_config(11), booster_config()).expect("pair model");
    pair_model
        .fit(
            4,
            &edges,
            None,
            &row_nodes,
            Some(&row_targets),
            Some(&dense),
            &target,
        )
        .expect("fit pair");
    let pair_predictions = pair_model
        .predict(&row_nodes, Some(&row_targets), Some(&dense))
        .expect("predict pair");
    assert_eq!(pair_predictions.len(), target.len());
    assert!(pair_predictions.iter().all(|value| value.is_finite()));
    assert!(pair_model.predict(&row_nodes, None, Some(&dense)).is_err());

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("node2vec_pair.json");
    pair_model.save_artifact_json(&path).expect("save pair");
    let restored = Node2VecRegressor::load_artifact_json(&path).expect("load pair");
    assert_eq!(
        restored
            .predict(&row_nodes, Some(&row_targets), Some(&dense))
            .expect("predict restored"),
        pair_predictions
    );

    let mut node_model =
        Node2VecRegressor::new(node2vec_config(13), booster_config()).expect("node model");
    node_model
        .fit(4, &edges, None, &row_nodes, None, Some(&dense), &target)
        .expect("fit node");
    assert_eq!(
        node_model
            .predict(&row_nodes, None, Some(&dense))
            .expect("predict node")
            .len(),
        target.len()
    );
}

#[test]
fn graphsage_regressor_predicts_reloads_and_rejects_wrong_mode() {
    let node_features = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.25],
    ];
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    let row_nodes = vec![0, 1, 2, 3];
    let row_targets = vec![1, 2, 3, 0];
    let target = vec![1.2, 1.7, 2.1, 0.5];
    let mut model =
        GraphSageRegressor::new(graphsage_config(17), 2, booster_config()).expect("model");

    model
        .fit(
            &node_features,
            &edges,
            &row_nodes,
            Some(&row_targets),
            None,
            &target,
        )
        .expect("fit");
    let predictions = model
        .predict(&node_features, &row_nodes, Some(&row_targets), None)
        .expect("predict");

    assert_eq!(predictions.len(), target.len());
    assert!(predictions.iter().all(|value| value.is_finite()));
    assert!(model
        .predict(&node_features, &row_nodes, None, None)
        .is_err());

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("graphsage.json");
    model.save_artifact_json(&path).expect("save");
    let restored = GraphSageRegressor::load_artifact_json(&path).expect("load");
    let restored_predictions = restored
        .predict(&node_features, &row_nodes, Some(&row_targets), None)
        .expect("predict restored");
    assert_vec_close(&restored_predictions, &predictions);
}

#[test]
fn hetero_graphsage_regressor_predicts_reloads_and_rejects_wrong_mode() {
    let fixture = typed_graph_fixture();
    let mut model =
        HeteroGraphSageRegressor::new(hetero_graphsage_config(31), 2, 2, booster_config())
            .expect("model");

    model
        .fit(
            &fixture.node_features,
            &fixture.typed_edges,
            &fixture.sources,
            Some(&fixture.destinations),
            None,
            &fixture.target,
        )
        .expect("fit");
    let predictions = model
        .predict(
            &fixture.node_features,
            &fixture.sources,
            Some(&fixture.destinations),
            None,
        )
        .expect("predict");

    assert_eq!(predictions.len(), fixture.target.len());
    assert!(predictions.iter().all(|value| value.is_finite()));
    assert!(model
        .predict(&fixture.node_features, &fixture.sources, None, None)
        .is_err());

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("hetero_graphsage.json");
    model.save_artifact_json(&path).expect("save");
    let restored = HeteroGraphSageRegressor::load_artifact_json(&path).expect("load");
    let restored_predictions = restored
        .predict(
            &fixture.node_features,
            &fixture.sources,
            Some(&fixture.destinations),
            None,
        )
        .expect("predict restored");
    assert_vec_close(&restored_predictions, &predictions);
}

#[test]
fn hinsage_regressor_predicts_reloads_and_rejects_wrong_mode() {
    let fixture = typed_graph_fixture();
    let edge_type_triples = vec![(0, 0, 1), (1, 1, 0)];
    let mut model = HinSageRegressor::new(
        hinsage_config(37),
        2,
        2,
        edge_type_triples,
        booster_config(),
    )
    .expect("model");

    model
        .fit(
            &fixture.node_features,
            &fixture.node_types,
            &fixture.typed_edges,
            &fixture.sources,
            Some(&fixture.destinations),
            None,
            &fixture.target,
        )
        .expect("fit");
    let predictions = model
        .predict(
            &fixture.node_features,
            &fixture.sources,
            Some(&fixture.destinations),
            None,
        )
        .expect("predict");

    assert_eq!(predictions.len(), fixture.target.len());
    assert!(predictions.iter().all(|value| value.is_finite()));
    assert!(model
        .predict(&fixture.node_features, &fixture.sources, None, None)
        .is_err());

    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("hinsage.json");
    model.save_artifact_json(&path).expect("save");
    let restored = HinSageRegressor::load_artifact_json(&path).expect("load");
    let restored_predictions = restored
        .predict(
            &fixture.node_features,
            &fixture.sources,
            Some(&fixture.destinations),
            None,
        )
        .expect("predict restored");
    assert_vec_close(&restored_predictions, &predictions);
}

#[test]
fn link_predictors_score_candidate_pairs_and_roundtrip_artifacts() {
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    let pairs = vec![(0, 1), (0, 2), (1, 2), (1, 3)];

    let mut node2vec = Node2VecLinkPredictor::new(node2vec_config(23)).expect("node2vec link");
    node2vec.fit(4, &edges, None).expect("fit node2vec link");
    let node2vec_scores = node2vec.predict_scores(&pairs).expect("node2vec scores");
    assert_eq!(node2vec_scores.len(), pairs.len());
    assert!(node2vec_scores.iter().all(|value| value.is_finite()));

    let node_features = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.25],
    ];
    let mut graphsage =
        GraphSageLinkPredictor::new(graphsage_config(29), 2).expect("graphsage link");
    graphsage
        .fit(&node_features, &edges)
        .expect("fit graphsage");
    let graphsage_scores = graphsage
        .predict_scores(&node_features, &pairs)
        .expect("graphsage scores");
    assert_eq!(graphsage_scores.len(), pairs.len());
    assert!(graphsage_scores.iter().all(|value| value.is_finite()));

    let dir = tempdir().expect("tempdir");
    let node_path = dir.path().join("node2vec_link.json");
    node2vec
        .save_artifact_json(&node_path)
        .expect("save node2vec");
    let restored_node =
        Node2VecLinkPredictor::load_artifact_json(&node_path).expect("load node2vec");
    assert_eq!(
        restored_node
            .predict_scores(&pairs)
            .expect("restored node2vec scores"),
        node2vec_scores
    );

    let graph_path = dir.path().join("graphsage_link.json");
    graphsage
        .save_artifact_json(&graph_path)
        .expect("save graphsage");
    let restored_graph =
        GraphSageLinkPredictor::load_artifact_json(&graph_path).expect("load graphsage");
    let restored_graphsage_scores = restored_graph
        .predict_scores(&node_features, &pairs)
        .expect("restored graphsage scores");
    assert_vec_close(&restored_graphsage_scores, &graphsage_scores);
}

#[test]
fn typed_link_predictors_score_candidate_pairs_and_roundtrip_artifacts() {
    let fixture = typed_graph_fixture();
    let pairs = vec![(0, 1), (0, 2), (1, 2), (1, 3)];

    let mut hetero =
        HeteroGraphSageLinkPredictor::new(hetero_graphsage_config(41), 2, 2).expect("hetero link");
    hetero
        .fit(&fixture.node_features, &fixture.typed_edges)
        .expect("fit hetero");
    let hetero_scores = hetero
        .predict_scores(&fixture.node_features, &pairs)
        .expect("hetero scores");
    assert_eq!(hetero_scores.len(), pairs.len());
    assert!(hetero_scores.iter().all(|value| value.is_finite()));

    let edge_type_triples = vec![(0, 0, 1), (1, 1, 0)];
    let mut hinsage =
        HinSageLinkPredictor::new(hinsage_config(43), 2, 2, edge_type_triples).expect("hin link");
    hinsage
        .fit(
            &fixture.node_features,
            &fixture.node_types,
            &fixture.typed_edges,
        )
        .expect("fit hinsage");
    let hinsage_scores = hinsage
        .predict_scores(&fixture.node_features, &pairs)
        .expect("hinsage scores");
    assert_eq!(hinsage_scores.len(), pairs.len());
    assert!(hinsage_scores.iter().all(|value| value.is_finite()));

    let dir = tempdir().expect("tempdir");
    let hetero_path = dir.path().join("hetero_graphsage_link.json");
    hetero
        .save_artifact_json(&hetero_path)
        .expect("save hetero");
    let restored_hetero =
        HeteroGraphSageLinkPredictor::load_artifact_json(&hetero_path).expect("load hetero");
    let restored_hetero_scores = restored_hetero
        .predict_scores(&fixture.node_features, &pairs)
        .expect("restored hetero scores");
    assert_vec_close(&restored_hetero_scores, &hetero_scores);

    let hinsage_path = dir.path().join("hinsage_link.json");
    hinsage
        .save_artifact_json(&hinsage_path)
        .expect("save hinsage");
    let restored_hinsage =
        HinSageLinkPredictor::load_artifact_json(&hinsage_path).expect("load hinsage");
    let restored_hinsage_scores = restored_hinsage
        .predict_scores(&fixture.node_features, &pairs)
        .expect("restored hinsage scores");
    assert_vec_close(&restored_hinsage_scores, &hinsage_scores);
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
fn metal_graphsage_link_predictor_matches_cpu_backend() {
    if !cartoboost_neural::available_backends()
        .iter()
        .any(|backend| backend == "metal")
    {
        return;
    }

    let node_features = vec![
        vec![0.1_f32, 0.2],
        vec![0.4_f32, 0.5],
        vec![0.7_f32, 0.8],
        vec![1.0_f32, 1.1],
    ];
    let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0)];
    let pairs = vec![(0, 1), (0, 2), (1, 3), (3, 2)];
    let mut cpu_config = graphsage_config(808);
    cpu_config.backend = select_backend(Some("cpu")).unwrap();
    let mut metal_config = graphsage_config(808);
    metal_config.backend = select_backend(Some("metal")).unwrap();

    let mut cpu = GraphSageLinkPredictor::new(cpu_config, 2).unwrap();
    let mut metal = GraphSageLinkPredictor::new(metal_config, 2).unwrap();
    cpu.fit(&node_features, &edges).unwrap();
    metal.fit(&node_features, &edges).unwrap();

    let cpu_scores = cpu.predict_scores(&node_features, &pairs).unwrap();
    let metal_scores = metal.predict_scores(&node_features, &pairs).unwrap();
    for (left, right) in metal_scores.iter().zip(&cpu_scores) {
        assert!((left - right).abs() < 1.0e-5);
    }
}
