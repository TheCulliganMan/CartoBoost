fn run_regression_request(request: BrowserRegressionRequest) -> Result<BrowserRegressionResponse> {
    if request.rows.len() < 4 {
        return Err(CartoBoostError::InvalidInput(
            "regression modeling requires at least four rows".to_string(),
        ));
    }
    if request.feature_names.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "regression modeling requires at least one feature".to_string(),
        ));
    }
    if !request.options.holdout_fraction.is_finite()
        || request.options.holdout_fraction <= 0.0
        || request.options.holdout_fraction >= 0.8
    {
        return Err(CartoBoostError::InvalidInput(
            "holdout_fraction must be finite and between 0 and 0.8".to_string(),
        ));
    }
    let feature_count = request.feature_names.len();
    let sparse_feature_count = request.sparse_feature_names.len();
    let mut features = Vec::with_capacity(request.rows.len());
    let mut sparse_rows = Vec::with_capacity(request.rows.len());
    let mut targets = Vec::with_capacity(request.rows.len());
    for row in request.rows {
        if row.features.len() != feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "feature row has {} columns but feature_names has {feature_count}",
                row.features.len()
            )));
        }
        if row.sparse_sets.len() != sparse_feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "sparse feature row has {} columns but sparse_feature_names has {sparse_feature_count}",
                row.sparse_sets.len()
            )));
        }
        if row.features.iter().any(|value| !value.is_finite()) || !row.target.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "regression features and targets must be finite".to_string(),
            ));
        }
        features.push(row.features);
        sparse_rows.push(row.sparse_sets);
        targets.push(row.target);
    }

    let requested_holdout =
        ((features.len() as f64) * request.options.holdout_fraction).round() as usize;
    let holdout_rows = requested_holdout.clamp(1, features.len().saturating_sub(2));
    let train_rows = features.len() - holdout_rows;
    let schema = regression_feature_schema(
        &request.feature_names,
        &request.sparse_feature_names,
        &request.options,
    )?;
    let train_x = Dataset::mixed(
        features[..train_rows].to_vec(),
        sparse_columns_from_rows(&sparse_rows[..train_rows], sparse_feature_count),
        Some(schema.clone()),
    )?;
    let holdout_x = Dataset::mixed(
        features[train_rows..].to_vec(),
        sparse_columns_from_rows(&sparse_rows[train_rows..], sparse_feature_count),
        Some(schema),
    )?;
    let train_y = &targets[..train_rows];
    let holdout_y = &targets[train_rows..];

    let model =
        Booster::new(regression_booster_config(&request.options)?).fit(&train_x, train_y, None)?;
    let predictions = model.try_predict(&holdout_x)?;
    let interval_predictions =
        regression_interval_predictions(&request.options, &train_x, train_y, &holdout_x)?;
    let metrics = regression_metrics(holdout_y, &predictions, train_rows, holdout_rows)?;
    let prediction_rows = predictions
        .iter()
        .zip(holdout_y.iter())
        .enumerate()
        .map(
            |(offset, (prediction, actual))| BrowserRegressionPrediction {
                row_index: train_rows + offset,
                actual: *actual,
                prediction: *prediction,
                lower_prediction: interval_predictions
                    .as_ref()
                    .map(|(lower, _)| lower[offset]),
                upper_prediction: interval_predictions
                    .as_ref()
                    .map(|(_, upper)| upper[offset]),
                residual: actual - prediction,
            },
        )
        .collect::<Vec<_>>();
    let feature_importance = feature_importance(
        &model.trees,
        &request.feature_names,
        &request.sparse_feature_names,
    );

    Ok(BrowserRegressionResponse {
        metadata: json!({
            "model": "cartoboost_regressor",
            "featureNames": request.feature_names,
            "sparseFeatureNames": request.sparse_feature_names,
            "trainingConfig": model.training_config,
            "splitterMode": request.options.splitter_mode.as_deref().unwrap_or("auto"),
            "loss": regression_loss_label(&request.options),
            "intervalLowerAlpha": request.options.interval_lower_alpha,
            "intervalUpperAlpha": request.options.interval_upper_alpha,
            "monotonicConstraints": request.options.monotonic_constraints,
            "treeCount": model.trees.len(),
        }),
        metrics,
        predictions: prediction_rows,
        feature_importance,
        model_visualization: request
            .options
            .include_model_visualization
            .unwrap_or(false)
            .then(|| {
                model_visualization(
                    &model.trees,
                    &request.feature_names,
                    &request.sparse_feature_names,
                )
            }),
    })
}

fn run_neural_request(request: BrowserNeuralRequest) -> Result<BrowserNeuralResponse> {
    if request.rows.len() < 4 {
        return Err(CartoBoostError::InvalidInput(
            "neural modeling requires at least four rows".to_string(),
        ));
    }
    if !request.options.holdout_fraction.is_finite()
        || request.options.holdout_fraction <= 0.0
        || request.options.holdout_fraction >= 0.8
    {
        return Err(CartoBoostError::InvalidInput(
            "holdout_fraction must be finite and between 0 and 0.8".to_string(),
        ));
    }
    let dense_width = request.dense_feature_names.len();
    let mut dense = Vec::with_capacity(request.rows.len());
    let mut targets = Vec::with_capacity(request.rows.len());
    for row in &request.rows {
        if row.dense.len() != dense_width {
            return Err(CartoBoostError::InvalidInput(format!(
                "neural dense row has {} columns but dense_feature_names has {dense_width}",
                row.dense.len()
            )));
        }
        if row.dense.iter().any(|value| !value.is_finite()) || !row.target.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "neural dense features and targets must be finite".to_string(),
            ));
        }
        dense.push(row.dense.clone());
        targets.push(row.target);
    }

    let requested_holdout =
        ((dense.len() as f64) * request.options.holdout_fraction).round() as usize;
    let holdout_rows = requested_holdout.clamp(1, dense.len().saturating_sub(2));
    let train_rows = dense.len() - holdout_rows;
    let pipeline = request.pipeline.trim().to_ascii_lowercase();
    let (predictions, feature_names, trees, metadata) = match pipeline.as_str() {
        "" | "embedding" | "embedding_table" | "neural_embedding" => {
            run_embedding_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "node2vec" | "node2vec_graph" | "graph_node2vec" => {
            run_node2vec_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "graphsage" | "graph_sage" | "graphsage_graph" => {
            run_graphsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "hetero_graphsage" | "heterographsage" | "typed_graphsage" => {
            run_hetero_graphsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        "hinsage" | "hin_sage" | "typed_hinsage" => {
            run_hinsage_neural_pipeline(&request, &dense, &targets, train_rows)?
        }
        other => {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported browser neural pipeline {other:?}"
            )));
        }
    };

    let holdout_y = &targets[train_rows..];
    let metrics = regression_metrics(holdout_y, &predictions, train_rows, holdout_rows)?;
    let prediction_rows = predictions
        .iter()
        .zip(holdout_y.iter())
        .enumerate()
        .map(
            |(offset, (prediction, actual))| BrowserRegressionPrediction {
                row_index: train_rows + offset,
                actual: *actual,
                prediction: *prediction,
                lower_prediction: None,
                upper_prediction: None,
                residual: actual - prediction,
            },
        )
        .collect::<Vec<_>>();
    let feature_importance = feature_importance(&trees, &feature_names, &[]);

    Ok(BrowserNeuralResponse {
        metadata: json!({
            "model": metadata["model"].as_str().unwrap_or("cartoboost_neural"),
            "pipeline": pipeline,
            "denseFeatureNames": request.dense_feature_names,
            "treeCount": trees.len(),
            "details": metadata,
        }),
        metrics,
        predictions: prediction_rows,
        feature_importance,
        model_visualization: request
            .options
            .include_model_visualization
            .unwrap_or(false)
            .then(|| model_visualization(&trees, &feature_names, &[])),
    })
}

fn run_embedding_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let ids = request
        .rows
        .iter()
        .map(|row| {
            row.id.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "embedding neural pipeline requires an id column".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut model = NeuralEmbeddingRegressor::new(
        request.options.embedding_dim.unwrap_or(8),
        ArtifactFallbackKind::GlobalMeanVector,
        request.options.random_state,
        request.options.support_prior_strength.unwrap_or(1.0),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &ids[..train_rows],
            &targets[..train_rows],
            Some(&dense[..train_rows]),
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(&ids[train_rows..], Some(&dense[train_rows..]))
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "embedding",
        artifact.dim,
        &request.dense_feature_names,
        None,
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "neural_embedding_regressor",
            "embeddingDim": artifact.dim,
            "embeddingRows": artifact.table.rows.len(),
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_node2vec_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let sources = request
        .rows
        .iter()
        .map(|row| {
            row.source.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "Node2Vec neural pipeline requires a source column".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let target_nodes = request
        .rows
        .iter()
        .map(|row| row.target_node)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "Node2Vec neural pipeline requires a target node column".to_string(),
            )
        })?;
    let edges = sources
        .iter()
        .zip(target_nodes.iter())
        .map(|(source, target)| (*source, *target))
        .collect::<Vec<_>>();
    let edge_weights = request
        .rows
        .iter()
        .map(|row| row.edge_weight.unwrap_or(1.0))
        .collect::<Vec<_>>();
    let node_count = edges
        .iter()
        .flat_map(|(source, target)| [*source, *target])
        .max()
        .map(|max_node| max_node + 1)
        .unwrap_or(0);
    let mut model = Node2VecRegressor::new(
        node2vec_config(&request.options),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            node_count,
            &edges,
            Some(&edge_weights),
            &sources[..train_rows],
            Some(&target_nodes[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &sources[train_rows..],
            Some(&target_nodes[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "node2vec",
        artifact.encoder.output_dim,
        &request.dense_feature_names,
        Some("target_node2vec"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "node2vec_regressor",
            "mode": artifact.mode,
            "embeddingDim": artifact.encoder.output_dim,
            "nodeCount": artifact.encoder.node_count,
            "edgeCount": edges.len(),
            "lossCurve": artifact.encoder.loss_curve,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_graphsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "GraphSAGE")?;
    let config = graph_sage_config(&request.options);
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let mut model = GraphSageRegressor::new(
        config,
        graph.input_dim,
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "graphsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_graphsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "graphsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.edges.len(),
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_hetero_graphsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "HeteroGraphSAGE")?;
    let config = hetero_graph_sage_config(&request.options);
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let relation_count = graph
        .typed_edges
        .iter()
        .map(|(_, _, relation)| *relation)
        .max()
        .map(|relation| relation + 1)
        .unwrap_or(1);
    let mut model = HeteroGraphSageRegressor::new(
        config,
        graph.input_dim,
        relation_count,
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.typed_edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "hetero_graphsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_hetero_graphsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "hetero_graphsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.typed_edges.len(),
            "relationCount": relation_count,
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

fn run_hinsage_neural_pipeline(
    request: &BrowserNeuralRequest,
    dense: &[Vec<f64>],
    targets: &[f64],
    train_rows: usize,
) -> Result<BrowserNeuralPipelineOutput> {
    let graph = browser_graph_inputs(request, "HinSAGE")?;
    let config = hin_sage_config(&request.options);
    let embedding_dim = graph_sage_dim(&config.hidden_dims);
    let node_type_count = graph
        .node_types
        .iter()
        .max()
        .map(|node_type| node_type + 1)
        .unwrap_or(1);
    let edge_type_triples = if request.edge_type_triples.is_empty() {
        vec![(0, 0, 0)]
    } else {
        request.edge_type_triples.clone()
    };
    let mut model = HinSageRegressor::new(
        config,
        graph.input_dim,
        node_type_count,
        edge_type_triples.clone(),
        standalone_booster_config(&request.options),
    )
    .map_err(neural_to_core)?;
    model
        .fit(
            &graph.node_features,
            &graph.node_types,
            &graph.typed_edges,
            &graph.sources[..train_rows],
            Some(&graph.targets[..train_rows]),
            Some(&dense[..train_rows]),
            &targets[..train_rows],
        )
        .map_err(neural_to_core)?;
    let predictions = model
        .predict(
            &graph.node_features,
            &graph.sources[train_rows..],
            Some(&graph.targets[train_rows..]),
            Some(&dense[train_rows..]),
        )
        .map_err(neural_to_core)?;
    let artifact = model.to_artifact().map_err(neural_to_core)?;
    let feature_names = embedding_feature_names(
        "hinsage",
        embedding_dim,
        &request.dense_feature_names,
        Some("target_hinsage"),
    );
    Ok((
        predictions,
        feature_names,
        artifact.model.trees,
        json!({
            "model": "hinsage_regressor",
            "mode": artifact.mode,
            "embeddingDim": embedding_dim,
            "nodeCount": graph.node_features.len(),
            "edgeCount": graph.typed_edges.len(),
            "nodeTypeCount": node_type_count,
            "edgeTypeTriples": edge_type_triples,
            "inputDim": graph.input_dim,
            "denseWidth": artifact.dense_width,
        }),
    ))
}

struct BrowserGraphInputs {
    node_features: Vec<Vec<f32>>,
    node_types: Vec<usize>,
    sources: Vec<usize>,
    targets: Vec<usize>,
    edges: Vec<(usize, usize)>,
    typed_edges: Vec<(usize, usize, usize)>,
    input_dim: usize,
}

fn browser_graph_inputs(
    request: &BrowserNeuralRequest,
    pipeline_name: &str,
) -> Result<BrowserGraphInputs> {
    if request.node_features.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} neural pipeline requires inferred node features"
        )));
    }
    let input_dim = request.node_features[0].len();
    if input_dim == 0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} neural pipeline requires at least one node feature"
        )));
    }
    for features in &request.node_features {
        if features.len() != input_dim || features.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "{pipeline_name} node features must be finite and rectangular"
            )));
        }
    }
    let sources = request
        .rows
        .iter()
        .map(|row| {
            row.source.ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "{pipeline_name} neural pipeline requires a source column"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let targets = request
        .rows
        .iter()
        .map(|row| row.target_node)
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "{pipeline_name} neural pipeline requires a target node column"
            ))
        })?;
    let node_count = request.node_features.len();
    if sources
        .iter()
        .chain(targets.iter())
        .any(|node| *node >= node_count)
    {
        return Err(CartoBoostError::InvalidInput(format!(
            "{pipeline_name} graph rows reference node ids outside node_features"
        )));
    }
    let edges = sources
        .iter()
        .zip(targets.iter())
        .map(|(source, target)| (*source, *target))
        .collect::<Vec<_>>();
    let typed_edges = request
        .rows
        .iter()
        .zip(edges.iter())
        .map(|(row, (source, target))| (*source, *target, row.edge_type.unwrap_or(0)))
        .collect::<Vec<_>>();
    let node_types = if request.node_types.is_empty() {
        vec![0; node_count]
    } else {
        if request.node_types.len() != node_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "{pipeline_name} node_types length must match node_features"
            )));
        }
        request.node_types.clone()
    };
    Ok(BrowserGraphInputs {
        node_features: request.node_features.clone(),
        node_types,
        sources,
        targets,
        edges,
        typed_edges,
        input_dim,
    })
}

fn embedding_feature_names(
    prefix: &str,
    dim: usize,
    dense_feature_names: &[String],
    secondary_prefix: Option<&str>,
) -> Vec<String> {
    let mut names = (0..dim)
        .map(|idx| format!("{prefix}_{idx}"))
        .collect::<Vec<_>>();
    if let Some(secondary_prefix) = secondary_prefix {
        names.extend((0..dim).map(|idx| format!("{secondary_prefix}_{idx}")));
    }
    names.extend(dense_feature_names.iter().cloned());
    names
}

fn standalone_booster_config(options: &BrowserNeuralOptions) -> StandaloneBoosterConfig {
    StandaloneBoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(80),
        learning_rate: options.learning_rate.unwrap_or(0.07),
        max_depth: options.max_depth.unwrap_or(4),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(2),
        min_gain: 0.0,
    }
}

fn node2vec_config(options: &BrowserNeuralOptions) -> Node2VecConfig {
    let mut config = Node2VecConfig::default();
    if let Some(dim) = options.embedding_dim {
        config.dim = dim;
    }
    if let Some(walk_length) = options.node2vec_walk_length {
        config.walk_length = walk_length;
    }
    if let Some(walks_per_node) = options.node2vec_walks_per_node {
        config.walks_per_node = walks_per_node;
    }
    if let Some(window_size) = options.node2vec_window_size {
        config.window_size = window_size;
    }
    if let Some(epochs) = options.node2vec_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.node2vec_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(p) = options.node2vec_p {
        config.p = p;
    }
    if let Some(q) = options.node2vec_q {
        config.q = q;
    }
    if let Some(seed) = options.node2vec_seed {
        config.seed = seed;
    }
    config
}

fn graph_sage_config(options: &BrowserNeuralOptions) -> GraphSageConfig {
    let mut config = GraphSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        ..GraphSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    config
}

fn hetero_graph_sage_config(options: &BrowserNeuralOptions) -> HeteroGraphSageConfig {
    let mut config = HeteroGraphSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        ..HeteroGraphSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    config
}

fn hin_sage_config(options: &BrowserNeuralOptions) -> HinSageConfig {
    let mut config = HinSageConfig {
        hidden_dims: vec![options.embedding_dim.unwrap_or(8)],
        ..HinSageConfig::default()
    };
    if let Some(epochs) = options.graph_sage_epochs {
        config.epochs = epochs;
    }
    if let Some(learning_rate) = options.graph_sage_learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(negative_samples) = options.graph_sage_negative_samples {
        config.negative_samples = negative_samples;
    }
    if let Some(seed) = options.graph_sage_seed.or(options.random_state) {
        config.seed = seed;
    }
    config
}

fn graph_sage_dim(hidden_dims: &[usize]) -> usize {
    hidden_dims.last().copied().unwrap_or(8)
}

fn neural_to_core(error: cartoboost_neural::NeuralError) -> CartoBoostError {
    CartoBoostError::InvalidInput(error.to_string())
}

fn sparse_columns_from_rows(
    sparse_rows: &[Vec<Vec<u64>>],
    sparse_feature_count: usize,
) -> Vec<SparseSetColumn> {
    (0..sparse_feature_count)
        .map(|feature_idx| {
            SparseSetColumn::new(
                sparse_rows
                    .iter()
                    .map(|row| row.get(feature_idx).cloned().unwrap_or_default())
                    .collect(),
            )
        })
        .collect()
}

fn regression_booster_config(options: &BrowserRegressionOptions) -> Result<BoosterConfig> {
    Ok(BoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(120),
        learning_rate: options.learning_rate.unwrap_or(0.06),
        max_depth: options.max_depth.unwrap_or(3),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(4),
        splitters: regression_splitters(options),
        loss: regression_loss_config(options)?,
        monotonic_constraints: options.monotonic_constraints.clone().unwrap_or_default(),
        ..Default::default()
    })
}

fn regression_interval_predictions(
    options: &BrowserRegressionOptions,
    train_x: &Dataset,
    train_y: &[f64],
    holdout_x: &Dataset,
) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
    let Some(lower_alpha) = options.interval_lower_alpha else {
        return Ok(None);
    };
    let Some(upper_alpha) = options.interval_upper_alpha else {
        return Ok(None);
    };
    if !lower_alpha.is_finite()
        || !upper_alpha.is_finite()
        || lower_alpha <= 0.0
        || upper_alpha >= 1.0
        || lower_alpha >= upper_alpha
    {
        return Err(CartoBoostError::InvalidInput(
            "interval alphas must be finite with 0 < lower < upper < 1".to_string(),
        ));
    }
    let lower_model = Booster::new(regression_booster_config_with_loss(
        options,
        LossConfig::Quantile(QuantileLossConfig { alpha: lower_alpha }),
    ))
    .fit(train_x, train_y, None)?;
    let upper_model = Booster::new(regression_booster_config_with_loss(
        options,
        LossConfig::Quantile(QuantileLossConfig { alpha: upper_alpha }),
    ))
    .fit(train_x, train_y, None)?;
    let lower = lower_model.try_predict(holdout_x)?;
    let upper = upper_model.try_predict(holdout_x)?;
    let (lower, upper): (Vec<_>, Vec<_>) = lower
        .into_iter()
        .zip(upper)
        .map(|(left, right)| {
            if left <= right {
                (left, right)
            } else {
                (right, left)
            }
        })
        .unzip();
    Ok(Some((lower, upper)))
}

fn regression_booster_config_with_loss(
    options: &BrowserRegressionOptions,
    loss: LossConfig,
) -> BoosterConfig {
    BoosterConfig {
        n_estimators: options.n_estimators.unwrap_or(120),
        learning_rate: options.learning_rate.unwrap_or(0.06),
        max_depth: options.max_depth.unwrap_or(3),
        min_samples_leaf: options.min_samples_leaf.unwrap_or(4),
        splitters: regression_splitters(options),
        loss,
        monotonic_constraints: options.monotonic_constraints.clone().unwrap_or_default(),
        ..Default::default()
    }
}

fn regression_loss_label(options: &BrowserRegressionOptions) -> String {
    options
        .loss
        .as_deref()
        .unwrap_or("l2")
        .trim()
        .to_ascii_lowercase()
}

fn regression_loss_config(options: &BrowserRegressionOptions) -> Result<LossConfig> {
    match regression_loss_label(options).as_str() {
        "" | "l2" | "squared_error" => Ok(LossConfig::L2),
        "l1" | "absolute_error" | "median" => Ok(LossConfig::L1),
        "huber" => Ok(LossConfig::Huber(HuberLossConfig {
            delta: options.huber_delta.unwrap_or(1.0),
        })),
        "log_l2" | "logl2" => Ok(LossConfig::LogL2(LogL2LossConfig {
            offset: options.log_offset.unwrap_or(1.0),
        })),
        "quantile" => Ok(LossConfig::Quantile(QuantileLossConfig {
            alpha: options.quantile_alpha.unwrap_or(0.5),
        })),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unsupported browser regression loss {other:?}"
        ))),
    }
}

fn regression_splitters(options: &BrowserRegressionOptions) -> Vec<SplitterKind> {
    match options
        .splitter_mode
        .as_deref()
        .unwrap_or("auto")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "axis" | "dense_axis" => vec![SplitterKind::Axis],
        "spatial" => vec![
            SplitterKind::Axis,
            SplitterKind::Diagonal2D,
            SplitterKind::Gaussian2D,
        ],
        "periodic" => vec![
            SplitterKind::Axis,
            SplitterKind::Periodic {
                period: default_periodic_period(options),
            },
        ],
        "sparse" | "sparse_set" | "sparse_sets" => {
            vec![SplitterKind::Axis, SplitterKind::SparseSet]
        }
        "full" | "toolkit" | "spatial_periodic" => vec![
            SplitterKind::Axis,
            SplitterKind::Diagonal2D,
            SplitterKind::Gaussian2D,
            SplitterKind::Periodic {
                period: default_periodic_period(options),
            },
            SplitterKind::SparseSet,
        ],
        _ => vec![SplitterKind::Auto],
    }
}

fn default_periodic_period(options: &BrowserRegressionOptions) -> f64 {
    options
        .periodic_periods
        .values()
        .next()
        .copied()
        .unwrap_or(24) as f64
}

fn regression_feature_schema(
    feature_names: &[String],
    sparse_feature_names: &[String],
    options: &BrowserRegressionOptions,
) -> Result<FeatureSchema> {
    let mut kinds = feature_names
        .iter()
        .map(|name| {
            let kind = options
                .feature_kinds
                .get(name)
                .map(|value| value.trim().to_ascii_lowercase())
                .unwrap_or_else(|| "numeric".to_string());
            match kind.as_str() {
                "" | "numeric" => Ok(FeatureKind::Numeric),
                "spatial" => Ok(FeatureKind::Spatial),
                "periodic" => {
                    let period = options.periodic_periods.get(name).copied().unwrap_or(24);
                    if period == 0 {
                        return Err(CartoBoostError::InvalidInput(format!(
                            "periodic feature {name:?} must have a positive period"
                        )));
                    }
                    Ok(FeatureKind::Periodic { period })
                }
                other => Err(CartoBoostError::InvalidInput(format!(
                    "unsupported browser regression feature kind {other:?} for {name:?}"
                ))),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    kinds.extend(
        sparse_feature_names
            .iter()
            .map(|_| FeatureKind::SparseSet)
            .collect::<Vec<_>>(),
    );
    let mut names = feature_names.to_vec();
    names.extend(sparse_feature_names.iter().cloned());
    Ok(FeatureSchema { names, kinds })
}

fn regression_metrics(
    actuals: &[f64],
    predictions: &[f64],
    train_rows: usize,
    holdout_rows: usize,
) -> Result<BrowserRegressionMetrics> {
    if actuals.len() != predictions.len() || actuals.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "actual and prediction lengths must match and be non-empty".to_string(),
        ));
    }
    let mut squared_error_sum = 0.0;
    let mut absolute_error_sum = 0.0;
    let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
    let mut total_sum_squares = 0.0;
    for (actual, prediction) in actuals.iter().zip(predictions.iter()) {
        let residual = actual - prediction;
        squared_error_sum += residual * residual;
        absolute_error_sum += residual.abs();
        total_sum_squares += (actual - mean_actual).powi(2);
    }
    let rmse = (squared_error_sum / actuals.len() as f64).sqrt();
    let mae = absolute_error_sum / actuals.len() as f64;
    let r2 = if total_sum_squares <= f64::EPSILON {
        0.0
    } else {
        1.0 - squared_error_sum / total_sum_squares
    };
    Ok(BrowserRegressionMetrics {
        rmse,
        mae,
        r2,
        train_rows,
        holdout_rows,
    })
}

fn feature_importance(
    trees: &[cartoboost_core::Tree],
    feature_names: &[String],
    sparse_feature_names: &[String],
) -> Vec<BrowserFeatureImportance> {
    let dense_feature_count = feature_names.len();
    let mut names = feature_names.to_vec();
    names.extend(sparse_feature_names.iter().cloned());
    let mut counts = vec![0usize; names.len()];
    for tree in trees {
        count_split_features(&tree.root, &mut counts, dense_feature_count);
    }
    let mut importance = names
        .iter()
        .enumerate()
        .map(|(idx, feature)| BrowserFeatureImportance {
            feature: feature.clone(),
            split_count: counts[idx],
        })
        .collect::<Vec<_>>();
    importance.sort_by(|left, right| {
        right
            .split_count
            .cmp(&left.split_count)
            .then_with(|| left.feature.cmp(&right.feature))
    });
    importance
}

fn count_split_features(node: &Node, counts: &mut [usize], dense_feature_count: usize) {
    if let Node::Branch {
        split, left, right, ..
    } = node
    {
        count_split(split, counts, dense_feature_count);
        count_split_features(left, counts, dense_feature_count);
        count_split_features(right, counts, dense_feature_count);
    }
}

fn count_split(split: &Split, counts: &mut [usize], dense_feature_count: usize) {
    match split {
        Split::Axis { feature, .. }
        | Split::PeriodicInterval { feature, .. }
        | Split::SparseSetContainsAny { feature, .. } => increment_feature(*feature, counts),
        Split::Diagonal2D {
            x_feature,
            y_feature,
            ..
        }
        | Split::Gaussian2D {
            x_feature,
            y_feature,
            ..
        } => {
            increment_feature(*x_feature, counts);
            increment_feature(*y_feature, counts);
        }
        Split::SparseListContainsAny { sparse_feature, .. } => {
            increment_feature(dense_feature_count + *sparse_feature, counts);
        }
        Split::Fuzzy { base, .. } => count_split(base, counts, dense_feature_count),
    }
}

fn increment_feature(feature: usize, counts: &mut [usize]) {
    if let Some(count) = counts.get_mut(feature) {
        *count += 1;
    }
}

fn model_visualization(
    trees: &[cartoboost_core::Tree],
    feature_names: &[String],
    sparse_feature_names: &[String],
) -> BrowserModelVisualization {
    let mut names = feature_names.to_vec();
    names.extend(sparse_feature_names.iter().cloned());
    let mut totals = TreeStats::default();
    let mut split_kind_counts = BTreeMap::<String, usize>::new();
    let mut splitter_rules = BTreeMap::<(String, String), SplitterRuleAccumulator>::new();
    let mut feature_split_counts = BTreeMap::<(String, String), SplitterRuleAccumulator>::new();
    let mut depth_counts = BTreeMap::<usize, usize>::new();
    for tree in trees {
        let mut context = TreeStatsContext {
            dense_feature_count: feature_names.len(),
            feature_names: &names,
            stats: &mut totals,
            split_kind_counts: &mut split_kind_counts,
            splitter_rules: &mut splitter_rules,
            feature_split_counts: &mut feature_split_counts,
            depth_counts: &mut depth_counts,
        };
        collect_tree_stats(&tree.root, 0, &mut context);
    }
    let tree_blueprints = trees
        .iter()
        .take(8)
        .enumerate()
        .map(|(tree_index, tree)| tree_blueprint(tree_index, tree, feature_names.len(), &names))
        .collect();
    BrowserModelVisualization {
        summary: BrowserModelVisualizationSummary {
            tree_count: trees.len(),
            node_count: totals.node_count,
            branch_count: totals.branch_count,
            leaf_count: totals.leaf_count,
            max_depth: totals.max_depth,
            mean_leaf_value: finite_ratio(totals.leaf_value_sum, totals.leaf_count),
            mean_gain: finite_ratio(totals.gain_sum, totals.branch_count),
        },
        split_kinds: split_kind_counts
            .into_iter()
            .map(|(kind, count)| BrowserSplitKindCount { kind, count })
            .collect(),
        splitter_rules: top_splitter_rules(splitter_rules),
        feature_split_counts: top_feature_split_counts(feature_split_counts),
        depth_histogram: depth_counts
            .into_iter()
            .map(|(depth, count)| BrowserDepthCount { depth, count })
            .collect(),
        tree_blueprints,
    }
}

fn tree_blueprint(
    tree_index: usize,
    tree: &cartoboost_core::Tree,
    dense_feature_count: usize,
    feature_names: &[String],
) -> BrowserTreeBlueprint {
    let mut stats = TreeStats::default();
    let mut split_kind_counts = BTreeMap::<String, usize>::new();
    let mut splitter_rules = BTreeMap::<(String, String), SplitterRuleAccumulator>::new();
    let mut feature_split_counts = BTreeMap::<(String, String), SplitterRuleAccumulator>::new();
    let mut depth_counts = BTreeMap::<usize, usize>::new();
    let mut context = TreeStatsContext {
        dense_feature_count,
        feature_names,
        stats: &mut stats,
        split_kind_counts: &mut split_kind_counts,
        splitter_rules: &mut splitter_rules,
        feature_split_counts: &mut feature_split_counts,
        depth_counts: &mut depth_counts,
    };
    collect_tree_stats(&tree.root, 0, &mut context);
    let mut next_id = 0;
    BrowserTreeBlueprint {
        tree_index,
        node_count: stats.node_count,
        branch_count: stats.branch_count,
        leaf_count: stats.leaf_count,
        max_depth: stats.max_depth,
        total_gain: stats.gain_sum,
        root: tree_node_blueprint(
            &tree.root,
            0,
            dense_feature_count,
            feature_names,
            &mut next_id,
        ),
    }
}

#[derive(Default)]
struct TreeStats {
    node_count: usize,
    branch_count: usize,
    leaf_count: usize,
    max_depth: usize,
    gain_sum: f64,
    leaf_value_sum: f64,
}

#[derive(Default)]
struct SplitterRuleAccumulator {
    count: usize,
    total_gain: f64,
}

struct TreeStatsContext<'a> {
    dense_feature_count: usize,
    feature_names: &'a [String],
    stats: &'a mut TreeStats,
    split_kind_counts: &'a mut BTreeMap<String, usize>,
    splitter_rules: &'a mut BTreeMap<(String, String), SplitterRuleAccumulator>,
    feature_split_counts: &'a mut BTreeMap<(String, String), SplitterRuleAccumulator>,
    depth_counts: &'a mut BTreeMap<usize, usize>,
}

fn collect_tree_stats(node: &Node, depth: usize, context: &mut TreeStatsContext<'_>) {
    context.stats.node_count += 1;
    context.stats.max_depth = context.stats.max_depth.max(depth);
    *context.depth_counts.entry(depth).or_insert(0) += 1;
    match node {
        Node::Leaf { value, .. } => {
            context.stats.leaf_count += 1;
            context.stats.leaf_value_sum += *value;
        }
        Node::LinearLeaf { model, .. } => {
            context.stats.leaf_count += 1;
            context.stats.leaf_value_sum += model.intercept;
        }
        Node::Branch {
            split,
            left,
            right,
            gain,
            ..
        } => {
            context.stats.branch_count += 1;
            context.stats.gain_sum += *gain;
            let (kind, label) =
                split_display(split, context.dense_feature_count, context.feature_names);
            *context.split_kind_counts.entry(kind.clone()).or_insert(0) += 1;
            let rule = context
                .splitter_rules
                .entry((kind.clone(), label))
                .or_default();
            rule.count += 1;
            rule.total_gain += *gain;
            for feature in split_feature_indices(split, context.dense_feature_count) {
                let feature_name = feature_label(feature, context.feature_names);
                let feature_rule = context
                    .feature_split_counts
                    .entry((feature_name, kind.clone()))
                    .or_default();
                feature_rule.count += 1;
                feature_rule.total_gain += *gain;
            }
            collect_tree_stats(left, depth + 1, context);
            collect_tree_stats(right, depth + 1, context);
        }
    }
}

fn top_splitter_rules(
    splitter_rules: BTreeMap<(String, String), SplitterRuleAccumulator>,
) -> Vec<BrowserSplitterRuleSummary> {
    let mut rules = splitter_rules
        .into_iter()
        .map(|((kind, label), accumulator)| BrowserSplitterRuleSummary {
            kind,
            label,
            count: accumulator.count,
            total_gain: accumulator.total_gain,
            mean_gain: finite_ratio(accumulator.total_gain, accumulator.count),
        })
        .collect::<Vec<_>>();
    rules.sort_by(|left, right| {
        right
            .total_gain
            .total_cmp(&left.total_gain)
            .then_with(|| right.count.cmp(&left.count))
            .then_with(|| left.label.cmp(&right.label))
    });
    rules.truncate(16);
    rules
}

fn top_feature_split_counts(
    feature_split_counts: BTreeMap<(String, String), SplitterRuleAccumulator>,
) -> Vec<BrowserFeatureSplitCount> {
    let mut rows = feature_split_counts
        .into_iter()
        .map(|((feature, kind), accumulator)| BrowserFeatureSplitCount {
            feature,
            kind,
            count: accumulator.count,
            total_gain: accumulator.total_gain,
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .total_gain
            .total_cmp(&left.total_gain)
            .then_with(|| right.count.cmp(&left.count))
            .then_with(|| left.feature.cmp(&right.feature))
            .then_with(|| left.kind.cmp(&right.kind))
    });
    rows.truncate(24);
    rows
}

fn tree_node_blueprint(
    node: &Node,
    depth: usize,
    dense_feature_count: usize,
    feature_names: &[String],
    next_id: &mut usize,
) -> BrowserTreeNode {
    let id = *next_id;
    *next_id += 1;
    match node {
        Node::Leaf {
            value,
            sample_weight_sum,
            ..
        } => BrowserTreeNode {
            id,
            depth,
            kind: "leaf".to_string(),
            label: format!("leaf {value:.3}"),
            value: Some(*value),
            gain: None,
            sample_weight_sum: Some(*sample_weight_sum),
            left: None,
            right: None,
        },
        Node::LinearLeaf {
            model,
            sample_weight_sum,
            ..
        } => BrowserTreeNode {
            id,
            depth,
            kind: "linear_leaf".to_string(),
            label: format!("linear leaf {:+.3}", model.intercept),
            value: Some(model.intercept),
            gain: None,
            sample_weight_sum: Some(*sample_weight_sum),
            left: None,
            right: None,
        },
        Node::Branch {
            split,
            left,
            right,
            gain,
            sample_weight_sum,
        } => {
            let (kind, label) = split_display(split, dense_feature_count, feature_names);
            let should_expand = depth < 3;
            BrowserTreeNode {
                id,
                depth,
                kind,
                label,
                value: None,
                gain: Some(*gain),
                sample_weight_sum: Some(*sample_weight_sum),
                left: should_expand.then(|| {
                    Box::new(tree_node_blueprint(
                        left,
                        depth + 1,
                        dense_feature_count,
                        feature_names,
                        next_id,
                    ))
                }),
                right: should_expand.then(|| {
                    Box::new(tree_node_blueprint(
                        right,
                        depth + 1,
                        dense_feature_count,
                        feature_names,
                        next_id,
                    ))
                }),
            }
        }
    }
}

fn split_feature_indices(split: &Split, dense_feature_count: usize) -> Vec<usize> {
    let mut features = BTreeSet::new();
    collect_split_feature_indices(split, dense_feature_count, &mut features);
    features.into_iter().collect()
}

fn collect_split_feature_indices(
    split: &Split,
    dense_feature_count: usize,
    features: &mut BTreeSet<usize>,
) {
    match split {
        Split::Axis { feature, .. }
        | Split::PeriodicInterval { feature, .. }
        | Split::SparseSetContainsAny { feature, .. } => {
            features.insert(*feature);
        }
        Split::Diagonal2D {
            x_feature,
            y_feature,
            ..
        }
        | Split::Gaussian2D {
            x_feature,
            y_feature,
            ..
        } => {
            features.insert(*x_feature);
            features.insert(*y_feature);
        }
        Split::SparseListContainsAny { sparse_feature, .. } => {
            features.insert(dense_feature_count + *sparse_feature);
        }
        Split::Fuzzy { base, .. } => {
            collect_split_feature_indices(base, dense_feature_count, features);
        }
    }
}

fn split_display(
    split: &Split,
    dense_feature_count: usize,
    feature_names: &[String],
) -> (String, String) {
    match split {
        Split::Axis {
            feature, threshold, ..
        } => (
            "axis".to_string(),
            format!(
                "{} <= {:.3}",
                feature_label(*feature, feature_names),
                threshold
            ),
        ),
        Split::Diagonal2D {
            x_feature,
            y_feature,
            normal_x,
            normal_y,
            threshold,
            ..
        } => (
            "diagonal_2d".to_string(),
            format!(
                "{:.2}*{} + {:.2}*{} <= {:.3}",
                normal_x,
                feature_label(*x_feature, feature_names),
                normal_y,
                feature_label(*y_feature, feature_names),
                threshold
            ),
        ),
        Split::Gaussian2D {
            x_feature,
            y_feature,
            center_x,
            center_y,
            radius,
            ..
        } => (
            "gaussian_2d".to_string(),
            format!(
                "{} / {} near {:.2}, {:.2} r{:.2}",
                feature_label(*x_feature, feature_names),
                feature_label(*y_feature, feature_names),
                center_x,
                center_y,
                radius
            ),
        ),
        Split::PeriodicInterval {
            feature,
            period,
            start,
            end,
            ..
        } => (
            "periodic".to_string(),
            format!(
                "{} in {:.2}..{:.2} mod {:.2}",
                feature_label(*feature, feature_names),
                start,
                end,
                period
            ),
        ),
        Split::SparseSetContainsAny { feature, ids, .. } => (
            "sparse_set".to_string(),
            format!(
                "{} has {}",
                feature_label(*feature, feature_names),
                id_summary(ids)
            ),
        ),
        Split::SparseListContainsAny {
            sparse_feature,
            ids,
            ..
        } => (
            "sparse_list".to_string(),
            format!(
                "{} has {}",
                feature_label(dense_feature_count + *sparse_feature, feature_names),
                id_summary(ids)
            ),
        ),
        Split::Fuzzy {
            base,
            bandwidth,
            kernel,
        } => {
            let (_, label) = split_display(base, dense_feature_count, feature_names);
            (
                "fuzzy".to_string(),
                format!("fuzzy {kernel:?} bw {:.3}: {label}", bandwidth),
            )
        }
    }
}

fn feature_label(feature: usize, feature_names: &[String]) -> String {
    feature_names
        .get(feature)
        .cloned()
        .unwrap_or_else(|| format!("feature_{feature}"))
}

fn id_summary(ids: &[u64]) -> String {
    let mut values = ids.iter().take(4).map(u64::to_string).collect::<Vec<_>>();
    if ids.len() > values.len() {
        values.push("...".to_string());
    }
    values.join(",")
}

fn finite_ratio(numerator: f64, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator / denominator as f64
    }
}

#[cfg(test)]
mod model_visualization_tests {
    use super::*;
    use cartoboost_core::Tree;

    #[test]
    fn model_visualization_summarizes_tree_shape_and_split_labels() {
        let trees = vec![Tree {
            root: Node::Branch {
                split: Split::Axis {
                    feature: 0,
                    threshold: 12.5,
                    missing_goes_left: true,
                },
                left: Box::new(Node::Leaf {
                    value: -1.25,
                    sample_weight_sum: 3.0,
                    training_loss: 0.4,
                }),
                right: Box::new(Node::Branch {
                    split: Split::PeriodicInterval {
                        feature: 1,
                        period: 24.0,
                        start: 7.0,
                        end: 10.0,
                        missing_goes_left: false,
                    },
                    left: Box::new(Node::Leaf {
                        value: 0.5,
                        sample_weight_sum: 2.0,
                        training_loss: 0.1,
                    }),
                    right: Box::new(Node::Leaf {
                        value: 1.5,
                        sample_weight_sum: 4.0,
                        training_loss: 0.2,
                    }),
                    gain: 1.25,
                    sample_weight_sum: 6.0,
                }),
                gain: 2.75,
                sample_weight_sum: 9.0,
            },
        }];
        let visualization = model_visualization(
            &trees,
            &["pickup_hour".to_string(), "pickup_dow".to_string()],
            &[],
        );

        assert_eq!(visualization.summary.tree_count, 1);
        assert_eq!(visualization.summary.node_count, 5);
        assert_eq!(visualization.summary.branch_count, 2);
        assert_eq!(visualization.summary.leaf_count, 3);
        assert_eq!(visualization.summary.max_depth, 2);
        assert_eq!(visualization.depth_histogram.len(), 3);
        assert_eq!(visualization.split_kinds[0].kind, "axis");
        assert_eq!(visualization.split_kinds[1].kind, "periodic");
        assert_eq!(visualization.splitter_rules.len(), 2);
        assert!(
            visualization.splitter_rules[0].total_gain
                >= visualization.splitter_rules[1].total_gain
        );
        assert!(visualization
            .feature_split_counts
            .iter()
            .any(|row| row.feature == "pickup_hour" && row.kind == "axis" && row.count == 1));
        assert!(visualization
            .feature_split_counts
            .iter()
            .any(|row| row.feature == "pickup_dow" && row.kind == "periodic" && row.count == 1));
        assert!(visualization.tree_blueprints[0]
            .root
            .label
            .contains("pickup_hour"));
    }
}

fn default_holdout_fraction() -> f64 {
    0.2
}

fn default_true() -> bool {
    true
}

fn default_neural_pipeline() -> String {
    "embedding".to_string()
}

