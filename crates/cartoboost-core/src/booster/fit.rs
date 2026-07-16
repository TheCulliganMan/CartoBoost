use crate::data::{validate_weights, Dataset};
use crate::graph_regularization::{
    CsrGraph, GraphLaplacian, GraphLeafSmoothing, GraphSplitRegularization,
};
use crate::loss::LossConfig;
use crate::objectives::{initial_margin_for_loss, pseudo_residual_for_loss};
use crate::profile;
use crate::tree::{
    FuzzyKernel, LeafPredictorKind, Model, PredictionTransform, SplitterKind,
    TrainingConfigMetadata, TrainingMetric, TreeBuilder, MODEL_ARTIFACT_VERSION,
};
use crate::{CartoBoostError, Result};
use cartoboost_accelerator::{select_backend_for_operations, BackendOperation, BackendSelection};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoosterConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    pub splitters: Vec<SplitterKind>,
    pub leaf_predictor: LeafPredictorKind,
    pub linear_leaf_features: Vec<usize>,
    pub linear_lambda_l2: f64,
    pub constant_lambda_l2: f64,
    pub fuzzy: bool,
    pub fuzzy_bandwidth: f64,
    pub fuzzy_kernel: FuzzyKernel,
    pub loss: LossConfig,
    pub monotonic_constraints: Vec<i8>,
    pub interaction_constraints: Vec<Vec<usize>>,
    pub graph_split_regularization: Option<GraphSplitRegularization>,
    pub graph_leaf_smoothing: Option<GraphLeafSmoothing>,
}

#[derive(Debug, Clone)]
pub struct Booster {
    pub config: BoosterConfig,
    backend: BackendSelection,
}

impl Default for BoosterConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.05,
            max_depth: 4,
            min_samples_leaf: 20,
            min_gain: 1e-8,
            splitters: vec![SplitterKind::Auto],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
            graph_leaf_smoothing: None,
        }
    }
}

impl Booster {
    pub fn new(config: BoosterConfig) -> Self {
        let operations = required_backend_operations(&config);
        Self {
            config,
            backend: select_backend_for_operations(Some("cpu"), &operations)
                .expect("CPU booster operations are always available"),
        }
    }

    pub fn new_with_backend(config: BoosterConfig, backend: Option<&str>) -> Result<Self> {
        let operations = required_backend_operations(&config);
        let backend = select_backend_for_operations(backend, &operations)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        Ok(Self { config, backend })
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.backend
    }

    pub fn fit(&self, x: &Dataset, y: &[f64], sample_weight: Option<&[f64]>) -> Result<Model> {
        let profile_enabled = profile::enabled();
        if profile_enabled {
            profile::reset();
        }
        let profile_started = profile::ProfileTimer::start();
        if !self.config.learning_rate.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "learning_rate must be finite".to_string(),
            ));
        }
        if x.n_rows() != y.len() {
            return Err(CartoBoostError::InvalidInput(
                "X row count must match y length".to_string(),
            ));
        }
        if y.par_iter().any(|v| !v.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "targets must be finite".to_string(),
            ));
        }
        if let LossConfig::LogL2(config) = self.config.loss {
            if y.par_iter().any(|value| *value + config.offset <= 0.0) {
                return Err(CartoBoostError::InvalidInput(
                    "log_l2 targets must be greater than -offset".to_string(),
                ));
            }
        }
        let weights = validate_weights(sample_weight, y.len())?;
        validate_training_config(
            &self.config,
            x.n_cols(),
            x.n_cols() + x.n_sparse_sets(),
            x.n_rows(),
        )?;
        let resolved_splitters = resolve_splitters(&self.config, x);
        let transformed_y;
        let training_targets = match self.config.loss {
            LossConfig::LogL2(config) => {
                transformed_y = y
                    .par_iter()
                    .map(|value| (*value + config.offset).ln())
                    .collect::<Vec<_>>();
                &transformed_y
            }
            _ => y,
        };
        let init_prediction =
            initial_margin_for_loss(&self.config.loss, training_targets, Some(&weights))?;
        let prediction_transform = match self.config.loss {
            LossConfig::LogL2(_) => PredictionTransform::Expm1,
            _ => PredictionTransform::Identity,
        };
        let mut pred = vec![init_prediction; training_targets.len()];
        let mut residuals = vec![0.0; training_targets.len()];
        let mut trees = Vec::with_capacity(self.config.n_estimators);
        let mut training_history = Vec::with_capacity(self.config.n_estimators);
        let builder = TreeBuilder {
            max_depth: self.config.max_depth,
            min_samples_leaf: self.config.min_samples_leaf,
            min_gain: self.config.min_gain,
            splitters: resolved_splitters.clone(),
            leaf_predictor: self.config.leaf_predictor.clone(),
            linear_leaf_features: self.config.linear_leaf_features.clone(),
            linear_lambda_l2: self.config.linear_lambda_l2,
            constant_lambda_l2: self.config.constant_lambda_l2,
            fuzzy: self.config.fuzzy,
            fuzzy_bandwidth: self.config.fuzzy_bandwidth,
            fuzzy_kernel: self.config.fuzzy_kernel,
            loss: self.config.loss.clone(),
            monotonic_constraints: self.config.monotonic_constraints.clone(),
            interaction_constraints: self.config.interaction_constraints.clone(),
            graph_split_regularization: self.config.graph_split_regularization.clone(),
        };
        let fit_context = profile::timed(profile::CONTEXT, || builder.fit_context(x));

        let tree_count = if self.config.max_depth == 0
            && matches!(self.config.leaf_predictor, LeafPredictorKind::Constant)
        {
            0
        } else {
            self.config.n_estimators
        };
        for iteration in 0..tree_count {
            profile::timed(profile::RESIDUAL, || {
                parallel_pseudo_residuals(
                    &mut residuals,
                    training_targets,
                    &pred,
                    &self.config.loss,
                )
            })?;
            let use_leaf_updates = !self.config.fuzzy
                && matches!(self.config.leaf_predictor, LeafPredictorKind::Constant);
            let use_leaf_updates = use_leaf_updates
                && matches!(self.config.loss, LossConfig::L2 | LossConfig::LogL2(_))
                && self.config.monotonic_constraints.is_empty()
                && self.config.graph_leaf_smoothing.is_none();
            let mut tree = if use_leaf_updates {
                let (tree, updates) = profile::timed(profile::TREE_FIT, || {
                    builder.fit_with_leaf_updates_in_context(x, &residuals, &weights, &fit_context)
                });
                profile::timed(profile::PRED_UPDATE, || {
                    parallel_prediction_update(&mut pred, &updates, self.config.learning_rate)
                })?;
                tree
            } else {
                profile::timed(profile::TREE_FIT, || {
                    builder.fit_in_context(x, &residuals, &weights, &fit_context)
                })
            };
            if let Some(smoothing) = &self.config.graph_leaf_smoothing {
                apply_graph_leaf_smoothing(&mut tree, x, smoothing, &self.backend)?;
            }
            if !use_leaf_updates || self.config.graph_leaf_smoothing.is_some() {
                profile::timed(profile::PRED_UPDATE, || {
                    let updates = (0..x.n_rows())
                        .into_par_iter()
                        .map(|row| tree.predict_dataset_row(x, row))
                        .collect::<Vec<_>>();
                    parallel_prediction_update(&mut pred, &updates, self.config.learning_rate)
                })?;
            }
            trees.push(tree);
            training_history.push(TrainingMetric {
                iteration: iteration + 1,
                name: "train/rmse".to_string(),
                value: weighted_rmse(training_targets, &pred, &weights),
            });
        }

        let model = Model {
            artifact_version: MODEL_ARTIFACT_VERSION,
            metadata: Some(Model::default_metadata()),
            init_prediction,
            learning_rate: self.config.learning_rate,
            feature_count: x.n_cols(),
            feature_schema: Some(x.feature_schema_or_default()),
            target_name: None,
            training_config: Some(TrainingConfigMetadata {
                backend: Some(self.backend.clone()),
                n_estimators: self.config.n_estimators,
                learning_rate: self.config.learning_rate,
                max_depth: self.config.max_depth,
                min_samples_leaf: self.config.min_samples_leaf,
                min_gain: self.config.min_gain,
                splitters: resolved_splitters,
                leaf_predictor: self.config.leaf_predictor.clone(),
                linear_leaf_features: self.config.linear_leaf_features.clone(),
                linear_lambda_l2: self.config.linear_lambda_l2,
                constant_lambda_l2: self.config.constant_lambda_l2,
                fuzzy: self.config.fuzzy,
                fuzzy_bandwidth: self.config.fuzzy_bandwidth,
                fuzzy_kernel: self.config.fuzzy_kernel,
                loss: self.config.loss.clone(),
                monotonic_constraints: self.config.monotonic_constraints.clone(),
                interaction_constraints: self.config.interaction_constraints.clone(),
                graph_split_regularization: self.config.graph_split_regularization.clone(),
                graph_leaf_smoothing: self.config.graph_leaf_smoothing.clone(),
            }),
            prediction_transform,
            training_history,
            trees,
        };
        profile::report("booster_fit", profile_started.elapsed());
        Ok(model)
    }
}

fn required_backend_operations(config: &BoosterConfig) -> Vec<BackendOperation> {
    let mut operations = vec![BackendOperation::Dense];
    if config.graph_leaf_smoothing.is_some() {
        operations.push(BackendOperation::CsrDiffusion);
    }
    operations
}

fn parallel_pseudo_residuals(
    residuals: &mut [f64],
    targets: &[f64],
    predictions: &[f64],
    loss: &LossConfig,
) -> Result<()> {
    residuals
        .par_iter_mut()
        .zip(targets.par_iter())
        .zip(predictions.par_iter())
        .for_each(|((residual, target), prediction)| {
            *residual = pseudo_residual_for_loss(loss, *target, *prediction);
        });
    Ok(())
}

pub(crate) fn parallel_prediction_update(
    predictions: &mut [f64],
    updates: &[f64],
    learning_rate: f64,
) -> Result<()> {
    predictions
        .par_iter_mut()
        .zip(updates.par_iter())
        .for_each(|(prediction, update)| {
            *prediction += learning_rate * update;
        });
    Ok(())
}

fn weighted_rmse(targets: &[f64], predictions: &[f64], weights: &[f64]) -> f64 {
    let (loss, weight_sum) = targets
        .par_iter()
        .zip(predictions.par_iter())
        .zip(weights.par_iter())
        .map(|((&target, &prediction), &weight)| {
            let residual = target - prediction;
            (weight * residual * residual, weight)
        })
        .reduce(
            || (0.0, 0.0),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );
    if weight_sum <= 0.0 {
        0.0
    } else {
        (loss / weight_sum).sqrt()
    }
}

fn validate_training_config(
    config: &BoosterConfig,
    feature_count: usize,
    total_feature_count: usize,
    row_count: usize,
) -> Result<()> {
    match config.loss {
        LossConfig::L2 => {}
        LossConfig::L1 => {
            if config.leaf_predictor != LeafPredictorKind::Constant {
                return Err(CartoBoostError::InvalidInput(
                    "l1 loss currently requires constant leaves".to_string(),
                ));
            }
        }
        LossConfig::Huber(loss) => {
            if !loss.delta.is_finite() || loss.delta <= 0.0 {
                return Err(CartoBoostError::InvalidInput(
                    "huber delta must be positive and finite".to_string(),
                ));
            }
            if config.leaf_predictor != LeafPredictorKind::Constant {
                return Err(CartoBoostError::InvalidInput(
                    "huber loss currently requires constant leaves".to_string(),
                ));
            }
        }
        LossConfig::LogL2(loss) => {
            if (loss.offset - 1.0).abs() > 1e-12 {
                return Err(CartoBoostError::InvalidInput(
                    "log_l2 currently supports offset=1.0".to_string(),
                ));
            }
            if config.leaf_predictor != LeafPredictorKind::Constant {
                return Err(CartoBoostError::InvalidInput(
                    "log_l2 loss currently requires constant leaves".to_string(),
                ));
            }
        }
        LossConfig::Quantile(loss) => {
            if !loss.alpha.is_finite() || loss.alpha <= 0.0 || loss.alpha >= 1.0 {
                return Err(CartoBoostError::InvalidInput(
                    "quantile alpha must be finite and in (0, 1)".to_string(),
                ));
            }
            if config.leaf_predictor != LeafPredictorKind::Constant {
                return Err(CartoBoostError::InvalidInput(
                    "quantile loss currently requires constant leaves".to_string(),
                ));
            }
        }
    }
    if !config.monotonic_constraints.is_empty() {
        if config.monotonic_constraints.len() != feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "monotonic_constraints has length {}, but X has {feature_count} features",
                config.monotonic_constraints.len()
            )));
        }
        if config
            .monotonic_constraints
            .iter()
            .any(|constraint| !matches!(*constraint, -1..=1))
        {
            return Err(CartoBoostError::InvalidInput(
                "monotonic_constraints values must be -1, 0, or 1".to_string(),
            ));
        }
        if config.leaf_predictor != LeafPredictorKind::Constant {
            return Err(CartoBoostError::InvalidInput(
                "monotonic constraints currently require constant leaves".to_string(),
            ));
        }
        if config.fuzzy {
            return Err(CartoBoostError::InvalidInput(
                "monotonic constraints currently require hard routing".to_string(),
            ));
        }
        if config.splitters.iter().any(|splitter| {
            !matches!(
                splitter,
                SplitterKind::Auto | SplitterKind::Axis | SplitterKind::AxisHistogram { .. }
            )
        }) {
            return Err(CartoBoostError::InvalidInput(
                "monotonic constraints currently support only axis splitters".to_string(),
            ));
        }
    }
    validate_interaction_constraints(&config.interaction_constraints, total_feature_count)?;
    if let Some(graph_split_regularization) = &config.graph_split_regularization {
        graph_split_regularization.validate_row_count(row_count)?;
    }
    if let Some(graph_leaf_smoothing) = &config.graph_leaf_smoothing {
        graph_leaf_smoothing.validate_row_count(row_count)?;
        if config.leaf_predictor != LeafPredictorKind::Constant {
            return Err(CartoBoostError::InvalidInput(
                "graph_leaf_smoothing requires constant leaves".to_string(),
            ));
        }
        if config.fuzzy {
            return Err(CartoBoostError::InvalidInput(
                "graph_leaf_smoothing requires hard routing".to_string(),
            ));
        }
    }
    Ok(())
}

pub(super) fn apply_graph_leaf_smoothing(
    tree: &mut crate::tree::Tree,
    x: &Dataset,
    smoothing: &GraphLeafSmoothing,
    backend: &BackendSelection,
) -> Result<()> {
    smoothing.validate_row_count(x.n_rows())?;
    if smoothing.lambda == 0.0 || smoothing.iterations == 0 {
        return Ok(());
    }
    let mut leaf_values = Vec::new();
    collect_leaf_values(&tree.root, &mut leaf_values);
    if leaf_values.len() <= 1 {
        return Ok(());
    }
    let row_leaf_ids = (0..x.n_rows())
        .map(|row| leaf_id_for_row(&tree.root, x, row, 0))
        .collect::<Vec<_>>();
    let leaf_graph = leaf_graph_from_row_graph(&smoothing.graph, &row_leaf_ids, leaf_values.len())?;
    let smoothed = smoothing.smoother().smooth_leaf_values_with_backend(
        &leaf_values,
        &GraphLaplacian::new(leaf_graph),
        Some(&backend.selected),
    )?;
    let mut next_leaf = 0usize;
    assign_leaf_values(&mut tree.root, &smoothed, &mut next_leaf);
    Ok(())
}

fn collect_leaf_values(node: &crate::tree::Node, values: &mut Vec<f64>) {
    match node {
        crate::tree::Node::Leaf { value, .. } => values.push(*value),
        crate::tree::Node::LinearLeaf { .. } => {}
        crate::tree::Node::Branch { left, right, .. } => {
            collect_leaf_values(left, values);
            collect_leaf_values(right, values);
        }
    }
}

fn assign_leaf_values(node: &mut crate::tree::Node, values: &[f64], next_leaf: &mut usize) {
    match node {
        crate::tree::Node::Leaf { value, .. } => {
            if let Some(smoothed) = values.get(*next_leaf) {
                *value = *smoothed;
            }
            *next_leaf += 1;
        }
        crate::tree::Node::LinearLeaf { .. } => {}
        crate::tree::Node::Branch { left, right, .. } => {
            assign_leaf_values(left, values, next_leaf);
            assign_leaf_values(right, values, next_leaf);
        }
    }
}

fn leaf_id_for_row(node: &crate::tree::Node, x: &Dataset, row: usize, offset: usize) -> usize {
    match node {
        crate::tree::Node::Leaf { .. } | crate::tree::Node::LinearLeaf { .. } => offset,
        crate::tree::Node::Branch {
            split, left, right, ..
        } => {
            let left_count = leaf_count(left);
            if split.hard_goes_left_dataset_row(x, row) {
                leaf_id_for_row(left, x, row, offset)
            } else {
                leaf_id_for_row(right, x, row, offset + left_count)
            }
        }
    }
}

fn leaf_count(node: &crate::tree::Node) -> usize {
    match node {
        crate::tree::Node::Leaf { .. } | crate::tree::Node::LinearLeaf { .. } => 1,
        crate::tree::Node::Branch { left, right, .. } => leaf_count(left) + leaf_count(right),
    }
}

fn leaf_graph_from_row_graph(
    row_graph: &CsrGraph,
    row_leaf_ids: &[usize],
    leaf_count: usize,
) -> Result<CsrGraph> {
    if row_graph.node_count != row_leaf_ids.len() {
        return Err(CartoBoostError::InvalidInput(
            "row graph node_count must match row leaf assignment count".to_string(),
        ));
    }
    let mut weights = std::collections::BTreeMap::<(usize, usize), f64>::new();
    for left_row in 0..row_graph.node_count {
        let left_leaf = row_leaf_ids[left_row];
        for (right_row, weight) in row_graph.neighbors(left_row)? {
            let right_leaf = row_leaf_ids[right_row];
            if left_leaf != right_leaf {
                *weights.entry((left_leaf, right_leaf)).or_insert(0.0) += weight;
            }
        }
    }
    let edges = weights
        .into_iter()
        .map(|((left, right), weight)| (left, right, weight))
        .collect::<Vec<_>>();
    CsrGraph::from_edges(leaf_count, &edges)
}

fn validate_interaction_constraints(
    interaction_constraints: &[Vec<usize>],
    feature_count: usize,
) -> Result<()> {
    for group in interaction_constraints {
        if group.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "interaction constraint groups must not be empty".to_string(),
            ));
        }
        if group.iter().any(|feature| *feature >= feature_count) {
            return Err(CartoBoostError::InvalidInput(
                "interaction constraint feature index out of range".to_string(),
            ));
        }
        if group.windows(2).any(|window| window[0] >= window[1]) {
            return Err(CartoBoostError::InvalidInput(
                "interaction constraint groups must be sorted and deduplicated".to_string(),
            ));
        }
    }
    Ok(())
}

fn resolve_splitters(config: &BoosterConfig, x: &Dataset) -> Vec<SplitterKind> {
    let fallback = if should_use_histogram_auto(config, x) {
        SplitterKind::AxisHistogram { bins: 512 }
    } else {
        SplitterKind::Axis
    };
    let mut resolved = Vec::with_capacity(config.splitters.len().max(1));
    for splitter in &config.splitters {
        match splitter {
            SplitterKind::Auto => resolved.push(fallback.clone()),
            other => resolved.push(other.clone()),
        }
    }
    if resolved.is_empty() {
        resolved.push(fallback);
    }
    resolved
}

fn should_use_histogram_auto(config: &BoosterConfig, x: &Dataset) -> bool {
    x.n_rows() >= 2_048
        && !config.fuzzy
        && config.monotonic_constraints.is_empty()
        && matches!(config.loss, LossConfig::L2 | LossConfig::LogL2(_))
        && matches!(config.leaf_predictor, LeafPredictorKind::Constant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_regularization::{CsrGraph, GraphLeafSmoothing, GraphSplitRegularization};
    use crate::loss::{L2Loss, LossConfig, QuantileLossConfig};
    use crate::tree::{Split, MODEL_ARTIFACT_VERSION};

    #[test]
    fn one_tree_booster_predicts_training_stump_with_learning_rate_one() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(model.artifact_version, MODEL_ARTIFACT_VERSION);
        assert_predictions_close(&model.predict(&x), &y);
    }

    #[test]
    fn accelerated_booster_training_updates_match_cpu() {
        let x = Dataset::from_rows(
            (0..256)
                .map(|row| vec![row as f64 / 32.0, (row as f64 * 0.17).sin()])
                .collect(),
        )
        .unwrap();
        let y = (0..256)
            .map(|row| if row % 7 < 3 { 2.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let config = BoosterConfig {
            n_estimators: 4,
            learning_rate: 0.1,
            max_depth: 2,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        };
        let cpu = Booster::new(config.clone()).fit(&x, &y, None).unwrap();
        for backend in cartoboost_accelerator::available_backends() {
            if backend == "cpu"
                || !cartoboost_accelerator::backend_supports_operation(
                    &backend,
                    BackendOperation::Dense,
                )
            {
                continue;
            }
            let accelerated = Booster::new_with_backend(config.clone(), Some(&backend))
                .unwrap()
                .fit(&x, &y, None)
                .unwrap();
            for (expected, actual) in cpu.predict(&x).iter().zip(accelerated.predict(&x)) {
                assert!(
                    (expected - actual).abs() < 1.0e-4,
                    "backend {backend}: {expected} != {actual}"
                );
            }
            let inference = accelerated
                .try_predict_with_backend(&x, Some(&backend))
                .unwrap();
            let persisted_backend_inference = accelerated.try_predict(&x).unwrap();
            assert_eq!(persisted_backend_inference.len(), inference.len());
            for (persisted, explicit) in persisted_backend_inference.iter().zip(&inference) {
                assert!(
                    (persisted - explicit).abs() < 1.0e-6,
                    "backend {backend} persisted selection was not reused"
                );
            }
            for (expected, actual) in cpu.predict(&x).iter().zip(inference) {
                assert!(
                    (expected - actual).abs() < 1.0e-4,
                    "backend {backend} inference: {expected} != {actual}"
                );
            }
            assert_eq!(
                accelerated
                    .training_config
                    .as_ref()
                    .unwrap()
                    .backend
                    .as_ref()
                    .unwrap()
                    .selected,
                backend
            );
        }
    }

    #[test]
    fn parallel_l2_residual_generation_matches_reference() {
        const ROWS: usize = 16_384;
        let targets = (0..ROWS)
            .map(|index| (index % 19) as f64 - 4.0)
            .collect::<Vec<_>>();
        let predictions = (0..ROWS)
            .map(|index| (index % 7) as f64 * 0.25)
            .collect::<Vec<_>>();
        let expected = targets
            .iter()
            .zip(&predictions)
            .map(|(target, prediction)| target - prediction)
            .collect::<Vec<_>>();
        let mut actual = vec![0.0; targets.len()];
        parallel_pseudo_residuals(&mut actual, &targets, &predictions, &LossConfig::L2).unwrap();
        for (left, right) in actual.iter().zip(&expected) {
            assert!((left - right).abs() < 1.0e-12, "{left} != {right}");
        }
    }

    fn assert_predictions_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn auto_splitter_resolves_to_exact_axis_for_small_training_sets() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(
            model.training_config.as_ref().unwrap().splitters,
            vec![SplitterKind::Axis]
        );
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Axis { .. },
                ..
            }
        ));
    }

    #[test]
    fn auto_splitter_resolves_to_histogram_for_large_dense_l2_training_sets() {
        let rows = 2_048;
        let cols = 2;
        let values = (0..rows * cols)
            .map(|index| {
                let row = index / cols;
                let col = index % cols;
                (row as f64 * 0.03 + col as f64).sin()
            })
            .collect::<Vec<_>>();
        let y = (0..rows)
            .map(|row| if row % 11 < 5 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let x = Dataset::from_flat(rows, cols, values).unwrap();
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(
            model.training_config.as_ref().unwrap().splitters,
            vec![SplitterKind::AxisHistogram { bins: 512 }]
        );
    }

    #[test]
    fn explicit_axis_splitter_preserves_exact_axis_metadata_on_large_training_sets() {
        let rows = 2_048;
        let x = Dataset::from_flat(rows, 1, (0..rows).map(|row| row as f64).collect::<Vec<_>>())
            .unwrap();
        let y = (0..rows)
            .map(|row| if row < rows / 2 { 0.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(
            model.training_config.as_ref().unwrap().splitters,
            vec![SplitterKind::Axis]
        );
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Axis { .. },
                ..
            }
        ));
    }

    #[test]
    fn booster_reduces_l2_loss_and_json_round_trips_predictions() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 3,
            learning_rate: 0.5,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let loss = L2Loss;
        let initial = vec![model.init_prediction; y.len()];
        let predictions = model.predict(&x);

        assert!(loss.value(&y, &predictions) < loss.value(&y, &initial));

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("model.json");
        model.save(&path).unwrap();
        let loaded = Model::load(&path).unwrap();

        assert_eq!(loaded.artifact_version, MODEL_ARTIFACT_VERSION);
        assert_eq!(loaded.predict(&x), predictions);
    }

    #[test]
    fn quantile_booster_uses_weighted_quantile_initial_prediction_and_metadata() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let weights = vec![1.0, 1.0, 10.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            loss: LossConfig::Quantile(QuantileLossConfig { alpha: 0.8 }),
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, Some(&weights)).unwrap();

        assert_eq!(model.init_prediction, 20.0);
        assert_eq!(
            model.training_config.as_ref().unwrap().loss,
            LossConfig::Quantile(QuantileLossConfig { alpha: 0.8 })
        );
        assert!(model
            .predict(&x)
            .iter()
            .all(|prediction| prediction.is_finite()));
    }

    #[test]
    fn l1_booster_uses_weighted_median_initial_prediction_and_predictions_are_finite() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let weights = vec![1.0, 1.0, 10.0, 1.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 0,
            min_samples_leaf: 1,
            min_gain: 0.0,
            loss: LossConfig::L1,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, Some(&weights)).unwrap();

        assert_eq!(model.init_prediction, 20.0);
        assert_eq!(model.training_config.as_ref().unwrap().loss, LossConfig::L1);
        assert_eq!(model.predict(&x), vec![20.0; 4]);
    }

    #[test]
    fn quantile_zero_depth_json_round_trips_constant_prediction() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 10.0, 20.0, 30.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 0,
            min_samples_leaf: 1,
            min_gain: 0.0,
            loss: LossConfig::Quantile(QuantileLossConfig { alpha: 0.8 }),
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("quantile.json");
        model.save(&path).unwrap();
        let loaded = Model::load(&path).unwrap();

        assert_eq!(model.predict(&x), vec![30.0; 4]);
        assert_eq!(loaded.predict(&x), model.predict(&x));
    }

    #[test]
    fn monotonic_constraints_reject_decreasing_axis_split_for_increasing_feature() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![10.0, 10.0, 0.0, 0.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            monotonic_constraints: vec![1],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Leaf { .. }
        ));
        assert_eq!(
            model
                .training_config
                .as_ref()
                .unwrap()
                .monotonic_constraints,
            vec![1]
        );
    }

    #[test]
    fn interaction_constraints_gate_second_level_split_search() {
        let x = Dataset::from_rows(vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ])
        .unwrap();
        let y = vec![0.0, 1.0, 10.0, 11.0];
        let constrained = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 2,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            interaction_constraints: vec![vec![0, 2]],
            ..BoosterConfig::default()
        })
        .fit(&x, &y, None)
        .unwrap();
        let allowed = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 2,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            interaction_constraints: vec![vec![0, 1]],
            ..BoosterConfig::default()
        })
        .fit(&x, &y, None)
        .unwrap();

        assert!(tree_uses_feature(&constrained.trees[0].root, 0));
        assert!(!tree_uses_feature(&constrained.trees[0].root, 1));
        assert!(tree_uses_feature(&allowed.trees[0].root, 1));
        assert_eq!(
            constrained
                .training_config
                .as_ref()
                .unwrap()
                .interaction_constraints,
            vec![vec![0, 2]]
        );
        assert!(
            L2Loss.value(&y, &allowed.predict(&x)) < L2Loss.value(&y, &constrained.predict(&x))
        );

        let encoded = serde_json::to_string(&constrained).unwrap();
        let decoded: Model = serde_json::from_str(&encoded).unwrap();
        assert_eq!(
            decoded
                .training_config
                .as_ref()
                .unwrap()
                .interaction_constraints,
            vec![vec![0, 2]]
        );
        assert_eq!(decoded.predict(&x), constrained.predict(&x));
    }

    #[test]
    fn interaction_constraints_validate_sorted_in_range_groups() {
        let x = Dataset::from_rows(vec![vec![0.0, 0.0], vec![1.0, 1.0]]).unwrap();
        let y = vec![0.0, 1.0];
        let unsorted = Booster::new(BoosterConfig {
            interaction_constraints: vec![vec![1, 0]],
            ..BoosterConfig::default()
        })
        .fit(&x, &y, None)
        .unwrap_err();
        assert!(unsorted.to_string().contains("sorted and deduplicated"));

        let out_of_range = Booster::new(BoosterConfig {
            interaction_constraints: vec![vec![2]],
            ..BoosterConfig::default()
        })
        .fit(&x, &y, None)
        .unwrap_err();
        assert!(out_of_range.to_string().contains("out of range"));
    }

    #[test]
    fn graph_split_regularization_lambda_zero_matches_baseline_predictions() {
        let x = Dataset::from_rows(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ])
        .unwrap();
        let y = vec![0.0, 10.0, 10.0, 10.0];
        let graph =
            CsrGraph::from_edges(4, &[(0, 2, 1.0), (2, 0, 1.0), (1, 3, 1.0), (3, 1, 1.0)]).unwrap();
        let config = BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        };
        let baseline = Booster::new(config.clone()).fit(&x, &y, None).unwrap();
        let regularized = Booster::new(BoosterConfig {
            graph_split_regularization: Some(GraphSplitRegularization::new(graph, 0.0).unwrap()),
            ..config
        })
        .fit(&x, &y, None)
        .unwrap();

        assert_eq!(regularized.predict(&x), baseline.predict(&x));
        assert_eq!(
            regularized
                .training_config
                .as_ref()
                .unwrap()
                .graph_split_regularization
                .as_ref()
                .unwrap()
                .lambda,
            0.0
        );
    }

    #[test]
    fn graph_split_regularization_penalizes_rough_candidate_updates() {
        let x = Dataset::from_rows(vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ])
        .unwrap();
        let y = vec![0.0, 10.0, 10.0, 10.0];
        let graph =
            CsrGraph::from_edges(4, &[(0, 2, 1.0), (2, 0, 1.0), (1, 3, 1.0), (3, 1, 1.0)]).unwrap();
        let baseline = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        })
        .fit(&x, &y, None)
        .unwrap();
        let regularized = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            graph_split_regularization: Some(GraphSplitRegularization::new(graph, 1.0).unwrap()),
            ..BoosterConfig::default()
        })
        .fit(&x, &y, None)
        .unwrap();

        assert!(root_uses_axis_feature(&baseline, 0));
        assert!(root_uses_axis_feature(&regularized, 1));

        let encoded = serde_json::to_string(&regularized).unwrap();
        let decoded: Model = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.predict(&x), regularized.predict(&x));
        assert!(decoded
            .training_config
            .as_ref()
            .unwrap()
            .graph_split_regularization
            .is_some());
    }

    #[test]
    fn graph_split_regularization_rejects_row_count_mismatch() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0]]).unwrap();
        let y = vec![0.0, 1.0];
        let graph = CsrGraph::from_edges(3, &[(0, 1, 1.0)]).unwrap();
        let err = Booster::new(BoosterConfig {
            graph_split_regularization: Some(GraphSplitRegularization::new(graph, 1.0).unwrap()),
            ..BoosterConfig::default()
        })
        .fit(&x, &y, None)
        .unwrap_err();

        assert!(err.to_string().contains("training row count"));
    }

    #[test]
    fn graph_leaf_smoothing_lambda_zero_matches_baseline_predictions() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0]]).unwrap();
        let y = vec![0.0, 10.0];
        let graph = CsrGraph::from_edges(2, &[(0, 1, 1.0), (1, 0, 1.0)]).unwrap();
        let config = BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        };
        let baseline = Booster::new(config.clone()).fit(&x, &y, None).unwrap();
        let smoothed = Booster::new(BoosterConfig {
            graph_leaf_smoothing: Some(GraphLeafSmoothing::new(graph, 0.0, 4).unwrap()),
            ..config
        })
        .fit(&x, &y, None)
        .unwrap();

        assert_eq!(smoothed.predict(&x), baseline.predict(&x));
    }

    #[test]
    fn graph_leaf_smoothing_smooths_constant_leaf_updates() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0]]).unwrap();
        let y = vec![0.0, 10.0];
        let graph = CsrGraph::from_edges(2, &[(0, 1, 1.0), (1, 0, 1.0)]).unwrap();
        let model = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            graph_leaf_smoothing: Some(GraphLeafSmoothing::new(graph, 1.0, 1).unwrap()),
            ..BoosterConfig::default()
        })
        .fit(&x, &y, None)
        .unwrap();

        assert_predictions_close(&model.predict(&x), &[5.0, 5.0]);
        assert!(model
            .training_config
            .as_ref()
            .unwrap()
            .graph_leaf_smoothing
            .is_some());

        let encoded = serde_json::to_string(&model).unwrap();
        let decoded: Model = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.predict(&x), model.predict(&x));
    }

    #[test]
    fn diagonal_splitter_solves_diagonal_boundary_stump() {
        let x = Dataset::from_rows(vec![
            vec![-2.0, -1.0],
            vec![-1.0, -1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
        ])
        .unwrap();
        let y = vec![-10.0, -10.0, 10.0, 10.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Diagonal2D],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_predictions_close(&model.predict(&x), &y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Diagonal2D { .. },
                ..
            }
        ));
    }

    #[test]
    fn gaussian_splitter_solves_radial_stump() {
        let x = Dataset::from_rows(vec![
            vec![0.0, 0.0],
            vec![0.25, 0.25],
            vec![3.0, 0.0],
            vec![0.0, 3.0],
        ])
        .unwrap();
        let y = vec![10.0, 10.0, -10.0, -10.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Gaussian2D],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(model.predict(&x), y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Gaussian2D { .. },
                ..
            }
        ));
    }

    #[test]
    fn periodic_splitter_handles_late_night_wraparound_stump() {
        let x = Dataset::from_rows(vec![vec![23.0], vec![1.0], vec![12.0], vec![15.0]]).unwrap();
        let y = vec![5.0, 5.0, -5.0, -5.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_predictions_close(&model.predict(&x), &y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::PeriodicInterval { .. },
                ..
            }
        ));
    }

    #[test]
    fn periodic_splitter_learns_shifted_interval_from_observed_boundaries() {
        let x = Dataset::from_rows(vec![
            vec![0.0],
            vec![1.0],
            vec![4.0],
            vec![5.0],
            vec![6.0],
            vec![7.0],
            vec![11.0],
            vec![14.0],
            vec![18.0],
            vec![21.0],
        ])
        .unwrap();
        let y = vec![-3.0, -3.0, 8.0, 8.0, 8.0, 8.0, -3.0, -3.0, -3.0, -3.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_predictions_close(&model.predict(&x), &y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::PeriodicInterval { .. },
                ..
            }
        ));
    }

    #[test]
    fn sparse_set_splitter_finds_integer_id_membership() {
        let x = Dataset::from_rows(vec![vec![7.0], vec![7.0], vec![3.0], vec![4.0]]).unwrap();
        let y = vec![25.0, 25.0, -5.0, -5.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_eq!(model.predict(&x), y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::SparseSetContainsAny { .. },
                ..
            }
        ));
    }

    #[test]
    fn sparse_list_boosting_updates_residuals_with_dataset_aware_prediction() {
        let dense = Dataset::from_rows(vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]]).unwrap();
        let x = dense
            .with_sparse_sets(vec![crate::data::SparseSetColumn::new(vec![
                vec![7, 11],
                vec![11],
                vec![3],
                vec![],
            ])])
            .unwrap();
        let y = vec![10.0, 10.0, 0.0, 0.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 2,
            learning_rate: 0.5,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let predictions = model.predict(&x);

        assert_predictions_close(&predictions, &[8.75, 8.75, 1.25, 1.25]);
        assert!(model.trees.iter().all(|tree| matches!(
            tree.root,
            crate::tree::Node::Branch {
                split: Split::SparseListContainsAny { .. },
                ..
            }
        )));
    }

    #[test]
    fn gaussian_splitter_learns_off_center_hotspot() {
        let x = Dataset::from_rows(vec![
            vec![4.8, -2.0],
            vec![5.0, -2.1],
            vec![5.2, -1.9],
            vec![5.1, -2.2],
            vec![-5.0, -5.0],
            vec![-4.0, 4.0],
            vec![0.0, 5.0],
            vec![5.0, 5.0],
            vec![-5.0, 0.0],
            vec![0.0, -5.0],
        ])
        .unwrap();
        let y = vec![12.0, 12.0, 12.0, 12.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Gaussian2D],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert_predictions_close(&model.predict(&x), &y);
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Gaussian2D { .. },
                ..
            }
        ));
    }

    #[test]
    fn linear_leaf_fits_gradient_residuals_with_learning_rate_shrinkage() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![3.0, 5.0, 7.0, 9.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 0.5,
            max_depth: 0,
            min_samples_leaf: 1,
            min_gain: 0.0,
            leaf_predictor: LeafPredictorKind::Linear,
            linear_leaf_features: vec![0],
            linear_lambda_l2: 0.0,
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let pred = model.predict(&x);

        assert_eq!(model.init_prediction, 6.0);
        for (actual, expected) in pred.iter().zip([4.5, 5.5, 6.5, 7.5]) {
            assert!(
                (actual - expected).abs() < 1e-12,
                "expected {expected}, got {actual}"
            );
        }
        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::LinearLeaf { .. }
        ));
    }

    #[test]
    fn fuzzy_training_uses_fractional_prediction_and_preserves_mass() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 10.0, 10.0];
        let booster = Booster::new(BoosterConfig {
            n_estimators: 1,
            learning_rate: 1.0,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            fuzzy: true,
            fuzzy_bandwidth: 1.0,
            splitters: vec![SplitterKind::Axis],
            ..BoosterConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();

        assert!(matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Fuzzy { .. },
                ..
            }
        ));
        assert_predictions_close(&model.predict(&x), &[1.25, 3.125, 6.875, 8.75]);
        assert_predictions_close(&[model.predict_one(&[1.5])], &[5.0]);
    }

    fn tree_uses_feature(node: &crate::tree::Node, feature: usize) -> bool {
        match node {
            crate::tree::Node::Branch {
                split, left, right, ..
            } => {
                split_uses_feature(split, feature)
                    || tree_uses_feature(left, feature)
                    || tree_uses_feature(right, feature)
            }
            crate::tree::Node::Leaf { .. } | crate::tree::Node::LinearLeaf { .. } => false,
        }
    }

    fn root_uses_axis_feature(model: &Model, feature: usize) -> bool {
        matches!(
            model.trees[0].root,
            crate::tree::Node::Branch {
                split: Split::Axis {
                    feature: split_feature,
                    ..
                },
                ..
            } if split_feature == feature
        )
    }

    fn split_uses_feature(split: &Split, feature: usize) -> bool {
        match split {
            Split::Axis {
                feature: split_feature,
                ..
            }
            | Split::PeriodicInterval {
                feature: split_feature,
                ..
            }
            | Split::SparseSetContainsAny {
                feature: split_feature,
                ..
            } => *split_feature == feature,
            Split::Diagonal2D {
                x_feature,
                y_feature,
                ..
            }
            | Split::Gaussian2D {
                x_feature,
                y_feature,
                ..
            } => *x_feature == feature || *y_feature == feature,
            Split::SparseListContainsAny { .. } => false,
            Split::Fuzzy { base, .. } => split_uses_feature(base, feature),
        }
    }
}
