use crate::data::{validate_weights, Dataset};
use crate::graph_regularization::GraphLeafSmoothing;
use crate::loss::LossConfig;
use crate::objectives::{
    BinaryLogLossObjective, GradientPair, MetricValue, MulticlassLogLossObjective, Objective,
    PredictionTransformKind,
};
use crate::tree::{
    FuzzyKernel, LeafPredictorKind, ModelMetadata, SplitterKind, TrainingMetric, Tree, TreeBuilder,
};
use crate::{CartoBoostError, Result};
use cartoboost_accelerator::{
    backend_csr_row_softmax_f32, backend_dense_layer_f32, select_backend_for,
    select_backend_for_operations, BackendOperation, BackendSelection,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CLASSIFIER_MODEL_ARTIFACT_VERSION: u32 = 1;
const CLASSIFIER_DENSE_DISPATCH_MIN_OPS: usize = 16_384;
const CLASSIFIER_SOFTMAX_DISPATCH_MIN_OPS: usize = 16_384;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassificationObjective {
    #[default]
    BinaryLogLoss,
    MulticlassLogLoss,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
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
    pub objective: ClassificationObjective,
    pub class_count: usize,
    pub class_weights: Vec<f64>,
    #[serde(default)]
    pub graph_leaf_smoothing: Option<GraphLeafSmoothing>,
}

#[derive(Debug, Clone)]
pub struct Classifier {
    pub config: ClassifierConfig,
    backend: BackendSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierTrainingConfigMetadata {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<BackendSelection>,
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    pub splitters: Vec<SplitterKind>,
    pub leaf_predictor: LeafPredictorKind,
    pub linear_leaf_features: Vec<usize>,
    pub linear_lambda_l2: f64,
    #[serde(default)]
    pub constant_lambda_l2: f64,
    pub fuzzy: bool,
    pub fuzzy_bandwidth: f64,
    #[serde(default)]
    pub fuzzy_kernel: FuzzyKernel,
    pub objective: ClassificationObjective,
    pub class_count: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub class_weights: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_leaf_smoothing: Option<GraphLeafSmoothing>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierModel {
    pub artifact_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ModelMetadata>,
    pub objective: ClassificationObjective,
    pub init_margins: Vec<f64>,
    pub learning_rate: f64,
    pub feature_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_schema: Option<crate::data::FeatureSchema>,
    pub class_values: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_config: Option<ClassifierTrainingConfigMetadata>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub training_history: Vec<TrainingMetric>,
    pub trees: Vec<Vec<Tree>>,
}

impl Default for ClassifierConfig {
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
            objective: ClassificationObjective::BinaryLogLoss,
            class_count: 2,
            class_weights: Vec::new(),
            graph_leaf_smoothing: None,
        }
    }
}

impl Classifier {
    pub fn new(config: ClassifierConfig) -> Self {
        let operations = classifier_backend_operations(&config);
        Self {
            config,
            backend: select_backend_for_operations(Some("cpu"), &operations)
                .expect("CPU classifier operations"),
        }
    }

    pub fn new_with_backend(config: ClassifierConfig, backend: Option<&str>) -> Result<Self> {
        let operations = classifier_backend_operations(&config);
        let backend = select_backend_for_operations(backend, &operations)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        Ok(Self { config, backend })
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.backend
    }

    pub fn fit(
        &self,
        x: &Dataset,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<ClassifierModel> {
        validate_classifier_config(&self.config, x.n_cols())?;
        validate_graph_leaf_smoothing(&self.config, x.n_rows())?;
        if x.n_rows() != y.len() {
            return Err(CartoBoostError::InvalidInput(
                "X row count must match y length".to_string(),
            ));
        }
        let class_count = resolve_class_count(&self.config, y)?;
        let class_values = (0..class_count)
            .map(|class| class as f64)
            .collect::<Vec<_>>();
        validate_class_targets(y, class_count)?;
        let base_weights = validate_weights(sample_weight, y.len())?;
        let effective_weights = apply_class_weights(&base_weights, y, &self.config.class_weights)?;
        let objective = make_objective(self.config.objective, class_count)?;
        let init_margins = objective.initial_margin(y, Some(&effective_weights))?;
        let output_dimension = objective.output_dimension();
        let mut raw_predictions = vec![0.0; y.len() * output_dimension];
        raw_predictions
            .par_chunks_mut(output_dimension)
            .for_each(|row| row.copy_from_slice(&init_margins));
        let mut trees = Vec::with_capacity(self.config.n_estimators);
        let mut training_history = Vec::with_capacity(self.config.n_estimators);
        let builder = TreeBuilder {
            max_depth: self.config.max_depth,
            min_samples_leaf: self.config.min_samples_leaf,
            min_gain: self.config.min_gain,
            splitters: self.config.splitters.clone(),
            leaf_predictor: self.config.leaf_predictor.clone(),
            linear_leaf_features: self.config.linear_leaf_features.clone(),
            linear_lambda_l2: self.config.linear_lambda_l2,
            constant_lambda_l2: self.config.constant_lambda_l2,
            fuzzy: self.config.fuzzy,
            fuzzy_bandwidth: self.config.fuzzy_bandwidth,
            fuzzy_kernel: self.config.fuzzy_kernel,
            loss: LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };
        let fit_context = builder.fit_context(x);

        for iteration in 0..self.config.n_estimators {
            let derivative_pairs = multiclass_derivatives_with_backend(
                self.config.objective,
                y,
                &raw_predictions,
                &effective_weights,
                output_dimension,
                &self.backend,
            )?
            .map_or_else(
                || {
                    objective.gradients_hessians(
                        y,
                        &raw_predictions,
                        Some(&effective_weights),
                        None,
                    )
                },
                Ok,
            )?;
            let mut iteration_trees = Vec::with_capacity(output_dimension);
            for output in 0..output_dimension {
                let mut targets = vec![0.0; y.len()];
                let mut hessian_weights = vec![1.0e-12; y.len()];
                targets
                    .par_iter_mut()
                    .zip(hessian_weights.par_iter_mut())
                    .enumerate()
                    .for_each(|(row, (target, hessian_weight))| {
                        let pair = derivative_pairs[row * output_dimension + output];
                        let hessian = pair.hessian.max(1.0e-12);
                        *target = -pair.gradient / hessian;
                        *hessian_weight = hessian;
                    });
                let mut tree = builder.fit_in_context(x, &targets, &hessian_weights, &fit_context);
                if let Some(smoothing) = &self.config.graph_leaf_smoothing {
                    super::fit::apply_graph_leaf_smoothing(&mut tree, x, smoothing, &self.backend)?;
                }
                let mut margins = raw_predictions
                    .chunks(output_dimension)
                    .map(|row| row[output])
                    .collect::<Vec<_>>();
                let updates = (0..x.n_rows())
                    .into_par_iter()
                    .map(|row| tree.predict_dataset_row(x, row))
                    .collect::<Vec<_>>();
                super::fit::parallel_prediction_update(
                    &mut margins,
                    &updates,
                    self.config.learning_rate,
                )?;
                for (row, value) in margins.into_iter().enumerate() {
                    raw_predictions[row * output_dimension + output] = value;
                }
                iteration_trees.push(tree);
            }
            trees.push(iteration_trees);
            let metric = multiclass_metric_with_backend(
                self.config.objective,
                y,
                &raw_predictions,
                output_dimension,
                &self.backend,
            )?
            .map_or_else(|| objective.default_metric(y, &raw_predictions, None), Ok)?;
            training_history.push(TrainingMetric {
                iteration: iteration + 1,
                name: format!("train/{}", metric.name),
                value: metric.value,
            });
        }

        Ok(ClassifierModel {
            artifact_version: CLASSIFIER_MODEL_ARTIFACT_VERSION,
            metadata: Some(crate::tree::Model::default_metadata()),
            objective: self.config.objective,
            init_margins,
            learning_rate: self.config.learning_rate,
            feature_count: x.n_cols(),
            feature_schema: Some(x.feature_schema_or_default()),
            class_values,
            training_config: Some(ClassifierTrainingConfigMetadata {
                backend: Some(self.backend.clone()),
                n_estimators: self.config.n_estimators,
                learning_rate: self.config.learning_rate,
                max_depth: self.config.max_depth,
                min_samples_leaf: self.config.min_samples_leaf,
                min_gain: self.config.min_gain,
                splitters: self.config.splitters.clone(),
                leaf_predictor: self.config.leaf_predictor.clone(),
                linear_leaf_features: self.config.linear_leaf_features.clone(),
                linear_lambda_l2: self.config.linear_lambda_l2,
                constant_lambda_l2: self.config.constant_lambda_l2,
                fuzzy: self.config.fuzzy,
                fuzzy_bandwidth: self.config.fuzzy_bandwidth,
                fuzzy_kernel: self.config.fuzzy_kernel,
                objective: self.config.objective,
                class_count,
                class_weights: self.config.class_weights.clone(),
                graph_leaf_smoothing: self.config.graph_leaf_smoothing.clone(),
            }),
            training_history,
            trees,
        })
    }
}

impl ClassifierModel {
    pub fn output_dimension(&self) -> usize {
        match self.objective {
            ClassificationObjective::BinaryLogLoss => 1,
            ClassificationObjective::MulticlassLogLoss => self.class_values.len(),
        }
    }

    pub fn requires_sparse_sets(&self) -> bool {
        self.trees
            .iter()
            .flatten()
            .any(Tree::contains_sparse_list_split)
    }

    pub fn raw_predict_dataset_row(&self, x: &Dataset, row: usize) -> Vec<f64> {
        let mut margins = self.init_margins.clone();
        for tree_group in &self.trees {
            for (output, tree) in tree_group.iter().enumerate() {
                margins[output] += self.learning_rate * tree.predict_dataset_row(x, row);
            }
        }
        margins
    }

    pub fn decision_function(&self, x: &Dataset) -> Result<Vec<Vec<f64>>> {
        if let Some(backend) = self.artifact_backend() {
            return self.decision_function_with_selection(x, backend);
        }
        self.decision_function_cpu(x)
    }

    fn decision_function_cpu(&self, x: &Dataset) -> Result<Vec<Vec<f64>>> {
        self.validate_dataset(x)?;
        Ok((0..x.n_rows())
            .into_par_iter()
            .map(|row| self.raw_predict_dataset_row(x, row))
            .collect())
    }

    pub fn decision_function_with_backend(
        &self,
        x: &Dataset,
        backend: Option<&str>,
    ) -> Result<Vec<Vec<f64>>> {
        let selection = select_backend_for(backend, BackendOperation::Dense)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        self.decision_function_with_selection(x, &selection)
    }

    fn decision_function_with_selection(
        &self,
        x: &Dataset,
        selection: &BackendSelection,
    ) -> Result<Vec<Vec<f64>>> {
        self.validate_dataset(x)?;
        let outputs = self.output_dimension();
        let width = outputs * (self.trees.len() + 1);
        if selection.selected == "cpu"
            || self.trees.is_empty()
            || x.n_rows().saturating_mul(width).saturating_mul(outputs)
                < CLASSIFIER_DENSE_DISPATCH_MIN_OPS
        {
            return self.decision_function_cpu(x);
        }
        let input = (0..x.n_rows())
            .into_par_iter()
            .map(|row| {
                self.init_margins
                    .iter()
                    .copied()
                    .chain(self.trees.iter().flat_map(move |group| {
                        group
                            .iter()
                            .map(move |tree| self.learning_rate * tree.predict_dataset_row(x, row))
                    }))
                    .map(|value| value as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut weights = vec![0.0_f32; width * outputs];
        for round in 0..=self.trees.len() {
            for output in 0..outputs {
                weights[(round * outputs + output) * outputs + output] = 1.0;
            }
        }
        let flat = backend_dense_layer_f32(selection, &input, &weights, &vec![0.0; outputs])
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        Ok(flat
            .into_iter()
            .map(|row| row.into_iter().map(f64::from).collect())
            .collect())
    }

    pub fn predict_proba(&self, x: &Dataset) -> Result<Vec<Vec<f64>>> {
        if let Some(backend) = self.artifact_backend() {
            return self.predict_proba_with_selection(x, backend);
        }
        self.predict_proba_with_backend(x, Some("cpu"))
    }

    pub fn predict_proba_with_backend(
        &self,
        x: &Dataset,
        backend: Option<&str>,
    ) -> Result<Vec<Vec<f64>>> {
        let selection = select_backend_for_operations(backend, &self.inference_operations())
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        self.predict_proba_with_selection(x, &selection)
    }

    fn predict_proba_with_selection(
        &self,
        x: &Dataset,
        selection: &BackendSelection,
    ) -> Result<Vec<Vec<f64>>> {
        let objective = make_objective(self.objective, self.class_values.len())?;
        let transform = objective.prediction_transform();
        let margins = self.decision_function_with_selection(x, selection)?;
        if transform == PredictionTransformKind::Softmax {
            let width = self.output_dimension();
            let logits = margins.iter().flatten().copied().collect::<Vec<_>>();
            if let Some(values) = accelerated_row_softmax(selection, &logits, margins.len(), width)?
            {
                return Ok(values
                    .chunks_exact(width)
                    .map(|row| row.iter().map(|value| f64::from(*value)).collect())
                    .collect());
            }
        }
        Ok(margins
            .into_par_iter()
            .map(|row| transform_margin_row(transform, &row))
            .collect())
    }

    pub fn predict(&self, x: &Dataset) -> Result<Vec<f64>> {
        Ok(self
            .predict_proba(x)?
            .into_iter()
            .map(|probabilities| {
                let class = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.class_values[class]
            })
            .collect())
    }

    fn artifact_backend(&self) -> Option<&BackendSelection> {
        self.training_config
            .as_ref()
            .and_then(|config| config.backend.as_ref())
    }

    fn inference_operations(&self) -> Vec<BackendOperation> {
        let mut operations = vec![BackendOperation::Dense];
        if self.objective == ClassificationObjective::MulticlassLogLoss {
            operations.push(BackendOperation::CsrRowSoftmax);
        }
        operations
    }

    pub fn predict_with_backend(&self, x: &Dataset, backend: Option<&str>) -> Result<Vec<f64>> {
        Ok(self
            .predict_proba_with_backend(x, backend)?
            .into_iter()
            .map(|probabilities| {
                let class = probabilities
                    .iter()
                    .enumerate()
                    .max_by(|left, right| left.1.total_cmp(right.1))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.class_values[class]
            })
            .collect())
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialize::save_json(self, path)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model: Self = crate::serialize::load_json(path)?;
        if model.artifact_version != CLASSIFIER_MODEL_ARTIFACT_VERSION {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported classifier model artifact version {}",
                model.artifact_version
            )));
        }
        Ok(model)
    }

    fn validate_dataset(&self, x: &Dataset) -> Result<()> {
        if x.n_cols() != self.feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "X has {} features, but model expects {}",
                x.n_cols(),
                self.feature_count
            )));
        }
        if self.requires_sparse_sets() && x.n_sparse_sets() == 0 {
            return Err(CartoBoostError::InvalidInput(
                "prediction requires sparse_sets for a model with list-valued sparse splits"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

fn classifier_backend_operations(config: &ClassifierConfig) -> Vec<BackendOperation> {
    let mut operations = vec![BackendOperation::Dense];
    if config.objective == ClassificationObjective::MulticlassLogLoss {
        operations.push(BackendOperation::CsrRowSoftmax);
    }
    if config.graph_leaf_smoothing.is_some() {
        operations.push(BackendOperation::CsrDiffusion);
    }
    operations
}

fn accelerated_row_softmax(
    backend: &BackendSelection,
    logits: &[f64],
    row_count: usize,
    width: usize,
) -> Result<Option<Vec<f32>>> {
    if backend.selected == "cpu"
        || row_count.saturating_mul(width) < CLASSIFIER_SOFTMAX_DISPATCH_MIN_OPS
    {
        return Ok(None);
    }
    let indptr = (0..=row_count)
        .map(|row| {
            u32::try_from(row.saturating_mul(width)).map_err(|_| {
                CartoBoostError::InvalidInput(
                    "classifier softmax tensor exceeds u32 indexing".to_string(),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let logits = logits.iter().map(|value| *value as f32).collect::<Vec<_>>();
    backend_csr_row_softmax_f32(backend, &indptr, &logits)
        .map(Some)
        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))
}

fn multiclass_derivatives_with_backend(
    objective: ClassificationObjective,
    targets: &[f64],
    raw_predictions: &[f64],
    weights: &[f64],
    class_count: usize,
    backend: &BackendSelection,
) -> Result<Option<Vec<GradientPair>>> {
    if objective != ClassificationObjective::MulticlassLogLoss {
        return Ok(None);
    }
    let Some(probabilities) =
        accelerated_row_softmax(backend, raw_predictions, targets.len(), class_count)?
    else {
        return Ok(None);
    };
    Ok(Some(
        probabilities
            .chunks_exact(class_count)
            .zip(targets)
            .zip(weights)
            .flat_map(|((row, target), weight)| {
                let target = *target as usize;
                row.iter().enumerate().map(move |(output, probability)| {
                    let probability = f64::from(*probability);
                    let label = if output == target { 1.0 } else { 0.0 };
                    GradientPair {
                        gradient: weight * (probability - label),
                        hessian: weight * probability * (1.0 - probability),
                    }
                })
            })
            .collect(),
    ))
}

fn multiclass_metric_with_backend(
    objective: ClassificationObjective,
    targets: &[f64],
    raw_predictions: &[f64],
    class_count: usize,
    backend: &BackendSelection,
) -> Result<Option<MetricValue>> {
    if objective != ClassificationObjective::MulticlassLogLoss {
        return Ok(None);
    }
    let Some(probabilities) =
        accelerated_row_softmax(backend, raw_predictions, targets.len(), class_count)?
    else {
        return Ok(None);
    };
    let loss = targets
        .iter()
        .zip(probabilities.chunks_exact(class_count))
        .map(|(target, row)| -f64::from(row[*target as usize]).clamp(1.0e-15, 1.0).ln())
        .sum::<f64>()
        / targets.len().max(1) as f64;
    Ok(Some(MetricValue {
        name: "logloss",
        value: loss,
    }))
}

fn validate_graph_leaf_smoothing(config: &ClassifierConfig, row_count: usize) -> Result<()> {
    if let Some(smoothing) = &config.graph_leaf_smoothing {
        smoothing.validate_row_count(row_count)?;
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

fn validate_classifier_config(config: &ClassifierConfig, feature_count: usize) -> Result<()> {
    if config.n_estimators == 0 {
        return Err(CartoBoostError::InvalidInput(
            "n_estimators must be positive".to_string(),
        ));
    }
    if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "learning_rate must be positive and finite".to_string(),
        ));
    }
    if config.min_samples_leaf == 0 {
        return Err(CartoBoostError::InvalidInput(
            "min_samples_leaf must be positive".to_string(),
        ));
    }
    if !config.min_gain.is_finite() || config.min_gain < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "min_gain must be finite and non-negative".to_string(),
        ));
    }
    if !config.constant_lambda_l2.is_finite() || config.constant_lambda_l2 < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "constant_lambda_l2 must be finite and non-negative".to_string(),
        ));
    }
    if !config.linear_lambda_l2.is_finite() || config.linear_lambda_l2 < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "linear_lambda_l2 must be finite and non-negative".to_string(),
        ));
    }
    if !config.fuzzy_bandwidth.is_finite() || config.fuzzy_bandwidth < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "fuzzy_bandwidth must be finite and non-negative".to_string(),
        ));
    }
    if config
        .linear_leaf_features
        .iter()
        .any(|feature| *feature >= feature_count)
    {
        return Err(CartoBoostError::InvalidInput(
            "linear_leaf_features contains an out-of-range feature index".to_string(),
        ));
    }
    if !config.class_weights.is_empty()
        && config
            .class_weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err(CartoBoostError::InvalidInput(
            "class_weights must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn resolve_class_count(config: &ClassifierConfig, targets: &[f64]) -> Result<usize> {
    let observed = targets
        .iter()
        .copied()
        .filter(|target| target.is_finite())
        .map(|target| target as usize)
        .max()
        .map_or(0, |max_class| max_class + 1);
    let class_count = config.class_count.max(observed);
    match config.objective {
        ClassificationObjective::BinaryLogLoss => {
            if class_count != 2 {
                return Err(CartoBoostError::InvalidInput(
                    "binary_logloss requires exactly two classes".to_string(),
                ));
            }
        }
        ClassificationObjective::MulticlassLogLoss => {
            if class_count < 2 {
                return Err(CartoBoostError::InvalidInput(
                    "multiclass_logloss requires at least two classes".to_string(),
                ));
            }
        }
    }
    if !config.class_weights.is_empty() && config.class_weights.len() != class_count {
        return Err(CartoBoostError::InvalidInput(format!(
            "class_weights has length {}, but classifier has {class_count} classes",
            config.class_weights.len()
        )));
    }
    Ok(class_count)
}

fn validate_class_targets(targets: &[f64], class_count: usize) -> Result<()> {
    if targets.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "classifier targets must not be empty".to_string(),
        ));
    }
    if targets
        .iter()
        .any(|target| !target.is_finite() || target.fract() != 0.0 || *target < 0.0)
    {
        return Err(CartoBoostError::InvalidInput(
            "classifier targets must be finite non-negative integer class ids".to_string(),
        ));
    }
    if targets.iter().any(|target| *target as usize >= class_count) {
        return Err(CartoBoostError::InvalidInput(
            "classifier target class id is out of range".to_string(),
        ));
    }
    let mut seen = vec![false; class_count];
    for target in targets {
        seen[*target as usize] = true;
    }
    if seen.iter().filter(|value| **value).count() < 2 {
        return Err(CartoBoostError::InvalidInput(
            "classifier targets must contain at least two classes".to_string(),
        ));
    }
    Ok(())
}

fn apply_class_weights(
    weights: &[f64],
    targets: &[f64],
    class_weights: &[f64],
) -> Result<Vec<f64>> {
    if class_weights.is_empty() {
        return Ok(weights.to_vec());
    }
    Ok(weights
        .iter()
        .zip(targets)
        .map(|(weight, target)| weight * class_weights[*target as usize])
        .collect())
}

fn make_objective(
    objective: ClassificationObjective,
    class_count: usize,
) -> Result<Box<dyn Objective + Send + Sync>> {
    match objective {
        ClassificationObjective::BinaryLogLoss => Ok(Box::new(BinaryLogLossObjective)),
        ClassificationObjective::MulticlassLogLoss => {
            Ok(Box::new(MulticlassLogLossObjective::new(class_count)?))
        }
    }
}

fn transform_margin_row(transform: PredictionTransformKind, margins: &[f64]) -> Vec<f64> {
    match transform {
        PredictionTransformKind::Identity => margins.to_vec(),
        PredictionTransformKind::Sigmoid => {
            let positive = sigmoid(margins[0]);
            vec![1.0 - positive, positive]
        }
        PredictionTransformKind::Softmax => {
            let max = margins.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_values = margins
                .iter()
                .map(|margin| (margin - max).exp())
                .collect::<Vec<_>>();
            let total = exp_values.iter().sum::<f64>();
            exp_values.into_iter().map(|value| value / total).collect()
        }
    }
}

fn sigmoid(raw_prediction: f64) -> f64 {
    if raw_prediction >= 0.0 {
        1.0 / (1.0 + (-raw_prediction).exp())
    } else {
        let exp_value = raw_prediction.exp();
        exp_value / (1.0 + exp_value)
    }
}

impl From<&ClassifierConfig> for ClassifierTrainingConfigMetadata {
    fn from(config: &ClassifierConfig) -> Self {
        Self {
            backend: None,
            n_estimators: config.n_estimators,
            learning_rate: config.learning_rate,
            max_depth: config.max_depth,
            min_samples_leaf: config.min_samples_leaf,
            min_gain: config.min_gain,
            splitters: config.splitters.clone(),
            leaf_predictor: config.leaf_predictor.clone(),
            linear_leaf_features: config.linear_leaf_features.clone(),
            linear_lambda_l2: config.linear_lambda_l2,
            constant_lambda_l2: config.constant_lambda_l2,
            fuzzy: config.fuzzy,
            fuzzy_bandwidth: config.fuzzy_bandwidth,
            fuzzy_kernel: config.fuzzy_kernel,
            objective: config.objective,
            class_count: config.class_count,
            class_weights: config.class_weights.clone(),
            graph_leaf_smoothing: config.graph_leaf_smoothing.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph_regularization::{CsrGraph, GraphLeafSmoothing};
    use crate::objectives::{
        LogisticBoostingObjective, ObjectiveTask, PredictionTransformKind, ProbabilityBooster,
    };

    #[test]
    fn binary_classifier_learns_separable_boundary_and_roundtrips() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let classifier = Classifier::new(ClassifierConfig {
            n_estimators: 8,
            learning_rate: 0.5,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..ClassifierConfig::default()
        });

        let model = classifier.fit(&x, &y, None).unwrap();
        let predictions = model.predict(&x).unwrap();
        let probabilities = model.predict_proba(&x).unwrap();

        assert_eq!(predictions, y);
        assert!(probabilities[0][1] < probabilities[3][1]);
        for backend in cartoboost_accelerator::available_backends()
            .into_iter()
            .filter(|name| name != "cpu")
        {
            let accelerated = model
                .predict_proba_with_backend(&x, Some(&backend))
                .unwrap_or_else(|error| panic!("{backend} classifier inference failed: {error}"));
            for (actual, expected) in accelerated
                .iter()
                .flatten()
                .zip(probabilities.iter().flatten())
            {
                assert!((actual - expected).abs() <= 1.0e-4);
            }
        }

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("classifier.json");
        model.save(&path).unwrap();
        let loaded = ClassifierModel::load(&path).unwrap();

        assert_eq!(loaded.predict(&x).unwrap(), y);
    }

    #[test]
    fn probability_booster_alias_returns_bounded_binary_probabilities() {
        let objective = LogisticBoostingObjective::default();
        assert_eq!(objective.task(), ObjectiveTask::BinaryClassification);
        assert_eq!(
            objective.prediction_transform(),
            PredictionTransformKind::Sigmoid
        );

        let x = Dataset::from_rows(vec![
            vec![0.0],
            vec![0.5],
            vec![1.0],
            vec![2.0],
            vec![2.5],
            vec![3.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let booster = ProbabilityBooster::new(ClassifierConfig {
            n_estimators: 10,
            learning_rate: 0.4,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            ..ClassifierConfig::default()
        });

        let model = booster.fit(&x, &y, None).unwrap();
        let probabilities = model.predict_proba(&x).unwrap();

        assert_eq!(probabilities.len(), y.len());
        assert!(probabilities.iter().all(|row| {
            row.len() == 2
                && row
                    .iter()
                    .all(|probability| (0.0..=1.0).contains(probability))
                && (row[0] + row[1] - 1.0).abs() < 1.0e-12
        }));
        assert!(probabilities[0][1] < probabilities[5][1]);
    }

    #[test]
    fn multiclass_classifier_returns_row_probabilities() {
        let x = Dataset::from_rows(vec![
            vec![0.0],
            vec![0.2],
            vec![2.0],
            vec![2.2],
            vec![4.0],
            vec![4.2],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let classifier = Classifier::new(ClassifierConfig {
            n_estimators: 12,
            learning_rate: 0.4,
            max_depth: 2,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            objective: ClassificationObjective::MulticlassLogLoss,
            class_count: 3,
            ..ClassifierConfig::default()
        });

        let model = classifier.fit(&x, &y, None).unwrap();
        let probabilities = model.predict_proba(&x).unwrap();

        assert_eq!(probabilities.len(), y.len());
        assert!(probabilities
            .iter()
            .all(|row| (row.iter().sum::<f64>() - 1.0).abs() < 1.0e-12));
        assert_eq!(model.predict(&x).unwrap(), y);
    }

    #[test]
    fn large_multiclass_softmax_matches_cpu_on_every_backend() {
        let training_x = Dataset::from_rows(vec![
            vec![0.0],
            vec![0.2],
            vec![2.0],
            vec![2.2],
            vec![4.0],
            vec![4.2],
        ])
        .unwrap();
        let classifier = Classifier::new(ClassifierConfig {
            n_estimators: 6,
            learning_rate: 0.4,
            max_depth: 2,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            objective: ClassificationObjective::MulticlassLogLoss,
            class_count: 3,
            ..ClassifierConfig::default()
        });
        let model = classifier
            .fit(&training_x, &[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], None)
            .unwrap();
        let inference_x = Dataset::from_rows(
            (0..6_000)
                .map(|row| vec![(row % 43) as f64 * 0.1])
                .collect(),
        )
        .unwrap();
        let expected = model
            .predict_proba_with_backend(&inference_x, Some("cpu"))
            .unwrap();
        for backend_name in cartoboost_accelerator::available_backends() {
            let actual = model
                .predict_proba_with_backend(&inference_x, Some(&backend_name))
                .unwrap_or_else(|error| panic!("{backend_name} multiclass inference: {error}"));
            for (actual_row, expected_row) in actual.iter().zip(&expected) {
                assert!((actual_row.iter().sum::<f64>() - 1.0).abs() <= 2.0e-5);
                for (actual, expected) in actual_row.iter().zip(expected_row) {
                    assert!(
                        (actual - expected).abs() <= 2.0e-4,
                        "{backend_name}: expected {expected}, got {actual}"
                    );
                }
            }
        }
    }

    #[test]
    fn large_multiclass_training_matches_cpu_on_every_backend() {
        let row_count = 6_000;
        let x = Dataset::from_rows(
            (0..row_count)
                .map(|row| {
                    let class = row % 3;
                    vec![class as f64 * 2.0 + (row % 17) as f64 * 0.001]
                })
                .collect(),
        )
        .unwrap();
        let y = (0..row_count)
            .map(|row| (row % 3) as f64)
            .collect::<Vec<_>>();
        let config = ClassifierConfig {
            n_estimators: 2,
            learning_rate: 0.4,
            max_depth: 2,
            min_samples_leaf: 8,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            objective: ClassificationObjective::MulticlassLogLoss,
            class_count: 3,
            ..ClassifierConfig::default()
        };
        let expected = Classifier::new_with_backend(config.clone(), Some("cpu"))
            .unwrap()
            .fit(&x, &y, None)
            .unwrap();
        let expected_predictions = expected.predict(&x).unwrap();
        for backend_name in cartoboost_accelerator::available_backends() {
            let actual = Classifier::new_with_backend(config.clone(), Some(&backend_name))
                .unwrap_or_else(|error| panic!("{backend_name} classifier selection: {error}"))
                .fit(&x, &y, None)
                .unwrap_or_else(|error| panic!("{backend_name} multiclass fit: {error}"));
            assert_eq!(actual.predict(&x).unwrap(), expected_predictions);
            for (actual_metric, expected_metric) in actual
                .training_history
                .iter()
                .zip(&expected.training_history)
            {
                assert!(
                    (actual_metric.value - expected_metric.value).abs() <= 2.0e-4,
                    "{backend_name}: expected {}, got {}",
                    expected_metric.value,
                    actual_metric.value
                );
            }
        }
    }

    #[test]
    fn classifier_applies_and_serializes_graph_leaf_smoothing() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let graph = CsrGraph::new(4, vec![0, 1, 2, 3, 4], vec![1, 0, 3, 2], vec![1.0; 4]).unwrap();
        let classifier = Classifier::new_with_backend(
            ClassifierConfig {
                n_estimators: 1,
                max_depth: 1,
                min_samples_leaf: 1,
                graph_leaf_smoothing: Some(GraphLeafSmoothing::new(graph, 0.5, 2).unwrap()),
                ..ClassifierConfig::default()
            },
            Some("cpu"),
        )
        .unwrap();
        let model = classifier.fit(&x, &[0.0, 0.0, 1.0, 1.0], None).unwrap();

        assert!(model
            .training_config
            .unwrap()
            .graph_leaf_smoothing
            .is_some());
    }
}
