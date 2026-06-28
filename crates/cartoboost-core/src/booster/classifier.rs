use crate::data::{validate_weights, Dataset};
use crate::loss::LossConfig;
use crate::objectives::{
    BinaryLogLossObjective, MulticlassLogLossObjective, Objective, PredictionTransformKind,
};
use crate::tree::{
    FuzzyKernel, LeafPredictorKind, ModelMetadata, SplitterKind, TrainingMetric, Tree, TreeBuilder,
};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const CLASSIFIER_MODEL_ARTIFACT_VERSION: u32 = 1;

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
}

#[derive(Debug, Clone)]
pub struct Classifier {
    pub config: ClassifierConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierTrainingConfigMetadata {
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
        }
    }
}

impl Classifier {
    pub fn new(config: ClassifierConfig) -> Self {
        Self { config }
    }

    pub fn fit(
        &self,
        x: &Dataset,
        y: &[f64],
        sample_weight: Option<&[f64]>,
    ) -> Result<ClassifierModel> {
        validate_classifier_config(&self.config, x.n_cols())?;
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
            let derivative_pairs = objective.gradients_hessians(
                y,
                &raw_predictions,
                Some(&effective_weights),
                None,
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
                let tree = builder.fit_in_context(x, &targets, &hessian_weights, &fit_context);
                raw_predictions
                    .par_chunks_mut(output_dimension)
                    .enumerate()
                    .for_each(|(row, margins)| {
                        margins[output] +=
                            self.config.learning_rate * tree.predict_dataset_row(x, row);
                    });
                iteration_trees.push(tree);
            }
            trees.push(iteration_trees);
            let metric = objective.default_metric(y, &raw_predictions, None)?;
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
        self.validate_dataset(x)?;
        Ok((0..x.n_rows())
            .into_par_iter()
            .map(|row| self.raw_predict_dataset_row(x, row))
            .collect())
    }

    pub fn predict_proba(&self, x: &Dataset) -> Result<Vec<Vec<f64>>> {
        let objective = make_objective(self.objective, self.class_values.len())?;
        let transform = objective.prediction_transform();
        let margins = self.decision_function(x)?;
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
