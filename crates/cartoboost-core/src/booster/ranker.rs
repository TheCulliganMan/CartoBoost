use crate::data::{validate_weights, Dataset};
use crate::loss::LossConfig;
use crate::objectives::{LambdaRankObjective, Objective, PairwiseLogitObjective};
use crate::tree::{
    FuzzyKernel, LeafPredictorKind, ModelMetadata, SplitterKind, TrainingMetric, Tree, TreeBuilder,
};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub const RANKER_MODEL_ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RankingObjective {
    PairwiseLogit,
    #[default]
    LambdaRank,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankerConfig {
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
    pub objective: RankingObjective,
}

#[derive(Debug, Clone)]
pub struct Ranker {
    pub config: RankerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankerTrainingConfigMetadata {
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
    pub objective: RankingObjective,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RankingMetricSet {
    pub ndcg: f64,
    pub map: f64,
    pub mrr: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankerModel {
    pub artifact_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<ModelMetadata>,
    pub objective: RankingObjective,
    pub init_score: f64,
    pub learning_rate: f64,
    pub feature_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_schema: Option<crate::data::FeatureSchema>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_config: Option<RankerTrainingConfigMetadata>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub training_history: Vec<TrainingMetric>,
    pub trees: Vec<Tree>,
}

impl Default for RankerConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.05,
            max_depth: 4,
            min_samples_leaf: 20,
            min_gain: 1.0e-8,
            splitters: vec![SplitterKind::Auto],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            objective: RankingObjective::LambdaRank,
        }
    }
}

impl Ranker {
    pub fn new(config: RankerConfig) -> Self {
        Self { config }
    }

    pub fn fit(
        &self,
        x: &Dataset,
        y: &[f64],
        groups: &[usize],
        sample_weight: Option<&[f64]>,
    ) -> Result<RankerModel> {
        validate_ranker_config(&self.config, x.n_cols())?;
        if x.n_rows() != y.len() {
            return Err(CartoBoostError::InvalidInput(
                "X row count must match y length".to_string(),
            ));
        }
        validate_targets(y)?;
        validate_groups(groups, y.len())?;
        let weights = validate_weights(sample_weight, y.len())?;
        let objective = make_objective(self.config.objective);
        let init_score = objective.initial_margin(y, Some(&weights))?[0];
        let mut scores = vec![init_score; y.len()];
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
            let derivative_pairs =
                objective.gradients_hessians(y, &scores, Some(&weights), Some(groups))?;
            let mut targets = vec![0.0; y.len()];
            let mut hessian_weights = vec![1.0e-12; y.len()];
            targets
                .par_iter_mut()
                .zip(hessian_weights.par_iter_mut())
                .enumerate()
                .for_each(|(row, (target, hessian_weight))| {
                    let pair = derivative_pairs[row];
                    let hessian = pair.hessian.max(1.0e-12);
                    *target = -pair.gradient / hessian;
                    *hessian_weight = hessian;
                });
            let tree = builder.fit_in_context(x, &targets, &hessian_weights, &fit_context);
            scores.par_iter_mut().enumerate().for_each(|(row, score)| {
                *score += self.config.learning_rate * tree.predict_dataset_row(x, row);
            });
            trees.push(tree);
            let metrics = ranking_metrics(y, &scores, groups)?;
            training_history.push(TrainingMetric {
                iteration: iteration + 1,
                name: "train/ndcg".to_string(),
                value: metrics.ndcg,
            });
            training_history.push(TrainingMetric {
                iteration: iteration + 1,
                name: "train/map".to_string(),
                value: metrics.map,
            });
            training_history.push(TrainingMetric {
                iteration: iteration + 1,
                name: "train/mrr".to_string(),
                value: metrics.mrr,
            });
        }

        Ok(RankerModel {
            artifact_version: RANKER_MODEL_ARTIFACT_VERSION,
            metadata: Some(crate::tree::Model::default_metadata()),
            objective: self.config.objective,
            init_score,
            learning_rate: self.config.learning_rate,
            feature_count: x.n_cols(),
            feature_schema: Some(x.feature_schema_or_default()),
            training_config: Some(RankerTrainingConfigMetadata {
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
            }),
            training_history,
            trees,
        })
    }
}

impl RankerModel {
    pub fn predict(&self, x: &Dataset) -> Result<Vec<f64>> {
        self.validate_dataset(x)?;
        Ok((0..x.n_rows())
            .into_par_iter()
            .map(|row| self.predict_dataset_row(x, row))
            .collect())
    }

    pub fn metrics(&self, x: &Dataset, y: &[f64], groups: &[usize]) -> Result<RankingMetricSet> {
        let scores = self.predict(x)?;
        ranking_metrics(y, &scores, groups)
    }

    pub fn requires_sparse_sets(&self) -> bool {
        self.trees.iter().any(Tree::contains_sparse_list_split)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialize::save_json(self, path)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model: Self = crate::serialize::load_json(path)?;
        if model.artifact_version != RANKER_MODEL_ARTIFACT_VERSION {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported ranker model artifact version {}",
                model.artifact_version
            )));
        }
        Ok(model)
    }

    fn predict_dataset_row(&self, x: &Dataset, row: usize) -> f64 {
        self.init_score
            + self
                .trees
                .iter()
                .map(|tree| self.learning_rate * tree.predict_dataset_row(x, row))
                .sum::<f64>()
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

pub fn ranking_metrics(y: &[f64], scores: &[f64], groups: &[usize]) -> Result<RankingMetricSet> {
    validate_targets(y)?;
    if scores.len() != y.len() {
        return Err(CartoBoostError::InvalidInput(
            "ranking scores length must match target length".to_string(),
        ));
    }
    if scores.iter().any(|score| !score.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "ranking scores must be finite".to_string(),
        ));
    }
    validate_groups(groups, y.len())?;
    Ok(RankingMetricSet {
        ndcg: ndcg_at_k(y, scores, groups, None)?,
        map: mean_average_precision(y, scores, groups)?,
        mrr: mean_reciprocal_rank(y, scores, groups)?,
    })
}

pub fn ndcg_at_k(y: &[f64], scores: &[f64], groups: &[usize], k: Option<usize>) -> Result<f64> {
    validate_targets(y)?;
    if scores.len() != y.len() {
        return Err(CartoBoostError::InvalidInput(
            "ranking scores length must match target length".to_string(),
        ));
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for range in group_ranges(groups, y.len())? {
        let limit = k.unwrap_or(range.len()).min(range.len());
        let mut predicted = range.clone().collect::<Vec<_>>();
        predicted.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]));
        let dcg = predicted
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(position, row)| ranking_gain(y[row]) * ranking_discount(position))
            .sum::<f64>();
        let mut ideal = range.collect::<Vec<_>>();
        ideal.sort_by(|left, right| y[*right].total_cmp(&y[*left]));
        let idcg = ideal
            .into_iter()
            .take(limit)
            .enumerate()
            .map(|(position, row)| ranking_gain(y[row]) * ranking_discount(position))
            .sum::<f64>();
        total += if idcg > 0.0 { dcg / idcg } else { 0.0 };
        count += 1;
    }
    Ok(if count == 0 {
        0.0
    } else {
        total / count as f64
    })
}

pub fn mean_average_precision(y: &[f64], scores: &[f64], groups: &[usize]) -> Result<f64> {
    validate_targets(y)?;
    if scores.len() != y.len() {
        return Err(CartoBoostError::InvalidInput(
            "ranking scores length must match target length".to_string(),
        ));
    }
    let mut total = 0.0;
    let mut group_count = 0usize;
    for range in group_ranges(groups, y.len())? {
        let mut predicted = range.clone().collect::<Vec<_>>();
        predicted.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]));
        let relevant_total = range.clone().filter(|row| y[*row] > 0.0).count();
        if relevant_total == 0 {
            group_count += 1;
            continue;
        }
        let mut relevant_seen = 0usize;
        let mut precision_sum = 0.0;
        for (position, row) in predicted.iter().enumerate() {
            if y[*row] > 0.0 {
                relevant_seen += 1;
                precision_sum += relevant_seen as f64 / (position + 1) as f64;
            }
        }
        total += precision_sum / relevant_total as f64;
        group_count += 1;
    }
    Ok(if group_count == 0 {
        0.0
    } else {
        total / group_count as f64
    })
}

pub fn mean_reciprocal_rank(y: &[f64], scores: &[f64], groups: &[usize]) -> Result<f64> {
    validate_targets(y)?;
    if scores.len() != y.len() {
        return Err(CartoBoostError::InvalidInput(
            "ranking scores length must match target length".to_string(),
        ));
    }
    let mut total = 0.0;
    let mut group_count = 0usize;
    for range in group_ranges(groups, y.len())? {
        let mut predicted = range.collect::<Vec<_>>();
        predicted.sort_by(|left, right| scores[*right].total_cmp(&scores[*left]));
        let reciprocal = predicted
            .iter()
            .position(|row| y[*row] > 0.0)
            .map_or(0.0, |position| 1.0 / (position + 1) as f64);
        total += reciprocal;
        group_count += 1;
    }
    Ok(if group_count == 0 {
        0.0
    } else {
        total / group_count as f64
    })
}

fn validate_ranker_config(config: &RankerConfig, feature_count: usize) -> Result<()> {
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
    Ok(())
}

fn validate_targets(targets: &[f64]) -> Result<()> {
    if targets.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "ranking targets must not be empty".to_string(),
        ));
    }
    if targets.iter().any(|target| !target.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "ranking targets must be finite".to_string(),
        ));
    }
    Ok(())
}

fn validate_groups(groups: &[usize], row_count: usize) -> Result<()> {
    if groups.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "ranking groups must not be empty".to_string(),
        ));
    }
    let mut total = 0usize;
    for &group_size in groups {
        if group_size == 0 {
            return Err(CartoBoostError::InvalidInput(
                "ranking group sizes must be positive".to_string(),
            ));
        }
        total += group_size;
    }
    if total != row_count {
        return Err(CartoBoostError::InvalidInput(format!(
            "ranking group sizes sum to {total}, but there are {row_count} rows"
        )));
    }
    Ok(())
}

fn group_ranges(groups: &[usize], row_count: usize) -> Result<Vec<std::ops::Range<usize>>> {
    validate_groups(groups, row_count)?;
    let mut start = 0usize;
    let mut ranges = Vec::with_capacity(groups.len());
    for &group_size in groups {
        let end = start + group_size;
        ranges.push(start..end);
        start = end;
    }
    Ok(ranges)
}

fn make_objective(objective: RankingObjective) -> Box<dyn Objective + Send + Sync> {
    match objective {
        RankingObjective::PairwiseLogit => Box::new(PairwiseLogitObjective),
        RankingObjective::LambdaRank => Box::new(LambdaRankObjective),
    }
}

fn ranking_gain(label: f64) -> f64 {
    2.0_f64.powf(label.max(0.0)) - 1.0
}

fn ranking_discount(position: usize) -> f64 {
    1.0 / ((position + 2) as f64).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranker_improves_ndcg_over_reversed_scores_and_roundtrips() {
        let x = Dataset::from_rows(vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ])
        .unwrap();
        let y = vec![0.0, 1.0, 3.0, 0.0, 2.0, 4.0];
        let groups = vec![3, 3];
        let ranker = Ranker::new(RankerConfig {
            n_estimators: 8,
            learning_rate: 0.4,
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            objective: RankingObjective::LambdaRank,
            ..RankerConfig::default()
        });

        let model = ranker.fit(&x, &y, &groups, None).unwrap();
        let metrics = model.metrics(&x, &y, &groups).unwrap();
        let reversed = vec![3.0, 2.0, 1.0, 3.0, 2.0, 1.0];
        let reversed_ndcg = ndcg_at_k(&y, &reversed, &groups, None).unwrap();

        assert!(metrics.ndcg > reversed_ndcg);
        assert!(metrics.map > 0.9);
        assert!(metrics.mrr > 0.9);

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("ranker.json");
        model.save(&path).unwrap();
        let loaded = RankerModel::load(&path).unwrap();

        assert_eq!(loaded.predict(&x).unwrap().len(), y.len());
    }
}
