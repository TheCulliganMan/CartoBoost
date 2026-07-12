use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceFrame {
    pub series: Vec<SequenceSeries>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceSeries {
    pub series_id: String,
    pub rows: Vec<SequenceRow>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceRow {
    pub row_id: String,
    pub position: f64,
    pub target: Option<f64>,
    pub reference_axis: Option<f64>,
    pub reference_signal: Option<f64>,
    pub auxiliary_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KnownPrefix {
    pub row_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionMask {
    pub row_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferenceSignal {
    pub axis: Vec<f64>,
    pub signal: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SequenceStateSpaceConfig {
    pub initial_axis_variance: f64,
    pub initial_rate_variance: f64,
    pub axis_process_variance: f64,
    pub rate_process_variance: f64,
    pub signal_observation_variance: f64,
    pub rate_observation_variance: Option<f64>,
    pub sigma_point_alpha: f64,
    pub sigma_point_beta: f64,
    pub sigma_point_kappa: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceKalmanPoint {
    pub row_id: String,
    pub position: f64,
    pub observed: Option<f64>,
    pub predicted_axis: f64,
    pub predicted_rate: f64,
    pub predicted_signal: f64,
    pub covariance: [[f64; 2]; 2],
    pub innovation: Option<f64>,
    pub innovation_variance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceKalmanResult {
    pub points: Vec<SequenceKalmanPoint>,
    pub log_likelihood: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ReferencePathConfig {
    pub emission_scale: f64,
    pub student_t_df: f64,
    pub transition_penalty: f64,
    pub smoothness_penalty: f64,
    pub start_axis: Option<f64>,
    pub start_penalty: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferencePathPoint {
    pub row_id: String,
    pub position: f64,
    pub axis: f64,
    pub signal: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReferencePathResult {
    pub points: Vec<ReferencePathPoint>,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceCandidatePrediction {
    pub series_id: String,
    pub row_id: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceCandidate {
    pub name: String,
    pub predictions: Vec<SequenceCandidatePrediction>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceBlendPrediction {
    pub series_id: String,
    pub row_id: String,
    pub value: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceCandidateEnsemble {
    pub weights: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceGroupPrediction {
    pub group_id: String,
    pub row_id: String,
    pub actual: f64,
    pub prediction: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SequenceGroupMetric {
    pub count: usize,
    pub rmse: f64,
    pub mae: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceOofCandidateRow {
    pub group_id: String,
    pub row_id: String,
    pub actual: f64,
    pub candidate_predictions: BTreeMap<String, f64>,
    pub train_group_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceOofFold {
    pub validation_group_id: String,
    pub train_group_ids: Vec<String>,
    pub actuals: Vec<SequenceCandidatePrediction>,
    pub candidates: Vec<SequenceCandidate>,
}

impl Default for SequenceStateSpaceConfig {
    fn default() -> Self {
        Self {
            initial_axis_variance: 1.0,
            initial_rate_variance: 1.0,
            axis_process_variance: 1e-3,
            rate_process_variance: 1e-3,
            signal_observation_variance: 1e-2,
            rate_observation_variance: None,
            sigma_point_alpha: 1e-3,
            sigma_point_beta: 2.0,
            sigma_point_kappa: 0.0,
        }
    }
}

impl Default for ReferencePathConfig {
    fn default() -> Self {
        Self {
            emission_scale: 1.0,
            student_t_df: 4.0,
            transition_penalty: 0.01,
            smoothness_penalty: 0.0,
            start_axis: None,
            start_penalty: 0.0,
        }
    }
}

impl SequenceFrame {
    pub fn new(series: Vec<SequenceSeries>) -> Result<Self> {
        let frame = Self { series };
        frame.validate()?;
        Ok(frame)
    }

    pub fn validate(&self) -> Result<()> {
        if self.series.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "sequence frame requires at least one series".to_string(),
            ));
        }
        let mut ids = BTreeSet::new();
        for series in &self.series {
            series.validate()?;
            if !ids.insert(series.series_id.as_str()) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate sequence series_id '{}'",
                    series.series_id
                )));
            }
        }
        Ok(())
    }
}

impl SequenceSeries {
    pub fn validate(&self) -> Result<KnownPrefix> {
        if self.series_id.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "sequence series_id must be non-empty".to_string(),
            ));
        }
        if self.rows.len() < 2 {
            return Err(CartoBoostError::InvalidInput(
                "sequence series requires at least two rows".to_string(),
            ));
        }
        let mut row_ids = BTreeSet::new();
        let mut prefix_count = 0usize;
        let mut saw_prediction = false;
        let mut previous_position = f64::NEG_INFINITY;
        for row in &self.rows {
            if row.row_id.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "sequence row_id must be non-empty".to_string(),
                ));
            }
            if !row_ids.insert(row.row_id.as_str()) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "duplicate sequence row_id '{}'",
                    row.row_id
                )));
            }
            if !row.position.is_finite() || row.position <= previous_position {
                return Err(CartoBoostError::InvalidInput(
                    "sequence positions must be finite and strictly ordered".to_string(),
                ));
            }
            previous_position = row.position;
            validate_optional_finite(row.target, "sequence targets")?;
            validate_optional_finite(row.reference_axis, "sequence reference axes")?;
            validate_optional_finite(row.reference_signal, "sequence reference signals")?;
            validate_optional_finite(row.auxiliary_rate, "sequence auxiliary rates")?;
            match row.target {
                Some(_) if saw_prediction => {
                    return Err(CartoBoostError::InvalidInput(
                        "target leakage: prediction rows must not contain targets after the known prefix"
                            .to_string(),
                    ));
                }
                Some(_) => prefix_count += 1,
                None => saw_prediction = true,
            }
        }
        if prefix_count == 0 {
            return Err(CartoBoostError::InvalidInput(
                "known prefix must be nonempty".to_string(),
            ));
        }
        if prefix_count == self.rows.len() {
            return Err(CartoBoostError::InvalidInput(
                "prediction suffix must be nonempty".to_string(),
            ));
        }
        Ok(KnownPrefix {
            row_count: prefix_count,
        })
    }

    pub fn prediction_mask(&self) -> Result<PredictionMask> {
        let prefix = self.validate()?;
        Ok(PredictionMask {
            row_ids: self.rows[prefix.row_count..]
                .iter()
                .map(|row| row.row_id.clone())
                .collect(),
        })
    }
}

impl ReferenceSignal {
    pub fn new(axis: Vec<f64>, signal: Vec<f64>) -> Result<Self> {
        let reference = Self { axis, signal };
        reference.validate()?;
        Ok(reference)
    }

    pub fn validate(&self) -> Result<()> {
        if self.axis.len() != self.signal.len() || self.axis.len() < 2 {
            return Err(CartoBoostError::InvalidInput(
                "reference signal requires matching axis/signal arrays with at least two points"
                    .to_string(),
            ));
        }
        let mut previous = f64::NEG_INFINITY;
        for (&axis, &signal) in self.axis.iter().zip(&self.signal) {
            if !axis.is_finite() || !signal.is_finite() {
                return Err(CartoBoostError::InvalidInput(
                    "reference axis and signal values must be finite".to_string(),
                ));
            }
            if axis <= previous {
                return Err(CartoBoostError::InvalidInput(
                    "reference axis must be strictly monotonic and deduplicated".to_string(),
                ));
            }
            previous = axis;
        }
        Ok(())
    }

    pub fn interpolate(&self, axis: f64) -> f64 {
        if axis <= self.axis[0] {
            return self.signal[0];
        }
        let last = self.axis.len() - 1;
        if axis >= self.axis[last] {
            return self.signal[last];
        }
        let upper = self.axis.partition_point(|value| *value < axis);
        let lower = upper - 1;
        let span = self.axis[upper] - self.axis[lower];
        let t = (axis - self.axis[lower]) / span;
        self.signal[lower] + t * (self.signal[upper] - self.signal[lower])
    }

    pub fn derivative(&self, axis: f64) -> f64 {
        if axis < self.axis[0] || axis > self.axis[self.axis.len() - 1] {
            return 0.0;
        }
        let upper = self.axis.partition_point(|value| *value < axis);
        let upper = upper.clamp(1, self.axis.len() - 1);
        let lower = upper - 1;
        (self.signal[upper] - self.signal[lower]) / (self.axis[upper] - self.axis[lower])
    }

    fn clamp_axis(&self, axis: f64) -> f64 {
        axis.clamp(self.axis[0], self.axis[self.axis.len() - 1])
    }
}

impl SequenceStateSpaceConfig {
    pub fn validate(&self) -> Result<()> {
        validate_positive(self.initial_axis_variance, "initial_axis_variance")?;
        validate_positive(self.initial_rate_variance, "initial_rate_variance")?;
        validate_nonnegative(self.axis_process_variance, "axis_process_variance")?;
        validate_nonnegative(self.rate_process_variance, "rate_process_variance")?;
        validate_positive(
            self.signal_observation_variance,
            "signal_observation_variance",
        )?;
        if let Some(value) = self.rate_observation_variance {
            validate_positive(value, "rate_observation_variance")?;
        }
        validate_positive(self.sigma_point_alpha, "sigma_point_alpha")?;
        validate_positive(self.sigma_point_beta, "sigma_point_beta")?;
        if !self.sigma_point_kappa.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "sigma_point_kappa must be finite".to_string(),
            ));
        }
        let sigma_scale = sigma_point_scale(*self);
        if !sigma_scale.is_finite() || sigma_scale <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "sigma-point scale alpha^2 * (state_dimension + kappa) must be finite and positive"
                    .to_string(),
            ));
        }
        Ok(())
    }
}

impl ReferencePathConfig {
    pub fn validate(&self) -> Result<()> {
        validate_positive(self.emission_scale, "emission_scale")?;
        validate_positive(self.student_t_df, "student_t_df")?;
        validate_nonnegative(self.transition_penalty, "transition_penalty")?;
        validate_nonnegative(self.smoothness_penalty, "smoothness_penalty")?;
        validate_nonnegative(self.start_penalty, "start_penalty")?;
        validate_optional_finite(self.start_axis, "start_axis")?;
        Ok(())
    }
}

pub fn forward_ekf(
    series: &SequenceSeries,
    reference: &ReferenceSignal,
    config: SequenceStateSpaceConfig,
) -> Result<SequenceKalmanResult> {
    run_extended_kalman(series, reference, config, false)
}

pub fn ukf_reference(
    series: &SequenceSeries,
    reference: &ReferenceSignal,
    config: SequenceStateSpaceConfig,
) -> Result<SequenceKalmanResult> {
    run_unscented_kalman(series, reference, config)
}

pub fn rts_smoother(
    series: &SequenceSeries,
    reference: &ReferenceSignal,
    config: SequenceStateSpaceConfig,
) -> Result<SequenceKalmanResult> {
    run_extended_kalman(series, reference, config, true)
}

pub fn missing_target_continuation(
    series: &SequenceSeries,
    reference: &ReferenceSignal,
    config: SequenceStateSpaceConfig,
) -> Result<Vec<SequenceKalmanPoint>> {
    let prefix = series.validate()?;
    let result = forward_ekf(series, reference, config)?;
    Ok(result.points[prefix.row_count..].to_vec())
}

pub fn reference_path_viterbi(
    series: &SequenceSeries,
    reference: &ReferenceSignal,
    config: ReferencePathConfig,
) -> Result<ReferencePathResult> {
    series.validate()?;
    reference.validate()?;
    config.validate()?;
    let n = series.rows.len();
    let m = reference.axis.len();
    let mut dp = vec![vec![f64::INFINITY; m]; n];
    let mut prev = vec![vec![0usize; m]; n];
    for (state, score) in dp[0].iter_mut().enumerate() {
        *score = emission_cost(series.rows[0].target, reference.signal[state], config)
            + start_cost(reference.axis[state], config);
    }
    for t in 1..n {
        let previous_scores = dp[t - 1].clone();
        for (state, score_slot) in dp[t].iter_mut().enumerate() {
            let emission = emission_cost(series.rows[t].target, reference.signal[state], config);
            let mut best_score = f64::INFINITY;
            let mut best_prev = 0usize;
            for (prior, previous_score) in previous_scores.iter().enumerate() {
                let prior_prior = (t > 1).then_some(prev[t - 1][prior]);
                let score = previous_score
                    + transition_cost(reference, prior, state, prior_prior, config)
                    + emission;
                if score < best_score {
                    best_score = score;
                    best_prev = prior;
                }
            }
            *score_slot = best_score;
            prev[t][state] = best_prev;
        }
    }
    let mut state = (0..m)
        .min_by(|&a, &b| dp[n - 1][a].total_cmp(&dp[n - 1][b]))
        .expect("non-empty reference states");
    let score = dp[n - 1][state];
    let mut states = vec![0usize; n];
    for t in (0..n).rev() {
        states[t] = state;
        if t > 0 {
            state = prev[t][state];
        }
    }
    Ok(ReferencePathResult {
        points: states
            .into_iter()
            .zip(&series.rows)
            .map(|(state, row)| ReferencePathPoint {
                row_id: row.row_id.clone(),
                position: row.position,
                axis: reference.axis[state],
                signal: reference.signal[state],
            })
            .collect(),
        score,
    })
}

pub fn reference_path_posterior_mean(
    series: &SequenceSeries,
    reference: &ReferenceSignal,
    config: ReferencePathConfig,
) -> Result<ReferencePathResult> {
    series.validate()?;
    reference.validate()?;
    config.validate()?;
    let n = series.rows.len();
    let m = reference.axis.len();
    let mut log_alpha = vec![vec![f64::NEG_INFINITY; m]; n];
    let mut log_beta = vec![vec![0.0; m]; n];
    for (state, log_alpha_value) in log_alpha[0].iter_mut().enumerate() {
        *log_alpha_value = -emission_cost(series.rows[0].target, reference.signal[state], config)
            - start_cost(reference.axis[state], config);
    }
    normalize_log_row(&mut log_alpha[0]);
    for t in 1..n {
        for state in 0..m {
            let terms = (0..m)
                .map(|prior| {
                    log_alpha[t - 1][prior] - transition_cost(reference, prior, state, None, config)
                })
                .collect::<Vec<_>>();
            log_alpha[t][state] = log_sum_exp(&terms)
                - emission_cost(series.rows[t].target, reference.signal[state], config);
        }
        normalize_log_row(&mut log_alpha[t]);
    }
    for t in (0..n - 1).rev() {
        for state in 0..m {
            let terms = (0..m)
                .map(|next| {
                    log_beta[t + 1][next]
                        - transition_cost(reference, state, next, None, config)
                        - emission_cost(series.rows[t + 1].target, reference.signal[next], config)
                })
                .collect::<Vec<_>>();
            log_beta[t][state] = log_sum_exp(&terms);
        }
        normalize_log_row(&mut log_beta[t]);
    }
    let points = (0..n)
        .map(|t| {
            let mut posterior = (0..m)
                .map(|state| log_alpha[t][state] + log_beta[t][state])
                .collect::<Vec<_>>();
            normalize_log_row(&mut posterior);
            let probs = posterior.iter().map(|v| v.exp()).collect::<Vec<_>>();
            let axis = probs
                .iter()
                .zip(&reference.axis)
                .map(|(prob, axis)| prob * axis)
                .sum::<f64>();
            ReferencePathPoint {
                row_id: series.rows[t].row_id.clone(),
                position: series.rows[t].position,
                axis: reference.clamp_axis(axis),
                signal: reference.interpolate(axis),
            }
        })
        .collect();
    Ok(ReferencePathResult { points, score: 0.0 })
}

impl SequenceCandidateEnsemble {
    pub fn fixed(weights: BTreeMap<String, f64>) -> Result<Self> {
        validate_weights(&weights)?;
        Ok(Self { weights })
    }

    pub fn validation_derived(
        candidates: &[SequenceCandidate],
        actuals: &[SequenceCandidatePrediction],
    ) -> Result<Self> {
        let actual_map = prediction_map(actuals)?;
        let mut raw = BTreeMap::new();
        for candidate in candidates {
            let candidate_map = prediction_map(&candidate.predictions)?;
            if candidate_map.keys().collect::<Vec<_>>() != actual_map.keys().collect::<Vec<_>>() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "candidate '{}' is not aligned to validation actual rows",
                    candidate.name
                )));
            }
            let mse = candidate_map
                .iter()
                .map(|(key, pred)| {
                    let err = pred - actual_map[key];
                    err * err
                })
                .sum::<f64>()
                / candidate_map.len() as f64;
            raw.insert(candidate.name.clone(), 1.0 / mse.max(1e-12));
        }
        normalize_weight_map(raw)
    }

    pub fn constrained_nonnegative_linear_blend(
        candidates: &[SequenceCandidate],
        actuals: &[SequenceCandidatePrediction],
    ) -> Result<Self> {
        let mut ensemble = Self::validation_derived(candidates, actuals)?;
        for _ in 0..200 {
            let gradient = blend_gradient(candidates, actuals, &ensemble.weights)?;
            for (name, grad) in gradient {
                let value = (ensemble.weights[&name] - 0.05 * grad).max(0.0);
                ensemble.weights.insert(name, value);
            }
            ensemble = normalize_weight_map(ensemble.weights)?;
        }
        Ok(ensemble)
    }

    pub fn predict(
        &self,
        candidates: &[SequenceCandidate],
    ) -> Result<Vec<SequenceBlendPrediction>> {
        validate_weights(&self.weights)?;
        let mut expected_keys: Option<Vec<(String, String)>> = None;
        let mut output: BTreeMap<(String, String), f64> = BTreeMap::new();
        for candidate in candidates {
            let Some(weight) = self.weights.get(&candidate.name) else {
                return Err(CartoBoostError::InvalidInput(format!(
                    "missing blend weight for candidate '{}'",
                    candidate.name
                )));
            };
            let map = prediction_map(&candidate.predictions)?;
            let keys = map.keys().cloned().collect::<Vec<_>>();
            if let Some(expected) = &expected_keys {
                if expected != &keys {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "candidate '{}' is not aligned to ensemble rows",
                        candidate.name
                    )));
                }
            } else {
                expected_keys = Some(keys);
            }
            for (key, value) in map {
                *output.entry(key).or_insert(0.0) += weight * value;
            }
        }
        for name in self.weights.keys() {
            if !candidates.iter().any(|candidate| &candidate.name == name) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "blend weight references unknown candidate '{name}'"
                )));
            }
        }
        Ok(output
            .into_iter()
            .map(|((series_id, row_id), value)| SequenceBlendPrediction {
                series_id,
                row_id,
                value,
            })
            .collect())
    }
}

pub fn generate_group_oof_candidate_rows(
    fold: &SequenceOofFold,
) -> Result<Vec<SequenceOofCandidateRow>> {
    if fold.validation_group_id.trim().is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "OOF validation_group_id must be non-empty".to_string(),
        ));
    }
    if fold.train_group_ids.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "OOF train_group_ids must be nonempty".to_string(),
        ));
    }
    if fold
        .train_group_ids
        .iter()
        .any(|group| group.trim().is_empty())
    {
        return Err(CartoBoostError::InvalidInput(
            "OOF train_group_ids must be non-empty".to_string(),
        ));
    }
    if fold
        .train_group_ids
        .iter()
        .any(|group| group == &fold.validation_group_id)
    {
        return Err(CartoBoostError::InvalidInput(format!(
            "leakage check failed: validation group '{}' appears in its training groups",
            fold.validation_group_id
        )));
    }
    if fold.candidates.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "OOF candidate generation requires at least one candidate".to_string(),
        ));
    }
    let actual_map = prediction_map(&fold.actuals)?;
    for (series_id, _) in actual_map.keys() {
        if series_id != &fold.validation_group_id {
            return Err(CartoBoostError::InvalidInput(format!(
                "OOF actual series_id '{series_id}' does not match validation group '{}'",
                fold.validation_group_id
            )));
        }
    }
    let mut candidate_names = BTreeSet::new();
    let mut candidate_maps = Vec::with_capacity(fold.candidates.len());
    for candidate in &fold.candidates {
        if candidate.name.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "OOF candidate names must be non-empty".to_string(),
            ));
        }
        if !candidate_names.insert(candidate.name.as_str()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "duplicate OOF candidate name '{}'",
                candidate.name
            )));
        }
        let map = prediction_map(&candidate.predictions)?;
        if map.keys().collect::<Vec<_>>() != actual_map.keys().collect::<Vec<_>>() {
            return Err(CartoBoostError::InvalidInput(format!(
                "OOF candidate '{}' is not aligned to validation actual rows",
                candidate.name
            )));
        }
        candidate_maps.push((candidate.name.as_str(), map));
    }
    let rows = actual_map
        .into_iter()
        .map(|((_, row_id), actual)| {
            let candidate_predictions = candidate_maps
                .iter()
                .map(|(name, map)| {
                    (
                        (*name).to_string(),
                        map[&(fold.validation_group_id.clone(), row_id.clone())],
                    )
                })
                .collect::<BTreeMap<_, _>>();
            SequenceOofCandidateRow {
                group_id: fold.validation_group_id.clone(),
                row_id,
                actual,
                candidate_predictions,
                train_group_ids: fold.train_group_ids.clone(),
            }
        })
        .collect::<Vec<_>>();
    validate_oof_meta_training(&rows)?;
    Ok(rows)
}

pub fn validate_oof_meta_training(rows: &[SequenceOofCandidateRow]) -> Result<()> {
    if rows.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "OOF meta-training rows must be nonempty".to_string(),
        ));
    }
    for row in rows {
        if row.group_id.trim().is_empty() || row.row_id.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "OOF group_id and row_id must be non-empty".to_string(),
            ));
        }
        if !row.actual.is_finite()
            || row
                .candidate_predictions
                .values()
                .any(|value| !value.is_finite())
        {
            return Err(CartoBoostError::InvalidInput(
                "OOF actuals and candidate predictions must be finite".to_string(),
            ));
        }
        if row
            .train_group_ids
            .iter()
            .any(|train_group| train_group == &row.group_id)
        {
            return Err(CartoBoostError::InvalidInput(format!(
                "leakage check failed: validation group '{}' appears in its training groups",
                row.group_id
            )));
        }
    }
    Ok(())
}

pub fn per_group_error_summary(
    predictions: &[SequenceGroupPrediction],
) -> Result<BTreeMap<String, SequenceGroupMetric>> {
    if predictions.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "group predictions must be nonempty".to_string(),
        ));
    }
    let mut sums: BTreeMap<String, (usize, f64, f64)> = BTreeMap::new();
    for row in predictions {
        if row.group_id.trim().is_empty() || row.row_id.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "group prediction ids must be non-empty".to_string(),
            ));
        }
        if !row.actual.is_finite() || !row.prediction.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "group prediction actuals and predictions must be finite".to_string(),
            ));
        }
        let err = row.actual - row.prediction;
        let entry = sums.entry(row.group_id.clone()).or_default();
        entry.0 += 1;
        entry.1 += err * err;
        entry.2 += err.abs();
    }
    Ok(sums
        .into_iter()
        .map(|(group, (count, sse, sae))| {
            (
                group,
                SequenceGroupMetric {
                    count,
                    rmse: (sse / count as f64).sqrt(),
                    mae: sae / count as f64,
                },
            )
        })
        .collect())
}

fn run_extended_kalman(
    series: &SequenceSeries,
    reference: &ReferenceSignal,
    config: SequenceStateSpaceConfig,
    smooth: bool,
) -> Result<SequenceKalmanResult> {
    series.validate()?;
    reference.validate()?;
    config.validate()?;
    let initial = initial_axis(series, reference);
    let mut state = [reference.clamp_axis(initial.axis), 0.0];
    let mut covariance = [
        [config.initial_axis_variance, 0.0],
        [0.0, config.initial_rate_variance],
    ];
    let mut filtered_states = Vec::with_capacity(series.rows.len());
    let mut filtered_covariances = Vec::with_capacity(series.rows.len());
    let mut prior_states = Vec::with_capacity(series.rows.len());
    let mut prior_covariances = Vec::with_capacity(series.rows.len());
    let mut points = Vec::with_capacity(series.rows.len());
    let mut log_likelihood = 0.0;
    for (idx, row) in series.rows.iter().enumerate() {
        if idx > 0 {
            let dt = row.position - series.rows[idx - 1].position;
            state = transition(state, dt);
            covariance = predict_covariance(covariance, dt, config);
            state[0] = reference.clamp_axis(state[0]);
        }
        prior_states.push(state);
        prior_covariances.push(covariance);
        let mut innovation = None;
        let mut innovation_variance = None;
        // A target-derived initial axis is already conditioned on row zero.
        if let Some(observed) = row
            .target
            .filter(|_| idx > 0 || !initial.conditioned_on_first_target)
        {
            let h = reference.interpolate(state[0]);
            let dh = reference.derivative(state[0]);
            let s = dh * covariance[0][0] * dh + config.signal_observation_variance;
            let s = positive_innovation_variance(s, "signal")?;
            let residual = observed - h;
            let k0 = (covariance[0][0] * dh) / s;
            let k1 = (covariance[1][0] * dh) / s;
            state[0] += k0 * residual;
            state[1] += k1 * residual;
            covariance = joseph_1d(
                covariance,
                [dh, 0.0],
                [k0, k1],
                config.signal_observation_variance,
            );
            log_likelihood += gaussian_log_likelihood(residual, s);
            innovation = Some(residual);
            innovation_variance = Some(s);
        }
        if let (Some(rate), Some(variance)) = (row.auxiliary_rate, config.rate_observation_variance)
        {
            let residual = rate - state[1];
            let s = positive_innovation_variance(covariance[1][1] + variance, "rate")?;
            let k0 = covariance[0][1] / s;
            let k1 = covariance[1][1] / s;
            state[0] += k0 * residual;
            state[1] += k1 * residual;
            covariance = joseph_1d(covariance, [0.0, 1.0], [k0, k1], variance);
        }
        state[0] = reference.clamp_axis(state[0]);
        filtered_states.push(state);
        filtered_covariances.push(covariance);
        points.push(SequenceKalmanPoint {
            row_id: row.row_id.clone(),
            position: row.position,
            observed: row.target,
            predicted_axis: state[0],
            predicted_rate: state[1],
            predicted_signal: reference.interpolate(state[0]),
            covariance,
            innovation,
            innovation_variance,
        });
    }
    if smooth {
        let smoothed = smooth_states(
            &series.rows,
            &filtered_states,
            &filtered_covariances,
            &prior_states,
            &prior_covariances,
            config,
        )?;
        for (point, (state, covariance)) in points.iter_mut().zip(smoothed) {
            point.predicted_axis = reference.clamp_axis(state[0]);
            point.predicted_rate = state[1];
            point.predicted_signal = reference.interpolate(point.predicted_axis);
            point.covariance = covariance;
        }
    }
    Ok(SequenceKalmanResult {
        points,
        log_likelihood,
    })
}

fn run_unscented_kalman(
    series: &SequenceSeries,
    reference: &ReferenceSignal,
    config: SequenceStateSpaceConfig,
) -> Result<SequenceKalmanResult> {
    series.validate()?;
    reference.validate()?;
    config.validate()?;
    let initial = initial_axis(series, reference);
    let mut state = [reference.clamp_axis(initial.axis), 0.0];
    let mut covariance = [
        [config.initial_axis_variance, 0.0],
        [0.0, config.initial_rate_variance],
    ];
    let mut points = Vec::with_capacity(series.rows.len());
    let mut log_likelihood = 0.0;
    for (idx, row) in series.rows.iter().enumerate() {
        if idx > 0 {
            let dt = row.position - series.rows[idx - 1].position;
            state = transition(state, dt);
            covariance = predict_covariance(covariance, dt, config);
            state[0] = reference.clamp_axis(state[0]);
        }
        let mut innovation = None;
        let mut innovation_variance = None;
        // Keep UKF initialization causal and avoid assimilating row zero twice.
        if let Some(observed) = row
            .target
            .filter(|_| idx > 0 || !initial.conditioned_on_first_target)
        {
            let sigma = sigma_points(state, covariance, config);
            let weights = sigma_weights(config);
            let z_points = sigma
                .iter()
                .map(|point| reference.interpolate(point[0]))
                .collect::<Vec<_>>();
            let z_mean = weights
                .iter()
                .zip(&z_points)
                .map(|(weight, z)| weight.0 * z)
                .sum::<f64>();
            let mut s = config.signal_observation_variance;
            let mut cross = [0.0, 0.0];
            for ((point, z), (wm, wc)) in sigma.iter().zip(&z_points).zip(&weights) {
                let dz = z - z_mean;
                s += wc * dz * dz;
                cross[0] += wc * (point[0] - state[0]) * dz;
                cross[1] += wc * (point[1] - state[1]) * dz;
                let _ = wm;
            }
            let s = positive_innovation_variance(s, "signal")?;
            let residual = observed - z_mean;
            let gain = [cross[0] / s, cross[1] / s];
            state[0] += gain[0] * residual;
            state[1] += gain[1] * residual;
            covariance[0][0] -= gain[0] * s * gain[0];
            covariance[0][1] -= gain[0] * s * gain[1];
            covariance[1][0] -= gain[1] * s * gain[0];
            covariance[1][1] -= gain[1] * s * gain[1];
            covariance = symmetrize(covariance);
            log_likelihood += gaussian_log_likelihood(residual, s);
            innovation = Some(residual);
            innovation_variance = Some(s);
        }
        if let (Some(rate), Some(variance)) = (row.auxiliary_rate, config.rate_observation_variance)
        {
            let residual = rate - state[1];
            let s = positive_innovation_variance(covariance[1][1] + variance, "rate")?;
            let gain = [covariance[0][1] / s, covariance[1][1] / s];
            state[0] += gain[0] * residual;
            state[1] += gain[1] * residual;
            covariance = joseph_1d(covariance, [0.0, 1.0], gain, variance);
        }
        state[0] = reference.clamp_axis(state[0]);
        points.push(SequenceKalmanPoint {
            row_id: row.row_id.clone(),
            position: row.position,
            observed: row.target,
            predicted_axis: state[0],
            predicted_rate: state[1],
            predicted_signal: reference.interpolate(state[0]),
            covariance,
            innovation,
            innovation_variance,
        });
    }
    Ok(SequenceKalmanResult {
        points,
        log_likelihood,
    })
}

#[derive(Debug, Clone, Copy)]
struct SequenceInitialAxis {
    axis: f64,
    conditioned_on_first_target: bool,
}

fn initial_axis(series: &SequenceSeries, reference: &ReferenceSignal) -> SequenceInitialAxis {
    let first = &series.rows[0];
    if let Some(axis) = first.reference_axis {
        return SequenceInitialAxis {
            axis,
            conditioned_on_first_target: false,
        };
    }
    let first_target = first
        .target
        .expect("validated sequence starts with a known target");
    let axis = reference
        .signal
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - first_target)
                .abs()
                .total_cmp(&(*b - first_target).abs())
        })
        .map(|(idx, _)| reference.axis[idx])
        .unwrap_or(reference.axis[0]);
    SequenceInitialAxis {
        axis,
        conditioned_on_first_target: true,
    }
}

fn transition(state: [f64; 2], dt: f64) -> [f64; 2] {
    [state[0] + dt * state[1], state[1]]
}

fn predict_covariance(
    p: [[f64; 2]; 2],
    dt: f64,
    config: SequenceStateSpaceConfig,
) -> [[f64; 2]; 2] {
    [
        [
            p[0][0] + dt * (p[1][0] + p[0][1]) + dt * dt * p[1][1] + config.axis_process_variance,
            p[0][1] + dt * p[1][1],
        ],
        [
            p[1][0] + dt * p[1][1],
            p[1][1] + config.rate_process_variance,
        ],
    ]
}

fn joseph_1d(p: [[f64; 2]; 2], h: [f64; 2], k: [f64; 2], r: f64) -> [[f64; 2]; 2] {
    let a = [
        [1.0 - k[0] * h[0], -k[0] * h[1]],
        [-k[1] * h[0], 1.0 - k[1] * h[1]],
    ];
    let ap = mat_mul(a, p);
    let apa = mat_mul(ap, transpose(a));
    symmetrize([
        [apa[0][0] + k[0] * r * k[0], apa[0][1] + k[0] * r * k[1]],
        [apa[1][0] + k[1] * r * k[0], apa[1][1] + k[1] * r * k[1]],
    ])
}

type SmoothedState = ([f64; 2], [[f64; 2]; 2]);

fn smooth_states(
    rows: &[SequenceRow],
    filtered_states: &[[f64; 2]],
    filtered_covariances: &[[[f64; 2]; 2]],
    prior_states: &[[f64; 2]],
    prior_covariances: &[[[f64; 2]; 2]],
    config: SequenceStateSpaceConfig,
) -> Result<Vec<SmoothedState>> {
    let n = rows.len();
    let mut states = filtered_states.to_vec();
    let mut covariances = filtered_covariances.to_vec();
    for idx in (0..n - 1).rev() {
        let dt = rows[idx + 1].position - rows[idx].position;
        let f = [[1.0, dt], [0.0, 1.0]];
        let predicted_covariance = if idx + 1 < prior_covariances.len() {
            prior_covariances[idx + 1]
        } else {
            predict_covariance(filtered_covariances[idx], dt, config)
        };
        let gain = mat_mul(
            mat_mul(filtered_covariances[idx], transpose(f)),
            inv2(predicted_covariance)?,
        );
        let predicted_state = prior_states[idx + 1];
        let delta = [
            states[idx + 1][0] - predicted_state[0],
            states[idx + 1][1] - predicted_state[1],
        ];
        states[idx] = [
            filtered_states[idx][0] + gain[0][0] * delta[0] + gain[0][1] * delta[1],
            filtered_states[idx][1] + gain[1][0] * delta[0] + gain[1][1] * delta[1],
        ];
        let covariance_delta = mat_sub(covariances[idx + 1], predicted_covariance);
        covariances[idx] = symmetrize(mat_add(
            filtered_covariances[idx],
            mat_mul(mat_mul(gain, covariance_delta), transpose(gain)),
        ));
    }
    Ok(states.into_iter().zip(covariances).collect())
}

fn sigma_points(
    state: [f64; 2],
    covariance: [[f64; 2]; 2],
    config: SequenceStateSpaceConfig,
) -> Vec<[f64; 2]> {
    let scale = sigma_point_scale(config);
    let a = (covariance[0][0].max(1e-12) * scale).sqrt();
    let b = covariance[1][0] / covariance[0][0].max(1e-12) * a;
    let c = ((covariance[1][1] * scale - b * b).max(1e-12)).sqrt();
    vec![
        state,
        [state[0] + a, state[1] + b],
        [state[0] - a, state[1] - b],
        [state[0], state[1] + c],
        [state[0], state[1] - c],
    ]
}

fn sigma_weights(config: SequenceStateSpaceConfig) -> Vec<(f64, f64)> {
    let n = 2.0;
    let scale = sigma_point_scale(config);
    let lambda = scale - n;
    let wm0 = lambda / scale;
    let wc0 = wm0 + (1.0 - config.sigma_point_alpha.powi(2) + config.sigma_point_beta);
    let wi = 1.0 / (2.0 * scale);
    vec![(wm0, wc0), (wi, wi), (wi, wi), (wi, wi), (wi, wi)]
}

fn sigma_point_scale(config: SequenceStateSpaceConfig) -> f64 {
    config.sigma_point_alpha.powi(2) * (2.0 + config.sigma_point_kappa)
}

fn positive_innovation_variance(value: f64, measurement: &str) -> Result<f64> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{measurement} innovation variance must be finite and positive"
        )));
    }
    Ok(value)
}

fn emission_cost(observed: Option<f64>, expected: f64, config: ReferencePathConfig) -> f64 {
    let Some(observed) = observed else {
        return 0.0;
    };
    let z = (observed - expected) / config.emission_scale;
    0.5 * (config.student_t_df + 1.0) * (1.0 + z * z / config.student_t_df).ln()
}

fn start_cost(axis: f64, config: ReferencePathConfig) -> f64 {
    config
        .start_axis
        .map(|start| config.start_penalty * (axis - start).powi(2))
        .unwrap_or(0.0)
}

fn transition_cost(
    reference: &ReferenceSignal,
    prior: usize,
    state: usize,
    prior_prior: Option<usize>,
    config: ReferencePathConfig,
) -> f64 {
    let delta = reference.axis[state] - reference.axis[prior];
    let smooth = prior_prior
        .map(|pp| {
            let prev_delta = reference.axis[prior] - reference.axis[pp];
            (delta - prev_delta).powi(2)
        })
        .unwrap_or(0.0);
    config.transition_penalty * delta.powi(2) + config.smoothness_penalty * smooth
}

fn prediction_map(
    predictions: &[SequenceCandidatePrediction],
) -> Result<BTreeMap<(String, String), f64>> {
    if predictions.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "candidate predictions must be nonempty".to_string(),
        ));
    }
    let mut map = BTreeMap::new();
    for row in predictions {
        if row.series_id.trim().is_empty() || row.row_id.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "candidate prediction ids must be non-empty".to_string(),
            ));
        }
        if !row.value.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "candidate prediction values must be finite".to_string(),
            ));
        }
        let key = (row.series_id.clone(), row.row_id.clone());
        if map.insert(key, row.value).is_some() {
            return Err(CartoBoostError::InvalidInput(
                "duplicate candidate prediction row".to_string(),
            ));
        }
    }
    Ok(map)
}

fn blend_gradient(
    candidates: &[SequenceCandidate],
    actuals: &[SequenceCandidatePrediction],
    weights: &BTreeMap<String, f64>,
) -> Result<BTreeMap<String, f64>> {
    let actual_map = prediction_map(actuals)?;
    let candidate_maps = candidates
        .iter()
        .map(|candidate| {
            Ok((
                candidate.name.clone(),
                prediction_map(&candidate.predictions)?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut gradient = BTreeMap::new();
    for name in weights.keys() {
        let mut grad = 0.0;
        for (key, actual) in &actual_map {
            let blended = weights
                .iter()
                .map(|(candidate_name, weight)| weight * candidate_maps[candidate_name][key])
                .sum::<f64>();
            grad += 2.0 * (blended - actual) * candidate_maps[name][key];
        }
        gradient.insert(name.clone(), grad / actual_map.len() as f64);
    }
    Ok(gradient)
}

fn normalize_weight_map(mut weights: BTreeMap<String, f64>) -> Result<SequenceCandidateEnsemble> {
    validate_weights(&weights)?;
    let total = weights.values().sum::<f64>();
    for value in weights.values_mut() {
        *value /= total;
    }
    Ok(SequenceCandidateEnsemble { weights })
}

fn validate_weights(weights: &BTreeMap<String, f64>) -> Result<()> {
    if weights.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "sequence blend weights must be nonempty".to_string(),
        ));
    }
    let mut total = 0.0;
    for (name, weight) in weights {
        if name.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "sequence blend candidate names must be non-empty".to_string(),
            ));
        }
        if !weight.is_finite() || *weight < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "sequence blend weights must be finite and non-negative".to_string(),
            ));
        }
        total += weight;
    }
    if total <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "sequence blend requires at least one positive weight".to_string(),
        ));
    }
    Ok(())
}

fn validate_optional_finite(value: Option<f64>, name: &str) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be finite when provided"
        )));
    }
    Ok(())
}

fn validate_positive(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be finite and positive"
        )));
    }
    Ok(())
}

fn validate_nonnegative(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be finite and non-negative"
        )));
    }
    Ok(())
}

fn gaussian_log_likelihood(residual: f64, variance: f64) -> f64 {
    -0.5 * ((2.0 * std::f64::consts::PI * variance).ln() + residual * residual / variance)
}

fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    max + values
        .iter()
        .map(|value| (value - max).exp())
        .sum::<f64>()
        .ln()
}

fn normalize_log_row(values: &mut [f64]) {
    let total = log_sum_exp(values);
    for value in values {
        *value -= total;
    }
}

fn mat_mul(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
        ],
    ]
}

fn mat_add(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] + b[0][0], a[0][1] + b[0][1]],
        [a[1][0] + b[1][0], a[1][1] + b[1][1]],
    ]
}

fn mat_sub(a: [[f64; 2]; 2], b: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [a[0][0] - b[0][0], a[0][1] - b[0][1]],
        [a[1][0] - b[1][0], a[1][1] - b[1][1]],
    ]
}

fn transpose(a: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}

fn inv2(a: [[f64; 2]; 2]) -> Result<[[f64; 2]; 2]> {
    if a.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "RTS predicted covariance must be finite".to_string(),
        ));
    }
    let scale = a
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 || !scale.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "RTS predicted covariance must be invertible".to_string(),
        ));
    }
    let normalized = [
        [a[0][0] / scale, a[0][1] / scale],
        [a[1][0] / scale, a[1][1] / scale],
    ];
    let symmetry_error = (normalized[0][1] - normalized[1][0]).abs();
    if normalized[0][0] <= 0.0 || normalized[1][1] <= 0.0 || symmetry_error > 64.0 * f64::EPSILON {
        return Err(CartoBoostError::InvalidInput(
            "RTS predicted covariance must be symmetric positive definite".to_string(),
        ));
    }
    let det = normalized[0][0] * normalized[1][1] - normalized[0][1] * normalized[1][0];
    if !det.is_finite() || det <= f64::EPSILON {
        return Err(CartoBoostError::InvalidInput(
            "RTS predicted covariance must be positive definite and well-conditioned".to_string(),
        ));
    }
    Ok([
        [
            normalized[1][1] / (det * scale),
            -normalized[0][1] / (det * scale),
        ],
        [
            -normalized[1][0] / (det * scale),
            normalized[0][0] / (det * scale),
        ],
    ])
}

fn symmetrize(mut a: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let off = 0.5 * (a[0][1] + a[1][0]);
    a[0][1] = off;
    a[1][0] = off;
    a[0][0] = a[0][0].max(1e-12);
    a[1][1] = a[1][1].max(1e-12);
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference() -> ReferenceSignal {
        ReferenceSignal::new(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0, 3.0]).unwrap()
    }

    fn series() -> SequenceSeries {
        SequenceSeries {
            series_id: "pickup_zone_1".to_string(),
            rows: vec![
                row("r0", 0.0, Some(0.0)),
                row("r1", 1.0, Some(1.0)),
                row("r2", 2.0, None),
                row("r3", 3.0, None),
            ],
        }
    }

    fn row(row_id: &str, position: f64, target: Option<f64>) -> SequenceRow {
        SequenceRow {
            row_id: row_id.to_string(),
            position,
            target,
            reference_axis: None,
            reference_signal: None,
            auxiliary_rate: None,
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1.0e-10,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn validates_known_prefix_and_prediction_suffix() {
        let mask = series().prediction_mask().unwrap();
        assert_eq!(mask.row_ids, vec!["r2", "r3"]);
        let mut leaky = series();
        leaky.rows[3].target = Some(3.0);
        assert!(leaky
            .validate()
            .unwrap_err()
            .to_string()
            .contains("leakage"));
    }

    #[test]
    fn rejects_duplicate_reference_axis() {
        let err = ReferenceSignal::new(vec![0.0, 1.0, 1.0], vec![0.0, 1.0, 2.0]).unwrap_err();
        assert!(err.to_string().contains("deduplicated"));
    }

    #[test]
    fn known_prefix_continuation_is_finite() {
        let points = missing_target_continuation(
            &series(),
            &reference(),
            SequenceStateSpaceConfig::default(),
        )
        .unwrap();
        assert_eq!(points.len(), 2);
        assert!(points
            .iter()
            .all(|point| point.predicted_signal.is_finite()));
    }

    #[test]
    fn ekf_ukf_and_smoother_shapes_are_aligned() {
        let ekf =
            forward_ekf(&series(), &reference(), SequenceStateSpaceConfig::default()).unwrap();
        let ukf =
            ukf_reference(&series(), &reference(), SequenceStateSpaceConfig::default()).unwrap();
        let rts =
            rts_smoother(&series(), &reference(), SequenceStateSpaceConfig::default()).unwrap();
        assert_eq!(ekf.points.len(), 4);
        assert_eq!(ukf.points.len(), 4);
        assert_eq!(
            rts.points.iter().map(|p| &p.row_id).collect::<Vec<_>>(),
            vec!["r0", "r1", "r2", "r3"]
        );
        assert!(ekf
            .points
            .iter()
            .all(|point| point.predicted_axis.is_finite()));
        assert!(ukf
            .points
            .iter()
            .all(|point| point.predicted_axis.is_finite()));
    }

    #[test]
    fn forward_filters_initialize_causally_and_condition_on_the_first_target_once() {
        let baseline = series();
        let mut changed_later_prefix = series();
        changed_later_prefix.rows[1].target = Some(3.0);
        let config = SequenceStateSpaceConfig::default();

        let baseline_ekf = forward_ekf(&baseline, &reference(), config).unwrap();
        let changed_ekf = forward_ekf(&changed_later_prefix, &reference(), config).unwrap();
        let baseline_ukf = ukf_reference(&baseline, &reference(), config).unwrap();
        let changed_ukf = ukf_reference(&changed_later_prefix, &reference(), config).unwrap();

        assert_eq!(baseline_ekf.points[0], changed_ekf.points[0]);
        assert_eq!(baseline_ukf.points[0], changed_ukf.points[0]);
        assert_eq!(baseline_ekf.points[0].predicted_axis, 0.0);
        assert_eq!(baseline_ukf.points[0].predicted_axis, 0.0);
        assert!(baseline_ekf.points[0].innovation.is_none());
        assert!(baseline_ukf.points[0].innovation.is_none());
    }

    #[test]
    fn ekf_signal_innovation_variance_is_h_p_h_transpose_plus_r() {
        let result =
            forward_ekf(&series(), &reference(), SequenceStateSpaceConfig::default()).unwrap();

        let second = &result.points[1];
        assert_close(second.innovation.unwrap(), 1.0);
        // P^-_00 = 1 + 1^2 * 1 + q_axis = 2.001, H = [1, 0], R = 0.01.
        assert_close(second.innovation_variance.unwrap(), 2.011);
    }

    #[test]
    fn reference_derivative_matches_the_clamped_interpolator_at_boundaries() {
        let reference = reference();

        assert_eq!(reference.derivative(-1.0), 0.0);
        assert_eq!(reference.derivative(4.0), 0.0);
        assert_eq!(reference.derivative(0.0), 1.0);
        assert_eq!(reference.derivative(3.0), 1.0);
    }

    #[test]
    fn ekf_clamps_transitioned_axis_before_boundary_measurement_update() {
        let mut boundary_series = SequenceSeries {
            series_id: "pickup_zone_1".to_string(),
            rows: vec![
                row("r0", 0.0, Some(0.0)),
                row("r1", 1.0, Some(3.0)),
                row("r2", 2.0, None),
            ],
        };
        boundary_series.rows[0].auxiliary_rate = Some(10.0);
        let config = SequenceStateSpaceConfig {
            rate_observation_variance: Some(1.0e-6),
            ..SequenceStateSpaceConfig::default()
        };

        let result = forward_ekf(&boundary_series, &reference(), config).unwrap();

        assert_eq!(result.points[1].predicted_axis, 3.0);
        assert!(result.points[1].innovation_variance.unwrap() > config.signal_observation_variance);
    }

    #[test]
    fn state_space_config_rejects_invalid_sigma_point_scale() {
        let invalid_kappa = SequenceStateSpaceConfig {
            sigma_point_kappa: -2.0,
            ..SequenceStateSpaceConfig::default()
        };
        let overflowing_alpha = SequenceStateSpaceConfig {
            sigma_point_alpha: f64::MAX,
            ..SequenceStateSpaceConfig::default()
        };

        assert!(invalid_kappa
            .validate()
            .unwrap_err()
            .to_string()
            .contains("sigma-point scale"));
        assert!(overflowing_alpha
            .validate()
            .unwrap_err()
            .to_string()
            .contains("sigma-point scale"));
    }

    #[test]
    fn rts_covariance_inverse_is_scale_aware_and_rejects_invalid_matrices() {
        let inverse = inv2([[1.0e-18, 0.0], [0.0, 2.0e-18]]).unwrap();
        assert_close(inverse[0][0] / 1.0e18, 1.0);
        assert_close(inverse[1][1] / 5.0e17, 1.0);
        assert!(inv2([[1.0, 1.0], [1.0, 1.0]]).is_err());
        assert!(inv2([[1.0, 2.0], [2.0, 1.0]]).is_err());
        assert!(inv2([[-1.0, 0.0], [0.0, -1.0]]).is_err());
        assert!(inv2([[1.0, 0.5], [0.0, 1.0]]).is_err());
    }

    #[test]
    fn rts_smoother_propagates_invalid_predicted_covariance_errors() {
        let rows = vec![row("r0", 0.0, Some(0.0)), row("r1", 1.0, None)];
        let states = [[0.0, 0.0], [0.0, 0.0]];
        let identity = [[1.0, 0.0], [0.0, 1.0]];
        let filtered_covariances = [identity, identity];
        let prior_covariances = [identity, [[1.0, 1.0], [1.0, 1.0]]];

        let err = smooth_states(
            &rows,
            &states,
            &filtered_covariances,
            &states,
            &prior_covariances,
            SequenceStateSpaceConfig::default(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("RTS predicted covariance"));
    }

    #[test]
    fn viterbi_chooses_obvious_path() {
        let result =
            reference_path_viterbi(&series(), &reference(), ReferencePathConfig::default())
                .unwrap();
        assert_eq!(result.points[0].axis, 0.0);
        assert_eq!(result.points[1].axis, 1.0);
    }

    #[test]
    fn posterior_mean_stays_within_reference_bounds() {
        let result =
            reference_path_posterior_mean(&series(), &reference(), ReferencePathConfig::default())
                .unwrap();
        assert!(result
            .points
            .iter()
            .all(|point| (0.0..=3.0).contains(&point.axis)));
    }

    #[test]
    fn oof_validation_rejects_training_on_validation_group() {
        let rows = vec![SequenceOofCandidateRow {
            group_id: "pickup_zone_1".to_string(),
            row_id: "r1".to_string(),
            actual: 1.0,
            candidate_predictions: BTreeMap::from([("candidate".to_string(), 1.1)]),
            train_group_ids: vec!["pickup_zone_1".to_string()],
        }];
        assert!(validate_oof_meta_training(&rows)
            .unwrap_err()
            .to_string()
            .contains("leakage"));
    }

    #[test]
    fn group_oof_generation_aligns_candidates_and_blocks_leakage() {
        let fold = SequenceOofFold {
            validation_group_id: "pickup_zone_1".to_string(),
            train_group_ids: vec!["pickup_zone_2".to_string()],
            actuals: vec![SequenceCandidatePrediction {
                series_id: "pickup_zone_1".to_string(),
                row_id: "hour_01".to_string(),
                value: 10.0,
            }],
            candidates: vec![SequenceCandidate {
                name: "candidate_a".to_string(),
                predictions: vec![SequenceCandidatePrediction {
                    series_id: "pickup_zone_1".to_string(),
                    row_id: "hour_01".to_string(),
                    value: 11.0,
                }],
            }],
        };
        let rows = generate_group_oof_candidate_rows(&fold).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].actual, 10.0);
        assert_eq!(rows[0].candidate_predictions["candidate_a"], 11.0);

        let mut leaky = fold;
        leaky.train_group_ids.push("pickup_zone_1".to_string());
        assert!(generate_group_oof_candidate_rows(&leaky)
            .unwrap_err()
            .to_string()
            .contains("leakage"));
    }

    #[test]
    fn viterbi_smoothness_penalty_affects_path_score() {
        let reference = ReferenceSignal::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]).unwrap();
        let series = SequenceSeries {
            series_id: "pickup_zone_1".to_string(),
            rows: vec![
                row("r0", 0.0, Some(0.0)),
                row("r1", 1.0, Some(2.0)),
                row("r2", 2.0, Some(0.0)),
                row("r3", 3.0, None),
            ],
        };
        let base =
            reference_path_viterbi(&series, &reference, ReferencePathConfig::default()).unwrap();
        let smooth = reference_path_viterbi(
            &series,
            &reference,
            ReferencePathConfig {
                smoothness_penalty: 10.0,
                ..ReferencePathConfig::default()
            },
        )
        .unwrap();
        assert!(smooth.score > base.score);
    }

    #[test]
    fn sequence_blend_aligns_rows() {
        let candidate_a = SequenceCandidate {
            name: "a".to_string(),
            predictions: vec![SequenceCandidatePrediction {
                series_id: "s".to_string(),
                row_id: "r".to_string(),
                value: 1.0,
            }],
        };
        let candidate_b = SequenceCandidate {
            name: "b".to_string(),
            predictions: vec![SequenceCandidatePrediction {
                series_id: "s".to_string(),
                row_id: "r".to_string(),
                value: 3.0,
            }],
        };
        let ensemble = SequenceCandidateEnsemble::fixed(BTreeMap::from([
            ("a".to_string(), 0.25),
            ("b".to_string(), 0.75),
        ]))
        .unwrap();
        let blended = ensemble.predict(&[candidate_a, candidate_b]).unwrap();
        assert_eq!(blended[0].value, 2.5);
    }
}
