use crate::forecasting::ForecastFrame;
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const AUTO_SELECTION_MIN_RELATIVE_GAIN: f64 = 0.03;
pub const AUTO_SELECTION_ROBUST_RELATIVE_TOLERANCE: f64 = 0.05;
pub const NATIVE_AUTO_RAW_KEEP_RELATIVE_GAIN: f64 = 0.50;
pub const FORECAST_MAGNITUDE_GUARD_MULTIPLIER: f64 = 100.0;
pub const FORECAST_MAGNITUDE_GUARD_ABSOLUTE_FLOOR: f64 = 1_000.0;

pub fn seasonal_naive_candidate_prediction(values: &[f64], season_length: usize) -> Result<f64> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "seasonal naive candidate requires non-empty history".to_string(),
        ));
    }
    if season_length == 0 {
        return Err(CartoBoostError::InvalidInput(
            "seasonal naive candidate season_length must be positive".to_string(),
        ));
    }
    let value = if values.len() >= season_length {
        values[values.len() - season_length]
    } else {
        values[values.len() - 1]
    };
    validate_finite_forecast_value(value, "seasonal naive candidate")
}

pub fn trend_candidate_prediction(
    values: &[f64],
    step: usize,
    season_length: usize,
    mode: &str,
) -> Result<f64> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "trend candidate requires non-empty history".to_string(),
        ));
    }
    if step == 0 {
        return Err(CartoBoostError::InvalidInput(
            "trend candidate step must be positive".to_string(),
        ));
    }
    if season_length == 0 {
        return Err(CartoBoostError::InvalidInput(
            "trend candidate season_length must be positive".to_string(),
        ));
    }
    validate_finite_series(values, "trend candidate history")?;
    if values.len() == 1 {
        return Ok(values[0]);
    }
    let step = step as f64;
    let slope = (values[values.len() - 1] - values[0]) / (values.len() - 1) as f64;
    let prediction = match mode {
        "drift" => values[values.len() - 1] + step * slope,
        "half_drift" => values[values.len() - 1] + 0.5 * step * slope,
        "seasonal_drift" => {
            let baseline = if season_length > 1 && values.len() >= season_length {
                values[values.len() - season_length]
            } else {
                values[values.len() - 1]
            };
            baseline + step * slope
        }
        mode if mode.starts_with("seasonal_cycle_drift_") => {
            if season_length <= 1 || values.len() < 2 * season_length {
                values[values.len() - 1]
            } else {
                let alpha = mode
                    .rsplit_once('_')
                    .and_then(|(_, suffix)| suffix.parse::<f64>().ok())
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "unsupported trend candidate mode '{mode}'"
                        ))
                    })?
                    / 100.0;
                let baseline = values[values.len() - season_length];
                let seasonal_slope = (values[values.len() - season_length]
                    - values[values.len() - 2 * season_length])
                    / season_length as f64;
                baseline + alpha * step * seasonal_slope
            }
        }
        _ => {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported trend candidate mode '{mode}'"
            )));
        }
    };
    validate_finite_forecast_value(prediction, "trend candidate")
}

pub fn calendar_profile_candidate_prediction(
    values: &[f64],
    day_of_months: &[u32],
    target_day_of_month: u32,
    mode: &str,
    elapsed_phase_period: Option<usize>,
) -> Result<f64> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "calendar profile candidate requires non-empty history".to_string(),
        ));
    }
    if values.len() != day_of_months.len() {
        return Err(CartoBoostError::InvalidInput(
            "calendar profile candidate history values and day indexes must align".to_string(),
        ));
    }
    validate_finite_series(values, "calendar profile candidate history")?;
    let fallback = if values.len() >= 7 {
        values[values.len() - 7]
    } else {
        values[values.len() - 1]
    };
    let prediction = match mode {
        "day_of_month" => {
            let matches = values
                .iter()
                .zip(day_of_months)
                .filter_map(|(&value, &day)| (day == target_day_of_month).then_some(value))
                .collect::<Vec<_>>();
            mean_or_fallback(&matches, fallback)
        }
        "elapsed_phase" => {
            let period = elapsed_phase_period.ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "elapsed phase calendar profile requires a phase period".to_string(),
                )
            })?;
            if period < 2 {
                return Err(CartoBoostError::InvalidInput(
                    "elapsed phase calendar profile period must be at least 2".to_string(),
                ));
            }
            let future_phase = values.len() % period;
            let matches = values
                .iter()
                .enumerate()
                .filter_map(|(index, &value)| (index % period == future_phase).then_some(value))
                .collect::<Vec<_>>();
            mean_or_fallback(&matches, fallback)
        }
        _ => {
            return Err(CartoBoostError::InvalidInput(format!(
                "unsupported calendar profile candidate mode '{mode}'"
            )));
        }
    };
    validate_finite_forecast_value(prediction, "calendar profile candidate")
}

pub fn validation_ensemble_weights(
    candidate_scores: &BTreeMap<String, f64>,
) -> Result<BTreeMap<String, f64>> {
    if candidate_scores.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "validation ensemble requires at least one candidate score".to_string(),
        ));
    }
    for (name, score) in candidate_scores {
        if name.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "validation ensemble candidate names must be non-empty".to_string(),
            ));
        }
        if !score.is_finite() || *score < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "validation ensemble score for '{name}' must be finite and non-negative"
            )));
        }
    }
    let mut scores = candidate_scores
        .iter()
        .map(|(name, score)| (name.clone(), *score))
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scores.truncate(4);

    let zero_count = scores.iter().take_while(|(_, score)| *score == 0.0).count();
    if zero_count > 0 {
        let weight = 1.0 / zero_count as f64;
        return Ok(scores
            .into_iter()
            .take(zero_count)
            .map(|(name, _)| (name, weight))
            .collect());
    }

    // Ratio to the best score before squaring. This is algebraically
    // equivalent to inverse-square weighting but avoids overflow for large
    // losses and overflow in 1 / score^2 for very small losses.
    let best_score = scores[0].1;
    let raw = scores
        .iter()
        .map(|(name, score)| (name.clone(), (best_score / score).powi(2)))
        .collect::<BTreeMap<_, _>>();
    normalize(raw)
}

pub fn forecast_magnitude_guard_allows(
    forecast_max_abs: f64,
    training_max_abs: f64,
) -> Result<bool> {
    if !forecast_max_abs.is_finite() || forecast_max_abs < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast magnitude must be finite and non-negative".to_string(),
        ));
    }
    if !training_max_abs.is_finite() || training_max_abs < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "training magnitude must be finite and non-negative".to_string(),
        ));
    }
    let limit = FORECAST_MAGNITUDE_GUARD_ABSOLUTE_FLOOR
        .max(training_max_abs.max(1.0) * FORECAST_MAGNITUDE_GUARD_MULTIPLIER);
    Ok(forecast_max_abs <= limit)
}

pub fn weighted_blend_candidate_forecast(
    primary_forecast: &[f64],
    secondary_forecast: &[f64],
    primary_weight: f64,
) -> Result<Vec<f64>> {
    if primary_forecast.len() != secondary_forecast.len() {
        return Err(CartoBoostError::InvalidInput(
            "weighted blend forecast inputs must have equal length".to_string(),
        ));
    }
    if !primary_weight.is_finite() || !(0.0..=1.0).contains(&primary_weight) {
        return Err(CartoBoostError::InvalidInput(
            "weighted blend forecast weight must be finite and between 0 and 1".to_string(),
        ));
    }
    let secondary_weight = 1.0 - primary_weight;
    primary_forecast
        .iter()
        .zip(secondary_forecast)
        .map(|(&primary, &secondary)| {
            if !primary.is_finite() || !secondary.is_finite() {
                return Err(CartoBoostError::InvalidInput(
                    "weighted blend forecast values must be finite".to_string(),
                ));
            }
            Ok(primary_weight * primary + secondary_weight * secondary)
        })
        .collect()
}

pub fn requires_lag_spine(source: &str, season_length: usize, horizon: usize) -> bool {
    if source != "low_frequency_competition" {
        return false;
    }
    if season_length == 12 {
        return true;
    }
    season_length == 1 && horizon > 6
}

pub fn validation_unavailable_candidate_choice(
    model: &str,
    validation_profile: &str,
    available_candidates: &[String],
) -> Result<String> {
    if model.trim().is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "model name must not be empty".to_string(),
        ));
    }
    if validation_profile.trim().is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "validation profile must not be empty".to_string(),
        ));
    }
    if available_candidates.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "validation fallback requires at least one available candidate".to_string(),
        ));
    }
    if matches!(
        validation_profile,
        "classical_competition" | "classical_competition_full"
    ) && model == "cartoboost_auto_forecast"
        && available_candidates
            .iter()
            .any(|candidate| candidate == "cartoboost_lag")
    {
        return Ok("cartoboost_lag".to_string());
    }
    if available_candidates
        .iter()
        .any(|candidate| candidate == model)
    {
        return Ok(model.to_string());
    }
    available_candidates
        .iter()
        .min_by_key(|candidate| candidate_complexity_rank(candidate))
        .cloned()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "validation fallback requires at least one available candidate".to_string(),
            )
        })
}

pub fn shared_candidate_names() -> Vec<String> {
    [
        "shared_seasonal_base",
        "shared_calendar_dom",
        "shared_calendar_elapsed_phase",
        "shared_drift",
        "shared_half_drift",
        "shared_seasonal_drift",
        "shared_seasonal_cycle_drift_050",
        "shared_seasonal_cycle_drift_075",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

pub fn selectable_candidate_names(model: &str, source: &str) -> Vec<String> {
    let mut candidates = shared_candidate_names();
    if model == "cartoboost_auto_forecast" {
        candidates.insert(0, "cartoboost_lag".to_string());
    }
    if matches!(
        source,
        "classical_competition" | "classical_competition_full"
    ) && model == "cartoboost_auto_forecast"
    {
        candidates.push("cartoboost_autostats_bank".to_string());
    }
    if source == "hierarchical_reconciliation" && model == "cartoboost_auto_forecast" {
        candidates.extend(
            [
                "cartoboost_autostats_bank",
                "shared_calendar_autostats_blend",
                "shared_elapsed_phase_total_reconciled_020",
                "shared_elapsed_phase_total_reconciled_035",
                "shared_elapsed_phase_total_reconciled_050",
                "shared_reconciled_autostats_blend",
                "shared_point_autostats_elapsed_phase_blend",
                "shared_total_reconciled_auto",
            ]
            .into_iter()
            .map(str::to_string),
        );
    }
    if source == "rank_portfolio" && model == "cartoboost_auto_forecast" {
        candidates.extend(
            [
                "shared_market_neutral_zero",
                "shared_elapsed_phase_rank_blend",
                "cartoboost_point_auto",
            ]
            .into_iter()
            .map(str::to_string),
        );
    }
    candidates
}

pub fn include_autostats_candidate(source: &str, season_length: usize, horizon: usize) -> bool {
    if source == "hierarchical_reconciliation" || source == "classical_competition_full" {
        return true;
    }
    source == "classical_competition" && matches!(season_length, 1 | 4 | 12) && horizon <= 24
}

pub fn native_auto_raw_candidate_is_confident(
    selected_candidate: Option<&str>,
    inner_raw_relative_rmse_gain: Option<f64>,
) -> bool {
    selected_candidate == Some("cartoboost_raw")
        && inner_raw_relative_rmse_gain
            .is_some_and(|gain| gain.is_finite() && gain >= NATIVE_AUTO_RAW_KEEP_RELATIVE_GAIN)
}

pub fn relative_loss_displacement_allowed(
    baseline_loss: f64,
    candidate_loss: f64,
    min_relative_gain: f64,
) -> Result<bool> {
    if !baseline_loss.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "baseline loss must be finite".to_string(),
        ));
    }
    if !candidate_loss.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "candidate loss must be finite".to_string(),
        ));
    }
    if !min_relative_gain.is_finite() || !(0.0..=1.0).contains(&min_relative_gain) {
        return Err(CartoBoostError::InvalidInput(
            "displacement relative gain must be finite and between 0 and 1".to_string(),
        ));
    }
    let improvement = baseline_loss - candidate_loss;
    let scale = baseline_loss.abs().max(1.0e-12);
    Ok(improvement / scale >= min_relative_gain)
}

pub fn lag_origin_consistency_guard(
    candidate: &str,
    source: &str,
    lag_scores: &[f64],
    candidate_scores: &[f64],
) -> Result<Option<Value>> {
    if matches!(
        source,
        "classical_competition"
            | "classical_competition_full"
            | "hierarchical_reconciliation"
            | "rank_portfolio"
    ) || candidate == "cartoboost_lag"
    {
        return Ok(None);
    }
    let paired = candidate_scores
        .iter()
        .zip(lag_scores)
        .filter_map(|(&candidate_loss, &lag_loss)| {
            (candidate_loss.is_finite() && lag_loss.is_finite() && lag_loss > 0.0)
                .then_some((candidate_loss, lag_loss))
        })
        .collect::<Vec<_>>();
    if paired.len() < 2 {
        return Ok(None);
    }
    let losing_origin_count = paired
        .iter()
        .filter(|(candidate_loss, lag_loss)| candidate_loss > lag_loss)
        .count();
    if losing_origin_count == 0 {
        return Ok(None);
    }
    let gains = paired
        .iter()
        .map(|(candidate_loss, lag_loss)| 1.0 - candidate_loss / lag_loss)
        .collect::<Vec<_>>();
    let min_relative_gain_vs_lag = gains.iter().copied().fold(f64::INFINITY, f64::min);
    let mean_relative_gain_vs_lag = gains.iter().sum::<f64>() / gains.len() as f64;
    Ok(Some(json!({
        "candidate": candidate,
        "reason": "candidate_lost_at_least_one_inner_origin_to_lag",
        "origin_count": paired.len(),
        "losing_origin_count": losing_origin_count,
        "min_relative_gain_vs_lag": min_relative_gain_vs_lag,
        "mean_relative_gain_vs_lag": mean_relative_gain_vs_lag,
    })))
}

pub fn stable_magnitude_candidate_choice(
    selected_candidate: &str,
    candidate_scores: &BTreeMap<String, f64>,
    candidate_forecast_max_abs: &BTreeMap<String, f64>,
    training_max_abs: f64,
    inner_origin_count: Option<usize>,
) -> Result<String> {
    let mut stable_scores = BTreeMap::new();
    for (candidate, loss) in candidate_scores {
        if !loss.is_finite() {
            continue;
        }
        let Some(forecast_max_abs) = candidate_forecast_max_abs.get(candidate) else {
            continue;
        };
        if forecast_magnitude_guard_allows(*forecast_max_abs, training_max_abs)? {
            stable_scores.insert(candidate.clone(), *loss);
        }
    }
    if stable_scores.contains_key(selected_candidate) {
        return Ok(selected_candidate.to_string());
    }
    if stable_scores.is_empty() {
        return Ok("cartoboost_lag".to_string());
    }
    CandidateSelectionPolicy::new("hierarchical_reconciliation", inner_origin_count)?
        .select(&stable_scores)
        .map(|selection| selection.candidate)
}

fn validate_finite_series(values: &[f64], label: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} values must be finite"
        )));
    }
    Ok(())
}

fn validate_finite_forecast_value(value: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} forecast must be finite"
        )));
    }
    Ok(value)
}

fn mean_or_fallback(values: &[f64], fallback: f64) -> f64 {
    if values.is_empty() {
        fallback
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpertScore {
    pub expert: String,
    pub series_id: Option<String>,
    pub horizon: Option<usize>,
    pub metric: String,
    pub value: f64,
}

impl ExpertScore {
    pub fn global(expert: impl Into<String>, metric: impl Into<String>, value: f64) -> Self {
        Self {
            expert: expert.into(),
            series_id: None,
            horizon: None,
            metric: metric.into(),
            value,
        }
    }

    pub fn for_series(
        expert: impl Into<String>,
        series_id: impl Into<String>,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            expert: expert.into(),
            series_id: Some(series_id.into()),
            horizon: None,
            metric: metric.into(),
            value,
        }
    }

    pub fn for_horizon(
        expert: impl Into<String>,
        horizon: usize,
        metric: impl Into<String>,
        value: f64,
    ) -> Self {
        Self {
            expert: expert.into(),
            series_id: None,
            horizon: Some(horizon),
            metric: metric.into(),
            value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationScoreTable {
    scores: Vec<ExpertScore>,
}

impl ValidationScoreTable {
    pub fn new(scores: Vec<ExpertScore>) -> Result<Self> {
        if scores.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "validation score table requires at least one score".to_string(),
            ));
        }
        let mut seen = BTreeMap::new();
        for score in &scores {
            if score.expert.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "validation score experts must be non-empty".to_string(),
                ));
            }
            if score.metric.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "validation score metrics must be non-empty".to_string(),
                ));
            }
            if !score.value.is_finite() || score.value < 0.0 {
                return Err(CartoBoostError::InvalidInput(
                    "validation scores must be finite and non-negative".to_string(),
                ));
            }
            if let Some(horizon) = score.horizon {
                if horizon == 0 {
                    return Err(CartoBoostError::InvalidInput(
                        "validation score horizons must be positive".to_string(),
                    ));
                }
            }
            let key = (
                score.expert.clone(),
                score.series_id.clone(),
                score.horizon,
                score.metric.clone(),
            );
            if seen.insert(key, ()).is_some() {
                return Err(CartoBoostError::InvalidInput(
                    "duplicate validation score entry".to_string(),
                ));
            }
        }
        Ok(Self { scores })
    }

    pub fn scores(&self) -> &[ExpertScore] {
        &self.scores
    }

    pub fn experts(&self) -> Vec<String> {
        let mut experts = Vec::new();
        for score in &self.scores {
            if !experts.contains(&score.expert) {
                experts.push(score.expert.clone());
            }
        }
        experts
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleBasedGating {
    metric: String,
    score_table: ValidationScoreTable,
    error_floor: f64,
    top_k: Option<usize>,
    hard_winner_relative_gain: Option<f64>,
    min_weight: Option<f64>,
    max_weight: Option<f64>,
    baseline: Option<String>,
    baseline_displacement_gain: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleBasedGatingGuardrails {
    pub error_floor: f64,
    pub top_k: Option<usize>,
    pub hard_winner_relative_gain: Option<f64>,
    pub weight_bounds: Option<(f64, f64)>,
    pub baseline: Option<String>,
    pub baseline_displacement_gain: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateSelection {
    pub candidate: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateSelectionPolicy {
    pub source: String,
    pub inner_origin_count: Option<usize>,
    pub robust_relative_tolerance: f64,
    pub min_relative_gain: f64,
}

impl CandidateSelectionPolicy {
    pub fn new(source: impl Into<String>, inner_origin_count: Option<usize>) -> Result<Self> {
        Self::with_thresholds(
            source,
            inner_origin_count,
            AUTO_SELECTION_ROBUST_RELATIVE_TOLERANCE,
            AUTO_SELECTION_MIN_RELATIVE_GAIN,
        )
    }

    pub fn with_thresholds(
        source: impl Into<String>,
        inner_origin_count: Option<usize>,
        robust_relative_tolerance: f64,
        min_relative_gain: f64,
    ) -> Result<Self> {
        let source = source.into();
        if source.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "candidate selection source must be non-empty".to_string(),
            ));
        }
        validate_relative_gain(robust_relative_tolerance, "robust_relative_tolerance")?;
        validate_relative_gain(min_relative_gain, "min_relative_gain")?;
        Ok(Self {
            source,
            inner_origin_count,
            robust_relative_tolerance,
            min_relative_gain,
        })
    }

    pub fn select(&self, candidate_scores: &BTreeMap<String, f64>) -> Result<CandidateSelection> {
        if candidate_scores.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "candidate selection requires at least one score".to_string(),
            ));
        }
        let candidate = match self.source.as_str() {
            "classical_competition" | "classical_competition_full" | "rank_portfolio" => {
                select_lowest_finite(candidate_scores)
            }
            "hierarchical_reconciliation" => self.select_hierarchical(candidate_scores),
            _ => self.select_robust(candidate_scores),
        };
        Ok(CandidateSelection { candidate })
    }

    fn select_robust(&self, candidate_scores: &BTreeMap<String, f64>) -> String {
        let finite_scores = finite_candidate_scores(candidate_scores);
        if finite_scores.is_empty() {
            return lowest_by_raw_value(candidate_scores);
        }
        let best_loss = finite_scores
            .values()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let tolerance = best_loss * (1.0 + self.robust_relative_tolerance);
        finite_scores
            .iter()
            .filter(|(_, loss)| **loss <= tolerance)
            .min_by(|(left, left_loss), (right, right_loss)| {
                candidate_complexity_rank(left)
                    .cmp(&candidate_complexity_rank(right))
                    .then_with(|| left_loss.total_cmp(right_loss))
                    .then_with(|| left.cmp(right))
            })
            .map(|(candidate, _)| candidate.clone())
            .expect("finite scores are non-empty")
    }

    fn select_hierarchical(&self, candidate_scores: &BTreeMap<String, f64>) -> String {
        let finite_scores = finite_candidate_scores(candidate_scores);
        if finite_scores.is_empty() {
            return self.select_robust(candidate_scores);
        }
        let best_loss = finite_scores
            .values()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let lag_loss = finite_scores.get("cartoboost_lag").copied();
        let clears_lag_guard = |candidate: &str| -> bool {
            if candidate == "cartoboost_lag" {
                return true;
            }
            let Some(lag_loss) = lag_loss else {
                return true;
            };
            if lag_loss <= 0.0 {
                return true;
            }
            let Some(candidate_loss) = finite_scores.get(candidate) else {
                return false;
            };
            1.0 - candidate_loss / lag_loss >= self.min_relative_gain
        };

        let reconciled_blend = "shared_reconciled_autostats_blend";
        if finite_scores.get(reconciled_blend).is_some_and(|loss| {
            (self.inner_origin_count.is_none_or(|count| count <= 1))
                && *loss <= best_loss * 1.001
                && clears_lag_guard(reconciled_blend)
        }) {
            return reconciled_blend.to_string();
        }

        let point_blend = "shared_point_autostats_elapsed_phase_blend";
        if finite_scores.get(point_blend).is_some_and(|loss| {
            (self.inner_origin_count.is_none_or(|count| count <= 1))
                && *loss <= best_loss * 1.015
                && clears_lag_guard(point_blend)
        }) {
            return point_blend.to_string();
        }

        let reconciled = finite_scores
            .iter()
            .filter(|(candidate, loss)| {
                candidate.starts_with("shared_elapsed_phase_total_reconciled_")
                    && **loss <= best_loss * 1.015
                    && clears_lag_guard(candidate)
            })
            .max_by(|(left, left_loss), (right, right_loss)| {
                candidate_complexity_rank(left)
                    .cmp(&candidate_complexity_rank(right))
                    .then_with(|| right_loss.total_cmp(left_loss))
                    .then_with(|| left.cmp(right))
            })
            .map(|(candidate, _)| candidate.clone());
        if let Some(candidate) = reconciled {
            return candidate;
        }

        let selected = finite_scores
            .iter()
            .min_by(|(left, left_loss), (right, right_loss)| {
                left_loss
                    .total_cmp(right_loss)
                    .then_with(|| left.cmp(right))
            })
            .map(|(candidate, _)| candidate.clone())
            .expect("finite scores are non-empty");
        if !clears_lag_guard(&selected) && lag_loss.is_some() {
            return "cartoboost_lag".to_string();
        }
        selected
    }
}

impl Default for RuleBasedGatingGuardrails {
    fn default() -> Self {
        Self {
            error_floor: 1e-9,
            top_k: None,
            hard_winner_relative_gain: None,
            weight_bounds: None,
            baseline: None,
            baseline_displacement_gain: None,
        }
    }
}

pub fn candidate_complexity_rank(candidate: &str) -> usize {
    match candidate {
        "cartoboost_lag" => 0,
        "cartoboost_autostats_bank" => 1,
        "shared_seasonal_base" => 2,
        "shared_half_drift" => 3,
        "shared_drift" => 4,
        "shared_seasonal_drift" => 5,
        "shared_seasonal_cycle_drift_050" => 6,
        "shared_seasonal_cycle_drift_075" => 7,
        "cartoboost_auto_forecast" => 8,
        "cartoboost_validation_weighted_ensemble" => 9,
        "shared_calendar_dom" => 10,
        "shared_calendar_elapsed_phase" => 11,
        "shared_market_neutral_zero" => 12,
        "shared_elapsed_phase_rank_blend" => 13,
        "shared_calendar_autostats_blend" => 14,
        "shared_elapsed_phase_total_reconciled_020" => 15,
        "shared_elapsed_phase_total_reconciled_035" => 16,
        "shared_elapsed_phase_total_reconciled_050" => 17,
        "shared_reconciled_autostats_blend" => 19,
        "shared_point_autostats_elapsed_phase_blend" | "shared_total_reconciled_auto" => 20,
        "cartoboost_point_auto" => 24,
        _ => 20,
    }
}

impl RuleBasedGating {
    pub fn new(metric: impl Into<String>, score_table: ValidationScoreTable) -> Result<Self> {
        Self::with_options(metric, score_table, 1e-9, None)
    }

    pub fn with_options(
        metric: impl Into<String>,
        score_table: ValidationScoreTable,
        error_floor: f64,
        top_k: Option<usize>,
    ) -> Result<Self> {
        let metric = metric.into();
        if metric.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "gating metric must be non-empty".to_string(),
            ));
        }
        if !error_floor.is_finite() || error_floor <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "gating error_floor must be finite and positive".to_string(),
            ));
        }
        if top_k == Some(0) {
            return Err(CartoBoostError::InvalidInput(
                "gating top_k must be positive when provided".to_string(),
            ));
        }
        Ok(Self {
            metric,
            score_table,
            error_floor,
            top_k,
            hard_winner_relative_gain: None,
            min_weight: None,
            max_weight: None,
            baseline: None,
            baseline_displacement_gain: None,
        })
    }

    pub fn with_guardrails(
        metric: impl Into<String>,
        score_table: ValidationScoreTable,
        guardrails: RuleBasedGatingGuardrails,
    ) -> Result<Self> {
        let mut gating = Self::with_options(
            metric,
            score_table,
            guardrails.error_floor,
            guardrails.top_k,
        )?;
        if let Some(gain) = guardrails.hard_winner_relative_gain {
            validate_relative_gain(gain, "hard_winner_relative_gain")?;
        }
        let (min_weight, max_weight) = match guardrails.weight_bounds {
            Some((min_weight, max_weight)) => {
                validate_weight_bounds(min_weight, max_weight)?;
                (Some(min_weight), Some(max_weight))
            }
            None => (None, None),
        };
        if let Some(baseline) = &guardrails.baseline {
            if baseline.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "gating baseline must be non-empty".to_string(),
                ));
            }
        }
        if let Some(gain) = guardrails.baseline_displacement_gain {
            validate_relative_gain(gain, "baseline_displacement_gain")?;
        }
        if guardrails.baseline.is_none() && guardrails.baseline_displacement_gain.is_some() {
            return Err(CartoBoostError::InvalidInput(
                "baseline_displacement_gain requires a baseline".to_string(),
            ));
        }
        gating.hard_winner_relative_gain = guardrails.hard_winner_relative_gain;
        gating.min_weight = min_weight;
        gating.max_weight = max_weight;
        gating.baseline = guardrails.baseline;
        gating.baseline_displacement_gain = guardrails.baseline_displacement_gain;
        Ok(gating)
    }

    pub fn weights_for(
        &self,
        series_id: Option<&str>,
        horizon: Option<usize>,
    ) -> Result<BTreeMap<String, f64>> {
        let mut candidates = self.matching_scores(series_id, horizon);
        if candidates.is_empty() && series_id.is_some() {
            candidates = self.matching_scores(None, horizon);
        }
        if candidates.is_empty() && horizon.is_some() {
            candidates = self.matching_scores(series_id, None);
        }
        if candidates.is_empty() {
            candidates = self.matching_scores(None, None);
        }
        if candidates.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "no validation scores found for gating metric '{}'",
                self.metric
            )));
        }
        candidates.sort_by(|left, right| {
            left.value
                .total_cmp(&right.value)
                .then_with(|| left.expert.cmp(&right.expert))
        });
        if let Some(weights) = self.baseline_guardrail(&candidates)? {
            return Ok(weights);
        }
        if let Some(weights) = self.hard_winner_guardrail(&candidates) {
            return Ok(weights);
        }
        if let Some(top_k) = self.top_k {
            candidates.truncate(top_k.min(candidates.len()));
        }
        let mut raw = BTreeMap::new();
        for score in candidates {
            let weight = 1.0 / score.value.max(self.error_floor);
            raw.insert(score.expert.clone(), weight);
        }
        normalize_with_bounds(raw, self.min_weight, self.max_weight)
    }

    pub fn weights_for_frame(&self, frame: &ForecastFrame) -> Result<BTreeMap<String, f64>> {
        if frame.is_panel() {
            let mut totals: BTreeMap<String, f64> = BTreeMap::new();
            for series_id in frame.series_ids() {
                for (expert, weight) in self.weights_for(Some(&series_id), None)? {
                    *totals.entry(expert).or_insert(0.0) += weight;
                }
            }
            normalize(totals)
        } else {
            self.weights_for(None, None)
        }
    }

    pub fn metadata(&self) -> Value {
        json!({
            "metric": self.metric,
            "error_floor": self.error_floor,
            "top_k": self.top_k,
            "hard_winner_relative_gain": self.hard_winner_relative_gain,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "baseline": self.baseline,
            "baseline_displacement_gain": self.baseline_displacement_gain,
            "scores": self.score_table.scores(),
        })
    }

    fn matching_scores(
        &self,
        series_id: Option<&str>,
        horizon: Option<usize>,
    ) -> Vec<&ExpertScore> {
        self.score_table
            .scores()
            .iter()
            .filter(|score| score.metric == self.metric)
            .filter(|score| score.series_id.as_deref() == series_id)
            .filter(|score| score.horizon == horizon)
            .collect()
    }

    fn baseline_guardrail(
        &self,
        candidates: &[&ExpertScore],
    ) -> Result<Option<BTreeMap<String, f64>>> {
        let Some(baseline) = &self.baseline else {
            return Ok(None);
        };
        let Some(required_gain) = self.baseline_displacement_gain else {
            return Ok(None);
        };
        let baseline_score = candidates
            .iter()
            .find(|score| &score.expert == baseline)
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "gating baseline '{baseline}' has no validation score for metric '{}'",
                    self.metric
                ))
            })?;
        let best = candidates.first().ok_or_else(|| {
            CartoBoostError::InvalidInput("gating requires at least one candidate".to_string())
        })?;
        if best.expert == *baseline {
            return Ok(Some(single_weight(baseline)));
        }
        let displacement_gain = relative_gain(best.value, baseline_score.value);
        if displacement_gain < required_gain {
            return Ok(Some(single_weight(baseline)));
        }
        Ok(None)
    }

    fn hard_winner_guardrail(&self, candidates: &[&ExpertScore]) -> Option<BTreeMap<String, f64>> {
        let required_gain = self.hard_winner_relative_gain?;
        if candidates.len() < 2 {
            return Some(single_weight(&candidates[0].expert));
        }
        let best = candidates[0];
        let second = candidates[1];
        if relative_gain(best.value, second.value) >= required_gain {
            return Some(single_weight(&best.expert));
        }
        None
    }
}

fn normalize(raw: BTreeMap<String, f64>) -> Result<BTreeMap<String, f64>> {
    if raw.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "gating produced no positive expert weights".to_string(),
        ));
    }
    for (expert, weight) in &raw {
        if expert.trim().is_empty() || !weight.is_finite() || *weight < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "gating weights must have non-empty expert names and finite non-negative values"
                    .to_string(),
            ));
        }
    }
    let max_weight = raw.values().copied().fold(0.0_f64, f64::max);
    if max_weight <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "gating produced no positive expert weights".to_string(),
        ));
    }
    let scaled_total = raw.values().map(|weight| weight / max_weight).sum::<f64>();
    if !scaled_total.is_finite() || scaled_total <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "gating weights could not be normalized".to_string(),
        ));
    }
    Ok(raw
        .into_iter()
        .map(|(expert, weight)| (expert, (weight / max_weight) / scaled_total))
        .collect())
}

fn finite_candidate_scores(candidate_scores: &BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    candidate_scores
        .iter()
        .filter(|(_, loss)| loss.is_finite())
        .map(|(candidate, loss)| (candidate.clone(), *loss))
        .collect()
}

fn select_lowest_finite(candidate_scores: &BTreeMap<String, f64>) -> String {
    let finite_scores = finite_candidate_scores(candidate_scores);
    if finite_scores.is_empty() {
        return lowest_by_raw_value(candidate_scores);
    }
    finite_scores
        .iter()
        .min_by(|(left, left_loss), (right, right_loss)| {
            left_loss
                .total_cmp(right_loss)
                .then_with(|| left.cmp(right))
        })
        .map(|(candidate, _)| candidate.clone())
        .expect("finite scores are non-empty")
}

fn lowest_by_raw_value(candidate_scores: &BTreeMap<String, f64>) -> String {
    candidate_scores
        .iter()
        .min_by(|(left, left_loss), (right, right_loss)| {
            left_loss
                .total_cmp(right_loss)
                .then_with(|| left.cmp(right))
        })
        .map(|(candidate, _)| candidate.clone())
        .expect("candidate scores are non-empty")
}

fn normalize_with_bounds(
    raw: BTreeMap<String, f64>,
    min_weight: Option<f64>,
    max_weight: Option<f64>,
) -> Result<BTreeMap<String, f64>> {
    let mut weights = normalize(raw)?;
    let Some(min_weight) = min_weight else {
        return Ok(weights);
    };
    let max_weight = max_weight.expect("weight bounds are set together");
    let n = weights.len();
    let selected_count = n as f64;
    if min_weight * selected_count > 1.0 + 1e-12 || max_weight * selected_count < 1.0 - 1e-12 {
        return Err(CartoBoostError::InvalidInput(
            "gating weight bounds cannot satisfy the number of selected experts".to_string(),
        ));
    }
    // Euclidean projection onto the bounded probability simplex. Clamping and
    // then renormalizing is incorrect: the renormalization can move weights
    // outside the requested bounds.
    let mut lower_shift = weights
        .values()
        .map(|weight| weight - max_weight)
        .fold(f64::INFINITY, f64::min);
    let mut upper_shift = weights
        .values()
        .map(|weight| weight - min_weight)
        .fold(f64::NEG_INFINITY, f64::max);
    for _ in 0..100 {
        let shift = 0.5 * (lower_shift + upper_shift);
        let total = weights
            .values()
            .map(|weight| (weight - shift).clamp(min_weight, max_weight))
            .sum::<f64>();
        if total > 1.0 {
            lower_shift = shift;
        } else {
            upper_shift = shift;
        }
    }
    let shift = 0.5 * (lower_shift + upper_shift);
    for weight in weights.values_mut() {
        *weight = (*weight - shift).clamp(min_weight, max_weight);
    }
    let mut residual = 1.0 - weights.values().sum::<f64>();
    if residual > 0.0 {
        for weight in weights.values_mut() {
            let adjustment = residual.min(max_weight - *weight);
            *weight += adjustment;
            residual -= adjustment;
            if residual <= 1.0e-15 {
                break;
            }
        }
    } else if residual < 0.0 {
        for weight in weights.values_mut() {
            let adjustment = (-residual).min(*weight - min_weight);
            *weight -= adjustment;
            residual += adjustment;
            if residual >= -1.0e-15 {
                break;
            }
        }
    }
    if residual.abs() > 1.0e-12 {
        return Err(CartoBoostError::InvalidInput(
            "gating bounded weights could not be projected onto the probability simplex"
                .to_string(),
        ));
    }
    Ok(weights)
}

fn single_weight(expert: &str) -> BTreeMap<String, f64> {
    BTreeMap::from([(expert.to_string(), 1.0)])
}

fn relative_gain(candidate_loss: f64, baseline_loss: f64) -> f64 {
    if baseline_loss <= 0.0 {
        if candidate_loss <= 0.0 {
            0.0
        } else {
            f64::NEG_INFINITY
        }
    } else {
        ((baseline_loss - candidate_loss) / baseline_loss).max(0.0)
    }
}

fn validate_relative_gain(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(CartoBoostError::InvalidInput(format!(
            "gating {name} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn validate_weight_bounds(min_weight: f64, max_weight: f64) -> Result<()> {
    if !min_weight.is_finite()
        || !max_weight.is_finite()
        || min_weight < 0.0
        || max_weight <= 0.0
        || min_weight > max_weight
        || max_weight > 1.0
    {
        return Err(CartoBoostError::InvalidInput(
            "gating weight bounds must satisfy 0 <= min <= max <= 1".to_string(),
        ));
    }
    Ok(())
}
