use crate::booster::BoosterConfig;
use crate::forecasting::lag_features::{history_by_series, lag_config_supported_by_prior};
use crate::forecasting::{
    CalendarFeature, ClassicalExpertBank, ExpertScore, ForecastActual, ForecastFrame,
    ForecastObjective, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
    IntermittentDemandConfig, IntermittentDemandForecaster, LagFeatureConfig, LagPlusConfig,
    LagPlusForecaster, LocalStandardScaledForecaster, Log1pForecaster,
    RectifiedRecursiveForecaster, RuleBasedGating, RuleBasedGatingGuardrails, ValidationScoreTable,
};
use crate::forecasting::{
    CartoBoostDirectForecaster, CartoBoostLagForecaster, GlobalForecastSampleWeightMode,
    GlobalForecastTargetMode,
};
use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const LAG_EXPERT: &str = "cartoboost_lag";
const RECENCY_WEIGHTED_LAG_EXPERT: &str = "recency_weighted_lag";
const SCALED_LAG_EXPERT: &str = "scaled_lag";
const DELTA_LAG_EXPERT: &str = "delta_lag";
const SCALED_DELTA_LAG_EXPERT: &str = "scaled_delta_lag";
const SEASONAL_DELTA_LAG_EXPERT: &str = "seasonal_delta_lag";
const SCALED_SEASONAL_DELTA_LAG_EXPERT: &str = "scaled_seasonal_delta_lag";
const EWM_LAG_EXPERT: &str = "ewm_lag";
const DIRECT_EXPERT: &str = "cartoboost_direct";
const RECTIFIED_RECURSIVE_EXPERT: &str = "cartoboost_rectified_recursive";
const LOG1P_SCALED_LAG_EXPERT: &str = "log1p_scaled_lag";
const LAG_PLUS_EXPERT: &str = "lag_plus";
const INTERMITTENT_DEMAND_EXPERT: &str = "intermittent_demand";
const CLASSICAL_EXPERT: &str = "classical_expert_bank";
const MIN_AUTO_TRAIN_HISTORY: usize = 4;
const MIN_SERIES_WEIGHT_VALIDATION_POINTS: usize = 4;
// Keep the stable selector focused on candidates with deterministic fit
// behavior across short, sparse panels. The broad classical expert bank is
// still available as a standalone supported model, but is not part of the
// stable AutoForecaster roster because one unstable ARIMA member can make an
// otherwise valid panel selection fail.
const AUTO_CANDIDATES: [&str; 13] = [
    LAG_EXPERT,
    RECENCY_WEIGHTED_LAG_EXPERT,
    SCALED_LAG_EXPERT,
    DELTA_LAG_EXPERT,
    SCALED_DELTA_LAG_EXPERT,
    SEASONAL_DELTA_LAG_EXPERT,
    SCALED_SEASONAL_DELTA_LAG_EXPERT,
    EWM_LAG_EXPERT,
    DIRECT_EXPERT,
    RECTIFIED_RECURSIVE_EXPERT,
    LOG1P_SCALED_LAG_EXPERT,
    LAG_PLUS_EXPERT,
    INTERMITTENT_DEMAND_EXPERT,
];

pub type AutoForecastObjective = ForecastObjective;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoForecastConfig {
    pub lag_config: LagFeatureConfig,
    pub booster_config: BoosterConfig,
    pub target_mode: GlobalForecastTargetMode,
    pub season_length: usize,
    pub validation_window: Option<usize>,
    pub validation_origin_count: usize,
    pub objective: AutoForecastObjective,
    pub baseline_displacement_gain: f64,
    pub hard_winner_relative_gain: f64,
    pub min_blend_weight: f64,
    pub max_blend_weight: f64,
    pub max_direct_horizon: usize,
    pub max_candidate_count: Option<usize>,
    #[serde(default = "default_backend")]
    pub backend: String,
}

impl Default for AutoForecastConfig {
    fn default() -> Self {
        Self {
            lag_config: LagFeatureConfig {
                lags: vec![1, 2, 3, 7, 14, 28],
                rolling_mean_windows: vec![7, 14, 28],
                partial_rolling_mean_windows: Vec::new(),
                rolling_std_windows: vec![7, 14, 28],
                rolling_min_windows: vec![7, 14, 28],
                rolling_max_windows: vec![7, 14, 28],
                ewm_alpha_percents: Vec::new(),
                calendar_features: vec![
                    CalendarFeature::DayOfWeek,
                    CalendarFeature::Month,
                    CalendarFeature::Day,
                ],
                difference_lags: vec![2, 3, 7, 14, 28],
                rolling_trend_windows: vec![7, 14, 28],
                covariate_features: Vec::new(),
                covariate_indicator_values: Default::default(),
                covariate_calendar_interactions: false,
            },
            booster_config: BoosterConfig::default(),
            target_mode: GlobalForecastTargetMode::Level,
            season_length: 7,
            validation_window: None,
            validation_origin_count: 2,
            objective: ForecastObjective::RmseWape,
            baseline_displacement_gain: 0.03,
            hard_winner_relative_gain: 0.05,
            min_blend_weight: 0.15,
            max_blend_weight: 0.85,
            max_direct_horizon: 28,
            max_candidate_count: None,
            backend: default_backend(),
        }
    }
}

fn default_backend() -> String {
    "cpu".to_string()
}

pub struct AutoForecastModel {
    config: AutoForecastConfig,
    fitted: Option<FittedAutoForecastModel>,
}

impl Clone for AutoForecastModel {
    fn clone(&self) -> Self {
        // Backtesting clones an unfitted model for each fold. A fitted auto
        // selector contains trait-object experts and is intentionally not
        // copied across folds; each fold reruns native selection.
        Self {
            config: self.config.clone(),
            fitted: None,
        }
    }
}

struct FittedAutoForecastModel {
    members: Vec<FittedMember>,
    weights: BTreeMap<String, f64>,
    horizon_weights: BTreeMap<usize, BTreeMap<String, f64>>,
    series_weights: BTreeMap<String, BTreeMap<String, f64>>,
    effective_lag_config: LagFeatureConfig,
    validation_scores: Vec<ExpertScore>,
    validation_window: usize,
    validation_origin_count: usize,
    member_metadata: BTreeMap<String, Value>,
    nonnegative_output: bool,
}

struct FittedMember {
    name: String,
    forecaster: Box<dyn Forecaster>,
}

enum DirectAutoMember {
    Direct(Box<CartoBoostDirectForecaster>),
    Rectified(Box<RectifiedRecursiveForecaster>),
}

struct FixedHorizonDirectForecaster {
    member: DirectAutoMember,
    fit_horizon: usize,
    model_name: &'static str,
}

impl FixedHorizonDirectForecaster {
    fn new(member: DirectAutoMember, fit_horizon: usize, model_name: &'static str) -> Result<Self> {
        if fit_horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "auto direct fit horizon must be positive".to_string(),
            ));
        }
        Ok(Self {
            member,
            fit_horizon,
            model_name,
        })
    }
}

impl Forecaster for FixedHorizonDirectForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        match &mut self.member {
            DirectAutoMember::Direct(model) => model.fit_horizon(frame, self.fit_horizon),
            DirectAutoMember::Rectified(model) => model.fit_horizon(frame, self.fit_horizon),
        }
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        match &self.member {
            DirectAutoMember::Direct(model) => model.predict(horizon),
            DirectAutoMember::Rectified(model) => model.predict(horizon),
        }
    }

    fn model_name(&self) -> &'static str {
        self.model_name
    }

    fn metadata(&self) -> Value {
        let inner = match &self.member {
            DirectAutoMember::Direct(model) => model.metadata(),
            DirectAutoMember::Rectified(model) => model.metadata(),
        };
        json!({
            "model": self.model_name(),
            "fit_horizon": self.fit_horizon,
            "inner": inner,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ForecastKey {
    series_id: String,
    timestamp: chrono::NaiveDateTime,
    horizon: usize,
}

impl AutoForecastModel {
    pub fn new(config: AutoForecastConfig) -> Result<Self> {
        validate_config(&config)?;
        Ok(Self {
            config,
            fitted: None,
        })
    }

    pub fn config(&self) -> &AutoForecastConfig {
        &self.config
    }

    pub fn weights(&self) -> Option<&BTreeMap<String, f64>> {
        self.fitted.as_ref().map(|state| &state.weights)
    }
}

impl Forecaster for AutoForecastModel {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let validation_window = effective_validation_window(frame, self.config.validation_window)?;
        let validation_origin_count = effective_validation_origin_count(
            frame,
            validation_window,
            self.config.validation_origin_count,
        )?;
        let nonnegative_output = frame_is_nonnegative(frame.rows());
        let splits = rolling_validation_splits(frame, validation_window, validation_origin_count)?;
        let oldest_split = splits.first().ok_or_else(|| {
            CartoBoostError::InvalidInput("no auto forecast validation splits".to_string())
        })?;
        let effective_config =
            effective_auto_config_for_split(&self.config, &oldest_split.train, validation_window);
        let scores = score_auto_candidates(frame, &effective_config, &splits, nonnegative_output)?;
        if scores.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "no auto forecast model candidate could be validated".to_string(),
            ));
        }
        let baseline = scores
            .iter()
            .any(|score| score.expert == LAG_EXPERT)
            .then(|| LAG_EXPERT.to_string());
        let gating = RuleBasedGating::with_guardrails(
            self.config.objective.as_str(),
            ValidationScoreTable::new(scores.clone())?,
            RuleBasedGatingGuardrails {
                top_k: Some(2),
                hard_winner_relative_gain: Some(self.config.hard_winner_relative_gain),
                weight_bounds: Some((self.config.min_blend_weight, self.config.max_blend_weight)),
                baseline,
                baseline_displacement_gain: scores
                    .iter()
                    .any(|score| score.expert == LAG_EXPERT)
                    .then_some(self.config.baseline_displacement_gain),
                ..RuleBasedGatingGuardrails::default()
            },
        )?;
        let weights = gating.weights_for(None, None)?;
        let horizon_weights = weights_by_horizon(&gating, validation_window)?;
        let series_weights =
            weights_by_series(&gating, frame, validation_window, validation_origin_count)?;
        let selected_names = selected_member_names(&weights, &horizon_weights, &series_weights);
        let (members, member_metadata) =
            fit_selected_members(&selected_names, &effective_config, frame)?;
        self.fitted = Some(FittedAutoForecastModel {
            members,
            weights,
            horizon_weights,
            series_weights,
            effective_lag_config: effective_config.lag_config,
            validation_scores: scores,
            validation_window,
            validation_origin_count,
            member_metadata,
            nonnegative_output,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let fitted = self.fitted.as_ref().ok_or_else(|| {
            CartoBoostError::InvalidInput("auto forecast model must be fitted".to_string())
        })?;
        let mut weighted: BTreeMap<ForecastKey, f64> = BTreeMap::new();
        let mut expected_keys: Option<Vec<ForecastKey>> = None;
        for member in &fitted.members {
            let result = member.forecaster.predict(horizon)?;
            let mut current_keys = Vec::with_capacity(result.predictions().len());
            for prediction in result.predictions() {
                let weight = member_weight_for_prediction(
                    fitted,
                    &member.name,
                    &prediction.series_id,
                    prediction.horizon,
                );
                let key = ForecastKey {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                };
                current_keys.push(key.clone());
                *weighted.entry(key).or_insert(0.0) += weight * prediction.mean;
            }
            if let Some(expected) = &expected_keys {
                if expected != &current_keys {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "auto forecast member '{}' produced a mismatched forecast index",
                        member.name
                    )));
                }
            } else {
                expected_keys = Some(current_keys);
            }
        }
        let predictions = weighted
            .into_iter()
            .map(|(key, mean)| ForecastPrediction {
                series_id: key.series_id,
                timestamp: key.timestamp,
                horizon: key.horizon,
                model: self.model_name().to_string(),
                mean: maybe_clamp_nonnegative(mean, fitted.nonnegative_output),
            })
            .collect();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "cartoboost_auto_forecast"
    }

    fn metadata(&self) -> Value {
        let fitted = self.fitted.as_ref();
        json!({
            "model": self.model_name(),
            "objective": self.config.objective.as_str(),
            "season_length": self.config.season_length,
            "validation_window": fitted.map(|state| state.validation_window),
            "validation_origin_count": fitted.map(|state| state.validation_origin_count),
            "series_weight_min_validation_points": MIN_SERIES_WEIGHT_VALIDATION_POINTS,
            "baseline": LAG_EXPERT,
            "baseline_displacement_gain": self.config.baseline_displacement_gain,
            "hard_winner_relative_gain": self.config.hard_winner_relative_gain,
            "max_direct_horizon": self.config.max_direct_horizon,
            "nonnegative_output": fitted.map(|state| state.nonnegative_output),
            "effective_lag_config": fitted.map(|state| &state.effective_lag_config),
            "blend_bounds": {
                "min": self.config.min_blend_weight,
                "max": self.config.max_blend_weight,
            },
            "weights": fitted.map(|state| &state.weights),
            "horizon_weights": fitted.map(|state| &state.horizon_weights),
            "series_weights": fitted.map(|state| &state.series_weights),
            "validation_scores": fitted.map(|state| &state.validation_scores),
            "members": fitted.map(|state| &state.member_metadata),
        })
    }
}

struct ValidationSplit {
    train: ForecastFrame,
    validation: Vec<ForecastRow>,
}

type ValidationScoreKey = (String, Option<String>, Option<usize>, String);

fn build_candidate(name: &str, config: &AutoForecastConfig) -> Result<Box<dyn Forecaster>> {
    build_candidate_with_direct_horizon(name, config, config.max_direct_horizon)
}

struct SelectedMemberFit {
    index: usize,
    name: String,
    member: FittedMember,
    metadata: Value,
}

fn fit_selected_members(
    selected_names: &[String],
    config: &AutoForecastConfig,
    frame: &ForecastFrame,
) -> Result<(Vec<FittedMember>, BTreeMap<String, Value>)> {
    let mut fitted = selected_names
        .par_iter()
        .enumerate()
        .map(|(index, name)| {
            let member_config = effective_config_for_candidate(name, config, frame);
            let mut forecaster = build_candidate(name, &member_config)?;
            forecaster.fit(frame)?;
            let metadata = forecaster.metadata();
            Ok(SelectedMemberFit {
                index,
                name: name.clone(),
                member: FittedMember {
                    name: name.clone(),
                    forecaster,
                },
                metadata,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    fitted.sort_by_key(|member| member.index);

    let mut members = Vec::with_capacity(fitted.len());
    let mut member_metadata = BTreeMap::new();
    for fit in fitted {
        member_metadata.insert(fit.name, fit.metadata);
        members.push(fit.member);
    }
    Ok((members, member_metadata))
}

fn build_candidate_with_direct_horizon(
    name: &str,
    config: &AutoForecastConfig,
    direct_horizon: usize,
) -> Result<Box<dyn Forecaster>> {
    match name {
        LAG_EXPERT => Ok(Box::new(CartoBoostLagForecaster::new_with_backend(
            config.lag_config.clone(),
            config.booster_config.clone(),
            config.target_mode,
            GlobalForecastSampleWeightMode::Uniform,
            Some(&config.backend),
        )?)),
        RECENCY_WEIGHTED_LAG_EXPERT => Ok(Box::new(CartoBoostLagForecaster::new_with_backend(
            config.lag_config.clone(),
            config.booster_config.clone(),
            config.target_mode,
            GlobalForecastSampleWeightMode::ExponentialRecency {
                half_life: recency_half_life(config),
            },
            Some(&config.backend),
        )?)),
        SCALED_LAG_EXPERT => Ok(Box::new(LocalStandardScaledForecaster::new(
            Box::new(CartoBoostLagForecaster::new_with_backend(
                config.lag_config.clone(),
                config.booster_config.clone(),
                config.target_mode,
                GlobalForecastSampleWeightMode::Uniform,
                Some(&config.backend),
            )?),
            1e-6,
            SCALED_LAG_EXPERT,
        )?)),
        DELTA_LAG_EXPERT => Ok(Box::new(CartoBoostLagForecaster::new_with_backend(
            config.lag_config.clone(),
            config.booster_config.clone(),
            GlobalForecastTargetMode::DeltaFromLast,
            GlobalForecastSampleWeightMode::Uniform,
            Some(&config.backend),
        )?)),
        SCALED_DELTA_LAG_EXPERT => Ok(Box::new(LocalStandardScaledForecaster::new(
            Box::new(CartoBoostLagForecaster::new_with_backend(
                config.lag_config.clone(),
                config.booster_config.clone(),
                GlobalForecastTargetMode::DeltaFromLast,
                GlobalForecastSampleWeightMode::Uniform,
                Some(&config.backend),
            )?),
            1e-6,
            SCALED_DELTA_LAG_EXPERT,
        )?)),
        SEASONAL_DELTA_LAG_EXPERT => Ok(Box::new(CartoBoostLagForecaster::new_with_backend(
            config.lag_config.clone(),
            config.booster_config.clone(),
            GlobalForecastTargetMode::SeasonalDelta {
                season_length: config.season_length,
            },
            GlobalForecastSampleWeightMode::Uniform,
            Some(&config.backend),
        )?)),
        SCALED_SEASONAL_DELTA_LAG_EXPERT => Ok(Box::new(LocalStandardScaledForecaster::new(
            Box::new(CartoBoostLagForecaster::new_with_backend(
                config.lag_config.clone(),
                config.booster_config.clone(),
                GlobalForecastTargetMode::SeasonalDelta {
                    season_length: config.season_length,
                },
                GlobalForecastSampleWeightMode::Uniform,
                Some(&config.backend),
            )?),
            1e-6,
            SCALED_SEASONAL_DELTA_LAG_EXPERT,
        )?)),
        EWM_LAG_EXPERT => {
            let mut lag_config = config.lag_config.clone();
            push_unique_u8(&mut lag_config.ewm_alpha_percents, 90);
            sort_dedup_u8(&mut lag_config.ewm_alpha_percents);
            Ok(Box::new(CartoBoostLagForecaster::new_with_backend(
                lag_config,
                config.booster_config.clone(),
                config.target_mode,
                GlobalForecastSampleWeightMode::Uniform,
                Some(&config.backend),
            )?))
        }
        DIRECT_EXPERT => Ok(Box::new(FixedHorizonDirectForecaster::new(
            DirectAutoMember::Direct(Box::new(CartoBoostDirectForecaster::new_with_backend(
                config.lag_config.clone(),
                config.booster_config.clone(),
                Some(&config.backend),
            )?)),
            direct_horizon,
            DIRECT_EXPERT,
        )?)),
        RECTIFIED_RECURSIVE_EXPERT => Ok(Box::new(FixedHorizonDirectForecaster::new(
            DirectAutoMember::Rectified(Box::new(RectifiedRecursiveForecaster::new_with_backend(
                config.lag_config.clone(),
                config.booster_config.clone(),
                Some(&config.backend),
            )?)),
            direct_horizon,
            RECTIFIED_RECURSIVE_EXPERT,
        )?)),
        LOG1P_SCALED_LAG_EXPERT => Ok(Box::new(Log1pForecaster::new(
            Box::new(LocalStandardScaledForecaster::new(
                Box::new(CartoBoostLagForecaster::new_with_backend(
                    config.lag_config.clone(),
                    config.booster_config.clone(),
                    config.target_mode,
                    GlobalForecastSampleWeightMode::Uniform,
                    Some(&config.backend),
                )?),
                1e-6,
                "log1p_scaled_lag_transformed",
            )?),
            LOG1P_SCALED_LAG_EXPERT,
        ))),
        LAG_PLUS_EXPERT => Ok(Box::new(LagPlusForecaster::new(LagPlusConfig {
            lag_config: config.lag_config.clone(),
            booster_config: config.booster_config.clone(),
            target_mode: config.target_mode,
            validation_window: config.validation_window,
            objective: config.objective,
            shrinkage_strength: 4.0,
            seasonal_bucket_period: Some(config.season_length),
            backend: config.backend.clone(),
        })?)),
        INTERMITTENT_DEMAND_EXPERT => Ok(Box::new(IntermittentDemandForecaster::new(
            IntermittentDemandConfig {
                validation_window: config.validation_window,
                objective: config.objective,
                ..IntermittentDemandConfig::default()
            },
        )?)),
        CLASSICAL_EXPERT => Ok(Box::new(ClassicalExpertBank::with_validation_window(
            ClassicalExpertBank::default_for_season_length(config.season_length)?
                .experts()
                .to_vec(),
            Some(config.validation_window.unwrap_or(1).max(1)),
        )?)),
        other => Err(CartoBoostError::InvalidInput(format!(
            "unknown auto forecast candidate '{other}'"
        ))),
    }
}

fn candidate_is_eligible(name: &str, frame: &ForecastFrame, config: &AutoForecastConfig) -> bool {
    match name {
        RECENCY_WEIGHTED_LAG_EXPERT => recent_level_shift_present(frame, config),
        LOG1P_SCALED_LAG_EXPERT => frame_is_nonnegative(frame.rows()),
        INTERMITTENT_DEMAND_EXPERT => {
            frame_is_nonnegative(frame.rows()) && zero_fraction(frame.rows()) >= 0.25
        }
        DIRECT_EXPERT => {
            zero_fraction(frame.rows()) < 0.25 && direct_history_sufficient(frame, config)
        }
        RECTIFIED_RECURSIVE_EXPERT => {
            zero_fraction(frame.rows()) < 0.25
                && direct_history_sufficient(frame, config)
                && minimum_series_history(frame) > config.max_direct_horizon.saturating_add(2)
        }
        _ => true,
    }
}

fn minimum_series_history(frame: &ForecastFrame) -> usize {
    history_by_series(frame.rows())
        .values()
        .map(Vec::len)
        .min()
        .unwrap_or(0)
}

fn direct_history_sufficient(frame: &ForecastFrame, config: &AutoForecastConfig) -> bool {
    let min_history = minimum_series_history(frame);
    if min_history == 0 {
        return false;
    }
    if min_history <= config.max_direct_horizon {
        return false;
    }
    let validation_window = config
        .validation_window
        .unwrap_or((min_history / 5).clamp(1, 8))
        .max(1);
    let max_origins = min_history.saturating_sub(MIN_AUTO_TRAIN_HISTORY) / validation_window;
    let origin_count = config.validation_origin_count.min(max_origins).max(1);
    let first_train_len = min_history.saturating_sub(validation_window * origin_count);
    first_train_len > MIN_AUTO_TRAIN_HISTORY
}

fn recency_half_life(config: &AutoForecastConfig) -> usize {
    config
        .season_length
        .max(config.validation_window.unwrap_or(1))
        .max(2)
}

fn recent_level_shift_present(frame: &ForecastFrame, config: &AutoForecastConfig) -> bool {
    let window = recency_half_life(config).clamp(2, 28);
    let mut eligible = 0usize;
    let mut shifted = 0usize;
    for history in history_by_series(frame.rows()).values() {
        if history.len() < window * 2 {
            continue;
        }
        eligible += 1;
        let recent = &history[history.len() - window..];
        let prior = &history[history.len() - window * 2..history.len() - window];
        let recent_mean = recent.iter().map(|row| row.target).sum::<f64>() / window as f64;
        let prior_mean = prior.iter().map(|row| row.target).sum::<f64>() / window as f64;
        let scale = prior
            .iter()
            .chain(recent.iter())
            .map(|row| row.target.abs())
            .sum::<f64>()
            / (window * 2) as f64;
        if scale > 0.0 && ((recent_mean - prior_mean).abs() / scale) >= 0.35 {
            shifted += 1;
        }
    }
    eligible > 0 && shifted * 4 >= eligible
}

fn zero_fraction(rows: &[ForecastRow]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    rows.iter().filter(|row| row.target == 0.0).count() as f64 / rows.len() as f64
}

fn frame_is_nonnegative(rows: &[ForecastRow]) -> bool {
    rows.iter().all(|row| row.target >= 0.0)
}

fn weights_by_series(
    gating: &RuleBasedGating,
    frame: &ForecastFrame,
    validation_window: usize,
    validation_origin_count: usize,
) -> Result<BTreeMap<String, BTreeMap<String, f64>>> {
    let mut by_series = BTreeMap::new();
    if !frame.is_panel() {
        return Ok(by_series);
    }
    let validation_points = validation_window.saturating_mul(validation_origin_count);
    if validation_points < MIN_SERIES_WEIGHT_VALIDATION_POINTS {
        return Ok(by_series);
    }
    for series_id in frame.series_ids() {
        by_series.insert(
            series_id.clone(),
            gating.weights_for(Some(&series_id), None)?,
        );
    }
    Ok(by_series)
}

fn weights_by_horizon(
    gating: &RuleBasedGating,
    validation_window: usize,
) -> Result<BTreeMap<usize, BTreeMap<String, f64>>> {
    let mut by_horizon = BTreeMap::new();
    for horizon in 1..=validation_window {
        by_horizon.insert(horizon, gating.weights_for(None, Some(horizon))?);
    }
    Ok(by_horizon)
}

fn selected_member_names(
    weights: &BTreeMap<String, f64>,
    horizon_weights: &BTreeMap<usize, BTreeMap<String, f64>>,
    series_weights: &BTreeMap<String, BTreeMap<String, f64>>,
) -> Vec<String> {
    let mut selected = BTreeSet::new();
    for name in weights.keys() {
        selected.insert(name.clone());
    }
    for weights in horizon_weights.values() {
        for name in weights.keys() {
            selected.insert(name.clone());
        }
    }
    for weights in series_weights.values() {
        for name in weights.keys() {
            selected.insert(name.clone());
        }
    }
    selected.into_iter().collect()
}

fn member_weight_for_prediction(
    fitted: &FittedAutoForecastModel,
    member_name: &str,
    series_id: &str,
    horizon: usize,
) -> f64 {
    if let Some(weights) = fitted.series_weights.get(series_id) {
        return weights.get(member_name).copied().unwrap_or(0.0);
    }
    if let Some(weights) = fitted.horizon_weights.get(&horizon) {
        return weights.get(member_name).copied().unwrap_or(0.0);
    }
    fitted.weights.get(member_name).copied().unwrap_or(0.0)
}

fn score_candidate(
    name: &str,
    config: &AutoForecastConfig,
    split: &ValidationSplit,
    nonnegative_output: bool,
) -> Result<Vec<ExpertScore>> {
    let horizon = validation_horizon(&split.validation);
    let minimum_history = history_by_series(split.train.rows())
        .values()
        .map(Vec::len)
        .min()
        .unwrap_or(0);
    let mut effective_config = config.clone();
    effective_config.validation_window = Some(horizon);
    let max_supported_prior = candidate_max_supported_prior(name, minimum_history, horizon);
    effective_config.lag_config =
        lag_config_supported_by_prior(&config.lag_config, max_supported_prior);
    let mut candidate = build_candidate_with_direct_horizon(name, &effective_config, horizon)?;
    candidate.fit(&split.train)?;
    let predictions = maybe_clamp_result(candidate.predict(horizon)?, nonnegative_output)?;
    let actuals = validation_actuals(&split.validation);
    let metrics = crate::forecasting::evaluate_forecast(&predictions, &actuals)?;
    let mut scores = vec![ExpertScore::global(
        name,
        config.objective.as_str(),
        config.objective.metric_value(&metrics),
    )];
    scores.extend(horizon_scores(
        name,
        config.objective,
        &predictions,
        &actuals,
    )?);
    scores.extend(series_scores(
        name,
        config.objective,
        &predictions,
        &actuals,
    )?);
    Ok(scores)
}

fn score_auto_candidates(
    frame: &ForecastFrame,
    config: &AutoForecastConfig,
    splits: &[ValidationSplit],
    nonnegative_output: bool,
) -> Result<Vec<ExpertScore>> {
    let mut scored = auto_candidate_roster(config)
        .par_iter()
        .enumerate()
        .map(|(index, name)| {
            if !candidate_is_eligible(name, frame, config)
                || !splits
                    .iter()
                    .all(|split| candidate_is_eligible_for_split(name, split, config))
            {
                return Ok(None);
            }
            let candidate_config = effective_config_for_candidate(name, config, frame);
            score_candidate_across_splits(name, &candidate_config, splits, nonnegative_output)
                .map(|scores| Some((index, scores)))
                .map_err(|error| {
                    CartoBoostError::InvalidInput(format!(
                        "auto forecast candidate {name:?} failed validation: {error}"
                    ))
                })
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    scored.sort_by_key(|(index, _)| *index);
    Ok(scored.into_iter().flat_map(|(_, scores)| scores).collect())
}

fn auto_candidate_roster(config: &AutoForecastConfig) -> Vec<&'static str> {
    let max_candidate_count = config
        .max_candidate_count
        .unwrap_or(AUTO_CANDIDATES.len())
        .min(AUTO_CANDIDATES.len());
    AUTO_CANDIDATES[..max_candidate_count].to_vec()
}

fn candidate_max_supported_prior(name: &str, minimum_history: usize, horizon: usize) -> usize {
    match name {
        DIRECT_EXPERT => minimum_history.saturating_sub(horizon),
        RECTIFIED_RECURSIVE_EXPERT => {
            let internal_holdout = (minimum_history / 5).clamp(1, 28).max(horizon);
            minimum_history
                .saturating_sub(internal_holdout)
                .saturating_sub(1)
        }
        LAG_PLUS_EXPERT => minimum_history.saturating_sub(horizon).saturating_sub(1),
        _ => minimum_history.saturating_sub(1),
    }
}

fn effective_config_for_candidate(
    name: &str,
    config: &AutoForecastConfig,
    full_frame: &ForecastFrame,
) -> AutoForecastConfig {
    let mut effective = config.clone();
    if !matches!(name, DIRECT_EXPERT | RECTIFIED_RECURSIVE_EXPERT) {
        return effective;
    }
    let full_history = minimum_series_history(full_frame);
    let validation_window = config.validation_window.unwrap_or(1);
    let oldest_history = full_history
        .saturating_sub(validation_window.saturating_mul(config.validation_origin_count));
    let score_support = candidate_max_supported_prior(name, oldest_history, validation_window);
    let full_support = candidate_max_supported_prior(name, full_history, config.max_direct_horizon);
    effective.lag_config =
        lag_config_supported_by_prior(&config.lag_config, score_support.min(full_support));
    effective
}

fn candidate_is_eligible_for_split(
    name: &str,
    split: &ValidationSplit,
    config: &AutoForecastConfig,
) -> bool {
    let minimum_history = minimum_series_history(&split.train);
    let horizon = validation_horizon(&split.validation);
    match name {
        SEASONAL_DELTA_LAG_EXPERT | SCALED_SEASONAL_DELTA_LAG_EXPERT => {
            minimum_history > config.season_length
        }
        DIRECT_EXPERT => minimum_history > horizon,
        RECTIFIED_RECURSIVE_EXPERT => minimum_history > horizon.saturating_add(1),
        LAG_PLUS_EXPERT => minimum_history > horizon,
        _ => true,
    }
}

fn score_candidate_across_splits(
    name: &str,
    config: &AutoForecastConfig,
    splits: &[ValidationSplit],
    nonnegative_output: bool,
) -> Result<Vec<ExpertScore>> {
    let mut scores_by_key: BTreeMap<ValidationScoreKey, Vec<f64>> = BTreeMap::new();
    for split in splits {
        for score in score_candidate(name, config, split, nonnegative_output)? {
            scores_by_key
                .entry((score.expert, score.series_id, score.horizon, score.metric))
                .or_default()
                .push(score.value);
        }
    }
    let mut scores = Vec::with_capacity(scores_by_key.len());
    for ((expert, series_id, horizon, metric), values) in scores_by_key {
        let value = values.iter().sum::<f64>() / values.len() as f64;
        scores.push(ExpertScore {
            expert,
            series_id,
            horizon,
            metric,
            value,
        });
    }
    Ok(scores)
}

fn horizon_scores(
    name: &str,
    objective: ForecastObjective,
    predictions: &ForecastResult,
    actuals: &[ForecastActual],
) -> Result<Vec<ExpertScore>> {
    let mut actuals_by_horizon: BTreeMap<usize, Vec<ForecastActual>> = BTreeMap::new();
    for actual in actuals {
        actuals_by_horizon
            .entry(actual.horizon)
            .or_default()
            .push(actual.clone());
    }
    let mut predictions_by_horizon: BTreeMap<usize, Vec<ForecastPrediction>> = BTreeMap::new();
    for prediction in predictions.predictions() {
        predictions_by_horizon
            .entry(prediction.horizon)
            .or_default()
            .push(prediction.clone());
    }
    let mut scores = Vec::with_capacity(actuals_by_horizon.len());
    for (horizon, horizon_actuals) in actuals_by_horizon {
        let horizon_predictions = predictions_by_horizon.get(&horizon).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "auto forecast candidate {name:?} omitted horizon {horizon} predictions"
            ))
        })?;
        let metrics = crate::forecasting::evaluate_forecast(
            &ForecastResult::new(horizon_predictions.clone())?,
            &horizon_actuals,
        )?;
        scores.push(ExpertScore::for_horizon(
            name,
            horizon,
            objective.as_str(),
            objective.metric_value(&metrics),
        ));
    }
    Ok(scores)
}

fn maybe_clamp_result(result: ForecastResult, nonnegative_output: bool) -> Result<ForecastResult> {
    if !nonnegative_output {
        return Ok(result);
    }
    let predictions = result
        .predictions()
        .iter()
        .map(|prediction| ForecastPrediction {
            series_id: prediction.series_id.clone(),
            timestamp: prediction.timestamp,
            horizon: prediction.horizon,
            model: prediction.model.clone(),
            mean: maybe_clamp_nonnegative(prediction.mean, true),
        })
        .collect();
    ForecastResult::new(predictions)
}

fn maybe_clamp_nonnegative(value: f64, nonnegative_output: bool) -> f64 {
    if nonnegative_output {
        value.max(0.0)
    } else {
        value
    }
}

fn series_scores(
    name: &str,
    objective: ForecastObjective,
    predictions: &ForecastResult,
    actuals: &[ForecastActual],
) -> Result<Vec<ExpertScore>> {
    let mut actuals_by_series: BTreeMap<String, Vec<ForecastActual>> = BTreeMap::new();
    for actual in actuals {
        actuals_by_series
            .entry(actual.series_id.clone())
            .or_default()
            .push(actual.clone());
    }
    if actuals_by_series.len() <= 1 {
        return Ok(Vec::new());
    }
    let mut predictions_by_series: BTreeMap<String, Vec<ForecastPrediction>> = BTreeMap::new();
    for prediction in predictions.predictions() {
        predictions_by_series
            .entry(prediction.series_id.clone())
            .or_default()
            .push(prediction.clone());
    }
    let mut scores = Vec::with_capacity(actuals_by_series.len());
    for (series_id, series_actuals) in actuals_by_series {
        let series_predictions = predictions_by_series.get(&series_id).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "auto forecast candidate {name:?} omitted predictions for series {series_id:?}"
            ))
        })?;
        let metrics = crate::forecasting::evaluate_forecast(
            &ForecastResult::new(series_predictions.clone())?,
            &series_actuals,
        )?;
        scores.push(ExpertScore::for_series(
            name,
            series_id,
            objective.as_str(),
            objective.metric_value(&metrics),
        ));
    }
    Ok(scores)
}

fn rolling_validation_splits(
    frame: &ForecastFrame,
    validation_window: usize,
    origin_count: usize,
) -> Result<Vec<ValidationSplit>> {
    if origin_count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast validation_origin_count must be positive".to_string(),
        ));
    }
    let mut splits = Vec::with_capacity(origin_count);
    for origin_index in 0..origin_count {
        splits.push(split_validation_frame_at_origin(
            frame,
            validation_window,
            origin_count - origin_index,
        )?);
    }
    Ok(splits)
}

fn split_validation_frame_at_origin(
    frame: &ForecastFrame,
    validation_window: usize,
    windows_from_end: usize,
) -> Result<ValidationSplit> {
    let mut train_rows = Vec::new();
    let mut validation_rows = Vec::new();
    for (_, rows) in history_by_series(frame.rows()) {
        let validation_end = rows
            .len()
            .checked_sub(validation_window * (windows_from_end - 1));
        let validation_start = rows.len().checked_sub(validation_window * windows_from_end);
        let (Some(validation_start), Some(validation_end)) = (validation_start, validation_end)
        else {
            return Err(CartoBoostError::InvalidInput(
                "not enough history for auto forecast rolling validation".to_string(),
            ));
        };
        if validation_start == 0 || validation_start >= validation_end {
            return Err(CartoBoostError::InvalidInput(
                "not enough history for auto forecast validation".to_string(),
            ));
        }
        train_rows.extend(rows[..validation_start].iter().cloned());
        validation_rows.extend(rows[validation_start..validation_end].iter().cloned());
    }
    Ok(ValidationSplit {
        train: ForecastFrame::with_metadata(
            train_rows,
            frame.frequency(),
            frame.metadata().clone(),
        )?,
        validation: validation_rows,
    })
}

fn validation_actuals(validation: &[ForecastRow]) -> Vec<ForecastActual> {
    let mut actuals = Vec::with_capacity(validation.len());
    for (_, rows) in history_by_series(validation) {
        for (index, row) in rows.iter().enumerate() {
            actuals.push(ForecastActual {
                series_id: row.series_id.clone(),
                timestamp: row.timestamp,
                horizon: index + 1,
                actual: row.target,
            });
        }
    }
    actuals
}

fn validation_horizon(validation: &[ForecastRow]) -> usize {
    history_by_series(validation)
        .values()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
}

fn effective_validation_window(frame: &ForecastFrame, configured: Option<usize>) -> Result<usize> {
    let min_history = history_by_series(frame.rows())
        .values()
        .map(Vec::len)
        .min()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "auto forecast requires at least one series for validation".to_string(),
            )
        })?;
    let max_feasible = min_history.checked_sub(MIN_AUTO_TRAIN_HISTORY).ok_or_else(|| {
        CartoBoostError::InvalidInput(format!(
            "auto forecast requires more than {MIN_AUTO_TRAIN_HISTORY} rows per series for a real holdout; minimum history is {min_history}"
        ))
    })?;
    if max_feasible == 0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "auto forecast requires more than {MIN_AUTO_TRAIN_HISTORY} rows per series for a real holdout; minimum history is {min_history}"
        )));
    }
    let automatic = (min_history / 5).clamp(1, 8);
    match configured {
        Some(window) if window <= max_feasible => Ok(window),
        Some(window) => Err(CartoBoostError::InvalidInput(format!(
            "auto forecast validation_window must be <= {max_feasible} to retain {MIN_AUTO_TRAIN_HISTORY} training rows per series; got {window}"
        ))),
        None => Ok(automatic.min(max_feasible)),
    }
}

fn effective_validation_origin_count(
    frame: &ForecastFrame,
    validation_window: usize,
    configured: usize,
) -> Result<usize> {
    let min_history = history_by_series(frame.rows())
        .values()
        .map(Vec::len)
        .min()
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "auto forecast requires at least one series for validation".to_string(),
            )
        })?;
    let max_origins = min_history.saturating_sub(MIN_AUTO_TRAIN_HISTORY) / validation_window;
    if configured > max_origins {
        return Err(CartoBoostError::InvalidInput(format!(
            "auto forecast validation_origin_count must be <= {max_origins} for validation_window={validation_window} while retaining {MIN_AUTO_TRAIN_HISTORY} training rows per series; got {configured}"
        )));
    }
    Ok(configured)
}

fn effective_auto_config_for_split(
    config: &AutoForecastConfig,
    train: &ForecastFrame,
    validation_window: usize,
) -> AutoForecastConfig {
    let mut effective = config.clone();
    effective.validation_window = Some(validation_window);
    let train_history = minimum_series_history(train);
    let roster = auto_candidate_roster(config);
    let mut max_supported_prior = train_history.saturating_sub(1);
    if roster.contains(&LAG_PLUS_EXPERT) {
        max_supported_prior = max_supported_prior.min(
            train_history
                .saturating_sub(validation_window)
                .saturating_sub(1),
        );
    }
    expand_lag_config_for_season(
        &mut effective.lag_config,
        config.season_length,
        max_supported_prior,
    );
    effective
}

fn expand_lag_config_for_season(
    lag_config: &mut LagFeatureConfig,
    season_length: usize,
    max_supported_window: usize,
) {
    *lag_config = lag_config_supported_by_prior(lag_config, max_supported_window);
    if season_length <= 1 || max_supported_window < season_length {
        sort_dedup_lag_config(lag_config);
        return;
    }
    for multiple in 1..=4 {
        let Some(window) = season_length.checked_mul(multiple) else {
            break;
        };
        if window > max_supported_window {
            break;
        }
        push_unique(&mut lag_config.lags, window);
        push_unique(&mut lag_config.rolling_mean_windows, window);
        push_unique(&mut lag_config.rolling_std_windows, window);
        push_unique(&mut lag_config.rolling_min_windows, window);
        push_unique(&mut lag_config.rolling_max_windows, window);
        push_unique(&mut lag_config.rolling_trend_windows, window.max(2));
        if window > 1 {
            push_unique(&mut lag_config.difference_lags, window);
        }
    }
    sort_dedup_lag_config(lag_config);
}

fn sort_dedup_lag_config(lag_config: &mut LagFeatureConfig) {
    sort_dedup(&mut lag_config.lags);
    sort_dedup(&mut lag_config.rolling_mean_windows);
    sort_dedup(&mut lag_config.rolling_std_windows);
    sort_dedup(&mut lag_config.rolling_min_windows);
    sort_dedup(&mut lag_config.rolling_max_windows);
    sort_dedup_u8(&mut lag_config.ewm_alpha_percents);
    sort_dedup(&mut lag_config.difference_lags);
    sort_dedup(&mut lag_config.rolling_trend_windows);
}

fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn sort_dedup(values: &mut Vec<usize>) {
    values.sort_unstable();
    values.dedup();
}

fn push_unique_u8(values: &mut Vec<u8>, value: u8) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn sort_dedup_u8(values: &mut Vec<u8>) {
    values.sort_unstable();
    values.dedup();
}

fn validate_config(config: &AutoForecastConfig) -> Result<()> {
    if config.season_length == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast season_length must be positive".to_string(),
        ));
    }
    if matches!(config.validation_window, Some(0)) {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast validation_window must be positive".to_string(),
        ));
    }
    if config.validation_origin_count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast validation_origin_count must be positive".to_string(),
        ));
    }
    if config.max_direct_horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast max_direct_horizon must be positive".to_string(),
        ));
    }
    if !config.baseline_displacement_gain.is_finite()
        || !(0.0..=1.0).contains(&config.baseline_displacement_gain)
    {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast baseline_displacement_gain must be between 0 and 1".to_string(),
        ));
    }
    if !config.hard_winner_relative_gain.is_finite()
        || !(0.0..=1.0).contains(&config.hard_winner_relative_gain)
    {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast hard_winner_relative_gain must be between 0 and 1".to_string(),
        ));
    }
    if !config.min_blend_weight.is_finite()
        || !config.max_blend_weight.is_finite()
        || config.min_blend_weight < 0.0
        || config.max_blend_weight > 1.0
        || config.min_blend_weight > config.max_blend_weight
    {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast blend weights must satisfy 0 <= min <= max <= 1".to_string(),
        ));
    }
    if matches!(config.max_candidate_count, Some(0)) {
        return Err(CartoBoostError::InvalidInput(
            "auto forecast max_candidate_count must be positive".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::parse_forecast_timestamp;

    fn ts(day: u32) -> chrono::NaiveDateTime {
        parse_forecast_timestamp(&format!("2026-01-{day:02}")).expect("timestamp")
    }

    #[test]
    fn series_weights_are_authoritative_when_present() {
        let fitted = FittedAutoForecastModel {
            members: Vec::new(),
            weights: BTreeMap::from([
                ("cartoboost_lag".to_string(), 0.6),
                ("cartoboost_direct".to_string(), 0.4),
            ]),
            horizon_weights: BTreeMap::from([(
                2,
                BTreeMap::from([("cartoboost_direct".to_string(), 1.0)]),
            )]),
            series_weights: BTreeMap::from([(
                "pickup_zone_1".to_string(),
                BTreeMap::from([("cartoboost_lag".to_string(), 1.0)]),
            )]),
            effective_lag_config: LagFeatureConfig::default(),
            validation_scores: Vec::new(),
            validation_window: 2,
            validation_origin_count: 2,
            member_metadata: BTreeMap::new(),
            nonnegative_output: true,
        };

        assert_eq!(
            member_weight_for_prediction(&fitted, "cartoboost_lag", "pickup_zone_1", 2),
            1.0
        );
        assert_eq!(
            member_weight_for_prediction(&fitted, "cartoboost_direct", "pickup_zone_1", 2),
            0.0
        );
        assert_eq!(
            member_weight_for_prediction(&fitted, "cartoboost_direct", "unknown_zone", 2),
            1.0
        );
        assert_eq!(
            member_weight_for_prediction(&fitted, "cartoboost_direct", "unknown_zone", 9),
            0.4
        );
    }

    #[test]
    fn season_aware_lag_config_adds_supported_multiples() {
        let mut config = LagFeatureConfig {
            lags: vec![1],
            rolling_mean_windows: Vec::new(),
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: Vec::new(),
            rolling_min_windows: Vec::new(),
            rolling_max_windows: Vec::new(),
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: Vec::new(),
            rolling_trend_windows: Vec::new(),
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        };

        expand_lag_config_for_season(&mut config, 12, 40);

        assert_eq!(config.lags, vec![1, 12, 24, 36]);
        assert_eq!(config.rolling_mean_windows, vec![12, 24, 36]);
        assert_eq!(config.rolling_std_windows, vec![12, 24, 36]);
        assert_eq!(config.rolling_min_windows, vec![12, 24, 36]);
        assert_eq!(config.rolling_max_windows, vec![12, 24, 36]);
        assert_eq!(config.difference_lags, vec![12, 24, 36]);
        assert_eq!(config.rolling_trend_windows, vec![12, 24, 36]);
    }

    #[test]
    fn season_aware_lag_config_skips_unsupported_short_history_windows() {
        let mut config = LagFeatureConfig {
            lags: vec![1, 7],
            rolling_mean_windows: vec![7],
            partial_rolling_mean_windows: Vec::new(),
            rolling_std_windows: vec![7],
            rolling_min_windows: vec![7],
            rolling_max_windows: vec![7],
            ewm_alpha_percents: Vec::new(),
            calendar_features: Vec::new(),
            difference_lags: vec![7],
            rolling_trend_windows: vec![7],
            covariate_features: Vec::new(),
            covariate_indicator_values: Default::default(),
            covariate_calendar_interactions: false,
        };

        expand_lag_config_for_season(&mut config, 24, 16);

        assert_eq!(config.lags, vec![1, 7]);
        assert_eq!(config.rolling_mean_windows, vec![7]);
        assert_eq!(config.rolling_std_windows, vec![7]);
        assert_eq!(config.rolling_min_windows, vec![7]);
        assert_eq!(config.rolling_max_windows, vec![7]);
        assert_eq!(config.difference_lags, vec![7]);
        assert_eq!(config.rolling_trend_windows, vec![7]);
    }

    #[test]
    fn auto_candidate_scoring_keeps_fixed_roster_order() {
        let frame = ForecastFrame::new(
            (1..=18)
                .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
                .collect(),
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let config = AutoForecastConfig {
            lag_config: LagFeatureConfig {
                lags: vec![1, 2],
                rolling_mean_windows: vec![2],
                partial_rolling_mean_windows: Vec::new(),
                rolling_std_windows: Vec::new(),
                rolling_min_windows: Vec::new(),
                rolling_max_windows: Vec::new(),
                ewm_alpha_percents: Vec::new(),
                calendar_features: Vec::new(),
                difference_lags: vec![2],
                rolling_trend_windows: vec![2],
                covariate_features: Vec::new(),
                covariate_indicator_values: Default::default(),
                covariate_calendar_interactions: false,
            },
            validation_window: Some(2),
            objective: ForecastObjective::RmseWape,
            season_length: 7,
            ..AutoForecastConfig::default()
        };
        let splits = rolling_validation_splits(&frame, 2, 2).expect("splits");
        let effective =
            effective_auto_config_for_split(&config, &splits.first().expect("oldest").train, 2);

        let scores =
            score_auto_candidates(&frame, &effective, &splits, true).expect("candidate scoring");
        let global_experts = scores
            .iter()
            .filter(|score| score.series_id.is_none() && score.horizon.is_none())
            .map(|score| score.expert.as_str())
            .collect::<Vec<_>>();

        let positions = global_experts
            .iter()
            .map(|expert| {
                AUTO_CANDIDATES
                    .iter()
                    .position(|candidate| candidate == expert)
                    .expect("candidate")
            })
            .collect::<Vec<_>>();
        let sorted = {
            let mut values = positions.clone();
            values.sort_unstable();
            values
        };
        assert!(!global_experts.is_empty());
        assert_eq!(positions, sorted);
    }

    #[test]
    fn auto_candidate_scoring_respects_candidate_budget() {
        let config = AutoForecastConfig {
            max_candidate_count: Some(2),
            ..AutoForecastConfig::default()
        };
        assert_eq!(
            auto_candidate_roster(&config),
            vec![LAG_EXPERT, RECENCY_WEIGHTED_LAG_EXPERT],
        );
    }

    #[test]
    fn auto_validation_origin_count_must_preserve_training_history() {
        let short_frame = ForecastFrame::new(
            (1..=7)
                .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
                .collect(),
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let long_frame = ForecastFrame::new(
            (1..=16)
                .map(|day| ForecastRow::single(ts(day), 10.0 + f64::from(day)))
                .collect(),
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");

        assert_eq!(
            effective_validation_origin_count(&short_frame, 3, 1).expect("one origin"),
            1
        );
        assert!(effective_validation_origin_count(&short_frame, 3, 2).is_err());
        assert_eq!(
            effective_validation_origin_count(&long_frame, 3, 4).expect("four origins"),
            4
        );
    }

    #[test]
    fn auto_forecast_rejects_oversized_explicit_validation_window() {
        let frame = ForecastFrame::new(
            ["PU4-DO7", "PU24-DO48", "PU132-DO236"]
                .into_iter()
                .flat_map(|series_id| {
                    (1..=20).map(move |day| {
                        ForecastRow::new(
                            series_id,
                            ts(day),
                            10.0 + f64::from(day % 7) + f64::from(series_id.len() as u32),
                        )
                    })
                })
                .collect(),
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let mut model = AutoForecastModel::new(AutoForecastConfig {
            lag_config: LagFeatureConfig {
                lags: vec![1, 24],
                rolling_mean_windows: vec![24],
                partial_rolling_mean_windows: Vec::new(),
                rolling_std_windows: Vec::new(),
                rolling_min_windows: Vec::new(),
                rolling_max_windows: Vec::new(),
                ewm_alpha_percents: Vec::new(),
                calendar_features: Vec::new(),
                difference_lags: Vec::new(),
                rolling_trend_windows: Vec::new(),
                covariate_features: Vec::new(),
                covariate_indicator_values: Default::default(),
                covariate_calendar_interactions: false,
            },
            validation_window: Some(24),
            validation_origin_count: 2,
            objective: ForecastObjective::RmseWape,
            season_length: 24,
            max_direct_horizon: 14,
            ..AutoForecastConfig::default()
        })
        .expect("model");

        let error = model
            .fit(&frame)
            .expect_err("explicit validation window must not be capped");
        assert!(error
            .to_string()
            .contains("validation_window must be <= 16"));
    }

    #[test]
    fn rolling_validation_splits_use_ordered_non_overlapping_origins() {
        let frame = ForecastFrame::new(
            (1..=8)
                .map(|day| ForecastRow::single(ts(day), f64::from(day)))
                .collect(),
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");

        let splits = rolling_validation_splits(&frame, 2, 2).expect("splits");

        assert_eq!(splits.len(), 2);
        assert_eq!(
            splits[0]
                .validation
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>(),
            vec![5.0, 6.0]
        );
        assert_eq!(
            splits[1]
                .validation
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>(),
            vec![7.0, 8.0]
        );
        assert_eq!(splits[0].train.rows().len(), 4);
        assert_eq!(splits[1].train.rows().len(), 6);
    }

    #[test]
    fn selected_member_fitting_keeps_requested_order() {
        let frame = ForecastFrame::new(
            (1..=20)
                .map(|day| ForecastRow::single(ts(day), 12.0 + f64::from(day % 7)))
                .collect(),
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let config = AutoForecastConfig {
            lag_config: LagFeatureConfig {
                lags: vec![1, 2],
                rolling_mean_windows: vec![2],
                partial_rolling_mean_windows: Vec::new(),
                rolling_std_windows: Vec::new(),
                rolling_min_windows: Vec::new(),
                rolling_max_windows: Vec::new(),
                ewm_alpha_percents: Vec::new(),
                calendar_features: Vec::new(),
                difference_lags: vec![2],
                rolling_trend_windows: vec![2],
                covariate_features: Vec::new(),
                covariate_indicator_values: Default::default(),
                covariate_calendar_interactions: false,
            },
            validation_window: Some(2),
            objective: ForecastObjective::RmseWape,
            season_length: 7,
            max_direct_horizon: 2,
            ..AutoForecastConfig::default()
        };
        let selected = vec![
            SCALED_LAG_EXPERT.to_string(),
            LAG_EXPERT.to_string(),
            DELTA_LAG_EXPERT.to_string(),
        ];

        let (members, metadata) =
            fit_selected_members(&selected, &config, &frame).expect("fit members");

        let member_names = members
            .iter()
            .map(|member| member.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            member_names,
            vec![SCALED_LAG_EXPERT, LAG_EXPERT, DELTA_LAG_EXPERT]
        );
        for name in selected {
            assert!(metadata.contains_key(&name));
        }
    }

    #[test]
    fn ewm_lag_candidate_fits_with_validation_gated_feature_config() {
        let frame = ForecastFrame::new(
            (1..=20)
                .map(|day| ForecastRow::single(ts(day), 12.0 + f64::from(day % 7)))
                .collect(),
            crate::forecasting::ForecastFrequency::Daily,
        )
        .expect("frame");
        let config = AutoForecastConfig {
            lag_config: LagFeatureConfig {
                lags: vec![1, 2],
                rolling_mean_windows: vec![2],
                partial_rolling_mean_windows: Vec::new(),
                rolling_std_windows: Vec::new(),
                rolling_min_windows: Vec::new(),
                rolling_max_windows: Vec::new(),
                ewm_alpha_percents: Vec::new(),
                calendar_features: Vec::new(),
                difference_lags: vec![2],
                rolling_trend_windows: vec![2],
                covariate_features: Vec::new(),
                covariate_indicator_values: Default::default(),
                covariate_calendar_interactions: false,
            },
            validation_window: Some(2),
            objective: ForecastObjective::RmseWape,
            season_length: 7,
            ..AutoForecastConfig::default()
        };
        let selected = vec![EWM_LAG_EXPERT.to_string()];

        let (_members, metadata) =
            fit_selected_members(&selected, &config, &frame).expect("fit EWM member");

        let ewm_metadata = metadata.get(EWM_LAG_EXPERT).expect("EWM metadata");
        assert_eq!(
            ewm_metadata["lag_config"]["ewm_alpha_percents"],
            serde_json::json!([90])
        );
    }
}
