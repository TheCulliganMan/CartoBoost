use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::local::{
    AutoARIMAForecaster, AutoETSForecaster, AutoKalmanForecaster, AutoLocalLevelKalmanForecaster,
    ETSForecaster, KalmanForecaster, LocalLevelKalmanForecaster, NaiveForecaster,
    OptimizedThetaForecaster, SeasonalNaiveForecaster, SeasonalWindowAverageForecaster,
    ThetaForecaster, ThetaSeasonality, WindowAverageForecaster,
};
use crate::forecasting::metrics::competition_mase_scale;
use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum ClassicalExpert {
    Naive,
    SeasonalNaive {
        season_length: usize,
    },
    WindowAverage {
        window_size: usize,
    },
    SeasonalWindowAverage {
        season_length: usize,
        window_count: usize,
    },
    Theta {
        theta: f64,
        alpha: f64,
    },
    OptimizedTheta {
        season_length: Option<usize>,
    },
    ETS {
        alpha: f64,
        beta: f64,
    },
    SeasonalETS {
        alpha: f64,
        beta: f64,
        gamma: f64,
        season_length: usize,
    },
    AutoETS {
        season_length: usize,
    },
    AutoARIMA {
        max_p: usize,
        max_d: usize,
    },
    LocalLevelKalman,
    Kalman,
    AutoLocalLevelKalman,
    AutoKalman,
}

pub struct ClassicalExpertBank {
    experts: Vec<ClassicalExpert>,
    validation_window: Option<usize>,
    validation_objective: ClassicalExpertValidationObjective,
    fitted: Option<FittedClassicalBankState>,
}

struct FittedClassicalBankState {
    selected: ClassicalExpert,
    fitted: Box<dyn Forecaster>,
    scores: Vec<ClassicalExpertScore>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassicalExpertScore {
    pub expert: ClassicalExpert,
    pub mse: f64,
    pub validation_loss: f64,
    pub validation_rows: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassicalExpertValidationObjective {
    MeanSquaredError,
    SmapeMaseAverage { seasonality: usize },
}

impl ClassicalExpertValidationObjective {
    pub fn name(self) -> &'static str {
        match self {
            Self::MeanSquaredError => "mean_squared_error",
            Self::SmapeMaseAverage { .. } => "smape_mase_average",
        }
    }
}

impl ClassicalExpert {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Naive => "naive",
            Self::SeasonalNaive { .. } => "seasonal_naive",
            Self::WindowAverage { .. } => "window_average",
            Self::SeasonalWindowAverage { .. } => "seasonal_window_average",
            Self::Theta { .. } => "theta",
            Self::OptimizedTheta { .. } => "optimized_theta",
            Self::ETS { .. } => "ets",
            Self::SeasonalETS { .. } => "seasonal_ets",
            Self::AutoETS { .. } => "auto_ets",
            Self::AutoARIMA { .. } => "auto_arima",
            Self::LocalLevelKalman => "local_level_kalman",
            Self::Kalman => "kalman",
            Self::AutoLocalLevelKalman => "auto_local_level_kalman",
            Self::AutoKalman => "auto_kalman",
        }
    }

    pub fn build(&self) -> Result<Box<dyn Forecaster>> {
        match *self {
            Self::Naive => Ok(Box::new(NaiveForecaster::new())),
            Self::SeasonalNaive { season_length } => {
                Ok(Box::new(SeasonalNaiveForecaster::new(season_length)?))
            }
            Self::WindowAverage { window_size } => {
                Ok(Box::new(WindowAverageForecaster::new(window_size)?))
            }
            Self::SeasonalWindowAverage {
                season_length,
                window_count,
            } => Ok(Box::new(SeasonalWindowAverageForecaster::new(
                season_length,
                window_count,
            )?)),
            Self::Theta { theta, alpha } => Ok(Box::new(ThetaForecaster::new(theta, alpha)?)),
            Self::OptimizedTheta { season_length } => {
                let seasonality = season_length.map(ThetaSeasonality::additive).transpose()?;
                Ok(Box::new(OptimizedThetaForecaster::with_seasonality(
                    vec![1.0, 1.5, 2.0, 2.5, 3.0],
                    vec![0.1, 0.2, 0.4, 0.6, 0.8],
                    seasonality,
                )?))
            }
            Self::ETS { alpha, beta } => Ok(Box::new(ETSForecaster::new(alpha, beta)?)),
            Self::SeasonalETS {
                alpha,
                beta,
                gamma,
                season_length,
            } => Ok(Box::new(ETSForecaster::with_additive_seasonality(
                alpha,
                beta,
                Some(gamma),
                Some(season_length),
            )?)),
            Self::AutoETS { season_length } => {
                Ok(Box::new(AutoETSForecaster::new(Some(season_length))?))
            }
            Self::AutoARIMA { max_p, max_d } => {
                Ok(Box::new(AutoARIMAForecaster::new(max_p, max_d)?))
            }
            Self::LocalLevelKalman => Ok(Box::new(LocalLevelKalmanForecaster::default())),
            Self::Kalman => Ok(Box::new(KalmanForecaster::default())),
            Self::AutoLocalLevelKalman => Ok(Box::new(AutoLocalLevelKalmanForecaster::default())),
            Self::AutoKalman => Ok(Box::new(AutoKalmanForecaster::default())),
        }
    }

    pub fn metadata(&self) -> Value {
        match self {
            Self::Naive => json!({"model": self.name()}),
            Self::SeasonalNaive { season_length } => {
                json!({"model": self.name(), "season_length": season_length})
            }
            Self::WindowAverage { window_size } => {
                json!({"model": self.name(), "window_size": window_size})
            }
            Self::SeasonalWindowAverage {
                season_length,
                window_count,
            } => {
                json!({
                    "model": self.name(),
                    "season_length": season_length,
                    "window_count": window_count,
                })
            }
            Self::Theta { theta, alpha } => {
                json!({"model": self.name(), "theta": theta, "alpha": alpha})
            }
            Self::OptimizedTheta { season_length } => {
                json!({"model": self.name(), "season_length": season_length})
            }
            Self::ETS { alpha, beta } => {
                json!({"model": self.name(), "alpha": alpha, "beta": beta})
            }
            Self::SeasonalETS {
                alpha,
                beta,
                gamma,
                season_length,
            } => {
                json!({
                    "model": self.name(),
                    "alpha": alpha,
                    "beta": beta,
                    "gamma": gamma,
                    "season_length": season_length,
                })
            }
            Self::AutoETS { season_length } => {
                json!({"model": self.name(), "season_length": season_length})
            }
            Self::AutoARIMA { max_p, max_d } => {
                json!({"model": self.name(), "max_p": max_p, "max_d": max_d})
            }
            Self::LocalLevelKalman
            | Self::Kalman
            | Self::AutoLocalLevelKalman
            | Self::AutoKalman => json!({"model": self.name()}),
        }
    }
}

impl ClassicalExpertBank {
    pub fn new(experts: Vec<ClassicalExpert>) -> Result<Self> {
        Self::with_validation_window(experts, None)
    }

    pub fn with_validation_window(
        experts: Vec<ClassicalExpert>,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        Self::with_validation_options(
            experts,
            validation_window,
            ClassicalExpertValidationObjective::MeanSquaredError,
        )
    }

    pub fn with_validation_options(
        experts: Vec<ClassicalExpert>,
        validation_window: Option<usize>,
        validation_objective: ClassicalExpertValidationObjective,
    ) -> Result<Self> {
        if experts.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "classical expert bank requires at least one expert".to_string(),
            ));
        }
        if matches!(validation_window, Some(0)) {
            return Err(CartoBoostError::InvalidInput(
                "validation_window must be positive when provided".to_string(),
            ));
        }
        if matches!(
            validation_objective,
            ClassicalExpertValidationObjective::SmapeMaseAverage { seasonality: 0 }
        ) {
            return Err(CartoBoostError::InvalidInput(
                "smape_mase_average seasonality must be positive".to_string(),
            ));
        }
        for (idx, expert) in experts.iter().enumerate() {
            expert.build().map_err(|error| {
                CartoBoostError::InvalidInput(format!(
                    "classical expert {} at index {idx} is invalid: {error}",
                    expert.name()
                ))
            })?;
            if experts[..idx].iter().any(|previous| previous == expert) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "classical expert bank contains duplicate expert {} at index {idx}",
                    expert.name()
                )));
            }
        }
        Ok(Self {
            experts,
            validation_window,
            validation_objective,
            fitted: None,
        })
    }

    pub fn default_for_season_length(season_length: usize) -> Result<Self> {
        Self::with_validation_window(default_experts(season_length), None)
    }

    pub fn experts(&self) -> &[ClassicalExpert] {
        &self.experts
    }

    pub fn selected_expert(&self) -> Option<&ClassicalExpert> {
        self.fitted.as_ref().map(|state| &state.selected)
    }

    pub fn validation_scores(&self) -> &[ClassicalExpertScore] {
        self.fitted
            .as_ref()
            .map(|state| state.scores.as_slice())
            .unwrap_or(&[])
    }
}

impl Forecaster for ClassicalExpertBank {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let validation_window = self.effective_validation_window(frame)?;
        let split = split_validation_frame(frame, validation_window)?;
        let minimum_train_history = history_by_series(split.train.rows())
            .values()
            .map(Vec::len)
            .min()
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "classical expert bank validation train split is empty".to_string(),
                )
            })?;
        let mut scores = Vec::new();
        for expert in &self.experts {
            if expert_supports_history(expert, minimum_train_history) {
                // A single unstable ARIMA/ETS candidate must not invalidate an
                // otherwise usable classical bank. Only candidates that fit
                // and score on the leakage-safe validation split participate.
                if let Ok(score) = score_expert(
                    expert,
                    &split.train,
                    &split.validation,
                    self.validation_objective,
                ) {
                    scores.push(score);
                }
            }
        }
        if scores.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "no classical expert has enough training history for an honest validation score"
                    .to_string(),
            ));
        }
        scores.sort_by_key(|score| {
            (
                OrderedF64(score.validation_loss),
                expert_rank(&score.expert),
                score.expert.name(),
            )
        });
        let mut fitted_choice = None;
        for score in &scores {
            let candidate = score.expert.clone();
            let Ok(mut fitted) = candidate.build() else {
                continue;
            };
            if fitted.fit(frame).is_ok() {
                fitted_choice = Some((candidate, fitted));
                break;
            }
        }
        let (selected, fitted) = fitted_choice.ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "no classical expert could be refit on the complete frame".to_string(),
            )
        })?;
        self.fitted = Some(FittedClassicalBankState {
            selected,
            fitted,
            scores,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.fitted.predict(horizon)?;
        let predictions = result
            .predictions()
            .iter()
            .map(|prediction| ForecastPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: self.model_name().to_string(),
                mean: prediction.mean,
            })
            .collect::<Vec<_>>();
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "classical_expert_bank"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "validation_window": self.validation_window,
            "validation_objective": self.validation_objective.name(),
            "experts": self.experts.iter().map(ClassicalExpert::metadata).collect::<Vec<_>>(),
            "selected_expert": self.selected_expert().map(ClassicalExpert::metadata),
            "validation_scores": self.validation_scores().iter().map(|score| {
                json!({
                    "expert": score.expert.metadata(),
                    "mse": score.mse,
                    "validation_loss": score.validation_loss,
                    "validation_rows": score.validation_rows,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Default for ClassicalExpertBank {
    fn default() -> Self {
        Self::default_for_season_length(24).expect("default classical bank is valid")
    }
}

struct ValidationSplit {
    train: ForecastFrame,
    validation: Vec<ForecastRow>,
}

fn default_experts(season_length: usize) -> Vec<ClassicalExpert> {
    let mut experts = vec![
        ClassicalExpert::Naive,
        ClassicalExpert::WindowAverage { window_size: 3 },
        ClassicalExpert::WindowAverage { window_size: 7 },
        ClassicalExpert::Theta {
            theta: 2.0,
            alpha: 0.3,
        },
        ClassicalExpert::Theta {
            theta: 2.0,
            alpha: 0.5,
        },
        ClassicalExpert::OptimizedTheta {
            season_length: None,
        },
        ClassicalExpert::ETS {
            alpha: 0.3,
            beta: 0.1,
        },
        ClassicalExpert::ETS {
            alpha: 0.5,
            beta: 0.1,
        },
        ClassicalExpert::AutoARIMA { max_p: 2, max_d: 1 },
        ClassicalExpert::LocalLevelKalman,
        ClassicalExpert::Kalman,
        ClassicalExpert::AutoLocalLevelKalman,
        ClassicalExpert::AutoKalman,
    ];
    if season_length > 1 {
        experts.extend([
            ClassicalExpert::SeasonalNaive { season_length },
            ClassicalExpert::SeasonalWindowAverage {
                season_length,
                window_count: 3,
            },
            ClassicalExpert::OptimizedTheta {
                season_length: Some(season_length),
            },
            ClassicalExpert::SeasonalETS {
                alpha: 0.3,
                beta: 0.1,
                gamma: 0.1,
                season_length,
            },
            ClassicalExpert::SeasonalETS {
                alpha: 0.5,
                beta: 0.1,
                gamma: 0.1,
                season_length,
            },
            ClassicalExpert::SeasonalETS {
                alpha: 0.5,
                beta: 0.1,
                gamma: 0.3,
                season_length,
            },
            ClassicalExpert::AutoETS { season_length },
        ]);
    }
    experts
}

fn split_validation_frame(
    frame: &ForecastFrame,
    validation_window: usize,
) -> Result<ValidationSplit> {
    let mut train_rows = Vec::new();
    let mut validation_rows = Vec::new();
    for (_, rows) in history_by_series(frame.rows()) {
        if rows.len() <= validation_window {
            return Err(CartoBoostError::InvalidInput(
                "not enough history for classical bank validation".to_string(),
            ));
        }
        let split_at = rows.len() - validation_window;
        train_rows.extend(rows[..split_at].iter().cloned());
        validation_rows.extend(rows[split_at..].iter().cloned());
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

fn score_expert(
    expert: &ClassicalExpert,
    train: &ForecastFrame,
    validation: &[ForecastRow],
    objective: ClassicalExpertValidationObjective,
) -> Result<ClassicalExpertScore> {
    let horizon = validation_horizon(validation);
    let mut model = expert.build()?;
    model.fit(train)?;
    let predictions = model.predict(horizon)?;
    let mut squared_error = 0.0;
    let mut abs_error = 0.0;
    let mut smape_sum = 0.0;
    let mut smape_count = 0usize;
    let mut count = 0usize;
    for actual in validation {
        let train_len = train.rows_for_series(&actual.series_id).len();
        let validation_idx = validation
            .iter()
            .filter(|row| row.series_id == actual.series_id && row.timestamp <= actual.timestamp)
            .count();
        let expected_horizon = validation_idx;
        if train_len == 0 || expected_horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "classical validation produced an invalid train length or horizon".to_string(),
            ));
        }
        let prediction = predictions
            .predictions()
            .iter()
            .find(|prediction| {
                prediction.series_id == actual.series_id && prediction.horizon == expected_horizon
            })
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "classical expert {} omitted validation prediction for series {}, horizon {}",
                    expert.name(),
                    actual.series_id,
                    expected_horizon
                ))
            })?;
        let err = prediction.mean - actual.target;
        squared_error += err * err;
        abs_error += err.abs();
        let denominator = prediction.mean.abs() + actual.target.abs();
        if denominator > 0.0 {
            smape_sum += 2.0 * err.abs() / denominator;
            smape_count += 1;
        }
        count += 1;
    }
    if count != validation.len() {
        return Err(CartoBoostError::InvalidInput(
            "classical expert did not produce a complete validation prediction set".to_string(),
        ));
    }
    let mse = squared_error / count as f64;
    let validation_loss = match objective {
        ClassicalExpertValidationObjective::MeanSquaredError => mse,
        ClassicalExpertValidationObjective::SmapeMaseAverage { seasonality } => {
            let training_series = train
                .series_ids()
                .iter()
                .map(|series_id| {
                    train
                        .rows_for_series(series_id)
                        .iter()
                        .map(|row| row.target)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let scale = competition_mase_scale(&training_series, seasonality)?;
            let mase = (abs_error / count as f64) / scale.max(1.0e-12);
            let smape = if smape_count == 0 {
                0.0
            } else {
                smape_sum / smape_count as f64
            };
            0.5 * (smape + mase)
        }
    };
    Ok(ClassicalExpertScore {
        expert: expert.clone(),
        mse,
        validation_loss,
        validation_rows: count,
    })
}

fn validation_horizon(validation: &[ForecastRow]) -> usize {
    history_by_series(validation)
        .values()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
}

impl ClassicalExpertBank {
    fn effective_validation_window(&self, frame: &ForecastFrame) -> Result<usize> {
        let min_history = history_by_series(frame.rows())
            .values()
            .map(Vec::len)
            .min()
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "classical expert bank requires at least one series".to_string(),
                )
            })?;
        if let Some(window) = self.validation_window {
            if window >= min_history {
                return Err(CartoBoostError::InvalidInput(format!(
                    "classical expert validation_window must be smaller than every series history; minimum history is {min_history}, got {window}"
                )));
            }
            return Ok(window);
        }
        if min_history < 2 {
            return Err(CartoBoostError::InvalidInput(
                "classical expert bank requires at least two rows per series for a real holdout"
                    .to_string(),
            ));
        }
        Ok((min_history / 5).clamp(1, 8))
    }
}

fn expert_supports_history(expert: &ClassicalExpert, history_len: usize) -> bool {
    let required = match expert {
        ClassicalExpert::Naive
        | ClassicalExpert::LocalLevelKalman
        | ClassicalExpert::AutoARIMA { .. } => 1,
        ClassicalExpert::SeasonalNaive { season_length } => *season_length,
        ClassicalExpert::WindowAverage { window_size } => *window_size,
        ClassicalExpert::SeasonalWindowAverage {
            season_length,
            window_count,
        } => match season_length.checked_mul(*window_count) {
            Some(required) => required,
            None => return false,
        },
        ClassicalExpert::Theta { .. }
        | ClassicalExpert::ETS { .. }
        | ClassicalExpert::Kalman
        | ClassicalExpert::AutoLocalLevelKalman => 2,
        ClassicalExpert::OptimizedTheta {
            season_length: None,
        } => 3,
        ClassicalExpert::OptimizedTheta {
            season_length: Some(season_length),
        }
        | ClassicalExpert::SeasonalETS { season_length, .. }
        | ClassicalExpert::AutoETS { season_length } => match season_length.checked_mul(2) {
            Some(required)
                if matches!(
                    expert,
                    ClassicalExpert::OptimizedTheta { .. } | ClassicalExpert::AutoETS { .. }
                ) =>
            {
                match required.checked_add(1) {
                    Some(required) => required,
                    None => return false,
                }
            }
            Some(required) => required,
            None => return false,
        },
        ClassicalExpert::AutoKalman => 3,
    };
    history_len >= required
}

fn validate_horizon(horizon: usize) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "forecast horizon must be positive".to_string(),
        ));
    }
    Ok(())
}

fn not_fitted() -> CartoBoostError {
    CartoBoostError::InvalidInput("forecaster must be fitted before predict".to_string())
}

fn expert_rank(expert: &ClassicalExpert) -> usize {
    match expert {
        ClassicalExpert::Naive => 0,
        ClassicalExpert::SeasonalNaive { .. } => 1,
        ClassicalExpert::WindowAverage { .. } => 2,
        ClassicalExpert::SeasonalWindowAverage { .. } => 3,
        ClassicalExpert::Theta { .. } => 4,
        ClassicalExpert::AutoETS { .. } => 5,
        ClassicalExpert::OptimizedTheta { .. } => 6,
        ClassicalExpert::ETS { .. } => 7,
        ClassicalExpert::SeasonalETS { .. } => 8,
        ClassicalExpert::AutoARIMA { .. } => 9,
        ClassicalExpert::LocalLevelKalman => 10,
        ClassicalExpert::Kalman => 11,
        ClassicalExpert::AutoLocalLevelKalman => 12,
        ClassicalExpert::AutoKalman => 13,
    }
}

#[cfg(test)]
fn robust_classical_expert(scores: &[ClassicalExpertScore]) -> Result<&ClassicalExpertScore> {
    let best = scores.first().ok_or_else(|| {
        CartoBoostError::InvalidInput("classical expert scores must not be empty".to_string())
    })?;
    let tolerance = best.validation_loss;
    scores
        .iter()
        .filter(|score| score.validation_loss <= tolerance)
        .min_by_key(|score| (expert_rank(&score.expert), score.expert.name()))
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "classical expert robust selection found no candidate".to_string(),
            )
        })
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_selector_uses_strict_best_validation_mse() {
        let scores = vec![
            ClassicalExpertScore {
                expert: ClassicalExpert::AutoLocalLevelKalman,
                mse: 100.0,
                validation_loss: 100.0,
                validation_rows: 8,
            },
            ClassicalExpertScore {
                expert: ClassicalExpert::Naive,
                mse: 100.5,
                validation_loss: 100.5,
                validation_rows: 8,
            },
        ];

        let selected = robust_classical_expert(&scores).expect("selected expert");

        assert_eq!(selected.expert, ClassicalExpert::AutoLocalLevelKalman);
    }

    #[test]
    fn robust_selector_uses_validation_loss_not_mse() {
        let scores = vec![
            ClassicalExpertScore {
                expert: ClassicalExpert::AutoLocalLevelKalman,
                mse: 100.0,
                validation_loss: 1.0,
                validation_rows: 8,
            },
            ClassicalExpertScore {
                expert: ClassicalExpert::Naive,
                mse: 50.0,
                validation_loss: 2.0,
                validation_rows: 8,
            },
        ];

        let selected = robust_classical_expert(&scores).expect("selected expert");

        assert_eq!(selected.expert, ClassicalExpert::AutoLocalLevelKalman);
    }
}
