use crate::forecasting::classical_bank::{
    ClassicalExpert, ClassicalExpertBank, ClassicalExpertScore, ClassicalExpertValidationObjective,
};
use crate::forecasting::{ForecastFrame, ForecastResult, Forecaster};
use crate::Result;
use serde_json::{json, Value};

pub struct AutoStatsBank {
    bank: ClassicalExpertBank,
    season_length: usize,
}

impl AutoStatsBank {
    pub fn new(season_length: usize) -> Result<Self> {
        Self::with_validation_window(season_length, None)
    }

    pub fn with_validation_window(
        season_length: usize,
        validation_window: Option<usize>,
    ) -> Result<Self> {
        Self::with_validation_objective(
            season_length,
            validation_window,
            ClassicalExpertValidationObjective::MeanSquaredError,
        )
    }

    pub fn with_validation_objective(
        season_length: usize,
        validation_window: Option<usize>,
        validation_objective: ClassicalExpertValidationObjective,
    ) -> Result<Self> {
        let mut experts = vec![
            ClassicalExpert::Naive,
            ClassicalExpert::WindowAverage { window_size: 3 },
            ClassicalExpert::WindowAverage { window_size: 7 },
            ClassicalExpert::Theta {
                theta: 2.0,
                alpha: 0.2,
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
        Ok(Self {
            bank: ClassicalExpertBank::with_validation_options(
                experts,
                validation_window,
                validation_objective,
            )?,
            season_length,
        })
    }

    pub fn selected_expert(&self) -> Option<&ClassicalExpert> {
        self.bank.selected_expert()
    }

    pub fn validation_scores(&self) -> &[ClassicalExpertScore] {
        self.bank.validation_scores()
    }
}

impl Forecaster for AutoStatsBank {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.bank.fit(frame)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let result = self.bank.predict(horizon)?;
        let predictions = result
            .predictions()
            .iter()
            .map(|prediction| crate::forecasting::ForecastPrediction {
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
        "autostats_bank"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "season_length": self.season_length,
            "selected_expert": self.selected_expert().map(ClassicalExpert::metadata),
            "classical_bank": self.bank.metadata(),
        })
    }
}

impl Default for AutoStatsBank {
    fn default() -> Self {
        Self::new(24).expect("default autostats bank is valid")
    }
}
