use cartoboost_core::forecasting::{
    ForecastFrame, ForecastFrequency, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use cartoboost_core::{CartoBoostError, Result as CoreResult};
use serde_json::{json, Value};
use std::collections::BTreeMap;

use super::dataloader::WindowDataset;
use super::nbeats::DeterministicMlp;
use super::scaler::StandardScaler;
use super::validate_window_config;
use crate::BackendSelection;

#[derive(Debug, Clone, PartialEq)]
pub struct NHiTSConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub pooling_size: usize,
    pub backend: BackendSelection,
}

impl Default for NHiTSConfig {
    fn default() -> Self {
        Self {
            input_size: 12,
            hidden_size: 16,
            epochs: 80,
            learning_rate: 0.01,
            pooling_size: 2,
            backend: BackendSelection::default(),
        }
    }
}

pub struct NHiTSForecaster {
    config: NHiTSConfig,
    model: DeterministicMlp,
    scaler: Option<StandardScaler>,
    frequency: Option<ForecastFrequency>,
    tails: BTreeMap<String, Vec<f64>>,
    last_rows: BTreeMap<String, ForecastRow>,
}

impl NHiTSForecaster {
    pub fn new(config: NHiTSConfig) -> crate::Result<Self> {
        validate_window_config(config.input_size, config.hidden_size, config.epochs)?;
        if !config.learning_rate.is_finite() || config.learning_rate <= 0.0 {
            return Err(crate::NeuralError::InvalidArgument(
                "learning_rate must be finite and positive".to_string(),
            ));
        }
        if config.pooling_size == 0 || config.pooling_size > config.input_size {
            return Err(crate::NeuralError::InvalidArgument(
                "pooling_size must be between 1 and input_size".to_string(),
            ));
        }
        let pooled_size = pooled_len(config.input_size, config.pooling_size);
        Ok(Self {
            model: DeterministicMlp::new(
                pooled_size,
                config.hidden_size,
                0.031,
                config.backend.clone(),
            ),
            config,
            scaler: None,
            frequency: None,
            tails: BTreeMap::new(),
            last_rows: BTreeMap::new(),
        })
    }

    pub fn config(&self) -> &NHiTSConfig {
        &self.config
    }
}

impl Forecaster for NHiTSForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> CoreResult<()> {
        let dataset = WindowDataset::from_frame(frame, self.config.input_size)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        let scaler = StandardScaler::fit(
            &frame
                .rows()
                .iter()
                .map(|row| row.target)
                .collect::<Vec<_>>(),
        )
        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        let examples = dataset
            .windows()
            .iter()
            .map(|window| {
                (
                    pool(
                        &scaler.transform_slice(&window.inputs),
                        self.config.pooling_size,
                    ),
                    scaler.transform(window.target),
                )
            })
            .collect::<Vec<_>>();
        self.model = DeterministicMlp::new(
            pooled_len(self.config.input_size, self.config.pooling_size),
            self.config.hidden_size,
            0.031,
            self.config.backend.clone(),
        );
        self.model
            .fit(&examples, self.config.epochs, self.config.learning_rate)
            .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
        self.scaler = Some(scaler);
        self.frequency = Some(dataset.frequency());
        self.tails = dataset.tails().clone();
        self.last_rows = frame
            .series_ids()
            .into_iter()
            .filter_map(|series_id| {
                frame
                    .rows_for_series(&series_id)
                    .last()
                    .map(|row| (series_id, (*row).clone()))
            })
            .collect();
        Ok(())
    }

    fn predict(&self, horizon: usize) -> CoreResult<ForecastResult> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        let scaler = self.scaler.ok_or_else(|| {
            CartoBoostError::InvalidInput("NHiTSForecaster must be fit before predict".to_string())
        })?;
        let frequency = self.frequency.ok_or_else(|| {
            CartoBoostError::InvalidInput("NHiTSForecaster must be fit before predict".to_string())
        })?;
        let series = self.tails.iter().collect::<Vec<_>>();
        let mut histories = series
            .iter()
            .map(|(_, tail)| (*tail).clone())
            .collect::<Vec<_>>();
        let mut predictions = vec![Vec::with_capacity(horizon); series.len()];
        for step in 1..=horizon {
            let inputs = histories
                .iter()
                .map(|history| pool(&scaler.transform_slice(history), self.config.pooling_size))
                .collect::<Vec<_>>();
            let scaled_predictions = self
                .model
                .predict_batch_with_backend(&inputs)
                .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
            for (series_index, ((series_id, _), scaled_prediction)) in
                series.iter().zip(scaled_predictions).enumerate()
            {
                let last_row = self.last_rows.get(*series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing fitted timestamp tail for series '{series_id}'"
                    ))
                })?;
                let mean = scaler.inverse_transform(scaled_prediction);
                predictions[series_index].push(ForecastPrediction {
                    series_id: (*series_id).clone(),
                    timestamp: frequency.advance(last_row.timestamp, step)?,
                    horizon: step,
                    model: self.model_name().to_string(),
                    mean,
                });
                histories[series_index].remove(0);
                histories[series_index].push(mean);
            }
        }
        ForecastResult::new(predictions.into_iter().flatten().collect())
    }

    fn model_name(&self) -> &'static str {
        "nhits"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "input_size": self.config.input_size,
            "hidden_size": self.config.hidden_size,
            "epochs": self.config.epochs,
            "learning_rate": self.config.learning_rate,
            "pooling_size": self.config.pooling_size,
            "backend": self.config.backend,
        })
    }
}

fn pooled_len(input_size: usize, pooling_size: usize) -> usize {
    input_size.div_ceil(pooling_size)
}

fn pool(values: &[f64], pooling_size: usize) -> Vec<f64> {
    values
        .chunks(pooling_size)
        .map(|chunk| chunk.iter().sum::<f64>() / chunk.len() as f64)
        .collect()
}
