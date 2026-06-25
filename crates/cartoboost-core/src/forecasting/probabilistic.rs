use crate::booster::{Booster, BoosterConfig};
use crate::data::Dataset;
use crate::loss::{LossConfig, QuantileLossConfig};
use crate::tree::Model;
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};

use super::quantiles::{
    default_quantile_levels, repair_non_crossing_quantiles, validate_quantile_grid,
    QuantileForecast,
};

pub trait ProbabilisticForecaster {
    fn predict_quantiles(&self, horizon: usize, quantiles: &[f64])
        -> Result<Vec<QuantileForecast>>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProbabilisticDirectForecaster {
    quantiles: Vec<f64>,
}

impl ProbabilisticDirectForecaster {
    pub fn new(quantiles: Vec<f64>) -> Result<Self> {
        validate_quantile_grid(&quantiles)?;
        Ok(Self { quantiles })
    }

    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    pub fn repair_horizon(&self, values: &[f64]) -> Result<QuantileForecast> {
        QuantileForecast::new(
            self.quantiles.clone(),
            repair_non_crossing_quantiles(values)?,
        )
    }

    pub fn repair_matrix(&self, horizon_values: &[Vec<f64>]) -> Result<Vec<QuantileForecast>> {
        horizon_values
            .iter()
            .map(|values| self.repair_horizon(values))
            .collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileRegressorSet {
    quantiles: Vec<f64>,
    models: Vec<Model>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantileRegressorSetConfig {
    pub quantiles: Vec<f64>,
    pub booster_config: BoosterConfig,
}

impl Default for QuantileRegressorSetConfig {
    fn default() -> Self {
        Self {
            quantiles: default_quantile_levels(),
            booster_config: BoosterConfig::default(),
        }
    }
}

impl QuantileRegressorSet {
    pub fn fit(
        x: &Dataset,
        y: &[f64],
        sample_weight: Option<&[f64]>,
        config: QuantileRegressorSetConfig,
    ) -> Result<Self> {
        validate_quantile_grid(&config.quantiles)?;
        if x.n_rows() != y.len() {
            return Err(CartoBoostError::InvalidInput(
                "X row count must match y length".to_string(),
            ));
        }
        if y.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "targets must be finite".to_string(),
            ));
        }
        let mut models = Vec::with_capacity(config.quantiles.len());
        for &quantile in &config.quantiles {
            let mut booster_config = config.booster_config.clone();
            booster_config.loss = LossConfig::Quantile(QuantileLossConfig { alpha: quantile });
            let mut model = Booster::new(booster_config).fit(x, y, sample_weight)?;
            model.target_name = Some(format!("quantile_{quantile:.3}"));
            models.push(model);
        }
        Ok(Self {
            quantiles: config.quantiles,
            models,
        })
    }

    pub fn new(quantiles: Vec<f64>, models: Vec<Model>) -> Result<Self> {
        validate_quantile_grid(&quantiles)?;
        if quantiles.len() != models.len() {
            return Err(CartoBoostError::InvalidInput(
                "quantiles and models must have the same length".to_string(),
            ));
        }
        Ok(Self { quantiles, models })
    }

    pub fn quantiles(&self) -> &[f64] {
        &self.quantiles
    }

    pub fn models(&self) -> &[Model] {
        &self.models
    }

    pub fn predict(&self, x: &Dataset) -> Result<Vec<QuantileForecast>> {
        if self.models.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "quantile regressor set must contain at least one model".to_string(),
            ));
        }
        let mut columns = Vec::with_capacity(self.models.len());
        for model in &self.models {
            columns.push(model.try_predict(x)?);
        }
        (0..x.n_rows())
            .map(|row| {
                let values = columns.iter().map(|column| column[row]).collect::<Vec<_>>();
                QuantileForecast::new(
                    self.quantiles.clone(),
                    repair_non_crossing_quantiles(&values)?,
                )
            })
            .collect()
    }

    pub fn predict_one(&self, row: &[f64]) -> Result<QuantileForecast> {
        if self.models.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "quantile regressor set must contain at least one model".to_string(),
            ));
        }
        let mut values = Vec::with_capacity(self.models.len());
        for model in &self.models {
            values.push(model.try_predict_one_dense(row)?);
        }
        QuantileForecast::new(
            self.quantiles.clone(),
            repair_non_crossing_quantiles(&values)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{LeafPredictorKind, SplitterKind};

    fn taxi_lane_fixture() -> (Dataset, Vec<f64>) {
        let mut rows = Vec::new();
        let mut y = Vec::new();
        for hour in 0..24 {
            for lane in 0..3 {
                let distance = 1.5 + lane as f64 * 2.0;
                let hour_f = hour as f64;
                rows.push(vec![distance, hour_f, lane as f64]);
                let rush = if (7..=9).contains(&hour) || (16..=18).contains(&hour) {
                    4.0
                } else {
                    0.0
                };
                let lane_offset = match lane {
                    0 => 0.0,
                    1 => 3.5,
                    _ => 7.0,
                };
                y.push(8.0 + 1.8 * distance + lane_offset + rush);
            }
        }
        (Dataset::from_rows(rows).unwrap(), y)
    }

    #[test]
    fn default_config_trains_requested_p10_to_p90_models() {
        let (x, y) = taxi_lane_fixture();
        let config = QuantileRegressorSetConfig {
            booster_config: BoosterConfig {
                n_estimators: 8,
                learning_rate: 0.25,
                max_depth: 2,
                min_samples_leaf: 2,
                splitters: vec![SplitterKind::Axis],
                leaf_predictor: LeafPredictorKind::Constant,
                ..BoosterConfig::default()
            },
            ..QuantileRegressorSetConfig::default()
        };

        let model = QuantileRegressorSet::fit(&x, &y, None, config).expect("fit");

        assert_eq!(model.quantiles(), &[0.10, 0.25, 0.50, 0.75, 0.90]);
        assert_eq!(model.models().len(), 5);
        for (quantile, fitted) in model.quantiles().iter().zip(model.models()) {
            assert_eq!(
                fitted.target_name.as_deref(),
                Some(format!("quantile_{quantile:.3}").as_str())
            );
        }
    }

    #[test]
    fn predictions_are_repaired_to_non_crossing_rows() {
        let (x, y) = taxi_lane_fixture();
        let config = QuantileRegressorSetConfig {
            quantiles: vec![0.1, 0.5, 0.9],
            booster_config: BoosterConfig {
                n_estimators: 6,
                learning_rate: 0.3,
                max_depth: 1,
                min_samples_leaf: 3,
                splitters: vec![SplitterKind::Axis],
                leaf_predictor: LeafPredictorKind::Constant,
                ..BoosterConfig::default()
            },
        };

        let model = QuantileRegressorSet::fit(&x, &y, None, config).expect("fit");
        let predictions = model.predict(&x).expect("predict");

        assert_eq!(predictions.len(), x.n_rows());
        for forecast in predictions {
            assert_eq!(forecast.quantiles, vec![0.1, 0.5, 0.9]);
            assert!(forecast.values.windows(2).all(|pair| pair[0] <= pair[1]));
        }
    }

    #[test]
    fn quantile_regressor_set_roundtrips_json() {
        let (x, y) = taxi_lane_fixture();
        let config = QuantileRegressorSetConfig {
            quantiles: vec![0.25, 0.5, 0.75],
            booster_config: BoosterConfig {
                n_estimators: 4,
                learning_rate: 0.2,
                max_depth: 1,
                min_samples_leaf: 2,
                splitters: vec![SplitterKind::Axis],
                leaf_predictor: LeafPredictorKind::Constant,
                ..BoosterConfig::default()
            },
        };
        let model = QuantileRegressorSet::fit(&x, &y, None, config).expect("fit");

        let encoded = serde_json::to_string(&model).expect("serialize");
        let decoded: QuantileRegressorSet = serde_json::from_str(&encoded).expect("deserialize");

        assert_eq!(decoded.quantiles(), model.quantiles());
        assert_eq!(
            decoded.predict(&x).expect("decoded predict"),
            model.predict(&x).expect("original predict")
        );
    }

    #[test]
    fn rejects_unsorted_quantile_grid() {
        let (x, y) = taxi_lane_fixture();
        let err = QuantileRegressorSet::fit(
            &x,
            &y,
            None,
            QuantileRegressorSetConfig {
                quantiles: vec![0.5, 0.1],
                ..QuantileRegressorSetConfig::default()
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("strictly increasing"));
    }
}
