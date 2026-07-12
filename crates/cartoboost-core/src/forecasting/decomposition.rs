use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::local::AutoARIMAForecaster;
use crate::forecasting::mstl::MSTLDecomposition;
use crate::forecasting::stl::STLDecomposition;
use crate::forecasting::{
    ForecastFrame, ForecastIntervalPrediction, ForecastPrediction, ForecastPredictionDetail,
    ForecastResult, ForecastRow, Forecaster,
};
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub struct STLCartoBoostForecaster {
    decomposition: STLDecomposition,
    remainder_forecaster: Box<dyn Forecaster>,
    fitted: Option<FittedSTLHybridState>,
}

pub struct MSTLCartoBoostForecaster {
    decomposition: MSTLDecomposition,
    remainder_forecaster: Box<dyn Forecaster>,
    fitted: Option<FittedMSTLHybridState>,
}

#[derive(Debug, Clone)]
struct FittedSTLHybridState {
    series: BTreeMap<String, FittedSTLSeries>,
}

#[derive(Debug, Clone)]
struct FittedMSTLHybridState {
    series: BTreeMap<String, FittedMSTLSeries>,
}

#[derive(Debug, Clone)]
struct FittedSTLSeries {
    history_len: usize,
    seasonal_pattern: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FittedMSTLSeries {
    history_len: usize,
    seasonal_patterns: Vec<(usize, Vec<f64>)>,
}

impl STLCartoBoostForecaster {
    pub fn new(season_length: usize) -> Result<Self> {
        Self::with_remainder_forecaster(
            STLDecomposition::new(season_length)?,
            Box::new(AutoARIMAForecaster::new(2, 1)?),
        )
    }

    pub fn with_remainder_forecaster(
        decomposition: STLDecomposition,
        remainder_forecaster: Box<dyn Forecaster>,
    ) -> Result<Self> {
        Ok(Self {
            decomposition,
            remainder_forecaster,
            fitted: None,
        })
    }

    pub fn decomposition(&self) -> &STLDecomposition {
        &self.decomposition
    }
}

impl MSTLCartoBoostForecaster {
    pub fn new(season_lengths: Vec<usize>) -> Result<Self> {
        Self::with_remainder_forecaster(
            MSTLDecomposition::new(season_lengths)?,
            Box::new(AutoARIMAForecaster::new(2, 1)?),
        )
    }

    pub fn with_remainder_forecaster(
        decomposition: MSTLDecomposition,
        remainder_forecaster: Box<dyn Forecaster>,
    ) -> Result<Self> {
        Ok(Self {
            decomposition,
            remainder_forecaster,
            fitted: None,
        })
    }

    pub fn decomposition(&self) -> &MSTLDecomposition {
        &self.decomposition
    }
}

impl Forecaster for STLCartoBoostForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = None;
        frame.require_regular_for_model(self.model_name())?;
        let mut adjusted_rows = Vec::with_capacity(frame.rows().len());
        let mut fitted_series = BTreeMap::new();
        for (series_id, rows) in history_by_series(frame.rows()) {
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            let decomposition = self.decomposition.decompose(&values)?;
            let pattern =
                final_phase_pattern(&decomposition.seasonal, self.decomposition.season_length())?;
            // The downstream model forecasts the seasonally adjusted series
            // (trend + remainder). The repeated STL seasonal component is then
            // added back at prediction time.
            for (row, seasonal) in rows.iter().zip(&decomposition.seasonal) {
                adjusted_rows.push(ForecastRow::new(
                    row.series_id.clone(),
                    row.timestamp,
                    row.target - seasonal,
                ));
            }
            fitted_series.insert(
                series_id,
                FittedSTLSeries {
                    history_len: values.len(),
                    seasonal_pattern: pattern,
                },
            );
        }
        let adjusted_frame = ForecastFrame::with_metadata(
            adjusted_rows,
            frame.frequency(),
            frame.metadata().clone(),
        )?;
        self.remainder_forecaster.fit(&adjusted_frame)?;
        self.fitted = Some(FittedSTLHybridState {
            series: fitted_series,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let adjusted = self.remainder_forecaster.predict(horizon)?;
        validate_adjusted_forecast(&adjusted, fitted.series.keys(), horizon, self.model_name())?;
        let predictions = adjusted
            .predictions()
            .iter()
            .map(|prediction| {
                let seasonal = stl_adjustment(fitted, &prediction.series_id, prediction.horizon)?;
                Ok(ForecastPrediction {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                    model: self.model_name().to_string(),
                    mean: prediction.mean + seasonal,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let intervals = adjusted
            .intervals()
            .iter()
            .map(|interval| {
                let seasonal = stl_adjustment(fitted, &interval.series_id, interval.horizon)?;
                Ok(ForecastIntervalPrediction {
                    series_id: interval.series_id.clone(),
                    timestamp: interval.timestamp,
                    horizon: interval.horizon,
                    model: self.model_name().to_string(),
                    level: interval.level,
                    lower: interval.lower + seasonal,
                    upper: interval.upper + seasonal,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let details = adjusted
            .details()
            .iter()
            .map(|detail| {
                let seasonal = stl_adjustment(fitted, &detail.series_id, detail.horizon)?;
                Ok(shift_detail(detail, self.model_name(), seasonal))
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new_with_intervals_and_details(predictions, intervals, details)
    }

    fn model_name(&self) -> &'static str {
        "stl_cartoboost"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "decomposition": self.decomposition.metadata(),
            "seasonally_adjusted_model": self.remainder_forecaster.metadata(),
            "adjusted_target": "observed_minus_seasonal",
            "seasonal_forecast": "repeat_final_cycle",
        })
    }
}

impl Forecaster for MSTLCartoBoostForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        self.fitted = None;
        frame.require_regular_for_model(self.model_name())?;
        let mut adjusted_rows = Vec::with_capacity(frame.rows().len());
        let mut fitted_series = BTreeMap::new();
        for (series_id, rows) in history_by_series(frame.rows()) {
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            let decomposition = self.decomposition.decompose(&values)?;
            let total_seasonal = decomposition.total_seasonal();
            for (row, seasonal) in rows.iter().zip(&total_seasonal) {
                adjusted_rows.push(ForecastRow::new(
                    row.series_id.clone(),
                    row.timestamp,
                    row.target - seasonal,
                ));
            }
            let seasonal_patterns = decomposition
                .seasonal_components
                .iter()
                .map(|component| {
                    Ok((
                        component.season_length,
                        final_phase_pattern(&component.values, component.season_length)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            fitted_series.insert(
                series_id,
                FittedMSTLSeries {
                    history_len: values.len(),
                    seasonal_patterns,
                },
            );
        }
        let adjusted_frame = ForecastFrame::with_metadata(
            adjusted_rows,
            frame.frequency(),
            frame.metadata().clone(),
        )?;
        self.remainder_forecaster.fit(&adjusted_frame)?;
        self.fitted = Some(FittedMSTLHybridState {
            series: fitted_series,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let adjusted = self.remainder_forecaster.predict(horizon)?;
        validate_adjusted_forecast(&adjusted, fitted.series.keys(), horizon, self.model_name())?;
        let predictions = adjusted
            .predictions()
            .iter()
            .map(|prediction| {
                let seasonal = mstl_adjustment(fitted, &prediction.series_id, prediction.horizon)?;
                Ok(ForecastPrediction {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                    model: self.model_name().to_string(),
                    mean: prediction.mean + seasonal,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let intervals = adjusted
            .intervals()
            .iter()
            .map(|interval| {
                let seasonal = mstl_adjustment(fitted, &interval.series_id, interval.horizon)?;
                Ok(ForecastIntervalPrediction {
                    series_id: interval.series_id.clone(),
                    timestamp: interval.timestamp,
                    horizon: interval.horizon,
                    model: self.model_name().to_string(),
                    level: interval.level,
                    lower: interval.lower + seasonal,
                    upper: interval.upper + seasonal,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let details = adjusted
            .details()
            .iter()
            .map(|detail| {
                let seasonal = mstl_adjustment(fitted, &detail.series_id, detail.horizon)?;
                Ok(shift_detail(detail, self.model_name(), seasonal))
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new_with_intervals_and_details(predictions, intervals, details)
    }

    fn model_name(&self) -> &'static str {
        "mstl_cartoboost"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "decomposition": self.decomposition.metadata(),
            "seasonally_adjusted_model": self.remainder_forecaster.metadata(),
            "adjusted_target": "observed_minus_total_seasonal",
            "seasonal_forecast": "repeat_final_cycle",
        })
    }
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

fn final_phase_pattern(values: &[f64], season_length: usize) -> Result<Vec<f64>> {
    if values.len() < season_length {
        return Err(CartoBoostError::InvalidInput(format!(
            "seasonal component has {} values but requires a complete final cycle of {season_length}",
            values.len()
        )));
    }
    let mut pattern = vec![0.0; season_length];
    for idx in values.len() - season_length..values.len() {
        pattern[idx % season_length] = values[idx];
    }
    Ok(pattern)
}

fn forecast_pattern(pattern: &[f64], history_len: usize, horizon: usize) -> Result<f64> {
    if pattern.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "seasonal forecast pattern must not be empty".to_string(),
        ));
    }
    let position = history_len
        .checked_add(horizon.saturating_sub(1))
        .ok_or_else(|| {
            CartoBoostError::InvalidInput("seasonal forecast horizon is too large".to_string())
        })?;
    Ok(pattern[position % pattern.len()])
}

fn stl_adjustment(fitted: &FittedSTLHybridState, series_id: &str, horizon: usize) -> Result<f64> {
    let series = fitted.series.get(series_id).ok_or_else(|| {
        CartoBoostError::InvalidInput(format!("missing STL decomposition for series {series_id}"))
    })?;
    forecast_pattern(&series.seasonal_pattern, series.history_len, horizon)
}

fn mstl_adjustment(fitted: &FittedMSTLHybridState, series_id: &str, horizon: usize) -> Result<f64> {
    let series = fitted.series.get(series_id).ok_or_else(|| {
        CartoBoostError::InvalidInput(format!("missing MSTL decomposition for series {series_id}"))
    })?;
    series
        .seasonal_patterns
        .iter()
        .try_fold(0.0, |total, (_, pattern)| {
            Ok(total + forecast_pattern(pattern, series.history_len, horizon)?)
        })
}

fn validate_adjusted_forecast<'a>(
    forecast: &ForecastResult,
    series_ids: impl Iterator<Item = &'a String>,
    horizon: usize,
    model_name: &str,
) -> Result<()> {
    let expected_series = series_ids.cloned().collect::<BTreeSet<_>>();
    let mut actual = BTreeSet::new();
    for prediction in forecast.predictions() {
        if !expected_series.contains(&prediction.series_id) {
            return Err(CartoBoostError::InvalidInput(format!(
                "{model_name} adjusted forecaster returned unknown series {}",
                prediction.series_id
            )));
        }
        if prediction.horizon > horizon {
            return Err(CartoBoostError::InvalidInput(format!(
                "{model_name} adjusted forecaster returned horizon {} beyond requested horizon {horizon}",
                prediction.horizon
            )));
        }
        if !actual.insert((prediction.series_id.clone(), prediction.horizon)) {
            return Err(CartoBoostError::InvalidInput(format!(
                "{model_name} adjusted forecaster returned multiple predictions for series {} horizon {}",
                prediction.series_id, prediction.horizon
            )));
        }
    }
    for series_id in expected_series {
        for step in 1..=horizon {
            if !actual.contains(&(series_id.clone(), step)) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "{model_name} adjusted forecaster omitted series {series_id} horizon {step}"
                )));
            }
        }
    }
    Ok(())
}

fn shift_detail(
    detail: &ForecastPredictionDetail,
    model_name: &str,
    seasonal: f64,
) -> ForecastPredictionDetail {
    ForecastPredictionDetail {
        series_id: detail.series_id.clone(),
        timestamp: detail.timestamp,
        horizon: detail.horizon,
        model: model_name.to_string(),
        base_mean: detail.base_mean.map(|value| value + seasonal),
        spatial_correction: detail.spatial_correction,
        kriging_variance: detail.kriging_variance,
        selected_neighbors: detail.selected_neighbors.clone(),
        component_decomposition: detail.component_decomposition.clone(),
        metadata: detail.metadata.clone(),
    }
}
