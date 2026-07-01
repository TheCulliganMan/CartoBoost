use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::local::AutoARIMAForecaster;
use crate::forecasting::mstl::MSTLDecomposition;
use crate::forecasting::stl::{seasonal_pattern, STLDecomposition};
use crate::forecasting::{
    ForecastFrame, ForecastPrediction, ForecastResult, ForecastRow, Forecaster,
};
use crate::{CartoBoostError, Result};
use serde_json::{json, Value};
use std::collections::BTreeMap;

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
    trend: Vec<f64>,
    seasonal_pattern: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FittedMSTLSeries {
    trend: Vec<f64>,
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
        frame.require_regular_for_model(self.model_name())?;
        let mut remainder_rows = Vec::with_capacity(frame.rows().len());
        let mut fitted_series = BTreeMap::new();
        for (series_id, rows) in history_by_series(frame.rows()) {
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            let decomposition = self.decomposition.decompose(&values)?;
            let pattern =
                seasonal_pattern(&decomposition.seasonal, self.decomposition.season_length());
            for (row, remainder) in rows.iter().zip(&decomposition.remainder) {
                remainder_rows.push(ForecastRow::new(
                    row.series_id.clone(),
                    row.timestamp,
                    *remainder,
                ));
            }
            fitted_series.insert(
                series_id,
                FittedSTLSeries {
                    trend: decomposition.trend,
                    seasonal_pattern: pattern,
                },
            );
        }
        let remainder_frame = ForecastFrame::with_metadata(
            remainder_rows,
            frame.frequency(),
            frame.metadata().clone(),
        )?;
        self.remainder_forecaster.fit(&remainder_frame)?;
        self.fitted = Some(FittedSTLHybridState {
            series: fitted_series,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let remainder = self.remainder_forecaster.predict(horizon)?;
        let predictions = remainder
            .predictions()
            .iter()
            .map(|prediction| {
                let series = fitted.series.get(&prediction.series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing STL decomposition for series {}",
                        prediction.series_id
                    ))
                })?;
                Ok(ForecastPrediction {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                    model: self.model_name().to_string(),
                    mean: prediction.mean
                        + forecast_trend(&series.trend, prediction.horizon)
                        + forecast_pattern(
                            &series.seasonal_pattern,
                            series.trend.len(),
                            prediction.horizon,
                        ),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "stl_cartoboost"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "decomposition": self.decomposition.metadata(),
            "remainder_model": self.remainder_forecaster.metadata(),
        })
    }
}

impl Forecaster for MSTLCartoBoostForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let mut remainder_rows = Vec::with_capacity(frame.rows().len());
        let mut fitted_series = BTreeMap::new();
        for (series_id, rows) in history_by_series(frame.rows()) {
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            let decomposition = self.decomposition.decompose(&values)?;
            for (row, remainder) in rows.iter().zip(&decomposition.remainder) {
                remainder_rows.push(ForecastRow::new(
                    row.series_id.clone(),
                    row.timestamp,
                    *remainder,
                ));
            }
            let seasonal_patterns = decomposition
                .seasonal_components
                .iter()
                .map(|component| {
                    (
                        component.season_length,
                        seasonal_pattern(&component.values, component.season_length),
                    )
                })
                .collect::<Vec<_>>();
            fitted_series.insert(
                series_id,
                FittedMSTLSeries {
                    trend: decomposition.trend,
                    seasonal_patterns,
                },
            );
        }
        let remainder_frame = ForecastFrame::with_metadata(
            remainder_rows,
            frame.frequency(),
            frame.metadata().clone(),
        )?;
        self.remainder_forecaster.fit(&remainder_frame)?;
        self.fitted = Some(FittedMSTLHybridState {
            series: fitted_series,
        });
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let remainder = self.remainder_forecaster.predict(horizon)?;
        let predictions = remainder
            .predictions()
            .iter()
            .map(|prediction| {
                let series = fitted.series.get(&prediction.series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing MSTL decomposition for series {}",
                        prediction.series_id
                    ))
                })?;
                let seasonal = series
                    .seasonal_patterns
                    .iter()
                    .map(|(_, pattern)| {
                        forecast_pattern(pattern, series.trend.len(), prediction.horizon)
                    })
                    .sum::<f64>();
                Ok(ForecastPrediction {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                    model: self.model_name().to_string(),
                    mean: prediction.mean
                        + forecast_trend(&series.trend, prediction.horizon)
                        + seasonal,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "mstl_cartoboost"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "decomposition": self.decomposition.metadata(),
            "remainder_model": self.remainder_forecaster.metadata(),
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

fn forecast_trend(trend: &[f64], horizon: usize) -> f64 {
    if trend.is_empty() {
        return 0.0;
    }
    let last = *trend.last().expect("checked non-empty");
    let slope_window = trend.len().saturating_sub(4);
    let tail = &trend[slope_window..];
    let slope = if tail.len() < 2 {
        0.0
    } else {
        tail.windows(2).map(|pair| pair[1] - pair[0]).sum::<f64>() / (tail.len() - 1) as f64
    };
    last + slope * horizon as f64
}

fn forecast_pattern(pattern: &[f64], history_len: usize, horizon: usize) -> f64 {
    if pattern.is_empty() {
        return 0.0;
    }
    pattern[(history_len + horizon - 1) % pattern.len()]
}
