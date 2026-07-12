use crate::forecasting::lag_features::history_by_series;
use crate::forecasting::{
    ForecastFrame, ForecastIntervalPrediction, ForecastPrediction, ForecastPredictionDetail,
    ForecastResult, ForecastRow, Forecaster,
};
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalStandardScaler {
    min_scale: f64,
    stats: BTreeMap<String, LocalScaleStats>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LocalScaleStats {
    pub mean: f64,
    pub scale: f64,
}

impl LocalStandardScaler {
    pub fn new(min_scale: f64) -> Result<Self> {
        if !min_scale.is_finite() || min_scale <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "local standard scaler min_scale must be finite and positive".to_string(),
            ));
        }
        Ok(Self {
            min_scale,
            stats: BTreeMap::new(),
        })
    }

    pub fn fit_transform(&mut self, frame: &ForecastFrame) -> Result<ForecastFrame> {
        let mut rows = Vec::with_capacity(frame.rows().len());
        let mut stats = BTreeMap::new();
        for (series_id, history) in history_by_series(frame.rows()) {
            let stat = fit_stats(&history, self.min_scale)?;
            stats.insert(series_id.clone(), stat);
            rows.extend(history.into_iter().map(|row| {
                ForecastRow::with_covariates(
                    row.series_id,
                    row.timestamp,
                    (row.target - stat.mean) / stat.scale,
                    row.covariates,
                )
            }));
        }
        self.stats = stats;
        ForecastFrame::with_metadata(rows, frame.frequency(), frame.metadata().clone())
    }

    pub fn inverse_result(
        &self,
        result: &ForecastResult,
        model_name: &str,
    ) -> Result<ForecastResult> {
        let predictions = result
            .predictions()
            .iter()
            .map(|prediction| {
                let stat = self.stats.get(&prediction.series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing local scale stats for series {}",
                        prediction.series_id
                    ))
                })?;
                Ok(ForecastPrediction {
                    series_id: prediction.series_id.clone(),
                    timestamp: prediction.timestamp,
                    horizon: prediction.horizon,
                    model: model_name.to_string(),
                    mean: prediction.mean * stat.scale + stat.mean,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let intervals = result
            .intervals()
            .iter()
            .map(|interval| {
                let stat = self.stats.get(&interval.series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing local scale stats for series {}",
                        interval.series_id
                    ))
                })?;
                Ok(ForecastIntervalPrediction {
                    series_id: interval.series_id.clone(),
                    timestamp: interval.timestamp,
                    horizon: interval.horizon,
                    model: model_name.to_string(),
                    level: interval.level,
                    lower: interval.lower * stat.scale + stat.mean,
                    upper: interval.upper * stat.scale + stat.mean,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let details = result
            .details()
            .iter()
            .map(|detail| {
                let stat = self.stats.get(&detail.series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing local scale stats for series {}",
                        detail.series_id
                    ))
                })?;
                Ok(ForecastPredictionDetail {
                    series_id: detail.series_id.clone(),
                    timestamp: detail.timestamp,
                    horizon: detail.horizon,
                    model: model_name.to_string(),
                    base_mean: detail.base_mean.map(|value| value * stat.scale + stat.mean),
                    spatial_correction: detail.spatial_correction.map(|value| value * stat.scale),
                    kriging_variance: detail
                        .kriging_variance
                        .map(|value| value * stat.scale * stat.scale),
                    selected_neighbors: detail.selected_neighbors.clone(),
                    component_decomposition: wrapped_component_decomposition(
                        detail.component_decomposition.as_ref(),
                        "local_standard_scaler",
                    ),
                    metadata: Some(json!({
                        "target_transform": "local_standard_scaler",
                        "inner_model": detail.model,
                        "inner_metadata": detail.metadata,
                    })),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new_with_intervals_and_details(predictions, intervals, details)
    }

    pub fn stats(&self) -> &BTreeMap<String, LocalScaleStats> {
        &self.stats
    }

    pub fn metadata(&self) -> Value {
        json!({
            "transform": "local_standard_scaler",
            "min_scale": self.min_scale,
            "series_count": self.stats.len(),
            "stats": self.stats,
        })
    }
}

fn fit_stats(history: &[ForecastRow], min_scale: f64) -> Result<LocalScaleStats> {
    if history.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "local standard scaler requires at least one row per series".to_string(),
        ));
    }
    let mean = history.iter().map(|row| row.target).sum::<f64>() / history.len() as f64;
    let variance = history
        .iter()
        .map(|row| {
            let centered = row.target - mean;
            centered * centered
        })
        .sum::<f64>()
        / history.len() as f64;
    let scale = variance.sqrt().max(min_scale);
    if !mean.is_finite() || !scale.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "local standard scaler produced non-finite stats".to_string(),
        ));
    }
    Ok(LocalScaleStats { mean, scale })
}

pub struct LocalStandardScaledForecaster {
    scaler: LocalStandardScaler,
    inner: Box<dyn Forecaster>,
    model_name: &'static str,
}

impl LocalStandardScaledForecaster {
    pub fn new(
        inner: Box<dyn Forecaster>,
        min_scale: f64,
        model_name: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            scaler: LocalStandardScaler::new(min_scale)?,
            inner,
            model_name,
        })
    }
}

impl Forecaster for LocalStandardScaledForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_observed_targets_for_model(self.model_name())?;
        let transformed = self.scaler.fit_transform(frame)?;
        self.inner.fit(&transformed)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let transformed = self.inner.predict(horizon)?;
        self.scaler.inverse_result(&transformed, self.model_name)
    }

    fn model_name(&self) -> &'static str {
        self.model_name
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "target_transform": self.scaler.metadata(),
            "inner": self.inner.metadata(),
        })
    }
}

pub struct Log1pForecaster {
    inner: Box<dyn Forecaster>,
    model_name: &'static str,
}

impl Log1pForecaster {
    pub fn new(inner: Box<dyn Forecaster>, model_name: &'static str) -> Self {
        Self { inner, model_name }
    }
}

impl Forecaster for Log1pForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_observed_targets_for_model(self.model_name())?;
        let rows = frame
            .rows()
            .iter()
            .map(|row| {
                if row.target < 0.0 {
                    return Err(CartoBoostError::InvalidInput(
                        "log1p target transform requires nonnegative targets".to_string(),
                    ));
                }
                Ok(ForecastRow::with_covariates(
                    row.series_id.clone(),
                    row.timestamp,
                    row.target.ln_1p(),
                    row.covariates.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let transformed =
            ForecastFrame::with_metadata(rows, frame.frequency(), frame.metadata().clone())?;
        self.inner.fit(&transformed)
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let transformed = self.inner.predict(horizon)?;
        let predictions = transformed
            .predictions()
            .iter()
            .map(|prediction| ForecastPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: self.model_name.to_string(),
                mean: prediction.mean.exp_m1().max(0.0),
            })
            .collect::<Vec<_>>();
        let intervals = transformed
            .intervals()
            .iter()
            .map(|interval| ForecastIntervalPrediction {
                series_id: interval.series_id.clone(),
                timestamp: interval.timestamp,
                horizon: interval.horizon,
                model: self.model_name.to_string(),
                level: interval.level,
                lower: inverse_log1p_nonnegative(interval.lower),
                upper: inverse_log1p_nonnegative(interval.upper),
            })
            .collect::<Vec<_>>();
        let details = transformed
            .details()
            .iter()
            .map(|detail| {
                let base_mean = detail.base_mean.map(inverse_log1p_nonnegative);
                let spatial_correction = match (detail.base_mean, detail.spatial_correction) {
                    (_, None) => None,
                    (Some(base), Some(correction)) => Some(
                        inverse_log1p_nonnegative(base + correction)
                            - inverse_log1p_nonnegative(base),
                    ),
                    (None, Some(_)) => {
                        return Err(CartoBoostError::InvalidInput(format!(
                            "cannot invert log1p spatial correction without base_mean for series {} at {}",
                            detail.series_id, detail.timestamp
                        )))
                    }
                };
                Ok(ForecastPredictionDetail {
                    series_id: detail.series_id.clone(),
                    timestamp: detail.timestamp,
                    horizon: detail.horizon,
                    model: self.model_name.to_string(),
                    base_mean,
                    spatial_correction,
                    // A variance does not have an exact nonlinear inverse without
                    // distributional assumptions. Keep the inner-scale value in
                    // metadata instead of presenting a delta-method approximation
                    // as an exact original-scale kriging variance.
                    kriging_variance: None,
                    selected_neighbors: detail.selected_neighbors.clone(),
                    component_decomposition: wrapped_component_decomposition(
                        detail.component_decomposition.as_ref(),
                        "log1p",
                    ),
                    metadata: Some(json!({
                        "target_transform": "log1p",
                        "inner_model": detail.model,
                        "inner_kriging_variance": detail.kriging_variance,
                        "inner_metadata": detail.metadata,
                    })),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new_with_intervals_and_details(predictions, intervals, details)
    }

    fn model_name(&self) -> &'static str {
        self.model_name
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "target_transform": {
                "transform": "log1p",
                "inverse": "expm1_clamped_nonnegative",
            },
            "inner": self.inner.metadata(),
        })
    }
}

fn inverse_log1p_nonnegative(value: f64) -> f64 {
    value.exp_m1().max(0.0)
}

fn wrapped_component_decomposition(component: Option<&Value>, transform: &str) -> Option<Value> {
    component.map(|value| {
        json!({
            "scale": "inner_transformed_target",
            "target_transform": transform,
            "value": value,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{ForecastFrameMetadata, ForecastFrequency};
    use chrono::NaiveDate;
    use std::sync::{Arc, Mutex};

    struct CapturingForecaster {
        fitted_frame: Arc<Mutex<Option<ForecastFrame>>>,
        result: ForecastResult,
    }

    impl Forecaster for CapturingForecaster {
        fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
            *self.fitted_frame.lock().expect("capture lock") = Some(frame.clone());
            Ok(())
        }

        fn predict(&self, _horizon: usize) -> Result<ForecastResult> {
            Ok(self.result.clone())
        }

        fn model_name(&self) -> &'static str {
            "capturing"
        }

        fn metadata(&self) -> Value {
            json!({"model": self.model_name()})
        }
    }

    #[test]
    fn local_standard_transform_preserves_covariates_and_inverts_rich_results() {
        let captured = Arc::new(Mutex::new(None));
        let inner = CapturingForecaster {
            fitted_frame: captured.clone(),
            result: rich_result("capturing", 1.0, 0.0, 2.0, Some(0.5), Some(0.5), Some(0.25)),
        };
        let mut wrapper =
            LocalStandardScaledForecaster::new(Box::new(inner), 1.0e-9, "scaled").expect("wrapper");
        let frame = frame_with_covariates(0.0, 4.0);

        wrapper.fit(&frame).expect("fit");
        let transformed = captured
            .lock()
            .expect("capture lock")
            .clone()
            .expect("captured frame");
        assert_eq!(transformed.rows()[0].covariates["trip_distance"], 1.5);
        assert_eq!(transformed.rows()[1].covariates["trip_distance"], 2.5);
        assert_eq!(transformed.metadata(), frame.metadata());

        let forecast = wrapper.predict(1).expect("predict");
        assert_eq!(forecast.predictions()[0].mean, 4.0);
        assert_eq!(forecast.intervals()[0].lower, 2.0);
        assert_eq!(forecast.intervals()[0].upper, 6.0);
        let detail = &forecast.details()[0];
        assert_eq!(detail.base_mean, Some(3.0));
        assert_eq!(detail.spatial_correction, Some(1.0));
        assert_eq!(detail.kriging_variance, Some(1.0));
        assert_eq!(detail.selected_neighbors, vec!["PU2->DO3"]);
        assert_eq!(
            detail.component_decomposition.as_ref().unwrap()["scale"],
            "inner_transformed_target"
        );
    }

    #[test]
    fn log1p_transform_preserves_covariates_and_inverts_intervals_and_details() {
        let captured = Arc::new(Mutex::new(None));
        let base = 2.0_f64.ln_1p();
        let corrected = 4.0_f64.ln_1p();
        let inner = CapturingForecaster {
            fitted_frame: captured.clone(),
            result: rich_result(
                "capturing",
                3.0_f64.ln_1p(),
                1.0_f64.ln_1p(),
                5.0_f64.ln_1p(),
                Some(base),
                Some(corrected - base),
                Some(0.4),
            ),
        };
        let mut wrapper = Log1pForecaster::new(Box::new(inner), "log1p");
        let frame = frame_with_covariates(0.0, 4.0);

        wrapper.fit(&frame).expect("fit");
        let transformed = captured
            .lock()
            .expect("capture lock")
            .clone()
            .expect("captured frame");
        assert_eq!(transformed.rows()[0].covariates["trip_distance"], 1.5);
        assert_eq!(transformed.rows()[1].covariates["trip_distance"], 2.5);

        let forecast = wrapper.predict(1).expect("predict");
        assert!((forecast.predictions()[0].mean - 3.0).abs() < 1.0e-12);
        assert!((forecast.intervals()[0].lower - 1.0).abs() < 1.0e-12);
        assert!((forecast.intervals()[0].upper - 5.0).abs() < 1.0e-12);
        let detail = &forecast.details()[0];
        assert!((detail.base_mean.unwrap() - 2.0).abs() < 1.0e-12);
        assert!((detail.spatial_correction.unwrap() - 2.0).abs() < 1.0e-12);
        assert_eq!(detail.kriging_variance, None);
        assert_eq!(
            detail.metadata.as_ref().unwrap()["inner_kriging_variance"],
            0.4
        );
    }

    #[test]
    fn log1p_transform_rejects_uninvertible_spatial_detail() {
        let inner = CapturingForecaster {
            fitted_frame: Arc::new(Mutex::new(None)),
            result: rich_result("capturing", 1.0, 0.5, 1.5, None, Some(0.25), None),
        };
        let wrapper = Log1pForecaster::new(Box::new(inner), "log1p");

        let error = wrapper
            .predict(1)
            .expect_err("correction without its base cannot be inverted exactly");
        assert!(error
            .to_string()
            .contains("cannot invert log1p spatial correction without base_mean"));
    }

    fn frame_with_covariates(first: f64, second: f64) -> ForecastFrame {
        let metadata = ForecastFrameMetadata {
            historical_covariates: vec!["trip_distance".to_string()],
            ..ForecastFrameMetadata::default()
        };
        ForecastFrame::with_metadata(
            vec![
                ForecastRow::with_covariates(
                    "PU1->DO2",
                    ts(1),
                    first,
                    BTreeMap::from([("trip_distance".to_string(), 1.5)]),
                ),
                ForecastRow::with_covariates(
                    "PU1->DO2",
                    ts(2),
                    second,
                    BTreeMap::from([("trip_distance".to_string(), 2.5)]),
                ),
            ],
            ForecastFrequency::Daily,
            metadata,
        )
        .expect("frame")
    }

    fn rich_result(
        model: &str,
        mean: f64,
        lower: f64,
        upper: f64,
        base_mean: Option<f64>,
        spatial_correction: Option<f64>,
        kriging_variance: Option<f64>,
    ) -> ForecastResult {
        ForecastResult::new_with_intervals_and_details(
            vec![ForecastPrediction {
                series_id: "PU1->DO2".to_string(),
                timestamp: ts(3),
                horizon: 1,
                model: model.to_string(),
                mean,
            }],
            vec![ForecastIntervalPrediction {
                series_id: "PU1->DO2".to_string(),
                timestamp: ts(3),
                horizon: 1,
                model: model.to_string(),
                level: 0.8,
                lower,
                upper,
            }],
            vec![ForecastPredictionDetail {
                series_id: "PU1->DO2".to_string(),
                timestamp: ts(3),
                horizon: 1,
                model: model.to_string(),
                base_mean,
                spatial_correction,
                kriging_variance,
                selected_neighbors: vec!["PU2->DO3".to_string()],
                component_decomposition: Some(json!({"trend": mean})),
                metadata: Some(json!({"source": "inner"})),
            }],
        )
        .expect("rich result")
    }

    fn ts(day: u32) -> chrono::NaiveDateTime {
        NaiveDate::from_ymd_opt(2024, 1, day)
            .expect("date")
            .and_hms_opt(0, 0, 0)
            .expect("time")
    }
}
