impl ThetaForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        let adjusted = forecast_theta_component(&series.component, step);
                        let fitted_seasonality =
                            series.seasonal_pattern.as_ref().and(self.seasonality);
                        let mean = reseasonalize_value(
                            adjusted,
                            series.n_obs + step - 1,
                            fitted_seasonality,
                            series.seasonal_pattern.as_deref(),
                        )?;
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl ArimaForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                series
                    .forecast_values(horizon)
                    .into_iter()
                    .enumerate()
                    .map(|(idx, mean)| {
                        let step = idx + 1;
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl FittedLocalState {
    fn from_frame(frame: &ForecastFrame) -> Self {
        Self::from_frame_with_anchor(frame, frame)
    }

    fn from_frame_with_anchor(frame: &ForecastFrame, anchor_frame: &ForecastFrame) -> Self {
        let mut history_by_series: BTreeMap<String, Vec<ForecastRow>> = BTreeMap::new();
        for row in frame.rows() {
            history_by_series
                .entry(row.series_id.clone())
                .or_default()
                .push(row.clone());
        }
        let mut anchor_timestamp_by_series = BTreeMap::new();
        for row in anchor_frame.rows() {
            anchor_timestamp_by_series.insert(row.series_id.clone(), row.timestamp);
        }
        Self {
            frame: frame.clone(),
            history_by_series,
            anchor_timestamp_by_series,
        }
    }
}

impl FittedETSState {
    fn from_frame(
        frame: &ForecastFrame,
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
        damping_phi: f64,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedETSSeries::fit(
                        series_id,
                        history,
                        alpha,
                        beta,
                        gamma,
                        season_length,
                        damping_phi,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .skip(1)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

impl FittedArimaState {
    fn from_frame(frame: &ForecastFrame, p: usize, d: usize, q: usize) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedArimaSeries::fit(series_id, history, p, d, q)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .skip(series.score_start)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    fn has_stable_ar_recursions(&self) -> bool {
        self.series
            .values()
            .all(|series| ar_recursion_is_stable(&series.ar_coefficients))
    }

    fn has_invertible_ma_recursions(&self) -> bool {
        self.series
            .values()
            .all(|series| ma_recursion_is_invertible(&series.ma_coefficients))
    }
}

impl FittedPiecewiseLinearSeasonalState {
    fn from_frame(frame: &ForecastFrame, config: PiecewiseLinearSeasonalConfig) -> Result<Self> {
        Self::from_frame_with_anchor(frame, frame, config)
    }

    fn from_frame_with_anchor(
        frame: &ForecastFrame,
        anchor_frame: &ForecastFrame,
        config: PiecewiseLinearSeasonalConfig,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedPiecewiseLinearSeasonalSeries::fit(series_id, history, &config)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mut anchor_timestamp_by_series = BTreeMap::new();
        for row in anchor_frame.rows() {
            anchor_timestamp_by_series.insert(row.series_id.clone(), row.timestamp);
        }
        Ok(Self {
            frame: frame.clone(),
            series,
            history_frame: Some(anchor_frame.clone()),
            anchor_timestamp_by_series,
        })
    }

    fn root_mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            (sum / count as f64).sqrt()
        }
    }
}

impl FittedPiecewiseLinearSeasonalSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for piecewise linear seasonal forecasting"
            )));
        }
        let start_timestamp = history[0].timestamp;
        let last_timestamp = history[history.len() - 1].timestamp;
        let elapsed = history
            .iter()
            .map(|row| elapsed_days(start_timestamp, row.timestamp))
            .collect::<Vec<_>>();
        let changepoints = select_piecewise_changepoints(
            series_id,
            start_timestamp,
            last_timestamp,
            &elapsed,
            config,
        )?;
        let regressor_stats = piecewise_regressor_stats(history, config)?;
        let feature_count = piecewise_linear_seasonal_feature_count(config, changepoints.len());
        let trend_coefficients =
            fit_piecewise_trend_coefficients(history, &elapsed, &changepoints, config)?;
        let compute_covariance = piecewise_needs_coefficient_covariance(config);
        let mut fit_result = fit_piecewise_linear_weighted_coefficients(
            history,
            &elapsed,
            &changepoints,
            config,
            &regressor_stats,
            &trend_coefficients,
            None,
            compute_covariance && config.fit_loss != PiecewiseLinearFitLoss::Huber,
        )?
        .ok_or_else(|| {
            CartoBoostError::InvalidInput(format!(
                "could not solve piecewise linear seasonal normal equations for series {series_id}"
            ))
        })?;
        let mut coefficients = fit_result.coefficients.clone();
        if config.fit_loss == PiecewiseLinearFitLoss::Huber && config.irls_iterations > 0 {
            for iteration in 0..config.irls_iterations {
                let residuals = piecewise_transformed_residuals(
                    history,
                    &elapsed,
                    &changepoints,
                    config,
                    &regressor_stats,
                    &trend_coefficients,
                    &coefficients,
                )?;
                let weights = huber_irls_weights(&residuals, config.huber_delta);
                fit_result = fit_piecewise_linear_weighted_coefficients(
                    history,
                    &elapsed,
                    &changepoints,
                    config,
                    &regressor_stats,
                    &trend_coefficients,
                    Some(&weights),
                    compute_covariance && iteration + 1 == config.irls_iterations,
                )?
                .ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "could not solve robust piecewise linear seasonal normal equations for series {series_id}"
                    ))
                })?;
                let max_delta =
                    max_abs_difference(coefficients.as_slice(), fit_result.coefficients.as_slice());
                coefficients = fit_result.coefficients.clone();
                if max_delta < 1.0e-8 {
                    if compute_covariance && fit_result.coefficient_covariance.is_empty() {
                        fit_result = fit_piecewise_linear_weighted_coefficients(
                            history,
                            &elapsed,
                            &changepoints,
                            config,
                            &regressor_stats,
                            &trend_coefficients,
                            Some(&weights),
                            true,
                        )?
                        .ok_or_else(|| {
                            CartoBoostError::InvalidInput(format!(
                                "could not solve robust piecewise linear covariance for series {series_id}"
                            ))
                        })?;
                        coefficients = fit_result.coefficients.clone();
                    }
                    break;
                }
            }
        }
        let transformed_residual_scale = if compute_covariance
            || !config.interval_levels.is_empty()
            || !config.quantile_levels.is_empty()
            || config.uncertainty_samples > 0
        {
            piecewise_transformed_residual_scale(
                history,
                &elapsed,
                &changepoints,
                config,
                &regressor_stats,
                &coefficients,
            )?
        } else {
            0.0
        };
        let residuals = history
            .iter()
            .zip(elapsed.iter())
            .map(|(row, &t)| {
                let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
                Ok(row.target
                    - inverse_piecewise_target(
                        predict_piecewise_linear_value(
                            t,
                            &coefficients,
                            &PiecewiseLinearFeatureContext {
                                series_id: None,
                                timestamp: row.timestamp,
                                covariates: Some(&row.covariates),
                                horizon_step: None,
                                component_multiplier: fit_component_multiplier(
                                    t,
                                    &coefficients,
                                    &changepoints,
                                    bounds,
                                    config,
                                ),
                                changepoints: &changepoints,
                                config,
                                regressor_stats: Some(&regressor_stats),
                            },
                        )?,
                        bounds,
                        config,
                    ))
            })
            .collect::<Result<Vec<_>>>()?;
        let trend_delta_scale = piecewise_trend_delta_scale(&coefficients, changepoints.len());
        Ok(Self {
            start_timestamp,
            last_timestamp,
            last_elapsed_days: elapsed.last().copied().unwrap_or(0.0),
            changepoints,
            coefficients,
            coefficient_covariance: fit_result.coefficient_covariance,
            feature_count,
            residuals,
            transformed_residual_scale,
            trend_delta_scale,
            regressor_stats,
        })
    }

    fn predict_component_record(
        &self,
        series_id: &str,
        elapsed_days: f64,
        timestamp: chrono::NaiveDateTime,
        step: usize,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Value> {
        debug_assert_eq!(self.feature_count, self.coefficients.len());
        let bounds = piecewise_bounds(Some(series_id), None, Some(step), config)?;
        let component_multiplier = self.component_multiplier(elapsed_days, bounds, config);
        let context = PiecewiseLinearFeatureContext {
            series_id: Some(series_id),
            timestamp,
            covariates: None,
            horizon_step: Some(step),
            component_multiplier,
            changepoints: &self.changepoints,
            config,
            regressor_stats: Some(&self.regressor_stats),
        };
        let mut features =
            vec![0.0; piecewise_linear_seasonal_feature_count(config, self.changepoints.len())];
        fill_piecewise_linear_seasonal_features(&mut features, elapsed_days, &context)?;
        let linear_predictor = features
            .iter()
            .zip(self.coefficients.iter())
            .map(|(feature, coefficient)| feature * coefficient)
            .sum::<f64>();
        let trend_linear =
            piecewise_trend_value(elapsed_days, &self.coefficients, &self.changepoints, config);
        let trend = inverse_piecewise_target(trend_linear, bounds, config);
        let trend_adjustment_multiplier = piecewise_trend_adjustment_multiplier(
            series_id,
            step,
            &config.trend_adjustments,
            &config.trend_adjustments_by_series,
        );
        let adjusted_trend = trend * trend_adjustment_multiplier;
        let trend_adjustment = adjusted_trend - trend;
        let residual_shock = self.residual_shock(step, config);
        let prediction = inverse_piecewise_target(linear_predictor, bounds, config)
            + trend_adjustment
            + residual_shock;
        let components = piecewise_component_contributions(
            &features,
            &self.coefficients,
            self.changepoints.len(),
            config,
        )?;
        Ok(json!({
            "series_id": series_id,
            "timestamp": timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "horizon": step,
            "prediction": prediction,
            "trend": trend,
            "adjusted_trend": adjusted_trend,
            "trend_adjustment_multiplier": trend_adjustment_multiplier,
            "trend_adjustment": trend_adjustment,
            "residual_shock": residual_shock,
            "linear_predictor": linear_predictor,
            "trend_linear": trend_linear,
            "component_scale": if config.growth == PiecewiseLinearGrowth::Logistic {
                "logistic_linear_predictor"
            } else {
                "prediction"
            },
            "components": components,
        }))
    }

    fn history_component_records(
        &self,
        series_id: &str,
        history: &[&ForecastRow],
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Vec<Value>> {
        let mut records = Vec::with_capacity(history.len());
        let mut previous_trend = None;
        let mut previous_fitted = None;
        for (idx, row) in history.iter().enumerate() {
            let record = self.history_component_record(
                series_id,
                row,
                idx,
                previous_trend,
                previous_fitted,
                config,
            )?;
            previous_trend = record["trend"].as_f64();
            previous_fitted = record["fitted"].as_f64();
            records.push(record);
        }
        Ok(records)
    }

    fn history_component_record(
        &self,
        series_id: &str,
        row: &ForecastRow,
        idx: usize,
        previous_trend: Option<f64>,
        previous_fitted: Option<f64>,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Value> {
        debug_assert_eq!(self.feature_count, self.coefficients.len());
        let elapsed = elapsed_days(self.start_timestamp, row.timestamp);
        let bounds = piecewise_bounds(None, Some(&row.covariates), None, config)?;
        let component_multiplier = self.component_multiplier(elapsed, bounds, config);
        let context = PiecewiseLinearFeatureContext {
            series_id: Some(series_id),
            timestamp: row.timestamp,
            covariates: Some(&row.covariates),
            horizon_step: None,
            component_multiplier,
            changepoints: &self.changepoints,
            config,
            regressor_stats: Some(&self.regressor_stats),
        };
        let mut features =
            vec![0.0; piecewise_linear_seasonal_feature_count(config, self.changepoints.len())];
        fill_piecewise_linear_seasonal_features(&mut features, elapsed, &context)?;
        let linear_predictor = features
            .iter()
            .zip(self.coefficients.iter())
            .map(|(feature, coefficient)| feature * coefficient)
            .sum::<f64>();
        let trend_linear =
            piecewise_trend_value(elapsed, &self.coefficients, &self.changepoints, config);
        let trend = inverse_piecewise_target(trend_linear, bounds, config);
        let fitted = inverse_piecewise_target(linear_predictor, bounds, config);
        let components = piecewise_component_contributions(
            &features,
            &self.coefficients,
            self.changepoints.len(),
            config,
        )?;
        Ok(json!({
            "series_id": series_id,
            "timestamp": row.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "index": idx,
            "actual": row.target,
            "fitted": fitted,
            "residual": row.target - fitted,
            "trend": trend,
            "adjusted_trend": trend,
            "trend_adjustment_multiplier": 1.0,
            "trend_adjustment": 0.0,
            "trend_movement": previous_trend.map(|previous| trend - previous),
            "fitted_movement": previous_fitted.map(|previous| fitted - previous),
            "linear_predictor": linear_predictor,
            "trend_linear": trend_linear,
            "component_scale": if config.growth == PiecewiseLinearGrowth::Logistic {
                "logistic_linear_predictor"
            } else {
                "prediction"
            },
            "components": components,
        }))
    }

    fn residual_scale(&self) -> f64 {
        if self.residuals.is_empty() {
            return 0.0;
        }
        let mse = self
            .residuals
            .iter()
            .map(|residual| residual * residual)
            .sum::<f64>()
            / self.residuals.len() as f64;
        mse.sqrt()
    }

    fn trend_uncertainty_offsets(
        &self,
        series_id: &str,
        elapsed_days: f64,
        timestamp: chrono::NaiveDateTime,
        step: usize,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<Vec<f64>> {
        if config.uncertainty_samples == 0
            || config.trend_uncertainty_scale <= 0.0
            || self.trend_delta_scale <= 0.0
            || config.growth == PiecewiseLinearGrowth::Flat
        {
            return Ok(Vec::new());
        }
        let future_elapsed = (elapsed_days - self.last_elapsed_days).max(0.0);
        if future_elapsed <= 0.0 {
            return Ok(Vec::new());
        }
        let scale = self.trend_delta_scale
            * config.trend_uncertainty_scale
            * future_elapsed
            * (step as f64).sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Ok(Vec::new());
        }
        let bounds = piecewise_bounds(Some(series_id), None, Some(step), config)?;
        let linear_predictor = predict_piecewise_linear_value(
            elapsed_days,
            &self.coefficients,
            &PiecewiseLinearFeatureContext {
                series_id: Some(series_id),
                timestamp,
                covariates: None,
                horizon_step: Some(step),
                component_multiplier: self.component_multiplier(elapsed_days, bounds, config),
                changepoints: &self.changepoints,
                config,
                regressor_stats: Some(&self.regressor_stats),
            },
        )?;
        let derivative =
            inverse_piecewise_target_derivative(linear_predictor, bounds, config).abs();
        if !derivative.is_finite() || derivative <= 0.0 {
            return Ok(Vec::new());
        }
        let series_hash = stable_hash64(series_id.as_bytes());
        Ok((0..config.uncertainty_samples)
            .map(|sample| {
                let draw = deterministic_trend_uncertainty_draw(
                    config.uncertainty_seed ^ series_hash,
                    step as u64,
                    sample as u64,
                    config.trend_uncertainty_policy,
                );
                draw * scale * derivative
            })
            .collect())
    }

    fn trend_uncertainty_linear_offsets(
        &self,
        series_id: &str,
        elapsed_days: f64,
        step: usize,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Vec<f64> {
        if config.uncertainty_samples == 0
            || config.trend_uncertainty_scale <= 0.0
            || self.trend_delta_scale <= 0.0
            || config.growth == PiecewiseLinearGrowth::Flat
        {
            return Vec::new();
        }
        let future_elapsed = (elapsed_days - self.last_elapsed_days).max(0.0);
        if future_elapsed <= 0.0 {
            return Vec::new();
        }
        let scale = self.trend_delta_scale
            * config.trend_uncertainty_scale
            * future_elapsed
            * (step as f64).sqrt();
        if !scale.is_finite() || scale <= 0.0 {
            return Vec::new();
        }
        let series_hash = stable_hash64(series_id.as_bytes());
        (0..config.uncertainty_samples)
            .map(|sample| {
                deterministic_trend_uncertainty_draw(
                    config.uncertainty_seed ^ series_hash,
                    step as u64,
                    sample as u64,
                    config.trend_uncertainty_policy,
                ) * scale
            })
            .collect()
    }

    fn prediction_terms_at(
        &self,
        series_id: &str,
        elapsed_days: f64,
        timestamp: chrono::NaiveDateTime,
        step: usize,
        bounds: PiecewiseBounds,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Result<PiecewisePredictionTerms> {
        let context = PiecewiseLinearFeatureContext {
            series_id: Some(series_id),
            timestamp,
            covariates: None,
            horizon_step: Some(step),
            component_multiplier: self.component_multiplier(elapsed_days, bounds, config),
            changepoints: &self.changepoints,
            config,
            regressor_stats: Some(&self.regressor_stats),
        };
        let mut features =
            vec![0.0; piecewise_linear_seasonal_feature_count(config, self.changepoints.len())];
        fill_piecewise_linear_seasonal_features(&mut features, elapsed_days, &context)?;
        let linear_predictor = features
            .iter()
            .zip(self.coefficients.iter())
            .map(|(feature, coefficient)| feature * coefficient)
            .sum::<f64>();
        let mean = inverse_piecewise_target(linear_predictor, bounds, config);
        let trend_linear =
            piecewise_trend_value(elapsed_days, &self.coefficients, &self.changepoints, config);
        let trend = inverse_piecewise_target(trend_linear, bounds, config);
        let trend_adjustment_multiplier = piecewise_trend_adjustment_multiplier(
            series_id,
            step,
            &config.trend_adjustments,
            &config.trend_adjustments_by_series,
        );
        let adjusted_trend = trend * trend_adjustment_multiplier;
        let residual_shock = self.residual_shock(step, config);
        let adjusted_mean = mean + adjusted_trend - trend + residual_shock;
        let variance = if config.coefficient_uncertainty_scale > 0.0
            && !self.coefficient_covariance.is_empty()
            && self.transformed_residual_scale > 0.0
        {
            quadratic_form(&features, &self.coefficient_covariance).max(0.0)
        } else {
            0.0
        };
        let linear_coefficient_scale = if variance.is_finite() && variance > 0.0 {
            config.coefficient_uncertainty_scale * self.transformed_residual_scale * variance.sqrt()
        } else {
            0.0
        };
        let derivative = inverse_piecewise_target_derivative(linear_predictor, bounds, config);
        let coefficient_scale = linear_coefficient_scale * derivative.abs();
        Ok(PiecewisePredictionTerms {
            mean: adjusted_mean,
            linear_predictor,
            coefficient_scale: if coefficient_scale.is_finite() {
                coefficient_scale
            } else {
                0.0
            },
            linear_coefficient_scale: if linear_coefficient_scale.is_finite() {
                linear_coefficient_scale
            } else {
                0.0
            },
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn predictive_sample_record(
        &self,
        series_id: &str,
        timestamp: chrono::NaiveDateTime,
        step: usize,
        sample: usize,
        mean: f64,
        linear_predictor: f64,
        bounds: PiecewiseBounds,
        residual_scale: f64,
        coefficient_scale: f64,
        linear_coefficient_scale: f64,
        trend_draw: f64,
        linear_trend_draw: f64,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> Value {
        let series_hash = stable_hash64(series_id.as_bytes());
        let residual_z = deterministic_standard_normal(
            config.uncertainty_seed ^ series_hash ^ 0x8f6b_3c2d_19e0_a457,
            step as u64,
            sample as u64,
        );
        let coefficient_z = deterministic_standard_normal(
            config.uncertainty_seed ^ series_hash ^ 0x4d31_2f9a_b875_c913,
            step as u64,
            sample as u64,
        );
        let residual_draw = residual_scale * residual_z;
        let coefficient_draw = coefficient_scale * coefficient_z;
        let prediction = match config.growth {
            PiecewiseLinearGrowth::Logistic => clamp_piecewise_logistic_interior_value(
                inverse_piecewise_target(
                    linear_predictor
                        + self.transformed_residual_scale * residual_z
                        + linear_coefficient_scale * coefficient_z
                        + linear_trend_draw,
                    bounds,
                    config,
                ),
                bounds,
                config,
            ),
            PiecewiseLinearGrowth::Linear | PiecewiseLinearGrowth::Flat => {
                mean + residual_draw + coefficient_draw + trend_draw
            }
        };
        json!({
            "series_id": series_id,
            "timestamp": timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
            "horizon": step,
            "sample": sample,
            "prediction": prediction,
            "mean": mean,
            "residual_draw": residual_draw,
            "coefficient_draw": coefficient_draw,
            "trend_draw": trend_draw,
        })
    }

    fn component_multiplier(
        &self,
        elapsed_days: f64,
        bounds: PiecewiseBounds,
        config: &PiecewiseLinearSeasonalConfig,
    ) -> f64 {
        match config.component_mode {
            PiecewiseLinearComponentMode::Additive => 1.0,
            PiecewiseLinearComponentMode::Multiplicative => {
                piecewise_component_multiplier_from_trend(
                    self.trend_component(elapsed_days, config),
                    bounds,
                    config,
                )
            }
        }
    }

    fn trend_component(&self, elapsed_days: f64, config: &PiecewiseLinearSeasonalConfig) -> f64 {
        piecewise_trend_value(elapsed_days, &self.coefficients, &self.changepoints, config)
    }

    fn residual_shock(&self, step: usize, config: &PiecewiseLinearSeasonalConfig) -> f64 {
        if config.residual_shock_window == 0 || config.residual_shock_scale <= 0.0 {
            return 0.0;
        }
        let window = config.residual_shock_window.min(self.residuals.len());
        if window == 0 {
            return 0.0;
        }
        let recent = &self.residuals[self.residuals.len() - window..];
        let first = recent[0];
        if first == 0.0 || !first.is_finite() {
            return 0.0;
        }
        let sign = first.signum();
        if recent
            .iter()
            .any(|residual| !residual.is_finite() || *residual == 0.0 || residual.signum() != sign)
        {
            return 0.0;
        }
        let average = recent.iter().sum::<f64>() / window as f64;
        average
            * config.residual_shock_scale
            * config
                .residual_shock_decay
                .powi(step.saturating_sub(1) as i32)
    }
}

fn piecewise_trend_adjustment_multiplier(
    series_id: &str,
    step: usize,
    global: &BTreeMap<usize, f64>,
    by_series: &BTreeMap<String, BTreeMap<usize, f64>>,
) -> f64 {
    by_series
        .get(series_id)
        .and_then(|values| values.get(&step))
        .or_else(|| global.get(&step))
        .copied()
        .unwrap_or(1.0)
}

#[allow(clippy::too_many_arguments)]
fn piecewise_prediction_intervals(
    prediction: &ForecastPrediction,
    residual_scale: f64,
    coefficient_scale: f64,
    linear_residual_scale: f64,
    linear_predictor: f64,
    linear_coefficient_scale: f64,
    mut trend_offsets: Vec<f64>,
    mut linear_trend_offsets: Vec<f64>,
    levels: &[f64],
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> Result<Vec<ForecastIntervalPrediction>> {
    if levels.is_empty() {
        return Ok(Vec::new());
    }
    if !residual_scale.is_finite() || residual_scale < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal residual scale must be finite and nonnegative".to_string(),
        ));
    }
    if !coefficient_scale.is_finite() || coefficient_scale < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "piecewise seasonal coefficient uncertainty scale must be finite and nonnegative"
                .to_string(),
        ));
    }
    let predictive_scale =
        (residual_scale * residual_scale + coefficient_scale * coefficient_scale).sqrt();
    let linear_predictive_scale = (linear_residual_scale * linear_residual_scale
        + linear_coefficient_scale * linear_coefficient_scale)
        .sqrt();
    if !trend_offsets.is_empty() {
        trend_offsets.sort_by(|a, b| a.total_cmp(b));
    }
    if !linear_trend_offsets.is_empty() {
        linear_trend_offsets.sort_by(|a, b| a.total_cmp(b));
    }
    levels
        .iter()
        .map(|&level| {
            let (lower, upper) = if config.growth == PiecewiseLinearGrowth::Logistic {
                piecewise_logistic_interval_bounds(
                    linear_predictor,
                    linear_predictive_scale,
                    &linear_trend_offsets,
                    level,
                    bounds,
                    config,
                )
            } else if trend_offsets.is_empty() {
                let alpha = (1.0 + level) / 2.0;
                let width = inverse_standard_normal_cdf(alpha) * predictive_scale;
                (prediction.mean - width, prediction.mean + width)
            } else {
                piecewise_sampled_interval_bounds(
                    prediction.mean,
                    predictive_scale,
                    &trend_offsets,
                    level,
                )
            };
            let (lower, upper) = clamp_piecewise_interval_bounds(lower, upper, bounds, config);
            Ok(ForecastIntervalPrediction {
                series_id: prediction.series_id.clone(),
                timestamp: prediction.timestamp,
                horizon: prediction.horizon,
                model: prediction.model.clone(),
                level,
                lower,
                upper,
            })
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn piecewise_prediction_quantiles(
    prediction: &ForecastPrediction,
    residual_scale: f64,
    coefficient_scale: f64,
    linear_residual_scale: f64,
    linear_predictor: f64,
    linear_coefficient_scale: f64,
    mut trend_offsets: Vec<f64>,
    mut linear_trend_offsets: Vec<f64>,
    levels: &[f64],
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> Vec<Value> {
    let predictive_scale =
        (residual_scale * residual_scale + coefficient_scale * coefficient_scale).sqrt();
    let linear_predictive_scale = (linear_residual_scale * linear_residual_scale
        + linear_coefficient_scale * linear_coefficient_scale)
        .sqrt();
    if !trend_offsets.is_empty() {
        trend_offsets.sort_by(|a, b| a.total_cmp(b));
    }
    if !linear_trend_offsets.is_empty() {
        linear_trend_offsets.sort_by(|a, b| a.total_cmp(b));
    }
    levels
        .iter()
        .map(|&level| {
            let z = inverse_standard_normal_cdf(level);
            let value = if config.growth == PiecewiseLinearGrowth::Logistic {
                let linear_value = if linear_trend_offsets.is_empty() {
                    linear_predictor + z * linear_predictive_scale
                } else {
                    linear_predictor
                        + quantile_from_sorted(&linear_trend_offsets, level)
                        + z * linear_predictive_scale
                };
                inverse_piecewise_target(linear_value, bounds, config)
            } else if trend_offsets.is_empty() {
                prediction.mean + z * predictive_scale
            } else {
                prediction.mean + quantile_from_sorted(&trend_offsets, level) + z * predictive_scale
            };
            let value = clamp_piecewise_logistic_interior_value(value, bounds, config);
            json!({
                "series_id": prediction.series_id.clone(),
                "timestamp": prediction.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                "horizon": prediction.horizon,
                "quantile": level,
                "prediction": value,
                "mean": prediction.mean,
            })
        })
        .collect()
}

fn clamp_piecewise_logistic_interior_value(
    value: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> f64 {
    if config.growth != PiecewiseLinearGrowth::Logistic {
        return value;
    }
    let cap = bounds.cap.expect("validated logistic cap");
    let span = cap - bounds.floor;
    let epsilon = (span.abs() * 1.0e-12).max(f64::EPSILON);
    value.max(bounds.floor + epsilon).min(cap - epsilon)
}

fn clamp_piecewise_interval_bounds(
    lower: f64,
    upper: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> (f64, f64) {
    if config.growth != PiecewiseLinearGrowth::Logistic {
        return (lower, upper);
    }
    let cap = bounds.cap.expect("validated logistic cap");
    (lower.max(bounds.floor), upper.min(cap))
}

fn piecewise_logistic_interval_bounds(
    linear_predictor: f64,
    linear_predictive_scale: f64,
    sorted_linear_trend_offsets: &[f64],
    level: f64,
    bounds: PiecewiseBounds,
    config: &PiecewiseLinearSeasonalConfig,
) -> (f64, f64) {
    let residual_width = inverse_standard_normal_cdf((1.0 + level) / 2.0) * linear_predictive_scale;
    let (linear_lower, linear_upper) = if sorted_linear_trend_offsets.is_empty() {
        (
            linear_predictor - residual_width,
            linear_predictor + residual_width,
        )
    } else {
        let lower_q = (1.0 - level) / 2.0;
        let upper_q = 1.0 - lower_q;
        (
            linear_predictor + quantile_from_sorted(sorted_linear_trend_offsets, lower_q)
                - residual_width,
            linear_predictor
                + quantile_from_sorted(sorted_linear_trend_offsets, upper_q)
                + residual_width,
        )
    };
    (
        inverse_piecewise_target(linear_lower, bounds, config),
        inverse_piecewise_target(linear_upper, bounds, config),
    )
}

fn piecewise_sampled_interval_bounds(
    mean: f64,
    residual_scale: f64,
    sorted_trend_offsets: &[f64],
    level: f64,
) -> (f64, f64) {
    let residual_width = inverse_standard_normal_cdf((1.0 + level) / 2.0) * residual_scale;
    let lower_q = (1.0 - level) / 2.0;
    let upper_q = 1.0 - lower_q;
    (
        mean + quantile_from_sorted(sorted_trend_offsets, lower_q) - residual_width,
        mean + quantile_from_sorted(sorted_trend_offsets, upper_q) + residual_width,
    )
}

fn quantile_from_sorted(values: &[f64], probability: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let bounded = probability.clamp(0.0, 1.0);
    let position = bounded * (values.len().saturating_sub(1)) as f64;
    let lower_idx = position.floor() as usize;
    let upper_idx = position.ceil() as usize;
    if lower_idx == upper_idx {
        values[lower_idx]
    } else {
        let weight = position - lower_idx as f64;
        values[lower_idx] * (1.0 - weight) + values[upper_idx] * weight
    }
}

fn inverse_standard_normal_cdf(probability: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if probability <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if probability >= 1.0 {
        return f64::INFINITY;
    }
    if probability < P_LOW {
        let q = (-2.0 * probability.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if probability <= P_HIGH {
        let q = probability - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - probability).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

impl KalmanForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean: series.level + step as f64 * series.trend,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl LocalLevelKalmanForecaster {
    fn predict_with_model_name(
        &self,
        horizon: usize,
        model_name: &'static str,
    ) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let predictions = fitted
            .series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, series)| {
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: model_name.to_string(),
                            mean: series.level,
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();
        ForecastResult::new(predictions)
    }
}

impl FittedKalmanState {
    fn from_frame(
        frame: &ForecastFrame,
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedKalmanSeries::fit(
                        series_id,
                        history,
                        level_process_variance,
                        trend_process_variance,
                        observation_variance,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }
}

impl FittedLocalLevelKalmanState {
    fn from_frame(
        frame: &ForecastFrame,
        level_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedLocalLevelKalmanSeries::fit(
                        series_id,
                        history,
                        level_process_variance,
                        observation_variance,
                    )?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }
}

impl FittedKalmanSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for local-linear kalman forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let config = LocalLinearKalmanConfig::new(
            level_process_variance,
            trend_process_variance,
            observation_variance,
        )?;
        let result = fit_local_linear_kalman(&values, config)
            .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
        Ok(Self {
            last_timestamp: history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?
                .timestamp,
            level: result.final_state.level,
            trend: result.final_state.trend,
        })
    }
}

impl FittedLocalLevelKalmanSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        level_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        if history.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least one row for local level kalman forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let config = LocalLevelKalmanConfig::new(level_process_variance, observation_variance)?;
        let result = fit_local_level_kalman(&values, config)
            .map_err(|err| CartoBoostError::InvalidInput(format!("{series_id}: {err}")))?;
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            level: result.final_level,
        })
    }
}

impl FittedKrigingState {
    fn from_frame(
        frame: &ForecastFrame,
        coordinates: &BTreeMap<String, (f64, f64)>,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let mut levels = BTreeMap::new();
        for (series_id, history) in &local.history_by_series {
            if !coordinates.contains_key(series_id) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "missing kriging coordinate for series {series_id}"
                )));
            }
            let last = history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?;
            levels.insert(series_id.clone(), last.target);
        }
        Ok(Self {
            frame: frame.clone(),
            levels,
        })
    }
}

impl FittedSpatialPiecewiseKrigingState {
    fn from_frame(
        frame: &ForecastFrame,
        config: &SpatialPiecewiseKrigingConfig,
        backend: &BackendSelection,
    ) -> Result<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let started = std::time::Instant::now();
        #[cfg(target_arch = "wasm32")]
        let started = ();
        let piecewise_config = spatial_piecewise_base_config(config);
        let modeled_frame = if config.uses_kriged_regressors() {
            kriged_regressor_frame(frame, config, backend)?
        } else {
            frame.clone()
        };
        let mut base = PiecewiseLinearSeasonalForecaster::new(piecewise_config)?;
        base.fit(&modeled_frame)?;
        let base_fitted = base.fitted.as_ref().ok_or_else(not_fitted)?;
        let local = FittedLocalState::from_frame(&modeled_frame);
        let mut residual_levels = BTreeMap::new();
        let mut residual_observation_series = Vec::new();
        let mut cutoff_timestamps = BTreeMap::new();
        for (series_id, history) in &local.history_by_series {
            let fitted_series = base_fitted.series.get(series_id).ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "missing fitted base series for spatial piecewise kriging series {series_id}"
                ))
            })?;
            let last_residual = fitted_series.residuals.last().copied().ok_or_else(|| {
                CartoBoostError::InvalidInput(format!(
                    "series {series_id} has no residuals for residual kriging"
                ))
            })?;
            if !last_residual.is_finite() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "series {series_id} residual is not finite"
                )));
            }
            let last_timestamp = history
                .last()
                .ok_or_else(|| CartoBoostError::InvalidInput("empty series history".to_string()))?
                .timestamp;
            residual_levels.insert(series_id.clone(), last_residual);
            residual_observation_series.push(series_id.clone());
            cutoff_timestamps.insert(series_id.clone(), last_timestamp);
        }
        residual_observation_series.sort();
        let residual_rmse = base_fitted.root_mean_squared_residual();
        let fit_metadata = json!({
            "cutoffs": cutoff_timestamps.iter().map(|(series_id, timestamp)| {
                (series_id.clone(), timestamp.format("%Y-%m-%dT%H:%M:%S").to_string())
            }).collect::<BTreeMap<_, _>>(),
            "residual_rmse": residual_rmse,
            "runtime_seconds": spatial_piecewise_runtime_seconds(&started),
        });
        Ok(Self {
            frame: modeled_frame,
            base,
            residual_levels,
            residual_observation_series,
            cutoff_timestamps,
            fit_metadata,
        })
    }

    fn residual_kriging_predictions(
        &self,
        config: &SpatialPiecewiseKrigingConfig,
        backend: &BackendSelection,
    ) -> Result<BTreeMap<String, SpatialKrigingCorrection>> {
        let observations = self
            .residual_observation_series
            .iter()
            .map(|series_id| {
                let coord = config.coordinates.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing spatial piecewise kriging coordinate for series {series_id}"
                    ))
                })?;
                let value = self
                    .residual_levels
                    .get(series_id)
                    .copied()
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput(format!(
                            "missing residual kriging level for series {series_id}"
                        ))
                    })?;
                Ok(KrigingObservation {
                    x: coord.0,
                    y: coord.1,
                    value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if observations.len() < 2 {
            return Err(CartoBoostError::InvalidInput(
                "residual kriging requires at least two spatial series so each target can be held out"
                    .to_string(),
            ));
        }
        let mut corrections = BTreeMap::new();
        for (held_out_idx, series_id) in self.residual_observation_series.iter().enumerate() {
            let target_observation = observations[held_out_idx];
            let training_rows = observations
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != held_out_idx)
                .map(|(idx, observation)| (idx, *observation))
                .collect::<Vec<_>>();
            let training = training_rows
                .iter()
                .map(|(_, observation)| *observation)
                .collect::<Vec<_>>();
            let target = (target_observation.x, target_observation.y);
            let correction = match ordinary_kriging_predict_many_with_backend(
                &training,
                &[target],
                config.kriging_config,
                Some(&backend.selected),
            ) {
                    Ok(mut predictions) => {
                        let mut prediction = predictions.remove(0);
                        prediction.neighbor_indices = prediction
                            .neighbor_indices
                            .iter()
                            .map(|local_idx| training_rows[*local_idx].0)
                            .collect();
                        SpatialKrigingCorrection {
                            prediction,
                            used_neighbor_fallback: false,
                        }
                    }
                    Err(error)
                        if config.allow_neighbor_fallback && is_neighbor_rule_error(&error) =>
                    {
                        SpatialKrigingCorrection {
                            prediction: crate::utilities::KrigingPrediction {
                                x: target.0,
                                y: target.1,
                                mean: 0.0,
                                variance: residual_level_variance(&self.residual_levels),
                                weights: Vec::new(),
                                neighbor_indices: Vec::new(),
                            },
                            used_neighbor_fallback: true,
                        }
                    }
                    Err(error) => return Err(error),
                };
            corrections.insert(series_id.clone(), correction);
        }
        Ok(corrections)
    }

    fn future_spatial_regressors(
        &self,
        horizon: usize,
        config: &SpatialPiecewiseKrigingConfig,
    ) -> Result<BTreeMap<String, BTreeMap<String, Vec<f64>>>> {
        let mut latest_by_series: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
        for row in self.frame.rows() {
            let entry = latest_by_series.entry(row.series_id.clone()).or_default();
            for name in &config.spatial_regressors {
                if let Some(value) = row.covariates.get(name) {
                    entry.insert(name.clone(), *value);
                }
            }
        }
        Ok(latest_by_series
            .into_iter()
            .map(|(series_id, values)| {
                let repeated = values
                    .into_iter()
                    .map(|(name, value)| (name, vec![value; horizon]))
                    .collect::<BTreeMap<_, _>>();
                (series_id, repeated)
            })
            .collect())
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn spatial_piecewise_runtime_seconds(started: &std::time::Instant) -> f64 {
    started.elapsed().as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
fn spatial_piecewise_runtime_seconds(_: &()) -> f64 {
    0.0
}

impl FittedETSSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        alpha: f64,
        beta: f64,
        gamma: Option<f64>,
        season_length: Option<usize>,
        damping_phi: f64,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for ETS forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let mut seasonals = match season_length {
            Some(length) => {
                if values.len() < length * 2 {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "series {series_id} requires at least two full seasonal cycles for ETS seasonality"
                    )));
                }
                let (_, pattern) = deseasonalize(
                    series_id,
                    &values,
                    Some(ThetaSeasonality::additive(length)?),
                )?;
                pattern
            }
            None => None,
        };
        let mut level = values[0] - seasonals.as_ref().map(|s| s[0]).unwrap_or(0.0);
        let mut trend = initial_trend(&values, seasonals.as_deref());
        let mut fitted_values = Vec::with_capacity(values.len());
        let mut residuals = Vec::with_capacity(values.len());
        let mut level_values = Vec::with_capacity(values.len());
        let mut trend_values = Vec::with_capacity(values.len());
        let mut seasonal_values = Vec::with_capacity(values.len());
        fitted_values.push(values[0]);
        residuals.push(0.0);
        level_values.push(level);
        trend_values.push(trend);
        seasonal_values.push(seasonals.as_ref().map(|s| s[0]).unwrap_or(0.0));

        for (idx, value) in values.iter().enumerate().skip(1) {
            let seasonal_idx = seasonals.as_ref().map(|seasonals| idx % seasonals.len());
            let seasonal = seasonal_idx
                .and_then(|seasonal_idx| {
                    seasonals.as_ref().map(|seasonals| seasonals[seasonal_idx])
                })
                .unwrap_or(0.0);
            let fitted = level + damping_phi * trend + seasonal;
            fitted_values.push(fitted);
            residuals.push(*value - fitted);

            let previous_level = level;
            level = alpha * (*value - seasonal) + (1.0 - alpha) * (level + damping_phi * trend);
            trend = beta * (level - previous_level) + (1.0 - beta) * damping_phi * trend;
            if let (Some(gamma), Some(seasonal_idx), Some(seasonals)) =
                (gamma, seasonal_idx, seasonals.as_mut())
            {
                seasonals[seasonal_idx] =
                    gamma * (*value - level) + (1.0 - gamma) * seasonals[seasonal_idx];
            }
            level_values.push(level);
            trend_values.push(trend);
            seasonal_values.push(seasonal);
        }

        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            n_obs: history.len(),
            level,
            trend,
            damping_phi,
            seasonals,
            fitted_values,
            residuals,
            level_values,
            trend_values,
            seasonal_values,
        })
    }

    fn forecast_values(&self, horizon: usize) -> Vec<f64> {
        (1..=horizon)
            .map(|step| {
                let seasonal = self
                    .seasonals
                    .as_ref()
                    .map(|seasonals| seasonals[(self.n_obs + step - 1) % seasonals.len()])
                    .unwrap_or(0.0);
                self.level + damped_trend_multiplier(self.damping_phi, step) * self.trend + seasonal
            })
            .collect()
    }
}

impl FittedArimaSeries {
    fn fit(series_id: &str, history: &[ForecastRow], p: usize, d: usize, q: usize) -> Result<Self> {
        validate_arima_order(p, d, q)?;
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let (effective_p, effective_d, effective_q) =
            arima_order_supported_by_history(values.len(), p, d, q);
        if (effective_p, effective_d, effective_q) != (p, d, q) {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} has {} rows, which cannot support requested ARIMA({p},{d},{q}); maximum supported order is ARIMA({effective_p},{effective_d},{effective_q})",
                values.len()
            )));
        }
        let differences = difference_series(&values, effective_d)?;
        let required_lags = effective_p.max(effective_q);
        if differences.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} has no differenced rows available for ARIMA fitting",
            )));
        }
        let (intercept, ar_coefficients, ma_coefficients, fitted_diff, residuals) =
            fit_arima_components(&differences, effective_p, effective_q)?;
        let fitted_values = undifference_fitted_values(&values, &fitted_diff, effective_d);
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            intercept,
            ar_coefficients,
            ma_coefficients,
            score_start: required_lags,
            differenced_history: differences,
            residual_history: residuals.clone(),
            last_differences: last_differences(&values, effective_d)?,
            fitted_values,
            residuals,
        })
    }

    fn forecast_values(&self, horizon: usize) -> Vec<f64> {
        let p = self.ar_coefficients.len();
        let q = self.ma_coefficients.len();
        let mut differenced = tail_values(&self.differenced_history, p);
        let mut residuals = tail_values(&self.residual_history, q);
        let mut levels = self.last_differences.clone();
        let mut forecasts = Vec::with_capacity(horizon);
        for _ in 0..horizon {
            let next_diff = forecast_arima_next(
                &differenced,
                &residuals,
                self.intercept,
                &self.ar_coefficients,
                &self.ma_coefficients,
            );
            push_tail(&mut differenced, p, next_diff);
            push_tail(&mut residuals, q, 0.0);
            let mut value = next_diff;
            for idx in (0..(levels.len() - 1)).rev() {
                levels[idx] += value;
                value = levels[idx];
            }
            forecasts.push(value);
        }
        forecasts
    }
}

impl FittedThetaState {
    fn from_frame(
        frame: &ForecastFrame,
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        let local = FittedLocalState::from_frame(frame);
        let series = local
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                Ok((
                    series_id.clone(),
                    FittedThetaSeries::fit(series_id, history, theta, alpha, seasonality)?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            frame: frame.clone(),
            series,
        })
    }

    fn mean_squared_residual(&self) -> f64 {
        let (sum, count) = self
            .series
            .values()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|series| {
                series
                    .residuals
                    .iter()
                    .skip(1)
                    .fold((0.0, 0usize), |(sum, count), residual| {
                        (sum + residual * residual, count + 1)
                    })
            })
            .reduce(
                || (0.0, 0usize),
                |left, right| (left.0 + right.0, left.1 + right.1),
            );
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }
}

impl FittedThetaSeries {
    fn fit(
        series_id: &str,
        history: &[ForecastRow],
        theta: f64,
        alpha: f64,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Self> {
        if history.len() < 2 {
            return Err(CartoBoostError::InvalidInput(format!(
                "series {series_id} requires at least two rows for theta forecasting"
            )));
        }
        let values = history.iter().map(|row| row.target).collect::<Vec<_>>();
        let (adjusted, pattern) = deseasonalize(series_id, &values, seasonality)?;
        let component = fit_theta_component(&adjusted, theta, alpha);
        let fitted_adjusted = fitted_theta_values(&adjusted, theta, alpha);
        let mut fitted_values = Vec::with_capacity(values.len());
        let mut residuals = Vec::with_capacity(values.len());
        for (idx, fitted) in fitted_adjusted.into_iter().enumerate() {
            let reseasonalized = reseasonalize_value(fitted, idx, seasonality, pattern.as_deref())?;
            fitted_values.push(reseasonalized);
            residuals.push(values[idx] - reseasonalized);
        }
        Ok(Self {
            last_timestamp: history.last().expect("history length checked").timestamp,
            n_obs: history.len(),
            component,
            seasonal_pattern: pattern,
            fitted_values,
            residuals,
        })
    }

    fn forecast_values(
        &self,
        horizon: usize,
        seasonality: Option<ThetaSeasonality>,
    ) -> Result<Vec<f64>> {
        (1..=horizon)
            .map(|step| {
                let adjusted = forecast_theta_component(&self.component, step);
                let fitted_seasonality = self.seasonal_pattern.as_ref().and(seasonality);
                reseasonalize_value(
                    adjusted,
                    self.n_obs + step - 1,
                    fitted_seasonality,
                    self.seasonal_pattern.as_deref(),
                )
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .expect("theta grid scores are finite")
    }
}

