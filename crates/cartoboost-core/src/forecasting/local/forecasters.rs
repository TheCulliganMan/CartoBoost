impl Forecaster for NaiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let observed = frame.observed_target_frame(self.model_name())?;
        self.fitted = Some(FittedLocalState::from_frame_with_anchor(&observed, frame));
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let anchor_timestamp = fitted
                    .anchor_timestamp_by_series
                    .get(series_id)
                    .copied()
                    .unwrap_or(last.timestamp);
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(anchor_timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean: last.target,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "naive"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name()})
    }
}

impl Forecaster for SeasonalNaiveForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let fitted = FittedLocalState::from_frame(frame);
        validate_local_history_requirement(
            &fitted,
            self.season_length,
            self.model_name(),
            "one complete requested season",
        )?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let base = history.len() - self.season_length;
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let seasonal_index = base + ((step - 1) % self.season_length);
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean: history[seasonal_index].target,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "seasonal_naive"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name(), "season_length": self.season_length})
    }
}

impl Forecaster for WindowAverageForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let observed = frame.observed_target_frame(self.model_name())?;
        let fitted = FittedLocalState::from_frame_with_anchor(&observed, frame);
        validate_local_history_requirement(
            &fitted,
            self.window_size,
            self.model_name(),
            "the requested averaging window",
        )?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let anchor_timestamp = fitted
                    .anchor_timestamp_by_series
                    .get(series_id)
                    .copied()
                    .unwrap_or(last.timestamp);
                let start = history.len() - self.window_size;
                let mean = history[start..].iter().map(|row| row.target).sum::<f64>()
                    / self.window_size as f64;
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(anchor_timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "window_average"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "window_size": self.window_size,
        })
    }
}

impl Forecaster for SeasonalWindowAverageForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let fitted = FittedLocalState::from_frame(frame);
        let required = self
            .season_length
            .checked_mul(self.window_count)
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "seasonal window history requirement overflowed usize".to_string(),
                )
            })?;
        validate_local_history_requirement(
            &fitted,
            required,
            self.model_name(),
            "the requested number of complete seasonal windows",
        )?;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let series_predictions = fitted
            .history_by_series
            .iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(series_id, history)| {
                let last = history.last().ok_or_else(|| {
                    CartoBoostError::InvalidInput("empty series history".to_string())
                })?;
                let model = self.model_name().to_string();
                let mut predictions = Vec::with_capacity(horizon);
                for step in 1..=horizon {
                    let phase_offset = (step - 1) % self.season_length;
                    let mut sum = 0.0;
                    for window in 0..self.window_count {
                        let base = history.len() - self.season_length * (window + 1);
                        sum += history[base + phase_offset].target;
                    }
                    predictions.push(ForecastPrediction {
                        series_id: series_id.clone(),
                        timestamp: fitted.frame.frequency().advance(last.timestamp, step)?,
                        horizon: step,
                        model: model.clone(),
                        mean: sum / self.window_count as f64,
                    });
                }
                Ok(predictions)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut predictions =
            Vec::with_capacity(fitted.history_by_series.len().saturating_mul(horizon));
        for series in series_predictions {
            predictions.extend(series);
        }
        ForecastResult::new(predictions)
    }

    fn model_name(&self) -> &'static str {
        "seasonal_window_average"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "season_length": self.season_length,
            "window_count": self.window_count,
        })
    }
}

impl Forecaster for ThetaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedThetaState::from_frame(
            frame,
            self.theta,
            self.alpha,
            self.seasonality,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "theta"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "theta": self.theta,
            "alpha": self.alpha,
            "seasonality": self.seasonality.map(ThetaSeasonality::name),
            "season_length": self.seasonality.map(|seasonality| seasonality.season_length),
        })
    }
}

impl Forecaster for OptimizedThetaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let history_by_series = FittedLocalState::from_frame(frame).history_by_series;
        let minimum_train_len = match self.seasonality {
            Some(seasonality) => seasonality.season_length.checked_mul(2).ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "optimized_theta seasonal history requirement overflowed usize".to_string(),
                )
            })?,
            None => 2,
        };
        let validation_window = automatic_model_validation_window(
            &history_by_series,
            minimum_train_len,
            self.model_name(),
        )?;
        let candidates = self
            .theta_grid
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(theta_idx, theta)| {
                self.alpha_grid
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(alpha_idx, alpha)| (theta_idx, alpha_idx, theta, alpha))
            })
            .collect::<Vec<_>>();
        let scored = candidates
            .into_par_iter()
            .map(|(theta_idx, alpha_idx, theta, alpha)| {
                let mse = score_theta_params(
                    &history_by_series,
                    theta,
                    alpha,
                    self.seasonality,
                    validation_window,
                )?;
                Ok((
                    theta_idx,
                    alpha_idx,
                    ThetaValidationScore { theta, alpha, mse },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(theta_idx, alpha_idx, _)| (*theta_idx, *alpha_idx));
        let scores = scored
            .into_iter()
            .map(|(_, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores
            .iter()
            .map(|score| {
                (
                    OrderedF64(score.mse),
                    OrderedF64(score.theta),
                    OrderedF64(score.alpha),
                )
            })
            .min();
        let (_, theta, alpha) = best.ok_or_else(|| {
            CartoBoostError::InvalidInput("theta validation grid must not be empty".to_string())
        })?;
        let mut fitted = ThetaForecaster::with_seasonality(theta.0, alpha.0, self.seasonality)?;
        fitted.fit(frame)?;
        self.selected_theta = Some(theta.0);
        self.selected_alpha = Some(alpha.0);
        self.validation_window = Some(validation_window);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "optimized_theta"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "selected_theta": self.selected_theta,
            "selected_alpha": self.selected_alpha,
            "validation_window": self.validation_window,
            "seasonality": self.seasonality.map(ThetaSeasonality::name),
            "season_length": self.seasonality.map(|seasonality| seasonality.season_length),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({"theta": score.theta, "alpha": score.alpha, "mse": score.mse})
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for ETSForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedETSState::from_frame(
            frame,
            self.alpha,
            self.beta,
            self.gamma,
            self.season_length,
            self.damping_phi,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
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
                        let seasonal = series
                            .seasonals
                            .as_ref()
                            .map(|seasonals| seasonals[(series.n_obs + step - 1) % seasonals.len()])
                            .unwrap_or(0.0);
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted
                                .frame
                                .frequency()
                                .advance(series.last_timestamp, step)?,
                            horizon: step,
                            model: self.model_name().to_string(),
                            mean: series.level
                                + damped_trend_multiplier(series.damping_phi, step) * series.trend
                                + seasonal,
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

    fn model_name(&self) -> &'static str {
        "ets"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "season_length": self.season_length,
            "damping_phi": self.damping_phi,
        })
    }
}

impl Forecaster for AutoETSForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let history_by_series = FittedLocalState::from_frame(frame).history_by_series;
        let minimum_train_len = match self.season_length {
            Some(season_length) => season_length.checked_mul(2).ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "auto_ets seasonal history requirement overflowed usize".to_string(),
                )
            })?,
            None => 2,
        };
        let validation_window = automatic_model_validation_window(
            &history_by_series,
            minimum_train_len,
            self.model_name(),
        )?;
        let mut candidates = Vec::new();
        for (alpha_idx, alpha) in self.alpha_grid.iter().copied().enumerate() {
            for (beta_idx, beta) in self.beta_grid.iter().copied().enumerate() {
                for (gamma_idx, gamma) in self.gamma_grid.iter().copied().enumerate() {
                    for (damping_idx, damping_phi) in
                        self.damping_phi_grid.iter().copied().enumerate()
                    {
                        candidates.push((
                            alpha_idx,
                            beta_idx,
                            gamma_idx,
                            damping_idx,
                            alpha,
                            beta,
                            gamma,
                            damping_phi,
                        ));
                    }
                }
            }
        }
        let scored = candidates
            .into_par_iter()
            .map(
                |(alpha_idx, beta_idx, gamma_idx, damping_idx, alpha, beta, gamma, damping_phi)| {
                    let mse = score_ets_params(
                        &history_by_series,
                        alpha,
                        beta,
                        gamma,
                        self.season_length,
                        damping_phi,
                        validation_window,
                    )?;
                    let params = ETSParameterSet {
                        alpha,
                        beta,
                        gamma,
                        damping_phi,
                    };
                    Ok((
                        alpha_idx,
                        beta_idx,
                        gamma_idx,
                        damping_idx,
                        ETSValidationScore { params, mse },
                    ))
                },
            )
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(alpha_idx, beta_idx, gamma_idx, damping_idx, _)| {
            (*alpha_idx, *beta_idx, *gamma_idx, *damping_idx)
        });
        let scores = scored
            .into_iter()
            .map(|(_, _, _, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores.iter().min_by(|left, right| {
            left.mse
                .total_cmp(&right.mse)
                .then_with(|| left.params.alpha.total_cmp(&right.params.alpha))
                .then_with(|| left.params.beta.total_cmp(&right.params.beta))
                .then_with(|| left.params.damping_phi.total_cmp(&right.params.damping_phi))
                .then_with(|| match (left.params.gamma, right.params.gamma) {
                    (Some(left), Some(right)) => left.total_cmp(&right),
                    (None, Some(_)) => std::cmp::Ordering::Less,
                    (Some(_), None) => std::cmp::Ordering::Greater,
                    (None, None) => std::cmp::Ordering::Equal,
                })
        });
        let params = best.map(|score| score.params).ok_or_else(|| {
            CartoBoostError::InvalidInput("auto_ets candidate grid must not be empty".to_string())
        })?;
        let mut fitted = ETSForecaster::with_additive_damped_trend(
            params.alpha,
            params.beta,
            params.gamma,
            self.season_length,
            params.damping_phi,
        )?;
        fitted.fit(frame)?;
        self.selected_params = Some(params);
        self.validation_window = Some(validation_window);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let result = fitted.predict(horizon)?;
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
        "auto_ets"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "season_length": self.season_length,
            "validation_window": self.validation_window,
            "selected_params": self.selected_params.map(|params| {
                json!({
                    "alpha": params.alpha,
                    "beta": params.beta,
                    "gamma": params.gamma,
                    "damping_phi": params.damping_phi,
                })
            }),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "alpha": score.params.alpha,
                    "beta": score.params.beta,
                    "gamma": score.params.gamma,
                    "damping_phi": score.params.damping_phi,
                    "mse": score.mse,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for ArimaForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let fitted = FittedArimaState::from_frame(frame, self.p, self.d, self.q)?;
        let enough_history = fitted
            .series
            .values()
            .map(|series| series.differenced_history.len())
            .min()
            .unwrap_or(0)
            > 2;
        let obviously_explosive = fitted.series.values().any(|series| {
            series
                .ar_coefficients
                .iter()
                .any(|coefficient| coefficient.abs() > 1.9)
        });
        if enough_history && obviously_explosive && !fitted.has_stable_ar_recursions() {
            return Err(CartoBoostError::InvalidInput(format!(
                "fitted ARIMA({},{},{}) has a non-stationary AR polynomial",
                self.p, self.d, self.q
            )));
        }
        if self.d == 0
            && enough_history
            && obviously_explosive
            && !fitted.has_invertible_ma_recursions()
        {
            return Err(CartoBoostError::InvalidInput(format!(
                "fitted ARIMA({},{},{}) has a non-invertible MA polynomial",
                self.p, self.d, self.q
            )));
        }
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "arima"
    }

    fn metadata(&self) -> Value {
        json!({"model": self.model_name(), "p": self.p, "d": self.d, "q": self.q})
    }
}

impl Forecaster for AutoARIMAForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let max_p = self.max_p;
        let max_d = self.max_d;
        let max_q = self.max_q;
        let min_history_len = FittedLocalState::from_frame(frame)
            .history_by_series
            .values()
            .map(Vec::len)
            .min()
            .unwrap_or(0);
        if min_history_len < 2 {
            return Err(CartoBoostError::InvalidInput(
                "auto_arima requires at least two rows per series for a real holdout".to_string(),
            ));
        }
        let validation_window = (min_history_len / 5).clamp(1, 8);
        let minimum_train_len = min_history_len - validation_window;
        let history_by_series = FittedLocalState::from_frame(frame).history_by_series;
        let mut candidate_orders = BTreeSet::new();
        for d in 0..=max_d {
            for p in 0..=max_p {
                for q in 0..=max_q {
                    candidate_orders.insert(arima_order_supported_by_history(
                        minimum_train_len,
                        p,
                        d,
                        q,
                    ));
                }
            }
        }
        let mut scores = candidate_orders
            .into_par_iter()
            .map(|(p, d, q)| {
                let (mse, validation_ar_stable, validation_ma_invertible) =
                    score_arima_order(&history_by_series, p, d, q, validation_window)?;
                let fitted = FittedArimaState::from_frame(frame, p, d, q)?;
                let ar_stable = validation_ar_stable && fitted.has_stable_ar_recursions();
                let ma_invertible =
                    validation_ma_invertible && fitted.has_invertible_ma_recursions();
                Ok(ArimaValidationScore {
                    p,
                    d,
                    q,
                    mse,
                    ar_stable,
                    ma_invertible,
                    stable: ar_stable && ma_invertible,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        scores.sort_by_key(|score| (score.d, score.p, score.q));
        let mut ranked_stable = scores
            .iter()
            .filter(|score| score.stable)
            .map(|score| (OrderedF64(score.mse), score.p, score.d, score.q))
            .collect::<Vec<_>>();
        ranked_stable.sort();
        let mut selected = None;
        let mut fitted = None;
        for (_, p, d, q) in ranked_stable {
            let mut candidate = ArimaForecaster::new(p, d, q)?;
            // A candidate can pass validation on the truncated training window
            // yet become unstable when refit on the complete frame. Skip that
            // candidate and keep the next stable validation result.
            if candidate.fit(frame).is_ok() {
                selected = Some((p, d, q));
                fitted = Some(candidate);
                break;
            }
        }
        let selected = selected.ok_or_else(|| {
            CartoBoostError::InvalidInput("auto_arima candidate grid must not be empty".to_string())
        })?;
        let fitted = fitted.expect("selected auto_arima candidate has a fitted model");
        self.selected_order = Some(selected);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_arima"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "max_p": self.max_p,
            "max_d": self.max_d,
            "max_q": self.max_q,
            "selected_order": self.selected_order.map(|(p, d, q)| json!({"p": p, "d": d, "q": q})),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "p": score.p,
                    "d": score.d,
                    "q": score.q,
                    "mse": score.mse,
                    "ar_stable": score.ar_stable,
                    "ma_invertible": score.ma_invertible,
                    "stable": score.stable,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for KalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedKalmanState::from_frame(
            frame,
            self.level_process_variance,
            self.trend_process_variance,
            self.observation_variance,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "level_process_variance": self.level_process_variance,
            "trend_process_variance": self.trend_process_variance,
            "observation_variance": self.observation_variance,
        })
    }
}

impl Forecaster for LocalLevelKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        self.fitted = Some(FittedLocalLevelKalmanState::from_frame(
            frame,
            self.level_process_variance,
            self.observation_variance,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        self.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "local_level_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "level_process_variance": self.level_process_variance,
            "observation_variance": self.observation_variance,
        })
    }
}

impl Forecaster for AutoKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let local = FittedLocalState::from_frame(frame);
        let level_grid = self.level_process_variance_grid.clone();
        let trend_grid = self.trend_process_variance_grid.clone();
        let observation_grid = self.observation_variance_grid.clone();
        let validation_window = self.validation_window;
        let mut candidates = Vec::new();
        for (level_idx, level_process_variance) in level_grid.iter().copied().enumerate() {
            for (trend_idx, trend_process_variance) in trend_grid.iter().copied().enumerate() {
                for (observation_idx, observation_variance) in
                    observation_grid.iter().copied().enumerate()
                {
                    candidates.push((
                        level_idx,
                        trend_idx,
                        observation_idx,
                        KalmanParameterSet {
                            level_process_variance,
                            trend_process_variance,
                            observation_variance,
                        },
                    ));
                }
            }
        }
        let scored = candidates
            .into_par_iter()
            .map(|(level_idx, trend_idx, observation_idx, params)| {
                let (mse, negative_log_likelihood) =
                    score_kalman_params(&local.history_by_series, params, validation_window)?;
                Ok((
                    level_idx,
                    trend_idx,
                    observation_idx,
                    KalmanValidationScore {
                        params,
                        mse,
                        negative_log_likelihood,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(level_idx, trend_idx, observation_idx, _)| {
            (*level_idx, *trend_idx, *observation_idx)
        });
        let scores = scored
            .into_iter()
            .map(|(_, _, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores.iter().min_by_key(|score| {
            (
                OrderedF64(score.negative_log_likelihood),
                OrderedF64(score.mse),
                OrderedF64(score.params.level_process_variance),
                OrderedF64(score.params.trend_process_variance),
                OrderedF64(score.params.observation_variance),
            )
        });
        let params = best
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "auto_kalman candidate grid must not be empty".to_string(),
                )
            })?
            .params;
        let mut fitted = KalmanForecaster::new(
            params.level_process_variance,
            params.trend_process_variance,
            params.observation_variance,
        )?;
        fitted.fit(frame)?;
        self.selected_params = Some(params);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "validation_window": self.validation_window,
            "selected_params": self.selected_params.map(|params| json!({
                "level_process_variance": params.level_process_variance,
                "trend_process_variance": params.trend_process_variance,
                "observation_variance": params.observation_variance,
            })),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "level_process_variance": score.params.level_process_variance,
                    "trend_process_variance": score.params.trend_process_variance,
                    "observation_variance": score.params.observation_variance,
                    "mse": score.mse,
                    "negative_log_likelihood": score.negative_log_likelihood,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for AutoLocalLevelKalmanForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_regular_for_model(self.model_name())?;
        let local = FittedLocalState::from_frame(frame);
        let candidates = self
            .level_process_variance_grid
            .iter()
            .copied()
            .enumerate()
            .flat_map(|(level_idx, level_process_variance)| {
                self.observation_variance_grid
                    .iter()
                    .copied()
                    .enumerate()
                    .map(move |(observation_idx, observation_variance)| {
                        (
                            level_idx,
                            observation_idx,
                            LocalLevelKalmanParameterSet {
                                level_process_variance,
                                observation_variance,
                            },
                        )
                    })
            })
            .collect::<Vec<_>>();
        let scored = candidates
            .into_par_iter()
            .map(|(level_idx, observation_idx, params)| {
                let (mse, negative_log_likelihood) = score_local_level_kalman_params(
                    &local.history_by_series,
                    params,
                    self.validation_window,
                )?;
                Ok((
                    level_idx,
                    observation_idx,
                    LocalLevelKalmanValidationScore {
                        params,
                        mse,
                        negative_log_likelihood,
                    },
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut scored = scored;
        scored.sort_by_key(|(level_idx, observation_idx, _)| (*level_idx, *observation_idx));
        let scores = scored
            .into_iter()
            .map(|(_, _, score)| score)
            .collect::<Vec<_>>();
        let best = scores.iter().min_by_key(|score| {
            (
                OrderedF64(score.negative_log_likelihood),
                OrderedF64(score.mse),
                OrderedF64(score.params.level_process_variance),
                OrderedF64(score.params.observation_variance),
            )
        });
        let params = best
            .ok_or_else(|| {
                CartoBoostError::InvalidInput(
                    "auto_local_level_kalman candidate grid must not be empty".to_string(),
                )
            })?
            .params;
        let mut fitted = LocalLevelKalmanForecaster::new(
            params.level_process_variance,
            params.observation_variance,
        )?;
        fitted.fit(frame)?;
        self.selected_params = Some(params);
        self.validation_scores = scores;
        self.fitted = Some(fitted);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        fitted.predict_with_model_name(horizon, self.model_name())
    }

    fn model_name(&self) -> &'static str {
        "auto_local_level_kalman"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "validation_window": self.validation_window,
            "selected_params": self.selected_params.map(|params| json!({
                "level_process_variance": params.level_process_variance,
                "observation_variance": params.observation_variance,
            })),
            "validation_scores": self.validation_scores.iter().map(|score| {
                json!({
                    "level_process_variance": score.params.level_process_variance,
                    "observation_variance": score.params.observation_variance,
                    "mse": score.mse,
                    "negative_log_likelihood": score.negative_log_likelihood,
                })
            }).collect::<Vec<_>>(),
        })
    }
}

impl Forecaster for KrigingForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_observed_targets_for_model(self.model_name())?;
        validate_common_spatial_cutoff(frame, self.model_name())?;
        self.fitted = Some(FittedKrigingState::from_frame(frame, &self.coordinates)?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let observations = fitted
            .levels
            .iter()
            .map(|(series_id, value)| {
                let coord = self.coordinates.get(series_id).ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing kriging coordinate for series {series_id}"
                    ))
                })?;
                Ok(KrigingObservation {
                    x: coord.0,
                    y: coord.1,
                    value: *value,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let series_ids = fitted.levels.keys().cloned().collect::<Vec<_>>();
        let means = ordinary_kriging_leave_one_out_with_backend(
            &observations,
            self.config,
            Some(&self.backend.selected),
        )?
            .into_iter()
            .map(|prediction| prediction.mean)
            .collect::<Vec<_>>();
        let predictions = series_ids
            .into_par_iter()
            .zip(means)
            .map(|(series_id, mean)| {
                let history = fitted.frame.rows_for_series(&series_id);
                let last_timestamp = history
                    .last()
                    .ok_or_else(|| {
                        CartoBoostError::InvalidInput("empty series history".to_string())
                    })?
                    .timestamp;
                (1..=horizon)
                    .map(|step| {
                        Ok(ForecastPrediction {
                            series_id: series_id.clone(),
                            timestamp: fitted.frame.frequency().advance(last_timestamp, step)?,
                            horizon: step,
                            model: self.model_name().to_string(),
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

    fn model_name(&self) -> &'static str {
        "kriging"
    }

    fn metadata(&self) -> Value {
        json!({
            "model": self.model_name(),
            "range": self.config.range,
            "nugget": self.config.nugget,
            "sill": self.config.sill,
            "variogram_model": format!("{:?}", self.config.variogram_model).to_lowercase(),
            "drift": format!("{:?}", self.config.drift).to_lowercase(),
            "anisotropy_angle_degrees": self.config.anisotropy_angle_degrees,
            "anisotropy_scaling": self.config.anisotropy_scaling,
            "max_neighbors": self.config.max_neighbors,
            "min_neighbors": self.config.min_neighbors,
            "max_distance": self.config.max_distance,
            "series_count": self.coordinates.len(),
            "target_policy": "leave_one_series_out",
            "backend": self.backend,
        })
    }
}

impl Forecaster for SpatialPiecewiseKrigingForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        frame.require_observed_targets_for_model(self.model_name())?;
        validate_spatial_piecewise_frame(frame, &self.config)?;
        self.fitted = Some(FittedSpatialPiecewiseKrigingState::from_frame(
            frame,
            &self.config,
            &self.backend,
        )?);
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        validate_horizon(horizon)?;
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let mut base = fitted.base.clone();
        if self.config.uses_kriged_regressors() {
            let future_regressors = fitted.future_spatial_regressors(horizon, &self.config)?;
            base.update_config(|config| {
                for (series_id, regressors) in future_regressors {
                    config
                        .future_regressors_by_series
                        .entry(series_id)
                        .or_default()
                        .extend(regressors);
                }
            })?;
        }
        let base_result = base.predict(horizon)?;
        let corrections = if self.config.uses_residual_kriging() {
            fitted.residual_kriging_predictions(&self.config, &self.backend)?
        } else {
            BTreeMap::new()
        };
        let component_records = base
            .predict_components_json_value(horizon)
            .ok()
            .and_then(|value| value.get("records").and_then(Value::as_array).cloned())
            .unwrap_or_default()
            .into_iter()
            .map(|record| {
                let key = component_record_key(&record);
                (key, record)
            })
            .collect::<BTreeMap<_, _>>();
        let mut predictions = Vec::with_capacity(base_result.predictions().len());
        let mut details = Vec::with_capacity(base_result.predictions().len());
        for base_prediction in base_result.predictions() {
            let correction = corrections.get(&base_prediction.series_id);
            let shrinkage = self
                .config
                .residual_shrinkage
                .powi((base_prediction.horizon - 1) as i32);
            let spatial_correction =
                correction.map(|correction| correction.prediction.mean * shrinkage);
            let unbounded_mean = base_prediction.mean + spatial_correction.unwrap_or(0.0);
            let bounds = piecewise_bounds(
                Some(&base_prediction.series_id),
                None,
                Some(base_prediction.horizon),
                &base.config,
            )?;
            let final_mean = if base.config.growth == PiecewiseLinearGrowth::Logistic {
                unbounded_mean
                    .max(bounds.floor)
                    .min(bounds.cap.expect("validated logistic cap"))
            } else {
                unbounded_mean
            };
            let spatial_correction = correction.map(|_| final_mean - base_prediction.mean);
            predictions.push(ForecastPrediction {
                series_id: base_prediction.series_id.clone(),
                timestamp: base_prediction.timestamp,
                horizon: base_prediction.horizon,
                model: self.model_name().to_string(),
                mean: final_mean,
            });
            let selected_neighbors = correction
                .map(|prediction| {
                    prediction
                        .prediction
                        .neighbor_indices
                        .iter()
                        .filter_map(|idx| fitted.residual_observation_series.get(*idx).cloned())
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            let neighbor_count = selected_neighbors.len();
            let kriging_variance = correction.map(|prediction| prediction.prediction.variance);
            let correction_magnitude = spatial_correction.map(f64::abs);
            let detail_key = prediction_lookup_key(
                &base_prediction.series_id,
                base_prediction.timestamp,
                base_prediction.horizon,
            );
            details.push(ForecastPredictionDetail {
                series_id: base_prediction.series_id.clone(),
                timestamp: base_prediction.timestamp,
                horizon: base_prediction.horizon,
                model: self.model_name().to_string(),
                base_mean: Some(base_prediction.mean),
                spatial_correction,
                kriging_variance,
                selected_neighbors,
                component_decomposition: component_records.get(&detail_key).cloned(),
                metadata: Some(json!({
                    "mode": self.config.mode.name(),
                    "residual_shrinkage": self.config.residual_shrinkage,
                    "neighbor_fallback": correction.is_some_and(|prediction| prediction.used_neighbor_fallback),
                    "neighbor_count": neighbor_count,
                    "correction_magnitude": correction_magnitude,
                    "kriging_variance": kriging_variance,
                    "fit_runtime_seconds": fitted.fit_metadata.get("runtime_seconds").cloned(),
                    "cutoff": fitted.cutoff_timestamps.get(&base_prediction.series_id).map(|timestamp| {
                        timestamp.format("%Y-%m-%dT%H:%M:%S").to_string()
                    }),
                })),
            });
        }
        let base_means = base_result
            .predictions()
            .iter()
            .map(|prediction| {
                (
                    prediction_lookup_key(
                        &prediction.series_id,
                        prediction.timestamp,
                        prediction.horizon,
                    ),
                    prediction.mean,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let intervals = base_result
            .intervals()
            .iter()
            .map(|interval| {
                let key = prediction_lookup_key(
                    &interval.series_id,
                    interval.timestamp,
                    interval.horizon,
                );
                let base_mean = base_means.get(&key).copied().ok_or_else(|| {
                    CartoBoostError::InvalidInput(format!(
                        "missing base prediction for spatial interval {key}"
                    ))
                })?;
                if interval.lower > base_mean || interval.upper < base_mean {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "base interval for {key} does not contain its prediction mean"
                    )));
                }
                let correction = corrections.get(&interval.series_id);
                let shrinkage = self
                    .config
                    .residual_shrinkage
                    .powi((interval.horizon - 1) as i32);
                let raw_correction = correction
                    .map(|correction| correction.prediction.mean * shrinkage)
                    .unwrap_or(0.0);
                let bounds = piecewise_bounds(
                    Some(&interval.series_id),
                    None,
                    Some(interval.horizon),
                    &base.config,
                )?;
                let corrected_mean = if base.config.growth == PiecewiseLinearGrowth::Logistic {
                    (base_mean + raw_correction)
                        .max(bounds.floor)
                        .min(bounds.cap.expect("validated logistic cap"))
                } else {
                    base_mean + raw_correction
                };
                let kriging_width = correction
                    .map(|correction| {
                        inverse_standard_normal_cdf((1.0 + interval.level) / 2.0)
                            * correction.prediction.variance.sqrt()
                            * shrinkage
                    })
                    .unwrap_or(0.0);
                let lower_width = (base_mean - interval.lower).hypot(kriging_width);
                let upper_width = (interval.upper - base_mean).hypot(kriging_width);
                let (lower, upper) = clamp_piecewise_interval_bounds(
                    corrected_mean - lower_width,
                    corrected_mean + upper_width,
                    bounds,
                    &base.config,
                );
                Ok(ForecastIntervalPrediction {
                    series_id: interval.series_id.clone(),
                    timestamp: interval.timestamp,
                    horizon: interval.horizon,
                    model: self.model_name().to_string(),
                    level: interval.level,
                    lower,
                    upper,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ForecastResult::new_with_intervals_and_details(predictions, intervals, details)
    }

    fn model_name(&self) -> &'static str {
        "spatial_piecewise_kriging"
    }

    fn metadata(&self) -> Value {
        let fit_metadata = self
            .fitted
            .as_ref()
            .map(|fitted| fitted.fit_metadata.clone())
            .unwrap_or_else(|| json!({}));
        json!({
            "model": self.model_name(),
            "mode": self.config.mode.name(),
            "piecewise": self.config.piecewise_config_metadata(),
            "variogram": kriging_config_metadata(self.config.kriging_config),
            "spatial_regressors": self.config.spatial_regressors,
            "series_count": self.config.coordinates.len(),
            "residual_shrinkage": self.config.residual_shrinkage,
            "allow_neighbor_fallback": self.config.allow_neighbor_fallback,
            "backend": self.backend,
            "fit": fit_metadata,
        })
    }
}

impl Forecaster for PiecewiseLinearSeasonalForecaster {
    fn fit(&mut self, frame: &ForecastFrame) -> Result<()> {
        let observed = frame.observed_target_frame(self.model_name())?;
        let resolved_config = resolve_piecewise_auto_seasonalities(&observed, &self.config);
        self.fitted = Some(FittedPiecewiseLinearSeasonalState::from_frame_with_anchor(
            &observed,
            frame,
            resolved_config.clone(),
        )?);
        self.config = resolved_config;
        Ok(())
    }

    fn predict(&self, horizon: usize) -> Result<ForecastResult> {
        let fitted = self.fitted.as_ref().ok_or_else(not_fitted)?;
        let schedule = self.horizon_schedule(fitted, horizon)?;
        self.predict_with_schedule(fitted, &schedule)
    }

    fn model_name(&self) -> &'static str {
        "piecewise_linear_seasonal"
    }

    fn metadata(&self) -> Value {
        let residual_rmse = self
            .fitted
            .as_ref()
            .map(FittedPiecewiseLinearSeasonalState::root_mean_squared_residual);
        let mut metadata = json!({
            "model": self.model_name(),
            "growth": self.config.growth.name(),
            "component_mode": self.config.component_mode.name(),
            "changepoints": self.config.changepoints,
            "changepoint_range": self.config.changepoint_range,
            "changepoint_timestamps": self.config.changepoint_timestamps.iter().map(|timestamp| {
                timestamp.format("%Y-%m-%dT%H:%M:%S").to_string()
            }).collect::<Vec<_>>(),
            "yearly_fourier_order": self.config.yearly_fourier_order,
            "weekly_fourier_order": self.config.weekly_fourier_order,
            "daily_fourier_order": self.config.daily_fourier_order,
            "auto_yearly_seasonality": self.config.auto_yearly_seasonality,
            "auto_weekly_seasonality": self.config.auto_weekly_seasonality,
            "auto_daily_seasonality": self.config.auto_daily_seasonality,
            "custom_seasonalities": self.config.custom_seasonalities.iter().map(|seasonality| {
                json!({
                    "name": seasonality.name,
                    "period_days": seasonality.period_days,
                    "fourier_order": seasonality.fourier_order,
                    "mode": seasonality.mode.map(PiecewiseLinearComponentMode::name),
                    "condition_name": seasonality.condition_name,
                    "l2_regularization": seasonality.l2_regularization,
                })
            }).collect::<Vec<_>>(),
            "changepoint_l2_regularization": self.config.changepoint_l2_regularization,
            "changepoint_l1_regularization": self.config.changepoint_l1_regularization,
            "seasonality_l2_regularization": self.config.seasonality_l2_regularization,
            "yearly_l2_regularization": self.config.yearly_l2_regularization,
            "weekly_l2_regularization": self.config.weekly_l2_regularization,
            "daily_l2_regularization": self.config.daily_l2_regularization,
            "event_l2_regularization": self.config.event_l2_regularization,
            "regressor_l2_regularization": self.config.regressor_l2_regularization,
            "event_l2_regularization_by_name": self.config.event_l2_regularization_by_name,
            "regressor_l2_regularization_by_name": self.config.regressor_l2_regularization_by_name,
            "events": self.config.events.iter().map(|event| {
                json!({
                    "name": event.name,
                    "timestamp": event.timestamp.format("%Y-%m-%dT%H:%M:%S").to_string(),
                    "lower_window": event.lower_window,
                    "upper_window": event.upper_window,
                })
            }).collect::<Vec<_>>(),
            "event_mode": self.config.event_mode.map(PiecewiseLinearComponentMode::name),
            "extra_regressors": self.config.extra_regressors,
            "regressor_modes": self.config.regressor_modes.iter().map(|(name, mode)| {
                (name.clone(), mode.name())
            }).collect::<BTreeMap<_, _>>(),
            "extra_regressor_monotonic_constraints": self.config.extra_regressor_monotonic_constraints,
            "regressor_standardization": self.config.regressor_standardization.name(),
            "future_regressors": self.config.future_regressors,
            "interval_levels": self.config.interval_levels,
            "quantile_levels": self.config.quantile_levels,
            "uncertainty_samples": self.config.uncertainty_samples,
            "trend_uncertainty_policy": self.config.trend_uncertainty_policy.name(),
            "trend_uncertainty_scale": self.config.trend_uncertainty_scale,
            "uncertainty_seed": self.config.uncertainty_seed,
            "cap": self.config.cap,
            "floor": self.config.floor,
            "cap_regressor": self.config.cap_regressor,
            "floor_regressor": self.config.floor_regressor,
            "residual_rmse": residual_rmse,
        });
        if let Value::Object(values) = &mut metadata {
            values.insert(
                "future_regressors_by_series".to_string(),
                json!(self.config.future_regressors_by_series),
            );
            values.insert(
                "trend_adjustments".to_string(),
                json!(self.config.trend_adjustments),
            );
            values.insert(
                "trend_adjustments_by_series".to_string(),
                json!(self.config.trend_adjustments_by_series),
            );
            values.insert(
                "residual_shock_window".to_string(),
                json!(self.config.residual_shock_window),
            );
            values.insert(
                "residual_shock_scale".to_string(),
                json!(self.config.residual_shock_scale),
            );
            values.insert(
                "residual_shock_decay".to_string(),
                json!(self.config.residual_shock_decay),
            );
            values.insert("fit_loss".to_string(), json!(self.config.fit_loss.name()));
            values.insert("huber_delta".to_string(), json!(self.config.huber_delta));
            values.insert(
                "irls_iterations".to_string(),
                json!(self.config.irls_iterations),
            );
            values.insert(
                "coefficient_uncertainty_scale".to_string(),
                json!(self.config.coefficient_uncertainty_scale),
            );
        }
        metadata
    }
}

