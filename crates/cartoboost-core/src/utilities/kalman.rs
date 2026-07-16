pub fn fit_local_level_kalman(
    values: &[f64],
    config: LocalLevelKalmanConfig,
) -> Result<LocalLevelKalmanResult> {
    let config = config.validate()?;
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman filter requires at least one observation".to_string(),
        ));
    }
    validate_numeric_series(values, "kalman observation")?;
    let mut level = values[0];
    let mut variance = config.observation_variance;
    let mut estimates = Vec::with_capacity(values.len().saturating_sub(1));
    let mut filtered_states = Vec::with_capacity(values.len());
    filtered_states.push(LocalLevelKalmanSmoothedState {
        step: 0,
        level,
        variance,
    });
    let mut total_log_likelihood = 0.0;
    for (idx, observed) in values.iter().enumerate().skip(1) {
        let prior_level = level;
        let prior_variance = variance + config.level_process_variance;
        let innovation = observed - prior_level;
        let innovation_variance = prior_variance + config.observation_variance;
        if innovation_variance <= 0.0 || !innovation_variance.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "local level kalman innovation variance is not positive".to_string(),
            ));
        }
        let gain = prior_variance / innovation_variance;
        let log_likelihood = gaussian_log_likelihood(innovation, innovation_variance);
        level = prior_level + gain * innovation;
        variance =
            (1.0 - gain).powi(2) * prior_variance + gain.powi(2) * config.observation_variance;
        total_log_likelihood += log_likelihood;
        estimates.push(LocalLevelKalmanEstimate {
            step: idx,
            observed: *observed,
            prior_level,
            prior_variance,
            level,
            variance,
            innovation,
            innovation_variance,
            gain,
            log_likelihood,
        });
        filtered_states.push(LocalLevelKalmanSmoothedState {
            step: idx,
            level,
            variance,
        });
    }
    let smoothed_states = smooth_local_level_states(&filtered_states, &estimates)?;
    let residual_summary =
        kalman_residual_summary(values.len(), &estimates, total_log_likelihood, 2);
    Ok(LocalLevelKalmanResult {
        final_level: level,
        final_variance: variance,
        estimates,
        smoothed_states,
        residual_summary,
        log_likelihood: total_log_likelihood,
    })
}

pub fn local_level_kalman_forecast(final_level: f64, horizon: usize) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman forecast horizon must be positive".to_string(),
        ));
    }
    if !final_level.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman final level must be finite".to_string(),
        ));
    }
    Ok(vec![final_level; horizon])
}

pub fn fit_local_linear_kalman(
    values: &[f64],
    config: LocalLinearKalmanConfig,
) -> Result<LocalLinearKalmanResult> {
    let config = config.validate()?;
    if values.len() < 2 {
        return Err(CartoBoostError::InvalidInput(
            "local linear kalman filter requires at least two observations".to_string(),
        ));
    }
    validate_numeric_series(values, "kalman observation")?;
    let mut level = values[0];
    // The first observation conditions the initial level. The trend remains an
    // unobserved zero-mean state until later observations update it; deriving
    // it from values[1] and then filtering values[1] double-counts that value.
    let mut trend = 0.0;
    let mut p00 = config.observation_variance;
    let mut p01 = 0.0;
    let mut p10 = 0.0;
    let mut p11 = config.observation_variance;
    let mut estimates = Vec::with_capacity(values.len() - 1);
    let mut filtered_states = Vec::with_capacity(values.len());
    filtered_states.push(LocalLinearKalmanSmoothedState {
        step: 0,
        level,
        trend,
        covariance: [[p00, p01], [p10, p11]],
    });
    let mut total_log_likelihood = 0.0;
    for (idx, observed) in values.iter().enumerate().skip(1) {
        let prior_level = level + trend;
        let prior_trend = trend;
        let pp00 = p00 + p01 + p10 + p11 + config.level_process_variance;
        let pp01 = p01 + p11;
        let pp10 = p10 + p11;
        let pp11 = p11 + config.trend_process_variance;

        let innovation = observed - prior_level;
        let innovation_variance = pp00 + config.observation_variance;
        if innovation_variance <= 0.0 || !innovation_variance.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "kalman innovation variance is not positive".to_string(),
            ));
        }
        let k0 = pp00 / innovation_variance;
        let k1 = pp10 / innovation_variance;
        let log_likelihood = gaussian_log_likelihood(innovation, innovation_variance);
        level = prior_level + k0 * innovation;
        trend = prior_trend + k1 * innovation;
        let posterior_covariance = kalman_joseph_covariance_update(
            [[pp00, pp01], [pp10, pp11]],
            [k0, k1],
            config.observation_variance,
        )?;
        p00 = posterior_covariance[0][0];
        p01 = posterior_covariance[0][1];
        p10 = posterior_covariance[1][0];
        p11 = posterior_covariance[1][1];
        total_log_likelihood += log_likelihood;
        estimates.push(LocalLinearKalmanEstimate {
            step: idx,
            observed: *observed,
            prior_level,
            prior_trend,
            prior_level_variance: pp00,
            prior_trend_variance: pp11,
            prior_covariance: [[pp00, pp01], [pp10, pp11]],
            level,
            trend,
            level_variance: p00,
            trend_variance: p11,
            covariance: [[p00, p01], [p10, p11]],
            innovation,
            innovation_variance,
            level_gain: k0,
            trend_gain: k1,
            log_likelihood,
        });
        filtered_states.push(LocalLinearKalmanSmoothedState {
            step: idx,
            level,
            trend,
            covariance: [[p00, p01], [p10, p11]],
        });
    }
    let smoothed_states = smooth_local_linear_states(&filtered_states, &estimates)?;
    let residual_summary =
        kalman_residual_summary(values.len(), &estimates, total_log_likelihood, 3);
    Ok(LocalLinearKalmanResult {
        final_state: LocalLinearKalmanState { level, trend },
        final_covariance: [[p00, p01], [p10, p11]],
        estimates,
        smoothed_states,
        residual_summary,
        log_likelihood: total_log_likelihood,
    })
}

pub fn intermittent_demand_forecast(
    values: &[f64],
    horizon: usize,
    alpha: f64,
    beta: f64,
    method: IntermittentDemandMethod,
) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand forecast horizon must be positive".to_string(),
        ));
    }
    validate_unit_interval("alpha", alpha)?;
    validate_unit_interval("beta", beta)?;
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "intermittent demand forecast requires at least one observation".to_string(),
        ));
    }
    validate_numeric_series(values, "intermittent demand observation")?;
    for (idx, value) in values.iter().enumerate() {
        if *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "intermittent demand observation at index {idx} must be non-negative"
            )));
        }
    }
    match method {
        IntermittentDemandMethod::Croston | IntermittentDemandMethod::Sba => {
            let estimate = croston_level(values, alpha)?;
            let adjusted = if method == IntermittentDemandMethod::Sba {
                estimate * (1.0 - alpha / 2.0)
            } else {
                estimate
            };
            Ok(vec![adjusted; horizon])
        }
        IntermittentDemandMethod::Tsb => {
            let estimate = tsb_level(values, alpha, beta)?;
            Ok(vec![estimate; horizon])
        }
    }
}

pub fn local_linear_kalman_forecast(
    state: LocalLinearKalmanState,
    horizon: usize,
) -> Result<Vec<f64>> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast horizon must be positive".to_string(),
        ));
    }
    if !state.level.is_finite() || !state.trend.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast state must be finite".to_string(),
        ));
    }
    Ok((1..=horizon)
        .map(|step| state.level + step as f64 * state.trend)
        .collect())
}

pub fn local_level_kalman_forecast_distribution(
    final_level: f64,
    final_variance: f64,
    config: LocalLevelKalmanConfig,
    horizon: usize,
    interval_z: f64,
) -> Result<Vec<KalmanForecastPoint>> {
    let config = config.validate()?;
    validate_kalman_forecast_inputs(horizon, interval_z)?;
    if !final_level.is_finite() || !final_variance.is_finite() || final_variance < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "local level kalman final state must be finite with non-negative variance".to_string(),
        ));
    }
    Ok((1..=horizon)
        .map(|step| {
            let variance = final_variance
                + step as f64 * config.level_process_variance
                + config.observation_variance;
            forecast_point(step, final_level, variance, interval_z)
        })
        .collect())
}

pub fn local_linear_kalman_forecast_distribution(
    state: LocalLinearKalmanState,
    covariance: [[f64; 2]; 2],
    config: LocalLinearKalmanConfig,
    horizon: usize,
    interval_z: f64,
) -> Result<Vec<KalmanForecastPoint>> {
    let config = config.validate()?;
    validate_kalman_forecast_inputs(horizon, interval_z)?;
    if !state.level.is_finite() || !state.trend.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kalman final state must be finite".to_string(),
        ));
    }
    let covariance = validate_covariance_2x2(covariance, "kalman final covariance")?;
    (1..=horizon)
        .map(|step| {
            let h = step as f64;
            let mean = state.level + h * state.trend;
            let trend_noise_multiplier = if step <= 1 {
                0.0
            } else {
                (h - 1.0) * h * (2.0 * h - 1.0) / 6.0
            };
            let state_variance = covariance[0][0]
                + h * (covariance[0][1] + covariance[1][0])
                + h * h * covariance[1][1]
                + h * config.level_process_variance
                + trend_noise_multiplier * config.trend_process_variance;
            let state_variance = checked_non_negative_variance(
                state_variance,
                covariance_scale(covariance)
                    + h * config.level_process_variance
                    + trend_noise_multiplier * config.trend_process_variance,
                "kalman forecast state variance",
            )?;
            let variance = state_variance + config.observation_variance;
            Ok(forecast_point(step, mean, variance, interval_z))
        })
        .collect()
}

fn gaussian_log_likelihood(innovation: f64, innovation_variance: f64) -> f64 {
    -0.5 * ((2.0 * std::f64::consts::PI).ln()
        + innovation_variance.ln()
        + innovation * innovation / innovation_variance)
}

fn kalman_joseph_covariance_update(
    prior: [[f64; 2]; 2],
    gain: [f64; 2],
    observation_variance: f64,
) -> Result<[[f64; 2]; 2]> {
    let update = [[1.0 - gain[0], 0.0], [-gain[1], 1.0]];
    let measurement = [
        [
            gain[0] * observation_variance * gain[0],
            gain[0] * observation_variance * gain[1],
        ],
        [
            gain[1] * observation_variance * gain[0],
            gain[1] * observation_variance * gain[1],
        ],
    ];
    let posterior = mat2_add(
        mat2_mul(mat2_mul(update, prior), mat2_transpose(update)),
        measurement,
    );
    validate_covariance_2x2(posterior, "kalman posterior covariance")
}

fn covariance_scale(matrix: [[f64; 2]; 2]) -> f64 {
    matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
}

fn checked_non_negative_variance(value: f64, scale: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} must be finite"
        )));
    }
    let tolerance = 128.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    if value < -tolerance {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} is negative ({value:e}); covariance inputs are invalid or numerically unstable"
        )));
    }
    Ok(value.max(0.0))
}

fn validate_covariance_2x2(matrix: [[f64; 2]; 2], label: &str) -> Result<[[f64; 2]; 2]> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} must be finite"
        )));
    }
    let scale = covariance_scale(matrix);
    let tolerance = 128.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE);
    if (matrix[0][1] - matrix[1][0]).abs() > tolerance {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} must be symmetric"
        )));
    }
    let p00 = checked_non_negative_variance(matrix[0][0], scale, label)?;
    let p11 = checked_non_negative_variance(matrix[1][1], scale, label)?;
    let mut off_diagonal = 0.5 * (matrix[0][1] + matrix[1][0]);
    let determinant = p00 * p11 - off_diagonal * off_diagonal;
    let determinant_tolerance = 256.0 * f64::EPSILON * scale.max(f64::MIN_POSITIVE).powi(2);
    if determinant < -determinant_tolerance {
        return Err(CartoBoostError::InvalidInput(format!(
            "{label} must be positive semidefinite"
        )));
    }
    if determinant < 0.0 {
        let bound = (p00 * p11).sqrt();
        off_diagonal = off_diagonal.clamp(-bound, bound);
    }
    Ok([[p00, off_diagonal], [off_diagonal, p11]])
}

fn validate_kalman_forecast_inputs(horizon: usize, interval_z: f64) -> Result<()> {
    if horizon == 0 {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast horizon must be positive".to_string(),
        ));
    }
    if !interval_z.is_finite() || interval_z < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "kalman forecast interval_z must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn forecast_point(step: usize, mean: f64, variance: f64, interval_z: f64) -> KalmanForecastPoint {
    debug_assert!(variance >= 0.0 && variance.is_finite());
    let standard_error = variance.sqrt();
    KalmanForecastPoint {
        step,
        mean,
        variance,
        lower: mean - interval_z * standard_error,
        upper: mean + interval_z * standard_error,
    }
}

trait KalmanInnovation {
    fn innovation(&self) -> f64;
    fn innovation_variance(&self) -> f64;
}

impl KalmanInnovation for LocalLevelKalmanEstimate {
    fn innovation(&self) -> f64 {
        self.innovation
    }

    fn innovation_variance(&self) -> f64 {
        self.innovation_variance
    }
}

impl KalmanInnovation for LocalLinearKalmanEstimate {
    fn innovation(&self) -> f64 {
        self.innovation
    }

    fn innovation_variance(&self) -> f64 {
        self.innovation_variance
    }
}

fn kalman_residual_summary<T: KalmanInnovation>(
    observation_count: usize,
    estimates: &[T],
    log_likelihood: f64,
    parameter_count: usize,
) -> KalmanResidualSummary {
    let fitted_count = estimates.len();
    if fitted_count == 0 {
        return KalmanResidualSummary {
            observation_count,
            fitted_count,
            log_likelihood,
            aic: 2.0 * parameter_count as f64 - 2.0 * log_likelihood,
            bic: f64::NAN,
            mse: f64::NAN,
            rmse: f64::NAN,
            mae: f64::NAN,
            mean_standardized_innovation: f64::NAN,
            max_abs_standardized_innovation: f64::NAN,
        };
    }
    let mut sse = 0.0;
    let mut sae = 0.0;
    let mut standardized_sum = 0.0;
    let mut standardized_abs_max = 0.0_f64;
    for estimate in estimates {
        let innovation = estimate.innovation();
        let standardized = innovation / estimate.innovation_variance().sqrt();
        sse += innovation * innovation;
        sae += innovation.abs();
        standardized_sum += standardized;
        standardized_abs_max = standardized_abs_max.max(standardized.abs());
    }
    let n = fitted_count as f64;
    let mse = sse / n;
    KalmanResidualSummary {
        observation_count,
        fitted_count,
        log_likelihood,
        aic: 2.0 * parameter_count as f64 - 2.0 * log_likelihood,
        bic: (parameter_count as f64) * n.ln() - 2.0 * log_likelihood,
        mse,
        rmse: mse.sqrt(),
        mae: sae / n,
        mean_standardized_innovation: standardized_sum / n,
        max_abs_standardized_innovation: standardized_abs_max,
    }
}

fn smooth_local_level_states(
    filtered_states: &[LocalLevelKalmanSmoothedState],
    estimates: &[LocalLevelKalmanEstimate],
) -> Result<Vec<LocalLevelKalmanSmoothedState>> {
    if filtered_states.is_empty() {
        return Ok(Vec::new());
    }
    let mut smoothed = filtered_states.to_vec();
    for idx in (0..filtered_states.len().saturating_sub(1)).rev() {
        let next_estimate = &estimates[idx];
        let filtered = filtered_states[idx];
        let next_smoothed = smoothed[idx + 1];
        if !next_estimate.prior_variance.is_finite() || next_estimate.prior_variance <= 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "local level kalman smoother prior variance must be positive and finite"
                    .to_string(),
            ));
        }
        let smoother_gain = filtered.variance / next_estimate.prior_variance;
        let level =
            filtered.level + smoother_gain * (next_smoothed.level - next_estimate.prior_level);
        let variance = filtered.variance
            + smoother_gain
                * smoother_gain
                * (next_smoothed.variance - next_estimate.prior_variance);
        let variance = checked_non_negative_variance(
            variance,
            filtered
                .variance
                .abs()
                .max(next_estimate.prior_variance.abs())
                .max(next_smoothed.variance.abs()),
            "local level kalman smoothed variance",
        )?;
        smoothed[idx] = LocalLevelKalmanSmoothedState {
            step: filtered.step,
            level,
            variance,
        };
    }
    Ok(smoothed)
}

fn smooth_local_linear_states(
    filtered_states: &[LocalLinearKalmanSmoothedState],
    estimates: &[LocalLinearKalmanEstimate],
) -> Result<Vec<LocalLinearKalmanSmoothedState>> {
    if filtered_states.is_empty() {
        return Ok(Vec::new());
    }
    let mut smoothed = filtered_states.to_vec();
    for idx in (0..filtered_states.len().saturating_sub(1)).rev() {
        let filtered = filtered_states[idx];
        let next_estimate = &estimates[idx];
        let predicted_inverse = invert_2x2(next_estimate.prior_covariance).ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "kalman smoother prior covariance is singular or numerically ill-conditioned"
                    .to_string(),
            )
        })?;
        let gain = mat2_mul(
            mat2_mul(filtered.covariance, [[1.0, 0.0], [1.0, 1.0]]),
            predicted_inverse,
        );
        let predicted_state = [next_estimate.prior_level, next_estimate.prior_trend];
        let next_smoothed = smoothed[idx + 1];
        let state_delta = [
            next_smoothed.level - predicted_state[0],
            next_smoothed.trend - predicted_state[1],
        ];
        let correction = mat2_vec_mul(gain, state_delta);
        let covariance_delta = mat2_sub(next_smoothed.covariance, next_estimate.prior_covariance);
        let covariance = mat2_add(
            filtered.covariance,
            mat2_mul(mat2_mul(gain, covariance_delta), mat2_transpose(gain)),
        );
        smoothed[idx] = LocalLinearKalmanSmoothedState {
            step: filtered.step,
            level: filtered.level + correction[0],
            trend: filtered.trend + correction[1],
            covariance: validate_covariance_2x2(covariance, "kalman smoothed covariance")?,
        };
    }
    Ok(smoothed)
}

fn invert_2x2(matrix: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    if matrix.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    let scale = covariance_scale(matrix);
    if scale == 0.0 {
        return None;
    }
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if !determinant.is_finite() || determinant.abs() <= 128.0 * f64::EPSILON * scale.powi(2) {
        return None;
    }
    Some([
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ])
}

fn mat2_mul(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ],
        [
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ],
    ]
}

fn mat2_vec_mul(matrix: [[f64; 2]; 2], vector: [f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

fn mat2_add(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [left[0][0] + right[0][0], left[0][1] + right[0][1]],
        [left[1][0] + right[1][0], left[1][1] + right[1][1]],
    ]
}

fn mat2_sub(left: [[f64; 2]; 2], right: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [
        [left[0][0] - right[0][0], left[0][1] - right[0][1]],
        [left[1][0] - right[1][0], left[1][1] - right[1][1]],
    ]
}

fn mat2_transpose(matrix: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]
}

