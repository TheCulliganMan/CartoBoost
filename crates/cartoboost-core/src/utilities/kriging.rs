pub fn ordinary_kriging_predict_many(
    observations: &[KrigingObservation],
    targets: &[(f64, f64)],
    config: OrdinaryKrigingConfig,
) -> Result<Vec<KrigingPrediction>> {
    let config = config.validate()?;
    validate_kriging_observations(observations)?;
    if targets.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging targets must not be empty".to_string(),
        ));
    }
    if !uses_local_neighbors(config) {
        return OrdinaryKrigingSystem::new(observations, config)?.predict_many(targets);
    }
    targets
        .par_iter()
        .map(|target| ordinary_kriging_predict_unchecked(observations, *target, config))
        .collect()
}

pub fn ordinary_kriging_predict(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Result<KrigingPrediction> {
    let config = config.validate()?;
    validate_kriging_observations(observations)?;
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging target coordinates must be finite".to_string(),
        ));
    }
    if !uses_local_neighbors(config) {
        return OrdinaryKrigingSystem::new(observations, config)?.predict(target);
    }
    ordinary_kriging_predict_unchecked(observations, target, config)
}

fn ordinary_kriging_predict_unchecked(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Result<KrigingPrediction> {
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging target coordinates must be finite".to_string(),
        ));
    }
    let selected = select_kriging_neighbors(observations, target, config)?;
    let n = selected.len();
    let drift_terms = drift_term_count(config.drift);
    if n < drift_terms {
        return Err(CartoBoostError::InvalidInput(format!(
            "kriging drift {:?} requires at least {drift_terms} neighbors; got {n}",
            config.drift
        )));
    }
    let selected_observations = selected
        .iter()
        .map(|(_, observation)| **observation)
        .collect::<Vec<_>>();
    let matrix = build_kriging_system_matrix(&selected_observations, config);
    let rhs = build_kriging_rhs(&selected_observations, target, config);
    let factorization = LinearSystemFactorization::factor(matrix).ok_or_else(|| {
        CartoBoostError::InvalidInput(
            "kriging system is singular or numerically ill-conditioned; adjust coordinates, variogram scale, or nugget".to_string(),
        )
    })?;
    let solution = factorization.solve(&rhs).ok_or_else(|| {
        CartoBoostError::InvalidInput("kriging solve produced a non-finite result".to_string())
    })?;
    kriging_prediction_from_solution(
        &selected_observations,
        target,
        config,
        &rhs,
        &solution,
        selected.into_iter().map(|(idx, _)| idx).collect(),
    )
}

pub fn ordinary_kriging_leave_one_out(
    observations: &[KrigingObservation],
    config: OrdinaryKrigingConfig,
) -> Result<Vec<KrigingPrediction>> {
    let config = config.validate()?;
    validate_kriging_observations(observations)?;
    if observations.len() < 2 {
        return Err(CartoBoostError::InvalidInput(
            "kriging leave-one-out requires at least two observations".to_string(),
        ));
    }
    observations
        .par_iter()
        .enumerate()
        .map(|(held_out_idx, held_out)| {
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
            let mut prediction =
                ordinary_kriging_predict_unchecked(&training, (held_out.x, held_out.y), config)?;
            prediction.neighbor_indices = prediction
                .neighbor_indices
                .iter()
                .map(|local_idx| training_rows[*local_idx].0)
                .collect();
            Ok(prediction)
        })
        .collect()
}

pub fn ordinary_kriging_leave_one_out_diagnostics(
    observations: &[KrigingObservation],
    config: OrdinaryKrigingConfig,
) -> Result<(Vec<KrigingPrediction>, KrigingLooDiagnostics)> {
    let predictions = ordinary_kriging_leave_one_out(observations, config)?;
    let diagnostics = kriging_loo_diagnostics(observations, &predictions)?;
    Ok((predictions, diagnostics))
}

pub fn empirical_variogram(
    observations: &[KrigingObservation],
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> Result<Vec<EmpiricalVariogramBin>> {
    validate_kriging_observations(observations)?;
    if observations.len() < 2 {
        return Err(CartoBoostError::InvalidInput(
            "empirical variogram requires at least two observations".to_string(),
        ));
    }
    if bin_count == 0 {
        return Err(CartoBoostError::InvalidInput(
            "variogram bin_count must be positive".to_string(),
        ));
    }
    let distance_config = OrdinaryKrigingConfig::new(1.0, 0.0)?
        .with_anisotropy(anisotropy_angle_degrees, anisotropy_scaling)?;
    let pairs = variogram_pairs(observations, distance_config, max_distance)?;
    if pairs.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "empirical variogram has no coordinate pairs within max_distance".to_string(),
        ));
    }
    let max_lag = max_distance.unwrap_or_else(|| {
        pairs
            .iter()
            .map(|(distance, _)| *distance)
            .fold(0.0, f64::max)
    });
    if max_lag <= 0.0 || !max_lag.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "empirical variogram max lag must be positive".to_string(),
        ));
    }
    let width = max_lag / bin_count as f64;
    let mut counts = vec![0usize; bin_count];
    let mut distance_sums = vec![0.0; bin_count];
    let mut semivariance_sums = vec![0.0; bin_count];
    for (distance, semivariance) in pairs {
        let mut bin = (distance / width).floor() as usize;
        if bin >= bin_count {
            bin = bin_count - 1;
        }
        counts[bin] += 1;
        distance_sums[bin] += distance;
        semivariance_sums[bin] += semivariance;
    }
    Ok((0..bin_count)
        .filter_map(|bin| {
            let pair_count = counts[bin];
            if pair_count == 0 {
                return None;
            }
            let lag_min = bin as f64 * width;
            let lag_max = (bin + 1) as f64 * width;
            Some(EmpiricalVariogramBin {
                lag_min,
                lag_max,
                lag_center: 0.5 * (lag_min + lag_max),
                mean_distance: distance_sums[bin] / pair_count as f64,
                semivariance: semivariance_sums[bin] / pair_count as f64,
                pair_count,
            })
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
pub fn fit_ordinary_kriging_variogram(
    observations: &[KrigingObservation],
    variogram_models: &[KrigingVariogramModel],
    range_candidates: &[f64],
    nugget_candidates: &[f64],
    sill_candidates: &[f64],
    bin_count: usize,
    anisotropy_angle_degrees: f64,
    anisotropy_scaling: f64,
) -> Result<KrigingVariogramFit> {
    let bins = empirical_variogram(
        observations,
        bin_count,
        None,
        anisotropy_angle_degrees,
        anisotropy_scaling,
    )?;
    let models = if variogram_models.is_empty() {
        vec![
            KrigingVariogramModel::Exponential,
            KrigingVariogramModel::Gaussian,
            KrigingVariogramModel::Spherical,
            KrigingVariogramModel::Linear,
        ]
    } else {
        variogram_models.to_vec()
    };
    let ranges = if range_candidates.is_empty() {
        default_variogram_ranges(&bins)
    } else {
        validate_variogram_candidates(range_candidates, "range_candidates")?;
        range_candidates.to_vec()
    };
    let nuggets = if nugget_candidates.is_empty() {
        default_variogram_nuggets(&bins)
    } else {
        validate_non_negative_candidates(nugget_candidates, "nugget_candidates")?;
        nugget_candidates.to_vec()
    };
    let sills = if sill_candidates.is_empty() {
        default_variogram_sills(&bins)
    } else {
        validate_variogram_candidates(sill_candidates, "sill_candidates")?;
        sill_candidates.to_vec()
    };
    let mut candidates =
        Vec::with_capacity(models.len() * ranges.len() * nuggets.len() * sills.len());
    for model in models {
        for &range in &ranges {
            for &nugget in &nuggets {
                for &sill in &sills {
                    candidates.push((model, range, nugget, sill));
                }
            }
        }
    }

    candidates
        .par_iter()
        .enumerate()
        .filter_map(|(index, &(model, range, nugget, sill))| {
            let config = OrdinaryKrigingConfig::new(range, nugget)
                .and_then(|config| config.with_sill(sill))
                .and_then(|config| {
                    config.with_anisotropy(anisotropy_angle_degrees, anisotropy_scaling)
                })
                .ok()?
                .with_variogram_model(model);
            let weighted_sse = variogram_weighted_sse(&bins, config);
            weighted_sse
                .is_finite()
                .then_some((index, config, weighted_sse))
        })
        .reduce_with(|left, right| {
            if right.2 < left.2 || (right.2 == left.2 && right.0 < left.0) {
                right
            } else {
                left
            }
        })
        .map(|(_, config, weighted_sse)| KrigingVariogramFit {
            config,
            bins,
            weighted_sse,
        })
        .ok_or_else(|| {
            CartoBoostError::InvalidInput("variogram fit found no valid candidate".to_string())
        })
}
pub fn validate_positive_finite(value: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must be positive and finite"
        )));
    }
    Ok(())
}

fn validate_kriging_target(target: (f64, f64)) -> Result<()> {
    if !target.0.is_finite() || !target.1.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging target coordinates must be finite".to_string(),
        ));
    }
    Ok(())
}

fn uses_local_neighbors(config: OrdinaryKrigingConfig) -> bool {
    config.max_neighbors.is_some() || config.max_distance.is_some()
}

fn kriging_loo_diagnostics(
    observations: &[KrigingObservation],
    predictions: &[KrigingPrediction],
) -> Result<KrigingLooDiagnostics> {
    if observations.len() != predictions.len() {
        return Err(CartoBoostError::InvalidInput(
            "kriging diagnostics require one prediction per observation".to_string(),
        ));
    }
    let n = observations.len();
    if n == 0 {
        return Err(CartoBoostError::InvalidInput(
            "kriging diagnostics require at least one prediction".to_string(),
        ));
    }
    let mut error_sum = 0.0;
    let mut abs_error_sum = 0.0;
    let mut squared_error_sum = 0.0;
    let mut standardized_sum = 0.0;
    let mut standardized_squared_sum = 0.0;
    let mut standardized_abs_max = 0.0_f64;
    let mut covered_95 = 0usize;
    let mut variance_sum = 0.0;
    for (observation, prediction) in observations.iter().zip(predictions.iter()) {
        let error = observation.value - prediction.mean;
        let variance = prediction.variance.max(f64::EPSILON);
        let standardized = error / variance.sqrt();
        error_sum += error;
        abs_error_sum += error.abs();
        squared_error_sum += error * error;
        standardized_sum += standardized;
        standardized_squared_sum += standardized * standardized;
        standardized_abs_max = standardized_abs_max.max(standardized.abs());
        variance_sum += prediction.variance;
        if standardized.abs() <= 1.959_963_984_540_054 {
            covered_95 += 1;
        }
    }
    let n_f64 = n as f64;
    Ok(KrigingLooDiagnostics {
        observation_count: n,
        mean_error: error_sum / n_f64,
        mae: abs_error_sum / n_f64,
        rmse: (squared_error_sum / n_f64).sqrt(),
        mean_standardized_error: standardized_sum / n_f64,
        rmse_standardized_error: (standardized_squared_sum / n_f64).sqrt(),
        max_abs_standardized_error: standardized_abs_max,
        interval_coverage_95: covered_95 as f64 / n_f64,
        average_variance: variance_sum / n_f64,
    })
}

fn variogram_pairs(
    observations: &[KrigingObservation],
    distance_config: OrdinaryKrigingConfig,
    max_distance: Option<f64>,
) -> Result<Vec<(f64, f64)>> {
    if let Some(max_distance) = max_distance {
        validate_positive_finite(max_distance, "max_distance")?;
    }
    Ok((0..observations.len())
        .into_par_iter()
        .flat_map_iter(|left_idx| {
            ((left_idx + 1)..observations.len()).filter_map(move |right_idx| {
                let left = observations[left_idx];
                let right = observations[right_idx];
                let distance =
                    transformed_distance((left.x, left.y), (right.x, right.y), distance_config);
                if max_distance
                    .map(|max_distance| distance > max_distance)
                    .unwrap_or(false)
                {
                    return None;
                }
                let diff = left.value - right.value;
                Some((distance, 0.5 * diff * diff))
            })
        })
        .collect())
}

fn validate_variogram_candidates(values: &[f64], name: &str) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must not be empty"
        )));
    }
    for value in values {
        validate_positive_finite(*value, name)?;
    }
    Ok(())
}

fn validate_non_negative_candidates(values: &[f64], name: &str) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(format!(
            "{name} must not be empty"
        )));
    }
    for value in values {
        if !value.is_finite() || *value < 0.0 {
            return Err(CartoBoostError::InvalidInput(format!(
                "{name} must contain finite non-negative values"
            )));
        }
    }
    Ok(())
}

fn default_variogram_ranges(bins: &[EmpiricalVariogramBin]) -> Vec<f64> {
    let max_distance = bins
        .iter()
        .map(|bin| bin.mean_distance)
        .fold(0.0, f64::max)
        .max(f64::EPSILON);
    [0.25, 0.5, 1.0, 1.5, 2.0]
        .iter()
        .map(|factor| max_distance * factor)
        .collect()
}

fn default_variogram_nuggets(bins: &[EmpiricalVariogramBin]) -> Vec<f64> {
    let first = bins
        .first()
        .map(|bin| bin.semivariance)
        .unwrap_or(0.0)
        .max(0.0);
    vec![0.0, first * 0.25, first * 0.5]
}

fn default_variogram_sills(bins: &[EmpiricalVariogramBin]) -> Vec<f64> {
    let max_semivariance = bins
        .iter()
        .map(|bin| bin.semivariance)
        .fold(0.0, f64::max)
        .max(f64::EPSILON);
    [0.5, 1.0, 1.5, 2.0]
        .iter()
        .map(|factor| max_semivariance * factor)
        .collect()
}

fn variogram_weighted_sse(bins: &[EmpiricalVariogramBin], config: OrdinaryKrigingConfig) -> f64 {
    bins.par_iter()
        .map(|bin| {
            let fitted = theoretical_semivariogram(bin.mean_distance, config);
            let residual = bin.semivariance - fitted;
            bin.pair_count as f64 * residual * residual
        })
        .sum()
}

fn theoretical_semivariogram(distance: f64, config: OrdinaryKrigingConfig) -> f64 {
    let ratio = distance / config.range;
    match config.variogram_model {
        KrigingVariogramModel::Exponential => config.nugget + config.sill * (1.0 - (-ratio).exp()),
        KrigingVariogramModel::Gaussian => {
            config.nugget + config.sill * (1.0 - (-(ratio * ratio)).exp())
        }
        KrigingVariogramModel::Spherical => {
            let structural = if ratio >= 1.0 {
                1.0
            } else {
                1.5 * ratio - 0.5 * ratio.powi(3)
            };
            config.nugget + config.sill * structural
        }
        // A linear variogram has no bounded covariance counterpart. `sill`
        // is its structural contribution at `range`, so sill/range is slope.
        KrigingVariogramModel::Linear => config.nugget + config.sill * ratio,
    }
}

fn build_kriging_system_matrix(
    observations: &[KrigingObservation],
    config: OrdinaryKrigingConfig,
) -> Vec<Vec<f64>> {
    let n = observations.len();
    let drift_terms = drift_term_count(config.drift);
    let system_size = n + drift_terms;
    (0..system_size)
        .into_par_iter()
        .map(|row_idx| {
            let mut row = vec![0.0; system_size];
            if row_idx < n {
                let left = observations[row_idx];
                for (col_idx, right) in observations.iter().enumerate() {
                    row[col_idx] = if row_idx == col_idx {
                        0.0
                    } else {
                        theoretical_semivariogram(
                            transformed_distance((left.x, left.y), (right.x, right.y), config),
                            config,
                        )
                    };
                }
                let basis = drift_basis((left.x, left.y), config.drift);
                for (basis_idx, basis_value) in basis.iter().enumerate() {
                    row[n + basis_idx] = *basis_value;
                }
            } else {
                let basis_idx = row_idx - n;
                for (col_idx, observation) in observations.iter().enumerate() {
                    row[col_idx] = drift_basis((observation.x, observation.y), config.drift)
                        .get(basis_idx)
                        .copied()
                        .unwrap_or(0.0);
                }
            }
            row
        })
        .collect()
}

fn build_kriging_rhs(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Vec<f64> {
    let drift_terms = drift_term_count(config.drift);
    let mut rhs = observations
        .par_iter()
        .map(|observation| {
            theoretical_semivariogram(
                transformed_distance((observation.x, observation.y), target, config),
                config,
            )
        })
        .collect::<Vec<_>>();
    rhs.extend(
        drift_basis(target, config.drift)
            .into_iter()
            .take(drift_terms),
    );
    rhs
}

fn kriging_prediction_from_solution(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
    rhs: &[f64],
    solution: &[f64],
    neighbor_indices: Vec<usize>,
) -> Result<KrigingPrediction> {
    let n = observations.len();
    let weights = solution.iter().copied().take(n).collect::<Vec<_>>();
    if weights.iter().any(|weight| !weight.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "kriging weights are not finite".to_string(),
        ));
    }
    validate_kriging_drift_constraints(observations, target, config.drift, &weights)?;
    let mean = observations
        .iter()
        .enumerate()
        .map(|(idx, observation)| weights[idx] * observation.value)
        .sum::<f64>();
    if !mean.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "kriging estimate is not finite".to_string(),
        ));
    }
    let raw_variance = solution
        .iter()
        .zip(rhs.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    let variance_scale = solution
        .iter()
        .zip(rhs.iter())
        .map(|(left, right)| (left * right).abs())
        .sum::<f64>()
        .max(config.sill + config.nugget);
    let variance =
        checked_non_negative_variance(raw_variance, variance_scale, "kriging prediction variance")?;
    Ok(KrigingPrediction {
        x: target.0,
        y: target.1,
        mean,
        variance,
        weights,
        neighbor_indices,
    })
}

