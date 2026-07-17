pub fn covariance(left: [f64; 2], right: [f64; 2], config: NngpConfig) -> f64 {
    let h = transformed_distance(left, right, config);
    let r = h / config.range;
    let corr = match config.kernel {
        CovarianceKernel::Exponential => (-r).exp(),
        CovarianceKernel::SquaredExponential => (-(r * r)).exp(),
        CovarianceKernel::Matern32 => {
            let z = 3.0_f64.sqrt() * r;
            (1.0 + z) * (-z).exp()
        }
        CovarianceKernel::Matern52 => {
            let z = 5.0_f64.sqrt() * r;
            (1.0 + z + z * z / 3.0) * (-z).exp()
        }
    };
    config.sill * corr
}

pub fn empirical_semivariogram(
    coords: &[[f64; 2]],
    values: &[f64],
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy: Anisotropy,
) -> Result<Vec<EmpiricalVariogramBin>> {
    if coords.len() != values.len() {
        return Err(GeostatsError::InvalidInput(
            "coords and values must have the same row count".to_string(),
        ));
    }
    if coords.len() < 2 || bin_count == 0 {
        return Err(GeostatsError::InvalidInput(
            "at least two observations and one bin are required".to_string(),
        ));
    }
    if let Some(max_distance) = max_distance {
        if !max_distance.is_finite() || max_distance <= 0.0 {
            return Err(GeostatsError::InvalidInput(
                "max variogram distance must be finite and positive".to_string(),
            ));
        }
    }
    for (idx, (coord, value)) in coords.iter().zip(values).enumerate() {
        if !coord[0].is_finite() || !coord[1].is_finite() {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram coordinates must be finite at row {idx}"
            )));
        }
        if !value.is_finite() {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram value must be finite at row {idx}"
            )));
        }
    }
    let distance_config = NngpConfig {
        anisotropy,
        ..NngpConfig::default()
    }
    .validate()?;
    let mut pairs = Vec::new();
    let mut observed_max: f64 = 0.0;
    for i in 0..coords.len() {
        for j in (i + 1)..coords.len() {
            let distance = transformed_distance(coords[i], coords[j], distance_config);
            if !distance.is_finite() {
                return Err(GeostatsError::InvalidInput(format!(
                    "variogram distance is not finite for rows {i} and {j}"
                )));
            }
            if max_distance.is_some_and(|max| distance > max) {
                continue;
            }
            let difference = values[i] - values[j];
            let semivariance = 0.5 * difference * difference;
            if !semivariance.is_finite() {
                return Err(GeostatsError::InvalidInput(format!(
                    "variogram semivariance is not finite for rows {i} and {j}"
                )));
            }
            observed_max = observed_max.max(distance);
            pairs.push((distance, semivariance));
        }
    }
    if pairs.is_empty() {
        return Err(GeostatsError::InvalidInput(
            "no coordinate pairs are available for variogram bins".to_string(),
        ));
    }
    let upper = max_distance.unwrap_or(observed_max);
    if upper <= 0.0 || !upper.is_finite() {
        return Err(GeostatsError::InvalidInput(
            "max variogram distance must be positive".to_string(),
        ));
    }
    let width = upper / bin_count as f64;
    if !width.is_finite() || width <= 0.0 {
        return Err(GeostatsError::InvalidInput(
            "variogram bin width must be finite and positive".to_string(),
        ));
    }
    let mut sums = vec![0.0; bin_count];
    let mut counts = vec![0usize; bin_count];
    for (distance, gamma) in pairs {
        let mut bin = (distance / width).floor() as usize;
        if bin >= bin_count {
            bin = bin_count - 1;
        }
        sums[bin] += gamma;
        counts[bin] += 1;
    }
    Ok((0..bin_count)
        .filter(|&bin| counts[bin] > 0)
        .map(|bin| {
            let lag_start = bin as f64 * width;
            let lag_end = lag_start + width;
            EmpiricalVariogramBin {
                lag_start,
                lag_end,
                lag_center: 0.5 * (lag_start + lag_end),
                semivariance: sums[bin] / counts[bin] as f64,
                pair_count: counts[bin],
            }
        })
        .collect())
}

pub fn binned_variogram(
    coords: &[[f64; 2]],
    values: &[f64],
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy: Anisotropy,
) -> Result<Vec<EmpiricalVariogramBin>> {
    empirical_semivariogram(coords, values, bin_count, max_distance, anisotropy)
}

pub fn fit_variogram_wls(
    bins: &[EmpiricalVariogramBin],
    kernels: &[CovarianceKernel],
    range_candidates: &[f64],
    sill_candidates: &[f64],
    nugget_candidates: &[f64],
) -> Result<VariogramFit> {
    if bins.is_empty()
        || range_candidates.is_empty()
        || sill_candidates.is_empty()
        || nugget_candidates.is_empty()
    {
        return Err(GeostatsError::InvalidInput(
            "variogram fitting requires bins and nonempty candidate grids".to_string(),
        ));
    }
    validate_variogram_bins(bins)?;
    validate_positive_candidates(range_candidates, "range candidates")?;
    validate_positive_candidates(sill_candidates, "sill candidates")?;
    validate_nonnegative_candidates(nugget_candidates, "nugget candidates")?;
    let kernels = if kernels.is_empty() {
        vec![CovarianceKernel::Exponential]
    } else {
        kernels.to_vec()
    };
    let mut best: Option<VariogramFit> = None;
    for &kernel in &kernels {
        for &range in range_candidates {
            for &sill in sill_candidates {
                for &nugget in nugget_candidates {
                    let config = NngpConfig {
                        kernel,
                        range,
                        sill,
                        nugget,
                        ..NngpConfig::default()
                    }
                    .validate()?;
                    let mut weighted_sse = 0.0;
                    for bin in bins {
                        let model = nugget
                            + sill
                                * (1.0
                                    - covariance(
                                        [0.0, 0.0],
                                        [bin.lag_center, 0.0],
                                        NngpConfig {
                                            nugget: 0.0,
                                            ..config
                                        },
                                    ) / sill);
                        if !model.is_finite() || model < 0.0 {
                            return Err(GeostatsError::InvalidInput(
                                "variogram candidate produced an invalid semivariance".to_string(),
                            ));
                        }
                        let residual = bin.semivariance - model;
                        let contribution = bin.pair_count as f64 * residual * residual;
                        if !contribution.is_finite() || contribution < 0.0 {
                            return Err(GeostatsError::InvalidInput(
                                "variogram candidate produced a non-finite weighted error"
                                    .to_string(),
                            ));
                        }
                        weighted_sse += contribution;
                        if !weighted_sse.is_finite() {
                            return Err(GeostatsError::InvalidInput(
                                "variogram weighted SSE is not finite".to_string(),
                            ));
                        }
                    }
                    let candidate = VariogramFit {
                        kernel,
                        range,
                        sill,
                        nugget,
                        weighted_sse,
                    };
                    if best
                        .as_ref()
                        .is_none_or(|current| candidate.weighted_sse < current.weighted_sse)
                    {
                        best = Some(candidate);
                    }
                }
            }
        }
    }
    best.ok_or_else(|| {
        GeostatsError::InvalidInput("no valid variogram candidates were supplied".to_string())
    })
}

fn validate_variogram_bins(bins: &[EmpiricalVariogramBin]) -> Result<()> {
    for (idx, bin) in bins.iter().enumerate() {
        if !bin.lag_start.is_finite() || bin.lag_start < 0.0 {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} lag_start must be finite and nonnegative"
            )));
        }
        if !bin.lag_end.is_finite() || bin.lag_end <= bin.lag_start {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} lag_end must be finite and greater than lag_start"
            )));
        }
        if !bin.lag_center.is_finite()
            || bin.lag_center < bin.lag_start
            || bin.lag_center > bin.lag_end
        {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} lag_center must be finite and within its lag bounds"
            )));
        }
        if !bin.semivariance.is_finite() || bin.semivariance < 0.0 {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} semivariance must be finite and nonnegative"
            )));
        }
        if bin.pair_count == 0 {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} pair_count must be positive"
            )));
        }
    }
    Ok(())
}

fn validate_positive_candidates(values: &[f64], name: &str) -> Result<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(GeostatsError::InvalidInput(format!(
            "{name} must contain only finite positive values"
        )));
    }
    Ok(())
}

fn validate_nonnegative_candidates(values: &[f64], name: &str) -> Result<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(GeostatsError::InvalidInput(format!(
            "{name} must contain only finite nonnegative values"
        )));
    }
    Ok(())
}

