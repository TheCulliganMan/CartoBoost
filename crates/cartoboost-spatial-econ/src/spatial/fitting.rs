fn fit_ols(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: &SpatialWeights,
) -> Result<SpatialRegressionModel> {
    let design = with_intercept(&x);
    require_residual_degrees_of_freedom(y.len(), design[0].len(), "OLS")?;
    let params = ols_params(&design, &y)?;
    let intercept = params[0];
    let coefficients = params[1..].to_vec();
    let fitted = predict_design(&design, &params);
    let model_residuals = residuals(&y, &fitted);
    let log_likelihood = gaussian_log_likelihood(&model_residuals, 0.0)?;
    finish_model(
        SpatialModelKind::Ols,
        intercept,
        coefficients,
        Vec::new(),
        None,
        None,
        fitted,
        y,
        weights,
        Some(log_likelihood),
    )
}

fn fit_spatial_lag_ml(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: &SpatialWeights,
    include_lag_x: bool,
) -> Result<SpatialRegressionModel> {
    let n_features = x[0].len();
    let lag_y = sparse_matvec(weights, &y)?;
    let lag_x = if include_lag_x {
        Some(sparse_matrix_lag(weights, &x)?)
    } else {
        None
    };
    let mut design = with_intercept(&x);
    if let Some(values) = &lag_x {
        append_columns(&mut design, values);
    }
    require_residual_degrees_of_freedom(
        y.len(),
        design[0].len() + 1,
        if include_lag_x {
            "spatial Durbin maximum likelihood"
        } else {
            "spatial lag maximum likelihood"
        },
    )?;
    let bound = spatial_parameter_bound(weights)?;
    let profile = maximize_spatial_profile(bound, |rho| {
        let transformed_y: Vec<f64> = y
            .iter()
            .zip(&lag_y)
            .map(|(value, lagged)| value - rho * lagged)
            .collect();
        profile_fit(rho, &design, &transformed_y, weights)
    })?;
    let intercept = profile.params[0];
    let coefficients = profile.params[1..1 + n_features].to_vec();
    let durbin_coefficients = if include_lag_x {
        profile.params[1 + n_features..].to_vec()
    } else {
        Vec::new()
    };
    let structural_mean = predict_design(&design, &profile.params);
    let fitted: Vec<f64> = structural_mean
        .iter()
        .zip(&lag_y)
        .map(|(mean, lagged_y)| mean + profile.parameter * lagged_y)
        .collect();
    finish_model(
        if include_lag_x {
            SpatialModelKind::SpatialDurbin
        } else {
            SpatialModelKind::SpatialLag
        },
        intercept,
        coefficients,
        durbin_coefficients,
        Some(profile.parameter),
        None,
        fitted,
        y,
        weights,
        Some(profile.log_likelihood),
    )
}

fn fit_spatial_error_ml(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: &SpatialWeights,
) -> Result<SpatialRegressionModel> {
    let design = with_intercept(&x);
    require_residual_degrees_of_freedom(
        y.len(),
        design[0].len() + 1,
        "spatial error maximum likelihood",
    )?;
    let lag_y = sparse_matvec(weights, &y)?;
    let lag_design = sparse_matrix_lag(weights, &design)?;
    let bound = spatial_parameter_bound(weights)?;
    let profile = maximize_spatial_profile(bound, |lambda| {
        let transformed_y: Vec<f64> = y
            .iter()
            .zip(&lag_y)
            .map(|(value, lagged)| value - lambda * lagged)
            .collect();
        let transformed_design: Vec<Vec<f64>> = design
            .iter()
            .zip(&lag_design)
            .map(|(row, lagged_row)| {
                row.iter()
                    .zip(lagged_row)
                    .map(|(value, lagged)| value - lambda * lagged)
                    .collect()
            })
            .collect();
        profile_fit(lambda, &transformed_design, &transformed_y, weights)
    })?;
    let intercept = profile.params[0];
    let coefficients = profile.params[1..].to_vec();
    let base_fitted = predict_design(&design, &profile.params);
    let disturbances = residuals(&y, &base_fitted);
    let lagged_disturbances = sparse_matvec(weights, &disturbances)?;
    let innovations: Vec<f64> = disturbances
        .iter()
        .zip(lagged_disturbances)
        .map(|(value, lagged)| value - profile.parameter * lagged)
        .collect();
    let fitted: Vec<f64> = y
        .iter()
        .zip(innovations)
        .map(|(truth, innovation)| truth - innovation)
        .collect();
    finish_model(
        SpatialModelKind::SpatialError,
        intercept,
        coefficients,
        Vec::new(),
        None,
        Some(profile.parameter),
        fitted,
        y,
        weights,
        Some(profile.log_likelihood),
    )
}

fn fit_two_stage_least_squares(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: &SpatialWeights,
) -> Result<SpatialRegressionModel> {
    let n_features = x[0].len();
    let lag_y = sparse_matvec(weights, &y)?;
    let lag_x = sparse_matrix_lag(weights, &x)?;
    let mut first_stage_design = with_intercept(&x);
    append_columns(&mut first_stage_design, &lag_x);
    require_residual_degrees_of_freedom(
        y.len(),
        first_stage_design[0].len(),
        "spatial two-stage least-squares first stage",
    )?;
    let first_stage_params = ols_params(&first_stage_design, &lag_y)?;
    let fitted_lag_y = predict_design(&first_stage_design, &first_stage_params);

    let mut second_stage_design = with_intercept(&x);
    append_column(&mut second_stage_design, &fitted_lag_y);
    require_residual_degrees_of_freedom(
        y.len(),
        second_stage_design[0].len(),
        "spatial two-stage least-squares second stage",
    )?;
    let params = ols_params(&second_stage_design, &y)?;
    let intercept = params[0];
    let coefficients = params[1..1 + n_features].to_vec();
    let rho_value = params[1 + n_features];
    validate_spatial_parameter(rho_value, spatial_parameter_bound(weights)?, "rho")?;
    let rho = Some(rho_value);
    let mut fitted = linear_predict(intercept, &coefficients, &x);
    if let Some(value) = rho {
        // The first-stage lag estimates coefficients; structural residuals use observed Wy.
        for (prediction, observed_lag_y) in fitted.iter_mut().zip(lag_y) {
            *prediction += value * observed_lag_y;
        }
    }
    finish_model(
        SpatialModelKind::SpatialTwoStageLeastSquares,
        intercept,
        coefficients,
        Vec::new(),
        rho,
        None,
        fitted,
        y,
        weights,
        None,
    )
}

#[derive(Clone, Debug)]
struct ProfileFit {
    parameter: f64,
    params: Vec<f64>,
    log_likelihood: f64,
}

fn profile_fit(
    parameter: f64,
    design: &[Vec<f64>],
    transformed_y: &[f64],
    weights: &SpatialWeights,
) -> Result<ProfileFit> {
    let params = ols_params(design, transformed_y)?;
    let innovations = residuals(transformed_y, &predict_design(design, &params));
    let log_likelihood = gaussian_log_likelihood(
        &innovations,
        spatial_log_abs_determinant(parameter, weights)?,
    )?;
    Ok(ProfileFit {
        parameter,
        params,
        log_likelihood,
    })
}

fn maximize_spatial_profile<F>(bound: f64, objective: F) -> Result<ProfileFit>
where
    F: Fn(f64) -> Result<ProfileFit>,
{
    const GRID_STEPS: usize = 200;
    const GOLDEN_ITERATIONS: usize = 80;
    let search_bound = 0.999 * bound;
    let mut scores = Vec::with_capacity(GRID_STEPS + 1);
    for index in 0..=GRID_STEPS {
        let parameter = -search_bound + 2.0 * search_bound * index as f64 / GRID_STEPS as f64;
        scores.push(objective(parameter)?);
    }
    let best_index = scores
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.log_likelihood.total_cmp(&right.log_likelihood))
        .map(|(index, _)| index)
        .ok_or_else(|| {
            SpatialEconError::InvalidInput(
                "spatial likelihood search produced no candidates".to_string(),
            )
        })?;
    if best_index == 0 || best_index == GRID_STEPS {
        return Err(SpatialEconError::InvalidInput(
            "spatial likelihood optimum reached the admissible stability boundary".to_string(),
        ));
    }

    let mut left = scores[best_index - 1].parameter;
    let mut right = scores[best_index + 1].parameter;
    let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut inner_left = right - golden_ratio * (right - left);
    let mut inner_right = left + golden_ratio * (right - left);
    let mut left_fit = objective(inner_left)?;
    let mut right_fit = objective(inner_right)?;
    for _ in 0..GOLDEN_ITERATIONS {
        if left_fit.log_likelihood > right_fit.log_likelihood {
            right = inner_right;
            inner_right = inner_left;
            right_fit = left_fit;
            inner_left = right - golden_ratio * (right - left);
            left_fit = objective(inner_left)?;
        } else {
            left = inner_left;
            inner_left = inner_right;
            left_fit = right_fit;
            inner_right = left + golden_ratio * (right - left);
            right_fit = objective(inner_right)?;
        }
    }
    let mut best = scores[best_index].clone();
    for candidate in [left_fit, right_fit] {
        if candidate.log_likelihood > best.log_likelihood {
            best = candidate;
        }
    }
    validate_spatial_parameter(best.parameter, bound, "spatial dependence parameter")?;
    Ok(best)
}

#[allow(clippy::too_many_arguments)]
fn finish_model(
    kind: SpatialModelKind,
    intercept: f64,
    coefficients: Vec<f64>,
    durbin_coefficients: Vec<f64>,
    rho: Option<f64>,
    lambda: Option<f64>,
    fitted_values: Vec<f64>,
    y: Vec<f64>,
    weights: &SpatialWeights,
    log_likelihood: Option<f64>,
) -> Result<SpatialRegressionModel> {
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(truth, prediction)| truth - prediction)
        .collect();
    let sigma2 = residuals.iter().map(|value| value * value).sum::<f64>() / y.len() as f64;
    let k = 1
        + coefficients.len()
        + durbin_coefficients.len()
        + usize::from(rho.is_some())
        + usize::from(lambda.is_some())
        + 1;
    if log_likelihood.is_some_and(|value| !value.is_finite()) {
        return Err(SpatialEconError::InvalidInput(
            "model log likelihood is not finite".to_string(),
        ));
    }
    let aic = log_likelihood.map(|ll| 2.0 * k as f64 - 2.0 * ll);
    let bic = log_likelihood.map(|ll| (k as f64) * (y.len() as f64).ln() - 2.0 * ll);
    let (direct_effects, indirect_effects, total_effects) =
        effects(rho, &coefficients, &durbin_coefficients, weights)?;
    let diagnostics = SpatialDiagnostics {
        residual_morans_i: morans_i(&residuals, weights)?,
        log_likelihood,
        aic,
        bic,
        rho,
        lambda,
        sigma2,
        n_samples: y.len(),
        n_features: coefficients.len(),
        isolated_rows: weights.isolated_nodes(),
        direct_effects,
        indirect_effects,
        total_effects,
    };
    Ok(SpatialRegressionModel {
        kind,
        intercept,
        coefficients,
        durbin_coefficients,
        rho,
        lambda,
        fitted_values,
        residuals,
        diagnostics,
    })
}

