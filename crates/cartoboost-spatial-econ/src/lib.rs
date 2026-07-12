pub use cartoboost_geo_core::SpatialWeights;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpatialEconError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("linear system is singular or ill-conditioned")]
    SingularSystem,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("geo core error: {0}")]
    GeoCore(#[from] cartoboost_geo_core::GeoCoreError),
}

pub type Result<T> = std::result::Result<T, SpatialEconError>;
type SpatialEffects = (Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>);

pub fn spatial_weights_from_coo(
    n_rows: usize,
    n_cols: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    values: Vec<f64>,
    row_standardize: bool,
) -> Result<SpatialWeights> {
    if n_rows != n_cols {
        return Err(SpatialEconError::InvalidInput(
            "spatial weights must be square".to_string(),
        ));
    }
    if rows.len() != cols.len() || rows.len() != values.len() {
        return Err(SpatialEconError::InvalidInput(
            "weights rows, cols, and values must have the same length".to_string(),
        ));
    }
    let mut edges = Vec::with_capacity(values.len());
    for ((row, col), value) in rows.into_iter().zip(cols).zip(values) {
        if !value.is_finite() {
            return Err(SpatialEconError::InvalidInput(
                "spatial weights must contain only finite values".to_string(),
            ));
        }
        if row == col && value != 0.0 {
            return Err(SpatialEconError::InvalidInput(
                "spatial econometric weights must have a zero diagonal".to_string(),
            ));
        }
        if value != 0.0 {
            edges.push((row, col, value));
        }
    }
    let weights = SpatialWeights::from_edges(n_rows, edges, false)?;
    Ok(if row_standardize {
        weights.row_normalize()
    } else {
        weights
    })
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpatialModelKind {
    Ols,
    SpatialLag,
    SpatialError,
    SpatialDurbin,
    SpatialTwoStageLeastSquares,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpatialDiagnostics {
    pub residual_morans_i: f64,
    pub log_likelihood: Option<f64>,
    pub aic: Option<f64>,
    pub bic: Option<f64>,
    pub rho: Option<f64>,
    pub lambda: Option<f64>,
    pub sigma2: f64,
    pub n_samples: usize,
    pub n_features: usize,
    pub isolated_rows: Vec<usize>,
    pub direct_effects: Option<Vec<f64>>,
    pub indirect_effects: Option<Vec<f64>>,
    pub total_effects: Option<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpatialRegressionModel {
    kind: SpatialModelKind,
    intercept: f64,
    coefficients: Vec<f64>,
    durbin_coefficients: Vec<f64>,
    rho: Option<f64>,
    lambda: Option<f64>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
    diagnostics: SpatialDiagnostics,
}

impl SpatialRegressionModel {
    pub fn fit(
        kind: SpatialModelKind,
        x: Vec<Vec<f64>>,
        y: Vec<f64>,
        weights: &SpatialWeights,
    ) -> Result<Self> {
        validate_xy(&x, &y, weights)?;
        match kind {
            SpatialModelKind::Ols => fit_ols(x, y, weights),
            SpatialModelKind::SpatialLag => fit_spatial_lag_ml(x, y, weights, false),
            SpatialModelKind::SpatialTwoStageLeastSquares => {
                fit_two_stage_least_squares(x, y, weights)
            }
            SpatialModelKind::SpatialError => fit_spatial_error_ml(x, y, weights),
            SpatialModelKind::SpatialDurbin => fit_spatial_lag_ml(x, y, weights, true),
        }
    }

    pub fn predict(&self, x: Vec<Vec<f64>>, weights: &SpatialWeights) -> Result<Vec<f64>> {
        validate_weights_structure(weights)?;
        validate_matrix(&x, weights.n_nodes, "X")?;
        if x[0].len() != self.coefficients.len() {
            return Err(SpatialEconError::InvalidInput(format!(
                "X has {} features, but model was fitted with {}",
                x[0].len(),
                self.coefficients.len()
            )));
        }
        let mut pred = linear_predict(self.intercept, &self.coefficients, &x);
        if !self.durbin_coefficients.is_empty() {
            let wx = sparse_matrix_lag(weights, &x)?;
            add_linear_part(&mut pred, &self.durbin_coefficients, &wx);
        }
        if let Some(rho) = self.rho {
            pred = solve_spatial_lag_mean(pred, rho, weights)?;
        }
        // A spatial-error coefficient changes the disturbance covariance, not E[y | X].
        // Training innovations must not be reused to correct arbitrary prediction rows.
        Ok(pred)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model: Self = serde_json::from_str(&fs::read_to_string(path)?)?;
        model.validate_loaded()?;
        Ok(model)
    }

    pub fn diagnostics(&self) -> &SpatialDiagnostics {
        &self.diagnostics
    }

    pub fn kind(&self) -> SpatialModelKind {
        self.kind
    }

    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub fn durbin_coefficients(&self) -> &[f64] {
        &self.durbin_coefficients
    }

    fn validate_loaded(&self) -> Result<()> {
        if self.coefficients.is_empty()
            || !self.intercept.is_finite()
            || self
                .coefficients
                .iter()
                .chain(&self.durbin_coefficients)
                .chain(self.fitted_values.iter())
                .chain(self.residuals.iter())
                .any(|value| !value.is_finite())
            || self.rho.is_some_and(|value| !value.is_finite())
            || self.lambda.is_some_and(|value| !value.is_finite())
        {
            return Err(SpatialEconError::InvalidInput(
                "serialized spatial regression model contains invalid numeric state".to_string(),
            ));
        }
        if self.fitted_values.len() != self.residuals.len()
            || self.fitted_values.is_empty()
            || self.fitted_values.len() != self.diagnostics.n_samples
            || self.coefficients.len() != self.diagnostics.n_features
            || (!self.durbin_coefficients.is_empty()
                && self.durbin_coefficients.len() != self.coefficients.len())
        {
            return Err(SpatialEconError::InvalidInput(
                "serialized spatial regression model has inconsistent dimensions".to_string(),
            ));
        }
        let kind_is_consistent = match self.kind {
            SpatialModelKind::Ols => {
                self.rho.is_none() && self.lambda.is_none() && self.durbin_coefficients.is_empty()
            }
            SpatialModelKind::SpatialLag | SpatialModelKind::SpatialTwoStageLeastSquares => {
                self.rho.is_some() && self.lambda.is_none() && self.durbin_coefficients.is_empty()
            }
            SpatialModelKind::SpatialError => {
                self.rho.is_none() && self.lambda.is_some() && self.durbin_coefficients.is_empty()
            }
            SpatialModelKind::SpatialDurbin => {
                self.rho.is_some()
                    && self.lambda.is_none()
                    && self.durbin_coefficients.len() == self.coefficients.len()
            }
        };
        if !kind_is_consistent {
            return Err(SpatialEconError::InvalidInput(
                "serialized spatial regression model kind does not match its parameters"
                    .to_string(),
            ));
        }
        let likelihood_is_consistent = match self.kind {
            SpatialModelKind::SpatialTwoStageLeastSquares => {
                self.diagnostics.log_likelihood.is_none()
                    && self.diagnostics.aic.is_none()
                    && self.diagnostics.bic.is_none()
            }
            _ => {
                self.diagnostics.log_likelihood.is_some()
                    && self.diagnostics.aic.is_some()
                    && self.diagnostics.bic.is_some()
            }
        };
        let effects_are_consistent = match self.rho {
            Some(_) => {
                self.diagnostics
                    .direct_effects
                    .as_ref()
                    .is_some_and(|values| {
                        values.len() == self.coefficients.len()
                            && values.iter().all(|value| value.is_finite())
                    })
                    && self
                        .diagnostics
                        .indirect_effects
                        .as_ref()
                        .is_some_and(|values| {
                            values.len() == self.coefficients.len()
                                && values.iter().all(|value| value.is_finite())
                        })
                    && self
                        .diagnostics
                        .total_effects
                        .as_ref()
                        .is_some_and(|values| {
                            values.len() == self.coefficients.len()
                                && values.iter().all(|value| value.is_finite())
                        })
            }
            None => {
                self.diagnostics.direct_effects.is_none()
                    && self.diagnostics.indirect_effects.is_none()
                    && self.diagnostics.total_effects.is_none()
            }
        };
        if self.diagnostics.rho != self.rho
            || self.diagnostics.lambda != self.lambda
            || !self.diagnostics.residual_morans_i.is_finite()
            || !self.diagnostics.sigma2.is_finite()
            || self.diagnostics.sigma2 < 0.0
            || self
                .diagnostics
                .log_likelihood
                .into_iter()
                .chain(self.diagnostics.aic)
                .chain(self.diagnostics.bic)
                .any(|value| !value.is_finite())
            || !likelihood_is_consistent
            || !effects_are_consistent
        {
            return Err(SpatialEconError::InvalidInput(
                "serialized spatial regression model has inconsistent diagnostics".to_string(),
            ));
        }
        Ok(())
    }
}

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

fn effects(
    rho: Option<f64>,
    coefficients: &[f64],
    durbin_coefficients: &[f64],
    weights: &SpatialWeights,
) -> Result<SpatialEffects> {
    let Some(rho) = rho else {
        return Ok((None, None, None));
    };
    if !durbin_coefficients.is_empty() && durbin_coefficients.len() != coefficients.len() {
        return Err(SpatialEconError::InvalidInput(
            "Durbin coefficients must match the feature coefficient count".to_string(),
        ));
    }
    let inverse = invert(spatial_system_matrix(rho, weights)?)?;
    let dense_weights = dense_weights(weights);
    let n = weights.n_nodes as f64;
    let mut direct = Vec::with_capacity(coefficients.len());
    let mut total = Vec::with_capacity(coefficients.len());
    for (feature, beta) in coefficients.iter().enumerate() {
        let theta = durbin_coefficients.get(feature).copied().unwrap_or(0.0);
        let mut direct_sum = 0.0;
        let mut total_sum = 0.0;
        for (row, inverse_row) in inverse.iter().enumerate() {
            for (middle, multiplier) in inverse_row.iter().enumerate() {
                for (col, spatial_weight) in dense_weights[middle].iter().enumerate() {
                    let base_effect =
                        (if middle == col { *beta } else { 0.0 }) + theta * spatial_weight;
                    let effect = multiplier * base_effect;
                    total_sum += effect;
                    if row == col {
                        direct_sum += effect;
                    }
                }
            }
        }
        direct.push(direct_sum / n);
        total.push(total_sum / n);
    }
    let indirect: Vec<f64> = total.iter().zip(&direct).map(|(t, d)| t - d).collect();
    if direct
        .iter()
        .chain(&indirect)
        .chain(&total)
        .any(|value| !value.is_finite())
    {
        return Err(SpatialEconError::InvalidInput(
            "spatial effects are not finite".to_string(),
        ));
    }
    Ok((Some(direct), Some(indirect), Some(total)))
}

fn morans_i(residuals: &[f64], weights: &SpatialWeights) -> Result<f64> {
    let n = residuals.len();
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = residuals.iter().map(|value| value - mean).collect();
    let denominator = centered.iter().map(|value| value * value).sum::<f64>();
    if denominator == 0.0 {
        return Err(SpatialEconError::InvalidInput(
            "residual Moran's I is undefined because residual variance is zero".to_string(),
        ));
    }
    let w_centered = sparse_matvec(weights, &centered)?;
    let numerator = centered
        .iter()
        .zip(w_centered)
        .map(|(value, lagged)| value * lagged)
        .sum::<f64>();
    let weight_sum: f64 = weights.data.iter().sum();
    if weight_sum <= 0.0 {
        return Err(SpatialEconError::InvalidInput(
            "residual Moran's I is undefined because spatial weights have zero total weight"
                .to_string(),
        ));
    }
    let value = n as f64 / weight_sum * numerator / denominator;
    if !value.is_finite() {
        return Err(SpatialEconError::InvalidInput(
            "residual Moran's I is not finite".to_string(),
        ));
    }
    Ok(value)
}

fn residuals(truth: &[f64], fitted: &[f64]) -> Vec<f64> {
    truth
        .iter()
        .zip(fitted)
        .map(|(truth, fitted)| truth - fitted)
        .collect()
}

fn gaussian_log_likelihood(innovations: &[f64], log_abs_determinant: f64) -> Result<f64> {
    let rss = innovations.iter().map(|value| value * value).sum::<f64>();
    let sigma2 = rss / innovations.len() as f64;
    if !sigma2.is_finite() || sigma2 <= 0.0 || !log_abs_determinant.is_finite() {
        return Err(SpatialEconError::InvalidInput(
            "Gaussian likelihood requires positive finite innovation variance".to_string(),
        ));
    }
    let value = log_abs_determinant
        - 0.5 * innovations.len() as f64 * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0);
    if !value.is_finite() {
        return Err(SpatialEconError::InvalidInput(
            "Gaussian log likelihood is not finite".to_string(),
        ));
    }
    Ok(value)
}

fn require_residual_degrees_of_freedom(
    n_samples: usize,
    n_mean_parameters: usize,
    model: &str,
) -> Result<()> {
    if n_samples <= n_mean_parameters {
        return Err(SpatialEconError::InvalidInput(format!(
            "{model} requires more observations than mean parameters ({n_samples} rows for {n_mean_parameters} parameters)"
        )));
    }
    Ok(())
}

fn spatial_parameter_bound(weights: &SpatialWeights) -> Result<f64> {
    let max_row_sum = (0..weights.n_nodes)
        .map(|row| {
            weights.data[weights.indptr[row]..weights.indptr[row + 1]]
                .iter()
                .sum()
        })
        .fold(0.0_f64, f64::max);
    if !max_row_sum.is_finite() || max_row_sum <= 0.0 {
        return Err(SpatialEconError::InvalidInput(
            "a spatial dependence model requires at least one positive spatial weight".to_string(),
        ));
    }
    Ok(1.0 / max_row_sum)
}

fn validate_spatial_parameter(value: f64, bound: f64, name: &str) -> Result<()> {
    if !value.is_finite() || value.abs() >= bound {
        return Err(SpatialEconError::InvalidInput(format!(
            "{name}={value} is outside the admissible interval (-{bound}, {bound})"
        )));
    }
    Ok(())
}

fn spatial_system_matrix(parameter: f64, weights: &SpatialWeights) -> Result<Vec<Vec<f64>>> {
    validate_spatial_parameter(
        parameter,
        spatial_parameter_bound(weights)?,
        "spatial parameter",
    )?;
    let mut matrix = vec![vec![0.0; weights.n_nodes]; weights.n_nodes];
    for (row, matrix_row) in matrix.iter_mut().enumerate() {
        matrix_row[row] = 1.0;
        for offset in weights.indptr[row]..weights.indptr[row + 1] {
            matrix_row[weights.indices[offset]] -= parameter * weights.data[offset];
        }
    }
    Ok(matrix)
}

fn solve_spatial_lag_mean(
    structural_mean: Vec<f64>,
    rho: f64,
    weights: &SpatialWeights,
) -> Result<Vec<f64>> {
    if structural_mean.len() != weights.n_nodes {
        return Err(SpatialEconError::InvalidInput(
            "spatial lag mean length must match spatial weights".to_string(),
        ));
    }
    solve(spatial_system_matrix(rho, weights)?, structural_mean)
}

fn spatial_log_abs_determinant(parameter: f64, weights: &SpatialWeights) -> Result<f64> {
    log_abs_determinant(spatial_system_matrix(parameter, weights)?)
}

fn log_abs_determinant(mut matrix: Vec<Vec<f64>>) -> Result<f64> {
    let n = matrix.len();
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = f64::EPSILON * scale * n.max(1) as f64 * 64.0;
    let mut log_abs = 0.0;
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|left, right| {
                matrix[*left][pivot]
                    .abs()
                    .total_cmp(&matrix[*right][pivot].abs())
            })
            .expect("pivot range is non-empty");
        if matrix[best][pivot].abs() <= tolerance {
            return Err(SpatialEconError::SingularSystem);
        }
        matrix.swap(pivot, best);
        let pivot_value = matrix[pivot][pivot];
        log_abs += pivot_value.abs().ln();
        let pivot_row = matrix[pivot].clone();
        for matrix_row in matrix.iter_mut().skip(pivot + 1) {
            let factor = matrix_row[pivot] / pivot_value;
            for (col, pivot_entry) in pivot_row.iter().enumerate().skip(pivot + 1) {
                matrix_row[col] -= factor * pivot_entry;
            }
        }
    }
    if !log_abs.is_finite() {
        return Err(SpatialEconError::InvalidInput(
            "spatial log determinant is not finite".to_string(),
        ));
    }
    Ok(log_abs)
}

fn dense_weights(weights: &SpatialWeights) -> Vec<Vec<f64>> {
    let mut dense = vec![vec![0.0; weights.n_nodes]; weights.n_nodes];
    for (row, dense_row) in dense.iter_mut().enumerate() {
        for offset in weights.indptr[row]..weights.indptr[row + 1] {
            dense_row[weights.indices[offset]] = weights.data[offset];
        }
    }
    dense
}

fn invert(matrix: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = f64::EPSILON * scale * n.max(1) as f64 * 64.0;
    let mut augmented = vec![vec![0.0; 2 * n]; n];
    for row in 0..n {
        augmented[row][..n].copy_from_slice(&matrix[row]);
        augmented[row][n + row] = 1.0;
    }
    for pivot in 0..n {
        let best = (pivot..n)
            .max_by(|left, right| {
                augmented[*left][pivot]
                    .abs()
                    .total_cmp(&augmented[*right][pivot].abs())
            })
            .expect("pivot range is non-empty");
        if augmented[best][pivot].abs() <= tolerance {
            return Err(SpatialEconError::SingularSystem);
        }
        augmented.swap(pivot, best);
        let diagonal = augmented[pivot][pivot];
        for value in &mut augmented[pivot] {
            *value /= diagonal;
        }
        let pivot_row = augmented[pivot].clone();
        for (row, augmented_row) in augmented.iter_mut().enumerate() {
            if row == pivot {
                continue;
            }
            let factor = augmented_row[pivot];
            for (value, pivot_value) in augmented_row.iter_mut().zip(&pivot_row) {
                *value -= factor * pivot_value;
            }
        }
    }
    Ok(augmented.into_iter().map(|row| row[n..].to_vec()).collect())
}

fn sparse_matvec(weights: &SpatialWeights, x: &[f64]) -> Result<Vec<f64>> {
    if x.len() != weights.n_nodes {
        return Err(SpatialEconError::InvalidInput(
            "vector length must match spatial weights columns".to_string(),
        ));
    }
    let mut out = vec![0.0; weights.n_nodes];
    for (row, out_value) in out.iter_mut().enumerate() {
        let mut sum = 0.0;
        for idx in weights.indptr[row]..weights.indptr[row + 1] {
            sum += weights.data[idx] * x[weights.indices[idx]];
        }
        *out_value = sum;
    }
    Ok(out)
}

fn sparse_matrix_lag(weights: &SpatialWeights, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    validate_matrix(x, weights.n_nodes, "X")?;
    let cols = x[0].len();
    let mut out = vec![vec![0.0; cols]; weights.n_nodes];
    for (row, out_row) in out.iter_mut().enumerate() {
        for idx in weights.indptr[row]..weights.indptr[row + 1] {
            let col = weights.indices[idx];
            let weight = weights.data[idx];
            for (j, value) in out_row.iter_mut().enumerate() {
                *value += weight * x[col][j];
            }
        }
    }
    Ok(out)
}

fn with_intercept(x: &[Vec<f64>]) -> Vec<Vec<f64>> {
    x.iter()
        .map(|row| {
            let mut out = Vec::with_capacity(row.len() + 1);
            out.push(1.0);
            out.extend(row);
            out
        })
        .collect()
}

fn append_column(x: &mut [Vec<f64>], values: &[f64]) {
    for (row, value) in x.iter_mut().zip(values) {
        row.push(*value);
    }
}

fn append_columns(x: &mut [Vec<f64>], values: &[Vec<f64>]) {
    for (row, extra) in x.iter_mut().zip(values) {
        row.extend(extra);
    }
}

fn linear_predict(intercept: f64, coefficients: &[f64], x: &[Vec<f64>]) -> Vec<f64> {
    x.iter()
        .map(|row| {
            intercept
                + coefficients
                    .iter()
                    .zip(row)
                    .map(|(coef, value)| coef * value)
                    .sum::<f64>()
        })
        .collect()
}

fn predict_design(design: &[Vec<f64>], params: &[f64]) -> Vec<f64> {
    design
        .iter()
        .map(|row| {
            row.iter()
                .zip(params)
                .map(|(value, param)| value * param)
                .sum()
        })
        .collect()
}

fn add_linear_part(pred: &mut [f64], coefficients: &[f64], x: &[Vec<f64>]) {
    for (prediction, row) in pred.iter_mut().zip(x) {
        *prediction += coefficients
            .iter()
            .zip(row)
            .map(|(coef, value)| coef * value)
            .sum::<f64>();
    }
}

fn ols_params(x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>> {
    let n_cols = x[0].len();
    let mut xtx = vec![vec![0.0; n_cols]; n_cols];
    let mut xty = vec![0.0; n_cols];
    for (row, target) in x.iter().zip(y) {
        for i in 0..n_cols {
            xty[i] += row[i] * target;
            for j in 0..n_cols {
                xtx[i][j] += row[i] * row[j];
            }
        }
    }
    solve(xtx, xty)
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    if n == 0
        || a.len() != n
        || a.iter().any(|row| row.len() != n)
        || a.iter().flatten().chain(&b).any(|value| !value.is_finite())
    {
        return Err(SpatialEconError::InvalidInput(
            "linear system must be a finite non-empty square matrix".to_string(),
        ));
    }
    let scale = a
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = f64::EPSILON * scale * n as f64 * 64.0;
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if a[row][pivot].abs() > a[best][pivot].abs() {
                best = row;
            }
        }
        if a[best][pivot].abs() <= tolerance {
            return Err(SpatialEconError::SingularSystem);
        }
        a.swap(pivot, best);
        b.swap(pivot, best);
        let diag = a[pivot][pivot];
        for value in a[pivot].iter_mut().take(n).skip(pivot) {
            *value /= diag;
        }
        b[pivot] /= diag;
        let pivot_row = a[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for (col, pivot_value) in pivot_row.iter().enumerate().take(n).skip(pivot) {
                a[row][col] -= factor * pivot_value;
            }
            b[row] -= factor * b[pivot];
        }
    }
    Ok(b)
}

fn validate_xy(x: &[Vec<f64>], y: &[f64], weights: &SpatialWeights) -> Result<()> {
    if y.is_empty() {
        return Err(SpatialEconError::InvalidInput(
            "y must contain at least one row".to_string(),
        ));
    }
    if x.len() != y.len() {
        return Err(SpatialEconError::InvalidInput(
            "X and y must contain the same number of rows".to_string(),
        ));
    }
    if weights.n_nodes != y.len() {
        return Err(SpatialEconError::InvalidInput(
            "spatial weights row count must match X and y".to_string(),
        ));
    }
    validate_weights_structure(weights)?;
    validate_matrix(x, y.len(), "X")?;
    validate_vector(y, "y")
}

fn validate_weights_structure(weights: &SpatialWeights) -> Result<()> {
    if weights.n_nodes == 0
        || weights.indptr.len() != weights.n_nodes + 1
        || weights.indptr.first().copied() != Some(0)
        || weights.indptr.last().copied() != Some(weights.indices.len())
        || weights.indices.len() != weights.data.len()
        || weights
            .indptr
            .windows(2)
            .any(|bounds| bounds[0] > bounds[1])
    {
        return Err(SpatialEconError::InvalidInput(
            "spatial weights must be a valid square CSR matrix".to_string(),
        ));
    }
    for row in 0..weights.n_nodes {
        for offset in weights.indptr[row]..weights.indptr[row + 1] {
            let col = weights.indices[offset];
            let value = weights.data[offset];
            if col >= weights.n_nodes || !value.is_finite() || value < 0.0 {
                return Err(SpatialEconError::InvalidInput(
                    "spatial weights must have valid columns and non-negative finite values"
                        .to_string(),
                ));
            }
            if row == col && value != 0.0 {
                return Err(SpatialEconError::InvalidInput(
                    "spatial econometric weights must have a zero diagonal".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_matrix(x: &[Vec<f64>], expected_rows: usize, name: &str) -> Result<()> {
    if x.len() != expected_rows {
        return Err(SpatialEconError::InvalidInput(format!(
            "{name} row count must match spatial weights"
        )));
    }
    let Some(first) = x.first() else {
        return Err(SpatialEconError::InvalidInput(format!(
            "{name} must contain at least one row"
        )));
    };
    if first.is_empty() {
        return Err(SpatialEconError::InvalidInput(format!(
            "{name} must contain at least one feature"
        )));
    }
    for row in x {
        if row.len() != first.len() {
            return Err(SpatialEconError::InvalidInput(format!(
                "{name} must be rectangular"
            )));
        }
        validate_vector(row, name)?;
    }
    Ok(())
}

fn validate_vector(values: &[f64], name: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SpatialEconError::InvalidInput(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagnostics(
        n_samples: usize,
        n_features: usize,
        rho: Option<f64>,
        lambda: Option<f64>,
    ) -> SpatialDiagnostics {
        SpatialDiagnostics {
            residual_morans_i: 0.0,
            log_likelihood: None,
            aic: None,
            bic: None,
            rho,
            lambda,
            sigma2: 1.0,
            n_samples,
            n_features,
            isolated_rows: Vec::new(),
            direct_effects: None,
            indirect_effects: None,
            total_effects: None,
        }
    }

    fn chain_weights() -> SpatialWeights {
        spatial_weights_from_coo(
            4,
            4,
            vec![0, 1, 1, 2, 2, 3],
            vec![1, 0, 2, 1, 3, 2],
            vec![1.0; 6],
            true,
        )
        .expect("weights")
    }

    fn ring_weights(n: usize) -> SpatialWeights {
        let mut rows = Vec::with_capacity(2 * n);
        let mut cols = Vec::with_capacity(2 * n);
        for row in 0..n {
            rows.extend([row, row]);
            cols.extend([(row + n - 1) % n, (row + 1) % n]);
        }
        spatial_weights_from_coo(n, n, rows, cols, vec![1.0; 2 * n], true).expect("ring weights")
    }

    fn fixture_x() -> Vec<Vec<f64>> {
        [0.0, 1.0, 4.0, 2.0, 7.0, 3.0, 9.0, 5.0, 11.0, 6.0, 10.0, 8.0]
            .into_iter()
            .map(|value| vec![value])
            .collect()
    }

    fn fixture_innovations() -> Vec<f64> {
        vec![
            0.30, -0.20, 0.10, -0.35, 0.25, 0.05, -0.15, 0.40, -0.25, 0.15, -0.05, -0.10,
        ]
    }

    fn spatial_lag_target(
        x: &[Vec<f64>],
        weights: &SpatialWeights,
        rho: f64,
        beta: f64,
        theta: f64,
    ) -> Vec<f64> {
        let wx = sparse_matrix_lag(weights, x).expect("WX");
        let innovations = fixture_innovations();
        let structural_mean: Vec<f64> = x
            .iter()
            .zip(wx)
            .zip(innovations)
            .map(|((row, lagged), innovation)| 1.5 + beta * row[0] + theta * lagged[0] + innovation)
            .collect();
        solve_spatial_lag_mean(structural_mean, rho, weights).expect("known SAR target")
    }

    #[test]
    fn spatial_lag_fits_known_toy_system() {
        let weights = ring_weights(12);
        let x = fixture_x();
        let y = spatial_lag_target(&x, &weights, 0.35, 1.2, 0.0);
        let model =
            SpatialRegressionModel::fit(SpatialModelKind::SpatialLag, x.clone(), y, &weights)
                .expect("fit");
        let pred = model.predict(x, &weights).expect("predict");
        assert_eq!(pred.len(), 12);
        assert!(model.diagnostics().rho.is_some());
        assert!(model.diagnostics().log_likelihood.is_some());
        assert!(model.diagnostics().direct_effects.is_some());
        assert!(model.diagnostics().residual_morans_i.is_finite());
        let rho = model.diagnostics().rho.expect("rho");
        let expected_likelihood = gaussian_log_likelihood(
            &model.residuals,
            spatial_log_abs_determinant(rho, &weights).expect("Jacobian"),
        )
        .expect("likelihood");
        assert!(
            (model.diagnostics().log_likelihood.expect("likelihood") - expected_likelihood).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn spatial_error_reports_lambda() {
        let weights = ring_weights(12);
        let x = fixture_x();
        let disturbances = solve_spatial_lag_mean(fixture_innovations(), 0.4, &weights)
            .expect("known SEM disturbances");
        let y: Vec<f64> = x
            .iter()
            .zip(disturbances)
            .map(|(row, disturbance)| 2.0 + 1.1 * row[0] + disturbance)
            .collect();
        let model = SpatialRegressionModel::fit(SpatialModelKind::SpatialError, x, y, &weights)
            .expect("fit");
        assert!(model.diagnostics().lambda.is_some());
        assert!(model.diagnostics().log_likelihood.is_some());
    }

    #[test]
    fn spatial_two_stage_least_squares_reports_rho() {
        let weights = chain_weights();
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.6, 4.4, 5.8];
        let model = SpatialRegressionModel::fit(
            SpatialModelKind::SpatialTwoStageLeastSquares,
            x.clone(),
            y,
            &weights,
        )
        .expect("fit");
        assert!(model.diagnostics().rho.is_some());
        assert!(model.diagnostics().log_likelihood.is_none());
        assert!(model.diagnostics().aic.is_none());
        assert!(model.diagnostics().bic.is_none());
        assert_eq!(model.predict(x, &weights).unwrap().len(), 4);
    }

    #[test]
    fn durbin_reports_effects_and_roundtrips() {
        let weights = ring_weights(12);
        let x = fixture_x();
        let y = spatial_lag_target(&x, &weights, 0.25, 1.1, 0.4);
        let model =
            SpatialRegressionModel::fit(SpatialModelKind::SpatialDurbin, x.clone(), y, &weights)
                .expect("fit");
        assert!(model.diagnostics().total_effects.is_some());
        let path = std::env::temp_dir().join("cartoboost-spatial-econ-test.json");
        model.save(&path).expect("save");
        let loaded = SpatialRegressionModel::load(&path).expect("load");
        let before = model.predict(x.clone(), &weights).unwrap();
        let after = loaded.predict(x, &weights).unwrap();
        assert!(before
            .iter()
            .zip(after)
            .all(|(left, right)| (left - right).abs() < 1.0e-12));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn durbin_effects_use_exact_spatial_multiplier() {
        let weights = spatial_weights_from_coo(2, 2, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        let (direct, indirect, total) =
            effects(Some(0.25), &[2.0], &[0.5], &weights).expect("effects");
        assert!((direct.unwrap()[0] - 2.266_666_666_666_666_6).abs() < 1.0e-12);
        assert!((total.unwrap()[0] - 3.333_333_333_333_333_5).abs() < 1.0e-12);
        assert!((indirect.unwrap()[0] - 1.066_666_666_666_666_9).abs() < 1.0e-12);
    }

    #[test]
    fn spatial_lag_prediction_solves_reduced_form_mean() {
        let weights = spatial_weights_from_coo(2, 2, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        let model = SpatialRegressionModel {
            kind: SpatialModelKind::SpatialLag,
            intercept: 1.0,
            coefficients: vec![2.0],
            durbin_coefficients: Vec::new(),
            rho: Some(0.25),
            lambda: None,
            fitted_values: Vec::new(),
            residuals: Vec::new(),
            diagnostics: diagnostics(2, 1, Some(0.25), None),
        };
        let prediction = model
            .predict(vec![vec![0.0], vec![1.0]], &weights)
            .expect("reduced-form prediction");
        let denominator = 1.0 - 0.25_f64.powi(2);
        assert!((prediction[0] - (1.0 + 0.25 * 3.0) / denominator).abs() < 1.0e-12);
        assert!((prediction[1] - (3.0 + 0.25) / denominator).abs() < 1.0e-12);
    }

    #[test]
    fn spatial_error_prediction_does_not_reuse_training_innovations() {
        let weights = spatial_weights_from_coo(2, 2, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        let model = SpatialRegressionModel {
            kind: SpatialModelKind::SpatialError,
            intercept: 1.0,
            coefficients: vec![2.0],
            durbin_coefficients: Vec::new(),
            rho: None,
            lambda: Some(0.75),
            fitted_values: vec![101.0, -97.0],
            residuals: vec![100.0, -100.0],
            diagnostics: diagnostics(2, 1, None, Some(0.75)),
        };
        assert_eq!(
            model
                .predict(vec![vec![0.0], vec![1.0]], &weights)
                .expect("SEM mean"),
            vec![1.0, 3.0]
        );
    }

    #[test]
    fn singular_design_is_not_hidden_by_ridge_regularization() {
        let weights = ring_weights(6);
        let x = vec![vec![1.0, 2.0]; 6];
        let y = vec![0.0, 1.0, 0.0, -1.0, 0.5, -0.5];
        assert!(matches!(
            SpatialRegressionModel::fit(SpatialModelKind::Ols, x, y, &weights),
            Err(SpatialEconError::SingularSystem)
        ));
    }

    #[test]
    fn saturated_durbin_fit_fails_clearly() {
        let error = SpatialRegressionModel::fit(
            SpatialModelKind::SpatialDurbin,
            vec![vec![1.0], vec![2.0], vec![4.0], vec![8.0]],
            vec![2.0, 3.0, 6.0, 10.0],
            &chain_weights(),
        )
        .expect_err("saturated likelihood must fail");
        assert!(error.to_string().contains("more observations"));
    }

    #[test]
    fn invalid_weights_fail_clearly() {
        let err = spatial_weights_from_coo(2, 3, vec![0], vec![1], vec![1.0], false)
            .expect_err("must fail");
        assert!(err.to_string().contains("square"));

        let err = spatial_weights_from_coo(2, 2, vec![0], vec![0], vec![1.0], false)
            .expect_err("self weights must fail");
        assert!(err.to_string().contains("zero diagonal"));
    }

    #[test]
    fn isolated_nodes_are_recorded() {
        let weights = spatial_weights_from_coo(3, 3, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        assert_eq!(weights.isolated_nodes(), vec![2]);
    }
}
