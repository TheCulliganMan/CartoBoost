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

