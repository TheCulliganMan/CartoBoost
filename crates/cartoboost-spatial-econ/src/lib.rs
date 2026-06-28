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
            SpatialModelKind::Ols => fit_augmented(kind, x, y, weights, false, false),
            SpatialModelKind::SpatialLag => fit_augmented(kind, x, y, weights, true, false),
            SpatialModelKind::SpatialTwoStageLeastSquares => {
                fit_two_stage_least_squares(x, y, weights)
            }
            SpatialModelKind::SpatialError => fit_spatial_error(x, y, weights),
            SpatialModelKind::SpatialDurbin => fit_augmented(kind, x, y, weights, true, true),
        }
    }

    pub fn predict(&self, x: Vec<Vec<f64>>, weights: &SpatialWeights) -> Result<Vec<f64>> {
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
        if let Some(lambda) = self.lambda {
            if self.residuals.len() == pred.len() {
                let wu = sparse_matvec(weights, &self.residuals)?;
                for (value, lagged_error) in pred.iter_mut().zip(wu) {
                    *value += lambda * lagged_error;
                }
            }
        }
        Ok(pred)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_str(&fs::read_to_string(path)?)?)
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
}

fn fit_augmented(
    kind: SpatialModelKind,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: &SpatialWeights,
    include_lag_y: bool,
    include_lag_x: bool,
) -> Result<SpatialRegressionModel> {
    let n_features = x[0].len();
    let mut design = with_intercept(&x);
    let lag_y = if include_lag_y {
        Some(sparse_matvec(weights, &y)?)
    } else {
        None
    };
    if let Some(values) = &lag_y {
        append_column(&mut design, values);
    }
    let lag_x = if include_lag_x {
        Some(sparse_matrix_lag(weights, &x)?)
    } else {
        None
    };
    if let Some(wx) = &lag_x {
        append_columns(&mut design, wx);
    }
    let params = ols_params(&design, &y)?;
    let intercept = params[0];
    let coefficients = params[1..1 + n_features].to_vec();
    let mut offset = 1 + n_features;
    let rho = if include_lag_y {
        let value = params[offset].clamp(-0.99, 0.99);
        offset += 1;
        Some(value)
    } else {
        None
    };
    let durbin_coefficients = if include_lag_x {
        params[offset..offset + n_features].to_vec()
    } else {
        Vec::new()
    };
    let mut fitted = linear_predict(intercept, &coefficients, &x);
    if let (Some(value), Some(wy)) = (rho, lag_y) {
        for (prediction, lagged_y) in fitted.iter_mut().zip(wy) {
            *prediction += value * lagged_y;
        }
    }
    if let Some(wx) = lag_x {
        add_linear_part(&mut fitted, &durbin_coefficients, &wx);
    }
    finish_model(
        kind,
        intercept,
        coefficients,
        durbin_coefficients,
        rho,
        None,
        fitted,
        y,
        weights,
    )
}

fn fit_spatial_error(
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: &SpatialWeights,
) -> Result<SpatialRegressionModel> {
    let params = ols_params(&with_intercept(&x), &y)?;
    let intercept = params[0];
    let coefficients = params[1..].to_vec();
    let base_fitted = linear_predict(intercept, &coefficients, &x);
    let base_residuals: Vec<f64> = y
        .iter()
        .zip(&base_fitted)
        .map(|(truth, prediction)| truth - prediction)
        .collect();
    let lagged_residuals = sparse_matvec(weights, &base_residuals)?;
    let denom = lagged_residuals.iter().map(|v| v * v).sum::<f64>();
    let lambda = if denom > 1e-12 {
        base_residuals
            .iter()
            .zip(&lagged_residuals)
            .map(|(u, wu)| u * wu)
            .sum::<f64>()
            / denom
    } else {
        0.0
    }
    .clamp(-0.99, 0.99);
    let mut fitted = base_fitted;
    for (prediction, lagged_error) in fitted.iter_mut().zip(lagged_residuals) {
        *prediction += lambda * lagged_error;
    }
    finish_model(
        SpatialModelKind::SpatialError,
        intercept,
        coefficients,
        Vec::new(),
        None,
        Some(lambda),
        fitted,
        y,
        weights,
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
    let first_stage_params = ols_params(&first_stage_design, &lag_y)?;
    let fitted_lag_y = predict_design(&first_stage_design, &first_stage_params);

    let mut second_stage_design = with_intercept(&x);
    append_column(&mut second_stage_design, &fitted_lag_y);
    let params = ols_params(&second_stage_design, &y)?;
    let intercept = params[0];
    let coefficients = params[1..1 + n_features].to_vec();
    let rho = Some(params[1 + n_features].clamp(-0.99, 0.99));
    let mut fitted = linear_predict(intercept, &coefficients, &x);
    if let Some(value) = rho {
        for (prediction, lagged_y_hat) in fitted.iter_mut().zip(fitted_lag_y) {
            *prediction += value * lagged_y_hat;
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
    )
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
        + usize::from(lambda.is_some());
    let log_likelihood = if sigma2 > 0.0 {
        Some(-0.5 * y.len() as f64 * ((2.0 * std::f64::consts::PI * sigma2).ln() + 1.0))
    } else {
        None
    };
    let aic = log_likelihood.map(|ll| 2.0 * k as f64 - 2.0 * ll);
    let bic = log_likelihood.map(|ll| (k as f64) * (y.len() as f64).ln() - 2.0 * ll);
    let (direct_effects, indirect_effects, total_effects) =
        effects(rho, &coefficients, &durbin_coefficients);
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

fn effects(rho: Option<f64>, coefficients: &[f64], durbin_coefficients: &[f64]) -> SpatialEffects {
    if durbin_coefficients.is_empty() {
        return (None, None, None);
    }
    let multiplier = 1.0 / (1.0 - rho.unwrap_or(0.0));
    let total: Vec<f64> = coefficients
        .iter()
        .zip(durbin_coefficients)
        .map(|(beta, theta)| (beta + theta) * multiplier)
        .collect();
    let direct = coefficients.to_vec();
    let indirect: Vec<f64> = total.iter().zip(&direct).map(|(t, d)| t - d).collect();
    (Some(direct), Some(indirect), Some(total))
}

fn morans_i(residuals: &[f64], weights: &SpatialWeights) -> Result<f64> {
    let n = residuals.len();
    let mean = residuals.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = residuals.iter().map(|value| value - mean).collect();
    let denominator = centered.iter().map(|value| value * value).sum::<f64>();
    if denominator <= 1e-12 {
        return Ok(0.0);
    }
    let w_centered = sparse_matvec(weights, &centered)?;
    let numerator = centered
        .iter()
        .zip(w_centered)
        .map(|(value, lagged)| value * lagged)
        .sum::<f64>();
    let weight_sum: f64 = weights.data.iter().sum();
    if weight_sum <= 0.0 {
        return Ok(0.0);
    }
    Ok(n as f64 / weight_sum * numerator / denominator)
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
    for (i, row) in xtx.iter_mut().enumerate() {
        row[i] += 1e-10;
    }
    solve(xtx, xty)
}

fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = b.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if a[row][pivot].abs() > a[best][pivot].abs() {
                best = row;
            }
        }
        if a[best][pivot].abs() < 1e-12 {
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
    validate_matrix(x, y.len(), "X")?;
    validate_vector(y, "y")
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

    #[test]
    fn spatial_lag_fits_known_toy_system() {
        let weights = chain_weights();
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.6, 4.4, 5.8];
        let model =
            SpatialRegressionModel::fit(SpatialModelKind::SpatialLag, x.clone(), y, &weights)
                .expect("fit");
        let pred = model.predict(x, &weights).expect("predict");
        assert_eq!(pred.len(), 4);
        assert!(model.diagnostics().rho.is_some());
        assert!(model.diagnostics().residual_morans_i.is_finite());
    }

    #[test]
    fn spatial_error_reports_lambda() {
        let weights = chain_weights();
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![1.0, 2.2, 4.0, 5.3];
        let model = SpatialRegressionModel::fit(SpatialModelKind::SpatialError, x, y, &weights)
            .expect("fit");
        assert!(model.diagnostics().lambda.is_some());
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
        assert_eq!(model.predict(x, &weights).unwrap().len(), 4);
    }

    #[test]
    fn durbin_reports_effects_and_roundtrips() {
        let weights = chain_weights();
        let x = vec![vec![1.0], vec![2.0], vec![4.0], vec![8.0]];
        let y = vec![2.0, 3.0, 6.0, 10.0];
        let model =
            SpatialRegressionModel::fit(SpatialModelKind::SpatialDurbin, x.clone(), y, &weights)
                .expect("fit");
        assert!(model.diagnostics().total_effects.is_some());
        let path = std::env::temp_dir().join("cartoboost-spatial-econ-test.json");
        model.save(&path).expect("save");
        let loaded = SpatialRegressionModel::load(&path).expect("load");
        assert_eq!(
            model.predict(x.clone(), &weights).unwrap(),
            loaded.predict(x, &weights).unwrap()
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn invalid_weights_fail_clearly() {
        let err = spatial_weights_from_coo(2, 3, vec![0], vec![1], vec![1.0], false)
            .expect_err("must fail");
        assert!(err.to_string().contains("square"));
    }

    #[test]
    fn isolated_nodes_are_recorded() {
        let weights = spatial_weights_from_coo(3, 3, vec![0, 1], vec![1, 0], vec![1.0, 1.0], true)
            .expect("weights");
        assert_eq!(weights.isolated_nodes(), vec![2]);
    }
}
