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

