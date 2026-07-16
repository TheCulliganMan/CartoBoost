fn solve_spd(mut a: Vec<Vec<f64>>, b: Vec<f64>) -> Result<Vec<f64>> {
    let mut jitter = 0.0;
    for attempt in 0..5 {
        if attempt > 0 {
            jitter = 10_f64.powi(attempt - 12);
            for (idx, row) in a.iter_mut().enumerate() {
                row[idx] += jitter;
            }
        }
        if let Some(chol) = cholesky(&a) {
            let y = forward_substitution(&chol, &b);
            return Ok(back_substitution_transpose(&chol, &y));
        }
    }
    Err(GeostatsError::LinearSolve(format!(
        "covariance matrix is not positive definite after jitter {jitter:e}"
    )))
}

fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum = (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                let value = a[i][i] - sum;
                if value <= 0.0 || !value.is_finite() {
                    return None;
                }
                l[i][j] = value.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let sum = (0..i).map(|j| l[i][j] * y[j]).sum::<f64>();
        y[i] = (b[i] - sum) / l[i][i];
    }
    y
}

fn back_substitution_transpose(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let sum = ((i + 1)..n).map(|j| l[j][i] * x[j]).sum::<f64>();
        x[i] = (y[i] - sum) / l[i][i];
    }
    x
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn checked_nonnegative(value: f64, scale: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(GeostatsError::LinearSolve(format!("{label} is not finite")));
    }
    let tolerance = 1.0e-10 * scale.abs().max(f64::MIN_POSITIVE);
    if value < -tolerance {
        return Err(GeostatsError::LinearSolve(format!(
            "{label} is materially negative ({value:e}, tolerance {tolerance:e})"
        )));
    }
    Ok(value.max(0.0))
}

