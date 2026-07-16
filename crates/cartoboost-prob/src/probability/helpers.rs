fn validate_quantile_rows(
    actual: &[f64],
    quantiles: &[f64],
    predictions: &[Vec<f64>],
) -> Result<()> {
    if actual.is_empty() {
        return invalid("actual must contain at least one value");
    }
    if actual.len() != predictions.len() {
        return invalid("actual and predictions must have the same row count");
    }
    for &q in quantiles {
        validate_alpha(q)?;
    }
    for row in predictions {
        if row.len() != quantiles.len() {
            return invalid("each prediction row must match quantiles length");
        }
        validate_finite(row, "prediction row")?;
    }
    validate_finite(actual, "actual")?;
    Ok(())
}

fn conformal_quantile(residuals: &[f64], alpha: f64) -> Result<f64> {
    validate_alpha(alpha)?;
    validate_finite(residuals, "residuals")?;
    if residuals.is_empty() {
        return invalid("residuals must contain at least one value");
    }
    let mut sorted = residuals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank = (((sorted.len() + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    Ok(sorted[rank.saturating_sub(1).min(sorted.len() - 1)])
}

fn validate_same_non_empty(
    left: &[f64],
    right: &[f64],
    left_name: &str,
    right_name: &str,
) -> Result<()> {
    validate_finite(left, left_name)?;
    validate_finite(right, right_name)?;
    if left.len() != right.len() {
        return invalid(&format!(
            "{left_name} and {right_name} must have the same length"
        ));
    }
    if left.is_empty() {
        return invalid(&format!(
            "{left_name} and {right_name} must contain at least one value"
        ));
    }
    Ok(())
}

fn validate_same_non_empty_matrix(
    matrix: &[Vec<f64>],
    values: &[f64],
    matrix_name: &str,
    values_name: &str,
) -> Result<()> {
    if matrix.is_empty() || values.is_empty() {
        return invalid(&format!(
            "{matrix_name} and {values_name} must contain at least one row"
        ));
    }
    if matrix.len() != values.len() {
        return invalid(&format!(
            "{matrix_name} and {values_name} must have the same row count"
        ));
    }
    let cols = matrix[0].len();
    if cols == 0 || matrix.iter().any(|row| row.len() != cols) {
        return invalid(&format!("{matrix_name} must have a fixed positive width"));
    }
    for row in matrix {
        validate_finite(row, matrix_name)?;
    }
    validate_finite(values, values_name)?;
    Ok(())
}

fn validate_panel(panel: &[Vec<f64>], name: &str) -> Result<()> {
    if panel.is_empty() {
        return invalid(&format!("{name} must contain at least one horizon row"));
    }
    let cols = panel[0].len();
    if cols == 0 {
        return invalid(&format!("{name} must contain at least one node column"));
    }
    for row in panel {
        if row.len() != cols {
            return invalid(&format!("{name} must have a fixed node width"));
        }
        validate_finite(row, name)?;
    }
    Ok(())
}

fn validate_edges(edges: &[DiffusionEdge], node_count: usize) -> Result<()> {
    for edge in edges {
        if edge.source >= node_count || edge.target >= node_count {
            return invalid("edge source and target must reference point_forecast columns");
        }
        if !edge.weight.is_finite() {
            return invalid("edge weights must be finite");
        }
    }
    Ok(())
}

fn validate_quantile_grid(quantiles: &[f64]) -> Result<()> {
    if quantiles.is_empty() {
        return invalid("quantiles must contain at least one value");
    }
    for pair in quantiles.windows(2) {
        if pair[0] >= pair[1] {
            return invalid("quantiles must be strictly increasing");
        }
    }
    for &q in quantiles {
        validate_alpha(q)?;
    }
    Ok(())
}

fn ridge_fit(features: &[Vec<f64>], target: &[f64], ridge: f64) -> Vec<f64> {
    let cols = features[0].len() + 1;
    let mut xtx = vec![vec![0.0; cols]; cols];
    let mut xty = vec![0.0; cols];
    for (row, &y) in features.iter().zip(target) {
        let mut x = vec![1.0];
        x.extend_from_slice(row);
        for r in 0..cols {
            xty[r] += x[r] * y;
            for c in 0..cols {
                xtx[r][c] += x[r] * x[c];
            }
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        row[idx] += ridge.max(1.0e-12);
    }
    solve_linear_system(xtx, xty)
}

fn predict_linear(features: &[Vec<f64>], weights: &[f64]) -> Result<Vec<f64>> {
    if weights.is_empty() {
        return invalid("linear weights must be non-empty");
    }
    let expected_width = weights.len() - 1;
    if features.iter().any(|row| row.len() != expected_width) {
        return invalid("feature width must match fitted flow artifact");
    }
    Ok(features
        .iter()
        .map(|row| {
            weights[0]
                + row
                    .iter()
                    .zip(&weights[1..])
                    .map(|(x, w)| x * w)
                    .sum::<f64>()
        })
        .collect())
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if best != pivot {
            matrix.swap(pivot, best);
            rhs.swap(pivot, best);
        }
        let diag = matrix[pivot][pivot];
        if diag.abs() < 1.0e-12 {
            continue;
        }
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= diag;
        }
        rhs[pivot] /= diag;
        let pivot_row = matrix[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for col in pivot..n {
                matrix[row][col] -= factor * pivot_row[col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    rhs
}

fn deterministic_standard_sample(row: usize, sample: usize) -> f64 {
    let value = ((row as u64 + 1) * 1_103_515_245 + (sample as u64 + 17) * 12_345) % 10_000;
    (value as f64 / 10_000.0 - 0.5) * 2.0
}

fn diffuse_residual_field(
    residual_field: &[Vec<f64>],
    edges: &[DiffusionEdge],
    node_count: usize,
) -> Vec<Vec<f64>> {
    let mut next = residual_field.to_vec();
    let mut inbound_weight = vec![0.0; node_count];
    for edge in edges {
        inbound_weight[edge.target] += edge.weight.abs();
    }
    for (t, row) in residual_field.iter().enumerate() {
        for edge in edges {
            let denom = inbound_weight[edge.target].max(1.0);
            next[t][edge.target] += 0.5 * edge.weight * row[edge.source] / denom;
        }
    }
    next
}

fn scenario_panel_mean(scenarios: &[Vec<Vec<f64>>], horizon: usize, nodes: usize) -> Vec<Vec<f64>> {
    let mut mean = vec![vec![0.0; nodes]; horizon];
    for scenario in scenarios {
        for t in 0..horizon {
            for node in 0..nodes {
                mean[t][node] += scenario[t][node] / scenarios.len() as f64;
            }
        }
    }
    mean
}

fn scenario_panel_variance(
    scenarios: &[Vec<Vec<f64>>],
    mean: &[Vec<f64>],
    horizon: usize,
    nodes: usize,
) -> Vec<Vec<f64>> {
    let mut variance = vec![vec![0.0; nodes]; horizon];
    for scenario in scenarios {
        for t in 0..horizon {
            for node in 0..nodes {
                let delta = scenario[t][node] - mean[t][node];
                variance[t][node] += delta * delta / scenarios.len() as f64;
            }
        }
    }
    variance
}

fn mean_abs_panel_delta(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
    let count = a.iter().map(Vec::len).sum::<usize>().max(1);
    a.iter()
        .zip(b)
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .map(|(&x, &y)| (x - y).abs())
                .sum::<f64>()
        })
        .sum::<f64>()
        / count as f64
}

fn scenario_spatial_correlation(panel: &[Vec<f64>], edges: &[DiffusionEdge]) -> f64 {
    if edges.is_empty() {
        return 0.0;
    }
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for row in panel {
        let mean = row.iter().sum::<f64>() / row.len() as f64;
        let centered = row.iter().map(|value| value - mean).collect::<Vec<_>>();
        let variance = centered.iter().map(|value| value * value).sum::<f64>();
        if variance <= 1.0e-12 {
            continue;
        }
        for edge in edges {
            numerator += edge.weight * centered[edge.source] * centered[edge.target];
            denominator += edge.weight.abs() * variance;
        }
    }
    if denominator <= 1.0e-12 {
        0.0
    } else {
        (numerator / denominator).clamp(-1.0, 1.0)
    }
}

fn normal_quantile_proxy(q: f64) -> f64 {
    // Smooth symmetric approximation sufficient for deterministic quantile surfaces.
    ((q / (1.0 - q)).ln() * 0.5513).clamp(-4.0, 4.0)
}

fn gaussian_log_likelihood(y: f64, loc: f64, scale: f64) -> f64 {
    let var = scale * scale;
    -0.5 * (((y - loc) * (y - loc)) / var + var.ln() + (2.0 * std::f64::consts::PI).ln())
}

fn quantile_mean(rows: &[Vec<f64>], idx: usize) -> Result<f64> {
    if rows.is_empty() || rows.iter().any(|row| idx >= row.len()) {
        return invalid("quantile index is out of bounds");
    }
    Ok(rows.iter().map(|row| row[idx]).sum::<f64>() / rows.len() as f64)
}

fn joint_path_calibration(actual: &[f64], paths: &[Vec<f64>]) -> f64 {
    if paths.is_empty() {
        return 0.0;
    }
    let actual_sum = actual.iter().sum::<f64>();
    let covered = paths
        .iter()
        .filter(|path| path.iter().sum::<f64>() >= actual_sum)
        .count();
    covered as f64 / paths.len() as f64
}

fn tail_event_calibration(actual: &[f64], upper: &[f64]) -> f64 {
    actual.iter().zip(upper).filter(|(y, hi)| y > hi).count() as f64 / actual.len() as f64
}

fn validate_finite(values: &[f64], name: &str) -> Result<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return invalid(&format!("{name} must contain only finite values"));
    }
    Ok(())
}

fn validate_alpha(alpha: f64) -> Result<()> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return invalid("alpha must be finite and in (0, 1)");
    }
    Ok(())
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(CartoBoostError::InvalidInput(message.to_string()))
}

