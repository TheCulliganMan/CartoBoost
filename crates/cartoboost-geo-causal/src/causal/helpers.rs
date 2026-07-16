fn unit_coordinates(panel: &GeoCausalPanel) -> BTreeMap<String, (f64, f64)> {
    let mut coords = BTreeMap::new();
    for row in panel.rows() {
        if let (Some(lat), Some(lon)) = (row.latitude, row.longitude) {
            coords.entry(row.unit_id.clone()).or_insert((lat, lon));
        }
    }
    coords
}

fn mean(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = mean(values.iter().copied());
    (values
        .iter()
        .map(|value| (value - avg).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

fn validate_representation_inputs(
    features: &[Vec<f64>],
    outcomes: &[f64],
    regions: &[String],
    heldout_region: &str,
) -> Result<()> {
    if features.is_empty() || features[0].is_empty() {
        return Err(GeoCausalError::InvalidInput(
            "features must be a non-empty matrix".to_string(),
        ));
    }
    if outcomes.len() != features.len() || regions.len() != features.len() {
        return Err(GeoCausalError::InvalidInput(
            "features, outcomes, and regions must have matching row counts".to_string(),
        ));
    }
    let dim = features[0].len();
    for row in features {
        if row.len() != dim || row.iter().any(|value| !value.is_finite()) {
            return Err(GeoCausalError::InvalidInput(
                "feature rows must be finite with fixed width".to_string(),
            ));
        }
    }
    if outcomes.iter().any(|value| !value.is_finite()) {
        return Err(GeoCausalError::InvalidInput(
            "outcomes must be finite".to_string(),
        ));
    }
    if !regions.iter().any(|region| region == heldout_region)
        || !regions.iter().any(|region| region != heldout_region)
    {
        return Err(GeoCausalError::InvalidInput(
            "heldout_region must have both held-out and training rows".to_string(),
        ));
    }
    Ok(())
}

fn column_mean(features: &[Vec<f64>]) -> Vec<f64> {
    let mut out = vec![0.0; features[0].len()];
    for row in features {
        for (idx, value) in row.iter().enumerate() {
            out[idx] += value / features.len() as f64;
        }
    }
    out
}

fn region_feature_means(
    features: &[Vec<f64>],
    regions: &[String],
    dim: usize,
) -> BTreeMap<String, Vec<f64>> {
    let mut sums: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (row, region) in features.iter().zip(regions) {
        let entry = sums.entry(region.clone()).or_insert(vec![0.0; dim]);
        for (idx, value) in row.iter().enumerate() {
            entry[idx] += value;
        }
        *counts.entry(region.clone()).or_insert(0) += 1;
    }
    sums.into_iter()
        .map(|(region, values)| {
            let count = counts[&region] as f64;
            (
                region,
                values.into_iter().map(|value| value / count).collect(),
            )
        })
        .collect()
}

fn ridge_fit_indexed(features: &[Vec<f64>], outcomes: &[f64], indices: &[usize]) -> Vec<f64> {
    let cols = features[0].len() + 1;
    let mut xtx = vec![vec![0.0; cols]; cols];
    let mut xty = vec![0.0; cols];
    for &idx in indices {
        let mut x = vec![1.0];
        x.extend_from_slice(&features[idx]);
        for r in 0..cols {
            xty[r] += x[r] * outcomes[idx];
            for c in 0..cols {
                xtx[r][c] += x[r] * x[c];
            }
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        row[idx] += 1.0e-6;
    }
    solve_linear(xtx, xty)
}

fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
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

fn indexed_rmse(
    features: &[Vec<f64>],
    outcomes: &[f64],
    indices: &[usize],
    weights: &[f64],
) -> f64 {
    indexed_mse(features, outcomes, indices, weights).sqrt()
}

fn indexed_mse(features: &[Vec<f64>], outcomes: &[f64], indices: &[usize], weights: &[f64]) -> f64 {
    indices
        .iter()
        .map(|&idx| {
            let pred = weights[0]
                + features[idx]
                    .iter()
                    .zip(&weights[1..])
                    .map(|(x, w)| x * w)
                    .sum::<f64>();
            let err = pred - outcomes[idx];
            err * err
        })
        .sum::<f64>()
        / indices.len() as f64
}

fn mean_region_distance(features: &[Vec<f64>], regions: &[String]) -> f64 {
    let means = region_feature_means(features, regions, features[0].len());
    let global = column_mean(features);
    means
        .values()
        .map(|row| {
            row.iter()
                .zip(&global)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum::<f64>()
        / means.len() as f64
}

fn mean_row_variation(features: &[Vec<f64>]) -> f64 {
    if features.len() < 2 {
        return 0.0;
    }
    features
        .windows(2)
        .map(|pair| {
            pair[0]
                .iter()
                .zip(&pair[1])
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
        })
        .sum::<f64>()
        / (features.len() - 1) as f64
}

