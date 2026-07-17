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

