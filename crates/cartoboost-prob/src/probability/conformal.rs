pub fn split_conformal_residual_quantile(
    actual: &[f64],
    prediction: &[f64],
    alpha: f64,
    order: SplitOrder,
) -> Result<f64> {
    order.validate()?;
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    let mut residuals = actual
        .iter()
        .zip(prediction)
        .map(|(&y, &p)| (y - p).abs())
        .collect::<Vec<_>>();
    residuals.sort_by(f64::total_cmp);
    let rank = (((residuals.len() + 1) as f64) * (1.0 - alpha)).ceil() as usize;
    Ok(residuals[rank.saturating_sub(1).min(residuals.len() - 1)])
}

pub fn weighted_conformal_residual_quantile(
    actual: &[f64],
    prediction: &[f64],
    weights: &[f64],
    alpha: f64,
    order: SplitOrder,
) -> Result<f64> {
    order.validate()?;
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    validate_same_non_empty(actual, weights, "actual", "weights")?;
    if weights.iter().any(|w| *w <= 0.0) {
        return invalid("weights must be positive");
    }
    let mut pairs = actual
        .iter()
        .zip(prediction)
        .zip(weights)
        .map(|((&y, &p), &w)| ((y - p).abs(), w))
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total_weight = pairs.iter().map(|(_, w)| *w).sum::<f64>();
    let threshold = (1.0 - alpha) * total_weight;
    let mut cumulative = 0.0;
    for (residual, weight) in pairs {
        cumulative += weight;
        if cumulative >= threshold {
            return Ok(residual);
        }
    }
    invalid("weighted conformal calibration failed")
}

pub fn group_conformal_residual_quantiles(
    actual: &[f64],
    prediction: &[f64],
    groups: &[String],
    alpha: f64,
    order: SplitOrder,
) -> Result<BTreeMap<String, f64>> {
    order.validate()?;
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    if groups.len() != actual.len() {
        return invalid("groups length must match actual length");
    }
    let mut grouped: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for ((&y, &p), group) in actual.iter().zip(prediction).zip(groups) {
        let entry = grouped.entry(group.clone()).or_default();
        entry.0.push(y);
        entry.1.push(p);
    }
    grouped
        .into_iter()
        .map(|(group, (y, p))| {
            split_conformal_residual_quantile(&y, &p, alpha, order).map(|q| (group, q))
        })
        .collect()
}

pub fn spatial_block_conformal_residual_quantiles(
    actual: &[f64],
    prediction: &[f64],
    block_ids: &[String],
    alpha: f64,
    order: SplitOrder,
) -> Result<BTreeMap<String, f64>> {
    group_conformal_residual_quantiles(actual, prediction, block_ids, alpha, order)
}

#[allow(clippy::too_many_arguments)]
pub fn nearest_calibration_residual_quantiles(
    actual: &[f64],
    prediction: &[f64],
    calibration_x: &[f64],
    calibration_y: &[f64],
    query_x: &[f64],
    query_y: &[f64],
    neighbor_count: usize,
    alpha: f64,
    order: SplitOrder,
) -> Result<Vec<f64>> {
    order.validate()?;
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    validate_same_non_empty(
        calibration_x,
        calibration_y,
        "calibration_x",
        "calibration_y",
    )?;
    validate_same_non_empty(calibration_x, actual, "calibration_x", "actual")?;
    validate_same_non_empty(query_x, query_y, "query_x", "query_y")?;
    if neighbor_count == 0 {
        return invalid("neighbor_count must be positive");
    }
    let residuals = actual
        .iter()
        .zip(prediction)
        .map(|(&y, &p)| (y - p).abs())
        .collect::<Vec<_>>();
    query_x
        .iter()
        .zip(query_y)
        .map(|(&x, &y)| {
            let mut distances = calibration_x
                .iter()
                .zip(calibration_y)
                .zip(&residuals)
                .map(|((&cx, &cy), &residual)| {
                    let dx = cx - x;
                    let dy = cy - y;
                    (dx * dx + dy * dy, residual)
                })
                .collect::<Vec<_>>();
            distances.sort_by(|a, b| a.0.total_cmp(&b.0));
            let local = distances
                .iter()
                .take(neighbor_count.min(distances.len()))
                .map(|(_, residual)| *residual)
                .collect::<Vec<_>>();
            conformal_quantile(&local, alpha)
        })
        .collect()
}

pub fn rolling_origin_conformal_residual_quantiles(
    actual: &[f64],
    prediction: &[f64],
    cutoffs: &[usize],
    alpha: f64,
) -> Result<Vec<f64>> {
    validate_alpha(alpha)?;
    validate_same_non_empty(actual, prediction, "actual", "prediction")?;
    if cutoffs.is_empty() {
        return invalid("cutoffs must contain at least one cutoff");
    }
    let mut result = Vec::with_capacity(cutoffs.len());
    for &cutoff in cutoffs {
        if cutoff == 0 || cutoff > actual.len() {
            return invalid("each cutoff must be inside the calibration history");
        }
        let order = SplitOrder {
            train_end_exclusive: 1,
            calibration_start: 1,
            calibration_end_exclusive: cutoff,
            test_start: cutoff,
        };
        result.push(split_conformal_residual_quantile(
            &actual[..cutoff],
            &prediction[..cutoff],
            alpha,
            order,
        )?);
    }
    Ok(result)
}

pub fn benchmark_calibration_report_fields(
    actual: &[f64],
    lower: &[f64],
    upper: &[f64],
    horizons: &[usize],
    spatial_blocks: &[String],
    residual_morans_i_after_calibration: Option<f64>,
) -> Result<BenchmarkCalibrationReportFields> {
    validate_same_non_empty(actual, lower, "actual", "lower")?;
    validate_same_non_empty(actual, upper, "actual", "upper")?;
    if horizons.len() != actual.len() {
        return invalid("horizons length must match actual length");
    }
    if spatial_blocks.len() != actual.len() {
        return invalid("spatial_blocks length must match actual length");
    }
    if let Some(value) = residual_morans_i_after_calibration {
        if !value.is_finite() {
            return invalid("residual_morans_i_after_calibration must be finite when provided");
        }
    }
    let mut coverage_by_horizon = BTreeMap::new();
    let mut width_by_horizon = BTreeMap::new();
    for horizon in horizons
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let mut y = Vec::new();
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for idx in 0..actual.len() {
            if horizons[idx] == horizon {
                y.push(actual[idx]);
                lo.push(lower[idx]);
                hi.push(upper[idx]);
            }
        }
        coverage_by_horizon.insert(horizon, interval_coverage(&y, &lo, &hi)?);
        width_by_horizon.insert(horizon, mean_interval_width(&lo, &hi)?);
    }
    let mut coverage_by_spatial_block = BTreeMap::new();
    for block in spatial_blocks
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>()
    {
        let mut y = Vec::new();
        let mut lo = Vec::new();
        let mut hi = Vec::new();
        for idx in 0..actual.len() {
            if spatial_blocks[idx] == block {
                y.push(actual[idx]);
                lo.push(lower[idx]);
                hi.push(upper[idx]);
            }
        }
        coverage_by_spatial_block.insert(block, interval_coverage(&y, &lo, &hi)?);
    }
    Ok(BenchmarkCalibrationReportFields {
        coverage_by_horizon,
        coverage_by_spatial_block,
        width_by_horizon,
        residual_morans_i_after_calibration,
    })
}

