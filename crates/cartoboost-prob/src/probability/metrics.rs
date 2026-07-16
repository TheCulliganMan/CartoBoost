pub fn pinball_loss(actual: &[f64], prediction: &[f64], quantile: f64) -> Result<f64> {
    cartoboost_core::forecasting::pinball_loss(actual, prediction, quantile)
}

pub fn interval_coverage(actual: &[f64], lower: &[f64], upper: &[f64]) -> Result<f64> {
    cartoboost_core::forecasting::interval_coverage(actual, lower, upper)
}

pub fn mean_interval_width(lower: &[f64], upper: &[f64]) -> Result<f64> {
    cartoboost_core::forecasting::mean_interval_width(lower, upper)
}

pub fn interval_metrics(actual: &[f64], lower: &[f64], upper: &[f64]) -> Result<IntervalMetrics> {
    Ok(IntervalMetrics {
        coverage: interval_coverage(actual, lower, upper)?,
        mean_width: mean_interval_width(lower, upper)?,
    })
}

pub fn crps_approximation(
    actual: &[f64],
    quantiles: &[f64],
    predictions: &[Vec<f64>],
) -> Result<f64> {
    validate_quantile_rows(actual, quantiles, predictions)?;
    let mut total = 0.0;
    for (idx, row) in predictions.iter().enumerate() {
        for (&level, &prediction) in quantiles.iter().zip(row) {
            total += 2.0
                * cartoboost_core::forecasting::pinball_loss(&[actual[idx]], &[prediction], level)?;
        }
    }
    Ok(total / (actual.len() * quantiles.len()) as f64)
}

pub fn weighted_interval_score(
    actual: &[f64],
    median: &[f64],
    intervals: &[(f64, Vec<f64>, Vec<f64>)],
) -> Result<f64> {
    validate_same_non_empty(actual, median, "actual", "median")?;
    if intervals.is_empty() {
        return invalid("intervals must contain at least one central interval");
    }
    let mut score = 0.5
        * actual
            .iter()
            .zip(median)
            .map(|(&y, &m)| (y - m).abs())
            .sum::<f64>();
    let mut weight_sum = 0.5;
    for (alpha, lower, upper) in intervals {
        validate_alpha(*alpha)?;
        validate_same_non_empty(actual, lower, "actual", "lower")?;
        validate_same_non_empty(actual, upper, "actual", "upper")?;
        let weight = *alpha / 2.0;
        weight_sum += weight;
        for ((&y, &lo), &hi) in actual.iter().zip(lower).zip(upper) {
            if lo > hi {
                return invalid("lower bounds must be less than or equal to upper bounds");
            }
            let below = if y < lo { (lo - y) * 2.0 / alpha } else { 0.0 };
            let above = if y > hi { (y - hi) * 2.0 / alpha } else { 0.0 };
            score += weight * (hi - lo + below + above);
        }
    }
    Ok(score / (actual.len() as f64 * weight_sum))
}

pub fn pit_bins(
    actual: &[f64],
    quantiles: &[f64],
    predictions: &[Vec<f64>],
    bins: usize,
) -> Result<PitBins> {
    validate_quantile_rows(actual, quantiles, predictions)?;
    if bins == 0 {
        return invalid("bins must be positive");
    }
    let mut counts = vec![0usize; bins];
    for (&y, row) in actual.iter().zip(predictions) {
        let mut pit = 0.0;
        for (&level, &prediction) in quantiles.iter().zip(row) {
            if y >= prediction {
                pit = level;
            } else {
                break;
            }
        }
        let idx = ((pit * bins as f64).floor() as usize).min(bins - 1);
        counts[idx] += 1;
    }
    let edges = (0..=bins).map(|idx| idx as f64 / bins as f64).collect();
    Ok(PitBins { edges, counts })
}

