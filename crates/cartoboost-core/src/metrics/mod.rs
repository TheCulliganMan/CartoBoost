use crate::{CartoBoostError, Result};
use rayon::prelude::*;
use std::collections::HashMap;

pub mod pinball;
pub mod rank_portfolio;
pub mod rps;
pub mod wrmsse;
pub use pinball::pinball_loss;
pub use rank_portfolio::{
    calibrated_rank_bucket_probabilities, evaluate_rank_portfolio_metrics,
    extreme_portfolio_decisions, portfolio_summary, rank_buckets, rank_hit_rates,
    rank_portfolio_combined_score, rank_portfolio_decision_loss, rank_portfolio_summary,
    rank_probability_calibration, rank_scored_assets, PortfolioAsset, PortfolioDecision,
    PortfolioDecisionRow, PortfolioSide, PortfolioSummary, RankBucketPrediction,
    RankHitRateSummary, RankPortfolioMetricSummary, RankPortfolioSummary,
    RankProbabilityCalibration, RankProbabilityCalibrationMetadata, RankScoredAsset,
};
pub use rps::rank_probability_score;
pub use wrmsse::{
    aggregate_equal_level_wrmsse, ordered_nonnegative_weights, rmsse_scale, wrmsse,
    AggregateWrmsseScore, LevelWrmsseScore, WrmsseScore, WrmsseSeries, WrmsseSeriesScore,
};

pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .par_iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / y_true.len().max(1) as f64
}

pub fn rmse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    (y_true
        .par_iter()
        .zip(y_pred)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / y_true.len().max(1) as f64)
        .sqrt()
}

/// Coefficient of determination with explicit finite, aligned input validation.
pub fn r_squared(y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
    if y_true.is_empty()
        || y_true.len() != y_pred.len()
        || y_true.iter().chain(y_pred).any(|v| !v.is_finite())
    {
        return Err(CartoBoostError::InvalidInput(
            "R-squared requires aligned nonempty finite vectors".into(),
        ));
    }
    // Summing a constant vector such as [0.1, 0.1, 0.1] can round its mean away
    // from the constant, manufacturing variance in the subtraction below.
    if y_true.iter().all(|value| *value == y_true[0]) {
        return Ok(if y_true == y_pred { 1.0 } else { 0.0 });
    }
    let mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
    let mut residual = 0.;
    let mut total = 0.;
    for (actual, predicted) in y_true.iter().zip(y_pred) {
        residual += (actual - predicted).powi(2);
        total += (actual - mean).powi(2);
    }
    if !residual.is_finite() || !total.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "R-squared calculation overflow".into(),
        ));
    }
    let score = 1. - residual / total;
    if !score.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "R-squared calculation overflow".into(),
        ));
    }
    Ok(score)
}

pub fn volatility(pred: &[f64]) -> f64 {
    if pred.len() < 2 {
        return 0.0;
    }
    pred.par_windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum::<f64>()
        / (pred.len() - 1) as f64
}

/// Binary ROC-AUC, awarding half credit to tied scores.
///
/// Labels above 0.5 are positive. Without weights, a missing class returns zero
/// for graph evaluation compatibility. Weighted inputs require positive weight
/// in both classes. Sorting score ties avoids enumerating positive/negative pairs.
pub fn binary_roc_auc(labels: &[f64], scores: &[f64], weights: Option<&[f64]>) -> Result<f64> {
    validate_binary_metric_inputs(labels, scores)?;
    let mut scales = [1.0_f64; 2];
    if let Some(weights) = weights {
        if weights.len() != labels.len()
            || weights
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(CartoBoostError::InvalidInput(
                "sample_weight must align with scores and contain finite non-negative values"
                    .to_string(),
            ));
        }
        scales = [0.0; 2];
        for (&label, &weight) in labels.iter().zip(weights) {
            let class = usize::from(label > 0.5);
            scales[class] = scales[class].max(weight);
        }
        if scales.contains(&0.0) {
            return Err(CartoBoostError::InvalidInput(
                "weighted ROC-AUC requires positive weight for both labels".to_string(),
            ));
        }
    } else if !labels.iter().any(|label| *label > 0.5) || !labels.iter().any(|label| *label <= 0.5)
    {
        return Ok(0.0);
    }
    // Separate class scaling preserves the statistic and avoids overflow even
    // when positive and negative weights differ by hundreds of orders of magnitude.
    let weight_at = |row: usize| {
        weights.map_or(1.0, |weights| {
            weights[row] / scales[usize::from(labels[row] > 0.5)]
        })
    };
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| scores[left].total_cmp(&scores[right]));
    let mut positive_total = 0.0;
    let mut negative_total = 0.0;
    let mut concordant = 0.0;
    let mut start = 0;
    while start < order.len() {
        let mut stop = start + 1;
        while stop < order.len() && scores[order[stop]] == scores[order[start]] {
            stop += 1;
        }
        let mut positive = 0.0;
        let mut negative = 0.0;
        for &row in &order[start..stop] {
            if labels[row] > 0.5 {
                positive += weight_at(row);
            } else {
                negative += weight_at(row);
            }
        }
        concordant += positive * (negative_total + 0.5 * negative);
        positive_total += positive;
        negative_total += negative;
        start = stop;
    }
    if positive_total == 0.0 || negative_total == 0.0 {
        return Ok(0.0);
    }
    Ok((concordant / (positive_total * negative_total)).clamp(0.0, 1.0))
}

/// Graph MRR over arbitrary, possibly interleaved query IDs. Ties retain input
/// order and groups without a positive label are excluded from the mean.
pub fn graph_mean_reciprocal_rank(
    labels: &[f64],
    scores: &[f64],
    query_ids: &[usize],
) -> Result<f64> {
    validate_binary_metric_inputs(labels, scores)?;
    if query_ids.len() != labels.len() {
        return Err(CartoBoostError::InvalidInput(
            "labels, scores, and query_ids must align".to_string(),
        ));
    }
    let mut groups = HashMap::new();
    let mut best: Vec<Option<usize>> = Vec::new();
    for row in 0..labels.len() {
        let next = groups.len();
        let group = *groups.entry(query_ids[row]).or_insert(next);
        if group == best.len() {
            best.push(None);
        }
        if labels[row] > 0.5 && best[group].is_none_or(|previous| scores[row] > scores[previous]) {
            best[group] = Some(row);
        }
    }
    let mut ranks = vec![1_usize; best.len()];
    for row in 0..scores.len() {
        let group = groups[&query_ids[row]];
        if let Some(positive) = best[group] {
            if scores[row] > scores[positive] || (scores[row] == scores[positive] && row < positive)
            {
                ranks[group] += 1;
            }
        }
    }
    let mut count = 0;
    let mut total = 0.0;
    for (positive, rank) in best.iter().zip(ranks) {
        if positive.is_some() {
            count += 1;
            total += 1.0 / rank as f64;
        }
    }
    Ok(if count == 0 {
        0.0
    } else {
        total / count as f64
    })
}

fn validate_binary_metric_inputs(labels: &[f64], scores: &[f64]) -> Result<()> {
    if labels.len() != scores.len() {
        return Err(CartoBoostError::InvalidInput(
            "labels and scores must have equal length".to_string(),
        ));
    }
    if labels.iter().chain(scores).any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "labels and scores must contain only finite values".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod binary_metric_tests {
    use super::*;

    #[test]
    fn sorted_auc_matches_pairwise_reference_with_ties_and_zero_weights() {
        for seed in 0..20 {
            let labels = (0..63)
                .map(|row| if (row + seed) % 3 == 0 { 1.0 } else { 0.0 })
                .collect::<Vec<_>>();
            let scores = (0..63)
                .map(|row| ((row * 7 + seed) % 9) as f64)
                .collect::<Vec<_>>();
            let weights = (0..63)
                .map(|row| ((row * 17 + seed) % 11) as f64 / 4.0)
                .collect::<Vec<_>>();
            for weighted in [false, true] {
                let weight = |row| if weighted { weights[row] } else { 1.0 };
                let mut wins = 0.0;
                let mut total = 0.0;
                for positive in (0..labels.len()).filter(|&row| labels[row] > 0.5) {
                    for negative in (0..labels.len()).filter(|&row| labels[row] <= 0.5) {
                        let pair_weight = weight(positive) * weight(negative);
                        total += pair_weight;
                        wins += pair_weight
                            * if scores[positive] > scores[negative] {
                                1.0
                            } else if scores[positive] == scores[negative] {
                                0.5
                            } else {
                                0.0
                            };
                    }
                }
                assert!(
                    (binary_roc_auc(&labels, &scores, weighted.then_some(weights.as_slice()))
                        .unwrap()
                        - wins / total)
                        .abs()
                        < 1e-12
                );
            }
        }
    }

    #[test]
    fn auc_preserves_extreme_weights_missing_classes_and_signed_zero_ties() {
        assert_eq!(
            binary_roc_auc(&[0.0, 1.0], &[-0.0, 0.0], None).unwrap(),
            0.5
        );
        assert_eq!(
            binary_roc_auc(&[0.0, 1.0], &[0.0, 1.0], Some(&[1e-308, 1e308])).unwrap(),
            1.0
        );
        assert_eq!(
            binary_roc_auc(&[0.0, 1.0], &[0.0, 0.0], Some(&[f64::MAX, f64::MAX])).unwrap(),
            0.5
        );
        assert_eq!(binary_roc_auc(&[], &[], None).unwrap(), 0.0);
        assert_eq!(binary_roc_auc(&[1.0], &[1.0], None).unwrap(), 0.0);
        assert!(binary_roc_auc(&[0.0, 1.0], &[0.0, 1.0], Some(&[0.0, 1.0])).is_err());
        assert!(binary_roc_auc(&[0.0], &[f64::NAN], None).is_err());
        assert!(binary_roc_auc(&[0.0, 1.0], &[0.0, 1.0], Some(&[-1.0, 1.0])).is_err());
        assert!(binary_roc_auc(&[0.0], &[], None).is_err());
    }

    #[test]
    fn graph_mrr_preserves_stable_ties_and_skips_empty_relevance_groups() {
        assert_eq!(
            graph_mean_reciprocal_rank(
                &[0.0, 0.0, 1.0, 1.0, 0.0],
                &[2.0, 9.0, 2.0, 1.0, 0.0],
                &[3, usize::MAX, 3, 7, usize::MAX]
            )
            .unwrap(),
            0.75
        );
        assert_eq!(graph_mean_reciprocal_rank(&[], &[], &[]).unwrap(), 0.0);
        assert_eq!(
            graph_mean_reciprocal_rank(&[0.0], &[1.0], &[0]).unwrap(),
            0.0
        );
        assert!(graph_mean_reciprocal_rank(&[1.0], &[f64::INFINITY], &[0]).is_err());
        assert!(graph_mean_reciprocal_rank(&[1.0], &[1.0], &[]).is_err());
    }
}
