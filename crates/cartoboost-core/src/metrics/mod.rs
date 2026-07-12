use rayon::prelude::*;

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

pub fn volatility(pred: &[f64]) -> f64 {
    if pred.len() < 2 {
        return 0.0;
    }
    pred.par_windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum::<f64>()
        / (pred.len() - 1) as f64
}
