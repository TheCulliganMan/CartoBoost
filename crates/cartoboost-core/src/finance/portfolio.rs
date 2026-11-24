use crate::{CartoBoostError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioSignal {
    pub asset_id: String,
    pub expected_return: f64,
}

impl PortfolioSignal {
    pub fn new(asset_id: impl Into<String>, expected_return: f64) -> Result<Self> {
        let asset_id = asset_id.into();
        if asset_id.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "asset_id must not be empty".to_string(),
            ));
        }
        if !expected_return.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "expected_return must be finite".to_string(),
            ));
        }
        Ok(Self {
            asset_id,
            expected_return,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PortfolioPosition {
    pub asset_id: String,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Portfolio {
    pub positions: Vec<PortfolioPosition>,
}

impl Portfolio {
    pub fn deterministic_long_short(
        signals: &[PortfolioSignal],
        long_count: usize,
        short_count: usize,
    ) -> Result<Self> {
        if signals.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "signals must contain at least one asset".to_string(),
            ));
        }
        if long_count == 0 && short_count == 0 {
            return Err(CartoBoostError::InvalidInput(
                "at least one long or short position is required".to_string(),
            ));
        }
        if long_count > signals.len() || short_count > signals.len() - long_count {
            return Err(CartoBoostError::InvalidInput(
                "long_count plus short_count must not exceed asset count".to_string(),
            ));
        }
        if signals
            .iter()
            .any(|signal| signal.asset_id.trim().is_empty() || !signal.expected_return.is_finite())
        {
            return Err(CartoBoostError::InvalidInput(
                "signals must have non-empty asset ids and finite expected returns".to_string(),
            ));
        }
        let mut ranked = signals.iter().collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .expected_return
                .total_cmp(&left.expected_return)
                .then_with(|| left.asset_id.cmp(&right.asset_id))
        });

        let long_weight = if long_count == 0 {
            0.0
        } else {
            1.0 / long_count as f64
        };
        let short_weight = if short_count == 0 {
            0.0
        } else {
            -1.0 / short_count as f64
        };
        let mut positions = Vec::with_capacity(long_count + short_count);
        positions.extend(
            ranked
                .iter()
                .take(long_count)
                .map(|signal| PortfolioPosition {
                    asset_id: signal.asset_id.clone(),
                    weight: long_weight,
                }),
        );
        positions.extend(
            ranked
                .iter()
                .rev()
                .take(short_count)
                .map(|signal| PortfolioPosition {
                    asset_id: signal.asset_id.clone(),
                    weight: short_weight,
                }),
        );
        positions.sort_by(|left, right| left.asset_id.cmp(&right.asset_id));
        Ok(Self { positions })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portfolio_rejects_overflowing_counts_and_invalid_public_signals() {
        let signal = PortfolioSignal::new("PU100", 1.0).expect("signal");
        assert!(Portfolio::deterministic_long_short(&[signal], usize::MAX, 1).is_err());
        assert!(Portfolio::deterministic_long_short(
            &[PortfolioSignal {
                asset_id: "PU100".into(),
                expected_return: f64::NAN
            }],
            1,
            0,
        )
        .is_err());
    }

    #[test]
    fn portfolio_ranking_keeps_deterministic_ties_and_weights() {
        let signals = [("PU102", 3.0), ("PU100", 3.0), ("PU101", 1.0)]
            .into_iter()
            .map(|(id, value)| PortfolioSignal::new(id, value).expect("signal"))
            .collect::<Vec<_>>();
        let portfolio = Portfolio::deterministic_long_short(&signals, 1, 1).expect("portfolio");
        assert_eq!(
            portfolio.positions,
            vec![
                PortfolioPosition {
                    asset_id: "PU100".into(),
                    weight: 1.0
                },
                PortfolioPosition {
                    asset_id: "PU101".into(),
                    weight: -1.0
                },
            ]
        );
    }
}
