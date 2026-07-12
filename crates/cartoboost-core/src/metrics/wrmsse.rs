use crate::{CartoBoostError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct WrmsseSeries {
    pub id: String,
    pub train: Vec<f64>,
    pub actual: Vec<f64>,
    pub forecast: Vec<f64>,
    pub weight: f64,
}

impl WrmsseSeries {
    pub fn new(
        id: impl Into<String>,
        train: Vec<f64>,
        actual: Vec<f64>,
        forecast: Vec<f64>,
        weight: f64,
    ) -> Self {
        Self {
            id: id.into(),
            train,
            actual,
            forecast,
            weight,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct WrmsseSeriesScore {
    pub id: String,
    pub weight: f64,
    pub normalized_weight: f64,
    pub scale: f64,
    pub rmsse: f64,
    pub contribution: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WrmsseScore {
    pub score: f64,
    pub series: Vec<WrmsseSeriesScore>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LevelWrmsseScore {
    pub level: String,
    pub wrmsse: f64,
    pub level_weight: f64,
    pub contribution: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AggregateWrmsseScore {
    pub score: f64,
    pub levels: Vec<LevelWrmsseScore>,
}

pub fn wrmsse(series: &[WrmsseSeries], seasonal_period: usize) -> Result<WrmsseScore> {
    if series.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "WRMSSE requires at least one series".to_string(),
        ));
    }
    if seasonal_period == 0 {
        return Err(CartoBoostError::InvalidInput(
            "WRMSSE seasonal_period must be positive".to_string(),
        ));
    }
    let total_weight = series.iter().try_fold(0.0, |acc, row| {
        validate_series(row, seasonal_period)?;
        Ok::<f64, CartoBoostError>(acc + row.weight)
    })?;
    if total_weight <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "WRMSSE weights must sum to a positive value".to_string(),
        ));
    }

    let mut total = 0.0;
    let mut scores = Vec::with_capacity(series.len());
    for row in series {
        let scale = rmsse_scale(&row.train, seasonal_period)?;
        let mse = row
            .actual
            .iter()
            .zip(&row.forecast)
            .map(|(actual, forecast)| {
                let error = actual - forecast;
                error * error
            })
            .sum::<f64>()
            / row.actual.len() as f64;
        let rmsse = (mse / scale).sqrt();
        let normalized_weight = row.weight / total_weight;
        let contribution = normalized_weight * rmsse;
        total += contribution;
        scores.push(WrmsseSeriesScore {
            id: row.id.clone(),
            weight: row.weight,
            normalized_weight,
            scale,
            rmsse,
            contribution,
        });
    }
    Ok(WrmsseScore {
        score: total,
        series: scores,
    })
}

pub fn aggregate_equal_level_wrmsse(
    level_scores: &[(String, f64)],
) -> Result<AggregateWrmsseScore> {
    if level_scores.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "aggregate WRMSSE requires at least one hierarchy level".to_string(),
        ));
    }
    let mut score = 0.0;
    let level_weight = 1.0 / level_scores.len() as f64;
    let mut levels = Vec::with_capacity(level_scores.len());
    for (level, wrmsse) in level_scores {
        if level.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "hierarchy level names must be non-empty".to_string(),
            ));
        }
        if !wrmsse.is_finite() || *wrmsse < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "hierarchy level WRMSSE values must be finite and non-negative".to_string(),
            ));
        }
        let contribution = level_weight * wrmsse;
        score += contribution;
        levels.push(LevelWrmsseScore {
            level: level.clone(),
            wrmsse: *wrmsse,
            level_weight,
            contribution,
        });
    }
    Ok(AggregateWrmsseScore { score, levels })
}

pub fn ordered_nonnegative_weights(
    ids: &[String],
    raw_weights: &[(String, f64)],
) -> Result<Vec<(String, f64)>> {
    if ids.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "ordered weights require at least one id".to_string(),
        ));
    }
    let mut weights = std::collections::BTreeMap::new();
    for (id, weight) in raw_weights {
        if id.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "ordered weight ids must be non-empty".to_string(),
            ));
        }
        if !weight.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "ordered weights must be finite".to_string(),
            ));
        }
        weights.insert(id.clone(), *weight);
    }
    let positive_total = ids
        .iter()
        .map(|id| weights.get(id).copied().unwrap_or(0.0).max(0.0))
        .sum::<f64>();
    if positive_total <= 0.0 {
        return Ok(ids.iter().map(|id| (id.clone(), 1.0)).collect());
    }
    Ok(ids
        .iter()
        .map(|id| (id.clone(), weights.get(id).copied().unwrap_or(0.0).max(0.0)))
        .collect())
}

pub fn rmsse_scale(train: &[f64], seasonal_period: usize) -> Result<f64> {
    if seasonal_period == 0 {
        return Err(CartoBoostError::InvalidInput(
            "RMSSE seasonal_period must be positive".to_string(),
        ));
    }
    if train.len() <= seasonal_period {
        return Err(CartoBoostError::InvalidInput(
            "RMSSE training history must be longer than seasonal_period".to_string(),
        ));
    }
    if !train.iter().all(|value| value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "RMSSE training values must be finite".to_string(),
        ));
    }
    let scale = train
        .iter()
        .skip(seasonal_period)
        .zip(train)
        .map(|(current, lagged)| {
            let diff = current - lagged;
            diff * diff
        })
        .sum::<f64>()
        / (train.len() - seasonal_period) as f64;
    if scale <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "RMSSE scale is zero".to_string(),
        ));
    }
    Ok(scale)
}

fn validate_series(row: &WrmsseSeries, seasonal_period: usize) -> Result<()> {
    if row.id.trim().is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "WRMSSE series ids must be non-empty".to_string(),
        ));
    }
    if row.actual.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "WRMSSE actual values must contain at least one horizon".to_string(),
        ));
    }
    if row.actual.len() != row.forecast.len() {
        return Err(CartoBoostError::InvalidInput(format!(
            "WRMSSE actual and forecast lengths differ for series '{}'",
            row.id
        )));
    }
    if !row.actual.iter().all(|value| value.is_finite())
        || !row.forecast.iter().all(|value| value.is_finite())
    {
        return Err(CartoBoostError::InvalidInput(
            "WRMSSE actual and forecast values must be finite".to_string(),
        ));
    }
    if !row.weight.is_finite() || row.weight < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "WRMSSE weights must be finite and non-negative".to_string(),
        ));
    }
    rmsse_scale(&row.train, seasonal_period)?;
    Ok(())
}
