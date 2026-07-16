use crate::forecasting::{ForecastFrame, ForecastRow};
use crate::{CartoBoostError, Result};
use chrono::{Datelike, NaiveDateTime};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CalendarFeature {
    DayOfWeek,
    DayOfWeekSin,
    DayOfWeekCos,
    Month,
    MonthSin,
    MonthCos,
    Day,
    DaySin,
    DayCos,
    MonthStart,
    MonthMiddle,
    MonthEnd,
    DayOfYear,
    ElapsedIndex,
    ElapsedPhase(usize),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LagFeatureConfig {
    pub lags: Vec<usize>,
    pub rolling_mean_windows: Vec<usize>,
    #[serde(default)]
    pub partial_rolling_mean_windows: Vec<usize>,
    #[serde(default)]
    pub rolling_std_windows: Vec<usize>,
    #[serde(default)]
    pub rolling_min_windows: Vec<usize>,
    #[serde(default)]
    pub rolling_max_windows: Vec<usize>,
    #[serde(default)]
    pub ewm_alpha_percents: Vec<u8>,
    pub calendar_features: Vec<CalendarFeature>,
    #[serde(default)]
    pub difference_lags: Vec<usize>,
    #[serde(default)]
    pub rolling_trend_windows: Vec<usize>,
    #[serde(default)]
    pub covariate_features: Vec<String>,
    #[serde(default)]
    pub covariate_indicator_values: BTreeMap<String, Vec<f64>>,
    #[serde(default)]
    pub covariate_calendar_interactions: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LagFeatureRow {
    pub series_id: String,
    pub timestamp: NaiveDateTime,
    pub target: f64,
    pub features: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LagFeatureBuilder {
    config: LagFeatureConfig,
    feature_names: Vec<String>,
}

// Lag-feature stages share this module namespace.
include!("lag_features/configuration.rs");
include!("lag_features/builder.rs");
include!("lag_features/validation.rs");
include!("lag_features/calendar.rs");
include!("lag_features/tests.rs");
