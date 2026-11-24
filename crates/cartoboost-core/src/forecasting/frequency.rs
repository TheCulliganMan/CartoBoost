use crate::{CartoBoostError, Result};
use chrono::{DateTime, Duration, NaiveDate, NaiveDateTime};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastFrequency {
    Hourly,
    Daily,
    Weekly,
}

impl ForecastFrequency {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "h" | "H" | "hour" | "hourly" => Ok(Self::Hourly),
            "d" | "D" | "day" | "daily" => Ok(Self::Daily),
            "w" | "W" | "week" | "weekly" => Ok(Self::Weekly),
            other => Err(CartoBoostError::InvalidInput(format!(
                "unsupported forecast frequency {other:?}; expected H, D, or W"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Hourly => "hourly",
            Self::Daily => "daily",
            Self::Weekly => "weekly",
        }
    }

    pub fn step(self) -> Duration {
        match self {
            Self::Hourly => Duration::hours(1),
            Self::Daily => Duration::days(1),
            Self::Weekly => Duration::weeks(1),
        }
    }

    pub fn advance(self, timestamp: NaiveDateTime, steps: usize) -> Result<NaiveDateTime> {
        let step_count = i32::try_from(steps).map_err(|_| {
            CartoBoostError::InvalidInput("forecast horizon is too large".to_string())
        })?;
        timestamp
            .checked_add_signed(self.step() * step_count)
            .ok_or_else(|| CartoBoostError::InvalidInput("forecast timestamp overflow".to_string()))
    }
}

pub fn parse_forecast_timestamp(value: &str) -> Result<NaiveDateTime> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecast timestamp must not be empty".to_string(),
        ));
    }
    for format in [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
    ] {
        if let Ok(timestamp) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(timestamp);
        }
    }
    if let Ok(date) = NaiveDate::parse_from_str(trimmed, "%Y-%m-%d") {
        return date.and_hms_opt(0, 0, 0).ok_or_else(|| {
            CartoBoostError::InvalidInput(format!("invalid forecast timestamp {value:?}"))
        });
    }
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(trimmed) {
        return Ok(timestamp.naive_utc());
    }
    // Keep the existing common formats on their fast path. Naive ISO strings
    // from dataframe timestamps can also carry subsecond precision.
    for format in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(timestamp) = NaiveDateTime::parse_from_str(trimmed, format) {
            return Ok(timestamp);
        }
    }
    Err(CartoBoostError::InvalidInput(format!(
        "forecast timestamp {value:?} is not parseable"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Timelike;

    #[test]
    fn parses_subsecond_naive_and_offset_timestamps_without_losing_precision() {
        for (text, nanos) in [
            ("2026-01-01T00:00:00.125000", 125_000_000),
            ("2026-01-01 00:00:00.123456789", 123_456_789),
            (" 2026-01-01T00:00:00.000000001 ", 1),
            ("2026-01-01T01:00:00.123456789+01:00", 123_456_789),
        ] {
            let timestamp = parse_forecast_timestamp(text).unwrap();
            assert_eq!(
                timestamp.date(),
                NaiveDate::from_ymd_opt(2026, 1, 1).unwrap()
            );
            assert_eq!(timestamp.hour(), 0);
            assert_eq!(timestamp.nanosecond(), nanos);
            let serialized = timestamp.format("%Y-%m-%dT%H:%M:%S%.f").to_string();
            assert_eq!(parse_forecast_timestamp(&serialized).unwrap(), timestamp);
        }
    }

    #[test]
    fn preserves_existing_timestamp_forms_and_rejects_invalid_subseconds() {
        let expected = NaiveDate::from_ymd_opt(2026, 1, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap();
        for text in [
            "2026-01-01",
            "2026-01-01 00:00",
            "2026-01-01T00:00",
            "2026-01-01 00:00:00",
            "2026-01-01T00:00:00",
            "2026-01-01T00:00:00Z",
        ] {
            assert_eq!(parse_forecast_timestamp(text).unwrap(), expected);
        }
        for text in ["", "2026-02-30T00:00:00.125", "2026-01-01T00:00:00.xyz"] {
            assert!(parse_forecast_timestamp(text).is_err());
        }
    }
}
