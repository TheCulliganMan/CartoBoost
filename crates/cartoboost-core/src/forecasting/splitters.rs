use crate::forecasting::{ForecastFrame, ForecastRow};
use crate::{CartoBoostError, Result};
use chrono::NaiveDateTime;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ForecastWindow {
    Expanding,
    Sliding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RollingOriginSplitter {
    pub horizon: usize,
    pub step: usize,
    pub min_train_size: usize,
    pub max_train_size: Option<usize>,
    pub n_splits: Option<usize>,
    pub window: ForecastWindow,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastFold {
    pub fold_id: String,
    pub train_indices: Vec<usize>,
    pub validation_indices: Vec<usize>,
    pub train_start: NaiveDateTime,
    pub train_end: NaiveDateTime,
    pub validation_start: NaiveDateTime,
    pub validation_end: NaiveDateTime,
    pub horizon: usize,
    pub step: usize,
    pub metadata: ForecastFoldMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForecastFoldMetadata {
    pub origin_timestamp: NaiveDateTime,
    pub train_size: usize,
    pub validation_size: usize,
    pub train_timestamp_count: usize,
    pub validation_timestamp_count: usize,
    pub series_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CandidateValidationCutoffSchedule {
    pub validation_profile: Option<String>,
    pub horizon: usize,
    pub timestamp_count: usize,
    pub cutoff_indices: Vec<usize>,
}

#[allow(dead_code)]
impl CandidateValidationCutoffSchedule {
    pub fn new(
        timestamp_count: usize,
        horizon: usize,
        validation_profile: Option<&str>,
    ) -> Result<Self> {
        if horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "candidate validation horizon must be positive".to_string(),
            ));
        }
        let max_origins = match validation_profile {
            Some("rank_portfolio") => 3,
            Some("hierarchical_reconciliation") => 2,
            Some("classical_competition") | Some("classical_competition_full") | None => 3,
            Some(_) => 2,
        };
        let mut cutoff_indices = Vec::new();
        if horizon < timestamp_count.saturating_sub(2) {
            for origin in (1..=max_origins).rev() {
                let Some(offset) = horizon.checked_mul(origin) else {
                    continue;
                };
                let Some(index) = timestamp_count.checked_sub(offset) else {
                    continue;
                };
                let min_train = horizon.max(14);
                if index >= min_train && index + horizon <= timestamp_count {
                    cutoff_indices.push(index);
                }
            }
        }
        Ok(Self {
            validation_profile: validation_profile.map(str::to_string),
            horizon,
            timestamp_count,
            cutoff_indices,
        })
    }
}

impl RollingOriginSplitter {
    pub fn expanding(horizon: usize, min_train_size: usize) -> Result<Self> {
        Self::new(
            horizon,
            1,
            min_train_size,
            None,
            None,
            ForecastWindow::Expanding,
        )
    }

    pub fn sliding(horizon: usize, min_train_size: usize, max_train_size: usize) -> Result<Self> {
        Self::new(
            horizon,
            1,
            min_train_size,
            Some(max_train_size),
            None,
            ForecastWindow::Sliding,
        )
    }

    pub fn new(
        horizon: usize,
        step: usize,
        min_train_size: usize,
        max_train_size: Option<usize>,
        n_splits: Option<usize>,
        window: ForecastWindow,
    ) -> Result<Self> {
        let splitter = Self {
            horizon,
            step,
            min_train_size,
            max_train_size,
            n_splits,
            window,
        };
        splitter.validate()?;
        Ok(splitter)
    }

    fn validate(&self) -> Result<()> {
        if self.horizon == 0 {
            return Err(CartoBoostError::InvalidInput(
                "forecast horizon must be positive".to_string(),
            ));
        }
        if self.step == 0 {
            return Err(CartoBoostError::InvalidInput(
                "rolling-origin step must be positive".to_string(),
            ));
        }
        if self.min_train_size == 0 {
            return Err(CartoBoostError::InvalidInput(
                "min_train_size must be positive".to_string(),
            ));
        }
        if let Some(max_train_size) = self.max_train_size {
            if max_train_size < self.min_train_size {
                return Err(CartoBoostError::InvalidInput(
                    "max_train_size must be greater than or equal to min_train_size".to_string(),
                ));
            }
        }
        if matches!(self.window, ForecastWindow::Sliding) && self.max_train_size.is_none() {
            return Err(CartoBoostError::InvalidInput(
                "sliding forecast windows require max_train_size".to_string(),
            ));
        }
        if self.n_splits == Some(0) {
            return Err(CartoBoostError::InvalidInput(
                "n_splits must be positive when provided".to_string(),
            ));
        }
        Ok(())
    }

    pub fn split(&self, frame: &ForecastFrame) -> Result<Vec<ForecastFold>> {
        // Fields are public and deserializable, so constructor checks alone
        // cannot guard zero steps or overflowing window arithmetic.
        self.validate()?;
        let timestamps = unique_timestamps(frame.rows());
        if timestamps.len() <= self.horizon {
            return Err(CartoBoostError::InvalidInput(format!(
                "rolling-origin split requires more than {} distinct timestamps for horizon {}, got {}",
                self.horizon,
                self.horizon,
                timestamps.len()
            )));
        }
        let last_origin = timestamps.len() - self.horizon;
        let first_cutoff = self.min_train_size - 1;
        let origin_count = if first_cutoff < last_origin {
            (last_origin - 1 - first_cutoff) / self.step + 1
        } else {
            0
        };
        let skip_origins = self
            .n_splits
            .map_or(0, |count| origin_count.saturating_sub(count));
        let mut candidates = Vec::with_capacity(origin_count - skip_origins);
        // Select the retained origins before allocating fold index vectors.
        // Each timestamp window is nonempty and contains at least one row,
        // so this is exactly the tail previously retained after construction.
        for origin in skip_origins..origin_count {
            let cutoff = first_cutoff + origin * self.step;
            let train_start_pos = self.max_train_size.map_or(0, |max_train_size| {
                (cutoff + 1).saturating_sub(max_train_size)
            });
            let train_times = &timestamps[train_start_pos..=cutoff];
            let validation_times = &timestamps[cutoff + 1..=cutoff + self.horizon];

            let train_indices = indices_for_timestamps(frame.rows(), train_times);
            let validation_indices = indices_for_timestamps(frame.rows(), validation_times);
            if train_times.len() >= self.min_train_size
                && !train_indices.is_empty()
                && !validation_indices.is_empty()
            {
                let train_end = *train_times
                    .last()
                    .expect("non-empty train timestamp window");
                let validation_start = validation_times[0];
                if train_end >= validation_start {
                    return Err(CartoBoostError::InvalidInput(
                        "forecast split leakage: max(train timestamp) must be < min(validation timestamp)"
                            .to_string(),
                    ));
                }
                let series_count = validation_indices
                    .iter()
                    .map(|index| frame.rows()[*index].series_id.as_str())
                    .collect::<BTreeSet<_>>()
                    .len();
                candidates.push(ForecastFold {
                    fold_id: String::new(),
                    train_indices,
                    validation_indices,
                    train_start: train_times[0],
                    train_end,
                    validation_start,
                    validation_end: *validation_times
                        .last()
                        .expect("non-empty validation timestamp window"),
                    horizon: self.horizon,
                    step: self.step,
                    metadata: ForecastFoldMetadata {
                        origin_timestamp: timestamps[cutoff],
                        train_size: 0,
                        validation_size: 0,
                        train_timestamp_count: train_times.len(),
                        validation_timestamp_count: validation_times.len(),
                        series_count,
                    },
                });
            }
        }
        if candidates.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "rolling-origin split produced no folds: {} timestamps cannot satisfy min_train_size {} and horizon {}",
                timestamps.len(), self.min_train_size, self.horizon
            )));
        }
        for (index, fold) in candidates.iter_mut().enumerate() {
            fold.fold_id = format!("fold_{index:04}");
            fold.metadata.train_size = fold.train_indices.len();
            fold.metadata.validation_size = fold.validation_indices.len();
        }
        Ok(candidates)
    }
}

pub(crate) fn frame_from_indices(
    frame: &ForecastFrame,
    indices: &[usize],
) -> Result<ForecastFrame> {
    let rows = indices
        .iter()
        .map(|index| frame.rows()[*index].clone())
        .collect::<Vec<_>>();
    ForecastFrame::with_metadata(rows, frame.frequency(), frame.metadata().clone())
}

fn unique_timestamps(rows: &[ForecastRow]) -> Vec<NaiveDateTime> {
    rows.iter()
        .map(|row| row.timestamp)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn indices_for_timestamps(rows: &[ForecastRow], timestamps: &[NaiveDateTime]) -> Vec<usize> {
    let (Some(start), Some(end)) = (timestamps.first(), timestamps.last()) else {
        return Vec::new();
    };
    // Callers pass contiguous slices of the complete sorted timestamp list.
    rows.iter()
        .enumerate()
        .filter_map(|(index, row)| {
            (*start <= row.timestamp && row.timestamp <= *end).then_some(index)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forecasting::{ForecastFrameMetadata, ForecastFrequency};
    use chrono::NaiveDate;

    fn ts(day: u32) -> NaiveDateTime {
        NaiveDate::from_ymd_opt(2026, 1, day)
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .expect("timestamp")
    }

    #[test]
    fn infeasible_split_is_an_error_instead_of_an_empty_protocol() {
        let frame = ForecastFrame::new(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
            ],
            ForecastFrequency::Daily,
        )
        .expect("frame");
        let splitter = RollingOriginSplitter::expanding(2, 1).expect("splitter");

        let error = splitter
            .split(&frame)
            .expect_err("infeasible split must fail");
        assert!(error.to_string().contains("requires more than"));
    }

    #[test]
    fn split_frames_preserve_forecast_metadata() {
        let metadata = ForecastFrameMetadata {
            timestamp_col: Some("pickup_hour".to_string()),
            target_col: Some("trip_count".to_string()),
            known_future_covariates: vec!["is_airport_hour".to_string()],
            ..ForecastFrameMetadata::default()
        };
        let frame = ForecastFrame::with_metadata(
            vec![
                ForecastRow::single(ts(1), 1.0),
                ForecastRow::single(ts(2), 2.0),
                ForecastRow::single(ts(3), 3.0),
            ],
            ForecastFrequency::Daily,
            metadata.clone(),
        )
        .expect("frame");

        let split = frame_from_indices(&frame, &[0, 1]).expect("split frame");
        assert_eq!(split.metadata(), &metadata);
    }

    #[test]
    fn limited_origins_match_full_tail_for_irregular_panels() {
        let rows = (1..=20)
            .flat_map(|day| {
                ["PU100", "PU101", "PU102"]
                    .into_iter()
                    .enumerate()
                    .filter(move |(series, _)| (day as usize + series) % 4 != 0)
                    .map(move |(_, id)| ForecastRow::new(id, ts(day), day as f64))
            })
            .collect();
        let frame = ForecastFrame::with_metadata(
            rows,
            ForecastFrequency::Daily,
            ForecastFrameMetadata {
                allow_irregular: true,
                ..ForecastFrameMetadata::default()
            },
        )
        .expect("irregular panel");
        for step in [1, 2, 4, usize::MAX] {
            for window in [ForecastWindow::Expanding, ForecastWindow::Sliding] {
                let max_train_size = matches!(window, ForecastWindow::Sliding).then_some(6);
                let mut splitter =
                    RollingOriginSplitter::new(3, step, 4, max_train_size, None, window)
                        .expect("splitter");
                let all = splitter.split(&frame).expect("all folds");
                for count in [1, 3, 99] {
                    splitter.n_splits = Some(count);
                    let actual = splitter.split(&frame).expect("limited folds");
                    let mut expected = all[all.len().saturating_sub(count)..].to_vec();
                    for (index, fold) in expected.iter_mut().enumerate() {
                        fold.fold_id = format!("fold_{index:04}");
                    }
                    assert_eq!(actual, expected);
                    for fold in actual {
                        let expected_train = frame
                            .rows()
                            .iter()
                            .enumerate()
                            .filter_map(|(index, row)| {
                                (fold.train_start <= row.timestamp
                                    && row.timestamp <= fold.train_end)
                                    .then_some(index)
                            })
                            .collect::<Vec<_>>();
                        assert_eq!(fold.train_indices, expected_train);
                        assert!(fold.train_end < fold.validation_start);
                    }
                }
            }
        }
    }

    #[test]
    fn public_splitter_fields_are_validated_before_iteration() {
        let frame = ForecastFrame::new(
            (1..=5)
                .map(|day| ForecastRow::single(ts(day), day as f64))
                .collect(),
            ForecastFrequency::Daily,
        )
        .expect("frame");
        let base = RollingOriginSplitter::expanding(1, 2).expect("splitter");
        for invalid in [
            RollingOriginSplitter {
                step: 0,
                ..base.clone()
            },
            RollingOriginSplitter {
                min_train_size: 0,
                ..base.clone()
            },
            RollingOriginSplitter {
                horizon: 0,
                ..base.clone()
            },
            RollingOriginSplitter {
                n_splits: Some(0),
                ..base.clone()
            },
            RollingOriginSplitter {
                min_train_size: usize::MAX,
                ..base.clone()
            },
            RollingOriginSplitter {
                horizon: usize::MAX,
                ..base.clone()
            },
        ] {
            assert!(invalid.split(&frame).is_err());
        }
        assert!(CandidateValidationCutoffSchedule::new(10, usize::MAX, None)
            .expect("empty oversized schedule")
            .cutoff_indices
            .is_empty());
    }
}
