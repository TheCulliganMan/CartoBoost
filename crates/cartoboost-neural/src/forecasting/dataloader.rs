use cartoboost_core::forecasting::{ForecastFrame, ForecastFrequency};
use std::collections::BTreeMap;

use crate::{NeuralError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ForecastWindow {
    pub series_id: String,
    pub inputs: Vec<f64>,
    pub target: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WindowDataset {
    windows: Vec<ForecastWindow>,
    tails: BTreeMap<String, Vec<f64>>,
    frequency: ForecastFrequency,
}

impl WindowDataset {
    pub fn from_frame(frame: &ForecastFrame, input_size: usize) -> Result<Self> {
        if input_size == 0 {
            return Err(NeuralError::InvalidArgument(
                "input_size must be positive".to_string(),
            ));
        }
        let mut windows = Vec::new();
        let mut tails = BTreeMap::new();
        for rows in frame
            .rows()
            .chunk_by(|left, right| left.series_id == right.series_id)
        {
            let series_id = &rows[0].series_id;
            if rows.len() <= input_size {
                return Err(NeuralError::InvalidArgument(format!(
                    "series '{series_id}' needs more than {input_size} rows"
                )));
            }
            let values = rows.iter().map(|row| row.target).collect::<Vec<_>>();
            for end in input_size..values.len() {
                windows.push(ForecastWindow {
                    series_id: series_id.clone(),
                    inputs: values[end - input_size..end].to_vec(),
                    target: values[end],
                });
            }
            tails.insert(
                series_id.clone(),
                values[values.len() - input_size..].to_vec(),
            );
        }
        Ok(Self {
            windows,
            tails,
            frequency: frame.frequency(),
        })
    }

    pub fn windows(&self) -> &[ForecastWindow] {
        &self.windows
    }

    pub fn tails(&self) -> &BTreeMap<String, Vec<f64>> {
        &self.tails
    }

    pub fn frequency(&self) -> ForecastFrequency {
        self.frequency
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartoboost_core::forecasting::{parse_forecast_timestamp, ForecastRow};

    #[test]
    fn grouped_windows_preserve_interleaved_panel_tails_and_order() {
        let frame = ForecastFrame::new(
            (0..6)
                .rev()
                .flat_map(|index| {
                    ["PU3-DO4", "PU1-DO2"].into_iter().map(move |series_id| {
                        ForecastRow::new(
                            series_id,
                            parse_forecast_timestamp(&format!("2026-01-{:02}", index + 1)).unwrap(),
                            index as f64 + if series_id == "PU1-DO2" { 10.0 } else { 20.0 },
                        )
                    })
                })
                .collect(),
            ForecastFrequency::Daily,
        )
        .unwrap();
        let dataset = WindowDataset::from_frame(&frame, 3).unwrap();
        let expected = frame
            .series_ids()
            .into_iter()
            .flat_map(|series_id| {
                let rows = frame.rows_for_series(&series_id);
                (3..rows.len())
                    .map(|end| ForecastWindow {
                        series_id: series_id.clone(),
                        inputs: rows[end - 3..end].iter().map(|row| row.target).collect(),
                        target: rows[end].target,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(dataset.windows(), expected);
        assert_eq!(dataset.tails()["PU1-DO2"], vec![13.0, 14.0, 15.0]);
        assert_eq!(dataset.tails()["PU3-DO4"], vec![23.0, 24.0, 25.0]);
    }
}
