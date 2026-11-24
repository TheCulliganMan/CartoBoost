use crate::{CartoBoostError, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalAggregation {
    pub name: String,
    pub factor: usize,
}

impl TemporalAggregation {
    pub fn new(name: impl Into<String>, factor: usize) -> Result<Self> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "temporal aggregation name must be non-empty".to_string(),
            ));
        }
        if factor == 0 {
            return Err(CartoBoostError::InvalidInput(
                "temporal aggregation factor must be positive".to_string(),
            ));
        }
        Ok(Self { name, factor })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalHierarchy {
    base_name: String,
    aggregations: Vec<TemporalAggregation>,
}

impl TemporalHierarchy {
    pub fn new(
        base_name: impl Into<String>,
        aggregations: Vec<TemporalAggregation>,
    ) -> Result<Self> {
        let base_name = base_name.into();
        if base_name.trim().is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "temporal base name must be non-empty".to_string(),
            ));
        }
        let mut previous = 1;
        for aggregation in &aggregations {
            if aggregation.factor <= previous {
                return Err(CartoBoostError::InvalidInput(
                    "temporal aggregation factors must be strictly increasing".to_string(),
                ));
            }
            if aggregation.factor % previous != 0 {
                return Err(CartoBoostError::InvalidInput(
                    "temporal aggregation factors must be nested multiples".to_string(),
                ));
            }
            previous = aggregation.factor;
        }
        Ok(Self {
            base_name,
            aggregations,
        })
    }

    pub fn base_name(&self) -> &str {
        &self.base_name
    }

    pub fn aggregations(&self) -> &[TemporalAggregation] {
        &self.aggregations
    }

    pub fn level_names(&self) -> Vec<&str> {
        std::iter::once(self.base_name.as_str())
            .chain(
                self.aggregations
                    .iter()
                    .map(|aggregation| aggregation.name.as_str()),
            )
            .collect()
    }

    pub fn aggregate_series(&self, values: &[f64]) -> Result<Vec<TemporalLevelValues>> {
        validate_values(values)?;
        let mut out = Vec::with_capacity(self.aggregations.len() + 1);
        out.push(TemporalLevelValues {
            name: self.base_name.clone(),
            factor: 1,
            values: values.to_vec(),
        });
        for aggregation in &self.aggregations {
            out.push(TemporalLevelValues {
                name: aggregation.name.clone(),
                factor: aggregation.factor,
                values: aggregate_non_overlapping(values, aggregation.factor)?,
            });
        }
        Ok(out)
    }

    pub fn aggregate_panel(
        &self,
        rows: &[TemporalSeries],
    ) -> Result<Vec<TemporalSeriesAggregation>> {
        if rows.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "temporal panel must contain at least one series".to_string(),
            ));
        }
        rows.iter()
            .map(|row| {
                Ok(TemporalSeriesAggregation {
                    series_id: row.series_id.clone(),
                    levels: self.aggregate_series(&row.values)?,
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemporalLevelValues {
    pub name: String,
    pub factor: usize,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemporalSeries {
    pub series_id: String,
    pub values: Vec<f64>,
}

impl TemporalSeries {
    pub fn new(series_id: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            series_id: series_id.into(),
            values,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemporalSeriesAggregation {
    pub series_id: String,
    pub levels: Vec<TemporalLevelValues>,
}

pub fn aggregate_non_overlapping(values: &[f64], factor: usize) -> Result<Vec<f64>> {
    validate_values(values)?;
    if factor == 0 {
        return Err(CartoBoostError::InvalidInput(
            "temporal aggregation factor must be positive".to_string(),
        ));
    }
    if values.len() % factor != 0 {
        return Err(CartoBoostError::InvalidInput(format!(
            "series length {} is not divisible by temporal aggregation factor {}",
            values.len(),
            factor
        )));
    }
    Ok(values
        .chunks_exact(factor)
        .map(|chunk| chunk.iter().sum())
        .collect())
}

fn validate_values(values: &[f64]) -> Result<()> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "temporal series must contain at least one value".to_string(),
        ));
    }
    if !values.iter().all(|value| value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "temporal series values must be finite".to_string(),
        ));
    }
    Ok(())
}
