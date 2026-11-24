use crate::error::{NeuralError, Result};
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
pub struct NeuralFeatureBlock {
    pub name: String,
    pub dim: usize,
    pub values: Vec<f32>,
}

impl NeuralFeatureBlock {
    pub fn new(name: impl Into<String>, dim: usize, values: Vec<f32>) -> Result<Self> {
        if dim == 0 {
            return Err(NeuralError::InvalidArgument(
                "embedding dimension must be greater than zero".to_string(),
            ));
        }
        if values.len() % dim != 0 {
            let row_count = values.len() / dim;
            return Err(NeuralError::InvalidRowCount {
                expected: row_count.saturating_mul(dim),
                actual: values.len(),
            });
        }

        Ok(Self {
            name: name.into(),
            dim,
            values,
        })
    }

    pub fn row_count(&self) -> usize {
        self.values.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn feature_names(&self) -> Vec<String> {
        (0..self.dim)
            .map(|index| format!("{}_{}", self.name, index_format(index)))
            .collect()
    }

    pub fn append_to_dense_f32(&self, dense: &mut [Vec<f32>]) -> Result<()> {
        let expected_rows = dense.len();
        if self.row_count() != expected_rows {
            return Err(NeuralError::InvalidRowCount {
                expected: expected_rows,
                actual: self.row_count(),
            });
        }

        dense
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_index, row)| {
                let start = row_index * self.dim;
                let end = start + self.dim;
                row.extend(&self.values[start..end]);
            });

        Ok(())
    }

    pub fn append_to_dense_f64(&self, dense: &mut [Vec<f64>]) -> Result<()> {
        let expected_rows = dense.len();
        if self.row_count() != expected_rows {
            return Err(NeuralError::InvalidRowCount {
                expected: expected_rows,
                actual: self.row_count(),
            });
        }

        dense
            .par_iter_mut()
            .enumerate()
            .for_each(|(row_index, row)| {
                let start = row_index * self.dim;
                let end = start + self.dim;
                row.extend(
                    self.values[start..end]
                        .iter()
                        .map(|value| f64::from(*value)),
                );
            });

        Ok(())
    }
}

fn index_format(value: usize) -> String {
    format!("{value:02}")
}
