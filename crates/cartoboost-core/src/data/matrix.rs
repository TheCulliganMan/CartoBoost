use super::{FeatureKind, FeatureSchema, SparseSetColumn};
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
pub struct Dataset {
    rows: usize,
    cols: usize,
    values: Vec<f64>,
    #[serde(default)]
    sparse_sets: Vec<SparseSetColumn>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    schema: Option<FeatureSchema>,
}

// Deserialization must establish the same invariants as the constructors.
// Otherwise malformed dimensions or duplicate sparse IDs can reach unchecked
// training indices and invalidate histogram counts.
impl<'de> Deserialize<'de> for Dataset {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct DatasetFields {
            rows: usize,
            cols: usize,
            values: Vec<f64>,
            #[serde(default)]
            sparse_sets: Vec<SparseSetColumn>,
            #[serde(default)]
            schema: Option<FeatureSchema>,
        }
        let fields = DatasetFields::deserialize(deserializer)?;
        let dataset = Self::from_flat(fields.rows, fields.cols, fields.values)
            .and_then(|dataset| dataset.with_sparse_sets(fields.sparse_sets))
            .and_then(|dataset| match fields.schema {
                Some(schema) => dataset.with_schema(schema),
                None => Ok(dataset),
            });
        dataset.map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct SampleRef<'a> {
    dataset: &'a Dataset,
    row: usize,
}

impl Dataset {
    pub fn from_rows(rows: Vec<Vec<f64>>) -> Result<Self> {
        let rows_len = rows.len();
        let cols = rows.first().map_or(0, Vec::len);
        if rows_len == 0 {
            return Err(CartoBoostError::InvalidInput(
                "dataset must contain at least one row".to_string(),
            ));
        }
        if cols == 0 {
            return Err(CartoBoostError::InvalidInput(
                "dataset must contain at least one column".to_string(),
            ));
        }
        if rows.iter().any(|row| row.len() != cols) {
            return Err(CartoBoostError::InvalidInput(
                "all rows must have the same number of columns".to_string(),
            ));
        }
        let mut values = Vec::with_capacity(rows_len * cols);
        for row in rows {
            for value in row {
                values.push(value);
            }
        }
        Self::from_flat(rows_len, cols, values)
    }

    pub fn from_flat(rows: usize, cols: usize, values: Vec<f64>) -> Result<Self> {
        if rows == 0 {
            return Err(CartoBoostError::InvalidInput(
                "dataset must contain at least one row".to_string(),
            ));
        }
        if cols == 0 {
            return Err(CartoBoostError::InvalidInput(
                "dataset must contain at least one column".to_string(),
            ));
        }
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(CartoBoostError::InvalidInput(format!(
                "matrix shape {rows}x{cols} does not match {} values",
                values.len()
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "dataset values must be finite".to_string(),
            ));
        }
        Ok(Self {
            rows,
            cols,
            values,
            sparse_sets: Vec::new(),
            schema: None,
        })
    }

    pub fn with_sparse_sets(mut self, mut sparse_sets: Vec<SparseSetColumn>) -> Result<Self> {
        for column in &mut sparse_sets {
            column.normalize();
        }
        for (idx, column) in sparse_sets.iter().enumerate() {
            if column.len() != self.rows {
                return Err(CartoBoostError::InvalidInput(format!(
                    "sparse set column {idx} has {} rows but dense matrix has {} rows",
                    column.len(),
                    self.rows
                )));
            }
        }
        self.sparse_sets = sparse_sets;
        if let Some(schema) = &self.schema {
            validate_schema_layout(schema, self.cols, self.sparse_sets.len())?;
        }
        Ok(self)
    }

    pub fn with_schema(mut self, schema: FeatureSchema) -> Result<Self> {
        schema.validate()?;
        validate_schema_layout(&schema, self.cols, self.sparse_sets.len())?;
        self.schema = Some(schema);
        Ok(self)
    }

    pub fn mixed(
        dense_rows: Vec<Vec<f64>>,
        sparse_sets: Vec<SparseSetColumn>,
        schema: Option<FeatureSchema>,
    ) -> Result<Self> {
        let dataset = Self::from_rows(dense_rows)?.with_sparse_sets(sparse_sets)?;
        match schema {
            Some(schema) => dataset.with_schema(schema),
            None => Ok(dataset),
        }
    }

    pub fn n_rows(&self) -> usize {
        self.rows
    }

    pub fn n_cols(&self) -> usize {
        self.cols
    }

    pub fn n_sparse_sets(&self) -> usize {
        self.sparse_sets.len()
    }

    pub fn feature_schema(&self) -> Option<&FeatureSchema> {
        self.schema.as_ref()
    }

    pub fn feature_schema_or_default(&self) -> FeatureSchema {
        self.schema.clone().unwrap_or_else(|| {
            let mut names = (0..self.cols)
                .map(|idx| format!("feature_{idx}"))
                .collect::<Vec<_>>();
            let mut kinds = vec![FeatureKind::Numeric; self.cols];
            names.extend((0..self.sparse_sets.len()).map(|idx| format!("sparse_set_{idx}")));
            kinds.extend(vec![FeatureKind::SparseSet; self.sparse_sets.len()]);
            FeatureSchema { names, kinds }
        })
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.values[row * self.cols + col]
    }

    pub(crate) fn dense_row(&self, row: usize) -> &[f64] {
        let start = row * self.cols;
        &self.values[start..start + self.cols]
    }

    pub fn sparse_set_row(&self, row: usize, sparse_col: usize) -> Option<&[u64]> {
        self.sparse_sets
            .get(sparse_col)
            .and_then(|column| column.row(row))
    }

    pub fn sparse_set_contains_any(&self, row: usize, sparse_col: usize, ids: &[u64]) -> bool {
        // Dataset construction and deserialization normalize these private
        // columns once; membership need not rescan the row for sortedness.
        self.sparse_set_row(row, sparse_col)
            .is_some_and(|values| ids.iter().any(|id| values.binary_search(id).is_ok()))
    }

    pub fn row(&self, row: usize) -> SampleRef<'_> {
        SampleRef { dataset: self, row }
    }

    pub fn rows(&self) -> impl Iterator<Item = SampleRef<'_>> {
        (0..self.rows).map(|row| self.row(row))
    }
}

fn validate_schema_layout(
    schema: &FeatureSchema,
    dense_cols: usize,
    sparse_cols: usize,
) -> Result<()> {
    if schema.is_empty() {
        return Ok(());
    }

    let expected = dense_cols + sparse_cols;
    if schema.len() != expected {
        return Err(CartoBoostError::InvalidInput(format!(
            "feature schema length {} does not match dataset feature count {expected} ({dense_cols} dense + {sparse_cols} sparse-set)",
            schema.len()
        )));
    }

    for (idx, kind) in schema.kinds.iter().take(dense_cols).enumerate() {
        if matches!(kind, FeatureKind::SparseSet) {
            return Err(CartoBoostError::InvalidInput(format!(
                "feature schema entry '{}' at dense index {idx} is sparse_set; sparse_set entries must correspond to sparse-set columns after the {dense_cols} dense features",
                schema.names[idx]
            )));
        }
    }

    for sparse_idx in 0..sparse_cols {
        let schema_idx = dense_cols + sparse_idx;
        if !matches!(schema.kinds[schema_idx], FeatureKind::SparseSet) {
            return Err(CartoBoostError::InvalidInput(format!(
                "feature schema entry '{}' at sparse-set index {sparse_idx} must be sparse_set",
                schema.names[schema_idx]
            )));
        }
    }

    Ok(())
}

impl SampleRef<'_> {
    pub fn get(&self, col: usize) -> f64 {
        self.dataset.get(self.row, col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_ragged_or_non_finite_rows() {
        assert!(Dataset::from_rows(vec![vec![1.0], vec![2.0, 3.0]]).is_err());
        assert!(Dataset::from_rows(vec![vec![f64::NAN]]).is_err());
    }

    #[test]
    fn mixed_schema_counts_dense_and_sparse_features() {
        let schema = FeatureSchema {
            names: vec![
                "distance".to_string(),
                "hour".to_string(),
                "route_cells".to_string(),
            ],
            kinds: vec![
                FeatureKind::Numeric,
                FeatureKind::Periodic { period: 24 },
                FeatureKind::SparseSet,
            ],
        };

        let dataset = Dataset::mixed(
            vec![vec![1.0, 7.0], vec![2.0, 8.0]],
            vec![SparseSetColumn::new(vec![vec![3, 1, 3], vec![]])],
            Some(schema),
        )
        .unwrap();

        assert_eq!(dataset.n_cols(), 2);
        assert_eq!(dataset.n_sparse_sets(), 1);
        assert_eq!(dataset.sparse_set_row(0, 0), Some(&[1, 3][..]));
    }

    #[test]
    fn with_sparse_sets_normalizes_public_column_values() {
        let dataset = Dataset::from_rows(vec![vec![1.0], vec![2.0]])
            .unwrap()
            .with_sparse_sets(vec![SparseSetColumn {
                values: vec![vec![9, 3, 9, 1], vec![4, 4]],
            }])
            .unwrap();

        assert_eq!(dataset.sparse_set_row(0, 0), Some(&[1, 3, 9][..]));
        assert_eq!(dataset.sparse_set_row(1, 0), Some(&[4][..]));
    }

    #[test]
    fn schema_length_mismatch_reports_dense_and_sparse_counts() {
        let err = Dataset::mixed(
            vec![vec![1.0], vec![2.0]],
            vec![SparseSetColumn::new(vec![vec![1], vec![2]])],
            Some(FeatureSchema {
                names: vec!["only_dense".to_string()],
                kinds: vec![FeatureKind::Numeric],
            }),
        )
        .unwrap_err();

        assert!(err.to_string().contains("1 dense + 1 sparse-set"));
    }

    #[test]
    fn sparse_set_schema_entries_must_correspond_to_sparse_columns() {
        let dense_sparse_err = Dataset::from_rows(vec![vec![1.0], vec![2.0]])
            .unwrap()
            .with_schema(FeatureSchema {
                names: vec!["route_cells".to_string()],
                kinds: vec![FeatureKind::SparseSet],
            })
            .unwrap_err();
        assert!(dense_sparse_err
            .to_string()
            .contains("sparse_set entries must correspond"));

        let sparse_kind_err = Dataset::mixed(
            vec![vec![1.0], vec![2.0]],
            vec![SparseSetColumn::new(vec![vec![1], vec![2]])],
            Some(FeatureSchema {
                names: vec!["distance".to_string(), "route_cells".to_string()],
                kinds: vec![FeatureKind::Numeric, FeatureKind::Numeric],
            }),
        )
        .unwrap_err();
        assert!(sparse_kind_err.to_string().contains("must be sparse_set"));
    }

    #[test]
    fn with_sparse_sets_revalidates_existing_schema() {
        let err = Dataset::from_rows(vec![vec![1.0], vec![2.0]])
            .unwrap()
            .with_schema(FeatureSchema {
                names: vec!["distance".to_string()],
                kinds: vec![FeatureKind::Numeric],
            })
            .unwrap()
            .with_sparse_sets(vec![SparseSetColumn::new(vec![vec![1], vec![2]])])
            .unwrap_err();

        assert!(err.to_string().contains("dataset feature count 2"));
    }
    #[test]
    fn deserialization_revalidates_dimensions_sparse_rows_and_schema() {
        for payload in [
            r#"{"rows":2,"cols":1,"values":[1.0]}"#,
            r#"{"rows":1,"cols":0,"values":[]}"#,
            r#"{"rows":1,"cols":1,"values":[1.0],"sparse_sets":[{"values":[]}]}"#,
            r#"{"rows":1,"cols":1,"values":[1.0],"schema":{"names":["x"],"kinds":[]}}"#,
        ] {
            assert!(
                serde_json::from_str::<Dataset>(payload).is_err(),
                "{payload}"
            );
        }
    }

    #[test]
    fn deserialization_normalizes_sparse_sets_and_preserves_valid_artifacts() {
        let payload =
            r#"{"rows":1,"cols":1,"values":[1.0],"sparse_sets":[{"values":[[9,3,9,1]]}]}"#;
        let dataset: Dataset = serde_json::from_str(payload).unwrap();
        assert_eq!(dataset.sparse_set_row(0, 0), Some(&[1, 3, 9][..]));
        let serialized = serde_json::to_string(&dataset).unwrap();
        let restored: Dataset = serde_json::from_str(&serialized).unwrap();
        assert_eq!(serde_json::to_string(&restored).unwrap(), serialized);
    }
}
