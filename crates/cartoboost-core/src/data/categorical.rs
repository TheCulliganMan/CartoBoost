use super::{validate_weights, Dataset, FeatureKind, FeatureSchema};
use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const ARTIFACT_TYPE: &str = "cartoboost.categorical_encoder";
const ARTIFACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalEncodingConfig {
    pub low_cardinality_threshold: usize,
    pub smoothing: f64,
}

impl Default for CategoricalEncodingConfig {
    fn default() -> Self {
        Self {
            low_cardinality_threshold: 16,
            smoothing: 10.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CategoricalEncodingStrategy {
    OneHot,
    Partition,
    Ordinal,
    TargetMean,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalColumnEncoder {
    pub index: usize,
    pub name: String,
    pub kind: FeatureKind,
    pub strategy: CategoricalEncodingStrategy,
    pub categories: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partitions: Vec<Vec<String>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub mapping: BTreeMap<String, f64>,
    pub unknown_value: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoothing: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoricalEncoder {
    pub artifact_type: String,
    pub artifact_version: u32,
    pub original_feature_count: usize,
    pub encoded_feature_names: Vec<String>,
    pub columns: Vec<CategoricalColumnEncoder>,
}

/// Borrowed categorical tokens or already parsed numeric column values.
/// Numeric columns preserve their f64 values without a text round trip.
#[derive(Clone, Copy, Debug)]
pub enum CategoricalInputColumn<'a> {
    Numeric(&'a [f64]),
    Tokens(&'a [String]),
}

impl<'a> CategoricalInputColumn<'a> {
    fn len(&self) -> usize {
        match self {
            Self::Numeric(values) => values.len(),
            Self::Tokens(values) => values.len(),
        }
    }

    fn tokens(&self, name: &str) -> Result<&'a [String]> {
        match self {
            Self::Tokens(values) => Ok(values),
            Self::Numeric(_) => Err(CartoBoostError::InvalidInput(format!(
                "categorical feature '{name}' requires category tokens"
            ))),
        }
    }

    fn numeric_values(&self, name: &str) -> Result<Vec<f64>> {
        match self {
            Self::Tokens(values) => parse_numeric_column(values, name),
            Self::Numeric(values) => {
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "feature '{name}' must contain finite numeric values"
                    )));
                }
                Ok(values.to_vec())
            }
        }
    }
}

impl CategoricalEncoder {
    pub fn fit_transform_rows(
        rows: &[Vec<String>],
        targets: &[f64],
        schema: Option<&FeatureSchema>,
        sample_weight: Option<&[f64]>,
        config: CategoricalEncodingConfig,
    ) -> Result<(Dataset, Self)> {
        let columns = rows_to_columns(rows)?;
        Self::fit_transform_columns(&columns, targets, schema, sample_weight, config)
    }

    pub fn fit_transform_columns(
        columns: &[Vec<String>],
        targets: &[f64],
        schema: Option<&FeatureSchema>,
        sample_weight: Option<&[f64]>,
        config: CategoricalEncodingConfig,
    ) -> Result<(Dataset, Self)> {
        let columns = columns
            .iter()
            .map(|column| CategoricalInputColumn::Tokens(column))
            .collect::<Vec<_>>();
        Self::fit_transform_typed_columns(&columns, targets, schema, sample_weight, config)
    }

    pub fn fit_transform_typed_columns(
        columns: &[CategoricalInputColumn<'_>],
        targets: &[f64],
        schema: Option<&FeatureSchema>,
        sample_weight: Option<&[f64]>,
        config: CategoricalEncodingConfig,
    ) -> Result<(Dataset, Self)> {
        let row_count = validate_columns(columns)?;
        validate_config(&config)?;
        if targets.len() != row_count {
            return Err(CartoBoostError::InvalidInput(
                "target length must match categorical rows".to_string(),
            ));
        }
        if targets.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "categorical targets must be finite".to_string(),
            ));
        }
        let weights = validate_weights(sample_weight, row_count)?;
        let schema = normalized_schema(schema, columns.len())?;
        let global_mean = weighted_mean(targets, &weights);

        let mut encoders = Vec::new();
        let mut encoded_names = Vec::new();
        let mut encoded_columns = Vec::new();

        for (idx, column) in columns.iter().enumerate() {
            let name = schema.names[idx].clone();
            match &schema.kinds[idx] {
                FeatureKind::Categorical => {
                    let column = column.tokens(&name)?;
                    let categories = sorted_categories(column);
                    if should_use_partition_encoding(
                        categories.len(),
                        config.low_cardinality_threshold,
                    ) {
                        let partitions = category_partitions(&categories);
                        for partition in &partitions {
                            encoded_names.push(partition_feature_name(&name, partition));
                            encoded_columns.push(partition_column(column, partition));
                        }
                        encoders.push(CategoricalColumnEncoder {
                            index: idx,
                            name,
                            kind: FeatureKind::Categorical,
                            strategy: CategoricalEncodingStrategy::Partition,
                            categories,
                            partitions,
                            mapping: BTreeMap::new(),
                            unknown_value: 0.0,
                            smoothing: None,
                        });
                    } else if categories.len() <= config.low_cardinality_threshold {
                        for category in &categories {
                            encoded_names.push(format!("{name}={category}"));
                            encoded_columns.push(one_hot_column(column, category));
                        }
                        encoders.push(CategoricalColumnEncoder {
                            index: idx,
                            name,
                            kind: FeatureKind::Categorical,
                            strategy: CategoricalEncodingStrategy::OneHot,
                            categories,
                            partitions: Vec::new(),
                            mapping: BTreeMap::new(),
                            unknown_value: 0.0,
                            smoothing: None,
                        });
                    } else {
                        let mapping = target_mean_mapping(
                            column,
                            targets,
                            &weights,
                            global_mean,
                            config.smoothing,
                        );
                        encoded_names.push(format!("{name}_target_mean"));
                        encoded_columns.push(target_mean_training_column(
                            column,
                            targets,
                            &weights,
                            global_mean,
                            config.smoothing,
                        ));
                        encoders.push(CategoricalColumnEncoder {
                            index: idx,
                            name,
                            kind: FeatureKind::Categorical,
                            strategy: CategoricalEncodingStrategy::TargetMean,
                            categories,
                            partitions: Vec::new(),
                            mapping,
                            unknown_value: global_mean,
                            smoothing: Some(config.smoothing),
                        });
                    }
                }
                FeatureKind::Ordinal => {
                    let column = column.tokens(&name)?;
                    let categories = sorted_categories(column);
                    let mapping = categories
                        .iter()
                        .enumerate()
                        .map(|(order, category)| (category.clone(), order as f64))
                        .collect::<BTreeMap<_, _>>();
                    encoded_names.push(name.clone());
                    encoded_columns.push(
                        column
                            .iter()
                            .map(|token| mapping[token])
                            .collect::<Vec<f64>>(),
                    );
                    encoders.push(CategoricalColumnEncoder {
                        index: idx,
                        name,
                        kind: FeatureKind::Ordinal,
                        strategy: CategoricalEncodingStrategy::Ordinal,
                        categories,
                        partitions: Vec::new(),
                        mapping,
                        unknown_value: -1.0,
                        smoothing: None,
                    });
                }
                FeatureKind::SparseSet => {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "categorical encoder cannot encode sparse-set feature '{}'",
                        schema.names[idx]
                    )));
                }
                _ => {
                    encoded_names.push(name.clone());
                    encoded_columns.push(column.numeric_values(&name)?);
                }
            }
        }

        let encoder = Self {
            artifact_type: ARTIFACT_TYPE.to_string(),
            artifact_version: ARTIFACT_VERSION,
            original_feature_count: columns.len(),
            encoded_feature_names: encoded_names,
            columns: encoders,
        };
        let dataset = columns_to_dataset(&encoded_columns)?.with_schema(FeatureSchema::numeric(
            encoder.encoded_feature_names.clone(),
        ))?;
        Ok((dataset, encoder))
    }

    pub fn transform_rows(&self, rows: &[Vec<String>]) -> Result<Dataset> {
        let columns = rows_to_columns(rows)?;
        self.transform_columns(&columns)
    }

    pub fn transform_columns(&self, columns: &[Vec<String>]) -> Result<Dataset> {
        let columns = columns
            .iter()
            .map(|column| CategoricalInputColumn::Tokens(column))
            .collect::<Vec<_>>();
        self.transform_typed_columns(&columns)
    }

    pub fn transform_typed_columns(
        &self,
        columns: &[CategoricalInputColumn<'_>],
    ) -> Result<Dataset> {
        let row_count = validate_columns(columns)?;
        if columns.len() != self.original_feature_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "categorical encoder expected {} features but received {}",
                self.original_feature_count,
                columns.len()
            )));
        }

        let encoder_lookup = self
            .columns
            .iter()
            .map(|encoder| (encoder.index, encoder))
            .collect::<BTreeMap<_, _>>();
        let mut encoded_columns = Vec::new();

        for (idx, column) in columns.iter().enumerate() {
            let Some(encoder) = encoder_lookup.get(&idx) else {
                let name = self
                    .encoded_feature_names
                    .get(encoded_columns.len())
                    .cloned()
                    .unwrap_or_else(|| format!("feature_{idx}"));
                encoded_columns.push(column.numeric_values(&name)?);
                continue;
            };

            let column = column.tokens(&encoder.name)?;
            match encoder.strategy {
                CategoricalEncodingStrategy::OneHot => {
                    for category in &encoder.categories {
                        encoded_columns.push(one_hot_column(column, category));
                    }
                }
                CategoricalEncodingStrategy::Partition => {
                    for partition in &encoder.partitions {
                        encoded_columns.push(partition_column(column, partition));
                    }
                }
                CategoricalEncodingStrategy::Ordinal => {
                    encoded_columns.push(
                        column
                            .iter()
                            .map(|token| encoder.mapping.get(token).copied().unwrap_or(-1.0))
                            .collect::<Vec<f64>>(),
                    );
                }
                CategoricalEncodingStrategy::TargetMean => {
                    encoded_columns.push(
                        column
                            .iter()
                            .map(|token| {
                                encoder
                                    .mapping
                                    .get(token)
                                    .copied()
                                    .unwrap_or(encoder.unknown_value)
                            })
                            .collect::<Vec<f64>>(),
                    );
                }
            }
        }

        debug_assert!(encoded_columns
            .iter()
            .all(|column| column.len() == row_count));
        columns_to_dataset(&encoded_columns)?
            .with_schema(FeatureSchema::numeric(self.encoded_feature_names.clone()))
    }
}

fn validate_config(config: &CategoricalEncodingConfig) -> Result<()> {
    if config.low_cardinality_threshold == 0 {
        return Err(CartoBoostError::InvalidInput(
            "low_cardinality_threshold must be positive".to_string(),
        ));
    }
    if !config.smoothing.is_finite() || config.smoothing < 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "categorical smoothing must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn normalized_schema(
    schema: Option<&FeatureSchema>,
    feature_count: usize,
) -> Result<FeatureSchema> {
    match schema {
        Some(schema) => {
            schema.validate()?;
            if schema.len() != feature_count {
                let len = schema.len();
                return Err(CartoBoostError::InvalidInput(format!(
                    "feature schema length {len} does not match categorical feature count \
                     {feature_count}",
                )));
            }
            Ok(schema.clone())
        }
        None => Ok(FeatureSchema::unnamed_numeric(feature_count)),
    }
}

fn rows_to_columns(rows: &[Vec<String>]) -> Result<Vec<Vec<String>>> {
    if rows.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "categorical rows must not be empty".to_string(),
        ));
    }
    let cols = rows.first().map_or(0, Vec::len);
    if cols == 0 {
        return Err(CartoBoostError::InvalidInput(
            "categorical rows must contain at least one feature".to_string(),
        ));
    }
    if rows.iter().any(|row| row.len() != cols) {
        return Err(CartoBoostError::InvalidInput(
            "categorical rows must be rectangular".to_string(),
        ));
    }
    let mut columns = vec![Vec::with_capacity(rows.len()); cols];
    for row in rows {
        for (idx, value) in row.iter().enumerate() {
            columns[idx].push(value.clone());
        }
    }
    Ok(columns)
}

fn validate_columns(columns: &[CategoricalInputColumn<'_>]) -> Result<usize> {
    if columns.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "categorical columns must not be empty".to_string(),
        ));
    }
    let rows = columns.first().map_or(0, CategoricalInputColumn::len);
    if rows == 0 {
        return Err(CartoBoostError::InvalidInput(
            "categorical columns must contain at least one row".to_string(),
        ));
    }
    if columns.iter().any(|column| column.len() != rows) {
        return Err(CartoBoostError::InvalidInput(
            "categorical columns must be rectangular".to_string(),
        ));
    }
    Ok(rows)
}

fn columns_to_dataset(columns: &[Vec<f64>]) -> Result<Dataset> {
    if columns.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "encoded categorical matrix must contain at least one column".to_string(),
        ));
    }
    let rows = columns[0].len();
    if columns.iter().any(|column| column.len() != rows) {
        return Err(CartoBoostError::InvalidInput(
            "encoded categorical columns must be rectangular".to_string(),
        ));
    }
    let count = rows.checked_mul(columns.len()).ok_or_else(|| {
        CartoBoostError::InvalidInput("encoded categorical matrix dimensions overflow".to_string())
    })?;
    let mut output = Vec::with_capacity(count);
    for row in 0..rows {
        output.extend(columns.iter().map(|column| column[row]));
    }
    Dataset::from_flat(rows, columns.len(), output)
}

fn sorted_categories(column: &[String]) -> Vec<String> {
    column
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn should_use_partition_encoding(category_count: usize, low_cardinality_threshold: usize) -> bool {
    (3..=4).contains(&category_count) && category_count <= low_cardinality_threshold
}

fn category_partitions(categories: &[String]) -> Vec<Vec<String>> {
    let category_count = categories.len();
    let max_mask = 1usize << category_count;
    let mut partitions = Vec::new();
    for mask in 1..(max_mask - 1) {
        let selected_count = mask.count_ones() as usize;
        if selected_count > category_count / 2 {
            continue;
        }
        if category_count % 2 == 0 && selected_count == category_count / 2 && (mask & 1) == 0 {
            continue;
        }
        partitions.push(
            categories
                .iter()
                .enumerate()
                .filter_map(|(idx, category)| {
                    ((mask & (1usize << idx)) != 0).then_some(category.clone())
                })
                .collect(),
        );
    }
    partitions
}

fn partition_feature_name(name: &str, partition: &[String]) -> String {
    format!("{name} in {{{}}}", partition.join("|"))
}

fn one_hot_column(column: &[String], category: &str) -> Vec<f64> {
    column
        .iter()
        .map(|value| if value == category { 1.0 } else { 0.0 })
        .collect()
}

fn partition_column(column: &[String], partition: &[String]) -> Vec<f64> {
    let members = partition.iter().collect::<BTreeSet<_>>();
    column
        .iter()
        .map(|value| if members.contains(value) { 1.0 } else { 0.0 })
        .collect()
}

fn parse_numeric_column(column: &[String], name: &str) -> Result<Vec<f64>> {
    column
        .iter()
        .map(|value| {
            value.parse::<f64>().map_err(|_| {
                CartoBoostError::InvalidInput(format!(
                    "feature '{name}' must be numeric or marked categorical"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()
        .and_then(|values| {
            if values.iter().any(|value| !value.is_finite()) {
                Err(CartoBoostError::InvalidInput(format!(
                    "feature '{name}' must contain finite numeric values"
                )))
            } else {
                Ok(values)
            }
        })
}

fn weighted_mean(targets: &[f64], weights: &[f64]) -> f64 {
    let (sum, weight_sum) = targets
        .iter()
        .zip(weights)
        .fold((0.0, 0.0), |(sum, weight_sum), (target, weight)| {
            (sum + target * weight, weight_sum + weight)
        });
    if weight_sum > 0.0 {
        sum / weight_sum
    } else {
        0.0
    }
}

fn target_mean_mapping(
    column: &[String],
    targets: &[f64],
    weights: &[f64],
    global_mean: f64,
    smoothing: f64,
) -> BTreeMap<String, f64> {
    let mut stats = BTreeMap::<String, (f64, f64)>::new();
    for ((token, target), weight) in column.iter().zip(targets).zip(weights) {
        let entry = stats.entry(token.clone()).or_insert((0.0, 0.0));
        entry.0 += target * weight;
        entry.1 += weight;
    }
    stats
        .into_iter()
        .map(|(token, (sum, weight_sum))| {
            let value = if weight_sum + smoothing > 0.0 {
                (sum + smoothing * global_mean) / (weight_sum + smoothing)
            } else {
                global_mean
            };
            (token, value)
        })
        .collect()
}

fn target_mean_training_column(
    column: &[String],
    targets: &[f64],
    weights: &[f64],
    global_mean: f64,
    smoothing: f64,
) -> Vec<f64> {
    let mut stats = BTreeMap::<String, (f64, f64)>::new();
    for ((token, target), weight) in column.iter().zip(targets).zip(weights) {
        let entry = stats.entry(token.clone()).or_insert((0.0, 0.0));
        entry.0 += target * weight;
        entry.1 += weight;
    }
    column
        .iter()
        .zip(targets)
        .zip(weights)
        .map(|((token, target), weight)| {
            let (sum, weight_sum) = stats.get(token).copied().unwrap_or((0.0, 0.0));
            let adjusted_sum = sum - target * weight;
            let adjusted_weight = weight_sum - weight;
            if adjusted_weight + smoothing > 0.0 {
                (adjusted_sum + smoothing * global_mean) / (adjusted_weight + smoothing)
            } else {
                global_mean
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_hot_encoding_is_stable_and_serializable() {
        let rows = vec![
            vec!["B".to_string(), "1.5".to_string()],
            vec!["A".to_string(), "2.0".to_string()],
            vec!["B".to_string(), "3.5".to_string()],
        ];
        let schema = FeatureSchema {
            names: vec!["zone".to_string(), "distance".to_string()],
            kinds: vec![FeatureKind::Categorical, FeatureKind::Numeric],
        };

        let (dataset, encoder) = CategoricalEncoder::fit_transform_rows(
            &rows,
            &[0.0, 1.0, 0.0],
            Some(&schema),
            None,
            CategoricalEncodingConfig::default(),
        )
        .unwrap();

        assert_eq!(
            encoder.encoded_feature_names,
            vec!["zone=A", "zone=B", "distance"]
        );
        assert_eq!(dataset.n_cols(), 3);
        assert_eq!(dataset.get(0, 0), 0.0);
        assert_eq!(dataset.get(0, 1), 1.0);
        assert_eq!(dataset.get(1, 0), 1.0);
        assert_eq!(dataset.get(1, 2), 2.0);

        let restored: CategoricalEncoder =
            serde_json::from_str(&serde_json::to_string(&encoder).unwrap()).unwrap();
        let transformed = restored
            .transform_rows(&[vec!["C".to_string(), "4.0".to_string()]])
            .unwrap();
        assert_eq!(transformed.get(0, 0), 0.0);
        assert_eq!(transformed.get(0, 1), 0.0);
        assert_eq!(transformed.get(0, 2), 4.0);
    }

    #[test]
    fn low_cardinality_partition_encoding_adds_subset_indicators() {
        let rows = vec![
            vec!["A".to_string()],
            vec!["B".to_string()],
            vec!["C".to_string()],
            vec!["D".to_string()],
        ];
        let schema = FeatureSchema {
            names: vec!["zone".to_string()],
            kinds: vec![FeatureKind::Categorical],
        };

        let (dataset, encoder) = CategoricalEncoder::fit_transform_rows(
            &rows,
            &[0.0, 1.0, 2.0, 3.0],
            Some(&schema),
            None,
            CategoricalEncodingConfig::default(),
        )
        .unwrap();

        assert_eq!(
            encoder.columns[0].strategy,
            CategoricalEncodingStrategy::Partition
        );
        assert_eq!(encoder.columns[0].partitions.len(), 7);
        assert_eq!(dataset.n_cols(), 7);
        assert_eq!(encoder.encoded_feature_names[2], "zone in {A|B}");

        let transformed = encoder.transform_rows(&[vec!["B".to_string()]]).unwrap();
        assert_eq!(transformed.get(0, 1), 1.0);
        assert_eq!(transformed.get(0, 2), 1.0);
        assert_eq!(transformed.get(0, 4), 0.0);
    }

    #[test]
    fn target_mean_encoding_uses_leave_one_out_training_values() {
        let rows = vec![
            vec!["z0".to_string()],
            vec!["z1".to_string()],
            vec!["z2".to_string()],
            vec!["z3".to_string()],
        ];
        let schema = FeatureSchema {
            names: vec!["zone".to_string()],
            kinds: vec![FeatureKind::Categorical],
        };
        let config = CategoricalEncodingConfig {
            low_cardinality_threshold: 1,
            smoothing: 1.0,
        };

        let (dataset, encoder) = CategoricalEncoder::fit_transform_rows(
            &rows,
            &[0.0, 2.0, 4.0, 6.0],
            Some(&schema),
            None,
            config,
        )
        .unwrap();

        assert_eq!(
            encoder.columns[0].strategy,
            CategoricalEncodingStrategy::TargetMean
        );
        assert_eq!(encoder.columns[0].unknown_value, 3.0);
        assert_eq!(dataset.get(0, 0), 3.0);

        let transformed = encoder
            .transform_rows(&[vec!["missing".to_string()]])
            .unwrap();
        assert_eq!(transformed.get(0, 0), 3.0);
    }

    #[test]
    fn target_mean_training_values_exclude_current_row_target() {
        let rows = vec![
            vec!["z0".to_string()],
            vec!["z0".to_string()],
            vec!["z1".to_string()],
            vec!["z2".to_string()],
        ];
        let schema = FeatureSchema {
            names: vec!["zone".to_string()],
            kinds: vec![FeatureKind::Categorical],
        };
        let config = CategoricalEncodingConfig {
            low_cardinality_threshold: 1,
            smoothing: 0.0,
        };

        let (dataset, encoder) = CategoricalEncoder::fit_transform_rows(
            &rows,
            &[2.0, 6.0, 10.0, 14.0],
            Some(&schema),
            None,
            config,
        )
        .unwrap();

        assert_eq!(
            encoder.columns[0].strategy,
            CategoricalEncodingStrategy::TargetMean
        );
        assert_eq!(dataset.get(0, 0), 6.0);
        assert_eq!(dataset.get(1, 0), 2.0);
        assert_eq!(dataset.get(2, 0), 8.0);
    }

    #[test]
    fn ordinal_encoding_keeps_unknowns_below_seen_categories() {
        let rows = vec![
            vec!["medium".to_string()],
            vec!["low".to_string()],
            vec!["high".to_string()],
        ];
        let schema = FeatureSchema {
            names: vec!["service_tier".to_string()],
            kinds: vec![FeatureKind::Ordinal],
        };

        let (_dataset, encoder) = CategoricalEncoder::fit_transform_rows(
            &rows,
            &[1.0, 2.0, 3.0],
            Some(&schema),
            None,
            CategoricalEncodingConfig::default(),
        )
        .unwrap();
        let transformed = encoder
            .transform_rows(&[vec!["unknown".to_string()]])
            .unwrap();

        assert_eq!(
            encoder.columns[0].strategy,
            CategoricalEncodingStrategy::Ordinal
        );
        assert_eq!(transformed.get(0, 0), -1.0);
    }
    #[test]
    fn typed_columns_preserve_encoder_artifacts_and_all_encoding_strategies() {
        let numeric = [0.125, 1.5, -2.25, 3.75, 4.0, 8.5];
        let numeric_tokens = numeric.iter().map(f64::to_string).collect::<Vec<_>>();
        let targets = [0.2, -0.1, 1.5, 0.0, 2.0, 7.25];
        let weights = [1.0, 0.5, 2.0, 0.75, 1.25, 3.0];
        for (kind, category_count, threshold) in [
            (FeatureKind::Categorical, 2, 16),
            (FeatureKind::Categorical, 4, 16),
            (FeatureKind::Categorical, 5, 2),
            (FeatureKind::Ordinal, 4, 16),
        ] {
            let tokens = (0..numeric.len())
                .map(|row| format!("zone{}", row % category_count))
                .collect::<Vec<_>>();
            let schema = FeatureSchema {
                names: vec!["zone".into(), "distance".into()],
                kinds: vec![kind, FeatureKind::Numeric],
            };
            for sample_weight in [None, Some(weights.as_slice())] {
                let config = CategoricalEncodingConfig {
                    low_cardinality_threshold: threshold,
                    smoothing: 1.5,
                };
                let (expected, expected_encoder) = CategoricalEncoder::fit_transform_columns(
                    &[tokens.clone(), numeric_tokens.clone()],
                    &targets,
                    Some(&schema),
                    sample_weight,
                    config.clone(),
                )
                .unwrap();
                let (actual, encoder) = CategoricalEncoder::fit_transform_typed_columns(
                    &[
                        CategoricalInputColumn::Tokens(&tokens),
                        CategoricalInputColumn::Numeric(&numeric),
                    ],
                    &targets,
                    Some(&schema),
                    sample_weight,
                    config,
                )
                .unwrap();
                assert_eq!(
                    serde_json::to_string(&encoder).unwrap(),
                    serde_json::to_string(&expected_encoder).unwrap()
                );
                for row in 0..actual.n_rows() {
                    for column in 0..actual.n_cols() {
                        assert_eq!(
                            actual.get(row, column).to_bits(),
                            expected.get(row, column).to_bits()
                        );
                    }
                }
                let unseen = vec!["unknown".to_string(), tokens[0].clone()];
                let query = [5.125, -0.25];
                let expected = expected_encoder
                    .transform_columns(&[
                        unseen.clone(),
                        query.iter().map(f64::to_string).collect(),
                    ])
                    .unwrap();
                let actual = encoder
                    .transform_typed_columns(&[
                        CategoricalInputColumn::Tokens(&unseen),
                        CategoricalInputColumn::Numeric(&query),
                    ])
                    .unwrap();
                for row in 0..actual.n_rows() {
                    for column in 0..actual.n_cols() {
                        assert_eq!(
                            actual.get(row, column).to_bits(),
                            expected.get(row, column).to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn typed_columns_reject_invalid_numeric_and_missing_category_tokens() {
        let schema = FeatureSchema {
            names: vec!["zone".into()],
            kinds: vec![FeatureKind::Categorical],
        };
        let result = CategoricalEncoder::fit_transform_typed_columns(
            &[CategoricalInputColumn::Numeric(&[1.0, 2.0])],
            &[0.0, 1.0],
            Some(&schema),
            None,
            CategoricalEncodingConfig::default(),
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("requires category tokens"));
        let result = CategoricalEncoder::fit_transform_typed_columns(
            &[CategoricalInputColumn::Numeric(&[1.0, f64::INFINITY])],
            &[0.0, 1.0],
            None,
            None,
            CategoricalEncodingConfig::default(),
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("finite numeric values"));
        let result = CategoricalEncoder::fit_transform_typed_columns(
            &[
                CategoricalInputColumn::Numeric(&[1.0]),
                CategoricalInputColumn::Tokens(&[]),
            ],
            &[0.0],
            None,
            None,
            CategoricalEncodingConfig::default(),
        );
        assert!(result.unwrap_err().to_string().contains("rectangular"));
    }
}
