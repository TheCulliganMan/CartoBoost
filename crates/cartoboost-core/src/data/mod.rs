mod categorical;
mod feature_schema;
#[cfg(test)]
mod feature_schema_tests;
mod matrix;
mod sparse_sets;
#[cfg(test)]
mod sparse_sets_tests;
mod weights;

pub use categorical::{
    CategoricalColumnEncoder, CategoricalEncoder, CategoricalEncodingConfig,
    CategoricalEncodingStrategy, CategoricalInputColumn,
};
pub use feature_schema::{FeatureKind, FeatureSchema};
pub use matrix::{Dataset, SampleRef};
pub use sparse_sets::SparseSetColumn;
pub use weights::validate_weights;
