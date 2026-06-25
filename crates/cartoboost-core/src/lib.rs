pub mod booster;
pub mod data;
pub mod explain;
pub mod finance;
pub mod forecasting;
pub mod geo;
pub mod graph_regularization;
pub mod loss;
pub mod metrics;
pub mod objectives;
pub mod predictors;
pub(crate) mod profile;
pub mod serialize;
pub mod splitters;
pub mod tree;
pub mod utilities;

pub use booster::{
    Booster, BoosterConfig, ClassificationObjective, Classifier, ClassifierConfig, ClassifierModel,
    Ranker, RankerConfig, RankerModel, RankingObjective,
};
pub use data::{
    CategoricalColumnEncoder, CategoricalEncoder, CategoricalEncodingConfig,
    CategoricalEncodingStrategy, Dataset, FeatureKind, FeatureSchema,
};
pub use tree::{Model, Tree};

#[derive(Debug, thiserror::Error)]
pub enum CartoBoostError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("model IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, CartoBoostError>;
