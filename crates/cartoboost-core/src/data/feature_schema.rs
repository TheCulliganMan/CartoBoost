use crate::{CartoBoostError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeatureKind {
    Numeric,
    Spatial,
    Categorical,
    Ordinal,
    Periodic { period: u32 },
    SparseSet,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSchema {
    #[serde(default)]
    pub names: Vec<String>,
    #[serde(default)]
    pub kinds: Vec<FeatureKind>,
}

impl FeatureSchema {
    pub fn numeric(names: Vec<String>) -> Self {
        let kinds = vec![FeatureKind::Numeric; names.len()];
        Self { names, kinds }
    }

    pub fn empty() -> Self {
        Self {
            names: Vec::new(),
            kinds: Vec::new(),
        }
    }

    pub fn unnamed_numeric(feature_count: usize) -> Self {
        let names = (0..feature_count)
            .map(|idx| format!("feature_{idx}"))
            .collect::<Vec<_>>();
        Self::numeric(names)
    }

    pub fn len(&self) -> usize {
        self.names.len().max(self.kinds.len())
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty() && self.kinds.is_empty()
    }

    pub fn validate(&self) -> Result<()> {
        if self.names.len() != self.kinds.len() {
            return Err(CartoBoostError::InvalidInput(format!(
                "feature schema names length {} does not match kinds length {}",
                self.names.len(),
                self.kinds.len()
            )));
        }
        let mut seen = HashSet::with_capacity(self.names.len());
        for (idx, name) in self.names.iter().enumerate() {
            if name.trim().is_empty() {
                return Err(CartoBoostError::InvalidInput(format!(
                    "feature schema name at index {idx} must not be empty"
                )));
            }
            if !seen.insert(name.as_str()) {
                return Err(CartoBoostError::InvalidInput(format!(
                    "feature schema contains duplicate feature name '{name}'"
                )));
            }
        }
        for (idx, kind) in self.kinds.iter().enumerate() {
            if let FeatureKind::Periodic { period } = kind {
                if *period == 0 {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "periodic feature '{}' at index {idx} must have a positive period",
                        self.names[idx]
                    )));
                }
            }
        }
        Ok(())
    }
}
