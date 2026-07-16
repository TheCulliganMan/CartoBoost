use chrono::{DateTime, NaiveDate, NaiveDateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EARTH_RADIUS_KM: f64 = 6_371.0;
pub const EARTH_RADIUS_METERS: f64 = EARTH_RADIUS_KM * 1_000.0;

#[derive(Debug, thiserror::Error)]
pub enum GeoCoreError {
    #[error("{0}")]
    InvalidInput(String),
    #[error("serialization failed: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, GeoCoreError>;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CoordinateMatrix {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub crs: Option<String>,
    pub id_col: Option<String>,
}

impl CoordinateMatrix {
    pub fn new(
        x: Vec<f64>,
        y: Vec<f64>,
        crs: Option<String>,
        id_col: Option<String>,
    ) -> Result<Self> {
        if x.len() != y.len() || x.is_empty() {
            return Err(GeoCoreError::InvalidInput(
                "x and y must have the same positive length".to_string(),
            ));
        }
        if x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
            return Err(GeoCoreError::InvalidInput(
                "coordinates must be finite".to_string(),
            ));
        }
        Ok(Self { x, y, crs, id_col })
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    pub fn from_json_str(value: &str) -> Result<Self> {
        let parsed: Self = serde_json::from_str(value)?;
        Self::new(parsed.x, parsed.y, parsed.crs, parsed.id_col)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TimeIndex {
    pub timestamps: Vec<DateTime<Utc>>,
    pub frequency: Option<String>,
    pub timezone: String,
}

impl TimeIndex {
    pub fn new(
        values: Vec<String>,
        frequency: Option<String>,
        timezone: Option<String>,
    ) -> Result<Self> {
        if values.is_empty() {
            return Err(GeoCoreError::InvalidInput(
                "timestamps must not be empty".to_string(),
            ));
        }
        let timestamps = values
            .iter()
            .map(|v| parse_timestamp(v))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            timestamps,
            frequency,
            timezone: timezone.unwrap_or_else(|| "UTC".to_string()),
        })
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timestamps.is_empty()
    }

    pub fn iso_strings(&self) -> Vec<String> {
        self.timestamps.iter().map(DateTime::to_rfc3339).collect()
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    pub fn from_json_str(value: &str) -> Result<Self> {
        let parsed: Self = serde_json::from_str(value)?;
        if parsed.timestamps.is_empty() {
            return Err(GeoCoreError::InvalidInput(
                "timestamps must not be empty".to_string(),
            ));
        }
        Ok(parsed)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PanelIndex {
    pub entity_ids: Vec<String>,
    pub time: Option<TimeIndex>,
}

impl PanelIndex {
    pub fn new(entity_ids: Vec<String>, time: Option<TimeIndex>) -> Result<Self> {
        if entity_ids.is_empty() || entity_ids.iter().any(|id| id.is_empty()) {
            return Err(GeoCoreError::InvalidInput(
                "entity ids must be non-empty".to_string(),
            ));
        }
        if let Some(time) = &time {
            if time.len() != entity_ids.len() {
                return Err(GeoCoreError::InvalidInput(
                    "time length must match entity ids".to_string(),
                ));
            }
        }
        Ok(Self { entity_ids, time })
    }

    pub fn len(&self) -> usize {
        self.entity_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entity_ids.is_empty()
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    pub fn from_json_str(value: &str) -> Result<Self> {
        let parsed: Self = serde_json::from_str(value)?;
        Self::new(parsed.entity_ids, parsed.time)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SpatialWeights {
    pub n_nodes: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub data: Vec<f64>,
    pub node_ids: Option<Vec<String>>,
    pub row_normalized: bool,
}

impl SpatialWeights {
    pub fn new(
        n_nodes: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        data: Vec<f64>,
        node_ids: Option<Vec<String>>,
        row_normalized: bool,
    ) -> Result<Self> {
        validate_csr(n_nodes, &indptr, &indices, &data, node_ids.as_ref())?;
        Ok(Self {
            n_nodes,
            indptr,
            indices,
            data,
            node_ids,
            row_normalized,
        })
    }

    pub fn from_edges(
        n_nodes: usize,
        edges: Vec<(usize, usize, f64)>,
        symmetric: bool,
    ) -> Result<Self> {
        if n_nodes == 0 {
            return Err(GeoCoreError::InvalidInput(
                "n_nodes must be positive".to_string(),
            ));
        }
        let mut rows: Vec<BTreeMap<usize, f64>> = vec![BTreeMap::new(); n_nodes];
        for (src, dst, weight) in edges {
            if src >= n_nodes || dst >= n_nodes || !weight.is_finite() || weight < 0.0 {
                return Err(GeoCoreError::InvalidInput(
                    "edges must reference valid nodes with non-negative finite weights".to_string(),
                ));
            }
            *rows[src].entry(dst).or_default() += weight;
            if symmetric && src != dst {
                *rows[dst].entry(src).or_default() += weight;
            }
        }
        let mut indptr = vec![0];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in rows {
            for (idx, weight) in row {
                indices.push(idx);
                data.push(weight);
            }
            indptr.push(indices.len());
        }
        Self::new(n_nodes, indptr, indices, data, None, false)
    }

    pub fn row_normalize(&self) -> Self {
        let mut out = self.clone();
        for row in 0..self.n_nodes {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            let sum: f64 = out.data[start..end].iter().sum();
            if sum > 0.0 {
                for value in &mut out.data[start..end] {
                    *value /= sum;
                }
            }
        }
        out.row_normalized = true;
        out
    }

    pub fn isolated_nodes(&self) -> Vec<usize> {
        (0..self.n_nodes)
            .filter(|row| self.indptr[*row] == self.indptr[*row + 1])
            .collect()
    }

    pub fn is_symmetric(&self, tolerance: f64) -> bool {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return false;
        }
        let mut values = BTreeMap::new();
        for row in 0..self.n_nodes {
            for offset in self.indptr[row]..self.indptr[row + 1] {
                values.insert((row, self.indices[offset]), self.data[offset]);
            }
        }
        values.iter().all(|(&(row, col), value)| {
            let reverse = values.get(&(col, row)).copied().unwrap_or(0.0);
            (*value - reverse).abs() <= tolerance
        })
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    pub fn from_json_str(value: &str) -> Result<Self> {
        let parsed: Self = serde_json::from_str(value)?;
        Self::new(
            parsed.n_nodes,
            parsed.indptr,
            parsed.indices,
            parsed.data,
            parsed.node_ids,
            parsed.row_normalized,
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GeoFrameMeta {
    pub dataset_fingerprint: String,
    pub coordinate_crs_note: String,
    pub model_version: String,
    pub dependency_versions: BTreeMap<String, String>,
    pub random_seed: Option<u64>,
    pub row_count: usize,
    pub split_id: Option<String>,
}

impl GeoFrameMeta {
    pub fn new(
        dataset_fingerprint: String,
        coordinate_crs_note: String,
        model_version: String,
        dependency_versions: BTreeMap<String, String>,
        random_seed: Option<u64>,
        row_count: usize,
        split_id: Option<String>,
    ) -> Result<Self> {
        if !dataset_fingerprint.starts_with("sha256:")
            || coordinate_crs_note.trim().is_empty()
            || model_version.trim().is_empty()
        {
            return Err(GeoCoreError::InvalidInput(
                "metadata requires sha256 dataset_fingerprint, CRS note, and model version"
                    .to_string(),
            ));
        }
        Ok(Self {
            dataset_fingerprint,
            coordinate_crs_note,
            model_version,
            dependency_versions,
            random_seed,
            row_count,
            split_id,
        })
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SplitFold {
    pub fold_id: String,
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SplitManifest {
    pub split_id: String,
    pub split_kind: String,
    pub row_count: usize,
    pub folds: Vec<SplitFold>,
    pub dataset_fingerprint: String,
    pub coordinate_crs_note: String,
    pub model_version: String,
    pub dependency_versions: BTreeMap<String, String>,
    pub random_seed: Option<u64>,
}

impl SplitManifest {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        split_id: String,
        split_kind: String,
        row_count: usize,
        folds: Vec<SplitFold>,
        dataset_fingerprint: String,
        coordinate_crs_note: String,
        model_version: String,
        dependency_versions: BTreeMap<String, String>,
        random_seed: Option<u64>,
    ) -> Result<Self> {
        if split_id.trim().is_empty() || split_kind.trim().is_empty() || folds.is_empty() {
            return Err(GeoCoreError::InvalidInput(
                "split manifests require ids and at least one fold".to_string(),
            ));
        }
        GeoFrameMeta::new(
            dataset_fingerprint.clone(),
            coordinate_crs_note.clone(),
            model_version.clone(),
            dependency_versions.clone(),
            random_seed,
            row_count,
            Some(split_id.clone()),
        )?;
        for fold in &folds {
            validate_fold(fold, row_count)?;
        }
        Ok(Self {
            split_id,
            split_kind,
            row_count,
            folds,
            dataset_fingerprint,
            coordinate_crs_note,
            model_version,
            dependency_versions,
            random_seed,
        })
    }

    pub fn hash(&self) -> Result<String> {
        let payload = serde_json::to_vec(self)?;
        Ok(format!("sha256:{}", stable_hex64(&payload)))
    }

    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(Into::into)
    }

    pub fn from_json_str(value: &str) -> Result<Self> {
        let parsed: Self = serde_json::from_str(value)?;
        Self::new(
            parsed.split_id,
            parsed.split_kind,
            parsed.row_count,
            parsed.folds,
            parsed.dataset_fingerprint,
            parsed.coordinate_crs_note,
            parsed.model_version,
            parsed.dependency_versions,
            parsed.random_seed,
        )
    }
}

// Cohesive implementation families share the crate namespace.
include!("geo/splits.rs");
include!("geo/geometry.rs");
include!("geo/tests.rs");
