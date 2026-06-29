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

pub fn spatial_block_cv(
    coords: &CoordinateMatrix,
    n_folds: usize,
    meta: GeoFrameMeta,
    split_id: String,
) -> Result<SplitManifest> {
    if n_folds < 2 || n_folds > coords.len() {
        return Err(GeoCoreError::InvalidInput(
            "n_folds must be between 2 and row count".to_string(),
        ));
    }
    let mut order: Vec<usize> = (0..coords.len()).collect();
    order.sort_by(|a, b| {
        coords.x[*a]
            .total_cmp(&coords.x[*b])
            .then(coords.y[*a].total_cmp(&coords.y[*b]))
    });
    let folds = partition_order(&order, n_folds, "block");
    manifest(split_id, "spatial_block_cv", coords.len(), folds, meta)
}

pub fn buffered_spatial_cv(
    coords: &CoordinateMatrix,
    n_folds: usize,
    buffer_distance: f64,
    meta: GeoFrameMeta,
    split_id: String,
) -> Result<SplitManifest> {
    if !buffer_distance.is_finite() || buffer_distance < 0.0 {
        return Err(GeoCoreError::InvalidInput(
            "buffer_distance must be finite and non-negative".to_string(),
        ));
    }
    let base = spatial_block_cv(coords, n_folds, meta.clone(), split_id.clone())?;
    let mut folds = Vec::new();
    for fold in base.folds {
        let test = fold.test_indices;
        let train = fold
            .train_indices
            .into_iter()
            .filter(|idx| {
                !test
                    .iter()
                    .any(|test_idx| distance(coords, *idx, *test_idx) <= buffer_distance)
            })
            .collect();
        folds.push(SplitFold {
            fold_id: fold.fold_id,
            train_indices: train,
            test_indices: test,
        });
    }
    manifest(split_id, "buffered_spatial_cv", coords.len(), folds, meta)
}

pub fn group_spatial_cv(
    groups: Vec<String>,
    n_folds: usize,
    meta: GeoFrameMeta,
    split_id: String,
) -> Result<SplitManifest> {
    if groups.len() != meta.row_count {
        return Err(GeoCoreError::InvalidInput(
            "groups length must match row_count".to_string(),
        ));
    }
    let unique: Vec<String> = BTreeSet::from_iter(groups.iter().cloned())
        .into_iter()
        .collect();
    if n_folds < 2 || n_folds > unique.len() {
        return Err(GeoCoreError::InvalidInput(
            "n_folds must fit the number of groups".to_string(),
        ));
    }
    let mut folds = Vec::new();
    for fold in 0..n_folds {
        let held_out: BTreeSet<&String> = unique
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| (idx % n_folds == fold).then_some(value))
            .collect();
        let mut train = Vec::new();
        let mut test = Vec::new();
        for (idx, group) in groups.iter().enumerate() {
            if held_out.contains(group) {
                test.push(idx);
            } else {
                train.push(idx);
            }
        }
        folds.push(SplitFold {
            fold_id: format!("group_{fold}"),
            train_indices: train,
            test_indices: test,
        });
    }
    manifest(split_id, "group_spatial_cv", groups.len(), folds, meta)
}

pub fn rolling_origin_panel_split(
    panel: &PanelIndex,
    min_train_size: usize,
    horizon: usize,
    step: usize,
    meta: GeoFrameMeta,
    split_id: String,
) -> Result<SplitManifest> {
    if min_train_size == 0 || horizon == 0 || step == 0 || panel.len() != meta.row_count {
        return Err(GeoCoreError::InvalidInput(
            "invalid rolling panel split sizes".to_string(),
        ));
    }
    let mut by_entity: BTreeMap<&String, Vec<usize>> = BTreeMap::new();
    for (idx, entity) in panel.entity_ids.iter().enumerate() {
        by_entity.entry(entity).or_default().push(idx);
    }
    let min_len = by_entity.values().map(Vec::len).min().unwrap_or(0);
    if min_len < min_train_size + horizon {
        return Err(GeoCoreError::InvalidInput(
            "not enough panel rows for requested horizon".to_string(),
        ));
    }
    let mut folds = Vec::new();
    let mut origin = min_train_size;
    while origin + horizon <= min_len {
        let mut train = Vec::new();
        let mut test = Vec::new();
        for indices in by_entity.values() {
            train.extend_from_slice(&indices[..origin]);
            test.extend_from_slice(&indices[origin..origin + horizon]);
        }
        train.sort_unstable();
        test.sort_unstable();
        folds.push(SplitFold {
            fold_id: format!("origin_{origin}"),
            train_indices: train,
            test_indices: test,
        });
        origin += step;
    }
    manifest(
        split_id,
        "rolling_origin_panel_split",
        panel.len(),
        folds,
        meta,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn spatial_temporal_blocked_split(
    coords: &CoordinateMatrix,
    time: &TimeIndex,
    n_spatial_folds: usize,
    min_train_time: usize,
    horizon: usize,
    meta: GeoFrameMeta,
    split_id: String,
) -> Result<SplitManifest> {
    if coords.len() != time.len()
        || coords.len() != meta.row_count
        || min_train_time + horizon > time.len()
    {
        return Err(GeoCoreError::InvalidInput(
            "coordinate, time, and temporal split sizes are inconsistent".to_string(),
        ));
    }
    let spatial = spatial_block_cv(coords, n_spatial_folds, meta.clone(), split_id.clone())?;
    let mut ordered: Vec<usize> = (0..time.len()).collect();
    ordered.sort_by_key(|idx| time.timestamps[*idx]);
    let train_time: BTreeSet<usize> = ordered[..min_train_time].iter().copied().collect();
    let test_time: BTreeSet<usize> = ordered[min_train_time..min_train_time + horizon]
        .iter()
        .copied()
        .collect();
    let folds = spatial
        .folds
        .into_iter()
        .map(|fold| {
            let spatial_test: BTreeSet<usize> = fold.test_indices.into_iter().collect();
            SplitFold {
                fold_id: fold.fold_id,
                train_indices: train_time.difference(&spatial_test).copied().collect(),
                test_indices: test_time.intersection(&spatial_test).copied().collect(),
            }
        })
        .collect();
    manifest(
        split_id,
        "spatial_temporal_blocked_split",
        coords.len(),
        folds,
        meta,
    )
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    if let Ok(ts) = DateTime::parse_from_rfc3339(value) {
        return Ok(ts.with_timezone(&Utc));
    }
    if let Ok(ts) = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S") {
        return Ok(ts.and_utc());
    }
    if let Ok(date) = NaiveDate::parse_from_str(value, "%Y-%m-%d") {
        return Ok(date
            .and_hms_opt(0, 0, 0)
            .expect("midnight is valid")
            .and_utc());
    }
    Err(GeoCoreError::InvalidInput(format!(
        "invalid timestamp {value:?}"
    )))
}

fn validate_csr(
    n_nodes: usize,
    indptr: &[usize],
    indices: &[usize],
    data: &[f64],
    node_ids: Option<&Vec<String>>,
) -> Result<()> {
    if n_nodes == 0
        || indptr.len() != n_nodes + 1
        || indptr.first() != Some(&0)
        || indptr.last() != Some(&indices.len())
    {
        return Err(GeoCoreError::InvalidInput(
            "invalid CSR indptr for n_nodes/nnz".to_string(),
        ));
    }
    if indices.len() != data.len() || indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(GeoCoreError::InvalidInput("invalid CSR shape".to_string()));
    }
    if indices.iter().any(|idx| *idx >= n_nodes) || data.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(GeoCoreError::InvalidInput(
            "CSR indices and data must be valid".to_string(),
        ));
    }
    if let Some(ids) = node_ids {
        if ids.len() != n_nodes {
            return Err(GeoCoreError::InvalidInput(
                "node_ids length must match n_nodes".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_fold(fold: &SplitFold, row_count: usize) -> Result<()> {
    let train: BTreeSet<usize> = fold.train_indices.iter().copied().collect();
    let test: BTreeSet<usize> = fold.test_indices.iter().copied().collect();
    if fold.fold_id.is_empty()
        || train.is_empty()
        || test.is_empty()
        || train.len() != fold.train_indices.len()
        || test.len() != fold.test_indices.len()
        || !train.is_disjoint(&test)
    {
        return Err(GeoCoreError::InvalidInput("invalid split fold".to_string()));
    }
    if train.iter().chain(test.iter()).any(|idx| *idx >= row_count) {
        return Err(GeoCoreError::InvalidInput(
            "fold index outside row_count".to_string(),
        ));
    }
    Ok(())
}

fn partition_order(order: &[usize], n_folds: usize, prefix: &str) -> Vec<SplitFold> {
    (0..n_folds)
        .map(|fold| {
            let start = fold * order.len() / n_folds;
            let end = (fold + 1) * order.len() / n_folds;
            let test: BTreeSet<usize> = order[start..end].iter().copied().collect();
            let train = order
                .iter()
                .copied()
                .filter(|idx| !test.contains(idx))
                .collect();
            SplitFold {
                fold_id: format!("{prefix}_{fold}"),
                train_indices: train,
                test_indices: test.into_iter().collect(),
            }
        })
        .collect()
}

fn manifest(
    split_id: String,
    split_kind: &str,
    row_count: usize,
    folds: Vec<SplitFold>,
    meta: GeoFrameMeta,
) -> Result<SplitManifest> {
    SplitManifest::new(
        split_id,
        split_kind.to_string(),
        row_count,
        folds,
        meta.dataset_fingerprint,
        meta.coordinate_crs_note,
        meta.model_version,
        meta.dependency_versions,
        meta.random_seed,
    )
}

fn distance(coords: &CoordinateMatrix, a: usize, b: usize) -> f64 {
    euclidean_distance([coords.x[a], coords.y[a]], [coords.x[b], coords.y[b]])
}

pub fn squared_euclidean_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    dx * dx + dy * dy
}

pub fn euclidean_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    squared_euclidean_distance(left, right).sqrt()
}

pub fn clockwise_bearing_unit_vector(origin: [f64; 2], destination: [f64; 2]) -> Option<[f64; 2]> {
    let dx = destination[0] - origin[0];
    let dy = destination[1] - origin[1];
    let distance = (dx * dx + dy * dy).sqrt();
    if distance == 0.0 || !distance.is_finite() {
        return None;
    }
    Some([dx / distance, dy / distance])
}

pub fn route_feature_vector(origin: [f64; 2], destination: [f64; 2]) -> Option<[f64; 5]> {
    let bearing = clockwise_bearing_unit_vector(origin, destination)?;
    Some([
        0.5 * (origin[0] + destination[0]),
        0.5 * (origin[1] + destination[1]),
        euclidean_distance(origin, destination),
        bearing[0],
        bearing[1],
    ])
}

pub fn radial_anchor_distances(point: [f64; 2], anchors: &[[f64; 2]]) -> Vec<f64> {
    anchors
        .iter()
        .map(|anchor| euclidean_distance(point, *anchor))
        .collect()
}

pub fn rbf_anchor_features(
    point: [f64; 2],
    anchors: &[[f64; 2]],
    length_scale: f64,
) -> Result<Vec<f64>> {
    if !length_scale.is_finite() || length_scale <= 0.0 {
        return Err(GeoCoreError::InvalidInput(
            "length_scale must be finite and positive".to_string(),
        ));
    }
    Ok(anchors
        .iter()
        .map(|anchor| {
            let distance_sq = squared_euclidean_distance(point, *anchor);
            (-0.5 * distance_sq / (length_scale * length_scale)).exp()
        })
        .collect())
}

pub fn local_frame_features(point: [f64; 2], origin: [f64; 2], axis: [f64; 2]) -> Option<[f64; 2]> {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    let east = axis[0] / norm;
    let north = axis[1] / norm;
    let dx = point[0] - origin[0];
    let dy = point[1] - origin[1];
    let along = dx * east + dy * north;
    let cross = -dx * north + dy * east;
    Some([along, cross])
}

pub fn initial_bearing_unit_vector_latlng(
    origin_latitude: f64,
    origin_longitude: f64,
    destination_latitude: f64,
    destination_longitude: f64,
) -> Option<[f64; 2]> {
    let lat1 = origin_latitude.to_radians();
    let lat2 = destination_latitude.to_radians();
    let dlon = (destination_longitude - origin_longitude).to_radians();
    let east = dlon.sin() * lat2.cos();
    let north = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * dlon.cos();
    let norm = (east * east + north * north).sqrt();
    if norm == 0.0 || !norm.is_finite() {
        return None;
    }
    Some([east / norm, north / norm])
}

pub fn anisotropic_euclidean_distance(
    left: [f64; 2],
    right: [f64; 2],
    angle_degrees: f64,
    scaling: f64,
) -> f64 {
    let left = transform_anisotropic_point(left, angle_degrees, scaling);
    let right = transform_anisotropic_point(right, angle_degrees, scaling);
    euclidean_distance(left, right)
}

pub fn transform_anisotropic_point(point: [f64; 2], angle_degrees: f64, scaling: f64) -> [f64; 2] {
    let angle = angle_degrees.to_radians();
    let cos = angle.cos();
    let sin = angle.sin();
    let x = point[0] * cos + point[1] * sin;
    let y = (-point[0] * sin + point[1] * cos) / scaling;
    [x, y]
}

pub fn haversine_distance_km(
    origin_latitude: f64,
    origin_longitude: f64,
    destination_latitude: f64,
    destination_longitude: f64,
) -> f64 {
    let lat1 = origin_latitude.to_radians();
    let lon1 = origin_longitude.to_radians();
    let lat2 = destination_latitude.to_radians();
    let lon2 = destination_longitude.to_radians();
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_KM * a.sqrt().asin()
}

pub fn haversine_distance_meters(
    origin_latitude: f64,
    origin_longitude: f64,
    destination_latitude: f64,
    destination_longitude: f64,
) -> f64 {
    haversine_distance_km(
        origin_latitude,
        origin_longitude,
        destination_latitude,
        destination_longitude,
    ) * 1_000.0
}

fn stable_hex64(bytes: &[u8]) -> String {
    let mut text = String::new();
    for salt in 0_u64..4 {
        let mut hash = 0xcbf29ce484222325_u64 ^ salt;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        text.push_str(&format!("{hash:016x}"));
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(row_count: usize) -> GeoFrameMeta {
        GeoFrameMeta::new(
            "sha256:test".to_string(),
            "EPSG:2263".to_string(),
            "0.2.32".to_string(),
            BTreeMap::from([("cartoboost".to_string(), "0.2.32".to_string())]),
            Some(42),
            row_count,
            None,
        )
        .unwrap()
    }

    #[test]
    fn containers_round_trip() {
        let coords = CoordinateMatrix::new(
            vec![0.0, 1.0],
            vec![2.0, 3.0],
            Some("EPSG:4326".to_string()),
            None,
        )
        .unwrap();
        assert_eq!(
            CoordinateMatrix::from_json_str(&coords.to_json_string().unwrap()).unwrap(),
            coords
        );
        let time = TimeIndex::new(
            vec!["2024-01-01".to_string(), "2024-01-02".to_string()],
            Some("D".to_string()),
            None,
        )
        .unwrap();
        let panel =
            PanelIndex::new(vec!["zone_a".to_string(), "zone_a".to_string()], Some(time)).unwrap();
        assert_eq!(
            PanelIndex::from_json_str(&panel.to_json_string().unwrap()).unwrap(),
            panel
        );
    }

    #[test]
    fn spatial_weights_csr_behaviors() {
        let weights = SpatialWeights::from_edges(3, vec![(0, 1, 2.0), (1, 0, 2.0)], false).unwrap();
        assert!(weights.is_symmetric(0.0));
        assert_eq!(weights.isolated_nodes(), vec![2]);
        let normalized = weights.row_normalize();
        assert_eq!(normalized.data, vec![1.0, 1.0]);
        assert_eq!(
            SpatialWeights::from_json_str(&normalized.to_json_string().unwrap()).unwrap(),
            normalized
        );
    }

    #[test]
    fn shared_distance_helpers_are_stable() {
        assert_eq!(squared_euclidean_distance([0.0, 0.0], [3.0, 4.0]), 25.0);
        assert_eq!(euclidean_distance([0.0, 0.0], [3.0, 4.0]), 5.0);
        assert_eq!(
            clockwise_bearing_unit_vector([0.0, 0.0], [0.0, 5.0]),
            Some([0.0, 1.0])
        );
        assert_eq!(
            clockwise_bearing_unit_vector([0.0, 0.0], [5.0, 0.0]),
            Some([1.0, 0.0])
        );
        assert_eq!(clockwise_bearing_unit_vector([1.0, 1.0], [1.0, 1.0]), None);
        assert_eq!(
            route_feature_vector([0.0, 0.0], [3.0, 4.0]),
            Some([1.5, 2.0, 5.0, 0.6, 0.8])
        );
        assert_eq!(
            radial_anchor_distances([3.0, 4.0], &[[0.0, 0.0], [3.0, 0.0]]),
            vec![5.0, 4.0]
        );
        assert_eq!(
            rbf_anchor_features([0.0, 0.0], &[[0.0, 0.0], [1.0, 0.0]], 1.0).unwrap(),
            vec![1.0, (-0.5_f64).exp()]
        );
        assert!(rbf_anchor_features([0.0, 0.0], &[[0.0, 0.0]], 0.0).is_err());
        assert_eq!(
            local_frame_features([2.0, 3.0], [1.0, 1.0], [0.0, 1.0]),
            Some([2.0, -1.0])
        );
        assert_eq!(
            transform_anisotropic_point([3.0, 4.0], 0.0, 2.0),
            [3.0, 2.0]
        );
        assert!(
            (anisotropic_euclidean_distance([0.0, 0.0], [3.0, 4.0], 0.0, 2.0) - 13.0_f64.sqrt())
                .abs()
                < 1.0e-12
        );
        let nyc_to_london_km = haversine_distance_km(40.7128, -74.0060, 51.5074, -0.1278);
        assert!((nyc_to_london_km - 5_570.0).abs() < 20.0);
        let north = initial_bearing_unit_vector_latlng(40.0, -73.0, 41.0, -73.0).unwrap();
        assert!(north[0].abs() < 1.0e-12);
        assert!((north[1] - 1.0).abs() < 1.0e-12);
        let northwest = initial_bearing_unit_vector_latlng(40.0, -73.0, 41.0, -74.0).unwrap();
        assert!(northwest[0] < 0.0);
        assert!(northwest[1] > 0.0);
        assert!((northwest[0] * northwest[0] + northwest[1] * northwest[1] - 1.0).abs() < 1.0e-12);
        assert!(
            (haversine_distance_meters(40.7128, -74.0060, 51.5074, -0.1278)
                - nyc_to_london_km * 1_000.0)
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn split_manifest_is_deterministic() {
        let coords =
            CoordinateMatrix::new(vec![0.0, 1.0, 2.0, 3.0], vec![0.0; 4], None, None).unwrap();
        let a = spatial_block_cv(&coords, 2, meta(4), "spatial_block".to_string()).unwrap();
        let b = spatial_block_cv(&coords, 2, meta(4), "spatial_block".to_string()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.hash().unwrap(), b.hash().unwrap());
        assert!(a.hash().unwrap().starts_with("sha256:"));
    }
}
