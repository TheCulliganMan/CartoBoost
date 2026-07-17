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

