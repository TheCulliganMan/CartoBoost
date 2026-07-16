fn graph_in_degrees(adjacency: &CsrAdjacency, nodes: usize) -> Vec<f64> {
    let mut result = vec![0.0; nodes];
    for target in &adjacency.indices {
        result[*target] += 1.0;
    }
    result
}

fn graph_out_degrees(adjacency: &CsrAdjacency, nodes: usize) -> Vec<f64> {
    (0..nodes)
        .map(|node| (adjacency.indptr[node + 1] - adjacency.indptr[node]) as f64)
        .collect()
}

/// Draw a reproducible 75% patch mask.  Randomized patch selection mirrors
/// LSTTN's pretraining policy without making fitted artifacts nondeterministic.
fn masked_patch_indices(patches: usize, step: u64) -> Vec<usize> {
    let mut indices = (0..patches).collect::<Vec<_>>();
    let mut state = step ^ 0x9e37_79b9_7f4a_7c15;
    for index in (1..indices.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        indices.swap(index, (state as usize) % (index + 1));
    }
    let masked_count = (patches * 3).div_ceil(4).min(patches.saturating_sub(1));
    indices.truncate(masked_count);
    indices.sort_unstable();
    indices
}

fn graph_temporal_training_fingerprint(frame: &GraphTemporalFrame) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    frame.node_ids.hash(&mut hasher);
    frame.timestamps.hash(&mut hasher);
    frame.horizon.hash(&mut hasher);
    frame.frequency.hash(&mut hasher);
    frame.adjacency.indptr.hash(&mut hasher);
    frame.adjacency.indices.hash(&mut hasher);
    for value in frame
        .target
        .iter()
        .flatten()
        .chain(frame.adjacency.data.iter())
        .chain(frame.covariates.iter().flatten().flatten().flatten())
    {
        value.to_bits().hash(&mut hasher);
    }
    hasher.finish()
}

fn write_lsttn_checkpoint(path: &Path, checkpoint: &LsttnTrainingCheckpoint) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.part");
    fs::write(&temporary, serde_json::to_vec(checkpoint)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

/// Divide one recurring traffic cycle into contiguous graph environments.  We
/// first estimate each phase's graph relation from the rank ordering of node
/// signals across all observed cycles, then use dynamic programming to choose
/// the contiguous partition with maximal within-environment Kendall coherence.
/// The resulting labels are used as the known source environments during the
/// spatial-shift paper's episodic expert training policy.
fn maximum_spatiotemporal_graph_division(
    values: &[Vec<f64>],
    periodicity: usize,
    experts: usize,
) -> Result<Vec<usize>> {
    if values.len() < periodicity {
        return Err(GeoStError::InvalidFrame(format!(
            "spatial-shift graph division requires at least one complete period ({periodicity} observations)"
        )));
    }
    let nodes = values
        .first()
        .ok_or_else(|| {
            GeoStError::InvalidFrame("graph division requires observations".to_string())
        })?
        .len();
    let signatures = (0..periodicity)
        .map(|phase| {
            let rows = values
                .iter()
                .skip(phase)
                .step_by(periodicity)
                .collect::<Vec<_>>();
            let mean = (0..nodes)
                .map(|node| rows.iter().map(|row| row[node]).sum::<f64>() / rows.len() as f64)
                .collect::<Vec<_>>();
            rank_signature(&mean)
        })
        .collect::<Vec<_>>();
    let groups = experts.min(periodicity).max(1);
    let mut score = vec![vec![0.0; periodicity + 1]; periodicity];
    for (start, score_row) in score.iter_mut().enumerate().take(periodicity) {
        for (end, segment_score) in score_row
            .iter_mut()
            .enumerate()
            .take(periodicity + 1)
            .skip(start + 1)
        {
            let mut total = 0.0;
            let mut count = 0usize;
            for left in start..end {
                for right in left + 1..end {
                    total += kendall_tau(&signatures[left], &signatures[right]);
                    count += 1;
                }
            }
            *segment_score = if count == 0 {
                0.0
            } else {
                total / count as f64
            };
        }
    }
    let mut dp = vec![vec![f64::NEG_INFINITY; periodicity + 1]; groups + 1];
    let mut parent = vec![vec![0usize; periodicity + 1]; groups + 1];
    dp[0][0] = 0.0;
    for group in 1..=groups {
        for end in group..=periodicity {
            for start in group - 1..end {
                let candidate = dp[group - 1][start] + score[start][end];
                if candidate > dp[group][end] {
                    dp[group][end] = candidate;
                    parent[group][end] = start;
                }
            }
        }
    }
    let mut boundaries = Vec::with_capacity(groups + 1);
    let mut end = periodicity;
    boundaries.push(end);
    for group in (1..=groups).rev() {
        end = parent[group][end];
        boundaries.push(end);
    }
    boundaries.reverse();
    let mut labels = vec![0usize; periodicity];
    for group in 0..groups {
        for label in labels
            .iter_mut()
            .take(boundaries[group + 1])
            .skip(boundaries[group])
        {
            *label = group;
        }
    }
    Ok(labels)
}

fn rank_signature(values: &[f64]) -> Vec<usize> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        values[*left]
            .total_cmp(&values[*right])
            .then_with(|| left.cmp(right))
    });
    let mut ranks = vec![0usize; values.len()];
    for (rank, node) in order.into_iter().enumerate() {
        ranks[node] = rank;
    }
    ranks
}

fn kendall_tau(left: &[usize], right: &[usize]) -> f64 {
    if left.len() < 2 || left.len() != right.len() {
        return 0.0;
    }
    let mut concordant = 0isize;
    let mut discordant = 0isize;
    for first in 0..left.len() {
        for second in first + 1..left.len() {
            let left_order = left[first].cmp(&left[second]);
            let right_order = right[first].cmp(&right[second]);
            if left_order == right_order {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    (concordant - discordant) as f64 / (left.len() * (left.len() - 1) / 2) as f64
}

pub fn graph_metrics(
    predictions: &[Vec<f64>],
    actual: &[Vec<f64>],
    node_ids: &[String],
    adjacency: &CsrAdjacency,
) -> GraphForecastMetrics {
    let horizons = predictions.len().min(actual.len());
    let nodes = node_ids.len();
    let by_horizon = (0..horizons)
        .map(|h| {
            let (mae, rmse, wape) = metric_values(&predictions[h], &actual[h]);
            HorizonMetric {
                horizon: h + 1,
                mae,
                rmse,
                wape,
            }
        })
        .collect();
    let by_node = (0..nodes)
        .map(|node| {
            let pred: Vec<f64> = (0..horizons).map(|h| predictions[h][node]).collect();
            let obs: Vec<f64> = (0..horizons).map(|h| actual[h][node]).collect();
            let (mae, rmse, wape) = metric_values(&pred, &obs);
            NodeMetric {
                node_id: node_ids[node].clone(),
                mae,
                rmse,
                wape,
            }
        })
        .collect();
    GraphForecastMetrics {
        by_horizon,
        by_node,
        graph_distance_residuals: distance_residuals(predictions, actual, adjacency, nodes),
    }
}

pub fn synthetic_graph_diffusion_frame() -> GraphTemporalFrame {
    let nodes = 4;
    let adjacency = CsrAdjacency::new(vec![0, 1, 2, 3, 4], vec![1, 2, 3, 0], vec![1.0; 4], nodes)
        .expect("fixture adjacency");
    let mut target = Vec::new();
    for t in 0..80 {
        let mut row = Vec::with_capacity(nodes);
        for node in 0..nodes {
            let phase = (t as f64 - node as f64) * 0.45;
            let upstream_phase = (t as f64 - (node + 1) as f64) * 0.45;
            row.push(12.0 + 2.4 * phase.sin() + 1.1 * upstream_phase.cos());
        }
        target.push(row);
    }
    GraphTemporalFrame::new(
        (0..nodes).map(|idx| format!("zone_{idx}")).collect(),
        (0..80).map(i64::from).collect(),
        target,
        None,
        adjacency,
        3,
        "hourly".to_string(),
    )
    .expect("fixture frame")
}

pub fn traffic_style_fixture_frame() -> GraphTemporalFrame {
    let adjacency = CsrAdjacency::new(
        vec![0, 2, 3, 4],
        vec![1, 2, 2, 0],
        vec![0.7, 0.3, 1.0, 1.0],
        3,
    )
    .expect("traffic adjacency");
    let mut target = Vec::new();
    for t in 0..48 {
        let hour = t as f64;
        target.push(vec![
            18.0 + (hour / 24.0 * std::f64::consts::TAU).sin() * 4.0,
            16.0 + ((hour - 1.0) / 24.0 * std::f64::consts::TAU).sin() * 3.5,
            14.0 + ((hour - 2.0) / 24.0 * std::f64::consts::TAU).sin() * 3.0,
        ]);
    }
    GraphTemporalFrame::new(
        vec!["sensor_a".into(), "sensor_b".into(), "sensor_c".into()],
        (0..48).map(i64::from).collect(),
        target,
        None,
        adjacency,
        4,
        "hourly".to_string(),
    )
    .expect("traffic frame")
}

fn dot(weights: &[f64], values: &[f64]) -> f64 {
    weights.iter().zip(values.iter()).map(|(w, v)| w * v).sum()
}

fn attention_pool(values: &[f64], query: &[f64], key: &[f64]) -> f64 {
    let width = values.len().min(query.len()).min(key.len());
    let scores = (0..width)
        .map(|idx| (query[idx] * values[idx] + key[idx]).tanh())
        .collect::<Vec<_>>();
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights = scores
        .iter()
        .map(|score| (score - max_score).exp())
        .collect::<Vec<_>>();
    let denom = weights.iter().sum::<f64>().max(1.0e-12);
    weights
        .iter()
        .zip(values)
        .take(width)
        .map(|(weight, value)| weight / denom * value)
        .sum()
}

fn sigmoid(value: f64) -> f64 {
    1.0 / (1.0 + (-value).exp())
}

fn blend_rows(actual: &[f64], predicted: &[f64], teacher_ratio: f64) -> Vec<f64> {
    actual
        .iter()
        .zip(predicted.iter())
        .map(|(&teacher, &model)| teacher_ratio * teacher + (1.0 - teacher_ratio) * model)
        .collect()
}

fn deterministic_weight_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    let scale = (cols.max(1) as f64).sqrt().recip();
    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                    (unit * 2.0 - 1.0) * scale
                })
                .collect()
        })
        .collect()
}

fn csr_edges(adjacency: &CsrAdjacency, node_count: usize) -> Vec<(usize, usize)> {
    let mut edges = Vec::with_capacity(adjacency.indices.len());
    for source in 0..node_count {
        for edge_idx in adjacency.indptr[source]..adjacency.indptr[source + 1] {
            edges.push((source, adjacency.indices[edge_idx]));
        }
    }
    edges
}

fn delayed_graph_signal(
    target: &[Vec<f64>],
    edges: &[(usize, usize)],
    weights: &[f64],
    delays: &[usize],
    time_idx: usize,
) -> Vec<f64> {
    let nodes = target.first().map_or(0, Vec::len);
    let mut signal = vec![0.0; nodes];
    let mut weight_sum = vec![0.0; nodes];
    for (edge_idx, &(source, target_node)) in edges.iter().enumerate() {
        let delay = delays[edge_idx];
        let lag_idx = (time_idx + 1).saturating_sub(delay);
        let weight = weights[edge_idx];
        signal[target_node] += weight * target[lag_idx][source];
        weight_sum[target_node] += weight.abs();
    }
    for node in 0..nodes {
        if weight_sum[node] > 1.0e-12 {
            signal[node] /= weight_sum[node];
        }
    }
    signal
}

fn quantize_prediction(value: f64) -> f64 {
    if value.is_finite() {
        (value * 1.0e12).round() / 1.0e12
    } else {
        value
    }
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        if best != pivot {
            matrix.swap(pivot, best);
            rhs.swap(pivot, best);
        }
        let diag = matrix[pivot][pivot];
        if diag.abs() < 1.0e-12 {
            continue;
        }
        for value in matrix[pivot].iter_mut().take(n).skip(pivot) {
            *value /= diag;
        }
        rhs[pivot] /= diag;
        let pivot_row = matrix[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for (col, pivot_value) in pivot_row.iter().enumerate().take(n).skip(pivot) {
                matrix[row][col] -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    rhs
}

fn target_center_scale(target: &[Vec<f64>]) -> (f64, f64) {
    let mut count = 0.0;
    let mut sum = 0.0;
    for row in target {
        for value in row {
            count += 1.0;
            sum += value;
        }
    }
    let mean = if count > 0.0 { sum / count } else { 0.0 };
    let mut variance = 0.0;
    for row in target {
        for value in row {
            let centered = value - mean;
            variance += centered * centered;
        }
    }
    let scale = if count > 0.0 {
        (variance / count).sqrt().max(1.0e-6)
    } else {
        1.0
    };
    (mean, scale)
}

fn metric_values(predictions: &[f64], actual: &[f64]) -> (f64, f64, f64) {
    let n = predictions.len().max(1) as f64;
    let mut abs = 0.0;
    let mut squared = 0.0;
    let mut denom = 0.0;
    for (&pred, &obs) in predictions.iter().zip(actual.iter()) {
        let err = pred - obs;
        abs += err.abs();
        squared += err * err;
        denom += obs.abs();
    }
    (
        abs / n,
        (squared / n).sqrt(),
        if denom > 0.0 { abs / denom } else { 0.0 },
    )
}

fn distance_residuals(
    predictions: &[Vec<f64>],
    actual: &[Vec<f64>],
    adjacency: &CsrAdjacency,
    nodes: usize,
) -> Vec<GraphDistanceResidual> {
    let distances = graph_distances(adjacency, nodes);
    let mut sums = vec![0.0; nodes];
    let mut counts = vec![0usize; nodes];
    for distance_row in distances.iter().take(nodes) {
        for (target, distance) in distance_row.iter().enumerate().take(nodes) {
            let distance = *distance;
            if distance < nodes {
                for h in 0..predictions.len().min(actual.len()) {
                    sums[distance] += (predictions[h][target] - actual[h][target]).abs();
                    counts[distance] += 1;
                }
            }
        }
    }
    sums.into_iter()
        .zip(counts)
        .enumerate()
        .filter_map(|(distance, (sum, count))| {
            (count > 0).then_some(GraphDistanceResidual {
                distance,
                mean_abs_residual: sum / count as f64,
                count,
            })
        })
        .collect()
}

fn graph_distances(adjacency: &CsrAdjacency, nodes: usize) -> Vec<Vec<usize>> {
    let mut all = vec![vec![usize::MAX / 4; nodes]; nodes];
    for (source, distances) in all.iter_mut().enumerate().take(nodes) {
        let mut queue = VecDeque::new();
        distances[source] = 0;
        queue.push_back(source);
        while let Some(node) = queue.pop_front() {
            let next_distance = distances[node] + 1;
            for edge in adjacency.indptr[node]..adjacency.indptr[node + 1] {
                let next = adjacency.indices[edge];
                if distances[next] > next_distance {
                    distances[next] = next_distance;
                    queue.push_back(next);
                }
            }
        }
    }
    all
}

