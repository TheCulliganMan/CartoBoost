fn looks_like_periodic_feature(
    x: &Dataset,
    indices: &[usize],
    feature: usize,
    period: f64,
) -> bool {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut count = 0usize;
    for &idx in indices {
        let value = x.get(idx, feature);
        if !value.is_finite() {
            continue;
        }
        if value < 0.0 || value > period {
            return false;
        }
        min = min.min(value);
        max = max.max(value);
        count += 1;
    }
    count >= 2 && min <= period * 0.25 && max >= period * 0.75
}

fn active_row_mask(rows: usize, indices: &[usize]) -> Vec<bool> {
    let mut active = vec![false; rows];
    for &idx in indices {
        active[idx] = true;
    }
    active
}

fn split_feature_indices(split: &Split, dense_feature_count: usize) -> Vec<usize> {
    match split {
        Split::Axis { feature, .. }
        | Split::PeriodicInterval { feature, .. }
        | Split::SparseSetContainsAny { feature, .. } => vec![*feature],
        Split::Diagonal2D {
            x_feature,
            y_feature,
            ..
        }
        | Split::Gaussian2D {
            x_feature,
            y_feature,
            ..
        } => {
            let mut features = vec![*x_feature, *y_feature];
            features.sort_unstable();
            features.dedup();
            features
        }
        Split::SparseListContainsAny { sparse_feature, .. } => {
            vec![dense_feature_count + *sparse_feature]
        }
        Split::Fuzzy { base, .. } => split_feature_indices(base, dense_feature_count),
    }
}

fn quantile_histogram_thresholds(mut values: Vec<f64>, bin_count: usize) -> Vec<f64> {
    if bin_count < 2 || values.len() < 2 {
        return Vec::new();
    }
    values.sort_by(f64::total_cmp);
    let mut unique_values = values.clone();
    unique_values.dedup();
    if unique_values.len() <= bin_count {
        return adjacent_value_thresholds(&unique_values);
    }
    if unique_values.len() * 10 <= values.len() {
        return fixed_width_histogram_thresholds(&values, bin_count);
    }
    if unique_values
        .iter()
        .all(|value| value.fract().abs() < 1e-12)
    {
        return fixed_width_histogram_thresholds(&values, bin_count);
    }

    let mut thresholds = Vec::with_capacity(bin_count - 1);
    let mut previous_threshold: Option<f64> = None;
    let last_index = values.len() - 1;
    for split in 1..bin_count {
        let mut index = (split * values.len()) / bin_count;
        if index == 0 {
            index = 1;
        } else if index > last_index {
            index = last_index;
        }

        while index <= last_index && values[index - 1] == values[index] {
            index += 1;
        }
        if index > last_index {
            break;
        }

        let threshold = (values[index - 1] + values[index]) / 2.0;
        if previous_threshold.is_none_or(|previous| threshold > previous) {
            thresholds.push(threshold);
            previous_threshold = Some(threshold);
        }
    }
    thresholds
}

fn adjacent_value_thresholds(values: &[f64]) -> Vec<f64> {
    values
        .windows(2)
        .filter_map(|window| {
            let threshold = (window[0] + window[1]) / 2.0;
            threshold.is_finite().then_some(threshold)
        })
        .collect()
}

fn fixed_width_histogram_thresholds(sorted_values: &[f64], bin_count: usize) -> Vec<f64> {
    let Some(&min_value) = sorted_values.first() else {
        return Vec::new();
    };
    let Some(&max_value) = sorted_values.last() else {
        return Vec::new();
    };
    if !min_value.is_finite() || min_value >= max_value {
        return Vec::new();
    }
    let scale = bin_count as f64 / (max_value - min_value);
    (0..(bin_count - 1))
        .filter_map(|split_bin| {
            let threshold = min_value + ((split_bin + 1) as f64 / scale);
            (threshold < max_value).then_some(threshold)
        })
        .collect()
}

fn prebinned_histogram_feature(
    x: &Dataset,
    feature: usize,
    bin_count: usize,
) -> Option<HistogramFeature> {
    if !dense_feature_allows_axis(x, feature) {
        return None;
    }

    let mut values = Vec::with_capacity(x.n_rows());
    for row in 0..x.n_rows() {
        let value = x.get(row, feature);
        if value.is_finite() {
            values.push(value);
        }
    }
    let thresholds = quantile_histogram_thresholds(values, bin_count);
    if thresholds.is_empty() {
        return None;
    }

    let bins = (0..x.n_rows())
        .map(|row| {
            let value = x.get(row, feature);
            if value.is_finite() {
                thresholds.partition_point(|threshold| value > *threshold) as u16
            } else {
                MISSING_BIN
            }
        })
        .collect();
    Some(HistogramFeature {
        bin_count: thresholds.len() + 1,
        thresholds,
        bins,
    })
}

fn dense_feature_kind(x: &Dataset, feature: usize) -> Option<&FeatureKind> {
    x.feature_schema()
        .and_then(|schema| schema.kinds.get(feature))
}

fn sparse_feature_kind(x: &Dataset, sparse_feature: usize) -> Option<&FeatureKind> {
    x.feature_schema()
        .and_then(|schema| schema.kinds.get(x.n_cols() + sparse_feature))
}

fn dense_feature_allows_axis(x: &Dataset, feature: usize) -> bool {
    !matches!(dense_feature_kind(x, feature), Some(FeatureKind::SparseSet))
}

fn dense_feature_allows_spatial(x: &Dataset, feature: usize) -> bool {
    matches!(
        dense_feature_kind(x, feature),
        None | Some(FeatureKind::Numeric) | Some(FeatureKind::Spatial)
    )
}

fn spatial_feature_indices(x: &Dataset) -> Vec<usize> {
    if let Some(schema) = x.feature_schema() {
        let spatial = (0..x.n_cols())
            .filter(|&feature| matches!(schema.kinds.get(feature), Some(FeatureKind::Spatial)))
            .collect::<Vec<_>>();
        if !spatial.is_empty() {
            return spatial;
        }
        return (0..x.n_cols())
            .filter(|&feature| dense_feature_allows_spatial(x, feature))
            .collect();
    }

    if x.n_cols() >= 2 {
        vec![0, 1]
    } else {
        Vec::new()
    }
}

fn dense_feature_allows_sparse_set(x: &Dataset, feature: usize) -> bool {
    match x.feature_schema() {
        Some(_) if x.n_sparse_sets() > 0 => {
            matches!(dense_feature_kind(x, feature), Some(FeatureKind::SparseSet))
        }
        Some(_) => matches!(
            dense_feature_kind(x, feature),
            None | Some(FeatureKind::Numeric)
        ),
        None => true,
    }
}

impl CandidateStats {
    #[inline(always)]
    fn add_row(&mut self, idx: usize, target: &[f64], weights: &[f64]) {
        let weight = weights[idx];
        let value = target[idx];
        self.count += 1;
        self.weight_sum += weight;
        self.weighted_target_sum += weight * value;
        self.weighted_target_square_sum += weight * value * value;
    }

    #[inline(always)]
    fn merge(&mut self, other: Self) {
        self.count += other.count;
        self.weight_sum += other.weight_sum;
        self.weighted_target_sum += other.weighted_target_sum;
        self.weighted_target_square_sum += other.weighted_target_square_sum;
    }

    #[inline(always)]
    fn minus(&self, other: &Self) -> Self {
        Self {
            count: self.count - other.count,
            weight_sum: self.weight_sum - other.weight_sum,
            weighted_target_sum: self.weighted_target_sum - other.weighted_target_sum,
            weighted_target_square_sum: self.weighted_target_square_sum
                - other.weighted_target_square_sum,
        }
    }

    #[inline(always)]
    fn sse(&self) -> f64 {
        weighted_sse_from_sums(
            self.weight_sum,
            self.weighted_target_sum,
            self.weighted_target_square_sum,
        )
    }
}

#[inline(always)]
fn constant_leaf_value(stats: CandidateStats, lambda_l2: f64) -> f64 {
    if stats.weight_sum <= 0.0 {
        0.0
    } else {
        stats.weighted_target_sum / (stats.weight_sum + lambda_l2.max(0.0))
    }
}

#[inline(always)]
fn constant_leaf_node(stats: CandidateStats, lambda_l2: f64) -> Node {
    Node::Leaf {
        value: constant_leaf_value(stats, lambda_l2),
        sample_weight_sum: stats.weight_sum,
        training_loss: stats.sse(),
    }
}

#[inline(always)]
fn add_histogram_stats_row(
    context: &FitContext,
    bins: usize,
    target: &[f64],
    weights: &[f64],
    idx: usize,
    stats: &mut [CandidateStats],
) {
    let weight = weights[idx];
    let value = target[idx];
    let weighted_target = weight * value;
    let weighted_target_square = weighted_target * value;
    let row_offset = idx * context.cols;
    macro_rules! add_feature {
        ($feature:expr) => {{
            let bin = usize::from(context.histogram_row_bins[row_offset + $feature]);
            if bin != usize::from(MISSING_BIN) {
                let item = &mut stats[($feature * bins) + bin];
                item.count += 1;
                item.weight_sum += weight;
                item.weighted_target_sum += weighted_target;
                item.weighted_target_square_sum += weighted_target_square;
            }
        }};
    }
    match context.cols {
        3 => {
            add_feature!(0);
            add_feature!(1);
            add_feature!(2);
            return;
        }
        4 => {
            add_feature!(0);
            add_feature!(1);
            add_feature!(2);
            add_feature!(3);
            return;
        }
        6 => {
            add_feature!(0);
            add_feature!(1);
            add_feature!(2);
            add_feature!(3);
            add_feature!(4);
            add_feature!(5);
            return;
        }
        8 => {
            add_feature!(0);
            add_feature!(1);
            add_feature!(2);
            add_feature!(3);
            add_feature!(4);
            add_feature!(5);
            add_feature!(6);
            add_feature!(7);
            return;
        }
        _ => {}
    }
    for (feature, &bin) in context.histogram_row_bins[row_offset..row_offset + context.cols]
        .iter()
        .enumerate()
    {
        if bin == MISSING_BIN {
            continue;
        }
        let item = &mut stats[feature * bins + usize::from(bin)];
        item.count += 1;
        item.weight_sum += weight;
        item.weighted_target_sum += weighted_target;
        item.weighted_target_square_sum += weighted_target_square;
    }
}

fn histogram_stats_for_indices(
    context: &FitContext,
    bins: usize,
    target: &[f64],
    weights: &[f64],
    indices: &[usize],
) -> Vec<CandidateStats> {
    if indices.len() < 32_768 {
        let mut stats = vec![CandidateStats::default(); context.cols * bins];
        for &idx in indices {
            add_histogram_stats_row(context, bins, target, weights, idx, &mut stats);
        }
        return stats;
    }
    let chunk_size = (indices.len() / rayon::current_num_threads().max(1)).max(16_384);
    let partials = indices
        .par_chunks(chunk_size)
        .map(|chunk| {
            let mut partial = vec![CandidateStats::default(); context.cols * bins];
            for &idx in chunk {
                add_histogram_stats_row(context, bins, target, weights, idx, &mut partial);
            }
            partial
        })
        .collect::<Vec<_>>();
    let mut stats = vec![CandidateStats::default(); context.cols * bins];
    for partial in partials {
        for (total, item) in stats.iter_mut().zip(partial) {
            total.merge(item);
        }
    }
    stats
}

#[inline(always)]
fn subtract_histogram_stats(
    parent: &[CandidateStats],
    child: &[CandidateStats],
) -> Vec<CandidateStats> {
    parent
        .iter()
        .zip(child)
        .map(|(parent, child)| parent.minus(child))
        .collect()
}

#[inline(always)]
fn histogram_node_stats(context: &FitContext, stats: &[CandidateStats]) -> Option<CandidateStats> {
    let bins = context.histogram_bins?;
    let feature = *context.histogram_feature_indices.first()?;
    let start = feature * bins;
    (start + bins <= stats.len()).then(|| histogram_node_stats_from_feature(feature, bins, stats))
}

#[inline(always)]
fn histogram_node_stats_from_feature(
    feature: usize,
    bins: usize,
    stats: &[CandidateStats],
) -> CandidateStats {
    let start = feature * bins;
    stats[start..start + bins]
        .iter()
        .fold(CandidateStats::default(), |mut total, stats| {
            total.count += stats.count;
            total.weight_sum += stats.weight_sum;
            total.weighted_target_sum += stats.weighted_target_sum;
            total.weighted_target_square_sum += stats.weighted_target_square_sum;
            total
        })
}

fn candidate_stats(
    indices: impl Iterator<Item = usize>,
    target: &[f64],
    weights: &[f64],
) -> CandidateStats {
    let mut stats = CandidateStats::default();
    for idx in indices {
        stats.add_row(idx, target, weights);
    }
    stats
}

fn materialize_sparse_list_split(
    sparse_feature: usize,
    id: u64,
    x: &Dataset,
    indices: &[usize],
    best: &mut Option<BestSplit>,
) {
    let Some(best) = best.as_mut() else {
        return;
    };
    best.left.clear();
    best.right.clear();
    best.left_weights = None;
    best.right_weights = None;
    for &idx in indices {
        if x.sparse_set_contains_any(idx, sparse_feature, &[id]) {
            best.left.push(idx);
        } else {
            best.right.push(idx);
        }
    }
}

fn materialize_dense_sparse_split(
    feature: usize,
    id: u64,
    x: &Dataset,
    indices: &[usize],
    best: &mut Option<BestSplit>,
) {
    let Some(best) = best.as_mut() else {
        return;
    };
    best.left.clear();
    best.right.clear();
    best.left_weights = None;
    best.right_weights = None;
    for &idx in indices {
        if super::sparse_set_value_contains_any(x.get(idx, feature), &[id]) {
            best.left.push(idx);
        } else {
            best.right.push(idx);
        }
    }
}

fn materialize_axis_split(
    feature: usize,
    threshold: f64,
    x: &Dataset,
    indices: &[usize],
    best: &mut Option<BestSplit>,
) {
    let started = profile::ProfileTimer::start();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &idx in indices {
        if x.get(idx, feature) <= threshold {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }
    *best = Some(BestSplit {
        split: Split::Axis {
            feature,
            threshold,
            missing_goes_left: true,
        },
        gain: 0.0,
        left,
        right,
        left_direct_node: None,
        right_direct_node: None,
        left_weights: None,
        right_weights: None,
        left_node_stats: None,
        right_node_stats: None,
        left_histogram_stats: None,
        right_histogram_stats: None,
    });
    profile::add(profile::MATERIALIZE, started.elapsed());
}

fn materialize_axis_candidate(
    candidate: &mut Option<BestAxisCandidate>,
    context: &FitContext,
    active: &[bool],
    best: &mut Option<BestSplit>,
) {
    let started = profile::ProfileTimer::start();
    let Some(candidate) = candidate.take() else {
        return;
    };
    if best
        .as_ref()
        .is_some_and(|old| !is_better_split(candidate.gain, &candidate.split, old))
    {
        return;
    }
    let Some(sorted_rows) = context.sorted_rows(candidate.feature) else {
        return;
    };
    let mut left: Vec<usize> = Vec::with_capacity(candidate.left_capacity);
    let mut right: Vec<usize> = Vec::with_capacity(candidate.right_capacity);
    let mut position = 0usize;
    for &idx in sorted_rows {
        if !active[idx] {
            continue;
        }
        if position <= candidate.split_position {
            left.push(idx);
        } else {
            right.push(idx);
        }
        position += 1;
    }

    *best = Some(BestSplit {
        split: candidate.split,
        gain: candidate.gain,
        left,
        right,
        left_direct_node: None,
        right_direct_node: None,
        left_weights: None,
        right_weights: None,
        left_node_stats: None,
        right_node_stats: None,
        left_histogram_stats: None,
        right_histogram_stats: None,
    });
    profile::add(profile::MATERIALIZE, started.elapsed());
}

fn materialize_ordered_candidate(
    pairs: &[(f64, usize)],
    candidate: BestOrderedCandidate,
) -> BestSplit {
    let started = profile::ProfileTimer::start();
    let mut left = Vec::with_capacity(candidate.left_capacity);
    let mut right = Vec::with_capacity(candidate.right_capacity);
    for (position, &(_, idx)) in pairs.iter().enumerate() {
        if position <= candidate.split_position {
            left.push(idx);
        } else {
            right.push(idx);
        }
    }
    debug_assert_eq!(left.len(), candidate.left_capacity);
    debug_assert_eq!(right.len(), candidate.right_capacity);
    profile::add(profile::MATERIALIZE, started.elapsed());
    BestSplit {
        split: candidate.split,
        gain: candidate.gain,
        left,
        right,
        left_direct_node: None,
        right_direct_node: None,
        left_weights: None,
        right_weights: None,
        left_node_stats: Some(candidate.left_stats),
        right_node_stats: Some(candidate.right_stats),
        left_histogram_stats: None,
        right_histogram_stats: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn materialize_histogram_candidate(
    candidate: &mut Option<BestHistogramCandidate>,
    context: &FitContext,
    bins: usize,
    indices: &[usize],
    target: &[f64],
    weights: &[f64],
    parent_histogram_stats: Option<&[CandidateStats]>,
    constant_lambda_l2: f64,
    min_samples_leaf: usize,
    build_child_histograms: bool,
    terminal_updates: Option<&mut [f64]>,
    best: &mut Option<BestSplit>,
) {
    let started = profile::ProfileTimer::start();
    let Some(candidate) = candidate.take() else {
        return;
    };
    let Some(feature) = candidate.feature() else {
        return;
    };
    let Some(histogram_feature) = context
        .histogram_features
        .get(feature)
        .and_then(Option::as_ref)
    else {
        return;
    };
    if best
        .as_ref()
        .is_some_and(|old| !is_better_split(candidate.gain, &candidate.split, old))
    {
        return;
    }

    if let Some(updates) = terminal_updates.filter(|_| context.histogram_all_features) {
        let left_value = constant_leaf_value(candidate.left_stats, constant_lambda_l2);
        let right_value = constant_leaf_value(candidate.right_stats, constant_lambda_l2);
        let split_bin = candidate.split_bin as u16;
        profile::timed(profile::MATERIALIZE_PARTITION, || {
            let mut left_len = 0usize;
            let mut right_len = 0usize;
            for &idx in indices {
                let bin = histogram_feature.bins[idx];
                if bin <= split_bin {
                    updates[idx] = left_value;
                    left_len += 1;
                } else {
                    updates[idx] = right_value;
                    right_len += 1;
                }
            }
            debug_assert_eq!(left_len, candidate.left_capacity);
            debug_assert_eq!(right_len, candidate.right_capacity);
        });
        *best = Some(BestSplit {
            split: candidate.split,
            gain: candidate.gain,
            left: Vec::new(),
            right: Vec::new(),
            left_direct_node: Some(constant_leaf_node(candidate.left_stats, constant_lambda_l2)),
            right_direct_node: Some(constant_leaf_node(
                candidate.right_stats,
                constant_lambda_l2,
            )),
            left_weights: None,
            right_weights: None,
            left_node_stats: Some(candidate.left_stats),
            right_node_stats: Some(candidate.right_stats),
            left_histogram_stats: None,
            right_histogram_stats: None,
        });
        profile::add(profile::MATERIALIZE, started.elapsed());
        return;
    }

    let mut left: Vec<usize> = Vec::with_capacity(candidate.left_capacity);
    let mut right: Vec<usize> = Vec::with_capacity(candidate.right_capacity);
    let left_can_split = candidate.left_capacity >= min_samples_leaf * 2;
    let right_can_split = candidate.right_capacity >= min_samples_leaf * 2;
    let build_left_histogram = context.histogram_all_features
        && build_child_histograms
        && parent_histogram_stats.is_some()
        && left_can_split
        && right_can_split
        && candidate.left_capacity <= candidate.right_capacity;
    let build_right_histogram = context.histogram_all_features
        && build_child_histograms
        && parent_histogram_stats.is_some()
        && left_can_split
        && right_can_split
        && candidate.right_capacity < candidate.left_capacity;
    let mut smaller_histogram_stats = if build_left_histogram || build_right_histogram {
        Some(profile::timed(profile::HIST_PREPARE, || {
            vec![CandidateStats::default(); context.cols * bins]
        }))
    } else {
        None
    };
    profile::timed(profile::MATERIALIZE_PARTITION, || {
        if context.histogram_all_features && indices.len() >= 32_768 {
            let (parallel_left, parallel_right): (Vec<_>, Vec<_>) =
                indices.par_iter().copied().partition_map(|idx| {
                    let bin = histogram_feature.bins[idx];
                    if bin <= candidate.split_bin as u16 {
                        Either::Left(idx)
                    } else {
                        Either::Right(idx)
                    }
                });
            left = parallel_left;
            right = parallel_right;
        } else if context.histogram_all_features {
            let split_bin = candidate.split_bin as u16;
            // SAFETY: capacities come from the same histogram counts and split bin used here.
            // Every input row is written exactly once to either `left` or `right`, and the
            // final lengths are set to the number of initialized elements.
            unsafe {
                let left_ptr = left.as_mut_ptr();
                let right_ptr = right.as_mut_ptr();
                let mut left_len = 0;
                let mut right_len = 0;
                for &idx in indices {
                    let bin = histogram_feature.bins[idx];
                    if bin <= split_bin {
                        debug_assert!(left_len < candidate.left_capacity);
                        left_ptr.add(left_len).write(idx);
                        left_len += 1;
                    } else {
                        debug_assert!(right_len < candidate.right_capacity);
                        right_ptr.add(right_len).write(idx);
                        right_len += 1;
                    }
                }
                debug_assert_eq!(left_len, candidate.left_capacity);
                debug_assert_eq!(right_len, candidate.right_capacity);
                left.set_len(left_len);
                right.set_len(right_len);
            }
        } else {
            for &idx in indices {
                let bin = histogram_feature.bins[idx];
                if bin != MISSING_BIN && usize::from(bin) <= candidate.split_bin {
                    left.push(idx);
                } else {
                    right.push(idx);
                }
            }
        }
    });

    let (left_histogram_stats, right_histogram_stats) =
        profile::timed(profile::MATERIALIZE_CHILD_HIST, || {
            if build_left_histogram || build_right_histogram {
                let stats = smaller_histogram_stats
                    .as_mut()
                    .expect("stats allocated for smaller child");
                let histogram_rows = if build_left_histogram {
                    left.as_slice()
                } else {
                    right.as_slice()
                };
                *stats =
                    histogram_stats_for_indices(context, bins, target, weights, histogram_rows);
            }
            match (parent_histogram_stats, smaller_histogram_stats) {
                (Some(parent_stats), Some(smaller_stats)) => {
                    if build_left_histogram {
                        let right_stats = subtract_histogram_stats(parent_stats, &smaller_stats);
                        (Some(smaller_stats), Some(right_stats))
                    } else {
                        let left_stats = subtract_histogram_stats(parent_stats, &smaller_stats);
                        (Some(left_stats), Some(smaller_stats))
                    }
                }
                _ => (None, None),
            }
        });

    *best = Some(BestSplit {
        split: candidate.split,
        gain: candidate.gain,
        left,
        right,
        left_direct_node: None,
        right_direct_node: None,
        left_weights: None,
        right_weights: None,
        left_node_stats: Some(candidate.left_stats),
        right_node_stats: Some(candidate.right_stats),
        left_histogram_stats,
        right_histogram_stats,
    });
    profile::add(profile::MATERIALIZE, started.elapsed());
}

fn sparse_feature_allows_sparse_set(x: &Dataset, sparse_feature: usize) -> bool {
    match x.feature_schema() {
        Some(_) => matches!(
            sparse_feature_kind(x, sparse_feature),
            Some(FeatureKind::SparseSet)
        ),
        None => true,
    }
}

fn periodic_period_for_feature(
    x: &Dataset,
    indices: &[usize],
    feature: usize,
    requested_period: f64,
) -> Option<f64> {
    match x.feature_schema() {
        Some(_) => match dense_feature_kind(x, feature) {
            Some(FeatureKind::Periodic { period }) => Some(*period as f64),
            _ => None,
        },
        None => looks_like_periodic_feature(x, indices, feature, requested_period)
            .then_some(requested_period),
    }
}

fn merge_best_split(best: &mut Option<BestSplit>, candidate: Option<BestSplit>) {
    let Some(candidate) = candidate else {
        return;
    };
    if best
        .as_ref()
        .is_none_or(|old| is_better_split(candidate.gain, &candidate.split, old))
    {
        *best = Some(candidate);
    }
}

fn split_objective_is_saturated(parent_sse: f64, best: Option<&BestSplit>) -> bool {
    best.is_some_and(|candidate| candidate.gain >= parent_sse.max(0.0) - 1e-12)
}

fn merge_best_ordered_split(
    best: &mut Option<BestOrderedSplitCandidate>,
    candidate: Option<BestOrderedSplitCandidate>,
) {
    let Some(candidate) = candidate else {
        return;
    };
    if best.as_ref().is_none_or(|old| {
        is_better_split_candidate(
            candidate.candidate.gain,
            &candidate.candidate.split,
            old.candidate.gain,
            &old.candidate.split,
        )
    }) {
        *best = Some(candidate);
    }
}

fn is_better_split(gain: f64, split: &Split, old: &BestSplit) -> bool {
    is_better_split_candidate(gain, split, old.gain, &old.split)
}

fn is_better_split_candidate(gain: f64, split: &Split, old_gain: f64, old_split: &Split) -> bool {
    if is_spatial_split(split) && !is_spatial_split(old_split) {
        let required_gain =
            old_gain + old_gain.abs().max(1e-12) * SPATIAL_SPLIT_RELATIVE_GAIN_MARGIN;
        return gain > required_gain;
    }
    if gain > old_gain + 1e-12 {
        return true;
    }
    if (gain - old_gain).abs() > 1e-12 {
        return false;
    }
    match (periodic_width(split), periodic_width(old_split)) {
        (Some(width), Some(old_width)) => width < old_width - 1e-12,
        _ => false,
    }
}

fn is_spatial_split(split: &Split) -> bool {
    matches!(split, Split::Diagonal2D { .. } | Split::Gaussian2D { .. })
}

fn weighted_sse_from_sums(weight_sum: f64, weighted_sum: f64, weighted_square_sum: f64) -> f64 {
    if weight_sum <= 0.0 {
        0.0
    } else {
        weighted_square_sum - (weighted_sum * weighted_sum / weight_sum)
    }
}

fn huber_loss(target: f64, prediction: f64, delta: f64) -> f64 {
    let residual = target - prediction;
    let abs = residual.abs();
    if abs <= delta {
        0.5 * residual * residual
    } else {
        delta * (abs - 0.5 * delta)
    }
}

fn periodic_width(split: &Split) -> Option<f64> {
    match split {
        Split::PeriodicInterval {
            period, start, end, ..
        } => Some((end - start).rem_euclid(*period)),
        _ => None,
    }
}

