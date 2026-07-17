impl TreeBuilder {
    #[allow(clippy::too_many_arguments)]
    fn periodic_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        period: f64,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        if self.uses_l2_split_score()
            && (!self.fuzzy || self.fuzzy_bandwidth <= 0.0)
            && self.graph_split_regularization.is_none()
        {
            self.periodic_candidates_grouped(
                x,
                target,
                weights,
                indices,
                parent_sse,
                period,
                active_features,
                best,
            );
            return;
        }

        if period <= 0.0 || !period.is_finite() {
            return;
        }
        let mut candidates = (0..x.n_cols())
            .into_par_iter()
            .filter_map(|feature| {
                self.best_periodic_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    period,
                    feature,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(feature, _)| *feature);
        for (_, candidate) in candidates {
            merge_best_split(best, Some(candidate));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn best_periodic_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        period: f64,
        feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[feature]) {
            return None;
        }
        let feature_period = periodic_period_for_feature(x, indices, feature, period)?;
        let mut values = indices
            .iter()
            .filter_map(|&idx| {
                let value = x.get(idx, feature);
                value
                    .is_finite()
                    .then_some(super::normalize_periodic(value, feature_period))
            })
            .collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        if values.len() < 2 {
            return None;
        }

        let mut boundaries = values.clone();
        for idx in 0..values.len() {
            let current = values[idx];
            let next = values[(idx + 1) % values.len()];
            let gap = (next - current).rem_euclid(feature_period);
            boundaries.push(super::normalize_periodic(
                current + gap / 2.0,
                feature_period,
            ));
        }

        let mut best = None;
        for start_idx in 0..boundaries.len() {
            for end_idx in 0..boundaries.len() {
                if start_idx == end_idx {
                    continue;
                }
                let split = Split::PeriodicInterval {
                    feature,
                    period: feature_period,
                    start: boundaries[start_idx],
                    end: boundaries[end_idx],
                    missing_goes_left: true,
                };
                merge_best_split(
                    &mut best,
                    self.evaluate_split_candidate(split, x, target, weights, indices, parent_sse),
                );
            }
        }
        best.map(|best| (feature, best))
    }

    #[allow(clippy::too_many_arguments)]
    fn periodic_candidates_grouped(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        period: f64,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        if period <= 0.0 || !period.is_finite() {
            return;
        }
        let mut candidates = (0..x.n_cols())
            .into_par_iter()
            .filter_map(|feature| {
                self.best_periodic_grouped_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    period,
                    feature,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(feature, _)| *feature);
        for (_, candidate) in candidates {
            merge_best_split(best, Some(candidate));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn best_periodic_grouped_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        period: f64,
        feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[feature]) {
            return None;
        }
        let feature_period = periodic_period_for_feature(x, indices, feature, period)?;

        let mut values = indices
            .iter()
            .filter_map(|&idx| {
                let value = x.get(idx, feature);
                value
                    .is_finite()
                    .then_some((super::normalize_periodic(value, feature_period), idx))
            })
            .collect::<Vec<_>>();
        values.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        if values.len() < self.min_samples_leaf * 2 {
            return None;
        }

        let mut groups: Vec<PeriodicValueGroup> = Vec::new();
        for (value, idx) in values {
            let weight = weights[idx];
            let target_value = target[idx];
            if let Some(group) = groups
                .last_mut()
                .filter(|group| (group.value - value).abs() < 1e-12)
            {
                group.count += 1;
                group.weight_sum += weight;
                group.weighted_target_sum += weight * target_value;
                group.weighted_target_square_sum += weight * target_value * target_value;
            } else {
                groups.push(PeriodicValueGroup {
                    value,
                    count: 1,
                    weight_sum: weight,
                    weighted_target_sum: weight * target_value,
                    weighted_target_square_sum: weight * target_value * target_value,
                });
            }
        }
        if groups.len() < 2 {
            return None;
        }

        let mut boundaries = groups.iter().map(|group| group.value).collect::<Vec<_>>();
        for idx in 0..groups.len() {
            let current = groups[idx].value;
            let next = groups[(idx + 1) % groups.len()].value;
            let gap = (next - current).rem_euclid(feature_period);
            boundaries.push(super::normalize_periodic(
                current + gap / 2.0,
                feature_period,
            ));
        }

        let total = groups
            .iter()
            .fold(CandidateStats::default(), |mut total, group| {
                total.count += group.count;
                total.weight_sum += group.weight_sum;
                total.weighted_target_sum += group.weighted_target_sum;
                total.weighted_target_square_sum += group.weighted_target_square_sum;
                total
            });
        let parent_loss = if parent_sse == 0.0 {
            total.sse()
        } else {
            parent_sse
        };
        let mut best: Option<(Split, f64, CandidateStats, CandidateStats)> = None;

        for &start in &boundaries {
            for &end in &boundaries {
                if (start - end).abs() < 1e-12 {
                    continue;
                }

                let mut left_stats = CandidateStats::default();
                for group in &groups {
                    if super::periodic_contains(group.value, feature_period, start, end) {
                        left_stats.count += group.count;
                        left_stats.weight_sum += group.weight_sum;
                        left_stats.weighted_target_sum += group.weighted_target_sum;
                        left_stats.weighted_target_square_sum += group.weighted_target_square_sum;
                    }
                }

                let right_stats = total.minus(&left_stats);
                if left_stats.count < self.min_samples_leaf
                    || right_stats.count < self.min_samples_leaf
                {
                    continue;
                }

                let gain = parent_loss - left_stats.sse() - right_stats.sse();
                let split = Split::PeriodicInterval {
                    feature,
                    period: feature_period,
                    start,
                    end,
                    missing_goes_left: true,
                };
                if best.as_ref().is_some_and(|(old_split, old_gain, _, _)| {
                    !is_better_split_candidate(gain, &split, *old_gain, old_split)
                }) {
                    continue;
                }

                best = Some((split, gain, left_stats, right_stats));
            }
        }

        let (split, gain, left_stats, right_stats) = best?;
        let mut left = Vec::with_capacity(left_stats.count);
        let mut right = Vec::with_capacity(right_stats.count);
        if let Split::PeriodicInterval { start, end, .. } = split {
            for &idx in indices {
                if super::periodic_contains(x.get(idx, feature), feature_period, start, end) {
                    left.push(idx);
                } else {
                    right.push(idx);
                }
            }
        }
        Some((
            feature,
            BestSplit {
                split,
                gain,
                left,
                right,
                left_direct_node: None,
                right_direct_node: None,
                left_weights: None,
                right_weights: None,
                left_node_stats: Some(left_stats),
                right_node_stats: Some(right_stats),
                left_histogram_stats: None,
                right_histogram_stats: None,
            },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn sparse_set_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        if self.uses_l2_split_score()
            && (!self.fuzzy || self.fuzzy_bandwidth <= 0.0)
            && self.graph_split_regularization.is_none()
        {
            self.sparse_set_candidates_grouped(x, target, weights, indices, active_features, best);
            return;
        }

        let mut sparse_candidates = (0..x.n_sparse_sets())
            .into_par_iter()
            .filter_map(|sparse_feature| {
                self.best_sparse_list_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    sparse_feature,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        sparse_candidates.sort_by_key(|(sparse_feature, _)| *sparse_feature);
        for (_, candidate) in sparse_candidates {
            merge_best_split(best, Some(candidate));
        }

        let mut dense_candidates = (0..x.n_cols())
            .into_par_iter()
            .filter_map(|feature| {
                self.best_dense_sparse_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    feature,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        dense_candidates.sort_by_key(|(feature, _)| *feature);
        for (_, candidate) in dense_candidates {
            merge_best_split(best, Some(candidate));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn best_sparse_list_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        sparse_feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[x.n_cols() + sparse_feature]) {
            return None;
        }
        if !sparse_feature_allows_sparse_set(x, sparse_feature) {
            return None;
        }
        let mut ids = Vec::new();
        for &idx in indices {
            if let Some(row_ids) = x.sparse_set_row(idx, sparse_feature) {
                ids.extend_from_slice(row_ids);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        let mut best = None;
        for id in ids {
            let split = Split::SparseListContainsAny {
                sparse_feature,
                ids: vec![id],
                missing_goes_left: false,
            };
            merge_best_split(
                &mut best,
                self.evaluate_split_candidate(split, x, target, weights, indices, parent_sse),
            );
        }
        best.map(|best| (sparse_feature, best))
    }

    #[allow(clippy::too_many_arguments)]
    fn best_dense_sparse_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[feature]) {
            return None;
        }
        if !dense_feature_allows_sparse_set(x, feature) {
            return None;
        }
        let mut ids = indices
            .iter()
            .filter_map(|&idx| {
                let value = x.get(idx, feature);
                let id = value as u64;
                (value.is_finite() && value >= 0.0 && value == id as f64).then_some(id)
            })
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        let mut best = None;
        for id in ids {
            let split = Split::SparseSetContainsAny {
                feature,
                ids: vec![id],
                missing_goes_left: false,
            };
            merge_best_split(
                &mut best,
                self.evaluate_split_candidate(split, x, target, weights, indices, parent_sse),
            );
        }
        best.map(|best| (feature, best))
    }

    fn sparse_set_candidates_grouped(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        let total = candidate_stats(indices.iter().copied(), target, weights);

        let mut sparse_candidates = (0..x.n_sparse_sets())
            .into_par_iter()
            .filter_map(|sparse_feature| {
                self.best_sparse_list_grouped_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    &total,
                    sparse_feature,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        sparse_candidates.sort_by_key(|(sparse_feature, _)| *sparse_feature);
        for (_, candidate) in sparse_candidates {
            merge_best_split(best, Some(candidate));
        }

        let mut dense_candidates = (0..x.n_cols())
            .into_par_iter()
            .filter_map(|feature| {
                self.best_dense_sparse_grouped_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    &total,
                    feature,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        dense_candidates.sort_by_key(|(feature, _)| *feature);
        for (_, candidate) in dense_candidates {
            merge_best_split(best, Some(candidate));
        }
    }

    fn grouped_binary_split_candidate(
        &self,
        split: Split,
        left_stats: CandidateStats,
        total_stats: &CandidateStats,
    ) -> Option<BestSplit> {
        let right_stats = total_stats.minus(&left_stats);
        if left_stats.count < self.min_samples_leaf || right_stats.count < self.min_samples_leaf {
            return None;
        }
        let gain = total_stats.sse() - left_stats.sse() - right_stats.sse();
        Some(BestSplit {
            split,
            gain,
            left: Vec::with_capacity(left_stats.count),
            right: Vec::with_capacity(right_stats.count),
            left_direct_node: None,
            right_direct_node: None,
            left_weights: None,
            right_weights: None,
            left_node_stats: Some(left_stats),
            right_node_stats: Some(right_stats),
            left_histogram_stats: None,
            right_histogram_stats: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn best_sparse_list_grouped_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        total: &CandidateStats,
        sparse_feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[x.n_cols() + sparse_feature]) {
            return None;
        }
        if !sparse_feature_allows_sparse_set(x, sparse_feature) {
            return None;
        }
        let mut by_id = BTreeMap::<u64, CandidateStats>::new();
        for &idx in indices {
            if let Some(row_ids) = x.sparse_set_row(idx, sparse_feature) {
                for &id in row_ids {
                    by_id.entry(id).or_default().add_row(idx, target, weights);
                }
            }
        }
        let mut best: Option<(u64, BestSplit)> = None;
        for (id, stats) in by_id {
            let split = Split::SparseListContainsAny {
                sparse_feature,
                ids: vec![id],
                missing_goes_left: false,
            };
            if let Some(candidate) = self.grouped_binary_split_candidate(split, stats, total) {
                if best
                    .as_ref()
                    .is_none_or(|(_, old)| is_better_split(candidate.gain, &candidate.split, old))
                {
                    best = Some((id, candidate));
                }
            }
        }
        let (id, candidate) = best?;
        let mut candidate = Some(candidate);
        materialize_sparse_list_split(sparse_feature, id, x, indices, &mut candidate);
        candidate.map(|candidate| (sparse_feature, candidate))
    }

    #[allow(clippy::too_many_arguments)]
    fn best_dense_sparse_grouped_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        total: &CandidateStats,
        feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[feature]) {
            return None;
        }
        if !dense_feature_allows_sparse_set(x, feature) {
            return None;
        }
        let mut by_id = BTreeMap::<u64, CandidateStats>::new();
        for &idx in indices {
            let value = x.get(idx, feature);
            let id = value as u64;
            if value.is_finite() && value >= 0.0 && value == id as f64 {
                by_id.entry(id).or_default().add_row(idx, target, weights);
            }
        }
        let mut best: Option<(u64, BestSplit)> = None;
        for (id, stats) in by_id {
            let split = Split::SparseSetContainsAny {
                feature,
                ids: vec![id],
                missing_goes_left: false,
            };
            if let Some(candidate) = self.grouped_binary_split_candidate(split, stats, total) {
                if best
                    .as_ref()
                    .is_none_or(|(_, old)| is_better_split(candidate.gain, &candidate.split, old))
                {
                    best = Some((id, candidate));
                }
            }
        }
        let (id, candidate) = best?;
        let mut candidate = Some(candidate);
        materialize_dense_sparse_split(feature, id, x, indices, &mut candidate);
        candidate.map(|candidate| (feature, candidate))
    }

}
