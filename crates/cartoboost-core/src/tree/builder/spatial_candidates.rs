impl TreeBuilder {
    #[allow(clippy::too_many_arguments)]
    fn diagonal_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        if x.n_cols() < 2 {
            return;
        }
        if self.uses_l2_split_score()
            && (!self.fuzzy || self.fuzzy_bandwidth <= 0.0)
            && self.monotonic_constraints.is_empty()
            && self.graph_split_regularization.is_none()
        {
            self.diagonal_candidates_ordered(
                x,
                target,
                weights,
                indices,
                parent_sse,
                active_features,
                best,
            );
            return;
        }
        let spatial_features = spatial_feature_indices(x);
        if spatial_features.len() < 2 {
            return;
        }
        let normals = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)];
        let mut candidates = spatial_features
            .par_iter()
            .copied()
            .flat_map_iter(|x_feature| {
                let mut work = Vec::new();
                for &y_feature in spatial_features
                    .iter()
                    .filter(|&&feature| feature > x_feature)
                {
                    for (normal_idx, (normal_x, normal_y)) in normals.iter().copied().enumerate() {
                        work.push((x_feature, y_feature, normal_idx, normal_x, normal_y));
                    }
                }
                work
            })
            .filter_map(|(x_feature, y_feature, normal_idx, normal_x, normal_y)| {
                self.best_diagonal_candidate_for_projection(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    x_feature,
                    y_feature,
                    normal_idx,
                    normal_x,
                    normal_y,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(x_feature, y_feature, normal_idx, _)| {
            (*x_feature, *y_feature, *normal_idx)
        });
        for (_, _, _, candidate) in candidates {
            merge_best_split(best, Some(candidate));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn diagonal_candidates_ordered(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        let spatial_features = spatial_feature_indices(x);
        if spatial_features.len() < 2 {
            return;
        }
        let normals = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0)];
        let mut candidates = spatial_features
            .par_iter()
            .copied()
            .flat_map_iter(|x_feature| {
                let mut work = Vec::new();
                for &y_feature in spatial_features
                    .iter()
                    .filter(|&&feature| feature > x_feature)
                {
                    for (normal_idx, (normal_x, normal_y)) in normals.iter().copied().enumerate() {
                        work.push((x_feature, y_feature, normal_idx, normal_x, normal_y));
                    }
                }
                work
            })
            .filter_map(|(x_feature, y_feature, normal_idx, normal_x, normal_y)| {
                self.best_diagonal_ordered_candidate_for_projection(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    x_feature,
                    y_feature,
                    normal_idx,
                    normal_x,
                    normal_y,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(x_feature, y_feature, normal_idx, _)| {
            (*x_feature, *y_feature, *normal_idx)
        });
        let mut ordered_best: Option<BestOrderedSplitCandidate> = None;
        for (_, _, _, candidate) in candidates {
            merge_best_ordered_split(&mut ordered_best, Some(candidate));
        }
        if let Some(candidate) = ordered_best {
            merge_best_split(
                best,
                Some(materialize_ordered_candidate(
                    &candidate.pairs,
                    candidate.candidate,
                )),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn best_diagonal_ordered_candidate_for_projection(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        x_feature: usize,
        y_feature: usize,
        normal_idx: usize,
        normal_x: f64,
        normal_y: f64,
        active_features: &[usize],
    ) -> Option<(usize, usize, usize, BestOrderedSplitCandidate)> {
        if !self.interaction_split_allowed(active_features, &[x_feature, y_feature]) {
            return None;
        }
        if !dense_feature_allows_spatial(x, x_feature)
            || !dense_feature_allows_spatial(x, y_feature)
        {
            return None;
        }
        let mut pairs = indices
            .iter()
            .map(|&idx| {
                (
                    normal_x * x.get(idx, x_feature) + normal_y * x.get(idx, y_feature),
                    idx,
                )
            })
            .collect::<Vec<_>>();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let candidate =
            self.best_ordered_candidate(&pairs, target, weights, parent_sse, |threshold| {
                Split::Diagonal2D {
                    x_feature,
                    y_feature,
                    normal_x,
                    normal_y,
                    threshold,
                    missing_goes_left: true,
                }
            })?;
        Some((
            x_feature,
            y_feature,
            normal_idx,
            BestOrderedSplitCandidate { candidate, pairs },
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn best_diagonal_candidate_for_projection(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        x_feature: usize,
        y_feature: usize,
        normal_idx: usize,
        normal_x: f64,
        normal_y: f64,
        active_features: &[usize],
    ) -> Option<(usize, usize, usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[x_feature, y_feature]) {
            return None;
        }
        if !dense_feature_allows_spatial(x, x_feature)
            || !dense_feature_allows_spatial(x, y_feature)
        {
            return None;
        }
        let mut pairs = indices
            .iter()
            .map(|&idx| {
                (
                    normal_x * x.get(idx, x_feature) + normal_y * x.get(idx, y_feature),
                    idx,
                )
            })
            .collect::<Vec<_>>();
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut best = None;
        for window in pairs.windows(2) {
            let threshold = (window[0].0 + window[1].0) / 2.0;
            if window[0].0 == window[1].0 {
                continue;
            }
            let split = Split::Diagonal2D {
                x_feature,
                y_feature,
                normal_x,
                normal_y,
                threshold,
                missing_goes_left: true,
            };
            merge_best_split(
                &mut best,
                self.evaluate_split_candidate(split, x, target, weights, indices, parent_sse),
            );
        }
        best.map(|best| (x_feature, y_feature, normal_idx, best))
    }

    #[allow(clippy::too_many_arguments)]
    fn gaussian_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        if x.n_cols() < 2 || indices.is_empty() {
            return;
        }
        if self.uses_l2_split_score()
            && (!self.fuzzy || self.fuzzy_bandwidth <= 0.0)
            && self.monotonic_constraints.is_empty()
            && self.graph_split_regularization.is_none()
        {
            self.gaussian_candidates_ordered(
                x,
                target,
                weights,
                indices,
                parent_sse,
                active_features,
                best,
            );
            return;
        }
        let spatial_features = spatial_feature_indices(x);
        if spatial_features.len() < 2 {
            return;
        }
        let mut candidates = spatial_features
            .par_iter()
            .copied()
            .flat_map_iter(|x_feature| {
                spatial_features
                    .iter()
                    .copied()
                    .filter(move |&y_feature| y_feature > x_feature)
                    .map(move |y_feature| (x_feature, y_feature))
                    .collect::<Vec<_>>()
            })
            .filter_map(|(x_feature, y_feature)| {
                self.best_gaussian_candidate_for_pair(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    x_feature,
                    y_feature,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(x_feature, y_feature, _)| (*x_feature, *y_feature));
        for (_, _, candidate) in candidates {
            merge_best_split(best, Some(candidate));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gaussian_candidates_ordered(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        let spatial_features = spatial_feature_indices(x);
        if spatial_features.len() < 2 {
            return;
        }
        let mut candidates = spatial_features
            .par_iter()
            .copied()
            .flat_map_iter(|x_feature| {
                spatial_features
                    .iter()
                    .copied()
                    .filter(move |&y_feature| y_feature > x_feature)
                    .map(move |y_feature| (x_feature, y_feature))
                    .collect::<Vec<_>>()
            })
            .filter_map(|(x_feature, y_feature)| {
                self.best_gaussian_ordered_candidate_for_pair(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    x_feature,
                    y_feature,
                    active_features,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(x_feature, y_feature, _)| (*x_feature, *y_feature));
        let mut ordered_best: Option<BestOrderedSplitCandidate> = None;
        for (_, _, candidate) in candidates {
            merge_best_ordered_split(&mut ordered_best, Some(candidate));
        }
        if let Some(candidate) = ordered_best {
            merge_best_split(
                best,
                Some(materialize_ordered_candidate(
                    &candidate.pairs,
                    candidate.candidate,
                )),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn best_gaussian_ordered_candidate_for_pair(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        x_feature: usize,
        y_feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, usize, BestOrderedSplitCandidate)> {
        if !self.interaction_split_allowed(active_features, &[x_feature, y_feature]) {
            return None;
        }
        if !dense_feature_allows_spatial(x, x_feature)
            || !dense_feature_allows_spatial(x, y_feature)
        {
            return None;
        }
        let mut best = None;
        for (center_x, center_y) in
            self.gaussian_centers(x, target, weights, indices, x_feature, y_feature)
        {
            let mut distances = indices
                .iter()
                .map(|&idx| {
                    (
                        (x.get(idx, x_feature) - center_x).powi(2)
                            + (x.get(idx, y_feature) - center_y).powi(2),
                        idx,
                    )
                })
                .filter(|(distance, _)| distance.is_finite())
                .collect::<Vec<_>>();
            distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let candidate = self
                .best_ordered_candidate(&distances, target, weights, parent_sse, |radius_sq| {
                    Split::Gaussian2D {
                        x_feature,
                        y_feature,
                        center_x,
                        center_y,
                        radius: radius_sq.max(0.0).sqrt(),
                        missing_goes_left: true,
                    }
                })
                .map(|candidate| BestOrderedSplitCandidate {
                    candidate,
                    pairs: distances,
                });
            merge_best_ordered_split(&mut best, candidate);
        }
        best.map(|best| (x_feature, y_feature, best))
    }

    #[allow(clippy::too_many_arguments)]
    fn best_gaussian_candidate_for_pair(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        x_feature: usize,
        y_feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[x_feature, y_feature]) {
            return None;
        }
        if !dense_feature_allows_spatial(x, x_feature)
            || !dense_feature_allows_spatial(x, y_feature)
        {
            return None;
        }
        let mut best = None;
        for (center_x, center_y) in
            self.gaussian_centers(x, target, weights, indices, x_feature, y_feature)
        {
            let mut distances = indices
                .iter()
                .map(|&idx| {
                    (
                        ((x.get(idx, x_feature) - center_x).powi(2)
                            + (x.get(idx, y_feature) - center_y).powi(2))
                        .sqrt(),
                        idx,
                    )
                })
                .filter(|(distance, _)| distance.is_finite())
                .collect::<Vec<_>>();
            distances.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for window in distances.windows(2) {
                if window[0].0 == window[1].0 {
                    continue;
                }
                let radius = (window[0].0 + window[1].0) / 2.0;
                let split = Split::Gaussian2D {
                    x_feature,
                    y_feature,
                    center_x,
                    center_y,
                    radius,
                    missing_goes_left: true,
                };
                merge_best_split(
                    &mut best,
                    self.evaluate_split_candidate(split, x, target, weights, indices, parent_sse),
                );
            }
        }
        best.map(|best| (x_feature, y_feature, best))
    }

    fn best_ordered_candidate(
        &self,
        pairs: &[(f64, usize)],
        target: &[f64],
        weights: &[f64],
        parent_sse: f64,
        make_split: impl Fn(f64) -> Split,
    ) -> Option<BestOrderedCandidate> {
        if pairs.len() < self.min_samples_leaf * 2 {
            return None;
        }
        let total = candidate_stats(pairs.iter().map(|(_, idx)| *idx), target, weights);
        let parent_loss = if parent_sse == 0.0 {
            total.sse()
        } else {
            parent_sse
        };
        let mut left_stats = CandidateStats::default();
        let mut left_count = 0usize;
        let mut previous: Option<(f64, usize)> = None;
        let mut best = None;

        for &(current_value, current_idx) in pairs {
            let Some((previous_value, previous_idx)) = previous else {
                previous = Some((current_value, current_idx));
                continue;
            };
            left_stats.add_row(previous_idx, target, weights);
            left_count += 1;

            if previous_value == current_value {
                previous = Some((current_value, current_idx));
                continue;
            }

            let right_count = pairs.len() - left_count;
            if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                previous = Some((current_value, current_idx));
                continue;
            }
            let right_stats = total.minus(&left_stats);
            let split = make_split((previous_value + current_value) / 2.0);
            let gain = parent_loss - left_stats.sse() - right_stats.sse();
            if best.as_ref().is_none_or(|old: &BestOrderedCandidate| {
                is_better_split_candidate(gain, &split, old.gain, &old.split)
            }) {
                best = Some(BestOrderedCandidate {
                    split,
                    gain,
                    split_position: left_count - 1,
                    left_capacity: left_count,
                    right_capacity: right_count,
                    left_stats,
                    right_stats,
                });
            }
            previous = Some((current_value, current_idx));
        }

        best
    }

    #[allow(clippy::too_many_arguments)]
    fn gaussian_centers(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        x_feature: usize,
        y_feature: usize,
    ) -> Vec<(f64, f64)> {
        let mut centers = Vec::new();
        let mut push_center = |center_x: f64, center_y: f64| {
            if !center_x.is_finite() || !center_y.is_finite() {
                return;
            }
            if centers.iter().any(|&(old_x, old_y): &(f64, f64)| {
                (old_x - center_x).abs() < 1e-12 && (old_y - center_y).abs() < 1e-12
            }) {
                return;
            }
            centers.push((center_x, center_y));
        };

        let weighted_centroid = |selected: &[usize]| -> Option<(f64, f64)> {
            let weight_sum = selected.iter().map(|&idx| weights[idx]).sum::<f64>();
            if weight_sum <= 0.0 {
                return None;
            }
            let center_x = selected
                .iter()
                .map(|&idx| x.get(idx, x_feature) * weights[idx])
                .sum::<f64>()
                / weight_sum;
            let center_y = selected
                .iter()
                .map(|&idx| x.get(idx, y_feature) * weights[idx])
                .sum::<f64>()
                / weight_sum;
            Some((center_x, center_y))
        };

        if let Some((center_x, center_y)) = weighted_centroid(indices) {
            push_center(center_x, center_y);
        }

        let weight_sum = indices.iter().map(|&idx| weights[idx]).sum::<f64>();
        let target_mean = if weight_sum > 0.0 {
            indices
                .iter()
                .map(|&idx| target[idx] * weights[idx])
                .sum::<f64>()
                / weight_sum
        } else {
            0.0
        };
        let above_mean = indices
            .iter()
            .copied()
            .filter(|&idx| target[idx] >= target_mean)
            .collect::<Vec<_>>();
        let below_mean = indices
            .iter()
            .copied()
            .filter(|&idx| target[idx] < target_mean)
            .collect::<Vec<_>>();
        for selected in [&above_mean, &below_mean] {
            if selected.len() >= self.min_samples_leaf {
                if let Some((center_x, center_y)) = weighted_centroid(selected) {
                    push_center(center_x, center_y);
                }
            }
        }

        centers
    }
}
