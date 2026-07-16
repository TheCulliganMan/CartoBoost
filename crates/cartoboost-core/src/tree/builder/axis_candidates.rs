impl TreeBuilder {
    #[allow(clippy::too_many_arguments)]
    fn best_split(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        context: &FitContext,
        node_histogram_stats: Option<&[CandidateStats]>,
        build_child_histograms: bool,
        terminal_updates: Option<&mut [f64]>,
        active_features: &[usize],
    ) -> Option<BestSplit> {
        let mut best: Option<BestSplit> = None;

        let splitters = self.splitters.as_slice();
        let pure_histogram = !splitters.is_empty()
            && splitters
                .iter()
                .all(|splitter| matches!(splitter, SplitterKind::AxisHistogram { .. }));
        let mut terminal_histogram_updates = if !build_child_histograms
            && pure_histogram
            && self.leaf_predictor == LeafPredictorKind::Constant
            && self.uses_l2_split_score()
            && self.monotonic_constraints.is_empty()
            && self.graph_split_regularization.is_none()
        {
            terminal_updates
        } else {
            None
        };
        let parent_sse = if pure_histogram && self.uses_l2_split_score() {
            0.0
        } else {
            profile::timed(profile::PARENT_SSE, || {
                self.node_loss(target, weights, indices)
            })
        };
        if splitters.is_empty() {
            self.axis_candidates(
                x,
                target,
                weights,
                indices,
                parent_sse,
                context,
                active_features,
                &mut best,
            );
            return best;
        }
        for splitter in splitters {
            match splitter {
                SplitterKind::Axis => profile::timed(profile::AXIS, || {
                    self.axis_candidates(
                        x,
                        target,
                        weights,
                        indices,
                        parent_sse,
                        context,
                        active_features,
                        &mut best,
                    )
                }),
                SplitterKind::Auto => profile::timed(profile::AXIS, || {
                    self.axis_candidates(
                        x,
                        target,
                        weights,
                        indices,
                        parent_sse,
                        context,
                        active_features,
                        &mut best,
                    )
                }),
                SplitterKind::AxisHistogram { bins } => {
                    if self.uses_l2_split_score()
                        && self.monotonic_constraints.is_empty()
                        && self.graph_split_regularization.is_none()
                    {
                        self.axis_histogram_candidates(
                            x,
                            target,
                            weights,
                            indices,
                            parent_sse,
                            *bins,
                            context,
                            node_histogram_stats,
                            build_child_histograms,
                            terminal_histogram_updates.as_deref_mut(),
                            active_features,
                            &mut best,
                        )
                    } else {
                        self.axis_histogram_exact_candidates(
                            x,
                            target,
                            weights,
                            indices,
                            parent_sse,
                            *bins,
                            context,
                            active_features,
                            &mut best,
                        )
                    }
                }
                SplitterKind::Diagonal2D => profile::timed(profile::DIAGONAL, || {
                    self.diagonal_candidates(
                        x,
                        target,
                        weights,
                        indices,
                        parent_sse,
                        active_features,
                        &mut best,
                    )
                }),
                SplitterKind::Gaussian2D => profile::timed(profile::GAUSSIAN, || {
                    self.gaussian_candidates(
                        x,
                        target,
                        weights,
                        indices,
                        parent_sse,
                        active_features,
                        &mut best,
                    )
                }),
                SplitterKind::Periodic { period } => profile::timed(profile::PERIODIC, || {
                    self.periodic_candidates(
                        x,
                        target,
                        weights,
                        indices,
                        parent_sse,
                        *period,
                        active_features,
                        &mut best,
                    )
                }),
                SplitterKind::SparseSet => profile::timed(profile::SPARSE_SET, || {
                    self.sparse_set_candidates(
                        x,
                        target,
                        weights,
                        indices,
                        parent_sse,
                        active_features,
                        &mut best,
                    )
                }),
            }
            if split_objective_is_saturated(parent_sse, best.as_ref()) {
                break;
            }
        }

        best
    }

    #[allow(clippy::too_many_arguments)]
    fn axis_histogram_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        bins: usize,
        context: &FitContext,
        node_histogram_stats: Option<&[CandidateStats]>,
        build_child_histograms: bool,
        mut terminal_updates: Option<&mut [f64]>,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        let bins = bins.clamp(2, 1024);
        if !context.histogram_row_bins.is_empty() {
            let started = profile::ProfileTimer::start();
            let mut computed_stats;
            let stats = if let Some(stats) = node_histogram_stats {
                stats
            } else {
                computed_stats = profile::timed(profile::HIST_PREPARE, || {
                    vec![CandidateStats::default(); context.cols * bins]
                });
                profile::timed(profile::HIST_ACCUMULATE, || {
                    if context.histogram_all_features && indices.len() >= 32_768 {
                        let chunk_size =
                            (indices.len() / rayon::current_num_threads().max(1)).max(16_384);
                        let partials = indices
                            .par_chunks(chunk_size)
                            .map(|chunk| {
                                let mut partial =
                                    vec![CandidateStats::default(); context.cols * bins];
                                for &idx in chunk {
                                    add_histogram_stats_row(
                                        context,
                                        bins,
                                        target,
                                        weights,
                                        idx,
                                        &mut partial,
                                    );
                                }
                                partial
                            })
                            .collect::<Vec<_>>();
                        for partial in partials {
                            for (total, item) in computed_stats.iter_mut().zip(partial) {
                                total.merge(item);
                            }
                        }
                    } else {
                        for &idx in indices {
                            add_histogram_stats_row(
                                context,
                                bins,
                                target,
                                weights,
                                idx,
                                &mut computed_stats,
                            );
                        }
                    }
                });
                &computed_stats
            };

            let mut histogram_candidate: Option<BestHistogramCandidate> = None;
            profile::timed(profile::HIST_SCORE, || {
                let common_total = context.histogram_all_features.then(|| {
                    histogram_node_stats_from_feature(
                        *context
                            .histogram_feature_indices
                            .first()
                            .expect("histogram_all_features requires at least one feature"),
                        bins,
                        stats,
                    )
                });
                let candidate_for_feature = |&feature: &usize| {
                    if !self.interaction_split_allowed(active_features, &[feature]) {
                        return None;
                    }
                    self.best_histogram_candidate_for_feature(
                        feature,
                        bins,
                        context,
                        stats,
                        common_total,
                        parent_sse,
                    )
                };
                // A small feature set is faster sequentially; spawning one
                // Rayon task per feature otherwise dominates histogram scoring
                // for the maintained 20-feature structured workload.
                let mut candidates = if context.histogram_feature_indices.len() < 64 {
                    context
                        .histogram_feature_indices
                        .iter()
                        .filter_map(candidate_for_feature)
                        .collect::<Vec<_>>()
                } else {
                    context
                        .histogram_feature_indices
                        .par_iter()
                        .filter_map(candidate_for_feature)
                        .collect::<Vec<_>>()
                };
                candidates.sort_by_key(|candidate| candidate.feature().unwrap_or(usize::MAX));
                for candidate in candidates {
                    if best
                        .as_ref()
                        .is_some_and(|old| !is_better_split(candidate.gain, &candidate.split, old))
                    {
                        continue;
                    }
                    if histogram_candidate.as_ref().is_none_or(|old| {
                        is_better_split_candidate(
                            candidate.gain,
                            &candidate.split,
                            old.gain,
                            &old.split,
                        )
                    }) {
                        histogram_candidate = Some(candidate);
                    }
                }
            });
            profile::add(profile::HISTOGRAM, started.elapsed());
            materialize_histogram_candidate(
                &mut histogram_candidate,
                context,
                bins,
                indices,
                target,
                weights,
                Some(stats),
                self.constant_lambda_l2,
                self.min_samples_leaf,
                build_child_histograms,
                terminal_updates.as_deref_mut(),
                best,
            );
            return;
        }

        let started = profile::ProfileTimer::start();
        let mut stats = vec![CandidateStats::default(); bins];
        let mut histogram_candidate: Option<BestHistogramCandidate> = None;
        for feature in 0..x.n_cols() {
            if !self.interaction_split_allowed(active_features, &[feature]) {
                continue;
            }
            if !dense_feature_allows_axis(x, feature) {
                continue;
            }
            if let Some(histogram_feature) = context.histogram_feature(feature, bins) {
                let Some(candidate) = self.axis_histogram_prebinned_candidate(
                    feature,
                    histogram_feature,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    &mut stats,
                ) else {
                    continue;
                };
                if best
                    .as_ref()
                    .is_some_and(|old| !is_better_split(candidate.gain, &candidate.split, old))
                {
                    continue;
                }
                if histogram_candidate.as_ref().is_none_or(|old| {
                    is_better_split_candidate(
                        candidate.gain,
                        &candidate.split,
                        old.gain,
                        &old.split,
                    )
                }) {
                    histogram_candidate = Some(candidate);
                }
                continue;
            }
            materialize_histogram_candidate(
                &mut histogram_candidate,
                context,
                bins,
                indices,
                target,
                weights,
                None,
                self.constant_lambda_l2,
                self.min_samples_leaf,
                false,
                None,
                best,
            );
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;
            for &idx in indices {
                let value = x.get(idx, feature);
                if value.is_finite() {
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
            }
            if !min_value.is_finite() || min_value >= max_value {
                continue;
            }

            let scale = bins as f64 / (max_value - min_value);
            stats.fill(CandidateStats::default());
            for &idx in indices {
                let value = x.get(idx, feature);
                if !value.is_finite() {
                    continue;
                }
                let bin = (((value - min_value) * scale) as usize).min(bins - 1);
                stats[bin].add_row(idx, target, weights);
            }

            let total = stats
                .iter()
                .fold(CandidateStats::default(), |mut total, item| {
                    total.count += item.count;
                    total.weight_sum += item.weight_sum;
                    total.weighted_target_sum += item.weighted_target_sum;
                    total.weighted_target_square_sum += item.weighted_target_square_sum;
                    total
                });
            if total.count < self.min_samples_leaf * 2 {
                continue;
            }

            let parent_loss = if parent_sse == 0.0 {
                total.sse()
            } else {
                parent_sse
            };
            let mut left_stats = CandidateStats::default();
            for (split_bin, bin_stats) in stats.iter().enumerate().take(bins - 1) {
                left_stats.count += bin_stats.count;
                left_stats.weight_sum += bin_stats.weight_sum;
                left_stats.weighted_target_sum += bin_stats.weighted_target_sum;
                left_stats.weighted_target_square_sum += bin_stats.weighted_target_square_sum;
                let right_count = total.count - left_stats.count;
                if left_stats.count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    continue;
                }
                let right_weight_sum = total.weight_sum - left_stats.weight_sum;
                let right_target_sum = total.weighted_target_sum - left_stats.weighted_target_sum;
                let right_target_square_sum =
                    total.weighted_target_square_sum - left_stats.weighted_target_square_sum;
                let threshold = min_value + ((split_bin + 1) as f64 / scale);
                if threshold >= max_value {
                    continue;
                }
                let gain = parent_loss
                    - left_stats.sse()
                    - weighted_sse_from_sums(
                        right_weight_sum,
                        right_target_sum,
                        right_target_square_sum,
                    );
                let split = Split::Axis {
                    feature,
                    threshold,
                    missing_goes_left: true,
                };
                if best
                    .as_ref()
                    .is_some_and(|old| !is_better_split(gain, &split, old))
                {
                    continue;
                }
                materialize_axis_split(feature, threshold, x, indices, best);
                if let Some(best) = best.as_mut() {
                    best.split = split;
                    best.gain = gain;
                }
            }
        }
        profile::add(profile::HISTOGRAM, started.elapsed());
        materialize_histogram_candidate(
            &mut histogram_candidate,
            context,
            bins,
            indices,
            target,
            weights,
            None,
            self.constant_lambda_l2,
            self.min_samples_leaf,
            false,
            terminal_updates,
            best,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn axis_histogram_prebinned_candidate(
        &self,
        feature: usize,
        histogram_feature: &HistogramFeature,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        stats: &mut [CandidateStats],
    ) -> Option<BestHistogramCandidate> {
        let bins = histogram_feature.bin_count;
        stats.fill(CandidateStats::default());
        for &idx in indices {
            let bin = histogram_feature.bins[idx];
            if bin != MISSING_BIN {
                stats[usize::from(bin)].add_row(idx, target, weights);
            }
        }

        profile::timed(profile::HIST_SCORE, || {
            let total = stats
                .iter()
                .fold(CandidateStats::default(), |mut total, item| {
                    total.count += item.count;
                    total.weight_sum += item.weight_sum;
                    total.weighted_target_sum += item.weighted_target_sum;
                    total.weighted_target_square_sum += item.weighted_target_square_sum;
                    total
                });
            if total.count < self.min_samples_leaf * 2 {
                return None;
            }

            let parent_loss = if parent_sse == 0.0 {
                total.sse()
            } else {
                parent_sse
            };
            let mut left_stats = CandidateStats::default();
            let mut candidate: Option<BestHistogramCandidate> = None;
            for (split_bin, bin_stats) in stats.iter().enumerate().take(bins - 1) {
                left_stats.count += bin_stats.count;
                left_stats.weight_sum += bin_stats.weight_sum;
                left_stats.weighted_target_sum += bin_stats.weighted_target_sum;
                left_stats.weighted_target_square_sum += bin_stats.weighted_target_square_sum;
                let right_count = total.count - left_stats.count;
                if left_stats.count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                    continue;
                }
                let right_weight_sum = total.weight_sum - left_stats.weight_sum;
                let right_target_sum = total.weighted_target_sum - left_stats.weighted_target_sum;
                let right_target_square_sum =
                    total.weighted_target_square_sum - left_stats.weighted_target_square_sum;
                let threshold = histogram_feature.thresholds[split_bin];
                let gain = parent_loss
                    - left_stats.sse()
                    - weighted_sse_from_sums(
                        right_weight_sum,
                        right_target_sum,
                        right_target_square_sum,
                    );
                let split = Split::Axis {
                    feature,
                    threshold,
                    missing_goes_left: true,
                };
                if candidate.as_ref().is_some_and(|old| {
                    !is_better_split_candidate(gain, &split, old.gain, &old.split)
                }) {
                    continue;
                }
                candidate = Some(BestHistogramCandidate {
                    split,
                    gain,
                    split_bin,
                    left_capacity: left_stats.count,
                    right_capacity: right_count,
                    left_stats,
                    right_stats: CandidateStats {
                        count: right_count,
                        weight_sum: right_weight_sum,
                        weighted_target_sum: right_target_sum,
                        weighted_target_square_sum: right_target_square_sum,
                    },
                });
            }

            candidate
        })
    }

    fn best_histogram_candidate_for_feature(
        &self,
        feature: usize,
        bins: usize,
        context: &FitContext,
        stats: &[CandidateStats],
        common_total: Option<CandidateStats>,
        parent_sse: f64,
    ) -> Option<BestHistogramCandidate> {
        let histogram_feature = context.histogram_features[feature]
            .as_ref()
            .expect("histogram_feature_indices contains prebinned features");
        let feature_stats = &stats[feature * bins..(feature + 1) * bins];
        let total = common_total.unwrap_or_else(|| {
            feature_stats
                .iter()
                .fold(CandidateStats::default(), |mut total, item| {
                    total.count += item.count;
                    total.weight_sum += item.weight_sum;
                    total.weighted_target_sum += item.weighted_target_sum;
                    total.weighted_target_square_sum += item.weighted_target_square_sum;
                    total
                })
        });
        if total.count < self.min_samples_leaf * 2 {
            return None;
        }

        let parent_loss = if parent_sse == 0.0 {
            total.sse()
        } else {
            parent_sse
        };
        let mut left_stats = CandidateStats::default();
        let mut candidate: Option<BestHistogramCandidate> = None;
        for (split_bin, bin_stats) in feature_stats.iter().enumerate().take(bins - 1) {
            left_stats.count += bin_stats.count;
            left_stats.weight_sum += bin_stats.weight_sum;
            left_stats.weighted_target_sum += bin_stats.weighted_target_sum;
            left_stats.weighted_target_square_sum += bin_stats.weighted_target_square_sum;
            let right_count = total.count - left_stats.count;
            if left_stats.count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                continue;
            }
            let right_weight_sum = total.weight_sum - left_stats.weight_sum;
            let right_target_sum = total.weighted_target_sum - left_stats.weighted_target_sum;
            let right_target_square_sum =
                total.weighted_target_square_sum - left_stats.weighted_target_square_sum;
            let threshold = histogram_feature.thresholds[split_bin];
            let gain = parent_loss
                - left_stats.sse()
                - weighted_sse_from_sums(
                    right_weight_sum,
                    right_target_sum,
                    right_target_square_sum,
                );
            let split = Split::Axis {
                feature,
                threshold,
                missing_goes_left: true,
            };
            if candidate
                .as_ref()
                .is_some_and(|old| !is_better_split_candidate(gain, &split, old.gain, &old.split))
            {
                continue;
            }
            candidate = Some(BestHistogramCandidate {
                split,
                gain,
                split_bin,
                left_capacity: left_stats.count,
                right_capacity: right_count,
                left_stats,
                right_stats: CandidateStats {
                    count: right_count,
                    weight_sum: right_weight_sum,
                    weighted_target_sum: right_target_sum,
                    weighted_target_square_sum: right_target_square_sum,
                },
            });
        }

        candidate
    }

    #[allow(clippy::too_many_arguments)]
    fn axis_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        context: &FitContext,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        if !self.interaction_constraints.is_empty() {
            self.axis_candidates_exact(
                x,
                target,
                weights,
                indices,
                parent_sse,
                context,
                active_features,
                best,
            );
            return;
        }
        if !self.uses_l2_split_score()
            || !self.monotonic_constraints.is_empty()
            || self.graph_split_regularization.is_some()
        {
            self.axis_candidates_exact(
                x,
                target,
                weights,
                indices,
                parent_sse,
                context,
                active_features,
                best,
            );
            return;
        }
        if !self.fuzzy || self.fuzzy_bandwidth <= 0.0 {
            self.axis_candidates_prefix(
                x,
                target,
                weights,
                indices,
                parent_sse,
                context,
                active_features,
                best,
            );
            return;
        }

        let active = active_row_mask(x.n_rows(), indices);
        let mut candidates = (0..x.n_cols())
            .into_par_iter()
            .filter_map(|feature| {
                self.best_axis_exact_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    feature,
                    parent_sse,
                    context,
                    &active,
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
    fn axis_candidates_exact(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        context: &FitContext,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        let active = active_row_mask(x.n_rows(), indices);
        let mut candidates = (0..x.n_cols())
            .into_par_iter()
            .filter_map(|feature| {
                self.best_axis_exact_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    feature,
                    parent_sse,
                    context,
                    &active,
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
    fn best_axis_exact_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        feature: usize,
        parent_sse: f64,
        context: &FitContext,
        active: &[bool],
        active_features: &[usize],
    ) -> Option<(usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[feature]) {
            return None;
        }
        if !dense_feature_allows_axis(x, feature) {
            return None;
        }
        let sorted_rows = context.sorted_rows(feature)?;
        let pairs = sorted_rows
            .iter()
            .copied()
            .filter(|&idx| active[idx])
            .filter_map(|idx| {
                let value = x.get(idx, feature);
                value.is_finite().then_some((value, idx))
            })
            .collect::<Vec<_>>();

        let mut best = None;
        for window in pairs.windows(2) {
            let (a, _) = window[0];
            let (b, _) = window[1];
            if a == b {
                continue;
            }
            let split = Split::Axis {
                feature,
                threshold: (a + b) / 2.0,
                missing_goes_left: true,
            };
            merge_best_split(
                &mut best,
                self.evaluate_split_candidate(split, x, target, weights, indices, parent_sse),
            );
        }
        best.map(|best| (feature, best))
    }

    #[allow(clippy::too_many_arguments)]
    fn axis_histogram_exact_candidates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        bins: usize,
        context: &FitContext,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        let bins = bins.clamp(2, 1024);
        let mut candidates = (0..x.n_cols())
            .into_par_iter()
            .filter_map(|feature| {
                self.best_axis_histogram_exact_candidate_for_feature(
                    x,
                    target,
                    weights,
                    indices,
                    parent_sse,
                    bins,
                    context,
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
    fn best_axis_histogram_exact_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        bins: usize,
        context: &FitContext,
        feature: usize,
        active_features: &[usize],
    ) -> Option<(usize, BestSplit)> {
        if !self.interaction_split_allowed(active_features, &[feature]) {
            return None;
        }
        if !dense_feature_allows_axis(x, feature) {
            return None;
        }
        let thresholds = if let Some(histogram_feature) = context.histogram_feature(feature, bins) {
            histogram_feature.thresholds.clone()
        } else {
            let mut values = Vec::with_capacity(indices.len());
            for &idx in indices {
                let value = x.get(idx, feature);
                if value.is_finite() {
                    values.push(value);
                }
            }
            quantile_histogram_thresholds(values, bins)
        };
        if thresholds.is_empty() {
            return None;
        }
        let mut best = None;
        for threshold in thresholds {
            let split = Split::Axis {
                feature,
                threshold,
                missing_goes_left: true,
            };
            merge_best_split(
                &mut best,
                self.evaluate_split_candidate(split, x, target, weights, indices, parent_sse),
            );
        }
        best.map(|best| (feature, best))
    }

    #[allow(clippy::too_many_arguments)]
    fn axis_candidates_prefix(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
        context: &FitContext,
        active_features: &[usize],
        best: &mut Option<BestSplit>,
    ) {
        let active = active_row_mask(x.n_rows(), indices);
        let mut axis_candidate: Option<BestAxisCandidate> = None;
        let mut candidates = (0..x.n_cols())
            .into_par_iter()
            .filter_map(|feature| {
                if !self.interaction_split_allowed(active_features, &[feature]) {
                    return None;
                }
                self.best_axis_prefix_candidate_for_feature(
                    x, target, weights, feature, parent_sse, context, &active,
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|candidate| candidate.feature);
        for candidate in candidates {
            if best
                .as_ref()
                .is_some_and(|old| !is_better_split(candidate.gain, &candidate.split, old))
            {
                continue;
            }
            if axis_candidate.as_ref().is_none_or(|old| {
                is_better_split_candidate(candidate.gain, &candidate.split, old.gain, &old.split)
            }) {
                axis_candidate = Some(candidate);
            }
        }
        materialize_axis_candidate(&mut axis_candidate, context, &active, best);
    }

    #[allow(clippy::too_many_arguments)]
    fn best_axis_prefix_candidate_for_feature(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        feature: usize,
        parent_sse: f64,
        context: &FitContext,
        active: &[bool],
    ) -> Option<BestAxisCandidate> {
        if !dense_feature_allows_axis(x, feature) {
            return None;
        }
        let sorted_rows = context.sorted_rows(feature)?;

        let mut total_weight = 0.0;
        let mut total_weighted_target = 0.0;
        let mut total_weighted_target_sq = 0.0;
        let mut active_count = 0usize;
        for &idx in sorted_rows {
            if !active[idx] {
                continue;
            }
            let weight = weights[idx];
            let value = target[idx];
            total_weight += weight;
            total_weighted_target += weight * value;
            total_weighted_target_sq += weight * value * value;
            active_count += 1;
        }
        if active_count < self.min_samples_leaf * 2 {
            return None;
        }

        let mut left_weight = 0.0;
        let mut left_weighted_target = 0.0;
        let mut left_weighted_target_sq = 0.0;
        let mut left_count = 0usize;
        let mut previous: Option<(f64, usize)> = None;
        let mut candidate: Option<BestAxisCandidate> = None;
        for &idx in sorted_rows {
            if !active[idx] {
                continue;
            }
            let current_value = x.get(idx, feature);
            let Some((previous_value, previous_idx)) = previous else {
                previous = Some((current_value, idx));
                continue;
            };
            let weight = weights[previous_idx];
            let value = target[previous_idx];
            left_weight += weight;
            left_weighted_target += weight * value;
            left_weighted_target_sq += weight * value * value;
            left_count += 1;

            if previous_value == current_value {
                previous = Some((current_value, idx));
                continue;
            }

            let right_count = active_count - left_count;
            if left_count < self.min_samples_leaf || right_count < self.min_samples_leaf {
                previous = Some((current_value, idx));
                continue;
            }

            let right_weight = total_weight - left_weight;
            let right_weighted_target = total_weighted_target - left_weighted_target;
            let right_weighted_target_sq = total_weighted_target_sq - left_weighted_target_sq;
            let gain = parent_sse
                - weighted_sse_from_sums(
                    left_weight,
                    left_weighted_target,
                    left_weighted_target_sq,
                )
                - weighted_sse_from_sums(
                    right_weight,
                    right_weighted_target,
                    right_weighted_target_sq,
                );
            let split = Split::Axis {
                feature,
                threshold: (previous_value + current_value) / 2.0,
                missing_goes_left: true,
            };
            if candidate
                .as_ref()
                .is_none_or(|old| is_better_split_candidate(gain, &split, old.gain, &old.split))
            {
                candidate = Some(BestAxisCandidate {
                    split,
                    gain,
                    feature,
                    split_position: left_count - 1,
                    left_capacity: left_count,
                    right_capacity: right_count,
                });
            }
            previous = Some((current_value, idx));
        }

        candidate
    }
}
