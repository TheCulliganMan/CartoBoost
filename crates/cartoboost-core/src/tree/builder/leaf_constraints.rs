impl TreeBuilder {
    fn uses_l2_split_score(&self) -> bool {
        matches!(self.loss, LossConfig::L2 | LossConfig::LogL2(_))
    }

    fn node_loss(&self, target: &[f64], weights: &[f64], indices: &[usize]) -> f64 {
        match self.loss {
            LossConfig::L2 | LossConfig::LogL2(_) => sse(target, weights, indices),
            LossConfig::L1 => weighted_absolute_loss(target, weights, indices),
            LossConfig::Huber(config) => {
                let value = self.leaf_value(target, weights, indices, None);
                indices
                    .iter()
                    .map(|&idx| weights[idx] * huber_loss(target[idx], value, config.delta))
                    .sum()
            }
            LossConfig::Quantile(config) => {
                weighted_pinball_loss(target, weights, indices, config.alpha)
            }
        }
    }

    fn leaf_value(
        &self,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        stats: Option<CandidateStats>,
    ) -> f64 {
        match self.loss {
            LossConfig::L2 | LossConfig::LogL2(_) | LossConfig::Huber(_) => stats.map_or_else(
                || {
                    let stats = candidate_stats(indices.iter().copied(), target, weights);
                    self.constant_leaf_value(stats)
                },
                |stats| self.constant_leaf_value(stats),
            ),
            LossConfig::L1 => {
                let values = indices.iter().map(|&idx| target[idx]).collect::<Vec<_>>();
                let selected_weights = indices.iter().map(|&idx| weights[idx]).collect::<Vec<_>>();
                weighted_quantile(&values, &selected_weights, 0.5)
            }
            LossConfig::Quantile(config) => {
                let values = indices.iter().map(|&idx| target[idx]).collect::<Vec<_>>();
                let selected_weights = indices.iter().map(|&idx| weights[idx]).collect::<Vec<_>>();
                weighted_quantile(&values, &selected_weights, config.alpha)
            }
        }
    }

    fn constant_leaf_value(&self, stats: CandidateStats) -> f64 {
        constant_leaf_value(stats, self.constant_lambda_l2)
    }

    fn leaf_training_loss(
        &self,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        value: f64,
    ) -> f64 {
        match self.loss {
            LossConfig::L2 | LossConfig::LogL2(_) => indices
                .iter()
                .map(|&idx| weights[idx] * (target[idx] - value).powi(2))
                .sum(),
            LossConfig::L1 => indices
                .iter()
                .map(|&idx| weights[idx] * absolute_loss(target[idx], value))
                .sum(),
            LossConfig::Huber(config) => indices
                .iter()
                .map(|&idx| weights[idx] * huber_loss(target[idx], value, config.delta))
                .sum(),
            LossConfig::Quantile(config) => indices
                .iter()
                .map(|&idx| weights[idx] * pinball_loss(target[idx], value, config.alpha))
                .sum(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn monotonic_split_allowed(
        &self,
        split: &Split,
        target: &[f64],
        left_weights: &[f64],
        right_weights: &[f64],
        left: &[usize],
        right: &[usize],
    ) -> bool {
        let Some((feature, direction)) = self.axis_monotonic_direction(split) else {
            return true;
        };
        if direction == 0 {
            return true;
        }
        let left_value = self.leaf_value(target, left_weights, left, None);
        let right_value = self.leaf_value(target, right_weights, right, None);
        let allowed = if direction > 0 {
            left_value <= right_value + 1e-12
        } else {
            left_value + 1e-12 >= right_value
        };
        let _ = feature;
        allowed
    }

    #[allow(clippy::too_many_arguments)]
    fn child_bounds(
        &self,
        split: &Split,
        target: &[f64],
        left_weights: &[f64],
        right_weights: &[f64],
        left: &[usize],
        right: &[usize],
        left_stats: Option<CandidateStats>,
        right_stats: Option<CandidateStats>,
        lower_bound: f64,
        upper_bound: f64,
    ) -> (f64, f64, f64, f64) {
        let Some((_feature, direction)) = self.axis_monotonic_direction(split) else {
            return (lower_bound, upper_bound, lower_bound, upper_bound);
        };
        if direction == 0 {
            return (lower_bound, upper_bound, lower_bound, upper_bound);
        }
        let left_value = self.leaf_value(target, left_weights, left, left_stats);
        let right_value = self.leaf_value(target, right_weights, right, right_stats);
        let middle = ((left_value + right_value) / 2.0).clamp(lower_bound, upper_bound);
        if direction > 0 {
            (lower_bound, middle, middle, upper_bound)
        } else {
            (middle, upper_bound, lower_bound, middle)
        }
    }

    fn axis_monotonic_direction(&self, split: &Split) -> Option<(usize, i8)> {
        let feature = match split {
            Split::Axis { feature, .. } => *feature,
            _ => return None,
        };
        self.monotonic_constraints
            .get(feature)
            .copied()
            .map(|direction| (feature, direction))
    }

    fn interaction_split_allowed(
        &self,
        active_features: &[usize],
        candidate_features: &[usize],
    ) -> bool {
        if self.interaction_constraints.is_empty() {
            return true;
        }
        let mut features = active_features.to_vec();
        features.extend(candidate_features.iter().copied());
        features.sort_unstable();
        features.dedup();
        self.interaction_constraints.iter().any(|group| {
            features
                .iter()
                .all(|feature| group.binary_search(feature).is_ok())
        })
    }

    fn child_active_features(
        &self,
        x: &Dataset,
        active_features: &[usize],
        split: &Split,
    ) -> Vec<usize> {
        if self.interaction_constraints.is_empty() {
            return Vec::new();
        }
        let mut features = active_features.to_vec();
        features.extend(split_feature_indices(split, x.n_cols()));
        features.sort_unstable();
        features.dedup();
        features
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_split_candidate(
        &self,
        split: Split,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        parent_sse: f64,
    ) -> Option<BestSplit> {
        let scoring_split = if self.fuzzy && self.fuzzy_bandwidth > 0.0 {
            Split::Fuzzy {
                base: Box::new(split),
                bandwidth: self.fuzzy_bandwidth,
                kernel: self.fuzzy_kernel,
            }
        } else {
            split
        };
        let mut left = Vec::new();
        let mut right = Vec::new();
        let mut left_weights = vec![0.0; weights.len()];
        let mut right_weights = vec![0.0; weights.len()];
        for &idx in indices {
            let branch_weights = scoring_split.branch_weights_dataset_row(x, idx);
            if branch_weights.left > 0.0 {
                left.push(idx);
                left_weights[idx] = weights[idx] * branch_weights.left;
            }
            if branch_weights.right > 0.0 {
                right.push(idx);
                right_weights[idx] = weights[idx] * branch_weights.right;
            }
        }
        if left.len() < self.min_samples_leaf || right.len() < self.min_samples_leaf {
            return None;
        }
        if !self.monotonic_split_allowed(
            &scoring_split,
            target,
            &left_weights,
            &right_weights,
            &left,
            &right,
        ) {
            return None;
        }
        let gain = parent_sse
            - self.node_loss(target, &left_weights, &left)
            - self.node_loss(target, &right_weights, &right);
        let gain = self
            .graph_adjusted_gain(gain, target, &left_weights, &right_weights, &left, &right)
            .ok()?;
        Some(BestSplit {
            split: scoring_split,
            gain,
            left_weights: Some(left_weights),
            right_weights: Some(right_weights),
            left,
            right,
            left_direct_node: None,
            right_direct_node: None,
            left_node_stats: None,
            right_node_stats: None,
            left_histogram_stats: None,
            right_histogram_stats: None,
        })
    }

    fn graph_adjusted_gain(
        &self,
        ordinary_gain: f64,
        target: &[f64],
        left_weights: &[f64],
        right_weights: &[f64],
        left: &[usize],
        right: &[usize],
    ) -> Result<f64> {
        let Some(regularization) = &self.graph_split_regularization else {
            return Ok(ordinary_gain);
        };
        if regularization.lambda == 0.0 {
            return Ok(ordinary_gain);
        }
        let left_value = self.leaf_value(target, left_weights, left, None);
        let right_value = self.leaf_value(target, right_weights, right, None);
        let mut updates = vec![0.0; target.len()];
        for &idx in left {
            updates[idx] = left_value;
        }
        for &idx in right {
            updates[idx] = right_value;
        }
        regularization.adjusted_gain(ordinary_gain, &updates)
    }
}

