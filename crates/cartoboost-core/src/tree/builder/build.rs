impl TreeBuilder {
    pub(crate) fn fit_context(&self, x: &Dataset) -> FitContext {
        FitContext::new(x, &self.splitters)
    }

    pub fn fit(&self, x: &Dataset, target: &[f64], weights: &[f64]) -> Tree {
        let context = FitContext::new(x, &self.splitters);
        self.fit_in_context(x, target, weights, &context)
    }

    pub(crate) fn fit_in_context(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        context: &FitContext,
    ) -> Tree {
        let indices = (0..x.n_rows()).collect::<Vec<_>>();
        Tree {
            root: self.build_node_inner(
                x,
                target,
                weights,
                &indices,
                0,
                context,
                None,
                None,
                None,
                f64::NEG_INFINITY,
                f64::INFINITY,
                &[],
            ),
        }
    }

    pub fn fit_with_leaf_updates(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
    ) -> (Tree, Vec<f64>) {
        let context = FitContext::new(x, &self.splitters);
        self.fit_with_leaf_updates_in_context(x, target, weights, &context)
    }

    pub(crate) fn fit_with_leaf_updates_in_context(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        context: &FitContext,
    ) -> (Tree, Vec<f64>) {
        let indices = (0..x.n_rows()).collect::<Vec<_>>();
        let mut updates = vec![0.0; x.n_rows()];
        let root = self.build_node_inner(
            x,
            target,
            weights,
            &indices,
            0,
            context,
            Some(&mut updates),
            None,
            None,
            f64::NEG_INFINITY,
            f64::INFINITY,
            &[],
        );
        (Tree { root }, updates)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_node_inner(
        &self,
        x: &Dataset,
        target: &[f64],
        weights: &[f64],
        indices: &[usize],
        depth: usize,
        context: &FitContext,
        mut updates: Option<&mut [f64]>,
        node_histogram_stats: Option<&[CandidateStats]>,
        node_stats: Option<CandidateStats>,
        lower_bound: f64,
        upper_bound: f64,
        active_features: &[usize],
    ) -> Node {
        let leaf = |updates: Option<&mut [f64]>| {
            let started = profile::ProfileTimer::start();
            let node_stats = node_stats
                .or_else(|| {
                    node_histogram_stats.and_then(|stats| histogram_node_stats(context, stats))
                })
                .unwrap_or_else(|| {
                    let mut stats = CandidateStats::default();
                    for &idx in indices {
                        stats.add_row(idx, target, weights);
                    }
                    stats
                });
            let weight_sum = node_stats.weight_sum;
            let raw_value = self.leaf_value(target, weights, indices, Some(node_stats));
            let value = raw_value.clamp(lower_bound, upper_bound);
            if let Some(updates) = updates {
                for &idx in indices {
                    updates[idx] = value;
                }
            }
            let training_loss = if matches!(self.loss, LossConfig::L2 | LossConfig::LogL2(_))
                && value == raw_value
            {
                node_stats.sse()
            } else {
                self.leaf_training_loss(target, weights, indices, value)
            };
            match self.leaf_predictor {
                LeafPredictorKind::Constant => {
                    profile::add(profile::LEAF, started.elapsed());
                    Node::Leaf {
                        value,
                        sample_weight_sum: weight_sum,
                        training_loss,
                    }
                }
                LeafPredictorKind::Linear => {
                    let features = if self.linear_leaf_features.is_empty() {
                        (0..x.n_cols()).collect()
                    } else {
                        self.linear_leaf_features.clone()
                    };
                    let rows = indices
                        .iter()
                        .map(|&idx| (0..x.n_cols()).map(|col| x.get(idx, col)).collect())
                        .collect::<Vec<Vec<f64>>>();
                    let leaf_targets = indices.iter().map(|&idx| target[idx]).collect::<Vec<_>>();
                    let leaf_weights = indices.iter().map(|&idx| weights[idx]).collect::<Vec<_>>();
                    let node = LinearLeafPredictor::fit_ridge(
                        &rows,
                        &leaf_targets,
                        &leaf_weights,
                        features,
                        self.linear_lambda_l2,
                    )
                    .map(|model| Node::LinearLeaf {
                        model,
                        sample_weight_sum: weight_sum,
                        training_loss,
                    })
                    .unwrap_or(Node::Leaf {
                        value,
                        sample_weight_sum: weight_sum,
                        training_loss,
                    });
                    profile::add(profile::LEAF, started.elapsed());
                    node
                }
            }
        };

        if depth >= self.max_depth || indices.len() < self.min_samples_leaf * 2 {
            return leaf(updates);
        }

        let Some(best) = self.best_split(
            x,
            target,
            weights,
            indices,
            context,
            node_histogram_stats,
            depth + 1 < self.max_depth,
            updates.as_deref_mut(),
            active_features,
        ) else {
            return leaf(updates);
        };
        if best.gain < self.min_gain {
            return leaf(updates);
        }

        let BestSplit {
            split,
            gain,
            left,
            right,
            left_direct_node,
            right_direct_node,
            left_weights,
            right_weights,
            left_node_stats,
            right_node_stats,
            left_histogram_stats,
            right_histogram_stats,
        } = best;
        let sample_weight_sum = node_stats
            .or_else(|| node_histogram_stats.and_then(|stats| histogram_node_stats(context, stats)))
            .map(|stats| stats.weight_sum)
            .unwrap_or_else(|| indices.iter().map(|&idx| weights[idx]).sum());
        let left_weight_values = left_weights.as_deref().unwrap_or(weights);
        let right_weight_values = right_weights.as_deref().unwrap_or(weights);
        let (left_lower_bound, left_upper_bound, right_lower_bound, right_upper_bound) = self
            .child_bounds(
                &split,
                target,
                left_weight_values,
                right_weight_values,
                &left,
                &right,
                left_node_stats,
                right_node_stats,
                lower_bound,
                upper_bound,
            );
        let left_node;
        let right_node;
        let child_active_features = self.child_active_features(x, active_features, &split);
        if let (Some(left_direct_node), Some(right_direct_node)) =
            (left_direct_node, right_direct_node)
        {
            left_node = left_direct_node;
            right_node = right_direct_node;
        } else if let Some(updates) = updates {
            left_node = self.build_node_inner(
                x,
                target,
                left_weight_values,
                &left,
                depth + 1,
                context,
                Some(updates),
                left_histogram_stats.as_deref(),
                left_node_stats,
                left_lower_bound,
                left_upper_bound,
                &child_active_features,
            );
            right_node = self.build_node_inner(
                x,
                target,
                right_weight_values,
                &right,
                depth + 1,
                context,
                Some(updates),
                right_histogram_stats.as_deref(),
                right_node_stats,
                right_lower_bound,
                right_upper_bound,
                &child_active_features,
            );
        } else {
            let (built_left, built_right) = rayon::join(
                || {
                    self.build_node_inner(
                        x,
                        target,
                        left_weight_values,
                        &left,
                        depth + 1,
                        context,
                        None,
                        left_histogram_stats.as_deref(),
                        left_node_stats,
                        left_lower_bound,
                        left_upper_bound,
                        &child_active_features,
                    )
                },
                || {
                    self.build_node_inner(
                        x,
                        target,
                        right_weight_values,
                        &right,
                        depth + 1,
                        context,
                        None,
                        right_histogram_stats.as_deref(),
                        right_node_stats,
                        right_lower_bound,
                        right_upper_bound,
                        &child_active_features,
                    )
                },
            );
            left_node = built_left;
            right_node = built_right;
        }
        Node::Branch {
            split,
            left: Box::new(left_node),
            right: Box::new(right_node),
            gain,
            sample_weight_sum,
        }
    }
}
