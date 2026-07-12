use super::{sse, FuzzyKernel, Node, Split, Tree};
use crate::data::{Dataset, FeatureKind};
use crate::graph_regularization::GraphSplitRegularization;
use crate::loss::{
    absolute_loss, pinball_loss, weighted_absolute_loss, weighted_pinball_loss, weighted_quantile,
    LossConfig,
};
use crate::predictors::LinearLeafPredictor;
use crate::profile;
use crate::Result;
use rayon::iter::Either;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct TreeBuilder {
    pub max_depth: usize,
    pub min_samples_leaf: usize,
    pub min_gain: f64,
    pub splitters: Vec<SplitterKind>,
    pub leaf_predictor: LeafPredictorKind,
    pub linear_leaf_features: Vec<usize>,
    pub linear_lambda_l2: f64,
    pub constant_lambda_l2: f64,
    pub fuzzy: bool,
    pub fuzzy_bandwidth: f64,
    pub fuzzy_kernel: FuzzyKernel,
    pub loss: LossConfig,
    pub monotonic_constraints: Vec<i8>,
    pub interaction_constraints: Vec<Vec<usize>>,
    pub graph_split_regularization: Option<GraphSplitRegularization>,
}

#[derive(Debug, Clone)]
struct BestSplit {
    split: Split,
    gain: f64,
    left: Vec<usize>,
    right: Vec<usize>,
    left_direct_node: Option<Node>,
    right_direct_node: Option<Node>,
    left_weights: Option<Vec<f64>>,
    right_weights: Option<Vec<f64>>,
    left_node_stats: Option<CandidateStats>,
    right_node_stats: Option<CandidateStats>,
    left_histogram_stats: Option<Vec<CandidateStats>>,
    right_histogram_stats: Option<Vec<CandidateStats>>,
}

#[derive(Debug, Clone)]
struct BestHistogramCandidate {
    split: Split,
    gain: f64,
    split_bin: usize,
    left_capacity: usize,
    right_capacity: usize,
    left_stats: CandidateStats,
    right_stats: CandidateStats,
}

impl BestHistogramCandidate {
    fn feature(&self) -> Option<usize> {
        match self.split {
            Split::Axis { feature, .. } => Some(feature),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
struct BestAxisCandidate {
    split: Split,
    gain: f64,
    feature: usize,
    split_position: usize,
    left_capacity: usize,
    right_capacity: usize,
}

#[derive(Debug, Clone)]
struct BestOrderedCandidate {
    split: Split,
    gain: f64,
    split_position: usize,
    left_capacity: usize,
    right_capacity: usize,
    left_stats: CandidateStats,
    right_stats: CandidateStats,
}

#[derive(Debug, Clone)]
struct BestOrderedSplitCandidate {
    candidate: BestOrderedCandidate,
    pairs: Vec<(f64, usize)>,
}

#[derive(Debug, Clone)]
struct PeriodicValueGroup {
    value: f64,
    count: usize,
    weight_sum: f64,
    weighted_target_sum: f64,
    weighted_target_square_sum: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct CandidateStats {
    count: usize,
    weight_sum: f64,
    weighted_target_sum: f64,
    weighted_target_square_sum: f64,
}

#[derive(Debug, Clone)]
pub(crate) struct FitContext {
    cols: usize,
    sorted_dense_rows: Vec<Option<Vec<usize>>>,
    histogram_bins: Option<usize>,
    histogram_features: Vec<Option<HistogramFeature>>,
    histogram_feature_indices: Vec<usize>,
    histogram_all_features: bool,
    histogram_row_bins: Vec<u16>,
}

#[derive(Debug, Clone)]
struct HistogramFeature {
    bin_count: usize,
    thresholds: Vec<f64>,
    bins: Vec<u16>,
}

const MISSING_BIN: u16 = u16::MAX;
const SPATIAL_SPLIT_RELATIVE_GAIN_MARGIN: f64 = 0.10;

impl FitContext {
    fn new(x: &Dataset, splitters: &[SplitterKind]) -> Self {
        let needs_exact_axis_order = splitters
            .iter()
            .any(|splitter| matches!(splitter, SplitterKind::Auto | SplitterKind::Axis));
        let histogram_bins = splitters.iter().find_map(|splitter| match splitter {
            SplitterKind::AxisHistogram { bins } => Some((*bins).clamp(2, 1024)),
            _ => None,
        });
        let sorted_dense_rows = if needs_exact_axis_order {
            (0..x.n_cols())
                .into_par_iter()
                .map(|feature| {
                    let mut rows = (0..x.n_rows())
                        .filter(|&row| x.get(row, feature).is_finite())
                        .collect::<Vec<_>>();
                    rows.sort_by(|&left, &right| {
                        x.get(left, feature)
                            .total_cmp(&x.get(right, feature))
                            .then(left.cmp(&right))
                    });
                    (!rows.is_empty()).then_some(rows)
                })
                .collect()
        } else {
            Vec::new()
        };
        let histogram_features: Vec<Option<HistogramFeature>> = histogram_bins
            .map(|bins| {
                (0..x.n_cols())
                    .into_par_iter()
                    .map(|feature| prebinned_histogram_feature(x, feature, bins))
                    .collect()
            })
            .unwrap_or_default();
        let histogram_row_bins = if histogram_bins.is_some() {
            let mut row_bins = vec![MISSING_BIN; x.n_rows() * x.n_cols()];
            row_bins
                .par_chunks_mut(x.n_cols())
                .enumerate()
                .for_each(|(row, row_bins)| {
                    for (feature, histogram_feature) in histogram_features.iter().enumerate() {
                        if let Some(histogram_feature) = histogram_feature {
                            row_bins[feature] = histogram_feature.bins[row];
                        }
                    }
                });
            row_bins
        } else {
            Vec::new()
        };
        let histogram_feature_indices = histogram_features
            .iter()
            .enumerate()
            .filter_map(|(feature, histogram_feature)| histogram_feature.as_ref().map(|_| feature))
            .collect::<Vec<_>>();
        let histogram_all_features =
            histogram_bins.is_some() && histogram_feature_indices.len() == x.n_cols();
        Self {
            cols: x.n_cols(),
            sorted_dense_rows,
            histogram_bins,
            histogram_features,
            histogram_feature_indices,
            histogram_all_features,
            histogram_row_bins,
        }
    }

    fn sorted_rows(&self, feature: usize) -> Option<&[usize]> {
        self.sorted_dense_rows
            .get(feature)
            .and_then(Option::as_deref)
    }

    fn histogram_feature(&self, feature: usize, bins: usize) -> Option<&HistogramFeature> {
        (self.histogram_bins == Some(bins))
            .then(|| self.histogram_features.get(feature))
            .flatten()
            .and_then(Option::as_ref)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum SplitterKind {
    #[default]
    Auto,
    Axis,
    AxisHistogram {
        bins: usize,
    },
    Diagonal2D,
    Gaussian2D,
    Periodic {
        period: f64,
    },
    SparseSet,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum LeafPredictorKind {
    #[default]
    Constant,
    Linear,
}

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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn quantile_histogram_thresholds_balance_skewed_features() {
        let mut values = vec![0.0; 50];
        values.extend((1..=50).map(|value| value as f64 + 0.25));

        let thresholds = quantile_histogram_thresholds(values, 4);

        assert_eq!(thresholds.len(), 2);
        assert!(thresholds[0] < thresholds[1]);
        assert!(
            thresholds[0] <= 1.0,
            "first threshold should stay near the dense mass, got {}",
            thresholds[0]
        );
        assert!(thresholds[1] > 20.0);
    }

    #[test]
    fn quantile_histogram_thresholds_collapse_duplicate_boundaries() {
        let thresholds = quantile_histogram_thresholds(vec![1.0, 1.0, 1.0, 2.0, 2.0], 8);

        assert_eq!(thresholds, vec![1.5]);
    }

    #[test]
    fn histogram_thresholds_keep_low_cardinality_features_exact() {
        let thresholds = quantile_histogram_thresholds(vec![0.0, 0.0, 1.0, 2.0, 2.0], 8);

        assert_eq!(thresholds, vec![0.5, 1.5]);
    }

    #[test]
    fn histogram_thresholds_keep_integer_id_features_fixed_width() {
        let values = (1..=100).map(|value| value as f64).collect::<Vec<_>>();

        let thresholds = quantile_histogram_thresholds(values, 4);

        assert_close(thresholds[0], 25.75);
        assert_close(thresholds[1], 50.5);
        assert_close(thresholds[2], 75.25);
    }

    #[test]
    fn histogram_thresholds_keep_repeated_encoded_features_fixed_width() {
        let mut values = Vec::new();
        for bucket in 0..20 {
            values.extend(std::iter::repeat_n(bucket as f64 / 10.0, 20));
        }

        let thresholds = quantile_histogram_thresholds(values, 4);

        assert_close(thresholds[0], 0.475);
        assert_close(thresholds[1], 0.95);
        assert_close(thresholds[2], 1.4249999999999998);
    }

    #[test]
    fn one_stump_finds_golden_axis_split_with_constant_leaves() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 1.0, 1.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        match tree.root {
            Node::Branch {
                split, left, right, ..
            } => {
                match split {
                    Split::Axis {
                        feature, threshold, ..
                    } => {
                        assert_eq!(feature, 0);
                        assert_close(threshold, 1.5);
                    }
                    other => panic!("expected axis split, got {other:?}"),
                }
                match (*left, *right) {
                    (Node::Leaf { value: left, .. }, Node::Leaf { value: right, .. }) => {
                        assert_close(left, 0.0);
                        assert_close(right, 1.0);
                    }
                    other => panic!("expected constant leaves, got {other:?}"),
                }
            }
            other => panic!("expected branch root, got {other:?}"),
        }
    }

    #[test]
    fn histogram_axis_splitter_fits_monotonic_stump() {
        let x = Dataset::from_rows(vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![4.0],
            vec![5.0],
        ])
        .unwrap();
        let y = vec![0.0, 0.0, 0.0, 5.0, 5.0, 5.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::AxisHistogram { bins: 4 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::Axis { .. },
                ..
            }
        ));
        assert!(tree.predict_dataset_row(&x, 0) < tree.predict_dataset_row(&x, 5));
    }

    #[test]
    fn fuzzy_training_uses_fractional_child_weights() {
        let x = Dataset::from_rows(vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]).unwrap();
        let y = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Axis],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: true,
            fuzzy_bandwidth: 2.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        match tree.root {
            Node::Branch {
                split, left, right, ..
            } => {
                assert!(matches!(split, Split::Fuzzy { .. }));
                match (*left, *right) {
                    (Node::Leaf { value: left, .. }, Node::Leaf { value: right, .. }) => {
                        assert!(left > 0.0 && left < 5.0, "left leaf was {left}");
                        assert!(right > 5.0 && right < 10.0, "right leaf was {right}");
                    }
                    other => panic!("expected constant leaves, got {other:?}"),
                }
            }
            other => panic!("expected branch root, got {other:?}"),
        }
    }

    #[test]
    fn sparse_set_splitter_trains_on_list_valued_rows() {
        let dense = Dataset::from_rows(vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0]]).unwrap();
        let x = dense
            .with_sparse_sets(vec![crate::data::SparseSetColumn::new(vec![
                vec![10, 20],
                vec![20, 30],
                vec![40],
                vec![],
            ])])
            .unwrap();
        let y = vec![7.0, 7.0, -2.0, -2.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert_eq!(
            (0..x.n_rows())
                .map(|row| tree.predict_dataset_row(&x, row))
                .collect::<Vec<_>>(),
            y
        );
        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::SparseListContainsAny { .. },
                ..
            }
        ));
    }

    #[test]
    fn schema_declared_periodic_feature_does_not_need_full_observed_cycle() {
        let x = Dataset::from_rows(vec![vec![7.0], vec![8.0], vec![9.0], vec![10.0]])
            .unwrap()
            .with_schema(crate::data::FeatureSchema {
                names: vec!["hour".to_string()],
                kinds: vec![FeatureKind::Periodic { period: 24 }],
            })
            .unwrap();
        let y = vec![3.0, 3.0, -1.0, -1.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert_eq!(
            (0..x.n_rows())
                .map(|row| tree.predict_dataset_row(&x, row))
                .collect::<Vec<_>>(),
            y
        );
        assert!(matches!(
            tree.root,
            Node::Branch {
                split: Split::PeriodicInterval { .. },
                ..
            }
        ));
    }

    #[test]
    fn schema_present_periodic_splitter_ignores_non_periodic_columns() {
        let x = Dataset::from_rows(vec![
            vec![0.0, 7.0],
            vec![1.0, 7.0],
            vec![23.0, 7.0],
            vec![24.0, 7.0],
        ])
        .unwrap()
        .with_schema(crate::data::FeatureSchema {
            names: vec!["numeric_covering_period".to_string(), "hour".to_string()],
            kinds: vec![FeatureKind::Numeric, FeatureKind::Periodic { period: 24 }],
        })
        .unwrap();
        let y = vec![0.0, 0.0, 10.0, 10.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 1,
            min_gain: 0.0,
            splitters: vec![SplitterKind::Periodic { period: 24.0 }],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(tree.root, Node::Leaf { .. }));
    }

    #[test]
    fn schema_present_sparse_splitter_ignores_dense_scalar_id_columns() {
        let x = Dataset::mixed(
            vec![vec![7.0], vec![7.0], vec![2.0], vec![2.0]],
            vec![crate::data::SparseSetColumn::new(vec![
                vec![100],
                vec![101],
                vec![102],
                vec![103],
            ])],
            Some(crate::data::FeatureSchema {
                names: vec!["dense_id_like".to_string(), "route_cells".to_string()],
                kinds: vec![FeatureKind::Numeric, FeatureKind::SparseSet],
            }),
        )
        .unwrap();
        let y = vec![9.0, 9.0, -4.0, -4.0];
        let weights = vec![1.0; y.len()];
        let builder = TreeBuilder {
            max_depth: 1,
            min_samples_leaf: 2,
            min_gain: 0.0,
            splitters: vec![SplitterKind::SparseSet],
            leaf_predictor: LeafPredictorKind::Constant,
            linear_leaf_features: Vec::new(),
            linear_lambda_l2: 1.0,
            constant_lambda_l2: 0.0,
            fuzzy: false,
            fuzzy_bandwidth: 0.0,
            fuzzy_kernel: FuzzyKernel::Linear,
            loss: crate::loss::LossConfig::L2,
            monotonic_constraints: Vec::new(),
            interaction_constraints: Vec::new(),
            graph_split_regularization: None,
        };

        let tree = builder.fit(&x, &y, &weights);

        assert!(matches!(tree.root, Node::Leaf { .. }));
    }
}
