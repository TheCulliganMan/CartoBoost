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

// Tree-building stages share this module namespace.
include!("builder/build.rs");
include!("builder/axis_candidates.rs");
include!("builder/spatial_candidates.rs");
include!("builder/periodic_sparse_candidates.rs");
include!("builder/leaf_constraints.rs");
include!("builder/helpers.rs");
include!("builder/tests.rs");
