use crate::{CartoBoostError, Result};
use cartoboost_accelerator::backend::{
    backend_csr_diffusion_f32, select_backend_for, BackendOperation,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const GRAPH_SMOOTHING_DISPATCH_MIN_VALUES: usize = 16_384;

fn should_accelerate_graph_smoothing(
    selected_backend: &str,
    edge_count: usize,
    iterations: usize,
) -> bool {
    selected_backend != "cpu"
        && edge_count.saturating_mul(iterations) >= GRAPH_SMOOTHING_DISPATCH_MIN_VALUES
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SymbolicRelationKind {
    OriginAdjacency,
    DestinationAdjacency,
    CorridorSimilarity,
    ReverseEdgeSimilarity,
    NeighborZoneSimilarity,
    SegmentSimilarity,
    EntitySimilarity,
    HierarchyParentChild,
    SmoothGroup,
    NoSmoothGroup,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CsrGraph {
    pub node_count: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub weights: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphLaplacian {
    pub graph: CsrGraph,
    pub degree: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GraphSmoother {
    pub lambda: f64,
    pub iterations: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphRegularizedBooster {
    pub baseline_predictions: Vec<f64>,
    pub laplacian: GraphLaplacian,
    pub lambda: f64,
    pub iterations: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphSplitRegularization {
    pub graph: CsrGraph,
    pub lambda: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphLeafSmoothing {
    pub graph: CsrGraph,
    pub lambda: f64,
    pub iterations: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicRelation {
    pub kind: SymbolicRelationKind,
    pub left: usize,
    pub right: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicRelationSet {
    pub node_count: usize,
    pub relations: Vec<SymbolicRelation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RelationPenalty {
    pub kind: SymbolicRelationKind,
    pub penalty: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleFeature {
    pub name: String,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintPenalty {
    pub name: String,
    pub penalty: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonotoneConstraint {
    pub feature: usize,
    pub direction: i8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MonotoneConstraintSet {
    pub constraints: Vec<MonotoneConstraint>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteractionConstraintSet {
    pub groups: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleCompilation {
    pub features: Vec<RuleFeature>,
    pub penalties: Vec<ConstraintPenalty>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuleCompiler {
    pub relation_set: SymbolicRelationSet,
    pub monotone_constraints: MonotoneConstraintSet,
    pub interaction_constraints: InteractionConstraintSet,
}

impl CsrGraph {
    pub fn new(
        node_count: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        weights: Vec<f64>,
    ) -> Result<Self> {
        if node_count == 0 {
            return Err(CartoBoostError::InvalidInput(
                "graph must contain at least one node".to_string(),
            ));
        }
        if indptr.len() != node_count + 1 {
            return Err(CartoBoostError::InvalidInput(
                "CSR indptr length must equal node_count + 1".to_string(),
            ));
        }
        if indptr.first().copied() != Some(0) || indptr.last().copied() != Some(indices.len()) {
            return Err(CartoBoostError::InvalidInput(
                "CSR indptr must start at 0 and end at edge count".to_string(),
            ));
        }
        if indices.len() != weights.len() {
            return Err(CartoBoostError::InvalidInput(
                "CSR indices and weights must have the same length".to_string(),
            ));
        }
        if indptr.windows(2).any(|window| window[1] < window[0]) {
            return Err(CartoBoostError::InvalidInput(
                "CSR indptr must be non-decreasing".to_string(),
            ));
        }
        if indices.iter().any(|&node| node >= node_count) {
            return Err(CartoBoostError::InvalidInput(
                "CSR edge index out of node range".to_string(),
            ));
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        {
            return Err(CartoBoostError::InvalidInput(
                "CSR weights must be finite and non-negative".to_string(),
            ));
        }
        Ok(Self {
            node_count,
            indptr,
            indices,
            weights,
        })
    }

    pub fn from_edges(node_count: usize, edges: &[(usize, usize, f64)]) -> Result<Self> {
        if edges.iter().any(|&(left, right, weight)| {
            left >= node_count || right >= node_count || !weight.is_finite() || weight < 0.0
        }) {
            return Err(CartoBoostError::InvalidInput(
                "graph edges must reference valid nodes with finite non-negative weights"
                    .to_string(),
            ));
        }
        let mut rows = vec![Vec::<(usize, f64)>::new(); node_count];
        for &(left, right, weight) in edges {
            if weight > 0.0 {
                rows[left].push((right, weight));
            }
        }
        for row in &mut rows {
            row.sort_by_key(|left| left.0);
        }
        let mut indptr = Vec::with_capacity(node_count + 1);
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        indptr.push(0);
        for row in rows {
            for (idx, weight) in row {
                indices.push(idx);
                weights.push(weight);
            }
            indptr.push(indices.len());
        }
        Self::new(node_count, indptr, indices, weights)
    }

    pub fn neighbors(&self, node: usize) -> Result<impl Iterator<Item = (usize, f64)> + '_> {
        if node >= self.node_count {
            return Err(CartoBoostError::InvalidInput(
                "node index out of graph range".to_string(),
            ));
        }
        let start = self.indptr[node];
        let end = self.indptr[node + 1];
        Ok(self.indices[start..end]
            .iter()
            .copied()
            .zip(self.weights[start..end].iter().copied()))
    }

    pub fn edge_count(&self) -> usize {
        self.indices.len()
    }
}

impl GraphLaplacian {
    pub fn new(graph: CsrGraph) -> Self {
        let degree = (0..graph.node_count)
            .map(|node| {
                let start = graph.indptr[node];
                let end = graph.indptr[node + 1];
                graph.weights[start..end].iter().sum()
            })
            .collect();
        Self { graph, degree }
    }

    pub fn penalty(&self, values: &[f64]) -> Result<f64> {
        self.validate_values(values)?;
        let mut total = 0.0;
        for left in 0..self.graph.node_count {
            for (right, weight) in self.graph.neighbors(left)? {
                total += weight * (values[left] - values[right]).powi(2);
            }
        }
        Ok(0.5 * total)
    }

    pub fn gradient(&self, values: &[f64]) -> Result<Vec<f64>> {
        self.validate_values(values)?;
        let mut gradient = vec![0.0; self.graph.node_count];
        for left in 0..self.graph.node_count {
            for (right, weight) in self.graph.neighbors(left)? {
                let delta = values[left] - values[right];
                gradient[left] += weight * delta;
                gradient[right] -= weight * delta;
            }
        }
        Ok(gradient)
    }

    fn validate_values(&self, values: &[f64]) -> Result<()> {
        if values.len() != self.graph.node_count {
            return Err(CartoBoostError::InvalidInput(
                "graph value count must match node_count".to_string(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "graph values must be finite".to_string(),
            ));
        }
        Ok(())
    }
}

impl GraphSmoother {
    pub fn smooth(&self, values: &[f64], laplacian: &GraphLaplacian) -> Result<Vec<f64>> {
        self.smooth_with_backend(values, laplacian, Some("cpu"))
    }

    pub fn smooth_with_backend(
        &self,
        values: &[f64],
        laplacian: &GraphLaplacian,
        backend: Option<&str>,
    ) -> Result<Vec<f64>> {
        laplacian.validate_values(values)?;
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "graph smoothing lambda must be finite and non-negative".to_string(),
            ));
        }
        if self.lambda == 0.0 || self.iterations == 0 {
            return Ok(values.to_vec());
        }
        let backend = select_backend_for(backend, BackendOperation::CsrDiffusion)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        let use_accelerator = should_accelerate_graph_smoothing(
            &backend.selected,
            laplacian.graph.indices.len(),
            self.iterations,
        );
        let accelerator_csr = if !use_accelerator {
            None
        } else {
            let too_large = || {
                CartoBoostError::InvalidInput(
                    "graph is too large for accelerator CSR indices".to_string(),
                )
            };
            Some((
                laplacian
                    .graph
                    .indptr
                    .iter()
                    .map(|&value| u32::try_from(value).map_err(|_| too_large()))
                    .collect::<Result<Vec<_>>>()?,
                laplacian
                    .graph
                    .indices
                    .iter()
                    .map(|&value| u32::try_from(value).map_err(|_| too_large()))
                    .collect::<Result<Vec<_>>>()?,
                laplacian
                    .graph
                    .weights
                    .iter()
                    .map(|&value| value as f32)
                    .collect::<Vec<_>>(),
            ))
        };
        let mut current = values.to_vec();
        let mut next = current.clone();
        for _ in 0..self.iterations {
            let neighbor_sums = if !use_accelerator {
                None
            } else {
                let (indptr, indices, weights) = accelerator_csr.as_ref().expect("accelerator CSR");
                let current_f32 = current
                    .iter()
                    .map(|&value| value as f32)
                    .collect::<Vec<_>>();
                Some(
                    backend_csr_diffusion_f32(&backend, indptr, indices, weights, 1, &current_f32)
                        .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
                )
            };
            for node in 0..laplacian.graph.node_count {
                let degree = laplacian.degree[node];
                if degree <= 0.0 {
                    next[node] = current[node];
                    continue;
                }
                let neighbor_sum = neighbor_sums.as_ref().map_or_else(
                    || {
                        laplacian.graph.neighbors(node).map(|neighbors| {
                            neighbors
                                .map(|(neighbor, weight)| weight * current[neighbor])
                                .sum::<f64>()
                        })
                    },
                    |sums| Ok(f64::from(sums[node])),
                )?;
                next[node] =
                    (values[node] + self.lambda * neighbor_sum) / (1.0 + self.lambda * degree);
            }
            std::mem::swap(&mut current, &mut next);
        }
        Ok(current)
    }

    pub fn smooth_residuals(
        &self,
        residuals: &[f64],
        laplacian: &GraphLaplacian,
    ) -> Result<Vec<f64>> {
        self.smooth(residuals, laplacian)
    }

    pub fn smooth_residuals_with_backend(
        &self,
        residuals: &[f64],
        laplacian: &GraphLaplacian,
        backend: Option<&str>,
    ) -> Result<Vec<f64>> {
        self.smooth_with_backend(residuals, laplacian, backend)
    }

    pub fn smooth_leaf_values(
        &self,
        leaf_values: &[f64],
        laplacian: &GraphLaplacian,
    ) -> Result<Vec<f64>> {
        self.smooth(leaf_values, laplacian)
    }

    pub fn smooth_leaf_values_with_backend(
        &self,
        leaf_values: &[f64],
        laplacian: &GraphLaplacian,
        backend: Option<&str>,
    ) -> Result<Vec<f64>> {
        self.smooth_with_backend(leaf_values, laplacian, backend)
    }
}

impl GraphRegularizedBooster {
    pub fn new(
        baseline_predictions: Vec<f64>,
        graph: CsrGraph,
        lambda: f64,
        iterations: usize,
    ) -> Result<Self> {
        let laplacian = GraphLaplacian::new(graph);
        laplacian.validate_values(&baseline_predictions)?;
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "graph regularization lambda must be finite and non-negative".to_string(),
            ));
        }
        Ok(Self {
            baseline_predictions,
            laplacian,
            lambda,
            iterations,
        })
    }

    pub fn predict(&self) -> Result<Vec<f64>> {
        self.predict_with_backend(Some("cpu"))
    }

    pub fn predict_with_backend(&self, backend: Option<&str>) -> Result<Vec<f64>> {
        GraphSmoother {
            lambda: self.lambda,
            iterations: self.iterations,
        }
        .smooth_with_backend(&self.baseline_predictions, &self.laplacian, backend)
    }

    pub fn objective_value(&self, targets: &[f64], predictions: &[f64]) -> Result<f64> {
        if targets.len() != predictions.len() || targets.len() != self.baseline_predictions.len() {
            return Err(CartoBoostError::InvalidInput(
                "targets, predictions, and graph nodes must have the same length".to_string(),
            ));
        }
        if targets
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
        {
            return Err(CartoBoostError::InvalidInput(
                "targets and predictions must be finite".to_string(),
            ));
        }
        let ordinary = targets
            .iter()
            .zip(predictions)
            .map(|(&target, &prediction)| (target - prediction).powi(2))
            .sum::<f64>()
            / targets.len().max(1) as f64;
        Ok(ordinary + self.lambda * self.laplacian.penalty(predictions)?)
    }
}

impl GraphSplitRegularization {
    pub fn new(graph: CsrGraph, lambda: f64) -> Result<Self> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "graph split regularization lambda must be finite and non-negative".to_string(),
            ));
        }
        Ok(Self { graph, lambda })
    }

    pub fn validate_row_count(&self, row_count: usize) -> Result<()> {
        if self.graph.node_count != row_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "graph split regularization node_count {} must match training row count {row_count}",
                self.graph.node_count
            )));
        }
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "graph split regularization lambda must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }

    pub fn split_penalty(&self, updates: &[f64]) -> Result<f64> {
        self.validate_row_count(updates.len())?;
        if self.lambda == 0.0 {
            return Ok(0.0);
        }
        Ok(self.lambda * GraphLaplacian::new(self.graph.clone()).penalty(updates)?)
    }

    pub fn adjusted_gain(&self, ordinary_gain: f64, updates: &[f64]) -> Result<f64> {
        if !ordinary_gain.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "ordinary split gain must be finite".to_string(),
            ));
        }
        Ok(ordinary_gain - self.split_penalty(updates)?)
    }
}

impl GraphLeafSmoothing {
    pub fn new(graph: CsrGraph, lambda: f64, iterations: usize) -> Result<Self> {
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "graph leaf smoothing lambda must be finite and non-negative".to_string(),
            ));
        }
        Ok(Self {
            graph,
            lambda,
            iterations,
        })
    }

    pub fn validate_row_count(&self, row_count: usize) -> Result<()> {
        if self.graph.node_count != row_count {
            return Err(CartoBoostError::InvalidInput(format!(
                "graph leaf smoothing node_count {} must match training row count {row_count}",
                self.graph.node_count
            )));
        }
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "graph leaf smoothing lambda must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }

    pub fn smoother(&self) -> GraphSmoother {
        GraphSmoother {
            lambda: self.lambda,
            iterations: self.iterations,
        }
    }
}

impl SymbolicRelationSet {
    pub fn new(node_count: usize, relations: Vec<SymbolicRelation>) -> Result<Self> {
        if node_count == 0 {
            return Err(CartoBoostError::InvalidInput(
                "relation set must contain at least one node".to_string(),
            ));
        }
        for relation in &relations {
            relation.validate(node_count)?;
        }
        Ok(Self {
            node_count,
            relations,
        })
    }

    pub fn to_graph(&self, included: &[SymbolicRelationKind]) -> Result<CsrGraph> {
        let included = included.iter().copied().collect::<BTreeSet<_>>();
        let mut weights = BTreeMap::<(usize, usize), f64>::new();
        for relation in &self.relations {
            if relation.kind == SymbolicRelationKind::NoSmoothGroup
                || (!included.is_empty() && !included.contains(&relation.kind))
            {
                continue;
            }
            *weights
                .entry((relation.left, relation.right))
                .or_insert(0.0) += relation.weight;
            *weights
                .entry((relation.right, relation.left))
                .or_insert(0.0) += relation.weight;
        }
        let edges = weights
            .into_iter()
            .map(|((left, right), weight)| (left, right, weight))
            .collect::<Vec<_>>();
        CsrGraph::from_edges(self.node_count, &edges)
    }

    pub fn relation_penalties(&self, predictions: &[f64]) -> Result<Vec<RelationPenalty>> {
        if predictions.len() != self.node_count
            || predictions.iter().any(|value| !value.is_finite())
        {
            return Err(CartoBoostError::InvalidInput(
                "prediction count must match relation node_count and be finite".to_string(),
            ));
        }
        let mut penalties = BTreeMap::<SymbolicRelationKind, f64>::new();
        for relation in &self.relations {
            if relation.kind == SymbolicRelationKind::NoSmoothGroup {
                continue;
            }
            *penalties.entry(relation.kind).or_insert(0.0) += relation.weight
                * (predictions[relation.left] - predictions[relation.right]).powi(2);
        }
        Ok(penalties
            .into_iter()
            .map(|(kind, penalty)| RelationPenalty { kind, penalty })
            .collect())
    }
}

impl SymbolicRelation {
    pub fn new(kind: SymbolicRelationKind, left: usize, right: usize, weight: f64) -> Self {
        Self {
            kind,
            left,
            right,
            weight,
        }
    }

    fn validate(&self, node_count: usize) -> Result<()> {
        if self.left >= node_count || self.right >= node_count {
            return Err(CartoBoostError::InvalidInput(
                "relation endpoints must be valid node indices".to_string(),
            ));
        }
        if !self.weight.is_finite() || self.weight < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "relation weight must be finite and non-negative".to_string(),
            ));
        }
        Ok(())
    }
}

impl RuleFeature {
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "rule feature name must not be empty".to_string(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "rule feature values must be finite".to_string(),
            ));
        }
        Ok(Self { name, values })
    }
}

impl MonotoneConstraintSet {
    pub fn new(constraints: Vec<MonotoneConstraint>) -> Result<Self> {
        let mut seen = BTreeSet::new();
        for constraint in &constraints {
            if !matches!(constraint.direction, -1..=1) {
                return Err(CartoBoostError::InvalidInput(
                    "monotone constraint direction must be -1, 0, or 1".to_string(),
                ));
            }
            if !seen.insert(constraint.feature) {
                return Err(CartoBoostError::InvalidInput(
                    "monotone constraints must not repeat feature indices".to_string(),
                ));
            }
        }
        Ok(Self { constraints })
    }

    pub fn split_allowed(
        &self,
        feature: usize,
        left_prediction: f64,
        right_prediction: f64,
    ) -> Result<bool> {
        if !left_prediction.is_finite() || !right_prediction.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "monotone split predictions must be finite".to_string(),
            ));
        }
        let direction = self
            .constraints
            .iter()
            .find(|constraint| constraint.feature == feature)
            .map(|constraint| constraint.direction)
            .unwrap_or(0);
        Ok(match direction {
            1 => left_prediction <= right_prediction,
            -1 => left_prediction >= right_prediction,
            _ => true,
        })
    }

    pub fn penalties(
        &self,
        feature_values: &[Vec<f64>],
        predictions: &[f64],
    ) -> Result<Vec<ConstraintPenalty>> {
        validate_feature_matrix(feature_values, predictions.len())?;
        validate_prediction_vector(predictions)?;
        let mut penalties = Vec::new();
        for constraint in &self.constraints {
            if constraint.direction == 0 {
                continue;
            }
            let Some(values) = feature_values.get(constraint.feature) else {
                return Err(CartoBoostError::InvalidInput(
                    "monotone constraint feature out of range".to_string(),
                ));
            };
            let mut penalty = 0.0;
            for left in 0..predictions.len() {
                for right in left + 1..predictions.len() {
                    let feature_delta = values[right] - values[left];
                    if feature_delta == 0.0 {
                        continue;
                    }
                    let expected_order = constraint.direction as f64 * feature_delta.signum();
                    let prediction_delta = predictions[right] - predictions[left];
                    let violation = -expected_order * prediction_delta;
                    if violation > 0.0 {
                        penalty += violation;
                    }
                }
            }
            penalties.push(ConstraintPenalty {
                name: format!("monotone_feature_{}", constraint.feature),
                penalty,
            });
        }
        Ok(penalties)
    }
}

impl InteractionConstraintSet {
    pub fn new(mut groups: Vec<Vec<usize>>) -> Result<Self> {
        for group in &mut groups {
            group.sort_unstable();
            group.dedup();
            if group.is_empty() {
                return Err(CartoBoostError::InvalidInput(
                    "interaction constraint groups must not be empty".to_string(),
                ));
            }
        }
        groups.sort();
        Ok(Self { groups })
    }

    pub fn split_allowed(&self, active_features: &[usize], candidate_feature: usize) -> bool {
        if self.groups.is_empty() || active_features.is_empty() {
            return true;
        }
        self.groups.iter().any(|group| {
            group.binary_search(&candidate_feature).is_ok()
                && active_features
                    .iter()
                    .all(|feature| group.binary_search(feature).is_ok())
        })
    }

    pub fn penalty(&self, active_features: &[usize]) -> ConstraintPenalty {
        let mut features = active_features.to_vec();
        features.sort_unstable();
        features.dedup();
        let allowed = features.is_empty()
            || self.groups.is_empty()
            || self.groups.iter().any(|group| {
                features
                    .iter()
                    .all(|feature| group.binary_search(feature).is_ok())
            });
        ConstraintPenalty {
            name: "interaction_constraints".to_string(),
            penalty: if allowed { 0.0 } else { 1.0 },
        }
    }
}

impl RuleCompiler {
    pub fn new(
        relation_set: SymbolicRelationSet,
        monotone_constraints: MonotoneConstraintSet,
        interaction_constraints: InteractionConstraintSet,
    ) -> Self {
        Self {
            relation_set,
            monotone_constraints,
            interaction_constraints,
        }
    }

    pub fn compile(
        &self,
        predictions: &[f64],
        feature_values: &[Vec<f64>],
        active_features: &[usize],
    ) -> Result<RuleCompilation> {
        validate_prediction_vector(predictions)?;
        if predictions.len() != self.relation_set.node_count {
            return Err(CartoBoostError::InvalidInput(
                "rule compiler predictions must match relation node_count".to_string(),
            ));
        }
        let mut features = self.compile_relation_features()?;
        features.push(RuleFeature::new(
            "interaction_violation".to_string(),
            vec![
                self.interaction_constraints
                    .penalty(active_features)
                    .penalty;
                self.relation_set.node_count
            ],
        )?);
        let penalties = self.relation_set.relation_penalties(predictions)?;
        let mut constraint_penalties = penalties
            .into_iter()
            .map(|penalty| ConstraintPenalty {
                name: format!("relation_{:?}", penalty.kind),
                penalty: penalty.penalty,
            })
            .collect::<Vec<_>>();
        constraint_penalties.extend(
            self.monotone_constraints
                .penalties(feature_values, predictions)?,
        );
        constraint_penalties.push(self.interaction_constraints.penalty(active_features));
        constraint_penalties.sort_by(|left, right| left.name.cmp(&right.name));
        features.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(RuleCompilation {
            features,
            penalties: constraint_penalties,
        })
    }

    pub fn compile_relation_features(&self) -> Result<Vec<RuleFeature>> {
        let mut degree_by_kind = BTreeMap::<SymbolicRelationKind, Vec<f64>>::from_iter(
            all_relation_kinds().map(|kind| (kind, vec![0.0; self.relation_set.node_count])),
        );
        let mut reverse_edge = vec![0.0; self.relation_set.node_count];
        let mut hierarchy = vec![0.0; self.relation_set.node_count];
        for relation in &self.relation_set.relations {
            if relation.kind == SymbolicRelationKind::NoSmoothGroup {
                continue;
            }
            let values = degree_by_kind
                .get_mut(&relation.kind)
                .expect("all relation kinds initialized");
            values[relation.left] += relation.weight;
            values[relation.right] += relation.weight;
            if relation.kind == SymbolicRelationKind::ReverseEdgeSimilarity {
                reverse_edge[relation.left] += 1.0;
                reverse_edge[relation.right] += 1.0;
            }
            if relation.kind == SymbolicRelationKind::HierarchyParentChild {
                hierarchy[relation.left] += 1.0;
                hierarchy[relation.right] -= 1.0;
            }
        }
        let mut features = Vec::new();
        for (kind, values) in degree_by_kind {
            features.push(RuleFeature::new(
                format!("relation_degree_{kind:?}"),
                values,
            )?);
        }
        features.push(RuleFeature::new("reverse_edge_count", reverse_edge)?);
        features.push(RuleFeature::new(
            "hierarchy_parent_child_balance",
            hierarchy,
        )?);
        Ok(features)
    }
}

fn all_relation_kinds() -> impl Iterator<Item = SymbolicRelationKind> {
    [
        SymbolicRelationKind::OriginAdjacency,
        SymbolicRelationKind::DestinationAdjacency,
        SymbolicRelationKind::CorridorSimilarity,
        SymbolicRelationKind::ReverseEdgeSimilarity,
        SymbolicRelationKind::NeighborZoneSimilarity,
        SymbolicRelationKind::SegmentSimilarity,
        SymbolicRelationKind::EntitySimilarity,
        SymbolicRelationKind::HierarchyParentChild,
        SymbolicRelationKind::SmoothGroup,
    ]
    .into_iter()
}

fn validate_prediction_vector(predictions: &[f64]) -> Result<()> {
    if predictions.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "predictions must contain at least one value".to_string(),
        ));
    }
    if predictions.iter().any(|value| !value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "predictions must contain only finite values".to_string(),
        ));
    }
    Ok(())
}

fn validate_feature_matrix(feature_values: &[Vec<f64>], row_count: usize) -> Result<()> {
    for values in feature_values {
        if values.len() != row_count || values.iter().any(|value| !value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "feature value columns must match prediction length and be finite".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartoboost_accelerator::backend::available_backends;

    fn chain_graph() -> CsrGraph {
        CsrGraph::from_edges(3, &[(0, 1, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 1, 1.0)]).unwrap()
    }

    #[test]
    fn graph_regularized_booster_lambda_zero_matches_baseline() {
        let baseline = vec![1.0, 10.0, 20.0];
        let model = GraphRegularizedBooster::new(baseline.clone(), chain_graph(), 0.0, 10).unwrap();

        assert_eq!(model.predict().unwrap(), baseline);
    }

    #[test]
    fn graph_structure_improves_cold_node_prediction_on_correlated_chain() {
        let baseline = vec![10.0, 12.0, 0.0];
        let target = [10.0, 12.0, 12.0];
        let model = GraphRegularizedBooster::new(baseline.clone(), chain_graph(), 2.0, 8).unwrap();
        let smoothed = model.predict().unwrap();
        let base_error = (baseline[2] - target[2]).abs();
        let smooth_error = (smoothed[2] - target[2]).abs();

        assert!(smooth_error < base_error);
    }

    #[test]
    fn graph_regularized_objective_is_ordinary_loss_plus_lambda_laplacian_penalty() {
        let model =
            GraphRegularizedBooster::new(vec![1.0, 3.0, 10.0], chain_graph(), 0.25, 1).unwrap();
        let targets = [2.0, 4.0, 8.0];
        let predictions = [1.0, 3.0, 10.0];

        let objective = model.objective_value(&targets, &predictions).unwrap();

        assert!((objective - 15.25).abs() < 1.0e-12);
    }

    #[test]
    fn graph_smoother_reduces_residual_sequence_roughness() {
        let laplacian = GraphLaplacian::new(chain_graph());
        let residuals = [0.0, 10.0, 0.0];
        let before = laplacian.penalty(&residuals).unwrap();
        let smoothed = GraphSmoother {
            lambda: 1.0,
            iterations: 4,
        }
        .smooth_residuals(&residuals, &laplacian)
        .unwrap();
        let after = laplacian.penalty(&smoothed).unwrap();

        assert!(after < before);
        assert!(smoothed.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn available_accelerators_match_cpu_graph_smoothing() {
        let laplacian = GraphLaplacian::new(chain_graph());
        let values = [0.0, 10.0, 2.0];
        let smoother = GraphSmoother {
            lambda: 0.75,
            iterations: 5,
        };
        let expected = smoother.smooth(&values, &laplacian).unwrap();
        for backend in available_backends()
            .into_iter()
            .filter(|name| name != "cpu")
        {
            let actual = smoother
                .smooth_with_backend(&values, &laplacian, Some(&backend))
                .unwrap_or_else(|error| panic!("{backend} graph smoothing failed: {error}"));
            for (actual, expected) in actual.iter().zip(&expected) {
                assert!(
                    (actual - expected).abs() <= 1.0e-4,
                    "{backend} produced {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn graph_smoothing_dispatch_avoids_small_device_launches() {
        for backend in available_backends() {
            assert!(!should_accelerate_graph_smoothing(&backend, 4, 5));
            assert_eq!(
                should_accelerate_graph_smoothing(&backend, 4_096, 4),
                backend != "cpu"
            );
        }
    }

    #[test]
    fn random_weak_graph_avoids_catastrophic_degradation() {
        let baseline = vec![2.0, 4.0, 8.0, 16.0];
        let graph =
            CsrGraph::from_edges(4, &[(0, 3, 0.01), (3, 0, 0.01), (1, 2, 0.01), (2, 1, 0.01)])
                .unwrap();
        let model = GraphRegularizedBooster::new(baseline.clone(), graph, 0.5, 4).unwrap();
        let smoothed = model.predict().unwrap();

        for (before, after) in baseline.iter().zip(smoothed) {
            assert!((before - after).abs() < 0.2);
        }
    }

    #[test]
    fn relation_constraints_affect_only_linked_predictions() {
        let relations = SymbolicRelationSet::new(
            4,
            vec![SymbolicRelation::new(
                SymbolicRelationKind::CorridorSimilarity,
                0,
                1,
                2.0,
            )],
        )
        .unwrap();
        let graph = relations
            .to_graph(&[SymbolicRelationKind::CorridorSimilarity])
            .unwrap();
        let smoothed = GraphSmoother {
            lambda: 1.0,
            iterations: 3,
        }
        .smooth(&[0.0, 10.0, 100.0, 200.0], &GraphLaplacian::new(graph))
        .unwrap();

        assert!(smoothed[0] > 0.0);
        assert!(smoothed[1] < 10.0);
        assert_eq!(smoothed[2], 100.0);
        assert_eq!(smoothed[3], 200.0);
    }

    #[test]
    fn relation_penalties_are_grouped_by_kind() {
        let relations = SymbolicRelationSet::new(
            3,
            vec![
                SymbolicRelation::new(SymbolicRelationKind::OriginAdjacency, 0, 1, 1.0),
                SymbolicRelation::new(SymbolicRelationKind::ReverseEdgeSimilarity, 1, 2, 0.5),
                SymbolicRelation::new(SymbolicRelationKind::NoSmoothGroup, 0, 2, 10.0),
            ],
        )
        .unwrap();
        let penalties = relations.relation_penalties(&[1.0, 3.0, 7.0]).unwrap();

        assert_eq!(penalties.len(), 2);
        assert_eq!(penalties[0].kind, SymbolicRelationKind::OriginAdjacency);
        assert_eq!(penalties[0].penalty, 4.0);
        assert_eq!(
            penalties[1].kind,
            SymbolicRelationKind::ReverseEdgeSimilarity
        );
        assert_eq!(penalties[1].penalty, 8.0);
    }

    #[test]
    fn graph_artifact_round_trips_exactly() {
        let model =
            GraphRegularizedBooster::new(vec![1.0, 2.0, 3.0], chain_graph(), 0.25, 2).unwrap();
        let text = serde_json::to_string(&model).unwrap();
        let restored: GraphRegularizedBooster = serde_json::from_str(&text).unwrap();

        assert_eq!(restored, model);
        assert_eq!(restored.predict().unwrap(), model.predict().unwrap());
    }

    #[test]
    fn large_sparse_graph_trains_successfully() {
        let node_count = 2_000;
        let edges = (0..node_count - 1)
            .flat_map(|idx| [(idx, idx + 1, 1.0), (idx + 1, idx, 1.0)])
            .collect::<Vec<_>>();
        let graph = CsrGraph::from_edges(node_count, &edges).unwrap();
        let values = (0..node_count).map(|idx| idx as f64).collect::<Vec<_>>();
        let smoothed = GraphSmoother {
            lambda: 0.1,
            iterations: 2,
        }
        .smooth(&values, &GraphLaplacian::new(graph))
        .unwrap();

        assert_eq!(smoothed.len(), node_count);
        assert!(smoothed.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn monotone_constraints_gate_candidate_splits_and_penalize_violations() {
        let constraints = MonotoneConstraintSet::new(vec![
            MonotoneConstraint {
                feature: 0,
                direction: 1,
            },
            MonotoneConstraint {
                feature: 1,
                direction: -1,
            },
        ])
        .unwrap();

        assert!(constraints.split_allowed(0, 1.0, 2.0).unwrap());
        assert!(!constraints.split_allowed(0, 2.0, 1.0).unwrap());
        assert!(constraints.split_allowed(1, 2.0, 1.0).unwrap());
        assert!(!constraints.split_allowed(1, 1.0, 2.0).unwrap());
        assert!(constraints.split_allowed(2, 10.0, -10.0).unwrap());

        let penalties = constraints
            .penalties(
                &[vec![0.0, 1.0, 2.0], vec![0.0, 1.0, 2.0]],
                &[3.0, 2.0, 1.0],
            )
            .unwrap();

        assert_eq!(penalties[0].name, "monotone_feature_0");
        assert!(penalties[0].penalty > 0.0);
        assert_eq!(penalties[1].name, "monotone_feature_1");
        assert_eq!(penalties[1].penalty, 0.0);
    }

    #[test]
    fn interaction_constraints_gate_feature_combinations() {
        let constraints = InteractionConstraintSet::new(vec![vec![0, 1], vec![2, 3]]).unwrap();

        assert!(constraints.split_allowed(&[], 0));
        assert!(constraints.split_allowed(&[0], 1));
        assert!(!constraints.split_allowed(&[0], 2));
        assert_eq!(constraints.penalty(&[0, 1]).penalty, 0.0);
        assert_eq!(constraints.penalty(&[0, 2]).penalty, 1.0);
    }

    #[test]
    fn rule_compiler_outputs_platform_stable_features_and_penalties() {
        let relations = SymbolicRelationSet::new(
            3,
            vec![
                SymbolicRelation::new(SymbolicRelationKind::ReverseEdgeSimilarity, 0, 1, 2.0),
                SymbolicRelation::new(SymbolicRelationKind::HierarchyParentChild, 1, 2, 1.0),
                SymbolicRelation::new(SymbolicRelationKind::NoSmoothGroup, 0, 2, 100.0),
            ],
        )
        .unwrap();
        let compiler = RuleCompiler::new(
            relations,
            MonotoneConstraintSet::new(vec![MonotoneConstraint {
                feature: 0,
                direction: 1,
            }])
            .unwrap(),
            InteractionConstraintSet::new(vec![vec![0, 1]]).unwrap(),
        );

        let compiled = compiler
            .compile(&[1.0, 3.0, 2.0], &[vec![0.0, 1.0, 2.0]], &[0, 2])
            .unwrap();
        let feature_names = compiled
            .features
            .iter()
            .map(|feature| feature.name.as_str())
            .collect::<Vec<_>>();
        let penalty_names = compiled
            .penalties
            .iter()
            .map(|penalty| penalty.name.as_str())
            .collect::<Vec<_>>();

        assert!(feature_names
            .windows(2)
            .all(|window| window[0] <= window[1]));
        assert!(penalty_names
            .windows(2)
            .all(|window| window[0] <= window[1]));
        assert_eq!(
            compiled
                .features
                .iter()
                .find(|feature| feature.name == "reverse_edge_count")
                .unwrap()
                .values,
            vec![1.0, 1.0, 0.0]
        );
        assert_eq!(
            compiled
                .features
                .iter()
                .find(|feature| feature.name == "hierarchy_parent_child_balance")
                .unwrap()
                .values,
            vec![0.0, 1.0, -1.0]
        );
        assert_eq!(
            compiled
                .penalties
                .iter()
                .find(|penalty| penalty.name == "interaction_constraints")
                .unwrap()
                .penalty,
            1.0
        );
        assert_eq!(
            serde_json::to_value(&compiled).unwrap(),
            serde_json::json!({
                "features": [
                    {"name": "hierarchy_parent_child_balance", "values": [0.0, 1.0, -1.0]},
                    {"name": "interaction_violation", "values": [1.0, 1.0, 1.0]},
                    {"name": "relation_degree_CorridorSimilarity", "values": [0.0, 0.0, 0.0]},
                    {"name": "relation_degree_DestinationAdjacency", "values": [0.0, 0.0, 0.0]},
                    {"name": "relation_degree_EntitySimilarity", "values": [0.0, 0.0, 0.0]},
                    {"name": "relation_degree_HierarchyParentChild", "values": [0.0, 1.0, 1.0]},
                    {"name": "relation_degree_NeighborZoneSimilarity", "values": [0.0, 0.0, 0.0]},
                    {"name": "relation_degree_OriginAdjacency", "values": [0.0, 0.0, 0.0]},
                    {"name": "relation_degree_ReverseEdgeSimilarity", "values": [2.0, 2.0, 0.0]},
                    {"name": "relation_degree_SegmentSimilarity", "values": [0.0, 0.0, 0.0]},
                    {"name": "relation_degree_SmoothGroup", "values": [0.0, 0.0, 0.0]},
                    {"name": "reverse_edge_count", "values": [1.0, 1.0, 0.0]}
                ],
                "penalties": [
                    {"name": "interaction_constraints", "penalty": 1.0},
                    {"name": "monotone_feature_0", "penalty": 1.0},
                    {"name": "relation_HierarchyParentChild", "penalty": 1.0},
                    {"name": "relation_ReverseEdgeSimilarity", "penalty": 8.0}
                ]
            })
        );
    }
}
