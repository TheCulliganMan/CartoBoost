use crate::forecasting::HierarchySpec;
use crate::{CartoBoostError, Result};
use cartoboost_accelerator::{
    backend_dense_layer_f32, select_backend_for, BackendOperation, BackendSelection,
};

const RECONCILIATION_DENSE_DISPATCH_MIN_OPS: usize = 16_384;

#[derive(Clone, Debug, PartialEq)]
pub enum ReconciliationMethod {
    BottomUp,
    TopDown,
    MiddleOut {
        level: usize,
    },
    Ols,
    Wls {
        variances: Vec<f64>,
    },
    MinTShrink {
        residuals: Vec<Vec<f64>>,
        shrinkage: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Reconciler {
    hierarchy: HierarchySpec,
    method: ReconciliationMethod,
    backend: BackendSelection,
}

pub fn proportional_total_reconciliation(
    base_values: &[f64],
    target_total: f64,
    gamma: f64,
) -> Result<Vec<f64>> {
    if base_values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "proportional reconciliation requires at least one base value".to_string(),
        ));
    }
    if !target_total.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "proportional reconciliation target total must be finite".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&gamma) || !gamma.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "proportional reconciliation gamma must be finite and between 0 and 1".to_string(),
        ));
    }
    if !base_values.iter().all(|value| value.is_finite()) {
        return Err(CartoBoostError::InvalidInput(
            "proportional reconciliation base values must be finite".to_string(),
        ));
    }
    let allocation_values = if target_total >= 0.0 {
        base_values
            .iter()
            .map(|value| value.max(0.0))
            .collect::<Vec<_>>()
    } else {
        base_values.to_vec()
    };
    let base_total = allocation_values.iter().sum::<f64>();
    if base_total.abs() <= 1.0e-12 {
        return Err(CartoBoostError::InvalidInput(
            "proportional reconciliation cannot allocate a target total when the allocation basis sums to zero"
                .to_string(),
        ));
    }
    let scale = target_total / base_total;
    Ok(base_values
        .iter()
        .zip(allocation_values.iter())
        .map(|(base, allocation)| {
            let reconciled = allocation * scale;
            (1.0 - gamma) * base + gamma * reconciled
        })
        .collect())
}

impl Reconciler {
    pub fn new(hierarchy: HierarchySpec, method: ReconciliationMethod) -> Self {
        Self::new_with_backend(hierarchy, method, Some("cpu")).expect("CPU dense backend")
    }

    pub fn new_with_backend(
        hierarchy: HierarchySpec,
        method: ReconciliationMethod,
        backend: Option<&str>,
    ) -> Result<Self> {
        let backend = select_backend_for(backend, BackendOperation::Dense)
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        Ok(Self {
            hierarchy,
            method,
            backend,
        })
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.backend
    }

    pub fn hierarchy(&self) -> &HierarchySpec {
        &self.hierarchy
    }

    pub fn method(&self) -> &ReconciliationMethod {
        &self.method
    }

    pub fn reconcile(&self, base_forecasts: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        validate_panel(base_forecasts, self.hierarchy.node_count())?;
        match &self.method {
            ReconciliationMethod::BottomUp => self.bottom_up(base_forecasts),
            ReconciliationMethod::TopDown => self.top_down(base_forecasts),
            ReconciliationMethod::MiddleOut { level } => self.middle_out(base_forecasts, *level),
            ReconciliationMethod::Ols => {
                self.project(base_forecasts, &vec![1.0; self.hierarchy.node_count()])
            }
            ReconciliationMethod::Wls { variances } => {
                if variances.len() != self.hierarchy.node_count() {
                    return Err(CartoBoostError::InvalidInput(format!(
                        "expected {} reconciliation variances, got {}",
                        self.hierarchy.node_count(),
                        variances.len()
                    )));
                }
                if !variances
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
                {
                    return Err(CartoBoostError::InvalidInput(
                        "reconciliation variances must be finite and positive".to_string(),
                    ));
                }
                self.project(base_forecasts, variances)
            }
            ReconciliationMethod::MinTShrink {
                residuals,
                shrinkage,
            } => {
                let precision =
                    shrink_precision(residuals, *shrinkage, self.hierarchy.node_count())?;
                self.project_with_node_precision(base_forecasts, &precision)
            }
        }
    }

    fn bottom_up(&self, base_forecasts: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let horizon = base_forecasts[0].len();
        let mut bottom_by_horizon = Vec::with_capacity(horizon);
        for (h, _) in base_forecasts[0].iter().enumerate() {
            let bottom = (0..self.hierarchy.node_count())
                .filter_map(|node_idx| {
                    self.hierarchy
                        .bottom_position_for_node(node_idx)
                        .map(|bottom_pos| (bottom_pos, base_forecasts[node_idx][h]))
                })
                .fold(
                    vec![0.0; self.hierarchy.bottom_count()],
                    |mut acc, (idx, value)| {
                        acc[idx] = value;
                        acc
                    },
                );
            bottom_by_horizon.push(bottom);
        }
        self.aggregate_bottom_horizons(&bottom_by_horizon)
    }

    fn top_down(&self, base_forecasts: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let horizon = base_forecasts[0].len();
        let root_idx = self.root_index()?;
        let mut bottom_by_horizon = Vec::with_capacity(horizon);
        for (h, _) in base_forecasts[0].iter().enumerate() {
            let bottom_base = self.bottom_base_for_horizon(base_forecasts, h);
            let proportions = normalized_positive_proportions(&bottom_base)?;
            let top = base_forecasts[root_idx][h];
            let bottom = proportions
                .iter()
                .map(|proportion| top * proportion)
                .collect::<Vec<_>>();
            bottom_by_horizon.push(bottom);
        }
        self.aggregate_bottom_horizons(&bottom_by_horizon)
    }

    fn middle_out(&self, base_forecasts: &[Vec<f64>], level: usize) -> Result<Vec<Vec<f64>>> {
        let middle_nodes = self.hierarchy.level_indices(level);
        if middle_nodes.is_empty() {
            return Err(CartoBoostError::InvalidInput(format!(
                "hierarchy has no nodes at middle-out level {level}"
            )));
        }
        let horizon = base_forecasts[0].len();
        let mut bottom_by_horizon = Vec::with_capacity(horizon);
        for h in 0..horizon {
            let mut bottom = vec![0.0; self.hierarchy.bottom_count()];
            for middle_idx in &middle_nodes {
                let descendants = self.hierarchy.descendants_bottom_positions(*middle_idx);
                if descendants.is_empty() {
                    continue;
                }
                let base = self.bottom_base_for_horizon(base_forecasts, h);
                let local_base = descendants
                    .iter()
                    .map(|bottom_idx| base[*bottom_idx])
                    .collect::<Vec<_>>();
                let proportions = normalized_positive_proportions(&local_base)?;
                for (offset, bottom_idx) in descendants.iter().enumerate() {
                    bottom[*bottom_idx] = base_forecasts[*middle_idx][h] * proportions[offset];
                }
            }
            bottom_by_horizon.push(bottom);
        }
        self.aggregate_bottom_horizons(&bottom_by_horizon)
    }

    fn project(&self, base_forecasts: &[Vec<f64>], variances: &[f64]) -> Result<Vec<Vec<f64>>> {
        let horizon = base_forecasts[0].len();
        let bottom_count = self.hierarchy.bottom_count();
        let mut gram = vec![vec![0.0; bottom_count]; bottom_count];
        for row in self.hierarchy.sparse_rows() {
            let inv_var = 1.0 / variances[row.node_index];
            for (left_idx, left_weight) in &row.bottom_weights {
                for (right_idx, right_weight) in &row.bottom_weights {
                    gram[*left_idx][*right_idx] += left_weight * right_weight * inv_var;
                }
            }
        }

        let mut bottom_by_horizon = Vec::with_capacity(horizon);
        for (h, _) in base_forecasts[0].iter().enumerate() {
            let mut rhs = vec![0.0; bottom_count];
            for row in self.hierarchy.sparse_rows() {
                let inv_var = 1.0 / variances[row.node_index];
                for (bottom_idx, weight) in &row.bottom_weights {
                    rhs[*bottom_idx] += weight * base_forecasts[row.node_index][h] * inv_var;
                }
            }
            let bottom = solve_linear_system(gram.clone(), rhs)?;
            bottom_by_horizon.push(bottom);
        }
        self.aggregate_bottom_horizons(&bottom_by_horizon)
    }

    fn project_with_node_precision(
        &self,
        base_forecasts: &[Vec<f64>],
        precision: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>> {
        let node_count = self.hierarchy.node_count();
        let bottom_count = self.hierarchy.bottom_count();
        if precision.len() != node_count || precision.iter().any(|row| row.len() != node_count) {
            return Err(CartoBoostError::InvalidInput(
                "precision matrix shape must match hierarchy node count".to_string(),
            ));
        }

        let horizon = base_forecasts[0].len();
        let mut gram = vec![vec![0.0; bottom_count]; bottom_count];
        for left_row in self.hierarchy.sparse_rows() {
            for right_row in self.hierarchy.sparse_rows() {
                let node_precision = precision[left_row.node_index][right_row.node_index];
                if node_precision == 0.0 {
                    continue;
                }
                for (left_bottom, left_weight) in &left_row.bottom_weights {
                    for (right_bottom, right_weight) in &right_row.bottom_weights {
                        gram[*left_bottom][*right_bottom] +=
                            left_weight * node_precision * right_weight;
                    }
                }
            }
        }

        let weighted_panel = if self.backend.selected == "cpu"
            || horizon
                .saturating_mul(node_count)
                .saturating_mul(node_count)
                < RECONCILIATION_DENSE_DISPATCH_MIN_OPS
        {
            None
        } else {
            let features = (0..horizon)
                .map(|h| {
                    (0..node_count)
                        .map(|node| base_forecasts[node][h] as f32)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let weights = (0..node_count)
                .flat_map(|input| {
                    (0..node_count).map(move |output| precision[output][input] as f32)
                })
                .collect::<Vec<_>>();
            Some(
                backend_dense_layer_f32(&self.backend, &features, &weights, &vec![0.0; node_count])
                    .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?,
            )
        };
        let mut bottom_by_horizon = Vec::with_capacity(horizon);
        for h in 0..horizon {
            let weighted_nodes = weighted_panel
                .as_ref()
                .map(|panel| panel[h].iter().map(|&value| f64::from(value)).collect())
                .unwrap_or_else(|| {
                    (0..node_count)
                        .map(|row| {
                            (0..node_count)
                                .map(|col| precision[row][col] * base_forecasts[col][h])
                                .sum()
                        })
                        .collect::<Vec<_>>()
                });

            let mut rhs = vec![0.0; bottom_count];
            for row in self.hierarchy.sparse_rows() {
                for (bottom_idx, weight) in &row.bottom_weights {
                    rhs[*bottom_idx] += weight * weighted_nodes[row.node_index];
                }
            }
            let bottom = solve_linear_system(gram.clone(), rhs)?;
            bottom_by_horizon.push(bottom);
        }
        self.aggregate_bottom_horizons(&bottom_by_horizon)
    }

    fn aggregate_bottom_horizons(&self, bottom_by_horizon: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let horizon = bottom_by_horizon.len();
        let nodes = self.hierarchy.node_count();
        if self.backend.selected == "cpu"
            || horizon
                .saturating_mul(self.hierarchy.bottom_count())
                .saturating_mul(nodes)
                < RECONCILIATION_DENSE_DISPATCH_MIN_OPS
        {
            let mut out = vec![vec![0.0; horizon]; nodes];
            for (h, bottom) in bottom_by_horizon.iter().enumerate() {
                for (node, value) in self
                    .hierarchy
                    .aggregate_bottom_values(bottom)?
                    .into_iter()
                    .enumerate()
                {
                    out[node][h] = value;
                }
            }
            return Ok(out);
        }
        let features = bottom_by_horizon
            .iter()
            .map(|row| row.iter().map(|value| *value as f32).collect())
            .collect::<Vec<Vec<_>>>();
        let mut weights = vec![0.0_f32; self.hierarchy.bottom_count() * nodes];
        for row in self.hierarchy.sparse_rows() {
            for (bottom, weight) in &row.bottom_weights {
                weights[*bottom * nodes + row.node_index] = *weight as f32;
            }
        }
        let panel = backend_dense_layer_f32(&self.backend, &features, &weights, &vec![0.0; nodes])
            .map_err(|error| CartoBoostError::InvalidInput(error.to_string()))?;
        let mut out = vec![vec![0.0; horizon]; nodes];
        for (h, row) in panel.into_iter().enumerate() {
            for (node, value) in row.into_iter().enumerate() {
                out[node][h] = f64::from(value);
            }
        }
        Ok(out)
    }

    fn root_index(&self) -> Result<usize> {
        let roots = self
            .hierarchy
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| node.parent.is_none().then_some(idx))
            .collect::<Vec<_>>();
        if roots.len() != 1 {
            return Err(CartoBoostError::InvalidInput(
                "top-down reconciliation requires exactly one hierarchy root".to_string(),
            ));
        }
        Ok(roots[0])
    }

    fn bottom_base_for_horizon(&self, base_forecasts: &[Vec<f64>], horizon: usize) -> Vec<f64> {
        (0..self.hierarchy.node_count())
            .filter_map(|node_idx| {
                self.hierarchy
                    .bottom_position_for_node(node_idx)
                    .map(|bottom_pos| (bottom_pos, base_forecasts[node_idx][horizon]))
            })
            .fold(
                vec![0.0; self.hierarchy.bottom_count()],
                |mut acc, (idx, value)| {
                    acc[idx] = value;
                    acc
                },
            )
    }
}

fn validate_panel(values: &[Vec<f64>], expected_rows: usize) -> Result<()> {
    if values.len() != expected_rows {
        return Err(CartoBoostError::InvalidInput(format!(
            "expected {} forecast rows, got {}",
            expected_rows,
            values.len()
        )));
    }
    let Some(first) = values.first() else {
        return Err(CartoBoostError::InvalidInput(
            "forecasts must contain at least one row".to_string(),
        ));
    };
    if first.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "forecasts must contain at least one horizon".to_string(),
        ));
    }
    for row in values {
        if row.len() != first.len() {
            return Err(CartoBoostError::InvalidInput(
                "all forecast rows must have the same horizon length".to_string(),
            ));
        }
        if !row.iter().all(|value| value.is_finite()) {
            return Err(CartoBoostError::InvalidInput(
                "forecasts must contain only finite values".to_string(),
            ));
        }
    }
    Ok(())
}

fn normalized_positive_proportions(values: &[f64]) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "reconciliation proportions require at least one value".to_string(),
        ));
    }
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(CartoBoostError::InvalidInput(
            "top-down and middle-out allocation bases must be finite and nonnegative".to_string(),
        ));
    }
    let total = values.iter().sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        return Err(CartoBoostError::InvalidInput(
            "top-down and middle-out allocation bases must have a positive finite sum".to_string(),
        ));
    }
    Ok(values.iter().map(|value| value / total).collect())
}

fn shrink_precision(
    residuals: &[Vec<f64>],
    shrinkage: f64,
    expected_rows: usize,
) -> Result<Vec<Vec<f64>>> {
    if !(0.0..=1.0).contains(&shrinkage) || !shrinkage.is_finite() {
        return Err(CartoBoostError::InvalidInput(
            "MinT shrinkage must be finite and between 0 and 1".to_string(),
        ));
    }
    validate_panel(residuals, expected_rows)?;
    let horizon = residuals[0].len();
    if horizon < 2 {
        return Err(CartoBoostError::InvalidInput(
            "MinT residuals require at least two observations".to_string(),
        ));
    }

    let means = residuals
        .iter()
        .map(|row| row.iter().sum::<f64>() / row.len() as f64)
        .collect::<Vec<_>>();
    let mut covariance = vec![vec![0.0; expected_rows]; expected_rows];
    for left in 0..expected_rows {
        for right in left..expected_rows {
            let value = (0..horizon)
                .map(|idx| {
                    (residuals[left][idx] - means[left]) * (residuals[right][idx] - means[right])
                })
                .sum::<f64>()
                / (horizon - 1) as f64;
            covariance[left][right] = value;
            covariance[right][left] = value;
        }
    }

    let mut shrunk = vec![vec![0.0; expected_rows]; expected_rows];
    for row in 0..expected_rows {
        for col in 0..expected_rows {
            let target = if row == col {
                covariance[row][row]
            } else {
                0.0
            };
            shrunk[row][col] = (1.0 - shrinkage) * covariance[row][col] + shrinkage * target;
        }
    }
    invert_matrix(shrunk)
}

fn invert_matrix(matrix: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut inverse = vec![vec![0.0; n]; n];
    for col in 0..n {
        let mut rhs = vec![0.0; n];
        rhs[col] = 1.0;
        let solution = solve_linear_system(matrix.clone(), rhs)?;
        for row in 0..n {
            inverse[row][col] = solution[row];
        }
    }
    Ok(inverse)
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        let mut best_abs = matrix[pivot][pivot].abs();
        for (row_idx, row) in matrix.iter().enumerate().skip(pivot + 1) {
            let candidate = row[pivot].abs();
            if candidate > best_abs {
                best = row_idx;
                best_abs = candidate;
            }
        }
        if best_abs <= 1e-12 {
            return Err(CartoBoostError::InvalidInput(
                "reconciliation normal equations are singular".to_string(),
            ));
        }
        if best != pivot {
            matrix.swap(best, pivot);
            rhs.swap(best, pivot);
        }
        let pivot_value = matrix[pivot][pivot];
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= pivot_value;
        }
        rhs[pivot] /= pivot_value;
        let pivot_tail = matrix[pivot][pivot..].to_vec();

        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor == 0.0 {
                continue;
            }
            for (offset, pivot_value) in pivot_tail.iter().enumerate() {
                matrix[row][pivot + offset] -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Ok(rhs)
}
