//! Reuse the exact elimination operations for repeated right-hand sides.
use super::{spatial_system_matrix, Result, SpatialEconError, SpatialWeights};
use std::sync::{Arc, Mutex};

const MAX_CACHE_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, Default)]
pub(super) struct SpatialSolveCache(Arc<Mutex<Option<Arc<PreparedSolve>>>>);

#[derive(Debug)]
struct PreparedSolve {
    weights: SpatialWeights,
    rho: f64,
    pivots: Vec<usize>,
    diagonals: Vec<f64>,
    factors: Vec<f64>,
}

impl SpatialSolveCache {
    pub(super) fn solve(
        &self,
        values: Vec<f64>,
        rho: f64,
        weights: &SpatialWeights,
    ) -> Result<Vec<f64>> {
        let bytes = weights
            .n_nodes
            .checked_mul(weights.n_nodes)
            .and_then(|n| n.checked_add(weights.indices.len().saturating_mul(2)))
            .and_then(|n| n.checked_add(weights.n_nodes.saturating_mul(4)))
            .and_then(|n| n.checked_mul(8));
        if bytes.is_none_or(|bytes| bytes > MAX_CACHE_BYTES) {
            return super::solve_spatial_lag_mean(values, rho, weights);
        }
        let prepared = {
            let mut entry = self.0.lock().map_err(|_| {
                SpatialEconError::InvalidInput("spatial solve cache lock was poisoned".into())
            })?;
            if !entry
                .as_ref()
                .is_some_and(|cache| cache.matches(rho, weights))
            {
                *entry = Some(Arc::new(PreparedSolve::new(rho, weights)?));
            }
            Arc::clone(entry.as_ref().ok_or_else(|| {
                SpatialEconError::InvalidInput("missing spatial factorization".into())
            })?)
        };
        prepared.solve(values)
    }
}

impl PreparedSolve {
    fn matches(&self, rho: f64, weights: &SpatialWeights) -> bool {
        self.rho == rho
            && self.weights.n_nodes == weights.n_nodes
            && self.weights.indptr == weights.indptr
            && self.weights.indices == weights.indices
            && self.weights.data == weights.data
    }

    fn new(rho: f64, weights: &SpatialWeights) -> Result<Self> {
        let mut a = spatial_system_matrix(rho, weights)?;
        let n = weights.n_nodes;
        let scale = a
            .iter()
            .flatten()
            .map(|v| v.abs())
            .fold(0_f64, f64::max)
            .max(1.);
        let tolerance = f64::EPSILON * scale * n as f64 * 64.;
        let mut prepared = Self {
            weights: weights.clone(),
            rho,
            pivots: Vec::with_capacity(n),
            diagonals: Vec::with_capacity(n),
            factors: vec![0.; n * n],
        };
        // Preserve the existing Gauss-Jordan pivoting and arithmetic exactly;
        // only the right-hand-side operations are replayed at prediction time.
        for pivot in 0..n {
            let mut best = pivot;
            for row in pivot + 1..n {
                if a[row][pivot].abs() > a[best][pivot].abs() {
                    best = row;
                }
            }
            if a[best][pivot].abs() <= tolerance {
                return Err(SpatialEconError::SingularSystem);
            }
            a.swap(pivot, best);
            let diag = a[pivot][pivot];
            prepared.pivots.push(best);
            prepared.diagonals.push(diag);
            for value in a[pivot].iter_mut().skip(pivot) {
                *value /= diag;
            }
            let pivot_row = a[pivot].clone();
            for (row, values) in a.iter_mut().enumerate() {
                if row == pivot {
                    continue;
                }
                let factor = values[pivot];
                prepared.factors[pivot * n + row] = factor;
                if factor == 0. {
                    continue;
                }
                for (col, pivot_value) in pivot_row.iter().enumerate().skip(pivot) {
                    values[col] -= factor * pivot_value;
                }
            }
        }
        Ok(prepared)
    }

    fn solve(&self, mut values: Vec<f64>) -> Result<Vec<f64>> {
        let n = self.weights.n_nodes;
        if values.len() != n || values.iter().any(|v| !v.is_finite()) {
            return Err(SpatialEconError::InvalidInput(
                "spatial solve requires a finite vector matching weights".into(),
            ));
        }
        for pivot in 0..n {
            values.swap(pivot, self.pivots[pivot]);
            values[pivot] /= self.diagonals[pivot];
            let value = values[pivot];
            for (row, current) in values.iter_mut().enumerate() {
                let factor = self.factors[pivot * n + row];
                if row != pivot && factor != 0. {
                    *current -= factor * value;
                }
            }
        }
        if values.iter().any(|v| !v.is_finite()) {
            return Err(SpatialEconError::SingularSystem);
        }
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial_weights_from_coo;

    #[test]
    fn replay_matches_original_solve_and_invalidates_for_every_system_component() {
        let cache = SpatialSolveCache::default();
        for n in [8, 13] {
            for offset in [1, 2] {
                for weight in [0.25, 0.5] {
                    let weights = spatial_weights_from_coo(
                        n,
                        n,
                        (0..n).collect(),
                        (0..n).map(|i| (i + offset) % n).collect(),
                        vec![weight; n],
                        false,
                    )
                    .unwrap();
                    for rho in [0.1, -0.6, 0.8] {
                        for shift in [0., 1.5] {
                            let values: Vec<_> = (0..n).map(|i| (i as f64).sin() + shift).collect();
                            let expected =
                                super::super::solve_spatial_lag_mean(values.clone(), rho, &weights)
                                    .unwrap();
                            assert_eq!(cache.solve(values, rho, &weights).unwrap(), expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn identical_graph_reuses_entry_and_clones_share_bounded_cache() {
        let weights =
            spatial_weights_from_coo(2, 2, vec![0, 1], vec![1, 0], vec![1., 1.], true).unwrap();
        let cache = SpatialSolveCache::default();
        cache.solve(vec![1., 2.], 0.4, &weights).unwrap();
        let first = Arc::clone(cache.0.lock().unwrap().as_ref().unwrap());
        cache.clone().solve(vec![4., 5.], 0.4, &weights).unwrap();
        assert!(Arc::ptr_eq(
            &first,
            cache.0.lock().unwrap().as_ref().unwrap()
        ));
        cache.solve(vec![4., 5.], 0.2, &weights).unwrap();
        assert!(!Arc::ptr_eq(
            &first,
            cache.0.lock().unwrap().as_ref().unwrap()
        ));
        assert!(cache.solve(vec![f64::NAN, 5.], 0.2, &weights).is_err());
    }
}
