fn validate_kriging_observations(observations: &[KrigingObservation]) -> Result<()> {
    if observations.is_empty() {
        return Err(CartoBoostError::InvalidInput(
            "kriging observations must not be empty".to_string(),
        ));
    }
    for (idx, observation) in observations.iter().enumerate() {
        if !observation.x.is_finite() || !observation.y.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging observation {idx} coordinates must be finite"
            )));
        }
        if !observation.value.is_finite() {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging observation {idx} value must be finite"
            )));
        }
    }
    Ok(())
}

fn select_kriging_neighbors(
    observations: &[KrigingObservation],
    target: (f64, f64),
    config: OrdinaryKrigingConfig,
) -> Result<Vec<(usize, &KrigingObservation)>> {
    let mut candidates = observations
        .iter()
        .enumerate()
        .map(|(idx, observation)| {
            (
                idx,
                observation,
                transformed_distance((observation.x, observation.y), target, config),
            )
        })
        .filter(|(_, _, distance)| {
            config
                .max_distance
                .map(|max_distance| *distance <= max_distance)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.2
            .partial_cmp(&right.2)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    if let Some(max_neighbors) = config.max_neighbors {
        candidates.truncate(max_neighbors);
    }
    if candidates.len() < config.min_neighbors {
        return Err(CartoBoostError::InvalidInput(format!(
            "kriging found {} neighbors, but min_neighbors is {}",
            candidates.len(),
            config.min_neighbors
        )));
    }
    Ok(candidates
        .into_iter()
        .map(|(idx, observation, _)| (idx, observation))
        .collect())
}

fn transformed_distance(left: (f64, f64), right: (f64, f64), config: OrdinaryKrigingConfig) -> f64 {
    let dx = left.0 - right.0;
    let dy = left.1 - right.1;
    if (config.anisotropy_angle_degrees.abs() <= f64::EPSILON)
        && ((config.anisotropy_scaling - 1.0).abs() <= f64::EPSILON)
    {
        return (dx * dx + dy * dy).sqrt();
    }
    let angle = config.anisotropy_angle_degrees.to_radians();
    let cos = angle.cos();
    let sin = angle.sin();
    let rotated_x = cos * dx + sin * dy;
    let rotated_y = -sin * dx + cos * dy;
    (rotated_x * rotated_x + (rotated_y / config.anisotropy_scaling).powi(2)).sqrt()
}

fn drift_term_count(drift: KrigingDrift) -> usize {
    match drift {
        KrigingDrift::Ordinary => 1,
        KrigingDrift::Linear => 3,
    }
}

fn drift_basis(point: (f64, f64), drift: KrigingDrift) -> Vec<f64> {
    match drift {
        KrigingDrift::Ordinary => vec![1.0],
        KrigingDrift::Linear => vec![1.0, point.0, point.1],
    }
}

fn validate_kriging_drift_constraints(
    observations: &[KrigingObservation],
    target: (f64, f64),
    drift: KrigingDrift,
    weights: &[f64],
) -> Result<()> {
    let target_basis = drift_basis(target, drift);
    for (basis_idx, expected) in target_basis.iter().enumerate() {
        let mut actual = 0.0;
        let mut scale = expected.abs().max(1.0);
        for (observation, weight) in observations.iter().zip(weights) {
            let basis = drift_basis((observation.x, observation.y), drift)[basis_idx];
            actual += weight * basis;
            scale += (weight * basis).abs();
        }
        if !actual.is_finite() || (actual - expected).abs() > 1.0e-9 * scale {
            return Err(CartoBoostError::InvalidInput(
                "kriging solution does not satisfy its unbiasedness constraints; system is numerically unstable"
                    .to_string(),
            ));
        }
    }
    Ok(())
}

impl LinearSystemFactorization {
    fn factor(mut matrix: Vec<Vec<f64>>) -> Option<Self> {
        let n = matrix.len();
        if n == 0
            || matrix
                .iter()
                .any(|row| row.len() != n || row.iter().any(|value| !value.is_finite()))
        {
            return None;
        }
        let mut permutation = (0..n).collect::<Vec<_>>();
        let mut row_scales = matrix
            .iter()
            .map(|row| row.iter().map(|value| value.abs()).fold(0.0, f64::max))
            .collect::<Vec<_>>();
        if row_scales.contains(&0.0) {
            return None;
        }
        for pivot in 0..n {
            let best = (pivot..n).max_by(|left, right| {
                let left_score = matrix[*left][pivot].abs() / row_scales[*left];
                let right_score = matrix[*right][pivot].abs() / row_scales[*right];
                left_score.total_cmp(&right_score)
            })?;
            let pivot_value = matrix[best][pivot].abs();
            let tolerance = 128.0 * f64::EPSILON * row_scales[best].max(f64::MIN_POSITIVE);
            if !pivot_value.is_finite() || pivot_value <= tolerance {
                return None;
            }
            matrix.swap(pivot, best);
            permutation.swap(pivot, best);
            row_scales.swap(pivot, best);
            #[allow(clippy::needless_range_loop)]
            for row in (pivot + 1)..n {
                let factor = matrix[row][pivot] / matrix[pivot][pivot];
                if !factor.is_finite() {
                    return None;
                }
                matrix[row][pivot] = factor;
                for column in (pivot + 1)..n {
                    matrix[row][column] -= factor * matrix[pivot][column];
                    if !matrix[row][column].is_finite() {
                        return None;
                    }
                }
            }
        }
        Some(Self {
            lu: matrix,
            permutation,
        })
    }

    fn solve(&self, rhs: &[f64]) -> Option<Vec<f64>> {
        let n = self.lu.len();
        if rhs.len() != n || rhs.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let mut solution = self
            .permutation
            .iter()
            .map(|index| rhs[*index])
            .collect::<Vec<_>>();
        for row in 0..n {
            for column in 0..row {
                solution[row] -= self.lu[row][column] * solution[column];
            }
        }
        for row in (0..n).rev() {
            for column in (row + 1)..n {
                solution[row] -= self.lu[row][column] * solution[column];
            }
            solution[row] /= self.lu[row][row];
        }
        solution
            .iter()
            .all(|value| value.is_finite())
            .then_some(solution)
    }
}

