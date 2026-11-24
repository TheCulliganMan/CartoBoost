//! Entity-token temporal learning and leakage-checked evaluation.
use crate::{
    backend_dense_layer_f32, backend_workload_decision, select_backend_for, BackendOperation,
    BackendSelection, BackendWorkloadDecision, NeuralError, Result,
};
use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

fn invalid(message: &str) -> NeuralError {
    NeuralError::InvalidArgument(message.to_owned())
}
fn validate(values: &[f64], rows: usize, cols: usize) -> Result<()> {
    if rows == 0
        || cols == 0
        || rows.checked_mul(cols) != Some(values.len())
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("expected a nonempty finite rectangular matrix"));
    }
    Ok(())
}
fn product(
    left: &[f64],
    rows: usize,
    inner: usize,
    right: &[f64],
    cols: usize,
    backend: &BackendSelection,
) -> Result<(Vec<f64>, BackendWorkloadDecision)> {
    let decision = backend_workload_decision(
        backend,
        BackendOperation::Dense,
        rows.saturating_mul(inner).saturating_mul(cols),
        16_384,
    );
    let values = if decision.accelerated {
        let left: Vec<Vec<f32>> = left
            .chunks_exact(inner)
            .map(|row| row.iter().map(|v| *v as f32).collect())
            .collect();
        let right: Vec<f32> = right.iter().map(|v| *v as f32).collect();
        if left.iter().flatten().chain(&right).any(|v| !v.is_finite()) {
            return Err(invalid("accelerated inputs exceed float32 range"));
        }
        backend_dense_layer_f32(backend, &left, &right, &vec![0.; cols])?
            .into_iter()
            .flatten()
            .map(f64::from)
            .collect()
    } else {
        ArrayView2::from_shape((rows, inner), left)
            .map_err(|_| invalid("invalid left matrix"))?
            .dot(
                &ArrayView2::from_shape((inner, cols), right)
                    .map_err(|_| invalid("invalid right matrix"))?,
            )
            .into_raw_vec_and_offset()
            .0
    };
    Ok((values, decision))
}

/// Minimum-norm least squares for the three-column temporal delta design.
/// One-sided Jacobi SVD avoids squaring the condition number through X'X.
/// The cutoff matches NumPy's `lstsq(rcond=None)` default.
pub fn least_squares_three(design: &[[f64; 3]], targets: &[f64]) -> Result<[f64; 3]> {
    if design.is_empty()
        || design.len() != targets.len()
        || design
            .iter()
            .flatten()
            .chain(targets)
            .any(|v| !v.is_finite())
    {
        return Err(invalid("invalid least-squares data"));
    }
    let scale = design.iter().flatten().fold(0.0_f64, |a, b| a.max(b.abs()));
    if scale == 0. {
        return Ok([0.; 3]);
    }
    let mut a: Vec<[f64; 3]> = design.iter().map(|r| r.map(|v| v / scale)).collect();
    let mut v = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    for _ in 0..64 {
        let mut changed = false;
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            let alpha: f64 = a.iter().map(|r| r[p] * r[p]).sum();
            let beta: f64 = a.iter().map(|r| r[q] * r[q]).sum();
            let gamma: f64 = a.iter().map(|r| r[p] * r[q]).sum();
            if gamma.abs() <= f64::EPSILON * (alpha * beta).sqrt() || gamma == 0. {
                continue;
            }
            let zeta = (beta - alpha) / (2. * gamma);
            let t = (if zeta >= 0. { 1. } else { -1. }) / (zeta.abs() + zeta.hypot(1.));
            let c = 1. / (1. + t * t).sqrt();
            let s = c * t;
            for row in &mut a {
                let ap = row[p];
                let aq = row[q];
                row[p] = c * ap - s * aq;
                row[q] = s * ap + c * aq;
            }
            for row in &mut v {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
            changed = true;
        }
        if !changed {
            break;
        }
    }
    let norms: [f64; 3] =
        std::array::from_fn(|j| a.iter().map(|r| r[j] * r[j]).sum::<f64>().sqrt());
    let tolerance =
        f64::EPSILON * design.len().max(3) as f64 * norms.iter().copied().fold(0., f64::max);
    let mut coefficients = [0.; 3];
    for j in 0..3 {
        if norms[j] <= tolerance {
            continue;
        }
        let projection =
            a.iter().zip(targets).map(|(r, y)| r[j] * y).sum::<f64>() / norms[j] / norms[j] / scale;
        for i in 0..3 {
            coefficients[i] += v[i][j] * projection;
        }
    }
    if coefficients.iter().any(|v| !v.is_finite()) {
        return Err(invalid("least-squares solution is non-finite"));
    }
    Ok(coefficients)
}

#[derive(Serialize, Deserialize)]
pub struct InvertedFit {
    pub last_values: Vec<f64>,
    pub local_trend: Vec<f64>,
    pub cross_entity_delta: Vec<f64>,
    pub delta_coef: Vec<[f64; 3]>,
    pub attention_weights: Vec<Vec<f64>>,
    pub backend: BackendWorkloadDecision,
    pub accelerated: bool,
    pub residual_scales: Vec<f64>,
}
pub fn fit(
    y: &[f64],
    rows: usize,
    entities: usize,
    lookback: usize,
    backend: &str,
) -> Result<InvertedFit> {
    validate(y, rows, entities)?;
    if rows < 3 || lookback == 0 {
        return Err(invalid("history requires three rows and positive lookback"));
    }
    let selection = select_backend_for(Some(backend), BackendOperation::Dense)?;
    let times = lookback.min(rows);
    let recent = &y[(rows - times) * entities..];
    let means: Vec<f64> = (0..entities)
        .map(|e| recent.chunks_exact(entities).map(|r| r[e]).sum::<f64>() / times as f64)
        .collect();
    let centered: Vec<f64> = recent
        .chunks_exact(entities)
        .flat_map(|r| r.iter().zip(&means).map(|(v, m)| v - m))
        .collect();
    let centered_ref = &centered;
    let transpose: Vec<f64> = (0..entities)
        .flat_map(|e| (0..times).map(move |t| centered_ref[t * entities + e]))
        .collect();
    let norms: Vec<f64> = (0..entities)
        .map(|e| {
            transpose[e * times..(e + 1) * times]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                .max(1e-12)
        })
        .collect();
    let (similarity, dispatch) =
        product(&transpose, entities, times, &centered, entities, &selection)?;
    let mut attention = vec![0.; entities * entities];
    for e in 0..entities {
        let logits: Vec<f64> = (0..entities)
            .map(|p| similarity[e * entities + p] / norms[e] / norms[p])
            .collect();
        let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits
            .iter()
            .map(|x| (x - max).clamp(-50., 50.).exp())
            .collect();
        let total: f64 = exps.iter().sum();
        for p in 0..entities {
            attention[e * entities + p] = exps[p] / total;
        }
    }
    let attention_ref = &attention;
    let attention_t: Vec<f64> = (0..entities)
        .flat_map(|e| (0..entities).map(move |p| attention_ref[p * entities + e]))
        .collect();
    let last = recent[(times - 1) * entities..].to_vec();
    let local: Vec<f64> = (0..entities)
        .map(|e| {
            if times >= 2 {
                last[e] - recent[(times - 2) * entities + e]
            } else {
                0.
            }
        })
        .collect();
    let (peers, peer_dispatch) = product(&last, 1, entities, &attention_t, entities, &selection)?;
    let cross = peers.iter().zip(&last).map(|(p, v)| p - v).collect();
    let mut coefficients = vec![[0., 1., 0.]; entities];
    let mut residual_scales = Vec::new();
    if times >= 3 {
        let (peer_levels, _) = product(
            &recent[entities..(times - 1) * entities],
            times - 2,
            entities,
            &attention_t,
            entities,
            &selection,
        )?;
        for e in 0..entities {
            let design: Vec<_> = (1..times - 1)
                .map(|t| {
                    [
                        1.,
                        recent[t * entities + e] - recent[(t - 1) * entities + e],
                        peer_levels[(t - 1) * entities + e] - recent[t * entities + e],
                    ]
                })
                .collect();
            let targets: Vec<_> = (1..times - 1)
                .map(|t| recent[(t + 1) * entities + e] - recent[t * entities + e])
                .collect();
            coefficients[e] = least_squares_three(&design, &targets)?;
            let rss: f64 = design
                .iter()
                .zip(&targets)
                .map(|(row, target)| {
                    let prediction: f64 =
                        row.iter().zip(&coefficients[e]).map(|(x, c)| x * c).sum();
                    (target - prediction).powi(2)
                })
                .sum();
            residual_scales.push((rss / targets.len() as f64).sqrt());
        }
    }
    let accelerated = dispatch.accelerated || peer_dispatch.accelerated;
    Ok(InvertedFit {
        last_values: last,
        local_trend: local,
        cross_entity_delta: cross,
        delta_coef: coefficients,
        attention_weights: attention
            .chunks_exact(entities)
            .map(<[f64]>::to_vec)
            .collect(),
        backend: dispatch,
        accelerated,
        residual_scales,
    })
}

pub fn predict(
    last: &[f64],
    local: &[f64],
    coefficients: &[f64],
    attention: &[f64],
    horizon: usize,
    cross_entity: bool,
    backend: &str,
) -> Result<Vec<Vec<f64>>> {
    let n = last.len();
    validate(last, 1, n)?;
    validate(local, 1, n)?;
    validate(coefficients, n, 3)?;
    validate(attention, n, n)?;
    if horizon == 0 {
        return Err(invalid("horizon must be positive"));
    }
    let selection = select_backend_for(Some(backend), BackendOperation::Dense)?;
    let transpose: Vec<f64> = (0..n)
        .flat_map(|e| (0..n).map(move |p| attention[p * n + e]))
        .collect();
    let mut current = last.to_vec();
    let mut delta = local.to_vec();
    let mut output = Vec::with_capacity(horizon);
    for _ in 0..horizon {
        let (peers, _) = product(&current, 1, n, &transpose, n, &selection)?;
        for e in 0..n {
            delta[e] = coefficients[e * 3]
                + delta[e] * coefficients[e * 3 + 1]
                + if cross_entity {
                    (peers[e] - current[e]) * coefficients[e * 3 + 2]
                } else {
                    0.
                };
            current[e] += delta[e];
        }
        if current.iter().any(|v| !v.is_finite()) {
            return Err(invalid("temporal forecast is non-finite"));
        }
        output.push(current.clone());
    }
    Ok(output)
}

/// Gaussian residual intervals based on fitted one-step residual scales.
/// These quantify residual dispersion; they are not calibrated per horizon.
pub fn temporal_quantiles(
    model: &crate::DeepTemporalEntityArtifact,
    horizon: usize,
) -> Result<Vec<Vec<Vec<f64>>>> {
    let point = crate::temporal_entity_predict(model, horizon)?;
    residual_quantiles(&point, &model.residual_scales)
}

pub fn residual_quantiles(point: &[Vec<f64>], scales: &[f64]) -> Result<Vec<Vec<Vec<f64>>>> {
    if point.is_empty()
        || scales.is_empty()
        || scales.iter().any(|v| !v.is_finite() || *v < 0.)
        || point
            .iter()
            .any(|row| row.len() != scales.len() || row.iter().any(|v| !v.is_finite()))
    {
        return Err(invalid(
            "finite nonnegative fitted residual scales must match entities",
        ));
    }
    let values: Vec<Vec<Vec<f64>>> = [-1.2815515655446004, 0., 1.2815515655446004]
        .iter()
        .map(|z| {
            point
                .iter()
                .map(|r| r.iter().zip(scales).map(|(v, s)| v + z * s).collect())
                .collect()
        })
        .collect();
    if values.iter().flatten().flatten().any(|v| !v.is_finite()) {
        return Err(invalid("residual quantiles exceed finite range"));
    }
    Ok(values)
}

#[derive(Serialize)]
pub struct TemporalFoldResult {
    pub predictions: Vec<Vec<f64>>,
    pub actual: Vec<Vec<f64>>,
    pub rmse: f64,
    pub mae: f64,
}
pub fn temporal_backtest(
    y: &[Vec<f64>],
    folds: &[(Vec<usize>, Vec<usize>)],
    lookback: usize,
    horizon: usize,
    backend: &str,
    inverted: bool,
) -> Result<Vec<TemporalFoldResult>> {
    if folds.is_empty() {
        return Err(invalid("backtest requires at least one valid fold"));
    }
    let mut result = Vec::with_capacity(folds.len());
    for (train, validation) in folds {
        if train.is_empty()
            || validation.is_empty()
            || train.iter().chain(validation).any(|i| *i >= y.len())
            || train
                .windows(2)
                .chain(validation.windows(2))
                .any(|w| w[0].checked_add(1) != Some(w[1]))
            || train.last().and_then(|i| i.checked_add(1)) != validation.first().copied()
        {
            return Err(invalid("backtest requires contiguous ordered training rows followed by held-out validation rows"));
        }
        let history: Vec<_> = train.iter().map(|i| y[*i].clone()).collect();
        let predictions = if inverted {
            let entities = history[0].len();
            if history.iter().any(|row| row.len() != entities) {
                return Err(invalid("panel rows must have equal widths"));
            }
            let flat: Vec<f64> = history.iter().flatten().copied().collect();
            let model = fit(&flat, history.len(), entities, lookback, backend)?;
            predict(
                &model.last_values,
                &model.local_trend,
                &model
                    .delta_coef
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
                &model
                    .attention_weights
                    .iter()
                    .flatten()
                    .copied()
                    .collect::<Vec<_>>(),
                validation.len(),
                true,
                backend,
            )?
        } else {
            let model = crate::temporal_entity_fit_with_backend(
                &history,
                lookback,
                horizon,
                Some(backend),
            )?;
            crate::temporal_entity_predict(&model, validation.len())?
        };
        let actual: Vec<_> = validation.iter().map(|i| y[*i].clone()).collect();
        if predictions
            .iter()
            .zip(&actual)
            .any(|(p, a)| p.len() != a.len() || a.iter().any(|v| !v.is_finite()))
        {
            return Err(invalid(
                "backtest validation targets must be finite and match entity count",
            ));
        }
        let errors: Vec<_> = predictions
            .iter()
            .flatten()
            .zip(actual.iter().flatten())
            .map(|(p, a)| p - a)
            .collect();
        let count = errors.len() as f64;
        result.push(TemporalFoldResult {
            predictions,
            actual,
            rmse: (errors.iter().map(|v| v * v).sum::<f64>() / count).sqrt(),
            mae: errors.iter().map(|v| v.abs()).sum::<f64>() / count,
        });
    }
    Ok(result)
}
