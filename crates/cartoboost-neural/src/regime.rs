//! Row-independent regime experts and their persisted training references.
use crate::backend::{
    backend_dense_layer_f32, backend_workload_decision, select_backend_for_operations,
    BackendOperation, BackendSelection, BackendWorkloadDecision,
};
use crate::{NeuralError, Result};
use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const EXPERT_NAMES: [&str; 6] = [
    "stable_recurring_pattern",
    "sparse_cold_start",
    "high_volume_hub",
    "volatile_shock",
    "long_distance_pair",
    "low_signal_fallback",
];

fn invalid(message: &str) -> NeuralError {
    NeuralError::InvalidArgument(message.to_owned())
}

fn matrix(values: &[f64], rows: usize, cols: usize) -> Result<()> {
    if cols == 0
        || rows.checked_mul(cols) != Some(values.len())
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("expected a finite fixed-width matrix"));
    }
    Ok(())
}

fn router_width(widths: &[usize]) -> Result<usize> {
    if widths.len() != 6 {
        return Err(invalid("six router input widths are required"));
    }
    widths.iter().try_fold(0_usize, |sum, width| {
        sum.checked_add(*width)
            .ok_or_else(|| invalid("router width overflow"))
    })
}

fn validate_router(values: &[f64], rows: usize, cols: usize) -> Result<()> {
    if rows.checked_mul(cols) != Some(values.len()) || values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "router must be a finite matrix matching input rows",
        ));
    }
    Ok(())
}

/// Safe dense multiplication, retaining the explicit accelerator dispatch policy.
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
    let output = if decision.accelerated {
        let left: Vec<Vec<f32>> = left
            .chunks_exact(inner)
            .map(|row| row.iter().map(|v| *v as f32).collect())
            .collect();
        let right: Vec<f32> = right.iter().map(|v| *v as f32).collect();
        if left
            .iter()
            .flatten()
            .chain(right.iter())
            .any(|v| !v.is_finite())
        {
            return Err(invalid("accelerated dense inputs exceed float32 range"));
        }
        backend_dense_layer_f32(backend, &left, &right, &vec![0.; cols])?
            .into_iter()
            .flatten()
            .map(f64::from)
            .collect()
    } else {
        ArrayView2::from_shape((rows, inner), left)
            .map_err(|_| invalid("invalid matrix shape"))?
            .dot(
                &ArrayView2::from_shape((inner, cols), right)
                    .map_err(|_| invalid("invalid matrix shape"))?,
            )
            .into_raw_vec_and_offset()
            .0
    };
    if output.iter().any(|v| !v.is_finite()) {
        return Err(invalid("dense calculation produced non-finite values"));
    }
    Ok((output, decision))
}

/// Ridge with an unpenalized intercept and a fallible, pivoted solve.
/// Coefficients are row-major `(features + 1, outputs)`, intercept first.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ridge_fit(
    x: &[f64],
    rows: usize,
    cols: usize,
    y: &[f64],
    outputs: usize,
    ridge: f64,
    backend: &BackendSelection,
) -> Result<(Vec<f64>, Vec<BackendWorkloadDecision>)> {
    matrix(x, rows, cols)?;
    matrix(y, rows, outputs)?;
    if rows == 0 || !ridge.is_finite() || ridge < 0. {
        return Err(invalid(
            "ridge must be finite and nonnegative; training rows must be nonempty",
        ));
    }
    let width = cols + 1;
    let mut design = Vec::with_capacity(rows * width);
    for row in x.chunks_exact(cols) {
        design.push(1.);
        design.extend_from_slice(row);
    }
    let mut transpose = vec![0.; design.len()];
    for row in 0..rows {
        for col in 0..width {
            transpose[col * rows + row] = design[row * width + col];
        }
    }
    let (mut gram, gram_dispatch) = product(&transpose, width, rows, &design, width, backend)?;
    let (mut rhs, rhs_dispatch) = product(&transpose, width, rows, y, outputs, backend)?;
    for col in 1..width {
        gram[col * width + col] += ridge;
    }
    for col in 0..width {
        let pivot = (col..width)
            .max_by(|&a, &b| {
                gram[a * width + col]
                    .abs()
                    .total_cmp(&gram[b * width + col].abs())
            })
            .unwrap_or(col);
        if gram[pivot * width + col] == 0. || !gram[pivot * width + col].is_finite() {
            return Err(invalid(
                "singular ridge system; increase ridge or remove dependent features",
            ));
        }
        for j in col..width {
            gram.swap(col * width + j, pivot * width + j);
        }
        for j in 0..outputs {
            rhs.swap(col * outputs + j, pivot * outputs + j);
        }
        for row in col + 1..width {
            let factor = gram[row * width + col] / gram[col * width + col];
            for j in col + 1..width {
                gram[row * width + j] -= factor * gram[col * width + j];
            }
            for j in 0..outputs {
                rhs[row * outputs + j] -= factor * rhs[col * outputs + j];
            }
        }
    }
    for col in (0..width).rev() {
        for out in 0..outputs {
            for j in col + 1..width {
                rhs[col * outputs + out] -= gram[col * width + j] * rhs[j * outputs + out];
            }
            rhs[col * outputs + out] /= gram[col * width + col];
        }
    }
    if rhs.iter().any(|v| !v.is_finite()) {
        return Err(invalid("ridge solver produced non-finite coefficients"));
    }
    Ok((rhs, vec![gram_dispatch, rhs_dispatch]))
}

pub(crate) fn dense_predict(
    x: &[f64],
    rows: usize,
    cols: usize,
    coef: &[f64],
    outputs: usize,
    backend: &BackendSelection,
) -> Result<Vec<f64>> {
    matrix(x, rows, cols)?;
    matrix(coef, cols + 1, outputs)?;
    if rows == 0 {
        return Ok(Vec::new());
    }
    let decision = backend_workload_decision(
        backend,
        BackendOperation::Dense,
        rows.saturating_mul(cols + 1).saturating_mul(outputs),
        16_384,
    );
    if decision.accelerated {
        let mut design = Vec::with_capacity(rows * (cols + 1));
        for row in x.chunks_exact(cols) {
            design.push(1.);
            design.extend_from_slice(row);
        }
        return Ok(product(&design, rows, cols + 1, coef, outputs, backend)?.0);
    }
    let mut values = Vec::with_capacity(rows * outputs);
    for row in x.chunks_exact(cols) {
        for out in 0..outputs {
            let mut value = coef[out];
            for (col, input) in row.iter().enumerate() {
                value += input * coef[(col + 1) * outputs + out];
            }
            if !value.is_finite() {
                return Err(invalid("prediction produced non-finite values"));
            }
            values.push(value);
        }
    }
    Ok(values)
}

fn legacy_version() -> u32 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegimeMoEModel {
    #[serde(default = "legacy_version")]
    artifact_version: u32,
    ridge: f64,
    backend: String,
    #[serde(default)]
    selection: Option<BackendSelection>,
    global_coef: Vec<f64>,
    mixer_coef: Vec<f64>,
    feature_count: usize,
    router_feature_count: usize,
    residual_mean: f64,
    #[serde(default)]
    base_mean: Option<f64>,
    #[serde(default)]
    first_feature_median: Option<f64>,
    #[serde(default)]
    router_widths: Vec<usize>,
    pub metadata: Value,
}

pub struct RegimePrediction {
    pub weights: Vec<f64>,
    pub experts: Vec<f64>,
    pub combined: Vec<f64>,
}

impl RegimeMoEModel {
    #[allow(clippy::too_many_arguments)]
    pub fn fit(
        x: &[f64],
        rows: usize,
        cols: usize,
        y: &[f64],
        router: &[f64],
        widths: &[usize],
        ridge: f64,
        backend: &str,
    ) -> Result<Self> {
        let router_cols = router_width(widths)?;
        validate_router(router, rows, router_cols)?;
        let selection = select_backend_for_operations(Some(backend), &[BackendOperation::Dense])?;
        let (global_coef, mut dispatch) = ridge_fit(x, rows, cols, y, 1, ridge, &selection)?;
        let base = dense_predict(x, rows, cols, &global_coef, 1, &selection)?;
        let base_mean = base.iter().sum::<f64>() / rows as f64;
        let mut first: Vec<_> = x.chunks_exact(cols).map(|row| row[0]).collect();
        first.sort_unstable_by(f64::total_cmp);
        let median = first[rows / 2] * 0.5 + first[(rows - 1) / 2] * 0.5;
        let mut model = Self {
            artifact_version: 2,
            ridge,
            backend: backend.to_owned(),
            selection: Some(selection.clone()),
            global_coef,
            mixer_coef: vec![0.; 7],
            feature_count: cols,
            router_feature_count: router_cols,
            residual_mean: 0.,
            base_mean: Some(base_mean),
            first_feature_median: Some(median),
            router_widths: widths.to_vec(),
            metadata: json!({}),
        };
        let (weights, experts) = model.experts(x, &base, router)?;
        let (mixer, mixer_dispatch) = ridge_fit(&experts, rows, 6, y, 1, ridge, &selection)?;
        dispatch.extend(mixer_dispatch);
        model.mixer_coef = mixer;
        let combined = dense_predict(&experts, rows, 6, &model.mixer_coef, 1, &selection)?;
        model.residual_mean =
            y.iter().zip(&combined).map(|(a, b)| a - b).sum::<f64>() / rows as f64;
        let rmse = (y
            .iter()
            .zip(&combined)
            .map(|(a, b)| (a - b - model.residual_mean).powi(2))
            .sum::<f64>()
            / rows as f64)
            .sqrt();
        let single_rmse = (y
            .iter()
            .zip(&base)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / rows as f64)
            .sqrt();
        let mut counts = [0_usize; 6];
        for row in weights.chunks_exact(6) {
            let mut best = 0;
            for j in 1..6 {
                if row[j] > row[best] {
                    best = j;
                }
            }
            counts[best] += 1;
        }
        let usage: serde_json::Map<String, Value> = EXPERT_NAMES
            .iter()
            .zip(counts)
            .map(|(name, count)| (name.to_string(), json!(count as f64 / rows as f64)))
            .collect();
        let entropy = -weights.iter().map(|v| v * v.max(1e-12).ln()).sum::<f64>() / rows as f64;
        if [
            base_mean,
            median,
            model.residual_mean,
            rmse,
            single_rmse,
            entropy,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            return Err(invalid("training diagnostics overflow"));
        }
        let accelerated = dispatch.iter().any(|d| d.accelerated);
        model.metadata = json!({"model_class":"RegimeMoEForecaster", "architecture":"regime_moe",
            "backend": {"requested": backend, "selected": if accelerated {&selection.selected} else {"cpu"}},
            "accelerated_operations": if accelerated {vec!["dense"]} else {vec![]},
            "expert_names": EXPERT_NAMES, "expert_usage": usage, "router_entropy": entropy,
            "train_metrics": {"rmse":rmse,"single_expert_rmse":single_rmse,"beats_single_expert":rmse<single_rmse},
            "router_inputs":["shared_regime_router","entity_id","pair_id","time_features","recent_volatility","recent_residuals","graph_centrality","historical_sparsity","candidate_value"],
            "shared_representation_consumed":false,"shared_representation":null,
            "outputs":["expert_weights","expert_predictions","combined_prediction","regime_metadata"],
            "save_load_parity_checked":false, "batch_independent":true});
        let restored = Self::from_json(&model.to_json()?)?;
        let probe = rows.min(32);
        let before = model
            .predict(
                &x[..probe * cols],
                probe,
                cols,
                &router[..probe * router_cols],
                widths,
                false,
            )?
            .combined;
        let after = restored
            .predict(
                &x[..probe * cols],
                probe,
                cols,
                &router[..probe * router_cols],
                widths,
                false,
            )?
            .combined;
        if before != after {
            return Err(invalid("model artifact roundtrip changed predictions"));
        }
        model.metadata["save_load_parity_checked"] = json!(true);
        Ok(model)
    }

    fn experts(&self, x: &[f64], base: &[f64], router: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let rows = base.len();
        if rows == 0 {
            return Ok((vec![], vec![]));
        }
        let mean = self
            .base_mean
            .unwrap_or_else(|| base.iter().sum::<f64>() / rows as f64);
        let median = self.first_feature_median.unwrap_or_else(|| {
            let mut values: Vec<_> = x.chunks_exact(self.feature_count).map(|r| r[0]).collect();
            values.sort_unstable_by(f64::total_cmp);
            values[rows / 2] * 0.5 + values[(rows - 1) / 2] * 0.5
        });
        let mut weights = Vec::with_capacity(rows * 6);
        let mut experts = Vec::with_capacity(rows * 6);
        for (row_index, (x, &base)) in x.chunks_exact(self.feature_count).zip(base).enumerate() {
            let r = &router[row_index * self.router_feature_count
                ..(row_index + 1) * self.router_feature_count];
            let mut padded = [0_f64; 6];
            let n = r.len().min(6);
            padded[..n].copy_from_slice(&r[..n]);
            let p = padded;
            let mut w = [
                -p[0].abs(),
                p[4],
                p[3],
                p[1].abs() + p[2].abs(),
                p[5],
                -p.iter().fold(0_f64, |norm, v| norm.hypot(*v)),
            ];
            for v in &mut w {
                *v *= 2.;
            }
            if w.iter().any(|v| !v.is_finite()) {
                return Err(invalid("router logits overflow"));
            }
            let max = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for v in &mut w {
                *v = (*v - max).clamp(-50., 50.).exp();
            }
            let sum: f64 = w.iter().sum();
            for v in &mut w {
                *v /= sum;
            }
            let residual = if self.artifact_version == 1 || self.router_widths[2] == 0 {
                0.
            } else {
                r[self.router_widths[0] + self.router_widths[1]]
            };
            let e = [
                base,
                mean + 0.35 * (base - mean),
                base + (x[0] - median).max(0.),
                base + residual * w[3],
                base + 0.5 * x[self.feature_count - 1],
                mean,
            ];
            if e.iter().any(|v| !v.is_finite()) {
                return Err(invalid("expert prediction overflow"));
            }
            weights.extend_from_slice(&w);
            experts.extend_from_slice(&e);
        }
        Ok((weights, experts))
    }

    pub fn predict(
        &self,
        x: &[f64],
        rows: usize,
        cols: usize,
        router: &[f64],
        widths: &[usize],
        components: bool,
    ) -> Result<RegimePrediction> {
        if cols != self.feature_count
            || router_width(widths)? != self.router_feature_count
            || (self.artifact_version == 2 && widths != self.router_widths)
        {
            return Err(invalid(
                "prediction feature and router input widths must match training",
            ));
        }
        validate_router(router, rows, self.router_feature_count)?;
        let selection = self
            .selection
            .as_ref()
            .ok_or_else(|| invalid("missing backend selection"))?;
        let base = dense_predict(x, rows, cols, &self.global_coef, 1, selection)?;
        let (weights, experts) = self.experts(x, &base, router)?;
        let mut combined = dense_predict(&experts, rows, 6, &self.mixer_coef, 1, selection)?;
        for value in &mut combined {
            *value += self.residual_mean;
        }
        if combined.iter().any(|v| !v.is_finite()) {
            return Err(invalid("combined prediction overflow"));
        }
        Ok(RegimePrediction {
            weights: if components { weights } else { vec![] },
            experts: if components { experts } else { vec![] },
            combined,
        })
    }

    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let mut model: Self = serde_json::from_str(raw)?;
        if !matches!(model.artifact_version, 1 | 2)
            || model.feature_count == 0
            || !model.ridge.is_finite()
            || model.ridge < 0.
            || !model.residual_mean.is_finite()
        {
            return Err(invalid("invalid regime artifact version or dimensions"));
        }
        matrix(
            &model.global_coef,
            model
                .feature_count
                .checked_add(1)
                .ok_or_else(|| invalid("artifact feature count overflow"))?,
            1,
        )?;
        matrix(&model.mixer_coef, 7, 1)?;
        if model.artifact_version == 2
            && (model.router_widths.len() != 6
                || router_width(&model.router_widths)? != model.router_feature_count
                || !model.base_mean.is_some_and(f64::is_finite)
                || !model.first_feature_median.is_some_and(f64::is_finite))
        {
            return Err(invalid(
                "regime artifact is missing finite training references",
            ));
        }
        if !["rmse", "single_expert_rmse"].iter().all(|key| {
            model.metadata["train_metrics"][key]
                .as_f64()
                .is_some_and(f64::is_finite)
        }) || !EXPERT_NAMES.iter().all(|name| {
            model.metadata["expert_usage"][name]
                .as_f64()
                .is_some_and(f64::is_finite)
        }) {
            return Err(invalid(
                "regime artifact requires finite training metrics and expert usage",
            ));
        }
        if model.artifact_version == 1 {
            model.base_mean = None;
            model.first_feature_median = None;
        }
        let requested = model
            .selection
            .as_ref()
            .map_or(model.backend.as_str(), |s| s.selected.as_str());
        let selection = select_backend_for_operations(Some(requested), &[BackendOperation::Dense])?;
        model.selection = Some(selection);
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coefficients_and_diagnostics_have_independent_oracles() {
        let x = [-2., 0., -1., 1., 0., -1., 1., 2., 2., -2., 3., 1.];
        let y: Vec<_> = x
            .chunks_exact(2)
            .map(|r| 4. + 2. * r[0] - 3. * r[1])
            .collect();
        let cpu = select_backend_for_operations(Some("cpu"), &[BackendOperation::Dense]).unwrap();
        let (coef, _) = ridge_fit(&x, 6, 2, &y, 1, 0., &cpu).unwrap();
        for (actual, expected) in coef.iter().zip([4., 2., -3.]) {
            assert!((actual - expected).abs() < 1e-12);
        }
        let router = vec![0.; 36];
        let model = RegimeMoEModel::fit(&x, 6, 2, &y, &router, &[1; 6], 1e-6, "cpu").unwrap();
        let all = model.predict(&x, 6, 2, &router, &[1; 6], true).unwrap();
        for row in 0..6 {
            let single = model
                .predict(
                    &x[row * 2..row * 2 + 2],
                    1,
                    2,
                    &router[row * 6..row * 6 + 6],
                    &[1; 6],
                    true,
                )
                .unwrap();
            assert_eq!(single.combined[0], all.combined[row]);
            assert_eq!(single.experts, all.experts[row * 6..row * 6 + 6]);
        }
        let rmse = (all
            .combined
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / 6.)
            .sqrt();
        assert!((model.metadata["train_metrics"]["rmse"].as_f64().unwrap() - rmse).abs() < 1e-14);
        let restored = RegimeMoEModel::from_json(&model.to_json().unwrap()).unwrap();
        assert_eq!(
            restored
                .predict(&x, 6, 2, &router, &[1; 6], true)
                .unwrap()
                .combined,
            all.combined
        );
    }

    #[test]
    fn malformed_or_singular_inputs_fail() {
        let cpu = select_backend_for_operations(Some("cpu"), &[BackendOperation::Dense]).unwrap();
        assert!(ridge_fit(&[1., 1.], 2, 1, &[2., 3.], 1, 0., &cpu).is_err());
        assert!(RegimeMoEModel::fit(&[], 0, 1, &[], &[], &[1; 6], 1e-6, "cpu").is_err());
        assert!(
            RegimeMoEModel::fit(&[f64::NAN], 1, 1, &[1.], &[0.; 6], &[1; 6], 1e-6, "cpu").is_err()
        );
        assert!(RegimeMoEModel::fit(&[1.], 1, 1, &[1.], &[0.; 6], &[0; 6], 1e-6, "cpu").is_err());
    }
}
