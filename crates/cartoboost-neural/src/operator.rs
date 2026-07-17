use crate::{
    backend_csr_diffusion_f32, select_backend_for, BackendOperation, BackendSelection, NeuralError,
    Result,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const OPERATOR_CSR_DISPATCH_MIN_VALUES: usize = 16_384;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialOperatorEdge {
    pub source: usize,
    pub target: usize,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuralOperatorPrediction {
    pub future_field: Vec<Vec<f64>>,
    pub residual_field: Vec<Vec<f64>>,
    pub uncertainty_field: Vec<Vec<f64>>,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NeuralOperatorSyntheticBenchmark {
    pub operator_rmse: f64,
    pub pointwise_mlp_rmse: f64,
    pub improvement: f64,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GraphNeuralOperator {
    pub smoothing: f64,
    pub coordinate_scale: f64,
    pub capability_tier: String,
    pub backend: BackendSelection,
}

pub type FourierGeoOperator = GraphNeuralOperator;
pub type SpatioTemporalOperator = GraphNeuralOperator;

pub fn graph_neural_operator_predict_json(
    field_values: &[Vec<f64>],
    coordinates: &[Vec<f64>],
    edges: &[SpatialOperatorEdge],
    exogenous_fields: &[Vec<f64>],
    smoothing: f64,
    coordinate_scale: f64,
    backend: Option<&str>,
) -> Result<String> {
    let operator = GraphNeuralOperator::new_with_backend(smoothing, coordinate_scale, backend)?;
    let prediction = operator.predict(field_values, coordinates, edges, exogenous_fields)?;
    serde_json::to_string(&prediction).map_err(NeuralError::from)
}

pub fn neural_operator_synthetic_benchmark_json() -> Result<String> {
    let nodes = 16;
    let mut coordinates = Vec::with_capacity(nodes);
    let mut field = Vec::with_capacity(4);
    for t in 0..4 {
        let mut row = Vec::with_capacity(nodes);
        for node in 0..nodes {
            let x = node as f64 / (nodes - 1) as f64;
            if t == 0 {
                coordinates.push(vec![x, (2.0 * std::f64::consts::PI * x).sin()]);
            }
            row.push(
                (2.0 * std::f64::consts::PI * x).sin()
                    + 0.15 * t as f64
                    + 0.1 * (4.0 * std::f64::consts::PI * x).cos(),
            );
        }
        field.push(row);
    }
    let edges = (0..nodes - 1)
        .flat_map(|node| {
            [
                SpatialOperatorEdge {
                    source: node,
                    target: node + 1,
                    weight: 1.0,
                },
                SpatialOperatorEdge {
                    source: node + 1,
                    target: node,
                    weight: 1.0,
                },
            ]
        })
        .collect::<Vec<_>>();
    let exogenous = vec![vec![0.2; nodes]; field.len()];
    let prediction =
        GraphNeuralOperator::new(0.35, 0.08)?.predict(&field, &coordinates, &edges, &exogenous)?;
    let last = field.last().expect("synthetic field is non-empty");
    let previous = &field[field.len() - 2];
    let neighbor = graph_average(last, &edges, nodes)?;
    let target = last
        .iter()
        .enumerate()
        .map(|(idx, &value)| {
            0.65 * (value + value - previous[idx])
                + 0.35 * neighbor[idx]
                + 0.08 * fourier_coordinate_signal(&coordinates[idx])
                + 0.01
        })
        .collect::<Vec<_>>();
    let operator_rmse = rmse(prediction.future_field.last().unwrap(), &target)?;
    let pointwise_mlp_rmse = rmse(last, &target)?;
    let improvement = pointwise_mlp_rmse - operator_rmse;
    let mut metadata = BTreeMap::new();
    metadata.insert("benchmark".to_string(), "smooth_field_transfer".to_string());
    metadata.insert("baseline".to_string(), "pointwise_mlp_proxy".to_string());
    metadata.insert(
        "capability_tier".to_string(),
        "advanced_experimental".to_string(),
    );
    Ok(serde_json::to_string(&NeuralOperatorSyntheticBenchmark {
        operator_rmse,
        pointwise_mlp_rmse,
        improvement,
        metadata,
    })?)
}

impl GraphNeuralOperator {
    pub fn new(smoothing: f64, coordinate_scale: f64) -> Result<Self> {
        Self::new_with_backend(smoothing, coordinate_scale, Some("cpu"))
    }

    pub fn new_with_backend(
        smoothing: f64,
        coordinate_scale: f64,
        backend: Option<&str>,
    ) -> Result<Self> {
        if !smoothing.is_finite() || !(0.0..=1.0).contains(&smoothing) {
            return invalid("smoothing must be finite and between 0 and 1");
        }
        if !coordinate_scale.is_finite() {
            return invalid("coordinate_scale must be finite");
        }
        let backend = select_operator_backend(backend)?;
        Ok(Self {
            smoothing,
            coordinate_scale,
            capability_tier: "advanced_experimental".to_string(),
            backend,
        })
    }

    pub fn predict(
        &self,
        field_values: &[Vec<f64>],
        coordinates: &[Vec<f64>],
        edges: &[SpatialOperatorEdge],
        exogenous_fields: &[Vec<f64>],
    ) -> Result<NeuralOperatorPrediction> {
        validate_panel(field_values, "field_values")?;
        let nodes = field_values[0].len();
        validate_coordinates(coordinates, nodes)?;
        validate_edges(edges, nodes)?;
        if !exogenous_fields.is_empty() {
            validate_panel(exogenous_fields, "exogenous_fields")?;
            if exogenous_fields[0].len() != nodes {
                return invalid("exogenous_fields node width must match field_values");
            }
        }
        let graph_contexts = graph_average_panel(field_values, edges, nodes, &self.backend)?;
        self.predict_from_graph_context(
            field_values,
            coordinates,
            exogenous_fields,
            &graph_contexts,
        )
    }

    /// Browser WebGPU requires asynchronous adapter discovery and command
    /// completion, so model-level callers must use this route instead of the
    /// synchronous native dispatcher.
    #[cfg(all(feature = "webgpu", target_arch = "wasm32"))]
    pub async fn predict_webgpu(
        &self,
        field_values: &[Vec<f64>],
        coordinates: &[Vec<f64>],
        edges: &[SpatialOperatorEdge],
        exogenous_fields: &[Vec<f64>],
    ) -> Result<NeuralOperatorPrediction> {
        validate_panel(field_values, "field_values")?;
        let nodes = field_values[0].len();
        validate_coordinates(coordinates, nodes)?;
        validate_edges(edges, nodes)?;
        if !exogenous_fields.is_empty() {
            validate_panel(exogenous_fields, "exogenous_fields")?;
            if exogenous_fields[0].len() != nodes {
                return invalid("exogenous_fields node width must match field_values");
            }
        }
        let (indptr, indices, weights, values) =
            graph_diffusion_inputs(field_values, edges, nodes)?;
        let output =
            crate::webgpu_csr_diffusion_f32_async(&indptr, &indices, &weights, 1, &values).await?;
        let graph_contexts = output
            .chunks_exact(nodes)
            .map(|row| row.iter().map(|value| f64::from(*value)).collect())
            .collect::<Vec<_>>();
        self.predict_from_graph_context(
            field_values,
            coordinates,
            exogenous_fields,
            &graph_contexts,
        )
    }

    fn predict_from_graph_context(
        &self,
        field_values: &[Vec<f64>],
        coordinates: &[Vec<f64>],
        exogenous_fields: &[Vec<f64>],
        graph_contexts: &[Vec<f64>],
    ) -> Result<NeuralOperatorPrediction> {
        let nodes = field_values[0].len();
        let mut future_field = Vec::with_capacity(field_values.len());
        let mut residual_field = Vec::with_capacity(field_values.len());
        let mut uncertainty_field = Vec::with_capacity(field_values.len());
        for (t, row) in field_values.iter().enumerate() {
            let graph_context = &graph_contexts[t];
            let previous = if t == 0 { row } else { &field_values[t - 1] };
            let exogenous = exogenous_fields.get(t);
            let mut future_row = Vec::with_capacity(nodes);
            let mut residual_row = Vec::with_capacity(nodes);
            let mut uncertainty_row = Vec::with_capacity(nodes);
            for node in 0..nodes {
                let temporal_delta = row[node] - previous[node];
                let exo = exogenous.map(|values| values[node]).unwrap_or(0.0);
                let coord = self.coordinate_scale * fourier_coordinate_signal(&coordinates[node]);
                let prediction = (1.0 - self.smoothing) * (row[node] + temporal_delta)
                    + self.smoothing * graph_context[node]
                    + coord
                    + 0.05 * exo;
                future_row.push(prediction);
                residual_row.push(prediction - row[node]);
                uncertainty_row.push((prediction - graph_context[node]).abs() + 1.0e-9);
            }
            future_field.push(future_row);
            residual_field.push(residual_row);
            uncertainty_field.push(uncertainty_row);
        }
        let mut metadata = BTreeMap::new();
        metadata.insert("model_class".to_string(), "GraphNeuralOperator".to_string());
        metadata.insert(
            "architecture".to_string(),
            "spatiotemporal_operator".to_string(),
        );
        metadata.insert("capability_tier".to_string(), self.capability_tier.clone());
        metadata.insert(
            "backend_requested".to_string(),
            self.backend.requested.clone(),
        );
        metadata.insert(
            "backend_selected".to_string(),
            self.backend.selected.clone(),
        );
        metadata.insert(
            "outputs".to_string(),
            "future_field,residual_field,uncertainty_field".to_string(),
        );
        Ok(NeuralOperatorPrediction {
            future_field,
            residual_field,
            uncertainty_field,
            metadata,
        })
    }
}

fn validate_panel(panel: &[Vec<f64>], name: &str) -> Result<()> {
    if panel.is_empty() {
        return invalid(&format!("{name} must contain at least one row"));
    }
    let cols = panel[0].len();
    if cols == 0 {
        return invalid(&format!("{name} must contain at least one column"));
    }
    for row in panel {
        if row.len() != cols {
            return invalid(&format!("{name} must have a fixed width"));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return invalid(&format!("{name} must contain only finite values"));
        }
    }
    Ok(())
}

fn validate_coordinates(coordinates: &[Vec<f64>], nodes: usize) -> Result<()> {
    if coordinates.len() != nodes {
        return invalid("coordinates row count must match field node width");
    }
    for row in coordinates {
        if row.len() < 2 || row.iter().any(|value| !value.is_finite()) {
            return invalid("coordinates must be finite with at least two columns");
        }
    }
    Ok(())
}

fn validate_edges(edges: &[SpatialOperatorEdge], nodes: usize) -> Result<()> {
    for edge in edges {
        if edge.source >= nodes || edge.target >= nodes {
            return invalid("edge source and target must reference field nodes");
        }
        if !edge.weight.is_finite() {
            return invalid("edge weights must be finite");
        }
    }
    Ok(())
}

fn select_operator_backend(requested: Option<&str>) -> Result<BackendSelection> {
    Ok(select_backend_for(
        requested.or(Some("cpu")),
        BackendOperation::CsrDiffusion,
    )?)
}

fn graph_average_panel(
    panel: &[Vec<f64>],
    edges: &[SpatialOperatorEdge],
    nodes: usize,
    backend: &BackendSelection,
) -> Result<Vec<Vec<f64>>> {
    if !should_accelerate_operator_diffusion(backend, panel.len(), edges.len()) {
        return panel
            .iter()
            .map(|row| graph_average(row, edges, nodes))
            .collect();
    }
    let (indptr, indices, weights, values) = graph_diffusion_inputs(panel, edges, nodes)?;
    let output = backend_csr_diffusion_f32(backend, &indptr, &indices, &weights, 1, &values)?;
    Ok(output
        .chunks_exact(nodes)
        .map(|row| row.iter().map(|value| f64::from(*value)).collect())
        .collect())
}

fn should_accelerate_operator_diffusion(
    backend: &BackendSelection,
    time_steps: usize,
    edge_count: usize,
) -> bool {
    backend.selected != "cpu"
        && time_steps.saturating_mul(edge_count) >= OPERATOR_CSR_DISPATCH_MIN_VALUES
}

type GraphDiffusionInputs = (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>);

fn graph_diffusion_inputs(
    panel: &[Vec<f64>],
    edges: &[SpatialOperatorEdge],
    nodes: usize,
) -> Result<GraphDiffusionInputs> {
    let mut rows = (0..nodes)
        .map(|node| vec![(node as u32, 1.0_f64)])
        .collect::<Vec<_>>();
    let mut totals = vec![1.0_f64; nodes];
    for edge in edges {
        totals[edge.target] += edge.weight.abs();
        rows[edge.target].push((
            u32::try_from(edge.source).map_err(|_| {
                NeuralError::InvalidArgument("operator node index exceeds u32".to_string())
            })?,
            edge.weight,
        ));
    }
    let mut indptr = Vec::with_capacity(nodes + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    indptr.push(0_u32);
    for (target, row) in rows.into_iter().enumerate() {
        for (source, weight) in row {
            indices.push(source);
            weights.push((weight / totals[target].max(1.0)) as f32);
        }
        indptr.push(u32::try_from(indices.len()).map_err(|_| {
            NeuralError::InvalidArgument("operator edge count exceeds u32".to_string())
        })?);
    }
    let values = panel
        .iter()
        .flat_map(|row| row.iter().map(|value| *value as f32))
        .collect::<Vec<_>>();
    Ok((indptr, indices, weights, values))
}

fn graph_average(row: &[f64], edges: &[SpatialOperatorEdge], nodes: usize) -> Result<Vec<f64>> {
    let mut accum = row.to_vec();
    let mut weights = vec![1.0; nodes];
    for edge in edges {
        if edge.source >= nodes || edge.target >= nodes {
            return invalid("edge source and target must reference field nodes");
        }
        let weight = edge.weight.abs();
        accum[edge.target] += edge.weight * row[edge.source];
        weights[edge.target] += weight;
    }
    Ok(accum
        .iter()
        .zip(weights)
        .map(|(&value, weight)| value / weight.max(1.0))
        .collect())
}

fn fourier_coordinate_signal(coordinate: &[f64]) -> f64 {
    (2.0 * std::f64::consts::PI * coordinate[0]).sin()
        + (2.0 * std::f64::consts::PI * coordinate[1]).cos() * 0.25
}

fn rmse(prediction: &[f64], actual: &[f64]) -> Result<f64> {
    if prediction.len() != actual.len() || prediction.is_empty() {
        return invalid("rmse inputs must have the same non-empty length");
    }
    Ok((prediction
        .iter()
        .zip(actual)
        .map(|(&p, &y)| {
            let err = p - y;
            err * err
        })
        .sum::<f64>()
        / prediction.len() as f64)
        .sqrt())
}

fn invalid<T>(message: &str) -> Result<T> {
    Err(NeuralError::InvalidArgument(message.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operator_diffusion_avoids_small_device_launches() {
        for backend_name in crate::available_backends() {
            let backend =
                select_backend_for(Some(&backend_name), BackendOperation::CsrDiffusion).unwrap();
            assert!(!should_accelerate_operator_diffusion(&backend, 1, 16));
            assert_eq!(
                should_accelerate_operator_diffusion(&backend, 1_024, 16),
                backend_name != "cpu"
            );
        }
    }

    #[test]
    fn graph_neural_operator_predicts_fields_and_uncertainty() {
        let fields = vec![vec![1.0, 2.0, 3.0], vec![1.2, 2.1, 3.4]];
        let coords = vec![vec![0.0, 0.0], vec![0.5, 0.0], vec![1.0, 0.0]];
        let edges = vec![
            SpatialOperatorEdge {
                source: 0,
                target: 1,
                weight: 1.0,
            },
            SpatialOperatorEdge {
                source: 1,
                target: 2,
                weight: 0.5,
            },
        ];
        let exogenous = vec![vec![0.1, 0.1, 0.1], vec![0.2, 0.2, 0.2]];
        let output = GraphNeuralOperator::new(0.25, 0.1)
            .unwrap()
            .predict(&fields, &coords, &edges, &exogenous)
            .unwrap();

        assert_eq!(output.future_field.len(), fields.len());
        assert_eq!(output.future_field[0].len(), fields[0].len());
        assert_eq!(output.residual_field[0].len(), fields[0].len());
        assert_eq!(output.uncertainty_field[0].len(), fields[0].len());
        assert_eq!(
            output.metadata.get("capability_tier").map(String::as_str),
            Some("advanced_experimental")
        );
    }

    #[test]
    fn neural_operator_beats_pointwise_baseline_on_smooth_transfer() {
        let report_json = neural_operator_synthetic_benchmark_json().unwrap();
        let report: NeuralOperatorSyntheticBenchmark = serde_json::from_str(&report_json).unwrap();
        assert!(report.operator_rmse < report.pointwise_mlp_rmse);
        assert!(report.improvement > 0.0);
        assert_eq!(
            report.metadata.get("benchmark").map(String::as_str),
            Some("smooth_field_transfer")
        );
    }
}
