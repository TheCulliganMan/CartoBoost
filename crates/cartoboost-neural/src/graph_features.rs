use crate::error::{NeuralError, Result};
use crate::{backend_csr_diffusion_f32, select_backend_for, BackendOperation, BackendSelection};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const MIN_ACCELERATED_DIRECTIONAL_AGGREGATION_VALUES: usize = 16_384;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DirectionalFeatureBlock {
    pub values: Vec<Vec<f32>>,
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SourceTargetPairExpansion {
    pub edges: Vec<(String, String, String)>,
    pub pair_node_ids: Vec<String>,
}

struct NeighborMeans {
    outgoing: Vec<Vec<f32>>,
    incoming: Vec<Vec<f32>>,
}

pub fn validate_directed_metapath(
    steps: &[String],
    edge_types: &[(String, String, String)],
) -> Result<()> {
    if steps.len() < 3 {
        return Err(NeuralError::InvalidArgument(
            "directed metapath must contain node, relation, node".to_string(),
        ));
    }
    if steps.len().is_multiple_of(2) {
        return Err(NeuralError::InvalidArgument(
            "directed metapath must alternate node/relation/node".to_string(),
        ));
    }
    if steps.iter().any(|step| step.is_empty()) {
        return Err(NeuralError::InvalidArgument(
            "directed metapath steps must be non-empty".to_string(),
        ));
    }

    let edge_set: HashSet<_> = edge_types.iter().cloned().collect();
    for relation_index in (1..steps.len()).step_by(2) {
        let edge = (
            steps[relation_index - 1].clone(),
            steps[relation_index].clone(),
            steps[relation_index + 1].clone(),
        );
        if !edge_set.contains(&edge) {
            return Err(NeuralError::InvalidArgument(format!(
                "metapath edge {edge:?} is not in schema"
            )));
        }
    }

    Ok(())
}

pub fn materialize_source_target_pair_nodes(
    edges: &[(String, String, String)],
    source_to_pair_relation: &str,
    pair_to_target_relation: &str,
    pair_node_prefix: &str,
    include_original_edges: bool,
) -> Result<SourceTargetPairExpansion> {
    if source_to_pair_relation.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "source_to_pair_relation must be non-empty".to_string(),
        ));
    }
    if pair_to_target_relation.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "pair_to_target_relation must be non-empty".to_string(),
        ));
    }
    if pair_node_prefix.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "pair_node_prefix must be non-empty".to_string(),
        ));
    }

    let mut expanded = if include_original_edges {
        edges.to_vec()
    } else {
        Vec::new()
    };
    let mut seen_pairs = HashSet::new();
    let mut pair_node_ids = Vec::new();

    for (source, target, _relation) in edges {
        let pair_node = format!("{pair_node_prefix}:{source}:{target}");
        if seen_pairs.insert(pair_node.clone()) {
            pair_node_ids.push(pair_node.clone());
        }
        expanded.push((
            source.clone(),
            pair_node.clone(),
            source_to_pair_relation.to_string(),
        ));
        expanded.push((
            pair_node,
            target.clone(),
            pair_to_target_relation.to_string(),
        ));
    }

    Ok(SourceTargetPairExpansion {
        edges: expanded,
        pair_node_ids,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn compute_directional_features(
    node_count: usize,
    edges: &[(usize, usize)],
    embeddings: &[Vec<f32>],
    edge_weights: Option<&[f32]>,
    edge_timestamps: Option<&[f32]>,
    feature_prefix: &str,
    requested_features: &[String],
) -> Result<DirectionalFeatureBlock> {
    compute_directional_features_with_backend(
        node_count,
        edges,
        embeddings,
        edge_weights,
        edge_timestamps,
        feature_prefix,
        requested_features,
        Some("cpu"),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn compute_directional_features_with_backend(
    node_count: usize,
    edges: &[(usize, usize)],
    embeddings: &[Vec<f32>],
    edge_weights: Option<&[f32]>,
    edge_timestamps: Option<&[f32]>,
    feature_prefix: &str,
    requested_features: &[String],
    backend: Option<&str>,
) -> Result<DirectionalFeatureBlock> {
    if feature_prefix.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "directional feature prefix must be non-empty".to_string(),
        ));
    }
    if embeddings.len() != node_count {
        return Err(NeuralError::InvalidRowCount {
            expected: node_count,
            actual: embeddings.len(),
        });
    }
    let embedding_width = embeddings.first().map_or(0, Vec::len);
    if embedding_width == 0
        || embeddings
            .iter()
            .any(|row| row.len() != embedding_width || row.iter().any(|value| !value.is_finite()))
    {
        return Err(NeuralError::InvalidArgument(
            "embedding rows must be finite with a consistent positive width".to_string(),
        ));
    }
    let weights = edge_values(edge_weights, edges.len(), 1.0, "edge_weights")?;
    let timestamps = edge_values(edge_timestamps, edges.len(), 0.0, "edge_timestamps")?;

    let mut out_degree = vec![0.0_f32; node_count];
    let mut in_degree = vec![0.0_f32; node_count];
    let mut out_time_weighted = vec![0.0_f32; node_count];
    let mut in_time_weighted = vec![0.0_f32; node_count];

    for (index, &(source, target)) in edges.iter().enumerate() {
        if source >= node_count || target >= node_count {
            return Err(NeuralError::InvalidArgument(
                "edge endpoint must be in [0, node_count)".to_string(),
            ));
        }
        let weight = weights[index];
        let timestamp = timestamps[index];
        out_degree[source] += weight;
        in_degree[target] += weight;
        out_time_weighted[source] += weight * timestamp;
        in_time_weighted[target] += weight * timestamp;
    }
    let requested_backend = select_backend_for(backend, BackendOperation::CsrDiffusion)?;
    let aggregation_work = edges.len().saturating_mul(embedding_width);
    let selected_backend = if requested_backend.selected != "cpu"
        && aggregation_work < MIN_ACCELERATED_DIRECTIONAL_AGGREGATION_VALUES
    {
        select_backend_for(Some("cpu"), BackendOperation::CsrDiffusion)?
    } else {
        requested_backend
    };
    let neighbor_means = neighbor_means_with_backend(
        node_count,
        edges,
        embeddings,
        embedding_width,
        &selected_backend,
    )?;

    let names = directional_feature_names(feature_prefix);
    let selected = selected_indices(&names, requested_features)?;
    let feature_names = selected
        .iter()
        .map(|&index| names[index].clone())
        .collect::<Vec<_>>();
    let values = (0..node_count)
        .into_par_iter()
        .map(|node| {
            let total = out_degree[node] + in_degree[node];
            let source_affinity = safe_divide(out_degree[node], total);
            let target_affinity = safe_divide(in_degree[node], total);
            let flow_asymmetry = safe_divide((out_degree[node] - in_degree[node]).abs(), total);
            let flow_imbalance = if total == 0.0 {
                0.0
            } else {
                (out_degree[node] - in_degree[node]) / total
            };
            let source_target_embedding =
                cosine_similarity(&embeddings[node], &neighbor_means.outgoing[node]);
            let target_source_embedding =
                cosine_similarity(&embeddings[node], &neighbor_means.incoming[node]);
            let forward_reverse_similarity_delta =
                source_target_embedding - target_source_embedding;
            let directed_temporal_drift = if out_degree[node] == 0.0 || in_degree[node] == 0.0 {
                0.0
            } else {
                safe_divide(out_time_weighted[node], out_degree[node])
                    - safe_divide(in_time_weighted[node], in_degree[node])
            };
            let row = [
                source_target_embedding,
                target_source_embedding,
                forward_reverse_similarity_delta,
                out_degree[node],
                in_degree[node],
                flow_imbalance,
                directed_temporal_drift,
                source_affinity,
                target_affinity,
                out_degree[node],
                in_degree[node],
                flow_asymmetry,
                out_degree[node],
                in_degree[node],
                source_target_embedding,
                target_source_embedding,
                out_degree[node],
                in_degree[node],
                out_degree[node],
                in_degree[node],
                directed_temporal_drift,
                source_affinity,
                flow_imbalance,
            ];
            Ok(selected.iter().map(|&index| row[index]).collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(DirectionalFeatureBlock {
        values,
        feature_names,
    })
}

fn neighbor_means_with_backend(
    node_count: usize,
    edges: &[(usize, usize)],
    embeddings: &[Vec<f32>],
    embedding_width: usize,
    backend: &BackendSelection,
) -> Result<NeighborMeans> {
    let outgoing = mean_neighbor_csr(node_count, edges, false)?;
    let incoming = mean_neighbor_csr(node_count, edges, true)?;
    let values = embeddings
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<_>>();
    let outgoing_values = backend_csr_diffusion_f32(
        backend,
        &outgoing.0,
        &outgoing.1,
        &outgoing.2,
        embedding_width,
        &values,
    )?;
    let incoming_values = backend_csr_diffusion_f32(
        backend,
        &incoming.0,
        &incoming.1,
        &incoming.2,
        embedding_width,
        &values,
    )?;
    Ok(NeighborMeans {
        outgoing: rows_from_flat(outgoing_values, embedding_width),
        incoming: rows_from_flat(incoming_values, embedding_width),
    })
}

fn mean_neighbor_csr(
    node_count: usize,
    edges: &[(usize, usize)],
    reverse: bool,
) -> Result<(Vec<u32>, Vec<u32>, Vec<f32>)> {
    let mut rows = vec![Vec::new(); node_count];
    for &(source, target) in edges {
        if source >= node_count || target >= node_count {
            return Err(NeuralError::InvalidArgument(
                "edge endpoint must be in [0, node_count)".to_string(),
            ));
        }
        if reverse {
            rows[target].push(source);
        } else {
            rows[source].push(target);
        }
    }
    let mut indptr = Vec::with_capacity(node_count + 1);
    let mut indices = Vec::with_capacity(edges.len());
    let mut weights = Vec::with_capacity(edges.len());
    indptr.push(0);
    for row in rows {
        let weight = if row.is_empty() {
            0.0
        } else {
            1.0 / row.len() as f32
        };
        for index in row {
            indices.push(u32::try_from(index).map_err(|_| {
                NeuralError::InvalidArgument("graph node index exceeds u32".to_string())
            })?);
            weights.push(weight);
        }
        indptr.push(u32::try_from(indices.len()).map_err(|_| {
            NeuralError::InvalidArgument("graph edge count exceeds u32".to_string())
        })?);
    }
    Ok((indptr, indices, weights))
}

fn rows_from_flat(values: Vec<f32>, width: usize) -> Vec<Vec<f32>> {
    values.chunks_exact(width).map(<[f32]>::to_vec).collect()
}

fn directional_feature_names(prefix: &str) -> Vec<String> {
    [
        "source_target_embedding",
        "target_source_embedding",
        "forward_reverse_similarity_delta",
        "source_outbound_strength",
        "target_inbound_strength",
        "flow_imbalance_ratio",
        "directed_temporal_drift",
        "source_target_affinity",
        "target_source_affinity",
        "forward_flow_weight",
        "reverse_flow_weight",
        "flow_asymmetry",
        "source_out_degree_weighted",
        "target_in_degree_weighted",
        "od_forward_similarity",
        "od_reverse_similarity",
        "origin_outbound_strength",
        "destination_inbound_strength",
        "forward_flow_volume_30d",
        "reverse_flow_volume_30d",
        "directional_market_drift",
        "directional_acceptance_rate",
        "directional_price_pressure",
    ]
    .iter()
    .map(|name| format!("{prefix}_{name}"))
    .collect()
}

fn selected_indices(names: &[String], requested_features: &[String]) -> Result<Vec<usize>> {
    if requested_features.is_empty() {
        return Ok((0..names.len()).collect());
    }
    let mut selected = Vec::new();
    for requested in requested_features {
        if let Some(index) = names.iter().position(|name| name == requested) {
            selected.push(index);
        }
    }
    if selected.is_empty() {
        return Err(NeuralError::InvalidArgument(
            "directional_features contains no recognized feature names".to_string(),
        ));
    }
    Ok(selected)
}

fn edge_values(
    values: Option<&[f32]>,
    expected: usize,
    default: f32,
    label: &str,
) -> Result<Vec<f32>> {
    match values {
        Some(values) if values.len() == expected => Ok(values.to_vec()),
        Some(_) => Err(NeuralError::InvalidArgument(format!(
            "{label} length must match edge count"
        ))),
        None => Ok(vec![default; expected]),
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let own_norm = l2_norm(left);
    if own_norm == 0.0 {
        return 0.0;
    }
    let mean_norm = l2_norm(right);
    if mean_norm == 0.0 {
        return 0.0;
    }
    dot(left, right) / (own_norm * mean_norm)
}

fn safe_divide(numerator: f32, denominator: f32) -> f32 {
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn l2_norm(values: &[f32]) -> f32 {
    dot(values, values).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directional_features_use_weights_and_timestamps() {
        let block = compute_directional_features(
            3,
            &[(0, 1), (1, 0), (0, 2)],
            &[vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]],
            Some(&[2.0, 5.0, 3.0]),
            Some(&[10.0, 20.0, 40.0]),
            "graph",
            &[
                "graph_forward_flow_weight".to_string(),
                "graph_reverse_flow_weight".to_string(),
                "graph_directed_temporal_drift".to_string(),
            ],
        )
        .unwrap();

        assert_eq!(block.feature_names.len(), 3);
        assert_eq!(block.values[0][0], 5.0);
        assert_eq!(block.values[0][1], 5.0);
        assert_eq!(block.values[0][2], 8.0);
    }

    #[test]
    fn directional_features_match_cpu_across_available_accelerators() {
        let node_count = 1_024;
        let width = 16;
        let edges = (0..node_count)
            .flat_map(|node| {
                [
                    (node, (node + 1) % node_count),
                    (node, (node + 17) % node_count),
                ]
            })
            .collect::<Vec<_>>();
        let embeddings = (0..node_count)
            .map(|node| {
                (0..width)
                    .map(|feature| ((node * 31 + feature * 7) % 101) as f32 / 101.0)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let requested = vec![
            "graph_source_target_embedding".to_string(),
            "graph_target_source_embedding".to_string(),
        ];
        let expected = compute_directional_features_with_backend(
            node_count,
            &edges,
            &embeddings,
            None,
            None,
            "graph",
            &requested,
            Some("cpu"),
        )
        .unwrap();

        for backend in crate::available_backends() {
            let actual = compute_directional_features_with_backend(
                node_count,
                &edges,
                &embeddings,
                None,
                None,
                "graph",
                &requested,
                Some(&backend),
            )
            .unwrap_or_else(|error| panic!("{backend} directional aggregation failed: {error}"));
            for (actual_row, expected_row) in actual.values.iter().zip(&expected.values) {
                for (actual_value, expected_value) in actual_row.iter().zip(expected_row) {
                    assert!(
                        (actual_value - expected_value).abs() <= 2.0e-4,
                        "{backend}: expected {expected_value}, got {actual_value}"
                    );
                }
            }
        }
    }

    #[test]
    fn metapath_validation_preserves_direction() {
        validate_directed_metapath(
            &[
                "source_h3".to_string(),
                "flows_to".to_string(),
                "target_h3".to_string(),
                "reverse_flows_to".to_string(),
                "source_h3".to_string(),
            ],
            &[
                (
                    "source_h3".to_string(),
                    "flows_to".to_string(),
                    "target_h3".to_string(),
                ),
                (
                    "target_h3".to_string(),
                    "reverse_flows_to".to_string(),
                    "source_h3".to_string(),
                ),
            ],
        )
        .unwrap();
    }

    #[test]
    fn pair_node_materialization_keeps_reverse_direction_distinct() {
        let expansion = materialize_source_target_pair_nodes(
            &[
                (
                    "Chicago".to_string(),
                    "Dallas".to_string(),
                    "flows_to".to_string(),
                ),
                (
                    "Dallas".to_string(),
                    "Chicago".to_string(),
                    "flows_to".to_string(),
                ),
            ],
            "source_to_pair",
            "pair_to_target",
            "od_pair",
            true,
        )
        .unwrap();

        assert!(expansion
            .pair_node_ids
            .contains(&"od_pair:Chicago:Dallas".to_string()));
        assert!(expansion
            .pair_node_ids
            .contains(&"od_pair:Dallas:Chicago".to_string()));
    }
}
