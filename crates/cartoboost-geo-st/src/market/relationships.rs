fn learn_relationships(
    frame: &MarketPanelFrame,
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    top_k: usize,
    floor: f64,
    neural_embeddings: Option<&[Vec<f32>]>,
) -> Result<Vec<Vec<MarketRelationship>>> {
    let mut index = BTreeMap::new();
    for (idx, lane) in frame.lane_ids.iter().enumerate() {
        index.insert(lane.as_str(), idx);
    }
    // A market panel may contain every observed origin-to-destination lane.
    // Comparing each lane with every other lane is quadratic and makes a
    // 40K+ lane city graph unusable.  Build the candidate topology from the
    // endpoint relationships that define this model instead: shared origin,
    // shared destination, reverse lanes, and caller-supplied expert edges.
    // Residual correlation and the learned kernel still rank those candidates
    // using train-only values; they are not used to manufacture a dense graph.
    let mut origins = BTreeMap::<&str, Vec<usize>>::new();
    let mut destinations = BTreeMap::<&str, Vec<usize>>::new();
    let mut endpoint_pairs = BTreeMap::<(&str, &str), Vec<usize>>::new();
    for lane in 0..frame.lane_ids.len() {
        origins
            .entry(frame.origin_ids[lane].as_str())
            .or_default()
            .push(lane);
        destinations
            .entry(frame.destination_ids[lane].as_str())
            .or_default()
            .push(lane);
        endpoint_pairs
            .entry((
                frame.origin_ids[lane].as_str(),
                frame.destination_ids[lane].as_str(),
            ))
            .or_default()
            .push(lane);
    }
    let mut priors = BTreeMap::<(usize, usize), &ExpertRelationshipPrior>::new();
    for prior in &frame.expert_priors {
        priors.insert(
            (
                *index
                    .get(prior.source_lane_id.as_str())
                    .ok_or_else(|| GeoStError::InvalidFrame("unknown expert source".to_string()))?,
                *index
                    .get(prior.target_lane_id.as_str())
                    .ok_or_else(|| GeoStError::InvalidFrame("unknown expert target".to_string()))?,
            ),
            prior,
        );
    }
    let mut output = vec![Vec::new(); frame.lane_ids.len()];
    for source in 0..frame.lane_ids.len() {
        // Build and release one lane's candidate list at a time. Retaining a
        // set for every lane would itself duplicate a full city graph.
        let mut candidate_indices = Vec::new();
        if let Some(rows) = origins.get(frame.origin_ids[source].as_str()) {
            candidate_indices.extend(rows.iter().copied());
        }
        if let Some(rows) = destinations.get(frame.destination_ids[source].as_str()) {
            candidate_indices.extend(rows.iter().copied());
        }
        if let Some(rows) = endpoint_pairs.get(&(
            frame.destination_ids[source].as_str(),
            frame.origin_ids[source].as_str(),
        )) {
            candidate_indices.extend(rows.iter().copied());
        }
        candidate_indices.extend(
            priors
                .keys()
                .filter_map(|&(prior_source, target)| (prior_source == source).then_some(target)),
        );
        candidate_indices.sort_unstable();
        candidate_indices.dedup();
        let mut candidates = Vec::new();
        for target in candidate_indices {
            if source == target {
                continue;
            }
            if let Some(prior) = priors.get(&(source, target)) {
                if !prior.allowed {
                    continue;
                }
            }
            let mut kinds = Vec::new();
            let mut score = 0.0;
            if frame.origin_ids[source] == frame.origin_ids[target] {
                kinds.push(RelationshipKind::SharedOrigin);
                score += 0.35;
            }
            if frame.destination_ids[source] == frame.destination_ids[target] {
                kinds.push(RelationshipKind::SharedDestination);
                score += 0.35;
            }
            if frame.origin_ids[source] == frame.destination_ids[target]
                && frame.destination_ids[source] == frame.origin_ids[target]
            {
                kinds.push(RelationshipKind::ReverseLane);
                score += 0.45;
            }
            let distance = endpoint_distance(frame.coordinates[source], frame.coordinates[target]);
            if distance < 2.0 {
                kinds.push(RelationshipKind::Geographic);
                score += 0.2 / (1.0 + distance);
            }
            let correlation = masked_correlation(residuals, observed, source, target);
            if correlation >= floor {
                kinds.push(RelationshipKind::ResidualCorrelation);
                score += 0.5 * correlation;
            }
            if let Some(embeddings) = neural_embeddings {
                let similarity = cosine_similarity(&embeddings[source], &embeddings[target]);
                if similarity > 0.0 {
                    // The kernel is learned from the candidate graph and static lane state;
                    // residual evidence remains the task-specific selection signal.
                    score += 0.2 * similarity;
                    kinds.push(RelationshipKind::NeuralKernel);
                }
            }
            if let Some(prior) = priors.get(&(source, target)) {
                kinds.push(RelationshipKind::Expert);
                score += prior.weight.max(0.01);
            }
            if !kinds.is_empty() && score > 0.0 {
                candidates.push((target, score, kinds));
            }
        }
        candidates.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| frame.lane_ids[a.0].cmp(&frame.lane_ids[b.0]))
        });
        candidates.truncate(top_k);
        let total: f64 = candidates.iter().map(|row| row.1).sum();
        output[source] = candidates
            .into_iter()
            .map(|(target, score, kinds)| MarketRelationship {
                source_lane_id: frame.lane_ids[source].clone(),
                target_lane_id: frame.lane_ids[target].clone(),
                weight: score / total.max(1e-12),
                periodic_weights: periodic_edge_weights(
                    source,
                    target,
                    residuals,
                    observed,
                    &frame.timestamps,
                ),
                kinds,
            })
            .collect();
    }
    Ok(output)
}

fn fit_graph_kernel(
    frame: &MarketPanelFrame,
    primary_means: &[f64],
    secondary_means: &[f64],
    relationships: &[Vec<MarketRelationship>],
    hidden_dim: usize,
    epochs: usize,
) -> Result<Vec<Vec<f32>>> {
    let features = lane_kernel_features(frame, primary_means, secondary_means);
    let lane_index = frame
        .lane_ids
        .iter()
        .enumerate()
        .map(|(index, lane_id)| (lane_id.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let edges = relationships
        .iter()
        .flat_map(|rows| rows.iter())
        .filter_map(|edge| {
            let source = *lane_index.get(edge.source_lane_id.as_str())?;
            let target = *lane_index.get(edge.target_lane_id.as_str())?;
            Some((source, target))
        })
        .collect::<Vec<_>>();
    if edges.is_empty() {
        return Ok(features);
    }
    let graph = HomogeneousGraph::from_directed_edges(frame.lane_ids.len(), &edges)
        .map_err(|err| GeoStError::InvalidFrame(format!("invalid learned market graph: {err}")))?;
    let config = GraphSageConfig {
        hidden_dims: vec![hidden_dim],
        epochs,
        ..GraphSageConfig::default()
    };
    GraphSageEncoder::new(config, features[0].len())
        .map_err(|err| GeoStError::InvalidFrame(format!("invalid market graph kernel: {err}")))?
        .fit(&graph, &features)
        .map(|embedding| embedding.into_inner())
        .map_err(|err| {
            GeoStError::InvalidFrame(format!("market graph kernel fitting failed: {err}"))
        })
}

fn lane_kernel_features(
    frame: &MarketPanelFrame,
    primary_means: &[f64],
    secondary_means: &[f64],
) -> Vec<Vec<f32>> {
    let mut rows = Vec::with_capacity(frame.lane_ids.len());
    for (idx, point) in frame.coordinates.iter().enumerate() {
        rows.push(vec![
            ((point[0] + point[2]) * 0.5) as f32,
            ((point[1] + point[3]) * 0.5) as f32,
            (point[2] - point[0]) as f32,
            (point[3] - point[1]) as f32,
            primary_means[idx] as f32,
            secondary_means[idx] as f32,
        ]);
    }
    standardize_kernel_features(&mut rows);
    rows
}

fn standardize_kernel_features(rows: &mut [Vec<f32>]) {
    for col in 0..rows[0].len() {
        let mean = rows.iter().map(|row| row[col] as f64).sum::<f64>() / rows.len() as f64;
        let scale = (rows
            .iter()
            .map(|row| (row[col] as f64 - mean).powi(2))
            .sum::<f64>()
            / rows.len() as f64)
            .sqrt()
            .max(1e-6);
        for row in rows.iter_mut() {
            row[col] = ((row[col] as f64 - mean) / scale) as f32;
        }
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
    let (mut dot, mut left_norm, mut right_norm) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (&a, &b) in left.iter().zip(right) {
        let a = a as f64;
        let b = b as f64;
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm <= 1e-12 || right_norm <= 1e-12 {
        0.0
    } else {
        dot / (left_norm * right_norm).sqrt()
    }
}

