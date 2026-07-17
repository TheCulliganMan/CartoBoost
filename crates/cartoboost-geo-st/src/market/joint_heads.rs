/// Fit point and distributional heads together over frozen GraphSAGE lane
/// embeddings. Freezing the encoder is intentional: relationships are
/// learned from the complete train-only panel first, then these heads consume
/// only one-step-ahead states available at each historical cutoff. This is a
/// compact adapter design rather than a post-hoc residual quantile adjustment.
#[allow(clippy::too_many_arguments)]
fn fit_joint_heads(
    frame: &MarketPanelFrame,
    primary: &[Vec<f64>],
    observed: &[Vec<bool>],
    secondary: &[Vec<f64>],
    primary_means: &[f64],
    secondary_means: &[f64],
    weekly_primary: &[Vec<f64>],
    weekly_secondary: &[Vec<f64>],
    primary_calendar_weights: &[f64],
    secondary_calendar_weights: &[f64],
    cross_target_couplings: &[f64],
    relationships: &[Vec<MarketRelationship>],
    embeddings: &[Vec<f32>],
    config: &MarketStructureConfig,
) -> Result<JointMarketHeads> {
    let has_samples = (1..primary.len()).any(|time| {
        (0..frame.lane_ids.len()).any(|lane| observed[time][lane] && observed[time - 1][lane])
    });
    if !has_samples {
        return Err(GeoStError::InvalidFrame(
            "market joint heads require at least one observed one-step training target".to_string(),
        ));
    }
    let width =
        embeddings.first().map_or(0, Vec::len) + 6 + frame.calendar.first().map_or(0, Vec::len);
    let mut heads = JointMarketHeads {
        primary_huber: vec![0.0; width],
        secondary_huber: vec![0.0; width],
        primary_quantiles: vec![vec![0.0; width]; config.quantile_levels.len()],
        secondary_quantiles: vec![vec![0.0; width]; config.quantile_levels.len()],
    };
    // Stream examples through every epoch. Materializing a feature vector for
    // every lane and timestamp duplicates a full city panel in memory and
    // prevents all-observed-lane fits. The updates are identical to the
    // former sample-cache loop, while peak memory stays bounded.
    for _ in 0..config.head_epochs {
        for time in 1..primary.len() {
            let calendar = frame.calendar[time].as_slice();
            let mix = frame.mix.as_ref().map(|rows| {
                rows[time]
                    .iter()
                    .map(|features| features[0])
                    .collect::<Vec<_>>()
            });
            for lane in 0..frame.lane_ids.len() {
                if !observed[time][lane] || !observed[time - 1][lane] {
                    continue;
                }
                let features = head_features(
                    lane,
                    &primary[time - 1],
                    &secondary[time - 1],
                    primary_means,
                    secondary_means,
                    relationships,
                    &frame.lane_ids,
                    frame.timestamps[time],
                    calendar,
                    mix.as_deref(),
                    embeddings,
                );
                let timestamp = frame.timestamps[time];
                let primary_peer = peer_value(
                    lane,
                    &primary[time - 1],
                    primary_means,
                    relationships,
                    &frame.lane_ids,
                    timestamp,
                );
                let secondary_peer = peer_value(
                    lane,
                    &secondary[time - 1],
                    secondary_means,
                    relationships,
                    &frame.lane_ids,
                    timestamp,
                );
                let primary_base = primary_means[lane]
                    + weekly_primary[timestamp.rem_euclid(7) as usize][lane]
                    + config.local_strength * (primary[time - 1][lane] - primary_means[lane])
                    + config.graph_strength * primary_peer
                    + calendar_effect(primary_calendar_weights, calendar);
                let secondary_base = secondary_means[lane]
                    + weekly_secondary[timestamp.rem_euclid(7) as usize][lane]
                    + config.local_strength * (secondary[time - 1][lane] - secondary_means[lane])
                    + config.graph_strength * secondary_peer
                    + calendar_effect(secondary_calendar_weights, calendar)
                    + cross_target_couplings[lane]
                        * (primary[time - 1][lane] - primary_means[lane]);
                let primary_target = primary[time][lane] - primary_base;
                let secondary_target = secondary[time][lane] - secondary_base;
                huber_step(&mut heads.primary_huber, &features, primary_target, config);
                huber_step(
                    &mut heads.secondary_huber,
                    &features,
                    secondary_target,
                    config,
                );
                for (idx, level) in config.quantile_levels.iter().enumerate() {
                    pinball_step(
                        &mut heads.primary_quantiles[idx],
                        &features,
                        primary_target,
                        *level,
                        config,
                    );
                    pinball_step(
                        &mut heads.secondary_quantiles[idx],
                        &features,
                        secondary_target,
                        *level,
                        config,
                    );
                }
            }
        }
    }
    Ok(heads)
}

fn huber_step(head: &mut [f64], features: &[f64], target: f64, config: &MarketStructureConfig) {
    let residual = dot(head, features) - target;
    let gradient = residual.clamp(-config.huber_delta, config.huber_delta);
    for (weight, feature) in head.iter_mut().zip(features) {
        *weight -= config.head_learning_rate * (gradient * feature + 1e-5 * weight.signum());
    }
}

fn pinball_step(
    head: &mut [f64],
    features: &[f64],
    target: f64,
    level: f64,
    config: &MarketStructureConfig,
) {
    let prediction = dot(head, features);
    let gradient = if target >= prediction {
        -level
    } else {
        1.0 - level
    };
    for (weight, feature) in head.iter_mut().zip(features) {
        *weight -= config.head_learning_rate * (gradient * feature + 1e-5 * weight.signum());
    }
}

fn dot(weights: &[f64], features: &[f64]) -> f64 {
    weights
        .iter()
        .zip(features)
        .map(|(weight, feature)| weight * feature)
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn head_features(
    lane: usize,
    primary: &[f64],
    secondary: &[f64],
    primary_means: &[f64],
    secondary_means: &[f64],
    relationships: &[Vec<MarketRelationship>],
    lane_ids: &[String],
    timestamp: i64,
    calendar: &[f64],
    mix: Option<&[f64]>,
    embeddings: &[Vec<f32>],
) -> Vec<f64> {
    let mut values = vec![
        1.0,
        primary[lane] - primary_means[lane],
        peer_value(
            lane,
            primary,
            primary_means,
            relationships,
            lane_ids,
            timestamp,
        ),
        secondary[lane] - secondary_means[lane],
        peer_value(
            lane,
            secondary,
            secondary_means,
            relationships,
            lane_ids,
            timestamp,
        ),
        mix.map_or(0.0, |rows| rows[lane]),
    ];
    values.extend_from_slice(calendar);
    values.extend(
        embeddings
            .get(lane)
            .into_iter()
            .flatten()
            .map(|value| *value as f64),
    );
    values
}

#[allow(clippy::too_many_arguments)]
fn forecast_head_features(
    lane: usize,
    primary: &[f64],
    secondary: &[f64],
    primary_means: &[f64],
    secondary_means: &[f64],
    relationships: &[Vec<MarketRelationship>],
    lane_ids: &[String],
    timestamp: i64,
    calendar: &[f64],
    mix: Option<&[f64]>,
    embeddings: &[Vec<f32>],
) -> Vec<f64> {
    head_features(
        lane,
        primary,
        secondary,
        primary_means,
        secondary_means,
        relationships,
        lane_ids,
        timestamp,
        calendar,
        mix,
        embeddings,
    )
}

