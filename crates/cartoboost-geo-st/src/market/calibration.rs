fn quantile_interval(
    levels: &[f64],
    values: &[f64],
    point: f64,
    fallback_spread: f64,
) -> (f64, f64) {
    let lower = levels
        .iter()
        .position(|level| *level <= 0.1)
        .map(|idx| values[idx]);
    let upper = levels
        .iter()
        .rposition(|level| *level >= 0.9)
        .map(|idx| values[idx]);
    // Quantile heads determine asymmetric tails. The train-only residual
    // radius remains a calibration floor so a narrow fitted head cannot make
    // the advertised interval spuriously overconfident on a new cutoff.
    let lower = lower
        .unwrap_or(point - fallback_spread)
        .min(point - fallback_spread);
    let upper = upper
        .unwrap_or(point + fallback_spread)
        .max(point + fallback_spread);
    (lower, upper)
}

#[allow(clippy::too_many_arguments)]
fn fit_expert_shift_calibration(
    frame: &MarketPanelFrame,
    primary: &[Vec<f64>],
    observed: &[Vec<bool>],
    means: &[f64],
    scales: &[f64],
    weekly: &[Vec<f64>],
    calendar_weights: &[f64],
    mix_coefficients: &[f64],
    relationships: &[Vec<MarketRelationship>],
    config: &MarketStructureConfig,
) -> Option<ExpertShiftCalibration> {
    let mut groups = [Vec::<[f64; 3]>::new(), Vec::new(), Vec::new()];
    for label in &frame.expert_labels {
        let lane = frame.lane_ids.iter().position(|id| id == &label.lane_id)?;
        let time = frame
            .timestamps
            .iter()
            .position(|time| *time == label.timestamp)?;
        if !observed[time][lane] {
            continue;
        }
        let timestamp = frame.timestamps[time];
        let peer = peer_value(
            lane,
            &primary[time],
            means,
            relationships,
            &frame.lane_ids,
            timestamp,
        );
        let market = means[lane]
            + weekly[(timestamp.rem_euclid(7)) as usize][lane]
            + calendar_effect(calendar_weights, &frame.calendar[time])
            + config.graph_strength * peer;
        let local = primary[time][lane] - market;
        let mix = frame
            .mix
            .as_ref()
            .map_or(0.0, |rows| mix_coefficients[lane] * rows[time][lane][0]);
        let scale = scales[lane].max(1e-8);
        let metrics = [peer.abs() / scale, local.abs() / scale, mix.abs() / scale];
        let group = match label.shift {
            MarketShiftKind::Market => 0,
            MarketShiftKind::LocalOrMix => 1,
            MarketShiftKind::NoShift => 2,
        };
        groups[group].push(metrics);
    }
    let centroids = groups.map(|values| {
        (values.len() >= 2).then(|| {
            let mut centroid = [0.0; 3];
            for values in &values {
                for (idx, value) in values.iter().enumerate() {
                    centroid[idx] += value;
                }
            }
            for value in &mut centroid {
                *value /= values.len() as f64;
            }
            centroid
        })
    });
    let calibration = ExpertShiftCalibration {
        market: centroids[0],
        local_or_mix: centroids[1],
        no_shift: centroids[2],
    };
    let trained_classes = [
        calibration.market.is_some(),
        calibration.local_or_mix.is_some(),
        calibration.no_shift.is_some(),
    ]
    .into_iter()
    .filter(|trained| *trained)
    .count();
    (trained_classes >= 2).then_some(calibration)
}

fn calibrated_shift(
    calibration: &ExpertShiftCalibration,
    metrics: [f64; 3],
) -> Option<MarketShiftKind> {
    let mut candidates = Vec::new();
    if let Some(centroid) = calibration.market {
        candidates.push((MarketShiftKind::Market, squared_distance(metrics, centroid)));
    }
    if let Some(centroid) = calibration.local_or_mix {
        candidates.push((
            MarketShiftKind::LocalOrMix,
            squared_distance(metrics, centroid),
        ));
    }
    if let Some(centroid) = calibration.no_shift {
        candidates.push((
            MarketShiftKind::NoShift,
            squared_distance(metrics, centroid),
        ));
    }
    candidates
        .into_iter()
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(shift, _)| shift)
}

