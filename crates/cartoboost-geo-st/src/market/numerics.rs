fn squared_distance(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| (left - right).powi(2))
        .sum()
}

fn peer_value(
    lane: usize,
    values: &[f64],
    means: &[f64],
    relationships: &[Vec<MarketRelationship>],
    lane_ids: &[String],
    timestamp: i64,
) -> f64 {
    let mut total = 0.0;
    for edge in &relationships[lane] {
        if let Some(target) = lane_ids.iter().position(|id| id == &edge.target_lane_id) {
            let period = timestamp.rem_euclid(7) as usize;
            let periodic = edge.periodic_weights.get(period).copied().unwrap_or(1.0);
            total += edge.weight * periodic * (values[target] - means[target]);
        }
    }
    total
}

fn periodic_edge_weights(
    source: usize,
    target: usize,
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    timestamps: &[i64],
) -> Vec<f64> {
    let overall = masked_correlation(residuals, observed, source, target).max(0.0);
    (0..7)
        .map(|period| {
            let (left, right): (Vec<_>, Vec<_>) = residuals
                .iter()
                .zip(observed)
                .zip(timestamps)
                .filter_map(|((row, mask), time)| {
                    (mask[source] && mask[target] && time.rem_euclid(7) as usize == period)
                        .then_some((row[source], row[target]))
                })
                .unzip();
            let local = if left.len() >= 3 {
                correlation(&left, &right).max(0.0)
            } else {
                overall
            };
            // Shrink periodic estimates toward the full-history estimate to reject short-lived look-alikes.
            (0.5 + 0.5 * (0.7 * overall + 0.3 * local)).clamp(0.25, 1.0)
        })
        .collect()
}

fn endpoint_distance(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let ex = a[2] - b[2];
    let ey = a[3] - b[3];
    ((dx * dx + dy * dy + ex * ex + ey * ey) / 2.0).sqrt()
}
fn matrix_map(input: &[Vec<f64>], f: impl Fn(f64) -> f64) -> Vec<Vec<f64>> {
    input
        .iter()
        .map(|row| row.iter().copied().map(&f).collect())
        .collect()
}

fn log_primary_with_missing(values: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<bool>>) {
    let observed = values
        .iter()
        .map(|row| row.iter().map(|value| !value.is_nan()).collect())
        .collect::<Vec<Vec<bool>>>();
    let logged = values
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| if value.is_nan() { 0.0 } else { value.ln() })
                .collect()
        })
        .collect();
    (logged, observed)
}

/// Estimate lane means with explicit partial pooling through caller-provided
/// parent groups. Group order is significant: callers should put the most
/// specific stable parent first. A lane with no observations must resolve to a
/// parent with observations; it is never silently filled from a global mean.
fn primary_means_with_hierarchy(
    frame: &MarketPanelFrame,
    values: &[Vec<f64>],
    observed: &[Vec<bool>],
) -> Result<Vec<f64>> {
    let mut groups = BTreeMap::<&str, Vec<f64>>::new();
    for (row, mask) in values.iter().zip(observed) {
        for (lane, value) in row.iter().enumerate() {
            if mask[lane] {
                for group in &frame.hierarchy_groups[lane] {
                    groups.entry(group.as_str()).or_default().push(*value);
                }
            }
        }
    }
    (0..frame.lane_ids.len())
        .map(|lane| {
            let own = values
                .iter()
                .zip(observed)
                .filter_map(|(row, mask)| mask[lane].then_some(row[lane]))
                .collect::<Vec<_>>();
            let parent = frame.hierarchy_groups[lane]
                .iter()
                .find_map(|group| groups.get(group.as_str()))
                .filter(|values| !values.is_empty())
                .map(|values| values.iter().sum::<f64>() / values.len() as f64);
            match (own.is_empty(), parent) {
                (false, Some(parent_mean)) => {
                    // Eight pseudo-observations makes the parent a stabilizing
                    // prior for intermittent lanes without erasing local data.
                    let own_mean = own.iter().sum::<f64>() / own.len() as f64;
                    Ok(
                        (own_mean * own.len() as f64 + parent_mean * 8.0)
                            / (own.len() as f64 + 8.0),
                    )
                }
                (false, None) => Ok(own.iter().sum::<f64>() / own.len() as f64),
                (true, Some(parent_mean)) => Ok(parent_mean),
                (true, None) => Err(GeoStError::InvalidFrame(format!(
                    "lane '{}' has no observed primary values in any supplied hierarchy group",
                    frame.lane_ids[lane]
                ))),
            }
        })
        .collect()
}

fn primary_scales_with_hierarchy(
    frame: &MarketPanelFrame,
    values: &[Vec<f64>],
    observed: &[Vec<bool>],
    means: &[f64],
) -> Result<Vec<f64>> {
    let mut group_values = BTreeMap::<&str, Vec<f64>>::new();
    for (row, mask) in values.iter().zip(observed) {
        for (lane, value) in row.iter().enumerate() {
            if mask[lane] {
                for group in &frame.hierarchy_groups[lane] {
                    group_values.entry(group.as_str()).or_default().push(*value);
                }
            }
        }
    }
    (0..frame.lane_ids.len())
        .map(|lane| {
            let own = values
                .iter()
                .zip(observed)
                .filter_map(|(row, mask)| mask[lane].then_some(row[lane] - means[lane]))
                .collect::<Vec<_>>();
            if !own.is_empty() {
                return Ok(rms_scale(&own));
            }
            let parent_values = frame.hierarchy_groups[lane]
                .iter()
                .find_map(|group| group_values.get(group.as_str()))
                .ok_or_else(|| {
                    GeoStError::InvalidFrame(format!(
                        "lane '{}' has no hierarchy observations for uncertainty",
                        frame.lane_ids[lane]
                    ))
                })?;
            let parent_mean = parent_values.iter().sum::<f64>() / parent_values.len() as f64;
            Ok(rms_scale(
                &parent_values
                    .iter()
                    .map(|value| value - parent_mean)
                    .collect::<Vec<_>>(),
            ))
        })
        .collect()
}

fn rms_scale(values: &[f64]) -> f64 {
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64)
        .sqrt()
        .max(1e-6)
}

fn primary_interval_radii(
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    fallback_scales: &[f64],
) -> Vec<f64> {
    (0..residuals[0].len())
        .map(|lane| {
            let mut absolute = residuals
                .iter()
                .zip(observed)
                .filter_map(|(row, mask)| mask[lane].then_some(row[lane].abs()))
                .collect::<Vec<_>>();
            if absolute.is_empty() {
                return 1.28155 * fallback_scales[lane];
            }
            absolute.sort_by(|left, right| left.total_cmp(right));
            // Finite-sample empirical conformal rank for nominal 80% coverage.
            let rank = ((absolute.len() + 1) * 4).div_ceil(5).saturating_sub(1);
            absolute[rank.min(absolute.len() - 1)].max(1e-6)
        })
        .collect()
}

fn interval_calibration_multiplier(
    frame: &MarketPanelFrame,
    config: &MarketStructureConfig,
) -> Result<f64> {
    let calibration_horizon = frame.horizon.min(frame.timestamps.len() / 4).max(1);
    let minimum_prefix = calibration_horizon + 1;
    if frame.timestamps.len() <= minimum_prefix + calibration_horizon {
        return Ok(1.0);
    }
    let mut calibration_config = config.clone();
    calibration_config.calibrate_intervals = false;
    let mut ratios = Vec::new();
    let mut prefixes = (1..=3)
        .filter_map(|origin| {
            frame
                .timestamps
                .len()
                .checked_sub(origin * calibration_horizon)
        })
        .filter(|prefix| *prefix >= minimum_prefix)
        .collect::<Vec<_>>();
    prefixes.sort_unstable();
    prefixes.dedup();
    for prefix in prefixes {
        let mut calibration_frame = frame.clone();
        calibration_frame.timestamps.truncate(prefix);
        calibration_frame.primary.truncate(prefix);
        calibration_frame.secondary.truncate(prefix);
        calibration_frame.calendar.truncate(prefix);
        if let Some(mix) = &mut calibration_frame.mix {
            mix.truncate(prefix);
        }
        let calibration_timestamps = calibration_frame.timestamps.clone();
        calibration_frame
            .expert_labels
            .retain(|label| calibration_timestamps.contains(&label.timestamp));
        calibration_frame.horizon = calibration_horizon;
        calibration_frame.validate()?;
        let mut model = MarketStructureForecaster::new(calibration_config.clone())?;
        model.fit(&calibration_frame)?;
        let future_calendar = (!frame.calendar.is_empty())
            .then_some(&frame.calendar[prefix..prefix + calibration_horizon]);
        for prediction in model.predict(calibration_horizon, future_calendar)? {
            let lane = frame
                .lane_ids
                .iter()
                .position(|lane_id| lane_id == &prediction.lane_id)
                .ok_or(GeoStError::NotFit)?;
            let step = prediction.horizon - 1;
            let actual = frame.primary[prefix + step][lane];
            if !actual.is_nan() {
                let radius = model.primary_interval_radii[lane].max(1e-6);
                ratios.push((actual.ln() - prediction.primary.ln()).abs() / radius);
            }
        }
    }
    if ratios.len() < 8 {
        return Ok(1.0);
    }
    ratios.sort_by(|left, right| left.total_cmp(right));
    // Use a conservative 90th-percentile rolling-origin multiplier: the
    // underlying radius is 80%, while multi-step graph propagation adds tail
    // risk that is only visible in held-out origins.
    let rank = ((ratios.len() + 1) * 9).div_ceil(10).saturating_sub(1);
    Ok(ratios[rank.min(ratios.len() - 1)].max(1.0))
}

fn centered_masked(values: &[Vec<f64>], means: &[f64], observed: &[Vec<bool>]) -> Vec<Vec<f64>> {
    values
        .iter()
        .zip(observed)
        .map(|(row, mask)| {
            row.iter()
                .enumerate()
                .map(|(lane, value)| if mask[lane] { value - means[lane] } else { 0.0 })
                .collect()
        })
        .collect()
}

fn last_observed_index(observed: &[Vec<bool>], lane: usize) -> Option<usize> {
    observed.iter().rposition(|row| row[lane])
}

fn last_observed_by_lane(values: &[Vec<f64>], observed: &[Vec<bool>], prior: &[f64]) -> Vec<f64> {
    (0..values[0].len())
        .map(|lane| {
            last_observed_index(observed, lane).map_or(prior[lane], |index| values[index][lane])
        })
        .collect()
}

fn last_observed_delta(values: &[Vec<f64>], observed: &[Vec<bool>], lane: usize) -> f64 {
    let indices = observed
        .iter()
        .enumerate()
        .filter_map(|(index, row)| row[lane].then_some(index))
        .collect::<Vec<_>>();
    match indices.as_slice() {
        [.., previous, current] => values[*current][lane] - values[*previous][lane],
        _ => 0.0,
    }
}
fn column_means(values: &[Vec<f64>]) -> Vec<f64> {
    (0..values[0].len())
        .map(|col| values.iter().map(|row| row[col]).sum::<f64>() / values.len() as f64)
        .collect()
}
fn centered(values: &[Vec<f64>], means: &[f64]) -> Vec<Vec<f64>> {
    values
        .iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| value - means[idx])
                .collect()
        })
        .collect()
}
fn column_scales(values: &[Vec<f64>]) -> Vec<f64> {
    (0..values[0].len())
        .map(|col| {
            (values.iter().map(|row| row[col] * row[col]).sum::<f64>() / values.len() as f64)
                .sqrt()
                .max(1e-6)
        })
        .collect()
}

fn calendar_weights(residuals: &[Vec<f64>], calendar: &[Vec<f64>]) -> Vec<f64> {
    let width = calendar.first().map_or(0, Vec::len);
    if width == 0 {
        return Vec::new();
    }
    let target = residuals
        .iter()
        .map(|row| row.iter().sum::<f64>() / row.len() as f64)
        .collect::<Vec<_>>();
    let target_mean = target.iter().sum::<f64>() / target.len() as f64;
    (0..width)
        .map(|feature| {
            let mean = calendar.iter().map(|row| row[feature]).sum::<f64>() / calendar.len() as f64;
            let (mut numerator, mut denominator) = (0.0, 0.0);
            for (row, &value) in calendar.iter().zip(&target) {
                let delta = row[feature] - mean;
                numerator += delta * (value - target_mean);
                denominator += delta * delta;
            }
            if denominator <= 1e-12 {
                0.0
            } else {
                (numerator / denominator).clamp(-0.25, 0.25)
            }
        })
        .collect()
}

fn calendar_weights_masked(
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    calendar: &[Vec<f64>],
) -> Vec<f64> {
    let width = calendar.first().map_or(0, Vec::len);
    if width == 0 {
        return Vec::new();
    }
    let target = residuals
        .iter()
        .zip(observed)
        .map(|(row, mask)| {
            let rows = row
                .iter()
                .enumerate()
                .filter_map(|(lane, value)| mask[lane].then_some(*value))
                .collect::<Vec<_>>();
            if rows.is_empty() {
                None
            } else {
                Some(rows.iter().sum::<f64>() / rows.len() as f64)
            }
        })
        .collect::<Vec<_>>();
    (0..width)
        .map(|feature| {
            let paired = target
                .iter()
                .enumerate()
                .filter_map(|(time, value)| value.map(|target| (calendar[time][feature], target)))
                .collect::<Vec<_>>();
            if paired.len() < 2 {
                return 0.0;
            }
            let x_mean = paired.iter().map(|(x, _)| x).sum::<f64>() / paired.len() as f64;
            let y_mean = paired.iter().map(|(_, y)| y).sum::<f64>() / paired.len() as f64;
            let (numerator, denominator) = paired.iter().fold((0.0, 0.0), |(num, den), (x, y)| {
                let delta = x - x_mean;
                (num + delta * (y - y_mean), den + delta * delta)
            });
            if denominator <= 1e-12 {
                0.0
            } else {
                (numerator / denominator).clamp(-0.25, 0.25)
            }
        })
        .collect()
}

fn calendar_effect(weights: &[f64], calendar: &[f64]) -> f64 {
    weights
        .iter()
        .zip(calendar)
        .map(|(weight, value)| weight * value)
        .sum()
}
fn weekly_effects(residuals: &[Vec<f64>], timestamps: &[i64]) -> Vec<Vec<f64>> {
    let lanes = residuals[0].len();
    (0..7)
        .map(|day| {
            (0..lanes)
                .map(|lane| {
                    let rows: Vec<_> = residuals
                        .iter()
                        .zip(timestamps)
                        .filter_map(|(row, timestamp)| {
                            if timestamp.rem_euclid(7) as usize == day {
                                Some(row[lane])
                            } else {
                                None
                            }
                        })
                        .collect();
                    if rows.is_empty() {
                        0.0
                    } else {
                        rows.iter().sum::<f64>() / rows.len() as f64
                    }
                })
                .collect()
        })
        .collect()
}

fn weekly_effects_masked(
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    timestamps: &[i64],
) -> Vec<Vec<f64>> {
    let lanes = residuals[0].len();
    (0..7)
        .map(|day| {
            (0..lanes)
                .map(|lane| {
                    let rows = residuals
                        .iter()
                        .zip(observed)
                        .zip(timestamps)
                        .filter_map(|((row, mask), timestamp)| {
                            (mask[lane] && timestamp.rem_euclid(7) as usize == day)
                                .then_some(row[lane])
                        })
                        .collect::<Vec<_>>();
                    if rows.is_empty() {
                        0.0
                    } else {
                        rows.iter().sum::<f64>() / rows.len() as f64
                    }
                })
                .collect()
        })
        .collect()
}
fn mix_coefficients(residuals: &[Vec<f64>], mix: Option<&Vec<Vec<Vec<f64>>>>) -> Vec<f64> {
    let lanes = residuals[0].len();
    match mix {
        None => vec![0.0; lanes],
        Some(mix) => (0..lanes)
            .map(|lane| {
                let (mut xy, mut xx) = (0.0, 0.0);
                for (time, row) in mix.iter().enumerate() {
                    let x = row[lane][0];
                    xy += x * residuals[time][lane];
                    xx += x * x;
                }
                if xx > 1e-12 {
                    xy / xx
                } else {
                    0.0
                }
            })
            .collect(),
    }
}

fn cross_target_couplings(primary: &[Vec<f64>], secondary: &[Vec<f64>]) -> Vec<f64> {
    (0..primary[0].len())
        .map(|lane| {
            let left = primary.iter().map(|row| row[lane]).collect::<Vec<_>>();
            let right = secondary.iter().map(|row| row[lane]).collect::<Vec<_>>();
            // Keep cross-target transfer conservative; the shared graph remains
            // the principal mechanism and this term only carries lane-local
            // co-movement between the caller-selected measures.
            (0.02 * correlation(&left, &right)).clamp(-0.02, 0.02)
        })
        .collect()
}
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / b.len() as f64;
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let x = x - ma;
        let y = y - mb;
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    if aa <= 1e-12 || bb <= 1e-12 {
        0.0
    } else {
        ab / (aa * bb).sqrt()
    }
}

fn masked_correlation(
    residuals: &[Vec<f64>],
    observed: &[Vec<bool>],
    source: usize,
    target: usize,
) -> f64 {
    let (mut count, mut sum_left, mut sum_right, mut sum_sq_left, mut sum_sq_right, mut sum_xy) =
        (0usize, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (row, mask) in residuals.iter().zip(observed) {
        if mask[source] && mask[target] {
            let left = row[source];
            let right = row[target];
            count += 1;
            sum_left += left;
            sum_right += right;
            sum_sq_left += left * left;
            sum_sq_right += right * right;
            sum_xy += left * right;
        }
    }
    if count < 3 {
        0.0
    } else {
        let count = count as f64;
        let covariance = sum_xy - sum_left * sum_right / count;
        let left_variance = sum_sq_left - sum_left * sum_left / count;
        let right_variance = sum_sq_right - sum_right * sum_right / count;
        if left_variance <= 1e-12 || right_variance <= 1e-12 {
            0.0
        } else {
            covariance / (left_variance * right_variance).sqrt()
        }
    }
}

