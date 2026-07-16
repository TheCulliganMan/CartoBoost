#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpilloverDiagnostics {
    pub adjacent_treated_control_pairs: Vec<(String, String, f64)>,
    pub min_treated_control_distance: Option<f64>,
    pub mean_treated_control_distance: Option<f64>,
    pub treated_weight_exposure: f64,
    pub control_weight_exposure: f64,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SpatialPlaceboTester {
    pub intervention_time: String,
    pub seed: u64,
}

impl SpatialPlaceboTester {
    pub fn placebo_estimates(&self, panel: GeoCausalPanel, n: usize) -> Result<Vec<f64>> {
        let mut estimator = SyntheticDIDEstimator::new(SyntheticDIDConfig {
            intervention_time: self.intervention_time.clone(),
            seed: self.seed,
        });
        estimator.fit(panel)?;
        estimator.placebo_test(n)
    }
}

pub fn spillover_diagnostics(panel: &GeoCausalPanel) -> SpilloverDiagnostics {
    let treated = treated_units(panel);
    spillover_diagnostics_for_units(panel, &treated)
}

pub fn spillover_diagnostics_for_units(
    panel: &GeoCausalPanel,
    treated: &[String],
) -> SpilloverDiagnostics {
    let treated_set: BTreeSet<_> = treated.iter().cloned().collect();
    let controls: BTreeSet<_> = panel
        .unit_ids()
        .into_iter()
        .filter(|unit| !treated_set.contains(unit))
        .collect();
    let adjacent_treated_control_pairs: Vec<_> = panel
        .spatial_weights()
        .iter()
        .filter(|edge| {
            (treated_set.contains(&edge.from_unit) && controls.contains(&edge.to_unit))
                || (treated_set.contains(&edge.to_unit) && controls.contains(&edge.from_unit))
        })
        .map(|edge| (edge.from_unit.clone(), edge.to_unit.clone(), edge.weight))
        .collect();
    let coords = unit_coordinates(panel);
    let mut distances = Vec::new();
    for treated_unit in &treated_set {
        for control_unit in &controls {
            if let (Some((tlat, tlon)), Some((clat, clon))) =
                (coords.get(treated_unit), coords.get(control_unit))
            {
                distances.push(cartoboost_geo_core::haversine_distance_km(
                    *tlat, *tlon, *clat, *clon,
                ));
            }
        }
    }
    let treated_weight_exposure = panel
        .spatial_weights()
        .iter()
        .filter(|edge| treated_set.contains(&edge.from_unit) || treated_set.contains(&edge.to_unit))
        .map(|edge| edge.weight)
        .sum();
    let control_weight_exposure = panel
        .spatial_weights()
        .iter()
        .filter(|edge| controls.contains(&edge.from_unit) && controls.contains(&edge.to_unit))
        .map(|edge| edge.weight)
        .sum();
    let mut warnings = Vec::new();
    if !adjacent_treated_control_pairs.is_empty() {
        warnings.push("treated and control units are adjacent under spatial weights; spillover may bias causal estimates".to_string());
    }
    SpilloverDiagnostics {
        adjacent_treated_control_pairs,
        min_treated_control_distance: distances.iter().copied().reduce(f64::min),
        mean_treated_control_distance: if distances.is_empty() {
            None
        } else {
            Some(mean(distances.iter().copied()))
        },
        treated_weight_exposure,
        control_weight_exposure,
        warnings,
    }
}

fn estimate_for_treated_units(
    panel: &GeoCausalPanel,
    config: &SyntheticDIDConfig,
    override_treated: Option<&[String]>,
    extra_warnings: Vec<String>,
) -> Result<SyntheticDIDEstimate> {
    if config.intervention_time.is_empty() {
        return Err(GeoCausalError::InvalidInput(
            "intervention_time must be provided".to_string(),
        ));
    }
    let treated = override_treated
        .map(|units| units.to_vec())
        .unwrap_or_else(|| treated_units(panel));
    if treated.is_empty() {
        return Err(GeoCausalError::InvalidInput(
            "at least one treated unit is required".to_string(),
        ));
    }
    let treated_set: BTreeSet<_> = treated.iter().cloned().collect();
    let controls: Vec<_> = panel
        .unit_ids()
        .into_iter()
        .filter(|unit| !treated_set.contains(unit))
        .collect();
    if controls.is_empty() {
        return Err(GeoCausalError::InvalidInput(
            "at least one control unit is required".to_string(),
        ));
    }
    let pre_times: Vec<_> = panel
        .times()
        .into_iter()
        .filter(|time| time < &config.intervention_time)
        .collect();
    let post_times: Vec<_> = panel
        .times()
        .into_iter()
        .filter(|time| time >= &config.intervention_time)
        .collect();
    if pre_times.is_empty() || post_times.is_empty() {
        return Err(GeoCausalError::InvalidInput(
            "synthetic DID requires non-empty pre and post periods".to_string(),
        ));
    }
    let pre_treated_mean = group_period_mean(panel, &treated, &pre_times)?;
    let mut unit_weights = BTreeMap::new();
    for control in &controls {
        let control_mean = group_period_mean(panel, std::slice::from_ref(control), &pre_times)?;
        unit_weights.insert(
            control.clone(),
            1.0 / ((control_mean - pre_treated_mean).abs() + 1e-6),
        );
    }
    normalize_weights(&mut unit_weights);
    let mut time_weights = pre_times
        .iter()
        .map(|time| (time.clone(), 1.0 / pre_times.len() as f64))
        .collect::<BTreeMap<_, _>>();
    normalize_weights(&mut time_weights);
    let pre_synthetic_mean = weighted_group_period_mean(panel, &unit_weights, &pre_times)?;
    let treated_post_mean = group_period_mean(panel, &treated, &post_times)?;
    let synthetic_post_mean = weighted_group_period_mean(panel, &unit_weights, &post_times)?;
    let mut warnings = spillover_warnings(panel, &treated);
    warnings.extend(extra_warnings);
    warnings.sort();
    warnings.dedup();
    Ok(SyntheticDIDEstimate {
        effect: (treated_post_mean - pre_treated_mean) - (synthetic_post_mean - pre_synthetic_mean),
        treated_post_mean,
        synthetic_post_mean,
        pre_treated_mean,
        pre_synthetic_mean,
        unit_weights,
        time_weights,
        placebo_estimates: Vec::new(),
        assumptions: vec![
            "causal interpretation requires no unmeasured shocks that differentially affect treated geos after intervention".to_string(),
            "control geos must represent untreated counterfactual behavior for marketing lift, policy rollout, store opening, or network-change analyses".to_string(),
            "spatial spillovers from treated to control geos are assumed absent or small enough to diagnose with reported warnings".to_string(),
            "reported effects are causal panel estimates, not forecasts or prediction accuracy claims".to_string(),
        ],
        warnings,
    })
}

fn treated_units(panel: &GeoCausalPanel) -> Vec<String> {
    panel
        .rows()
        .iter()
        .filter(|row| row.treatment)
        .map(|row| row.unit_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn control_units(panel: &GeoCausalPanel) -> Vec<String> {
    let treated: BTreeSet<_> = treated_units(panel).into_iter().collect();
    panel
        .unit_ids()
        .into_iter()
        .filter(|unit| !treated.contains(unit))
        .collect()
}

fn group_period_mean(panel: &GeoCausalPanel, units: &[String], times: &[String]) -> Result<f64> {
    let unit_set: BTreeSet<_> = units.iter().collect();
    let time_set: BTreeSet<_> = times.iter().collect();
    let values: Vec<_> = panel
        .rows()
        .iter()
        .filter(|row| unit_set.contains(&row.unit_id) && time_set.contains(&row.time))
        .map(|row| row.outcome)
        .collect();
    if values.is_empty() {
        return Err(GeoCausalError::InvalidInput(
            "panel is missing required unit/time observations".to_string(),
        ));
    }
    Ok(mean(values))
}

fn weighted_group_period_mean(
    panel: &GeoCausalPanel,
    unit_weights: &BTreeMap<String, f64>,
    times: &[String],
) -> Result<f64> {
    let time_set: BTreeSet<_> = times.iter().collect();
    let mut total = 0.0;
    let mut denom = 0.0;
    for row in panel
        .rows()
        .iter()
        .filter(|row| time_set.contains(&row.time))
    {
        if let Some(weight) = unit_weights.get(&row.unit_id) {
            total += row.outcome * weight;
            denom += weight;
        }
    }
    if denom <= 0.0 {
        return Err(GeoCausalError::InvalidInput(
            "weighted synthetic control has no observations".to_string(),
        ));
    }
    Ok(total / denom)
}

fn normalize_weights(weights: &mut BTreeMap<String, f64>) {
    let total: f64 = weights.values().sum();
    if total > 0.0 {
        for value in weights.values_mut() {
            *value /= total;
        }
    }
}

fn deterministic_pick(values: &[String], count: usize, seed: u64) -> Vec<String> {
    let mut scored: Vec<_> = values
        .iter()
        .map(|value| (hash_with_seed(value, seed), value.clone()))
        .collect();
    scored.sort_by_key(|row| row.0);
    scored.into_iter().take(count).map(|row| row.1).collect()
}

fn hash_with_seed(value: &str, seed: u64) -> u64 {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for byte in value.bytes() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(byte as u64 + 1442695040888963407);
    }
    state
}

fn spillover_warnings(panel: &GeoCausalPanel, treated: &[String]) -> Vec<String> {
    spillover_diagnostics_for_units(panel, treated).warnings
}

