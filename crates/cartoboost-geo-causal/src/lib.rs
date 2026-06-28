use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, thiserror::Error)]
pub enum GeoCausalError {
    #[error("{0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, GeoCausalError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoCausalRow {
    pub unit_id: String,
    pub time: String,
    pub outcome: f64,
    pub treatment: bool,
    pub covariates: BTreeMap<String, f64>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub region_id: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpatialWeight {
    pub from_unit: String,
    pub to_unit: String,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoCausalPanel {
    rows: Vec<GeoCausalRow>,
    spatial_weights: Vec<SpatialWeight>,
}

impl GeoCausalPanel {
    pub fn new(rows: Vec<GeoCausalRow>, spatial_weights: Vec<SpatialWeight>) -> Result<Self> {
        if rows.is_empty() {
            return Err(GeoCausalError::InvalidInput(
                "GeoCausalPanel requires at least one row".to_string(),
            ));
        }
        for (idx, row) in rows.iter().enumerate() {
            if row.unit_id.is_empty() {
                return Err(GeoCausalError::InvalidInput(format!(
                    "row {idx} has an empty unit_id"
                )));
            }
            if row.time.is_empty() {
                return Err(GeoCausalError::InvalidInput(format!(
                    "row {idx} has an empty time"
                )));
            }
            if !row.outcome.is_finite() {
                return Err(GeoCausalError::InvalidInput(format!(
                    "row {idx} outcome must be finite"
                )));
            }
            if row.latitude.is_some() != row.longitude.is_some() {
                return Err(GeoCausalError::InvalidInput(format!(
                    "row {idx} must provide both latitude and longitude or neither"
                )));
            }
        }
        let units: BTreeSet<_> = rows.iter().map(|row| row.unit_id.clone()).collect();
        for edge in &spatial_weights {
            if !units.contains(&edge.from_unit) || !units.contains(&edge.to_unit) {
                return Err(GeoCausalError::InvalidInput(format!(
                    "spatial weight references unknown units {} -> {}",
                    edge.from_unit, edge.to_unit
                )));
            }
            if !edge.weight.is_finite() || edge.weight < 0.0 {
                return Err(GeoCausalError::InvalidInput(
                    "spatial weights must be finite and non-negative".to_string(),
                ));
            }
        }
        Ok(Self {
            rows,
            spatial_weights,
        })
    }

    pub fn rows(&self) -> &[GeoCausalRow] {
        &self.rows
    }

    pub fn spatial_weights(&self) -> &[SpatialWeight] {
        &self.spatial_weights
    }

    pub fn unit_ids(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|row| row.unit_id.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    pub fn times(&self) -> Vec<String> {
        self.rows
            .iter()
            .map(|row| row.time.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyntheticDIDConfig {
    pub intervention_time: String,
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyntheticDIDEstimate {
    pub effect: f64,
    pub treated_post_mean: f64,
    pub synthetic_post_mean: f64,
    pub pre_treated_mean: f64,
    pub pre_synthetic_mean: f64,
    pub unit_weights: BTreeMap<String, f64>,
    pub time_weights: BTreeMap<String, f64>,
    pub placebo_estimates: Vec<f64>,
    pub assumptions: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct SyntheticDIDEstimator {
    config: SyntheticDIDConfig,
    panel: Option<GeoCausalPanel>,
    estimate: Option<SyntheticDIDEstimate>,
}

impl SyntheticDIDEstimator {
    pub fn new(config: SyntheticDIDConfig) -> Self {
        Self {
            config,
            panel: None,
            estimate: None,
        }
    }

    pub fn fit(&mut self, panel: GeoCausalPanel) -> Result<()> {
        let estimate = estimate_for_treated_units(&panel, &self.config, None, Vec::new())?;
        self.panel = Some(panel);
        self.estimate = Some(estimate);
        Ok(())
    }

    pub fn estimate_effect(&self) -> Result<SyntheticDIDEstimate> {
        self.estimate.clone().ok_or_else(|| {
            GeoCausalError::InvalidInput("SyntheticDIDEstimator must be fit first".to_string())
        })
    }

    pub fn placebo_test(&mut self, n: usize) -> Result<Vec<f64>> {
        let panel = self.panel.as_ref().ok_or_else(|| {
            GeoCausalError::InvalidInput("SyntheticDIDEstimator must be fit first".to_string())
        })?;
        let treated_count = treated_units(panel).len();
        let controls = control_units(panel);
        if treated_count == 0 || controls.len() <= treated_count {
            return Err(GeoCausalError::InvalidInput(
                "placebo tests require treated units and more control units than treated units"
                    .to_string(),
            ));
        }
        let mut estimates = Vec::with_capacity(n);
        for idx in 0..n {
            let pseudo =
                deterministic_pick(&controls, treated_count, self.config.seed + idx as u64);
            estimates.push(
                estimate_for_treated_units(panel, &self.config, Some(&pseudo), Vec::new())?.effect,
            );
        }
        if let Some(current) = &mut self.estimate {
            current.placebo_estimates = estimates.clone();
        }
        Ok(estimates)
    }

    pub fn summary_json(&self) -> Result<String> {
        serde_json::to_string_pretty(&self.estimate_effect()?)
            .map_err(|err| GeoCausalError::InvalidInput(err.to_string()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoExperimentDesign {
    pub candidate_test_geos: Vec<String>,
    pub balance_score: f64,
    pub estimated_detectable_lift: f64,
    pub placebo_estimates: Vec<f64>,
    pub assumptions: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct GeoExperimentDesigner {
    pub intervention_time: String,
    pub seed: u64,
}

impl GeoExperimentDesigner {
    pub fn design(
        &self,
        panel: &GeoCausalPanel,
        candidate_count: usize,
        placebo_n: usize,
    ) -> Result<GeoExperimentDesign> {
        if candidate_count == 0 {
            return Err(GeoCausalError::InvalidInput(
                "candidate_count must be positive".to_string(),
            ));
        }
        let pre_rows: Vec<_> = panel
            .rows()
            .iter()
            .filter(|row| row.time.as_str() < self.intervention_time.as_str())
            .collect();
        if pre_rows.is_empty() {
            return Err(GeoCausalError::InvalidInput(
                "design requires pre-period rows before intervention_time".to_string(),
            ));
        }
        let overall_mean = mean(pre_rows.iter().map(|row| row.outcome));
        let mut unit_scores: Vec<_> = panel
            .unit_ids()
            .into_iter()
            .map(|unit| {
                let unit_mean = mean(
                    pre_rows
                        .iter()
                        .filter(|row| row.unit_id == unit)
                        .map(|row| row.outcome),
                );
                let exposure = panel
                    .spatial_weights()
                    .iter()
                    .filter(|edge| edge.from_unit == unit || edge.to_unit == unit)
                    .map(|edge| edge.weight)
                    .sum::<f64>();
                (unit, (unit_mean - overall_mean).abs(), exposure)
            })
            .collect();
        unit_scores.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
                .then_with(|| a.0.cmp(&b.0))
        });
        let candidate_test_geos: Vec<String> = unit_scores
            .into_iter()
            .take(candidate_count)
            .map(|row| row.0)
            .collect();
        let config = SyntheticDIDConfig {
            intervention_time: self.intervention_time.clone(),
            seed: self.seed,
        };
        let estimate = estimate_for_treated_units(
            panel,
            &config,
            Some(&candidate_test_geos),
            spillover_warnings(panel, &candidate_test_geos),
        )?;
        let mut placebo_estimates = Vec::new();
        for idx in 0..placebo_n {
            let controls: Vec<_> = panel
                .unit_ids()
                .into_iter()
                .filter(|unit| !candidate_test_geos.contains(unit))
                .collect();
            if controls.len() <= candidate_test_geos.len() {
                break;
            }
            let pseudo =
                deterministic_pick(&controls, candidate_test_geos.len(), self.seed + idx as u64);
            placebo_estimates.push(
                estimate_for_treated_units(panel, &config, Some(&pseudo), Vec::new())?.effect,
            );
        }
        let baseline = estimate.pre_treated_mean.abs().max(1e-9);
        Ok(GeoExperimentDesign {
            candidate_test_geos,
            balance_score: (estimate.pre_treated_mean - estimate.pre_synthetic_mean).abs(),
            estimated_detectable_lift: 1.96 * sd(&placebo_estimates) / baseline,
            placebo_estimates,
            assumptions: estimate.assumptions,
            warnings: estimate.warnings,
        })
    }
}

pub type GeoLiftEstimator = GeoExperimentDesigner;

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

fn unit_coordinates(panel: &GeoCausalPanel) -> BTreeMap<String, (f64, f64)> {
    let mut coords = BTreeMap::new();
    for row in panel.rows() {
        if let (Some(lat), Some(lon)) = (row.latitude, row.longitude) {
            coords.entry(row.unit_id.clone()).or_insert((lat, lon));
        }
    }
    coords
}

fn mean(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = mean(values.iter().copied());
    (values
        .iter()
        .map(|value| (value - avg).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn known_effect_panel(effect: f64) -> GeoCausalPanel {
        let mut rows = Vec::new();
        for unit in ["treated", "control_a", "control_b", "control_c"] {
            for time in 0..8 {
                let post = time >= 4;
                let base = 10.0 + time as f64;
                let unit_shift = match unit {
                    "treated" | "control_a" => 1.0,
                    "control_b" => 0.8,
                    _ => 1.2,
                };
                rows.push(GeoCausalRow {
                    unit_id: unit.to_string(),
                    time: format!("2026-01-{time:02}"),
                    outcome: base
                        + unit_shift
                        + if unit == "treated" && post {
                            effect
                        } else {
                            0.0
                        },
                    treatment: unit == "treated" && post,
                    covariates: BTreeMap::new(),
                    latitude: Some(40.0 + if unit == "treated" { 0.0 } else { 1.0 }),
                    longitude: Some(-73.0),
                    region_id: Some(unit.to_string()),
                });
            }
        }
        GeoCausalPanel::new(
            rows,
            vec![SpatialWeight {
                from_unit: "treated".to_string(),
                to_unit: "control_a".to_string(),
                weight: 1.0,
            }],
        )
        .unwrap()
    }

    #[test]
    fn synthetic_did_recovers_known_effect() {
        let panel = known_effect_panel(5.0);
        let mut estimator = SyntheticDIDEstimator::new(SyntheticDIDConfig {
            intervention_time: "2026-01-04".to_string(),
            seed: 7,
        });
        estimator.fit(panel).unwrap();
        assert!((estimator.estimate_effect().unwrap().effect - 5.0).abs() < 0.25);
    }

    #[test]
    fn zero_effect_placebo_is_centered_near_zero_and_deterministic() {
        let panel = known_effect_panel(0.0);
        let mut a = SyntheticDIDEstimator::new(SyntheticDIDConfig {
            intervention_time: "2026-01-04".to_string(),
            seed: 42,
        });
        a.fit(panel.clone()).unwrap();
        let first = a.placebo_test(4).unwrap();
        let mut b = SyntheticDIDEstimator::new(SyntheticDIDConfig {
            intervention_time: "2026-01-04".to_string(),
            seed: 42,
        });
        b.fit(panel).unwrap();
        assert_eq!(first, b.placebo_test(4).unwrap());
        assert!(mean(first).abs() < 0.5);
    }

    #[test]
    fn spillover_warning_fires_for_adjacent_units() {
        let diagnostics = spillover_diagnostics(&known_effect_panel(1.0));
        assert!(!diagnostics.warnings.is_empty());
        assert_eq!(diagnostics.adjacent_treated_control_pairs.len(), 1);
    }
}
