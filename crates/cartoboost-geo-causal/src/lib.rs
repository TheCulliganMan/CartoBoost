use cartoboost_neural::{
    backend_affine_scores, backend_dense_layer_f32, backend_pairwise_squared_distances_f32,
    select_backend_for, select_backend_for_operations, BackendOperation, BackendSelection,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const GEO_CAUSAL_DENSE_DISPATCH_MIN_OPS: usize = 16_384;
const GEO_CAUSAL_PAIRWISE_DISPATCH_MIN_PAIRS: usize = 16_384;
const GEO_CAUSAL_AFFINE_DISPATCH_MIN_VALUES: usize = 16_384;

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
    backend: BackendSelection,
    panel: Option<GeoCausalPanel>,
    estimate: Option<SyntheticDIDEstimate>,
}

impl SyntheticDIDEstimator {
    pub fn new(config: SyntheticDIDConfig) -> Self {
        Self::new_with_backend(config, Some("cpu")).expect("CPU affine backend is always available")
    }

    pub fn new_with_backend(config: SyntheticDIDConfig, backend: Option<&str>) -> Result<Self> {
        let backend = select_backend_for(backend.or(Some("cpu")), BackendOperation::Affine)
            .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?;
        Ok(Self {
            config,
            backend,
            panel: None,
            estimate: None,
        })
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.backend
    }

    pub fn fit(&mut self, panel: GeoCausalPanel) -> Result<()> {
        let estimate =
            estimate_for_treated_units(&panel, &self.config, None, Vec::new(), &self.backend)?;
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
        let estimate = |idx: usize| {
            let pseudo =
                deterministic_pick(&controls, treated_count, self.config.seed + idx as u64);
            estimate_for_treated_units(
                panel,
                &self.config,
                Some(&pseudo),
                Vec::new(),
                &self.backend,
            )
            .map(|estimate| estimate.effect)
        };
        // CPU placebo fits are independent and deterministic, so Rayon can
        // saturate cores while indexed collection preserves seed order.
        // Device backends remain serial because concurrent command submission
        // to a shared adapter regresses small affine placebo workloads.
        let estimates: Vec<f64> = if self.backend.selected == "cpu" && n > 1 {
            (0..n)
                .into_par_iter()
                .map(estimate)
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..n).map(estimate).collect::<Result<Vec<_>>>()?
        };
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
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalRepresentationReport {
    pub transformed_features: Vec<Vec<f64>>,
    pub heldout_region: String,
    pub raw_rmse: f64,
    pub invariant_rmse: f64,
    pub improvement: f64,
    pub losses: BTreeMap<String, f64>,
    pub warnings: Vec<String>,
    pub metadata: BTreeMap<String, String>,
}

pub type InvariantRiskEncoder = CausalRepresentationReport;
pub type DomainAdversarialGeoEncoder = CausalRepresentationReport;
pub type CounterfactualRepresentationNet = CausalRepresentationReport;
pub type TreatmentEffectRepresentationHead = CausalRepresentationReport;

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
        self.design_with_backend(panel, candidate_count, placebo_n, Some("cpu"))
    }

    pub fn design_with_backend(
        &self,
        panel: &GeoCausalPanel,
        candidate_count: usize,
        placebo_n: usize,
        backend: Option<&str>,
    ) -> Result<GeoExperimentDesign> {
        let backend = select_backend_for(backend.or(Some("cpu")), BackendOperation::Affine)
            .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?;
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
            &backend,
        )?;
        let controls: Vec<_> = panel
            .unit_ids()
            .into_iter()
            .filter(|unit| !candidate_test_geos.contains(unit))
            .collect();
        let placebo_count = if controls.len() <= candidate_test_geos.len() {
            0
        } else {
            placebo_n
        };
        let placebo = |idx: usize| {
            let pseudo =
                deterministic_pick(&controls, candidate_test_geos.len(), self.seed + idx as u64);
            estimate_for_treated_units(panel, &config, Some(&pseudo), Vec::new(), &backend)
                .map(|estimate| estimate.effect)
        };
        let placebo_estimates: Vec<f64> = if backend.selected == "cpu" && placebo_count > 1 {
            (0..placebo_count)
                .into_par_iter()
                .map(placebo)
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..placebo_count)
                .map(placebo)
                .collect::<Result<Vec<_>>>()?
        };
        let baseline = estimate.pre_treated_mean.abs().max(1e-9);
        let mut metadata = BTreeMap::new();
        metadata.insert("backend_requested".to_string(), backend.requested.clone());
        metadata.insert("backend_selected".to_string(), backend.selected.clone());
        metadata.insert("accelerated_operation".to_string(), "affine".to_string());
        Ok(GeoExperimentDesign {
            candidate_test_geos,
            balance_score: (estimate.pre_treated_mean - estimate.pre_synthetic_mean).abs(),
            estimated_detectable_lift: 1.96 * sd(&placebo_estimates) / baseline,
            placebo_estimates,
            assumptions: estimate.assumptions,
            warnings: estimate.warnings,
            metadata,
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
        self.placebo_estimates_with_backend(panel, n, Some("cpu"))
    }

    pub fn placebo_estimates_with_backend(
        &self,
        panel: GeoCausalPanel,
        n: usize,
        backend: Option<&str>,
    ) -> Result<Vec<f64>> {
        let mut estimator = SyntheticDIDEstimator::new_with_backend(
            SyntheticDIDConfig {
                intervention_time: self.intervention_time.clone(),
                seed: self.seed,
            },
            backend,
        )?;
        estimator.fit(panel)?;
        estimator.placebo_test(n)
    }
}

pub fn spillover_diagnostics(panel: &GeoCausalPanel) -> SpilloverDiagnostics {
    let treated = treated_units(panel);
    spillover_diagnostics_for_units(panel, &treated)
}

pub fn spillover_diagnostics_with_backend(
    panel: &GeoCausalPanel,
    backend: Option<&str>,
) -> Result<SpilloverDiagnostics> {
    let treated = treated_units(panel);
    spillover_diagnostics_for_units_with_backend(panel, &treated, backend)
}

pub fn spillover_diagnostics_for_units(
    panel: &GeoCausalPanel,
    treated: &[String],
) -> SpilloverDiagnostics {
    spillover_diagnostics_for_units_with_backend(panel, treated, Some("cpu"))
        .expect("validated geographic panels always support CPU spillover diagnostics")
}

pub fn spillover_diagnostics_for_units_with_backend(
    panel: &GeoCausalPanel,
    treated: &[String],
    backend: Option<&str>,
) -> Result<SpilloverDiagnostics> {
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
    let selection = select_backend_for(backend, BackendOperation::PairwiseDistance)
        .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?;
    let treated_points = treated
        .iter()
        .filter_map(|unit| coords.get(unit).map(|coord| unit_sphere(*coord)))
        .collect::<Vec<_>>();
    let control_points = controls
        .iter()
        .filter_map(|unit| coords.get(unit).map(|coord| unit_sphere(*coord)))
        .collect::<Vec<_>>();
    let distances = if treated_points.is_empty() || control_points.is_empty() {
        Vec::new()
    } else {
        let workload = treated_points.len().saturating_mul(control_points.len());
        let cpu_backend = pairwise_distance_cpu_fallback(&selection, workload)?;
        backend_pairwise_squared_distances_f32(
            cpu_backend.as_ref().unwrap_or(&selection),
            &treated_points,
            &control_points,
        )
        .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?
        .into_iter()
        .flatten()
        .map(chord_squared_to_km)
        .collect::<Vec<_>>()
    };
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
    Ok(SpilloverDiagnostics {
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
    })
}

fn unit_sphere((latitude, longitude): (f64, f64)) -> Vec<f32> {
    let latitude = latitude.to_radians();
    let longitude = longitude.to_radians();
    vec![
        (latitude.cos() * longitude.cos()) as f32,
        (latitude.cos() * longitude.sin()) as f32,
        latitude.sin() as f32,
    ]
}

fn chord_squared_to_km(chord_squared: f32) -> f64 {
    const EARTH_RADIUS_KM: f64 = 6_371.008_8;
    let half_chord = (f64::from(chord_squared.max(0.0)).sqrt() * 0.5).min(1.0);
    2.0 * EARTH_RADIUS_KM * half_chord.asin()
}

fn estimate_for_treated_units(
    panel: &GeoCausalPanel,
    config: &SyntheticDIDConfig,
    override_treated: Option<&[String]>,
    extra_warnings: Vec<String>,
    backend: &BackendSelection,
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
    let pre_treated_mean = group_period_mean_with_backend(panel, &treated, &pre_times, backend)?;
    let mut unit_weights = BTreeMap::new();
    for control in &controls {
        let control_mean = group_period_mean_with_backend(
            panel,
            std::slice::from_ref(control),
            &pre_times,
            backend,
        )?;
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
    let pre_synthetic_mean = weighted_group_period_mean(panel, &unit_weights, &pre_times, backend)?;
    let treated_post_mean = group_period_mean_with_backend(panel, &treated, &post_times, backend)?;
    let synthetic_post_mean =
        weighted_group_period_mean(panel, &unit_weights, &post_times, backend)?;
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

fn group_period_mean_with_backend(
    panel: &GeoCausalPanel,
    units: &[String],
    times: &[String],
    backend: &BackendSelection,
) -> Result<f64> {
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
    let denominator = values.len() as f64;
    let weights = vec![1.0; values.len()];
    affine_weighted_mean(&values, &weights, denominator, backend)
}

fn weighted_group_period_mean(
    panel: &GeoCausalPanel,
    unit_weights: &BTreeMap<String, f64>,
    times: &[String],
    backend: &BackendSelection,
) -> Result<f64> {
    let time_set: BTreeSet<_> = times.iter().collect();
    let mut values = Vec::new();
    let mut weights = Vec::new();
    let mut denom = 0.0;
    for row in panel
        .rows()
        .iter()
        .filter(|row| time_set.contains(&row.time))
    {
        if let Some(weight) = unit_weights.get(&row.unit_id) {
            values.push(row.outcome);
            weights.push(*weight);
            denom += weight;
        }
    }
    if denom <= 0.0 {
        return Err(GeoCausalError::InvalidInput(
            "weighted synthetic control has no observations".to_string(),
        ));
    }
    affine_weighted_mean(&values, &weights, denom, backend)
}

fn pairwise_distance_cpu_fallback(
    backend: &BackendSelection,
    pair_count: usize,
) -> Result<Option<BackendSelection>> {
    if backend.selected != "cpu" && pair_count < GEO_CAUSAL_PAIRWISE_DISPATCH_MIN_PAIRS {
        return select_backend_for(Some("cpu"), BackendOperation::PairwiseDistance)
            .map(Some)
            .map_err(|error| GeoCausalError::InvalidInput(error.to_string()));
    }
    Ok(None)
}

fn affine_weighted_mean(
    values: &[f64],
    weights: &[f64],
    denominator: f64,
    backend: &BackendSelection,
) -> Result<f64> {
    let execution_backend =
        if backend.selected != "cpu" && values.len() < GEO_CAUSAL_AFFINE_DISPATCH_MIN_VALUES {
            select_backend_for(Some("cpu"), BackendOperation::Affine)
                .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?
        } else {
            backend.clone()
        };
    let total = backend_affine_scores(
        &execution_backend,
        &[values.to_vec()],
        &vec![0.0; weights.len()],
        weights,
        &[0.0],
    )
    .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?[0];
    Ok(total / denominator)
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

pub fn causal_representation_report_json(
    features: &[Vec<f64>],
    outcomes: &[f64],
    regions: &[String],
    heldout_region: &str,
) -> Result<String> {
    causal_representation_report_json_with_backend(
        features,
        outcomes,
        regions,
        heldout_region,
        Some("cpu"),
    )
}

pub fn causal_representation_report_json_with_backend(
    features: &[Vec<f64>],
    outcomes: &[f64],
    regions: &[String],
    heldout_region: &str,
    backend: Option<&str>,
) -> Result<String> {
    let report = causal_representation_report_with_backend(
        features,
        outcomes,
        regions,
        heldout_region,
        backend,
    )?;
    serde_json::to_string(&report).map_err(|err| GeoCausalError::InvalidInput(err.to_string()))
}

pub fn causal_representation_report(
    features: &[Vec<f64>],
    outcomes: &[f64],
    regions: &[String],
    heldout_region: &str,
) -> Result<CausalRepresentationReport> {
    causal_representation_report_with_backend(
        features,
        outcomes,
        regions,
        heldout_region,
        Some("cpu"),
    )
}

pub fn causal_representation_report_with_backend(
    features: &[Vec<f64>],
    outcomes: &[f64],
    regions: &[String],
    heldout_region: &str,
    backend: Option<&str>,
) -> Result<CausalRepresentationReport> {
    validate_representation_inputs(features, outcomes, regions, heldout_region)?;
    let selection = select_backend_for_operations(
        backend,
        &[
            BackendOperation::Affine,
            BackendOperation::Dense,
            BackendOperation::PairwiseDistance,
        ],
    )
    .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?;
    let dim = features[0].len();
    let global_mean = column_mean(features);
    let region_mean = region_feature_means(features, regions, dim);
    let transformed = features
        .iter()
        .zip(regions)
        .map(|(row, region)| {
            row.iter()
                .enumerate()
                .map(|(idx, value)| value - region_mean[region][idx] + global_mean[idx])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let train_indices = regions
        .iter()
        .enumerate()
        .filter_map(|(idx, region)| (region != heldout_region).then_some(idx))
        .collect::<Vec<_>>();
    let test_indices = regions
        .iter()
        .enumerate()
        .filter_map(|(idx, region)| (region == heldout_region).then_some(idx))
        .collect::<Vec<_>>();
    let raw_weights =
        ridge_fit_indexed_with_backend(features, outcomes, &train_indices, &selection)?;
    let invariant_weights =
        ridge_fit_indexed_with_backend(&transformed, outcomes, &train_indices, &selection)?;
    let raw_rmse =
        indexed_rmse_with_backend(features, outcomes, &test_indices, &raw_weights, &selection)?;
    let invariant_rmse = indexed_rmse_with_backend(
        &transformed,
        outcomes,
        &test_indices,
        &invariant_weights,
        &selection,
    )?;
    let mut losses = BTreeMap::new();
    losses.insert(
        "supervised_outcome_loss".to_string(),
        indexed_mse_with_backend(
            &transformed,
            outcomes,
            &train_indices,
            &invariant_weights,
            &selection,
        )?,
    );
    let domain_loss = mean_region_distance_with_backend(&transformed, regions, &selection)?;
    losses.insert("domain_adversarial_loss".to_string(), domain_loss);
    losses.insert(
        "invariant_risk_penalty".to_string(),
        (raw_rmse - invariant_rmse).abs(),
    );
    losses.insert("treatment_balance_penalty".to_string(), domain_loss);
    losses.insert(
        "representation_smoothness_penalty".to_string(),
        mean_row_variation(&transformed),
    );
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "model_class".to_string(),
        "InvariantRiskEncoder".to_string(),
    );
    metadata.insert(
        "domain_encoder".to_string(),
        "DomainAdversarialGeoEncoder".to_string(),
    );
    metadata.insert(
        "counterfactual_net".to_string(),
        "CounterfactualRepresentationNet".to_string(),
    );
    metadata.insert(
        "treatment_head".to_string(),
        "TreatmentEffectRepresentationHead".to_string(),
    );
    metadata.insert(
        "supplements".to_string(),
        "SyntheticDIDEstimator,GeoExperimentDesigner".to_string(),
    );
    metadata.insert("backend_requested".to_string(), selection.requested.clone());
    metadata.insert("backend_selected".to_string(), selection.selected.clone());
    metadata.insert(
        "accelerated_operations".to_string(),
        "dense_ridge_products,affine_scoring,pairwise_domain_distance".to_string(),
    );
    Ok(CausalRepresentationReport {
        transformed_features: transformed,
        heldout_region: heldout_region.to_string(),
        raw_rmse,
        invariant_rmse,
        improvement: raw_rmse - invariant_rmse,
        losses,
        warnings: vec![
            "Representation learning does not prove causal identification; use it only as a supplement to an identified design.".to_string(),
        ],
        metadata,
    })
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

fn validate_representation_inputs(
    features: &[Vec<f64>],
    outcomes: &[f64],
    regions: &[String],
    heldout_region: &str,
) -> Result<()> {
    if features.is_empty() || features[0].is_empty() {
        return Err(GeoCausalError::InvalidInput(
            "features must be a non-empty matrix".to_string(),
        ));
    }
    if outcomes.len() != features.len() || regions.len() != features.len() {
        return Err(GeoCausalError::InvalidInput(
            "features, outcomes, and regions must have matching row counts".to_string(),
        ));
    }
    let dim = features[0].len();
    for row in features {
        if row.len() != dim || row.iter().any(|value| !value.is_finite()) {
            return Err(GeoCausalError::InvalidInput(
                "feature rows must be finite with fixed width".to_string(),
            ));
        }
    }
    if outcomes.iter().any(|value| !value.is_finite()) {
        return Err(GeoCausalError::InvalidInput(
            "outcomes must be finite".to_string(),
        ));
    }
    if !regions.iter().any(|region| region == heldout_region)
        || !regions.iter().any(|region| region != heldout_region)
    {
        return Err(GeoCausalError::InvalidInput(
            "heldout_region must have both held-out and training rows".to_string(),
        ));
    }
    Ok(())
}

fn column_mean(features: &[Vec<f64>]) -> Vec<f64> {
    (0..features[0].len())
        .into_par_iter()
        .map(|idx| {
            features
                .iter()
                .map(|row| row[idx] / features.len() as f64)
                .sum()
        })
        .collect()
}

fn region_feature_means(
    features: &[Vec<f64>],
    regions: &[String],
    dim: usize,
) -> BTreeMap<String, Vec<f64>> {
    let mut sums: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for (row, region) in features.iter().zip(regions) {
        let entry = sums.entry(region.clone()).or_insert(vec![0.0; dim]);
        for (idx, value) in row.iter().enumerate() {
            entry[idx] += value;
        }
        *counts.entry(region.clone()).or_insert(0) += 1;
    }
    sums.into_iter()
        .map(|(region, values)| {
            let count = counts[&region] as f64;
            (
                region,
                values.into_iter().map(|value| value / count).collect(),
            )
        })
        .collect()
}

fn ridge_fit_indexed(features: &[Vec<f64>], outcomes: &[f64], indices: &[usize]) -> Vec<f64> {
    let cols = features[0].len() + 1;
    let mut xtx = vec![vec![0.0; cols]; cols];
    let mut xty = vec![0.0; cols];
    for &idx in indices {
        let mut x = vec![1.0];
        x.extend_from_slice(&features[idx]);
        for r in 0..cols {
            xty[r] += x[r] * outcomes[idx];
            for c in 0..cols {
                xtx[r][c] += x[r] * x[c];
            }
        }
    }
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        row[idx] += 1.0e-6;
    }
    solve_linear(xtx, xty)
}

fn ridge_fit_indexed_with_backend(
    features: &[Vec<f64>],
    outcomes: &[f64],
    indices: &[usize],
    backend: &BackendSelection,
) -> Result<Vec<f64>> {
    let cols = features[0].len() + 1;
    if backend.selected == "cpu"
        || indices.len().saturating_mul(cols).saturating_mul(cols)
            < GEO_CAUSAL_DENSE_DISPATCH_MIN_OPS
    {
        return Ok(ridge_fit_indexed(features, outcomes, indices));
    }
    let design = indices
        .iter()
        .map(|&idx| {
            std::iter::once(1.0_f32)
                .chain(features[idx].iter().map(|&value| value as f32))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let transposed = (0..cols)
        .map(|col| design.iter().map(|row| row[col]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let flattened_design = design.iter().flatten().copied().collect::<Vec<_>>();
    let xtx = backend_dense_layer_f32(backend, &transposed, &flattened_design, &vec![0.0; cols])
        .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?;
    let selected_outcomes = indices
        .iter()
        .map(|&idx| outcomes[idx] as f32)
        .collect::<Vec<_>>();
    let xty = backend_dense_layer_f32(backend, &transposed, &selected_outcomes, &[0.0])
        .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?;
    let mut xtx = xtx
        .into_iter()
        .map(|row| row.into_iter().map(f64::from).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    for (idx, row) in xtx.iter_mut().enumerate().skip(1) {
        row[idx] += 1.0e-6;
    }
    let xty = xty
        .into_iter()
        .map(|row| f64::from(row[0]))
        .collect::<Vec<_>>();
    Ok(solve_linear(xtx, xty))
}

fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for pivot in 0..n {
        let mut best = pivot;
        for row in pivot + 1..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let diag = matrix[pivot][pivot];
        if diag.abs() < 1.0e-12 {
            continue;
        }
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= diag;
        }
        rhs[pivot] /= diag;
        let pivot_row = matrix[pivot].clone();
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for (col, pivot_value) in pivot_row.iter().enumerate().take(n).skip(pivot) {
                matrix[row][col] -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    rhs
}

fn indexed_rmse_with_backend(
    features: &[Vec<f64>],
    outcomes: &[f64],
    indices: &[usize],
    weights: &[f64],
    backend: &BackendSelection,
) -> Result<f64> {
    Ok(indexed_mse_with_backend(features, outcomes, indices, weights, backend)?.sqrt())
}

fn indexed_mse_with_backend(
    features: &[Vec<f64>],
    outcomes: &[f64],
    indices: &[usize],
    weights: &[f64],
    backend: &BackendSelection,
) -> Result<f64> {
    let rows = indices
        .iter()
        .map(|&index| features[index].clone())
        .collect::<Vec<_>>();
    let cpu_backend =
        if should_accelerate_affine(backend, rows.len(), weights.len().saturating_sub(1)) {
            None
        } else {
            Some(
                select_backend_for(Some("cpu"), BackendOperation::Affine)
                    .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?,
            )
        };
    let execution_backend = cpu_backend.as_ref().unwrap_or(backend);
    let predictions = backend_affine_scores(
        execution_backend,
        &rows,
        &vec![0.0; weights.len() - 1],
        &weights[1..],
        &vec![weights[0]; rows.len()],
    )
    .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?;
    Ok(predictions
        .iter()
        .zip(indices)
        .map(|(prediction, &index)| {
            let error = prediction - outcomes[index];
            error * error
        })
        .sum::<f64>()
        / indices.len() as f64)
}

fn should_accelerate_affine(
    backend: &BackendSelection,
    row_count: usize,
    feature_count: usize,
) -> bool {
    backend.selected != "cpu"
        && row_count.saturating_mul(feature_count) >= GEO_CAUSAL_AFFINE_DISPATCH_MIN_VALUES
}

fn mean_region_distance(features: &[Vec<f64>], regions: &[String]) -> f64 {
    let means = region_feature_means(features, regions, features[0].len());
    let global = column_mean(features);
    means
        .values()
        .map(|row| {
            row.iter()
                .zip(&global)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .sum::<f64>()
        / means.len() as f64
}

fn mean_region_distance_with_backend(
    features: &[Vec<f64>],
    regions: &[String],
    backend: &BackendSelection,
) -> Result<f64> {
    let means = region_feature_means(features, regions, features[0].len());
    if backend.selected == "cpu"
        || means.len().saturating_mul(features[0].len()) < GEO_CAUSAL_PAIRWISE_DISPATCH_MIN_PAIRS
    {
        return Ok(mean_region_distance(features, regions));
    }
    let means = means
        .into_values()
        .map(|row| row.into_iter().map(|value| value as f32).collect())
        .collect::<Vec<Vec<f32>>>();
    let global = vec![column_mean(features)
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>()];
    let distances = backend_pairwise_squared_distances_f32(backend, &means, &global)
        .map_err(|error| GeoCausalError::InvalidInput(error.to_string()))?;
    Ok(distances
        .iter()
        .map(|row| f64::from(row[0]).max(0.0).sqrt())
        .sum::<f64>()
        / distances.len() as f64)
}

fn mean_row_variation(features: &[Vec<f64>]) -> f64 {
    if features.len() < 2 {
        return 0.0;
    }
    features
        .windows(2)
        .map(|pair| {
            pair[0]
                .iter()
                .zip(&pair[1])
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>()
        })
        .sum::<f64>()
        / (features.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use cartoboost_neural::{available_backends, backend_supports_operation};

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
    fn representation_affine_dispatch_avoids_small_device_launches() {
        for backend_name in available_backends() {
            let backend =
                select_backend_for(Some(&backend_name), BackendOperation::Affine).unwrap();
            assert!(!should_accelerate_affine(&backend, 1, 4));
            assert_eq!(
                should_accelerate_affine(&backend, 4_096, 4),
                backend_name != "cpu"
            );
        }
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
    fn synthetic_did_runs_on_every_available_affine_backend() {
        let panel = known_effect_panel(5.0);
        let cpu_effect = {
            let mut estimator = SyntheticDIDEstimator::new(SyntheticDIDConfig {
                intervention_time: "2026-01-04".to_string(),
                seed: 7,
            });
            estimator.fit(panel.clone()).unwrap();
            estimator.estimate_effect().unwrap().effect
        };
        for backend in available_backends() {
            if !backend_supports_operation(&backend, BackendOperation::Affine) {
                continue;
            }
            let mut estimator = SyntheticDIDEstimator::new_with_backend(
                SyntheticDIDConfig {
                    intervention_time: "2026-01-04".to_string(),
                    seed: 7,
                },
                Some(&backend),
            )
            .unwrap();
            estimator.fit(panel.clone()).unwrap();
            assert_eq!(estimator.backend().selected, backend);
            assert!(
                (estimator.estimate_effect().unwrap().effect - cpu_effect).abs() < 1.0e-4,
                "Synthetic DID mismatch on {backend}"
            );
            assert_eq!(estimator.placebo_test(2).unwrap().len(), 2);
        }
    }

    #[test]
    fn large_weighted_panel_mean_runs_on_every_affine_backend() {
        let rows = (0..GEO_CAUSAL_AFFINE_DISPATCH_MIN_VALUES)
            .map(|index| GeoCausalRow {
                unit_id: "control".to_string(),
                time: "pre".to_string(),
                outcome: (index % 113) as f64 / 113.0,
                treatment: false,
                covariates: BTreeMap::new(),
                latitude: None,
                longitude: None,
                region_id: None,
            })
            .collect();
        let panel = GeoCausalPanel::new(rows, Vec::new()).unwrap();
        let unit_weights = BTreeMap::from([("control".to_string(), 1.0)]);
        let times = vec!["pre".to_string()];
        let cpu = select_backend_for(Some("cpu"), BackendOperation::Affine).unwrap();
        let expected = weighted_group_period_mean(&panel, &unit_weights, &times, &cpu).unwrap();
        let expected_unweighted =
            group_period_mean_with_backend(&panel, &["control".to_string()], &times, &cpu).unwrap();
        for backend_name in available_backends() {
            if !backend_supports_operation(&backend_name, BackendOperation::Affine) {
                continue;
            }
            let backend =
                select_backend_for(Some(&backend_name), BackendOperation::Affine).unwrap();
            let actual = weighted_group_period_mean(&panel, &unit_weights, &times, &backend)
                .unwrap_or_else(|error| panic!("{backend_name} weighted mean failed: {error}"));
            let actual_unweighted =
                group_period_mean_with_backend(&panel, &["control".to_string()], &times, &backend)
                    .unwrap_or_else(|error| {
                        panic!("{backend_name} unweighted mean failed: {error}")
                    });
            assert!(
                (actual - expected).abs() <= 2.0e-4,
                "{backend_name}: expected {expected}, got {actual}"
            );
            assert!((actual_unweighted - expected_unweighted).abs() <= 2.0e-4);
        }
    }

    #[test]
    fn spillover_pairwise_dispatch_avoids_small_device_launches() {
        for backend_name in available_backends() {
            if !backend_supports_operation(&backend_name, BackendOperation::PairwiseDistance) {
                continue;
            }
            let backend =
                select_backend_for(Some(&backend_name), BackendOperation::PairwiseDistance)
                    .unwrap();
            assert_eq!(
                pairwise_distance_cpu_fallback(&backend, 1)
                    .unwrap()
                    .is_some(),
                backend_name != "cpu"
            );
            assert!(pairwise_distance_cpu_fallback(
                &backend,
                GEO_CAUSAL_PAIRWISE_DISPATCH_MIN_PAIRS
            )
            .unwrap()
            .is_none());
        }
    }

    #[test]
    fn geo_experiment_design_runs_on_every_available_backend() {
        let panel = known_effect_panel(0.0);
        let designer = GeoExperimentDesigner {
            intervention_time: "2026-01-04".to_string(),
            seed: 9,
        };
        let expected = designer
            .design_with_backend(&panel, 1, 2, Some("cpu"))
            .unwrap();
        for backend in available_backends() {
            let actual = designer
                .design_with_backend(&panel, 1, 2, Some(&backend))
                .unwrap_or_else(|error| panic!("{backend} design failed: {error}"));
            assert_eq!(actual.candidate_test_geos, expected.candidate_test_geos);
            assert_eq!(
                actual.placebo_estimates.len(),
                expected.placebo_estimates.len()
            );
            assert_eq!(actual.metadata["backend_selected"], backend);
            assert!((actual.balance_score - expected.balance_score).abs() < 1.0e-4);
        }
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

    #[test]
    fn causal_representation_improves_heldout_region_and_warns_on_identification() {
        let mut features = Vec::new();
        let mut outcomes = Vec::new();
        let mut regions = Vec::new();
        for (region, shift) in [("a", 0.0), ("b", 3.0), ("c", -4.0)] {
            for idx in 0..8 {
                let stable = idx as f64 / 4.0;
                features.push(vec![stable + shift, stable * 0.5 + shift]);
                outcomes.push(2.0 + 1.5 * stable);
                regions.push(region.to_string());
            }
        }
        let report = causal_representation_report(&features, &outcomes, &regions, "c").unwrap();

        assert!(report.invariant_rmse < report.raw_rmse);
        assert!(report.improvement > 0.0);
        assert!(report.losses.contains_key("domain_adversarial_loss"));
        assert!(report
            .warnings
            .iter()
            .any(|warning| warning.contains("does not prove causal identification")));
        assert_eq!(
            report.metadata.get("supplements").map(String::as_str),
            Some("SyntheticDIDEstimator,GeoExperimentDesigner")
        );

        for backend in available_backends() {
            if [
                BackendOperation::Affine,
                BackendOperation::Dense,
                BackendOperation::PairwiseDistance,
            ]
            .into_iter()
            .any(|operation| !backend_supports_operation(&backend, operation))
            {
                continue;
            }
            let accelerated = causal_representation_report_with_backend(
                &features,
                &outcomes,
                &regions,
                "c",
                Some(&backend),
            )
            .unwrap();
            assert!(
                (accelerated.raw_rmse - report.raw_rmse).abs() < 1.0e-4,
                "{backend}"
            );
            assert!(
                (accelerated.invariant_rmse - report.invariant_rmse).abs() < 1.0e-4,
                "{backend}"
            );
        }
    }
}
