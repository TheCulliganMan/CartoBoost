#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeoExperimentDesign {
    pub candidate_test_geos: Vec<String>,
    pub balance_score: f64,
    pub estimated_detectable_lift: f64,
    pub placebo_estimates: Vec<f64>,
    pub assumptions: Vec<String>,
    pub warnings: Vec<String>,
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

