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

