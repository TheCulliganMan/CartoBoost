pub fn conditional_flow_fit_json(
    hidden: &[Vec<f64>],
    residuals: &[f64],
    quantiles: &[f64],
    sample_count: usize,
) -> Result<String> {
    let model = ConditionalFlowDistributionHead::fit(hidden, residuals, quantiles, sample_count)?;
    serde_json::to_string(&model).map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
}

pub fn conditional_flow_predict_json(
    artifact_json: &str,
    hidden: &[Vec<f64>],
    actual: Option<&[f64]>,
) -> Result<String> {
    let model: ConditionalFlowDistributionHead = serde_json::from_str(artifact_json)
        .map_err(|err| CartoBoostError::InvalidInput(err.to_string()))?;
    let prediction = model.predict(hidden, actual)?;
    serde_json::to_string(&prediction).map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
}

pub fn diffusion_scenario_generate_json(
    point_forecast: &[Vec<f64>],
    edges: &[DiffusionEdge],
    scenario_count: usize,
    diffusion_steps: usize,
    shock_scale: f64,
) -> Result<String> {
    let model =
        GeoTemporalDiffusionScenarioModel::new(scenario_count, diffusion_steps, shock_scale)?;
    let prediction = model.generate(point_forecast, edges)?;
    serde_json::to_string(&prediction).map_err(|err| CartoBoostError::InvalidInput(err.to_string()))
}

impl ConditionalFlowDistributionHead {
    pub fn fit(
        hidden: &[Vec<f64>],
        residuals: &[f64],
        quantiles: &[f64],
        sample_count: usize,
    ) -> Result<Self> {
        validate_same_non_empty_matrix(hidden, residuals, "hidden", "residuals")?;
        validate_quantile_grid(quantiles)?;
        if sample_count == 0 {
            return invalid("sample_count must be positive");
        }
        let location_weights = ridge_fit(hidden, residuals, 1.0e-6);
        let predicted = predict_linear(hidden, &location_weights)?;
        let abs_residuals = residuals
            .iter()
            .zip(predicted.iter())
            .map(|(&actual, &pred)| (actual - pred).abs().ln_1p())
            .collect::<Vec<_>>();
        let scale_weights = ridge_fit(hidden, &abs_residuals, 1.0e-6);
        let residual_scale = (residuals.iter().map(|v| v * v).sum::<f64>()
            / residuals.len() as f64)
            .sqrt()
            .max(1.0e-9);
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "architecture".to_string(),
            "conditional_residual_sampler".to_string(),
        );
        metadata.insert("invertible_flow".to_string(), "false".to_string());
        metadata.insert("sample_count".to_string(), sample_count.to_string());
        metadata.insert(
            "quantiles".to_string(),
            serde_json::to_string(quantiles).unwrap(),
        );
        Ok(Self {
            quantiles: quantiles.to_vec(),
            sample_count,
            location_weights,
            scale_weights,
            residual_scale,
            metadata,
        })
    }

    pub fn predict(&self, hidden: &[Vec<f64>], actual: Option<&[f64]>) -> Result<FlowPrediction> {
        if hidden.is_empty() {
            return invalid("hidden must contain at least one row");
        }
        if let Some(actual) = actual {
            validate_same_non_empty(actual, &vec![0.0; hidden.len()], "actual", "hidden")?;
        }
        let location = predict_linear(hidden, &self.location_weights)?;
        let raw_scale = predict_linear(hidden, &self.scale_weights)?;
        let scale = raw_scale
            .iter()
            .map(|value| value.exp().max(1.0e-6) * self.residual_scale)
            .collect::<Vec<_>>();
        let samples = location
            .iter()
            .zip(scale.iter())
            .enumerate()
            .map(|(row, (&loc, &s))| {
                (0..self.sample_count)
                    .map(|sample| loc + s * deterministic_standard_sample(row, sample))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let marginal_quantiles = location
            .iter()
            .zip(scale.iter())
            .map(|(&loc, &s)| {
                self.quantiles
                    .iter()
                    .map(|&q| loc + s * normal_quantile_proxy(q))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let joint_scenario_paths = (0..self.sample_count)
            .map(|sample| samples.iter().map(|row| row[sample]).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let log_likelihood = match actual {
            Some(values) => values
                .iter()
                .zip(location.iter())
                .zip(scale.iter())
                .map(|((&y, &loc), &s)| gaussian_log_likelihood(y, loc, s))
                .collect(),
            None => vec![0.0; hidden.len()],
        };
        let mut tail_risk_metrics = BTreeMap::new();
        tail_risk_metrics.insert(
            "p05_mean".to_string(),
            quantile_mean(&marginal_quantiles, 0)?,
        );
        tail_risk_metrics.insert(
            "p95_mean".to_string(),
            quantile_mean(&marginal_quantiles, self.quantiles.len() - 1)?,
        );
        tail_risk_metrics.insert(
            "expected_shortfall_low".to_string(),
            samples
                .iter()
                .map(|row| row.iter().copied().fold(f64::INFINITY, f64::min))
                .sum::<f64>()
                / samples.len() as f64,
        );
        let mut metrics = BTreeMap::new();
        if let Some(values) = actual {
            metrics.insert(
                "log_likelihood_mean".to_string(),
                log_likelihood.iter().sum::<f64>() / log_likelihood.len() as f64,
            );
            metrics.insert(
                "crps".to_string(),
                crps_approximation(values, &self.quantiles, &marginal_quantiles)?,
            );
            let median_idx = self
                .quantiles
                .iter()
                .position(|q| (*q - 0.5).abs() < 1.0e-12)
                .unwrap_or(self.quantiles.len() / 2);
            let median = marginal_quantiles
                .iter()
                .map(|row| row[median_idx])
                .collect::<Vec<_>>();
            metrics.insert(
                "pinball_median".to_string(),
                pinball_loss(values, &median, self.quantiles[median_idx])?,
            );
            let lower = marginal_quantiles
                .iter()
                .map(|row| row[0])
                .collect::<Vec<_>>();
            let upper = marginal_quantiles
                .iter()
                .map(|row| row[self.quantiles.len() - 1])
                .collect::<Vec<_>>();
            metrics.insert(
                "interval_coverage".to_string(),
                interval_coverage(values, &lower, &upper)?,
            );
            metrics.insert(
                "interval_width".to_string(),
                mean_interval_width(&lower, &upper)?,
            );
            metrics.insert(
                "joint_path_calibration".to_string(),
                joint_path_calibration(values, &joint_scenario_paths),
            );
            metrics.insert(
                "tail_event_calibration".to_string(),
                tail_event_calibration(values, &upper),
            );
        }
        Ok(FlowPrediction {
            samples,
            log_likelihood,
            marginal_quantiles,
            joint_scenario_paths,
            tail_risk_metrics,
            metrics,
        })
    }
}

impl GeoTemporalDiffusionScenarioModel {
    pub fn new(scenario_count: usize, diffusion_steps: usize, shock_scale: f64) -> Result<Self> {
        if scenario_count == 0 {
            return invalid("scenario_count must be positive");
        }
        if diffusion_steps == 0 {
            return invalid("diffusion_steps must be positive");
        }
        if !shock_scale.is_finite() || shock_scale <= 0.0 {
            return invalid("shock_scale must be finite and positive");
        }
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "architecture".to_string(),
            "geo_temporal_diffusion_scenario".to_string(),
        );
        metadata.insert("capability_tier".to_string(), "experimental".to_string());
        metadata.insert("auto_geo_enabled".to_string(), "false".to_string());
        metadata.insert(
            "primary_benchmark_evidence".to_string(),
            "false".to_string(),
        );
        Ok(Self {
            scenario_count,
            diffusion_steps,
            shock_scale,
            capability_tier: "experimental".to_string(),
            metadata,
        })
    }

    pub fn generate(
        &self,
        point_forecast: &[Vec<f64>],
        edges: &[DiffusionEdge],
    ) -> Result<DiffusionScenarioPrediction> {
        validate_panel(point_forecast, "point_forecast")?;
        validate_edges(edges, point_forecast[0].len())?;
        let horizon = point_forecast.len();
        let nodes = point_forecast[0].len();
        let mut scenarios = Vec::with_capacity(self.scenario_count);
        for scenario_idx in 0..self.scenario_count {
            let mut scenario = point_forecast.to_vec();
            let mut residual_field = (0..horizon)
                .map(|t| {
                    (0..nodes)
                        .map(|node| {
                            self.shock_scale
                                * deterministic_standard_sample(
                                    scenario_idx * horizon + t,
                                    node + self.diffusion_steps,
                                )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            for _ in 0..self.diffusion_steps {
                residual_field = diffuse_residual_field(&residual_field, edges, nodes);
            }
            for t in 0..horizon {
                for node in 0..nodes {
                    scenario[t][node] += residual_field[t][node];
                }
            }
            scenarios.push(scenario);
        }
        let scenario_mean = scenario_panel_mean(&scenarios, horizon, nodes);
        let scenario_variance = scenario_panel_variance(&scenarios, &scenario_mean, horizon, nodes);
        let spatial_correlation = scenario_spatial_correlation(&scenario_mean, edges);
        let mut point_forecast_comparison = BTreeMap::new();
        point_forecast_comparison.insert(
            "mean_absolute_delta".to_string(),
            mean_abs_panel_delta(&scenario_mean, point_forecast),
        );
        point_forecast_comparison.insert(
            "mean_variance".to_string(),
            scenario_variance.iter().flatten().sum::<f64>() / (horizon * nodes) as f64,
        );
        let mut metadata = self.metadata.clone();
        metadata.insert(
            "scenario_count".to_string(),
            self.scenario_count.to_string(),
        );
        metadata.insert(
            "diffusion_steps".to_string(),
            self.diffusion_steps.to_string(),
        );
        metadata.insert("shock_scale".to_string(), self.shock_scale.to_string());
        Ok(DiffusionScenarioPrediction {
            scenarios,
            scenario_mean,
            scenario_variance,
            spatial_correlation,
            point_forecast_comparison,
            metadata,
        })
    }
}

