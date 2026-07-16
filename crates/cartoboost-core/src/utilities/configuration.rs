impl LocalLinearKalmanConfig {
    pub fn new(
        level_process_variance: f64,
        trend_process_variance: f64,
        observation_variance: f64,
    ) -> Result<Self> {
        Self {
            level_process_variance,
            trend_process_variance,
            observation_variance,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self> {
        validate_positive_finite(self.level_process_variance, "level_process_variance")?;
        validate_positive_finite(self.trend_process_variance, "trend_process_variance")?;
        validate_positive_finite(self.observation_variance, "observation_variance")?;
        Ok(self)
    }
}

impl LocalLevelKalmanConfig {
    pub fn new(level_process_variance: f64, observation_variance: f64) -> Result<Self> {
        Self {
            level_process_variance,
            observation_variance,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self> {
        validate_positive_finite(self.level_process_variance, "level_process_variance")?;
        validate_positive_finite(self.observation_variance, "observation_variance")?;
        Ok(self)
    }
}

impl OrdinaryKrigingConfig {
    pub fn new(range: f64, nugget: f64) -> Result<Self> {
        Self {
            range,
            nugget,
            sill: 1.0,
            variogram_model: KrigingVariogramModel::Exponential,
            drift: KrigingDrift::Ordinary,
            anisotropy_angle_degrees: 0.0,
            anisotropy_scaling: 1.0,
            max_neighbors: None,
            min_neighbors: 1,
            max_distance: None,
        }
        .validate()
    }

    pub fn validate(self) -> Result<Self> {
        validate_positive_finite(self.range, "range")?;
        if !self.nugget.is_finite() || self.nugget < 0.0 {
            return Err(CartoBoostError::InvalidInput(
                "nugget must be finite and non-negative".to_string(),
            ));
        }
        validate_positive_finite(self.sill, "sill")?;
        if !self.anisotropy_angle_degrees.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "anisotropy_angle_degrees must be finite".to_string(),
            ));
        }
        validate_positive_finite(self.anisotropy_scaling, "anisotropy_scaling")?;
        if self.min_neighbors == 0 {
            return Err(CartoBoostError::InvalidInput(
                "min_neighbors must be positive".to_string(),
            ));
        }
        if let Some(max_neighbors) = self.max_neighbors {
            if max_neighbors == 0 {
                return Err(CartoBoostError::InvalidInput(
                    "max_neighbors must be positive when provided".to_string(),
                ));
            }
            if self.min_neighbors > max_neighbors {
                return Err(CartoBoostError::InvalidInput(
                    "min_neighbors must be <= max_neighbors".to_string(),
                ));
            }
            let drift_terms = drift_term_count(self.drift);
            if max_neighbors < drift_terms {
                return Err(CartoBoostError::InvalidInput(format!(
                    "max_neighbors must be at least {drift_terms} for {:?} kriging drift",
                    self.drift
                )));
            }
        }
        if let Some(max_distance) = self.max_distance {
            validate_positive_finite(max_distance, "max_distance")?;
        }
        Ok(self)
    }

    pub fn with_sill(mut self, sill: f64) -> Result<Self> {
        self.sill = sill;
        self.validate()
    }

    pub fn with_variogram_model(mut self, variogram_model: KrigingVariogramModel) -> Self {
        self.variogram_model = variogram_model;
        self
    }

    pub fn with_drift(mut self, drift: KrigingDrift) -> Self {
        self.drift = drift;
        self
    }

    pub fn with_anisotropy(mut self, angle_degrees: f64, scaling: f64) -> Result<Self> {
        if !angle_degrees.is_finite() {
            return Err(CartoBoostError::InvalidInput(
                "anisotropy_angle_degrees must be finite".to_string(),
            ));
        }
        validate_positive_finite(scaling, "anisotropy_scaling")?;
        self.anisotropy_angle_degrees = angle_degrees;
        self.anisotropy_scaling = scaling;
        self.validate()
    }

    pub fn with_neighbor_limits(
        mut self,
        max_neighbors: Option<usize>,
        min_neighbors: usize,
        max_distance: Option<f64>,
    ) -> Result<Self> {
        self.max_neighbors = max_neighbors;
        self.min_neighbors = min_neighbors;
        self.max_distance = max_distance;
        self.validate()
    }
}

impl OrdinaryKrigingSystem {
    pub fn new(observations: &[KrigingObservation], config: OrdinaryKrigingConfig) -> Result<Self> {
        let config = config.validate()?;
        validate_kriging_observations(observations)?;
        if uses_local_neighbors(config) {
            return Err(CartoBoostError::InvalidInput(
                "OrdinaryKrigingSystem requires all-neighbor config; use ordinary_kriging_predict_many for max_neighbors or max_distance".to_string(),
            ));
        }
        if observations.len() < config.min_neighbors {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging found {} neighbors, but min_neighbors is {}",
                observations.len(),
                config.min_neighbors
            )));
        }
        let drift_terms = drift_term_count(config.drift);
        if observations.len() < drift_terms {
            return Err(CartoBoostError::InvalidInput(format!(
                "kriging drift {:?} requires at least {drift_terms} observations; got {}",
                config.drift,
                observations.len()
            )));
        }
        let matrix = build_kriging_system_matrix(observations, config);
        let factorization = LinearSystemFactorization::factor(matrix).ok_or_else(|| {
            CartoBoostError::InvalidInput(
                "kriging system is singular or numerically ill-conditioned; adjust coordinates, variogram scale, or nugget".to_string(),
            )
        })?;
        Ok(Self {
            observations: observations.to_vec(),
            config,
            factorization,
            drift_terms,
        })
    }

    pub fn predict(&self, target: (f64, f64)) -> Result<KrigingPrediction> {
        validate_kriging_target(target)?;
        let rhs = build_kriging_rhs(&self.observations, target, self.config);
        let solution = self.factorization.solve(&rhs).ok_or_else(|| {
            CartoBoostError::InvalidInput("kriging solve produced a non-finite result".to_string())
        })?;
        kriging_prediction_from_solution(
            &self.observations,
            target,
            self.config,
            &rhs,
            &solution,
            (0..self.observations.len()).collect(),
        )
    }

    pub fn predict_many(&self, targets: &[(f64, f64)]) -> Result<Vec<KrigingPrediction>> {
        if targets.is_empty() {
            return Err(CartoBoostError::InvalidInput(
                "kriging targets must not be empty".to_string(),
            ));
        }
        targets
            .par_iter()
            .map(|target| self.predict(*target))
            .collect()
    }

    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    pub fn drift_terms(&self) -> usize {
        self.drift_terms
    }
}

