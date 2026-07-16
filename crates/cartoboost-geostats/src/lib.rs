use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum GeostatsError {
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("model is not fitted")]
    NotFitted,
    #[error("linear solve failed: {0}")]
    LinearSolve(String),
}

pub type Result<T> = std::result::Result<T, GeostatsError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CovarianceKernel {
    Exponential,
    SquaredExponential,
    Matern32,
    Matern52,
}

impl CovarianceKernel {
    pub fn parse(name: &str) -> Result<Self> {
        match name.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "exponential" | "exp" => Ok(Self::Exponential),
            "squaredexponential" | "gaussian" | "rbf" => Ok(Self::SquaredExponential),
            "matern32" | "matern3/2" | "matern1.5" => Ok(Self::Matern32),
            "matern52" | "matern5/2" | "matern2.5" => Ok(Self::Matern52),
            other => Err(GeostatsError::InvalidInput(format!(
                "unknown covariance kernel {other:?}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Exponential => "exponential",
            Self::SquaredExponential => "squared_exponential",
            Self::Matern32 => "matern_3_2",
            Self::Matern52 => "matern_5_2",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Anisotropy {
    pub angle_degrees: f64,
    pub scaling: f64,
}

impl Default for Anisotropy {
    fn default() -> Self {
        Self {
            angle_degrees: 0.0,
            scaling: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct NngpConfig {
    pub kernel: CovarianceKernel,
    pub range: f64,
    pub sill: f64,
    pub nugget: f64,
    pub anisotropy: Anisotropy,
    pub n_neighbors: usize,
    pub brute_force_threshold: usize,
    pub duplicate_tolerance: f64,
}

impl Default for NngpConfig {
    fn default() -> Self {
        Self {
            kernel: CovarianceKernel::Exponential,
            range: 1.0,
            sill: 1.0,
            nugget: 1.0e-6,
            anisotropy: Anisotropy::default(),
            n_neighbors: 16,
            brute_force_threshold: 2048,
            duplicate_tolerance: 0.0,
        }
    }
}

impl NngpConfig {
    pub fn validate(self) -> Result<Self> {
        if !self.range.is_finite() || self.range <= 0.0 {
            return Err(GeostatsError::InvalidInput(
                "range must be finite and positive".to_string(),
            ));
        }
        if !self.sill.is_finite() || self.sill <= 0.0 {
            return Err(GeostatsError::InvalidInput(
                "sill must be finite and positive".to_string(),
            ));
        }
        if !self.nugget.is_finite() || self.nugget < 0.0 {
            return Err(GeostatsError::InvalidInput(
                "nugget must be finite and nonnegative".to_string(),
            ));
        }
        if !self.anisotropy.angle_degrees.is_finite() {
            return Err(GeostatsError::InvalidInput(
                "anisotropy angle must be finite".to_string(),
            ));
        }
        if !self.anisotropy.scaling.is_finite() || self.anisotropy.scaling <= 0.0 {
            return Err(GeostatsError::InvalidInput(
                "anisotropy scaling must be finite and positive".to_string(),
            ));
        }
        if self.n_neighbors == 0 {
            return Err(GeostatsError::InvalidInput(
                "n_neighbors must be positive".to_string(),
            ));
        }
        if !self.duplicate_tolerance.is_finite() || self.duplicate_tolerance < 0.0 {
            return Err(GeostatsError::InvalidInput(
                "duplicate_tolerance must be finite and nonnegative".to_string(),
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NngpPrediction {
    pub mean: f64,
    pub variance: f64,
    pub neighbor_indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmpiricalVariogramBin {
    pub lag_start: f64,
    pub lag_end: f64,
    pub lag_center: f64,
    pub semivariance: f64,
    pub pair_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VariogramFit {
    pub kernel: CovarianceKernel,
    pub range: f64,
    pub sill: f64,
    pub nugget: f64,
    pub weighted_sse: f64,
}

#[derive(Clone, Debug)]
pub struct NearestNeighborGPRegressor {
    config: NngpConfig,
    coords: Vec<[f64; 2]>,
    search_coords: Vec<[f64; 2]>,
    values: Vec<f64>,
    neighbor_index: NeighborIndex,
    mean: f64,
    fitted: bool,
}

impl NearestNeighborGPRegressor {
    pub fn new(config: NngpConfig) -> Result<Self> {
        Ok(Self {
            config: config.validate()?,
            coords: Vec::new(),
            search_coords: Vec::new(),
            values: Vec::new(),
            neighbor_index: NeighborIndex::BruteForce,
            mean: 0.0,
            fitted: false,
        })
    }

    pub fn fit(&mut self, coords: &[[f64; 2]], y: &[f64]) -> Result<()> {
        if coords.len() != y.len() {
            return Err(GeostatsError::InvalidInput(
                "coords and y must have the same row count".to_string(),
            ));
        }
        if coords.is_empty() {
            return Err(GeostatsError::InvalidInput(
                "at least one observation is required".to_string(),
            ));
        }
        for (idx, (coord, value)) in coords.iter().zip(y).enumerate() {
            if !coord[0].is_finite() || !coord[1].is_finite() || !value.is_finite() {
                return Err(GeostatsError::InvalidInput(format!(
                    "non-finite coordinate or target at row {idx}"
                )));
            }
        }
        reject_duplicate_coords(coords, self.config.duplicate_tolerance)?;
        self.coords = coords.to_vec();
        self.search_coords = coords
            .iter()
            .map(|coord| transform_point(*coord, self.config))
            .collect();
        self.values = y.to_vec();
        self.neighbor_index =
            NeighborIndex::fit(&self.search_coords, self.config.brute_force_threshold);
        self.mean = y.iter().sum::<f64>() / y.len() as f64;
        self.fitted = true;
        Ok(())
    }

    pub fn predict(&self, coords: &[[f64; 2]]) -> Result<Vec<NngpPrediction>> {
        if !self.fitted {
            return Err(GeostatsError::NotFitted);
        }
        coords
            .iter()
            .map(|coord| self.predict_one(*coord))
            .collect()
    }

    pub fn config(&self) -> NngpConfig {
        self.config
    }

    pub fn neighbor_index_kind(&self) -> &'static str {
        self.neighbor_index.kind()
    }

    fn predict_one(&self, coord: [f64; 2]) -> Result<NngpPrediction> {
        if !coord[0].is_finite() || !coord[1].is_finite() {
            return Err(GeostatsError::InvalidInput(
                "prediction coordinates must be finite".to_string(),
            ));
        }
        let search_coord = transform_point(coord, self.config);
        let neighbors = self.neighbor_index.neighbors(
            &self.search_coords,
            search_coord,
            self.config.n_neighbors,
        );
        let n = neighbors.len();
        let mut k_nn = vec![vec![0.0; n]; n];
        let mut k_star = vec![0.0; n];
        let mut centered = vec![0.0; n];
        for (row_pos, &row_idx) in neighbors.iter().enumerate() {
            centered[row_pos] = self.values[row_idx] - self.mean;
            k_star[row_pos] = covariance(coord, self.coords[row_idx], self.config);
            for (col_pos, &col_idx) in neighbors.iter().enumerate() {
                let mut value = covariance(self.coords[row_idx], self.coords[col_idx], self.config);
                if row_pos == col_pos {
                    value += self.config.nugget;
                }
                k_nn[row_pos][col_pos] = value;
            }
        }
        let weights = solve_spd(k_nn, k_star.clone())?;
        let mean = self.mean + dot(&weights, &centered);
        if !mean.is_finite() || weights.iter().any(|weight| !weight.is_finite()) {
            return Err(GeostatsError::LinearSolve(
                "GP prediction produced non-finite weights or mean".to_string(),
            ));
        }
        let prior_variance = self.config.sill + self.config.nugget;
        let variance_reduction = checked_nonnegative(
            dot(&k_star, &weights),
            prior_variance,
            "GP variance reduction",
        )?;
        let variance = checked_nonnegative(
            prior_variance - variance_reduction,
            prior_variance.max(variance_reduction),
            "GP prediction variance",
        )?;
        Ok(NngpPrediction {
            mean,
            variance,
            neighbor_indices: neighbors,
        })
    }
}

// Cohesive implementation families share the crate namespace.
include!("geostatistics/variogram.rs");
include!("geostatistics/neighbors.rs");
include!("geostatistics/linear_algebra.rs");
include!("geostatistics/tests.rs");
