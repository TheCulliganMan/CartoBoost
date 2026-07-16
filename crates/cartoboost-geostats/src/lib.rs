use cartoboost_neural::{
    backend_pairwise_squared_distances_f32, select_backend_for, BackendOperation, BackendSelection,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const GEOSTATS_PAIRWISE_DISPATCH_MIN_PAIRS: usize = 16_384;

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
    backend: BackendSelection,
}

impl NearestNeighborGPRegressor {
    pub fn new(config: NngpConfig) -> Result<Self> {
        Self::new_with_backend(config, Some("cpu"))
    }

    pub fn new_with_backend(config: NngpConfig, backend: Option<&str>) -> Result<Self> {
        let backend = select_backend_for(backend, BackendOperation::PairwiseDistance)
            .map_err(|error| GeostatsError::InvalidInput(error.to_string()))?;
        Ok(Self {
            config: config.validate()?,
            coords: Vec::new(),
            search_coords: Vec::new(),
            values: Vec::new(),
            neighbor_index: NeighborIndex::BruteForce,
            mean: 0.0,
            fitted: false,
            backend,
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
        reject_duplicate_coords_with_backend(
            coords,
            self.config.duplicate_tolerance,
            &self.backend,
        )?;
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
        if self.backend.selected != "cpu"
            && matches!(self.neighbor_index, NeighborIndex::BruteForce)
            && coords.len().saturating_mul(self.coords.len())
                >= GEOSTATS_PAIRWISE_DISPATCH_MIN_PAIRS
        {
            if coords
                .iter()
                .flatten()
                .any(|coordinate| !coordinate.is_finite())
            {
                return Err(GeostatsError::InvalidInput(
                    "prediction coordinates must be finite".to_string(),
                ));
            }
            let queries = coords
                .iter()
                .map(|coord| {
                    transform_point(*coord, self.config)
                        .into_iter()
                        .map(|value| value as f32)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let observations = self
                .search_coords
                .iter()
                .map(|coord| coord.iter().map(|value| *value as f32).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let distances =
                backend_pairwise_squared_distances_f32(&self.backend, &queries, &observations)
                    .map_err(|error| GeostatsError::InvalidInput(error.to_string()))?;
            return coords
                .par_iter()
                .zip(distances.into_par_iter())
                .map(|(coord, row)| {
                    let mut ranked = row.into_iter().enumerate().collect::<Vec<_>>();
                    ranked.sort_by(|left, right| {
                        left.1
                            .total_cmp(&right.1)
                            .then_with(|| left.0.cmp(&right.0))
                    });
                    let neighbors = ranked
                        .into_iter()
                        .take(self.config.n_neighbors.min(self.coords.len()))
                        .map(|(index, _)| index)
                        .collect::<Vec<_>>();
                    self.predict_one_with_neighbors(*coord, neighbors)
                })
                .collect();
        }
        coords
            .par_iter()
            .map(|coord| self.predict_one(*coord))
            .collect()
    }

    /// Completes NNGP prediction from an accelerator-computed query-by-observation
    /// squared-distance matrix. Covariance solves remain small and run on CPU.
    pub fn predict_from_squared_distances(
        &self,
        coords: &[[f64; 2]],
        distances: &[Vec<f32>],
    ) -> Result<Vec<NngpPrediction>> {
        if !self.fitted {
            return Err(GeostatsError::NotFitted);
        }
        if distances.len() != coords.len()
            || distances.iter().any(|row| {
                row.len() != self.coords.len()
                    || row.iter().any(|value| !value.is_finite() || *value < 0.0)
            })
        {
            return Err(GeostatsError::InvalidInput(
                "distance matrix must be finite query-by-observation squared distances".to_string(),
            ));
        }
        coords
            .par_iter()
            .zip(distances.par_iter())
            .map(|(coord, row)| {
                let mut ranked = row.iter().copied().enumerate().collect::<Vec<_>>();
                ranked.sort_by(|left, right| {
                    left.1
                        .total_cmp(&right.1)
                        .then_with(|| left.0.cmp(&right.0))
                });
                let neighbors = ranked
                    .into_iter()
                    .take(self.config.n_neighbors.min(self.coords.len()))
                    .map(|(index, _)| index)
                    .collect();
                self.predict_one_with_neighbors(*coord, neighbors)
            })
            .collect()
    }

    pub fn transformed_points(&self, coords: &[[f64; 2]]) -> Result<Vec<Vec<f32>>> {
        if coords.iter().flatten().any(|value| !value.is_finite()) {
            return Err(GeostatsError::InvalidInput(
                "coordinates must be finite".to_string(),
            ));
        }
        Ok(coords
            .iter()
            .map(|coord| {
                transform_point(*coord, self.config)
                    .into_iter()
                    .map(|value| value as f32)
                    .collect()
            })
            .collect())
    }

    pub fn transformed_observations(&self) -> Result<Vec<Vec<f32>>> {
        if !self.fitted {
            return Err(GeostatsError::NotFitted);
        }
        Ok(self
            .search_coords
            .iter()
            .map(|coord| coord.iter().map(|value| *value as f32).collect())
            .collect())
    }

    pub fn config(&self) -> NngpConfig {
        self.config
    }

    pub fn neighbor_index_kind(&self) -> &'static str {
        self.neighbor_index.kind()
    }

    pub fn backend(&self) -> &BackendSelection {
        &self.backend
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
        self.predict_one_with_neighbors(coord, neighbors)
    }

    fn predict_one_with_neighbors(
        &self,
        coord: [f64; 2],
        neighbors: Vec<usize>,
    ) -> Result<NngpPrediction> {
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

pub fn covariance(left: [f64; 2], right: [f64; 2], config: NngpConfig) -> f64 {
    let h = transformed_distance(left, right, config);
    let r = h / config.range;
    let corr = match config.kernel {
        CovarianceKernel::Exponential => (-r).exp(),
        CovarianceKernel::SquaredExponential => (-(r * r)).exp(),
        CovarianceKernel::Matern32 => {
            let z = 3.0_f64.sqrt() * r;
            (1.0 + z) * (-z).exp()
        }
        CovarianceKernel::Matern52 => {
            let z = 5.0_f64.sqrt() * r;
            (1.0 + z + z * z / 3.0) * (-z).exp()
        }
    };
    config.sill * corr
}

pub fn empirical_semivariogram(
    coords: &[[f64; 2]],
    values: &[f64],
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy: Anisotropy,
) -> Result<Vec<EmpiricalVariogramBin>> {
    empirical_semivariogram_with_backend(
        coords,
        values,
        bin_count,
        max_distance,
        anisotropy,
        Some("cpu"),
    )
}

pub fn empirical_semivariogram_with_backend(
    coords: &[[f64; 2]],
    values: &[f64],
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy: Anisotropy,
    backend: Option<&str>,
) -> Result<Vec<EmpiricalVariogramBin>> {
    let backend = select_backend_for(backend.or(Some("cpu")), BackendOperation::PairwiseDistance)
        .map_err(|error| GeostatsError::InvalidInput(error.to_string()))?;
    if coords.len() != values.len() {
        return Err(GeostatsError::InvalidInput(
            "coords and values must have the same row count".to_string(),
        ));
    }
    if coords.len() < 2 || bin_count == 0 {
        return Err(GeostatsError::InvalidInput(
            "at least two observations and one bin are required".to_string(),
        ));
    }
    if let Some(max_distance) = max_distance {
        if !max_distance.is_finite() || max_distance <= 0.0 {
            return Err(GeostatsError::InvalidInput(
                "max variogram distance must be finite and positive".to_string(),
            ));
        }
    }
    for (idx, (coord, value)) in coords.iter().zip(values).enumerate() {
        if !coord[0].is_finite() || !coord[1].is_finite() {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram coordinates must be finite at row {idx}"
            )));
        }
        if !value.is_finite() {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram value must be finite at row {idx}"
            )));
        }
    }
    let distance_config = NngpConfig {
        anisotropy,
        ..NngpConfig::default()
    }
    .validate()?;
    let accelerated_distances = if backend.selected == "cpu"
        || coords.len().saturating_mul(coords.len()) < GEOSTATS_PAIRWISE_DISPATCH_MIN_PAIRS
    {
        None
    } else {
        let transformed = coords
            .iter()
            .map(|coord| {
                transform_point(*coord, distance_config)
                    .into_iter()
                    .map(|value| value as f32)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Some(
            backend_pairwise_squared_distances_f32(&backend, &transformed, &transformed)
                .map_err(|error| GeostatsError::InvalidInput(error.to_string()))?,
        )
    };
    let mut pairs = Vec::new();
    let mut observed_max: f64 = 0.0;
    for i in 0..coords.len() {
        for j in (i + 1)..coords.len() {
            let distance = accelerated_distances.as_ref().map_or_else(
                || transformed_distance(coords[i], coords[j], distance_config),
                |distances| f64::from(distances[i][j]).sqrt(),
            );
            if !distance.is_finite() {
                return Err(GeostatsError::InvalidInput(format!(
                    "variogram distance is not finite for rows {i} and {j}"
                )));
            }
            if max_distance.is_some_and(|max| distance > max) {
                continue;
            }
            let difference = values[i] - values[j];
            let semivariance = 0.5 * difference * difference;
            if !semivariance.is_finite() {
                return Err(GeostatsError::InvalidInput(format!(
                    "variogram semivariance is not finite for rows {i} and {j}"
                )));
            }
            observed_max = observed_max.max(distance);
            pairs.push((distance, semivariance));
        }
    }
    if pairs.is_empty() {
        return Err(GeostatsError::InvalidInput(
            "no coordinate pairs are available for variogram bins".to_string(),
        ));
    }
    let upper = max_distance.unwrap_or(observed_max);
    if upper <= 0.0 || !upper.is_finite() {
        return Err(GeostatsError::InvalidInput(
            "max variogram distance must be positive".to_string(),
        ));
    }
    let width = upper / bin_count as f64;
    if !width.is_finite() || width <= 0.0 {
        return Err(GeostatsError::InvalidInput(
            "variogram bin width must be finite and positive".to_string(),
        ));
    }
    let mut sums = vec![0.0; bin_count];
    let mut counts = vec![0usize; bin_count];
    for (distance, gamma) in pairs {
        let mut bin = (distance / width).floor() as usize;
        if bin >= bin_count {
            bin = bin_count - 1;
        }
        sums[bin] += gamma;
        counts[bin] += 1;
    }
    Ok((0..bin_count)
        .filter(|&bin| counts[bin] > 0)
        .map(|bin| {
            let lag_start = bin as f64 * width;
            let lag_end = lag_start + width;
            EmpiricalVariogramBin {
                lag_start,
                lag_end,
                lag_center: 0.5 * (lag_start + lag_end),
                semivariance: sums[bin] / counts[bin] as f64,
                pair_count: counts[bin],
            }
        })
        .collect())
}

pub fn binned_variogram(
    coords: &[[f64; 2]],
    values: &[f64],
    bin_count: usize,
    max_distance: Option<f64>,
    anisotropy: Anisotropy,
) -> Result<Vec<EmpiricalVariogramBin>> {
    empirical_semivariogram(coords, values, bin_count, max_distance, anisotropy)
}

pub fn fit_variogram_wls(
    bins: &[EmpiricalVariogramBin],
    kernels: &[CovarianceKernel],
    range_candidates: &[f64],
    sill_candidates: &[f64],
    nugget_candidates: &[f64],
) -> Result<VariogramFit> {
    if bins.is_empty()
        || range_candidates.is_empty()
        || sill_candidates.is_empty()
        || nugget_candidates.is_empty()
    {
        return Err(GeostatsError::InvalidInput(
            "variogram fitting requires bins and nonempty candidate grids".to_string(),
        ));
    }
    validate_variogram_bins(bins)?;
    validate_positive_candidates(range_candidates, "range candidates")?;
    validate_positive_candidates(sill_candidates, "sill candidates")?;
    validate_nonnegative_candidates(nugget_candidates, "nugget candidates")?;
    let kernels = if kernels.is_empty() {
        vec![CovarianceKernel::Exponential]
    } else {
        kernels.to_vec()
    };
    let per_kernel = range_candidates.len() * sill_candidates.len() * nugget_candidates.len();
    let per_range = sill_candidates.len() * nugget_candidates.len();
    let candidate_count = kernels.len() * per_kernel;
    let candidates = (0..candidate_count)
        .into_par_iter()
        .map(|index| {
            let kernel = kernels[index / per_kernel];
            let within_kernel = index % per_kernel;
            let range = range_candidates[within_kernel / per_range];
            let within_range = within_kernel % per_range;
            let sill = sill_candidates[within_range / nugget_candidates.len()];
            let nugget = nugget_candidates[within_range % nugget_candidates.len()];
            evaluate_variogram_candidate(bins, kernel, range, sill, nugget)
                .map(|candidate| (index, candidate))
        })
        .collect::<Result<Vec<_>>>()?;
    candidates
        .into_iter()
        .min_by(|(left_index, left), (right_index, right)| {
            left.weighted_sse
                .total_cmp(&right.weighted_sse)
                .then_with(|| left_index.cmp(right_index))
        })
        .map(|(_, candidate)| candidate)
        .ok_or_else(|| {
            GeostatsError::InvalidInput("no valid variogram candidates were supplied".to_string())
        })
}

fn evaluate_variogram_candidate(
    bins: &[EmpiricalVariogramBin],
    kernel: CovarianceKernel,
    range: f64,
    sill: f64,
    nugget: f64,
) -> Result<VariogramFit> {
    let config = NngpConfig {
        kernel,
        range,
        sill,
        nugget,
        ..NngpConfig::default()
    }
    .validate()?;
    let mut weighted_sse = 0.0;
    for bin in bins {
        let model = nugget
            + sill
                * (1.0
                    - covariance(
                        [0.0, 0.0],
                        [bin.lag_center, 0.0],
                        NngpConfig {
                            nugget: 0.0,
                            ..config
                        },
                    ) / sill);
        if !model.is_finite() || model < 0.0 {
            return Err(GeostatsError::InvalidInput(
                "variogram candidate produced an invalid semivariance".to_string(),
            ));
        }
        let residual = bin.semivariance - model;
        let contribution = bin.pair_count as f64 * residual * residual;
        if !contribution.is_finite() || contribution < 0.0 {
            return Err(GeostatsError::InvalidInput(
                "variogram candidate produced a non-finite weighted error".to_string(),
            ));
        }
        weighted_sse += contribution;
        if !weighted_sse.is_finite() {
            return Err(GeostatsError::InvalidInput(
                "variogram weighted SSE is not finite".to_string(),
            ));
        }
    }
    Ok(VariogramFit {
        kernel,
        range,
        sill,
        nugget,
        weighted_sse,
    })
}

fn validate_variogram_bins(bins: &[EmpiricalVariogramBin]) -> Result<()> {
    for (idx, bin) in bins.iter().enumerate() {
        if !bin.lag_start.is_finite() || bin.lag_start < 0.0 {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} lag_start must be finite and nonnegative"
            )));
        }
        if !bin.lag_end.is_finite() || bin.lag_end <= bin.lag_start {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} lag_end must be finite and greater than lag_start"
            )));
        }
        if !bin.lag_center.is_finite()
            || bin.lag_center < bin.lag_start
            || bin.lag_center > bin.lag_end
        {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} lag_center must be finite and within its lag bounds"
            )));
        }
        if !bin.semivariance.is_finite() || bin.semivariance < 0.0 {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} semivariance must be finite and nonnegative"
            )));
        }
        if bin.pair_count == 0 {
            return Err(GeostatsError::InvalidInput(format!(
                "variogram bin {idx} pair_count must be positive"
            )));
        }
    }
    Ok(())
}

fn validate_positive_candidates(values: &[f64], name: &str) -> Result<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(GeostatsError::InvalidInput(format!(
            "{name} must contain only finite positive values"
        )));
    }
    Ok(())
}

fn validate_nonnegative_candidates(values: &[f64], name: &str) -> Result<()> {
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err(GeostatsError::InvalidInput(format!(
            "{name} must contain only finite nonnegative values"
        )));
    }
    Ok(())
}

pub fn deterministic_neighbors(coords: &[[f64; 2]], target: [f64; 2], k: usize) -> Vec<usize> {
    let mut distances = coords
        .iter()
        .enumerate()
        .map(|(idx, coord)| {
            let dx = coord[0] - target[0];
            let dy = coord[1] - target[1];
            (idx, dx * dx + dy * dy)
        })
        .collect::<Vec<_>>();
    distances.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    distances.into_iter().take(k).map(|(idx, _)| idx).collect()
}

#[derive(Clone, Debug)]
enum NeighborIndex {
    BruteForce,
    KdTree { root: Option<Box<KdNode>> },
}

impl NeighborIndex {
    fn fit(coords: &[[f64; 2]], brute_force_threshold: usize) -> Self {
        if coords.len() <= brute_force_threshold {
            Self::BruteForce
        } else {
            let mut indices = (0..coords.len()).collect::<Vec<_>>();
            Self::KdTree {
                root: KdNode::build(coords, &mut indices, 0),
            }
        }
    }

    fn neighbors(&self, coords: &[[f64; 2]], target: [f64; 2], k: usize) -> Vec<usize> {
        match self {
            Self::BruteForce => deterministic_neighbors(coords, target, k),
            Self::KdTree { root } => {
                let mut heap = NeighborHeap::new(k);
                if let Some(root) = root {
                    root.search(coords, target, &mut heap);
                }
                heap.sorted_indices()
            }
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::BruteForce => "brute_force",
            Self::KdTree { .. } => "kd_tree",
        }
    }
}

#[derive(Clone, Debug)]
struct KdNode {
    index: usize,
    axis: usize,
    left: Option<Box<KdNode>>,
    right: Option<Box<KdNode>>,
}

impl KdNode {
    fn build(coords: &[[f64; 2]], indices: &mut [usize], depth: usize) -> Option<Box<Self>> {
        if indices.is_empty() {
            return None;
        }
        let axis = depth % 2;
        indices.sort_by(|left, right| {
            coords[*left][axis]
                .total_cmp(&coords[*right][axis])
                .then_with(|| left.cmp(right))
        });
        let median = indices.len() / 2;
        let (left_indices, rest) = indices.split_at_mut(median);
        let (median_indices, right_indices) = rest.split_at_mut(1);
        Some(Box::new(Self {
            index: median_indices[0],
            axis,
            left: Self::build(coords, left_indices, depth + 1),
            right: Self::build(coords, right_indices, depth + 1),
        }))
    }

    fn search(&self, coords: &[[f64; 2]], target: [f64; 2], heap: &mut NeighborHeap) {
        let coord = coords[self.index];
        heap.push(self.index, squared_distance(coord, target));

        let delta = target[self.axis] - coord[self.axis];
        let (near, far) = if delta <= 0.0 {
            (&self.left, &self.right)
        } else {
            (&self.right, &self.left)
        };
        if let Some(node) = near {
            node.search(coords, target, heap);
        }
        if heap.should_visit(delta * delta) {
            if let Some(node) = far {
                node.search(coords, target, heap);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NeighborCandidate {
    index: usize,
    distance_sq: f64,
}

#[derive(Clone, Debug)]
struct NeighborHeap {
    k: usize,
    candidates: Vec<NeighborCandidate>,
}

impl NeighborHeap {
    fn new(k: usize) -> Self {
        Self {
            k,
            candidates: Vec::with_capacity(k),
        }
    }

    fn push(&mut self, index: usize, distance_sq: f64) {
        let candidate = NeighborCandidate { index, distance_sq };
        if self.candidates.len() < self.k {
            self.candidates.push(candidate);
            return;
        }
        if let Some((worst_pos, worst)) = self.worst_candidate() {
            if candidate_is_better(candidate, worst) {
                self.candidates[worst_pos] = candidate;
            }
        }
    }

    fn should_visit(&self, axis_distance_sq: f64) -> bool {
        self.candidates.len() < self.k
            || self
                .worst_candidate()
                .is_some_and(|(_, worst)| axis_distance_sq <= worst.distance_sq)
    }

    fn sorted_indices(mut self) -> Vec<usize> {
        self.candidates.sort_by(compare_candidates);
        self.candidates
            .into_iter()
            .map(|candidate| candidate.index)
            .collect()
    }

    fn worst_candidate(&self) -> Option<(usize, NeighborCandidate)> {
        self.candidates
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, left), (_, right)| compare_candidates(left, right))
    }
}

fn candidate_is_better(left: NeighborCandidate, right: NeighborCandidate) -> bool {
    compare_candidates(&left, &right).is_lt()
}

fn compare_candidates(left: &NeighborCandidate, right: &NeighborCandidate) -> std::cmp::Ordering {
    left.distance_sq
        .total_cmp(&right.distance_sq)
        .then_with(|| left.index.cmp(&right.index))
}

fn squared_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
    cartoboost_geo_core::squared_euclidean_distance(left, right)
}

fn transformed_distance(left: [f64; 2], right: [f64; 2], config: NngpConfig) -> f64 {
    cartoboost_geo_core::anisotropic_euclidean_distance(
        left,
        right,
        config.anisotropy.angle_degrees,
        config.anisotropy.scaling,
    )
}

fn transform_point(point: [f64; 2], config: NngpConfig) -> [f64; 2] {
    cartoboost_geo_core::transform_anisotropic_point(
        point,
        config.anisotropy.angle_degrees,
        config.anisotropy.scaling,
    )
}

fn reject_duplicate_coords_with_backend(
    coords: &[[f64; 2]],
    tolerance: f64,
    backend: &BackendSelection,
) -> Result<()> {
    if tolerance == 0.0 {
        let mut seen = std::collections::BTreeMap::<(u64, u64), usize>::new();
        for (index, coord) in coords.iter().enumerate() {
            let key = (
                if coord[0] == 0.0 {
                    0
                } else {
                    coord[0].to_bits()
                },
                if coord[1] == 0.0 {
                    0
                } else {
                    coord[1].to_bits()
                },
            );
            if let Some(previous) = seen.insert(key, index) {
                return Err(GeostatsError::InvalidInput(format!(
                    "duplicate coordinates at rows {previous} and {index}; jitter or aggregate duplicates before fitting"
                )));
            }
        }
        return Ok(());
    }
    if backend.selected != "cpu"
        && coords.len().saturating_mul(coords.len()) >= GEOSTATS_PAIRWISE_DISPATCH_MIN_PAIRS
    {
        let rows = coords
            .iter()
            .map(|coord| vec![coord[0] as f32, coord[1] as f32])
            .collect::<Vec<_>>();
        let distances = backend_pairwise_squared_distances_f32(backend, &rows, &rows)
            .map_err(|error| GeostatsError::InvalidInput(error.to_string()))?;
        let tolerance_squared = tolerance * tolerance;
        for (i, row) in distances.iter().enumerate().take(coords.len()) {
            for (j, distance) in row.iter().enumerate().take(coords.len()).skip(i + 1) {
                if f64::from(*distance) <= tolerance_squared {
                    return Err(GeostatsError::InvalidInput(format!("duplicate coordinates at rows {i} and {j}; jitter or aggregate duplicates before fitting")));
                }
            }
        }
        return Ok(());
    }
    for i in 0..coords.len() {
        for j in (i + 1)..coords.len() {
            if cartoboost_geo_core::euclidean_distance(coords[i], coords[j]) <= tolerance {
                return Err(GeostatsError::InvalidInput(format!("duplicate coordinates at rows {i} and {j}; jitter or aggregate duplicates before fitting")));
            }
        }
    }
    Ok(())
}

fn solve_spd(mut a: Vec<Vec<f64>>, b: Vec<f64>) -> Result<Vec<f64>> {
    let mut jitter = 0.0;
    for attempt in 0..5 {
        if attempt > 0 {
            jitter = 10_f64.powi(attempt - 12);
            for (idx, row) in a.iter_mut().enumerate() {
                row[idx] += jitter;
            }
        }
        if let Some(chol) = cholesky(&a) {
            let y = forward_substitution(&chol, &b);
            return Ok(back_substitution_transpose(&chol, &y));
        }
    }
    Err(GeostatsError::LinearSolve(format!(
        "covariance matrix is not positive definite after jitter {jitter:e}"
    )))
}

fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum = (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                let value = a[i][i] - sum;
                if value <= 0.0 || !value.is_finite() {
                    return None;
                }
                l[i][j] = value.sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    Some(l)
}

fn forward_substitution(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let sum = (0..i).map(|j| l[i][j] * y[j]).sum::<f64>();
        y[i] = (b[i] - sum) / l[i][i];
    }
    y
}

fn back_substitution_transpose(l: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = l.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let sum = ((i + 1)..n).map(|j| l[j][i] * x[j]).sum::<f64>();
        x[i] = (y[i] - sum) / l[i][i];
    }
    x
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn checked_nonnegative(value: f64, scale: f64, label: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(GeostatsError::LinearSolve(format!("{label} is not finite")));
    }
    let tolerance = 1.0e-10 * scale.abs().max(f64::MIN_POSITIVE);
    if value < -tolerance {
        return Err(GeostatsError::LinearSolve(format!(
            "{label} is materially negative ({value:e}, tolerance {tolerance:e})"
        )));
    }
    Ok(value.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_have_unit_sill_at_zero_distance() {
        for kernel in [
            CovarianceKernel::Exponential,
            CovarianceKernel::SquaredExponential,
            CovarianceKernel::Matern32,
            CovarianceKernel::Matern52,
        ] {
            let config = NngpConfig {
                kernel,
                sill: 2.5,
                ..NngpConfig::default()
            };
            assert!((covariance([0.0, 0.0], [0.0, 0.0], config) - 2.5).abs() < 1.0e-12);
        }
    }

    #[test]
    fn synthetic_field_recovery_and_variance_shape() {
        let coords = (0..40)
            .map(|i| {
                let x = i as f64 / 4.0;
                [x, (x * 0.7).sin()]
            })
            .collect::<Vec<_>>();
        let y = coords
            .iter()
            .map(|coord| (coord[0] * 0.4).sin() + 0.5 * coord[1])
            .collect::<Vec<_>>();
        let config = NngpConfig {
            kernel: CovarianceKernel::Matern32,
            range: 2.0,
            sill: 1.0,
            nugget: 1.0e-6,
            n_neighbors: 12,
            ..NngpConfig::default()
        };
        let mut model = NearestNeighborGPRegressor::new(config).expect("model");
        model.fit(&coords, &y).expect("fit");
        let train_pred = model.predict(&[coords[10]]).expect("train pred").remove(0);
        let far_pred = model.predict(&[[50.0, 50.0]]).expect("far pred").remove(0);
        assert!((train_pred.mean - y[10]).abs() < 1.0e-3);
        assert!(train_pred.variance >= 0.0);
        assert!(far_pred.variance >= train_pred.variance);
    }

    #[test]
    fn duplicate_coordinate_policy_is_explicit() {
        let mut model = NearestNeighborGPRegressor::new(NngpConfig::default()).expect("model");
        let err = model.fit(&[[0.0, 0.0], [0.0, 0.0]], &[1.0, 2.0]);
        assert!(err.is_err());
    }

    #[test]
    fn duplicate_tolerance_fit_runs_on_every_available_backend() {
        let config = NngpConfig {
            duplicate_tolerance: 0.01,
            ..NngpConfig::default()
        };
        for backend in cartoboost_neural::available_backends() {
            let mut valid =
                NearestNeighborGPRegressor::new_with_backend(config, Some(&backend)).unwrap();
            valid
                .fit(&[[0.0, 0.0], [0.5, 0.2], [1.0, -0.1]], &[1.0, 2.0, 3.0])
                .unwrap_or_else(|error| panic!("{backend} valid fit failed: {error}"));
            assert_eq!(valid.backend().selected, backend);

            let mut duplicate =
                NearestNeighborGPRegressor::new_with_backend(config, Some(&backend)).unwrap();
            let error = duplicate
                .fit(&[[0.0, 0.0], [0.005, 0.0]], &[1.0, 2.0])
                .expect_err("near duplicate must be rejected");
            assert!(error.to_string().contains("duplicate coordinates"));
        }
    }

    #[test]
    fn kd_tree_neighbors_match_brute_force_ordering() {
        let coords = (0..80)
            .map(|idx| {
                let x = (idx % 11) as f64 * 0.13 + (idx / 11) as f64 * 0.001;
                let y = (idx / 11) as f64 * 0.17 + (idx % 7) as f64 * 0.002;
                [x, y]
            })
            .collect::<Vec<_>>();
        let y = coords
            .iter()
            .map(|coord| coord[0].sin() + coord[1].cos())
            .collect::<Vec<_>>();
        let config = NngpConfig {
            n_neighbors: 9,
            brute_force_threshold: 8,
            anisotropy: Anisotropy {
                angle_degrees: 25.0,
                scaling: 1.7,
            },
            ..NngpConfig::default()
        };
        let mut model = NearestNeighborGPRegressor::new(config).expect("model");
        model.fit(&coords, &y).expect("fit");
        assert_eq!(model.neighbor_index_kind(), "kd_tree");
        let target = [0.57, 0.43];
        let transformed_coords = coords
            .iter()
            .map(|coord| transform_point(*coord, config))
            .collect::<Vec<_>>();
        let expected =
            deterministic_neighbors(&transformed_coords, transform_point(target, config), 9);
        let prediction = model.predict(&[target]).expect("predict").remove(0);
        assert_eq!(prediction.neighbor_indices, expected);
    }

    #[test]
    fn accelerator_distance_boundary_matches_regular_prediction() {
        let coords = (0..24)
            .map(|idx| [idx as f64 * 0.17, (idx as f64 * 0.31).sin()])
            .collect::<Vec<_>>();
        let targets = coords
            .iter()
            .map(|coord| coord[0].cos() + 0.25 * coord[1])
            .collect::<Vec<_>>();
        let config = NngpConfig {
            n_neighbors: 7,
            anisotropy: Anisotropy {
                angle_degrees: 31.0,
                scaling: 1.4,
            },
            ..NngpConfig::default()
        };
        let mut model = NearestNeighborGPRegressor::new(config).expect("model");
        model.fit(&coords, &targets).expect("fit");
        let queries = [[0.41, -0.2], [1.73, 0.8], [3.2, -0.4]];
        let expected = model.predict(&queries).expect("regular prediction");
        let query_rows = model.transformed_points(&queries).expect("queries");
        let observation_rows = model.transformed_observations().expect("observations");
        let distances = query_rows
            .iter()
            .map(|query| {
                observation_rows
                    .iter()
                    .map(|observation| {
                        query
                            .iter()
                            .zip(observation)
                            .map(|(left, right)| (left - right).powi(2))
                            .sum()
                    })
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<_>>();
        let accelerated = model
            .predict_from_squared_distances(&queries, &distances)
            .expect("accelerated prediction");
        for (expected, accelerated) in expected.iter().zip(&accelerated) {
            assert_eq!(expected.neighbor_indices, accelerated.neighbor_indices);
            assert!((expected.mean - accelerated.mean).abs() < 1.0e-12);
            assert!((expected.variance - accelerated.variance).abs() < 1.0e-12);
        }
    }

    #[test]
    fn variogram_fit_selects_candidate() {
        let coords = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]];
        let y = [0.0, 1.0, 1.5, 1.75];
        let bins =
            empirical_semivariogram(&coords, &y, 3, None, Anisotropy::default()).expect("bins");
        let fit = fit_variogram_wls(
            &bins,
            &[CovarianceKernel::Exponential, CovarianceKernel::Matern32],
            &[1.0, 2.0],
            &[0.5, 1.0],
            &[0.0, 0.05],
        )
        .expect("fit");
        assert!(fit.weighted_sse.is_finite());
    }

    #[test]
    fn empirical_variogram_runs_on_every_available_backend() {
        let coords = [[0.0, 0.0], [0.6, 0.2], [1.4, -0.1], [2.2, 0.4], [3.0, 0.0]];
        let values = [0.0, 1.0, 1.4, 1.8, 2.1];
        let anisotropy = Anisotropy {
            angle_degrees: 23.0,
            scaling: 1.3,
        };
        let expected = empirical_semivariogram_with_backend(
            &coords,
            &values,
            2,
            Some(10.0),
            anisotropy,
            Some("cpu"),
        )
        .unwrap();
        for backend in cartoboost_neural::available_backends() {
            let actual = empirical_semivariogram_with_backend(
                &coords,
                &values,
                2,
                Some(10.0),
                anisotropy,
                Some(&backend),
            )
            .unwrap_or_else(|error| panic!("{backend} variogram failed: {error}"));
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.pair_count, expected.pair_count);
                assert!((actual.semivariance - expected.semivariance).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn config_rejects_non_finite_anisotropy_angle() {
        let result = NngpConfig {
            anisotropy: Anisotropy {
                angle_degrees: f64::NAN,
                scaling: 1.0,
            },
            ..NngpConfig::default()
        }
        .validate();

        assert!(matches!(result, Err(GeostatsError::InvalidInput(_))));
    }

    #[test]
    fn empirical_variogram_rejects_invalid_numeric_inputs() {
        let valid_coords = [[0.0, 0.0], [1.0, 0.0]];
        let valid_values = [1.0, 2.0];

        assert!(empirical_semivariogram(
            &[[f64::NAN, 0.0], [1.0, 0.0]],
            &valid_values,
            2,
            None,
            Anisotropy::default(),
        )
        .is_err());
        assert!(empirical_semivariogram(
            &valid_coords,
            &[1.0, f64::INFINITY],
            2,
            None,
            Anisotropy::default(),
        )
        .is_err());
        assert!(empirical_semivariogram(
            &valid_coords,
            &valid_values,
            2,
            Some(-1.0),
            Anisotropy::default(),
        )
        .is_err());
        assert!(empirical_semivariogram(
            &valid_coords,
            &valid_values,
            2,
            None,
            Anisotropy {
                angle_degrees: f64::NAN,
                scaling: 1.0,
            },
        )
        .is_err());
        assert!(empirical_semivariogram(
            &[[f64::MAX, 0.0], [-f64::MAX, 0.0]],
            &valid_values,
            2,
            None,
            Anisotropy::default(),
        )
        .is_err());
    }

    #[test]
    fn empirical_variogram_keeps_collocated_pairs_as_nugget_evidence() {
        let bins = empirical_semivariogram(
            &[[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            &[0.0, 2.0, 1.0],
            2,
            None,
            Anisotropy::default(),
        )
        .expect("variogram");

        assert_eq!(bins.iter().map(|bin| bin.pair_count).sum::<usize>(), 3);
        assert!(bins
            .iter()
            .any(|bin| bin.lag_start == 0.0 && bin.semivariance >= 2.0));
    }

    #[test]
    fn variogram_fit_rejects_invalid_bins_candidates_and_objectives() {
        let valid_bin = EmpiricalVariogramBin {
            lag_start: 0.0,
            lag_end: 1.0,
            lag_center: 0.5,
            semivariance: 1.0,
            pair_count: 2,
        };
        let fit =
            |bins: &[EmpiricalVariogramBin], ranges: &[f64], sills: &[f64], nuggets: &[f64]| {
                fit_variogram_wls(
                    bins,
                    &[CovarianceKernel::Exponential],
                    ranges,
                    sills,
                    nuggets,
                )
            };

        assert!(fit(
            &[EmpiricalVariogramBin {
                semivariance: f64::NAN,
                ..valid_bin.clone()
            }],
            &[1.0],
            &[1.0],
            &[0.0],
        )
        .is_err());
        assert!(fit(
            &[EmpiricalVariogramBin {
                pair_count: 0,
                ..valid_bin.clone()
            }],
            &[1.0],
            &[1.0],
            &[0.0],
        )
        .is_err());
        assert!(fit(
            &[EmpiricalVariogramBin {
                lag_center: 2.0,
                ..valid_bin.clone()
            }],
            &[1.0],
            &[1.0],
            &[0.0],
        )
        .is_err());
        assert!(fit(
            std::slice::from_ref(&valid_bin),
            &[f64::NAN],
            &[1.0],
            &[0.0]
        )
        .is_err());
        assert!(fit(std::slice::from_ref(&valid_bin), &[1.0], &[0.0], &[0.0]).is_err());
        assert!(fit(std::slice::from_ref(&valid_bin), &[1.0], &[1.0], &[-1.0]).is_err());
        assert!(fit(
            &[EmpiricalVariogramBin {
                semivariance: f64::MAX,
                ..valid_bin
            }],
            &[1.0],
            &[1.0],
            &[0.0],
        )
        .is_err());
    }

    #[test]
    fn variance_validation_only_clamps_roundoff_scale_negatives() {
        assert_eq!(
            checked_nonnegative(-1.0e-12, 1.0, "variance").expect("tiny roundoff"),
            0.0
        );
        assert!(checked_nonnegative(-1.0e-4, 1.0, "variance").is_err());
        assert!(checked_nonnegative(f64::NAN, 1.0, "variance").is_err());
    }
}
