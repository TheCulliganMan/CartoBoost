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
    let mut pairs = Vec::new();
    let mut observed_max: f64 = 0.0;
    for i in 0..coords.len() {
        for j in (i + 1)..coords.len() {
            let distance = transformed_distance(coords[i], coords[j], distance_config);
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
    let mut best: Option<VariogramFit> = None;
    for &kernel in &kernels {
        for &range in range_candidates {
            for &sill in sill_candidates {
                for &nugget in nugget_candidates {
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
                                "variogram candidate produced a non-finite weighted error"
                                    .to_string(),
                            ));
                        }
                        weighted_sse += contribution;
                        if !weighted_sse.is_finite() {
                            return Err(GeostatsError::InvalidInput(
                                "variogram weighted SSE is not finite".to_string(),
                            ));
                        }
                    }
                    let candidate = VariogramFit {
                        kernel,
                        range,
                        sill,
                        nugget,
                        weighted_sse,
                    };
                    if best
                        .as_ref()
                        .is_none_or(|current| candidate.weighted_sse < current.weighted_sse)
                    {
                        best = Some(candidate);
                    }
                }
            }
        }
    }
    best.ok_or_else(|| {
        GeostatsError::InvalidInput("no valid variogram candidates were supplied".to_string())
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

fn reject_duplicate_coords(coords: &[[f64; 2]], tolerance: f64) -> Result<()> {
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
