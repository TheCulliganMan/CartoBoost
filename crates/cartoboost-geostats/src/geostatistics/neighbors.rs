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

