use cartoboost_geostats::{
    deterministic_neighbors, CovarianceKernel, NearestNeighborGPRegressor, NngpConfig,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn geostats_nngp_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("geostats_nngp_predict");
    for n in [128usize, 512, 2048] {
        for k in [8usize, 16, 32] {
            group.bench_with_input(
                BenchmarkId::new(format!("n{n}"), k),
                &(n, k),
                |bench, &(n, k)| {
                    let coords = synthetic_coords(n);
                    let values = synthetic_values(&coords);
                    let mut model = NearestNeighborGPRegressor::new(NngpConfig {
                        kernel: CovarianceKernel::Matern32,
                        range: 0.2,
                        sill: 1.0,
                        nugget: 1.0e-6,
                        n_neighbors: k,
                        ..NngpConfig::default()
                    })
                    .expect("config");
                    model.fit(&coords, &values).expect("fit");
                    let targets = synthetic_coords(64);
                    bench.iter(|| model.predict(&targets).expect("predict"));
                },
            );
        }
    }
    group.finish();
    let coords = synthetic_coords(10000);
    c.bench_function("geostats_neighbors_10000_k16", |bench| {
        bench.iter(|| black_box(deterministic_neighbors(black_box(&coords), [0.2, 0.3], 16)));
    });
}

fn synthetic_coords(n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|idx| {
            let x = (idx % 97) as f64 / 97.0;
            let y = (idx / 97) as f64 / 97.0 + (idx % 13) as f64 * 1.0e-5;
            [x, y]
        })
        .collect()
}

fn synthetic_values(coords: &[[f64; 2]]) -> Vec<f64> {
    coords
        .iter()
        .map(|coord| (coord[0] * 8.0).sin() + (coord[1] * 5.0).cos())
        .collect()
}

criterion_group!(benches, geostats_nngp_scaling);
criterion_main!(benches);
