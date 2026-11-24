use cartoboost_core::data::SparseSetColumn;
use cartoboost_core::tree::SplitterKind;
use cartoboost_core::{Booster, BoosterConfig, Dataset};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn training_fixture(rows: usize, cols: usize) -> (Dataset, Vec<f64>) {
    let values = (0..rows * cols)
        .map(|index| {
            let row = index / cols;
            let col = index % cols;
            (row as f64 * 0.07 + col as f64 * 0.19).sin()
        })
        .collect::<Vec<_>>();
    let target = (0..rows)
        .map(|row| {
            let first = values[row * cols];
            let last = values[row * cols + cols - 1];
            if first + 0.25 * last > 0.0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect();

    (
        Dataset::from_flat(rows, cols, values).expect("synthetic training data is valid"),
        target,
    )
}

fn booster_config(n_estimators: usize, max_depth: usize) -> BoosterConfig {
    BoosterConfig {
        n_estimators,
        learning_rate: 0.2,
        max_depth,
        min_samples_leaf: 2,
        min_gain: 0.0,
        splitters: vec![SplitterKind::Axis],
        ..BoosterConfig::default()
    }
}

fn histogram_booster_config(n_estimators: usize, max_depth: usize) -> BoosterConfig {
    BoosterConfig {
        splitters: vec![SplitterKind::AxisHistogram { bins: 64 }],
        ..booster_config(n_estimators, max_depth)
    }
}

fn auto_booster_config(n_estimators: usize, max_depth: usize) -> BoosterConfig {
    BoosterConfig {
        splitters: vec![SplitterKind::Auto],
        ..booster_config(n_estimators, max_depth)
    }
}

fn mixed_training_fixture(rows: usize) -> (Dataset, Vec<f64>) {
    let cols = 4;
    let mut values = Vec::with_capacity(rows * cols);
    let mut sparse_rows = Vec::with_capacity(rows);
    let mut target = Vec::with_capacity(rows);
    for row in 0..rows {
        let x = (row as f64 * 0.017).sin() * 10.0;
        let y = (row as f64 * 0.023).cos() * 10.0;
        let hour = (row % 24) as f64;
        let dense_id = (row % 17) as f64;
        values.extend([x, y, hour, dense_id]);
        sparse_rows.push(vec![(row % 31) as u64, ((row / 7) % 31) as u64]);
        let signal = if x + y > 0.0 { 1.0 } else { -1.0 }
            + if !(6.0..=20.0).contains(&hour) {
                0.5
            } else {
                -0.5
            }
            + if row % 31 == 3 { 1.5 } else { 0.0 };
        target.push(signal);
    }
    let dataset = Dataset::from_flat(rows, cols, values)
        .expect("synthetic mixed data is valid")
        .with_sparse_sets(vec![SparseSetColumn::new(sparse_rows)])
        .expect("synthetic sparse sets align");
    (dataset, target)
}

fn mixed_booster_config(n_estimators: usize, max_depth: usize) -> BoosterConfig {
    BoosterConfig {
        n_estimators,
        learning_rate: 0.2,
        max_depth,
        min_samples_leaf: 2,
        min_gain: 0.0,
        splitters: vec![
            SplitterKind::Axis,
            SplitterKind::Diagonal2D,
            SplitterKind::Gaussian2D,
            SplitterKind::Periodic { period: 24.0 },
            SplitterKind::SparseSet,
        ],
        ..BoosterConfig::default()
    }
}

fn bench_training(c: &mut Criterion) {
    let mut metrics = c.benchmark_group("binary_metric_scaling");
    metrics.sample_size(10);
    for rows in [1_024, 8_192, 32_768] {
        let labels = (0..rows).map(|row| (row % 2) as f64).collect::<Vec<_>>();
        let scores = (0..rows)
            .map(|row| ((row * 7) % 37) as f64)
            .collect::<Vec<_>>();
        let weights = (0..rows).map(|row| (row % 11) as f64).collect::<Vec<_>>();
        let query_ids = (0..rows).map(|row| row % 63).collect::<Vec<_>>();
        metrics.bench_with_input(BenchmarkId::new("weighted_auc", rows), &rows, |bench, _| {
            bench.iter(|| {
                cartoboost_core::metrics::binary_roc_auc(
                    black_box(&labels),
                    black_box(&scores),
                    Some(black_box(&weights)),
                )
                .unwrap()
            });
        });
        metrics.bench_with_input(BenchmarkId::new("graph_mrr", rows), &rows, |bench, _| {
            bench.iter(|| {
                cartoboost_core::metrics::graph_mean_reciprocal_rank(
                    black_box(&labels),
                    black_box(&scores),
                    black_box(&query_ids),
                )
                .unwrap()
            });
        });
    }
    metrics.finish();
    let relevance = (0..4_096).map(|row| (row % 5) as f64).collect::<Vec<_>>();
    let scores = (0..4_096)
        .map(|row| (row as f64 * 0.37).sin())
        .collect::<Vec<_>>();
    let groups = vec![64; 64];
    c.bench_function("ranking_metrics/4096rows_64groups", |bench| {
        bench.iter(|| {
            cartoboost_core::booster::ranking_metrics(
                black_box(&relevance),
                black_box(&scores),
                black_box(&groups),
            )
            .expect("ranking benchmark fixture is valid")
        });
    });
    let mut group = c.benchmark_group("training");
    group.sample_size(10);
    for &(rows, cols, estimators, depth) in &[(32, 3, 1, 1), (96, 4, 3, 2)] {
        let (dataset, target) = training_fixture(rows, cols);
        let booster = Booster::new(booster_config(estimators, depth));
        group.bench_with_input(
            BenchmarkId::new(
                "booster_fit",
                format!("{rows}x{cols}_{estimators}trees_depth{depth}"),
            ),
            &(booster, dataset, target),
            |bench, (booster, dataset, target)| {
                bench.iter(|| {
                    black_box(booster)
                        .fit(black_box(dataset), black_box(target), None)
                        .expect("benchmark fixture trains")
                });
            },
        );
    }
    for &(rows, cols, estimators, depth) in &[(1_000, 8, 20, 3), (25_000, 8, 100, 4)] {
        let (dataset, target) = training_fixture(rows, cols);
        let booster = Booster::new(auto_booster_config(estimators, depth));
        group.bench_with_input(
            BenchmarkId::new(
                "booster_fit_auto",
                format!("{rows}x{cols}_{estimators}trees_depth{depth}"),
            ),
            &(booster, dataset, target),
            |bench, (booster, dataset, target)| {
                bench.iter(|| {
                    black_box(booster)
                        .fit(black_box(dataset), black_box(target), None)
                        .expect("benchmark fixture trains")
                });
            },
        );
    }
    {
        let (rows, estimators, depth) = (1_000, 20, 3);
        let (dataset, target) = mixed_training_fixture(rows);
        let booster = Booster::new(mixed_booster_config(estimators, depth));
        group.bench_with_input(
            BenchmarkId::new(
                "booster_fit_mixed_splitters",
                format!("{rows}rows_{estimators}trees_depth{depth}"),
            ),
            &(booster, dataset, target),
            |bench, (booster, dataset, target)| {
                bench.iter(|| {
                    black_box(booster)
                        .fit(black_box(dataset), black_box(target), None)
                        .expect("benchmark fixture trains")
                });
            },
        );
    }
    for &(rows, cols, estimators, depth) in &[(25_000, 8, 100, 4), (25_000, 3, 100, 4)] {
        let (dataset, target) = training_fixture(rows, cols);
        let booster = Booster::new(histogram_booster_config(estimators, depth));
        group.bench_with_input(
            BenchmarkId::new(
                "booster_fit_axis_histogram",
                format!("{rows}x{cols}_{estimators}trees_depth{depth}"),
            ),
            &(booster, dataset, target),
            |bench, (booster, dataset, target)| {
                bench.iter(|| {
                    black_box(booster)
                        .fit(black_box(dataset), black_box(target), None)
                        .expect("benchmark fixture trains")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(training_benches, bench_training);
criterion_main!(training_benches);
