use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn synthetic_csv(rows: usize, cols: usize) -> String {
    let mut csv = String::new();
    for col in 0..cols {
        csv.push_str(&format!("f{col},"));
    }
    csv.push_str("target\n");

    for row in 0..rows {
        for col in 0..cols {
            let value = ((row * 31 + col * 17) as f32 * 0.001).sin();
            csv.push_str(&format!("{value:.6},"));
        }
        csv.push_str(&format!("{:.6}\n", (row as f32 * 0.01).cos()));
    }

    csv
}

fn parse_csv(input: &str, cols: usize) -> Result<(Vec<f32>, Vec<f32>), std::num::ParseFloatError> {
    let mut values = Vec::new();
    let mut target = Vec::new();

    for line in input.lines().skip(1) {
        if line.is_empty() {
            continue;
        }
        for (index, raw) in line.split(',').enumerate() {
            let value = raw.parse::<f32>()?;
            if index < cols {
                values.push(value);
            } else {
                target.push(value);
            }
        }
    }

    Ok((values, target))
}

fn bench_data_loading(c: &mut Criterion) {
    for features in [64, 4_096] {
        let schema = cartoboost_core::data::FeatureSchema::unnamed_numeric(features);
        c.bench_with_input(
            BenchmarkId::new("schema_validate", features),
            &schema,
            |bench, schema| {
                bench.iter(|| {
                    black_box(schema)
                        .validate()
                        .expect("schema fixture is valid")
                });
            },
        );
    }
    let mut group = c.benchmark_group("data_loading");
    for &(rows, cols) in &[(1_000, 16), (25_000, 32)] {
        let input = synthetic_csv(rows, cols);
        group.bench_with_input(
            BenchmarkId::new("csv_parse", format!("{rows}x{cols}")),
            &input,
            |bench, input| {
                bench.iter(|| {
                    parse_csv(black_box(input), black_box(cols))
                        .expect("synthetic CSV benchmark input must parse")
                });
            },
        );
    }
    group.finish();
}

criterion_group!(data_loading_benches, bench_data_loading);
criterion_main!(data_loading_benches);
