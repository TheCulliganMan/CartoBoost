use cartoboost_accelerator::{
    backend_csr_row_softmax_f32, backend_dense_layer_f32, backend_train_tanh_mlp_f32,
    select_backend,
};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn cpu_kernels(c: &mut Criterion) {
    let backend = select_backend(Some("cpu")).expect("CPU backend");
    let mut dense = c.benchmark_group("accelerator_cpu_dense");
    for (rows, inputs, outputs) in [(32, 32, 32), (256, 128, 128)] {
        let features = (0..rows)
            .map(|row| {
                (0..inputs)
                    .map(|col| ((row * inputs + col) as f32 * 0.01).sin())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let weights = (0..inputs * outputs)
            .map(|index| (index as f32 * 0.013).cos() / inputs as f32)
            .collect::<Vec<_>>();
        let biases = vec![0.125; outputs];
        dense.bench_with_input(
            BenchmarkId::new("rows_inputs_outputs", format!("{rows}_{inputs}_{outputs}")),
            &features,
            |bench, features| {
                bench.iter(|| {
                    backend_dense_layer_f32(
                        &backend,
                        black_box(features),
                        black_box(&weights),
                        black_box(&biases),
                    )
                    .expect("dense layer")
                });
            },
        );
    }
    dense.finish();

    let indptr = (0..=1024).map(|row| row * 32).collect::<Vec<_>>();
    let logits = (0..1024 * 32)
        .map(|index| (index as f32 * 0.013).sin() * 5.0)
        .collect::<Vec<_>>();
    c.bench_function("accelerator_cpu_softmax_1024x32", |bench| {
        bench.iter(|| {
            backend_csr_row_softmax_f32(&backend, black_box(&indptr), black_box(&logits))
                .expect("CSR softmax")
        });
    });

    let inputs = (0..64)
        .map(|row| {
            (0..32)
                .map(|col| ((row * 32 + col) as f32 * 0.01).sin())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let targets = inputs.iter().map(|row| row[0] * 0.2).collect::<Vec<_>>();
    let parameters = vec![0.01; 32 * 32 + 32 + 32 + 1];
    c.bench_function("accelerator_cpu_tanh_train_64x32x32", |bench| {
        bench.iter_batched(
            || parameters.clone(),
            |mut parameters| {
                backend_train_tanh_mlp_f32(
                    &backend,
                    black_box(&inputs),
                    black_box(&targets),
                    32,
                    8,
                    0.001,
                    &mut parameters,
                )
                .expect("tanh training");
                black_box(parameters)
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, cpu_kernels);
criterion_main!(benches);
