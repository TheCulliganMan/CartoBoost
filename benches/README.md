# CartoBoost Benchmarks

This directory contains low-level Rust timing benchmarks. Most data-science
model comparisons should use the benchmark guides under `docs/benchmarks/`,
especially the NYC taxi and lane-level temporal-spatial reports.

Files:

- `training.rs`: `Booster::fit` on bounded synthetic matrices.
- `prediction.rs`: `Model::predict` and flat prediction paths on synthetic
  batches.
- `serialize.rs`: in-memory JSON serialization and deserialization.

Commands:

```sh
cargo bench --workspace --no-run
cargo bench --workspace
uv run --group dev python scripts/plot_benchmarks.py
uv run --group dev python scripts/check_performance_thresholds.py
```

`benchmark_summary.json` records the maintained timing summary and its
`max_mean_ns` regression thresholds. Use these timings for local performance
checks. Use dataset benchmarks for claims about model quality.
