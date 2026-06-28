set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
    just --list

fmt:
    cargo fmt --all
    uv run ruff format python tests scripts

lint:
    uv run pre-commit run --all-files

test:
    cargo test --workspace
    uv run pytest

build:
    uv run maturin build --release --locked --out dist

develop:
    uv run maturin develop

pre-commit-install:
    uv run pre-commit install

pre-commit:
    uv run pre-commit run --all-files

sdist:
    uv run maturin sdist --out dist

wheel:
    uv run maturin build --release --locked --out dist

release-prepare version:
    python3 scripts/prepare_release_tag.py --version {{version}}

release-bump bump="patch":
    python3 scripts/prepare_release_tag.py --bump {{bump}}

release-tag:
    python3 scripts/prepare_release_tag.py --tag-current

validate:
    uv sync
    uv run pre-commit run --all-files
    cargo test --workspace
    uv run maturin develop
    uv run pytest
    uv run python scripts/run_full_validation.py
    uv run python scripts/run_v1_validation.py
    uv run python scripts/check_release_gates.py
    uv run python scripts/check_public_api_contract.py
    uv run python scripts/check_artifact_compatibility.py
    uv run python scripts/check_docs_examples.py
    uv run python scripts/check_official_geo_evidence.py
    uv run python scripts/check_performance_thresholds.py
    just autogeo-benchmark-gate
    cargo bench --workspace --no-run

nyc-quality-benchmark:
    uv run maturin develop --release
    PYTHONPATH=python uv run --group bench python scripts/run_nyc_taxi_quality_benchmarks.py

nyc-quality-benchmark-smoke:
    PYTHONPATH=python uv run --group bench python scripts/run_nyc_taxi_quality_benchmarks.py --synthetic-smoke --models mean

nyc-quality-benchmark-repeated:
    uv run maturin develop --release
    PYTHONPATH=python uv run --group bench python scripts/run_repeated_nyc_taxi_benchmarks.py --no-download

model-benchmark-suite:
    PYTHONPATH=python uv run --group bench python scripts/run_model_benchmark_suite.py

autogeo-benchmark-gate sample_size="90":
    PYTHONPATH=python uv run --group dev python scripts/run_autogeo_benchmark_gate.py --output-dir target/autogeo-gate --sample-size {{sample_size}} --n-splits 5

bench-setup:
    uv sync --group dev --group bench
    uv run --group dev maturin develop --release

bench-smoke:
    PYTHONPATH=python uv run --group bench python scripts/run_nyc_taxi_quality_benchmarks.py --synthetic-smoke --models mean --no-plots
    PYTHONPATH=python uv run --group bench python scripts/run_model_benchmark_suite.py --n-rows 500 --datasets normal --models mean,cartoboost --no-plots

bench-nyc tasks="duration_minutes,fare_amount" models="cartoboost,xgboost,lightgbm,hist_gradient_boosting" sample_size="50000" extra="":
    uv run --group dev maturin develop --release
    PYTHONPATH=python uv run --group bench python scripts/run_nyc_taxi_quality_benchmarks.py --tasks {{tasks}} --models {{models}} --sample-size {{sample_size}} {{extra}}

bench-nyc-repeated runs="3" tasks="duration_minutes,fare_amount" models="cartoboost,xgboost,lightgbm,hist_gradient_boosting" sample_size="50000" extra="--no-download":
    uv run --group dev maturin develop --release
    PYTHONPATH=python uv run --group bench python scripts/run_repeated_nyc_taxi_benchmarks.py --runs {{runs}} --tasks {{tasks}} --models {{models}} --sample-size {{sample_size}} {{extra}}

bench-models datasets="normal,neural,graph" models="cartoboost,xgboost,lightgbm,hist_gradient_boosting" n_rows="5000" extra="":
    PYTHONPATH=python uv run --group bench python scripts/run_model_benchmark_suite.py --datasets {{datasets}} --models {{models}} --n-rows {{n_rows}} {{extra}}

bench-rust extra="":
    cargo bench --workspace {{extra}}

bench-rust-build:
    cargo bench --workspace --no-run

clean:
    cargo clean
    rm -rf build dist target wheels *.egg-info .pytest_cache .ruff_cache .venv
