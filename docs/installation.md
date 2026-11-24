---
sidebar_label: Installation
description: Install CartoBoost, verify your Python environment, and add only the integrations you need.
---

# Install CartoBoost

One package gets you the Python modeling API and its native implementation.
NumPy is the only required Python runtime dependency. Add dataframe,
geospatial, and plotting packages when your workflow needs them.

## Install from PyPI

```sh
uv add cartoboost
```

The published wheels target CPython 3.10, 3.11, 3.12, 3.13, 3.14, and 3.15,
plus free-threaded CPython 3.14t and 3.15t, on:

- Linux x86_64 and aarch64 with manylinux2014 compatibility.
- macOS x86_64 and arm64.
- Windows x86_64 and arm64 (Python 3.11–3.15 and 3.14t/3.15t on Windows
  arm64; check release assets for the exact interpreter/platform matrix).

If no compatible wheel exists, `uv` may try to build from source, which requires
the project build toolchain.

## Add the integrations you need

Install only the packages used by your workflow. For example, a Polars
preparation step and H3 spatial cells need:

```sh
uv add polars h3
```

| Package | Adds | Use when |
| --- | --- | --- |
| `shap` | SHAP explanations. | You need feature-attribution diagnostics for a fitted regressor. |
| `h3` | Optional H3 latitude/longitude encoder. | Spatial cells are part of the tested feature design. |
| `holidays` | Country holiday calendar expansion for the piecewise linear seasonal forecaster. | You need Prophet-style `add_country_holidays` behavior. |
| `s2sphere` | Optional S2 latitude/longitude encoder. | S2 cells match the existing geography pipeline. |
| `duckdb` | DuckDB relation/query-result input support. | Taxi training data already lives in DuckDB queries. |
| `optuna` | Hyperparameter tuning examples and workflows. | You are tuning under a fixed validation protocol. |
| `polars` | Polars input support. | Data preparation uses Polars tables. |
| `pandas` | pandas input support. | Data preparation uses pandas tables or `ForecastFrame.from_pandas`. |
| `onnx` | ONNX export for the supported dense axis-tree subset. | Deployment requires ONNX and the model stays inside the supported subset. |
| `matplotlib`, `geopandas`, `shapely`, `pydeck` | Plotting helpers. | You need diagnostic plots, static spatial plots, or interactive taxi route maps. |

## Check your environment

```sh
uv run python -c "import cartoboost; print(cartoboost.__version__)"
```

Then create an estimator:

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(n_estimators=10, max_depth=2)
```

The [first-model walkthrough](getting-started.md) contains a complete NumPy
example you can run after installation. From a repository checkout,
`uv run python examples/quickstart.py` runs the maintained example.

## Build from Rust source

CPU builds support Rust 1.85 and later. The optional WebGPU backend requires
Rust 1.87 or later through wgpu. Use the stable toolchain for the complete
workspace, whose browser crate enables WebGPU by default. To check the CPU
workspace on Rust 1.85, use `cargo +1.85.0 check --workspace --no-default-features`.

### CUDA

CartoBoost's CUDA kernels are built with [cuda-oxide](https://nvlabs.github.io/cuda-oxide/).
This path is for native development and validation; published Python wheels do
not require a local CUDA toolkit unless they are built from source.

Install a CUDA toolkit, an NVIDIA driver, Rust nightly, and `cargo-oxide`. Make
the toolkit compiler visible before building:

```sh
rustup toolchain install nightly-2026-04-03
cargo install cargo-oxide
export PATH=/usr/local/cuda/bin:$PATH
```

Build for the installed GPU's compute capability. The command below derives the
target from `nvidia-smi`; use the resulting `sm_XX` value explicitly when the
machine does not expose `nvidia-smi` to the build environment.

```sh
CUDA_ARCH="sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')"
CUDA_OXIDE_DEBUG=off RUSTUP_TOOLCHAIN=nightly-2026-04-03 \
  cargo oxide build --arch "$CUDA_ARCH" -- -p cartoboost-neural --features cuda
```

Run the native dispatch check on the same architecture:

```sh
CUDA_OXIDE_DEBUG=off RUSTUP_TOOLCHAIN=nightly-2026-04-03 \
  cargo oxide test --arch "$CUDA_ARCH" -- \
  -p cartoboost-neural --features cuda cuda_dispatch_report_runs_vector_add_kernel -- --nocapture
```

The CUDA target must not exceed the installed GPU's capability. For example, a
Turing GPU uses `sm_75`; an `sm_80` artifact cannot run there.

### HIP / ROCm

CartoBoost's HIP backend supports AMD GPUs on Linux and Windows. It dynamically
loads HIP and HIPRTC, so ordinary builds remain independent of ROCm. Install the
AMD ROCm SDK and enable the `rocm` feature:

```sh
cargo test -p cartoboost-neural --features rocm rocm_ -- --nocapture
```

On machines with multiple AMD adapters, select the device in the order reported
by `hipInfo`:

```sh
# Linux
CARTOBOOST_HIP_DEVICE=1 cargo test -p cartoboost-neural --features rocm rocm_ -- --nocapture
```

```powershell
# Windows PowerShell
$env:CARTOBOOST_HIP_DEVICE = "1"
cargo test -p cartoboost-neural --features rocm rocm_ -- --nocapture
```

The runtime searches `HIP_PATH` and `ROCM_PATH`, standard Linux shared-library
names, and versioned Windows SDKs under `Program Files\\AMD\\ROCm`. Model
configuration accepts both `hip` and `rocm` for this backend.

### DirectML

On Windows 10 or later, build the native neural crate with `directml` to use a
DirectX 12 adapter through the system DirectML runtime. No CUDA or ROCm toolkit
is required:

```powershell
cargo test -p cartoboost-neural --features directml
```

Request this backend as `directml` (or `dml`). It is advertised only after a
Direct3D 12 adapter and DirectML device are created successfully; an explicit
request fails instead of falling back to CPU when the feature or device is
unavailable.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError` during import | Reinstall CartoBoost in a clean Python environment. |
| `uv` tries to compile from source | Use CPython 3.10-3.15 or free-threaded CPython 3.14t/3.15t on a supported platform, or install the project build toolchain before building. |
| `ModuleNotFoundError` after installation | Run the script with `uv run python` from the project where you installed CartoBoost. |
| `examples/quickstart.py` does not exist | That file belongs to the source checkout. Use the complete example in [Your first model](getting-started.md) for a PyPI install. |
| cuda-oxide reports an artifact for a newer GPU architecture | Rebuild with `cargo oxide ... --arch sm_XX`, where `sm_XX` matches `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`. |

**Ready to fit?** Continue with [Your first model](getting-started.md), or
[choose a model family](user-guide/model-types.md) for your prediction task.
