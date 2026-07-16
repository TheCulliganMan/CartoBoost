# Installation

CartoBoost is published on PyPI as `cartoboost`.

Install CartoBoost when you need structured tabular or panel prediction with
place, cyclic time, memberships, or direction in a Python environment. The
core package is NumPy-only; optional extras add dataframe and other integrations.

## Install From PyPI

```sh
uv add cartoboost
```

The published wheels target CPython 3.10, 3.11, 3.12, 3.13, and 3.14 on:

- Linux x86_64 and aarch64 with manylinux2014 compatibility.
- macOS x86_64 and arm64.
- Windows x86_64 and arm64 (Python 3.11–3.14 on Windows arm64; check the
  release assets for the exact interpreter/platform matrix).

If no compatible wheel exists, `uv` may try to build from source, which requires
the project build toolchain.

## Optional Extras

Install optional integrations only when the scientific workflow needs them:

```sh
uv add "cartoboost[explain]"
uv add "cartoboost[h3]"
uv add "cartoboost[holidays]"
uv add "cartoboost[s2]"
uv add "cartoboost[duckdb]"
uv add "cartoboost[optuna]"
uv add "cartoboost[polars]"
uv add "cartoboost[pandas]"
uv add "cartoboost[onnx]"
uv add "cartoboost[visualization]"
```

| Extra | Adds | Use when |
| --- | --- | --- |
| `explain` | SHAP explanations. | You need feature-attribution diagnostics for a fitted regressor. |
| `h3` | Optional H3 latitude/longitude encoder. | Spatial cells are part of the tested feature design. |
| `holidays` | Country holiday calendar expansion for the piecewise linear seasonal forecaster. | You need Prophet-style `add_country_holidays` behavior. |
| `s2` | Optional S2 latitude/longitude encoder. | S2 cells match the existing geography pipeline. |
| `duckdb` | DuckDB relation/query-result input support. | Taxi training data already lives in DuckDB queries. |
| `optuna` | Hyperparameter tuning examples and workflows. | You are tuning under a fixed validation protocol. |
| `polars` | Polars input support. | Data preparation uses Polars tables. |
| `pandas` | pandas input support. | Data preparation uses pandas tables or `ForecastFrame.from_pandas`. |
| `onnx` | ONNX export for the supported dense axis-tree subset. | Deployment requires ONNX and the model stays inside the supported subset. |
| `visualization` | Matplotlib, GeoPandas, Shapely, and PyDeck plotting helpers. | You need diagnostic plots, static spatial plots, or interactive taxi route maps. |

## Verify The Install

```sh
python -c "import cartoboost; print(cartoboost.__version__)"
python examples/quickstart.py
```

Python usage should work immediately after install:

```python
from cartoboost import CartoBoostRegressor

model = CartoBoostRegressor(n_estimators=10, max_depth=2)
```

## CUDA Development Build

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

## HIP/ROCm Development Build

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

## DirectML Development Build

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
| `uv` tries to compile from source | Use CPython 3.10-3.14 on a supported platform, or install the project build toolchain before building. |
| `examples/quickstart.py` cannot import CartoBoost | Make sure the Python environment where `cartoboost` was installed is active. |
| cuda-oxide reports an artifact for a newer GPU architecture | Rebuild with `cargo oxide ... --arch sm_XX`, where `sm_XX` matches `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`. |
