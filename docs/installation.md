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

CartoBoost also supports CPython 3.14's free-threaded build. Install the
matching `cp314t` wheel, or build from source with a free-threaded interpreter.

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

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ImportError` during import | Reinstall CartoBoost in a clean Python environment. |
| `uv` tries to compile from source | Use CPython 3.10-3.14 on a supported platform, or install the project build toolchain before building. |
| `examples/quickstart.py` cannot import CartoBoost | Make sure the Python environment where `cartoboost` was installed is active. |
