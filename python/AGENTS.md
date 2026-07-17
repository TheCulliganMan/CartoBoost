# Python Source Agent Guide

## Dev environment tips
- This folder contains the Python package source root.
- Keep the public package importable through `pythonpath = ["python"]`.
- Keep layout compatible with maturin's `python-source` setting.
- Keep estimator training native-extension-only; do not add a separate Python
  training path.
- Keep Python model classes as thin ergonomics wrappers over the native Rust behavior. Python-side helper utilities are appropriate for input adaptation, optional dependency integration, and sparse-set construction.
- Optional input integrations such as Polars, DuckDB, H3, and S2 must not be imported at module import time unless they are core dependencies. Detect them lazily and raise clear install-hint errors when a helper requires a missing optional package.
- Table input support should preserve feature names where possible and should match NumPy behavior. DuckDB/Polars support should be tested against the same model configuration as the NumPy path.
- Do not add silent fallback paths for missing optional geo encoders or required model inputs. If an operation cannot be performed correctly, raise a clear error.

## Testing instructions
- Run `uv run ruff format --check python`.
- Run relevant pytest files after Python package changes.
- For optional dependency adapters, run both targeted tests and a smoke command after installing the package directly, for example `uv add --dev duckdb`, `uv add --dev h3`, or `uv add --dev s2sphere`.

## PR instructions
- Summarize changed Python API surface.
- Mention whether native binding rebuilds are required.
- Mention any optional dependencies added or changed and whether `uv.lock` was refreshed.
