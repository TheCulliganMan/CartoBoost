Implemented.

The generic deep layer now lives in the existing repo layout:

- Rust native behavior in `crates/cartoboost-neural/src/deep.rs`
- PyO3 bindings in `crates/cartoboost-py/src/lib.rs`
- Python ergonomics in `python/cartoboost/deep/`
- wasm exports in `crates/cartoboost-wasm/src/lib.rs`
- user docs in `docs/user-guide/deep-models.md`

The original work items were removed after implementation.
