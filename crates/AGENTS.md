# Rust Crates Agent Guide

## Dev environment tips
- This folder contains the Rust workspace crates.
- Keep shared behavior in `cartoboost-core`, CLI-only behavior in `cartoboost-cli`, and PyO3 binding code in `cartoboost-py`.
- Use workspace dependencies and lint settings from the root `Cargo.toml`.
- Keep `lib.rs` and `main.rs` focused on module declarations, public re-exports, and minimal crate initialization. New implementation belongs in domain-named modules; split oversized crate roots instead of extending them. As a review rule, treat roughly 500 lines as the point where a crate root needs an explicit structural justification.

## Testing instructions
- Run `cargo fmt --all --check`.
- Run `cargo clippy --workspace --all-targets -- -D warnings`.
- Run `cargo test --workspace`.

## PR instructions
- Name the crate or crates affected.
- Mention any Python, CLI, or artifact compatibility impact from Rust changes.
