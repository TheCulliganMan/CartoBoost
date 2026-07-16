# Docs Agent Guide

## Dev environment tips
- This folder contains project documentation for API contracts, artifacts, schema behavior, sparse features, tests, benchmarks, and release readiness.
- Keep docs factual and aligned with code and tests.
- Write for Python developers and data scientists first. Lead with what the
  model is good for, when to use it, how to fit it, how to validate it, and
  what to compare against.
- Use domain-general examples by default: demand, duration, price, route,
  sensor, location, entity id, hour, day of week, and series id. Taxi examples
  are allowed as one concrete framing device, but do not make the package or a
  model page feel taxi-specific.
- Keep optional dependency docs split by extra. For example, document `cartoboost[h3]`, `cartoboost[s2]`, `cartoboost[duckdb]`, and `cartoboost[polars]` separately instead of implying one bundled geo/table extra.
- When documenting benchmark results, distinguish real NYC taxi runs from synthetic smoke or acceptance fixtures. Avoid broad superiority language unless the documented command and metrics support it.
- Keep forecast benchmark tables reader-facing: one ranked model-comparison table per dataset family when possible. If a family includes multiple artifacts, use a `Run` column instead of scattering several small tables.
- Avoid vague public table headers such as `Scope`. Use plain, concrete headers such as `Model`, `RMSE`, `MAE`, `WAPE`, `Read`, `Artifact`, `Details`, and `Result`.
- When benchmark code changes metric output or artifact schema, update public benchmark docs only from freshly run benchmark artifacts, not from stale remembered values.
- Keep public forecasting docs user-facing. Internal implementation rules such as
  Rust-first ownership, Python wrapper boundaries, and benchmark acceptance
  gates belong in AGENTS files unless they are explaining an actual public API
  contract.
- Model guide pages should follow the maintained user-guide shape:
  title, short purpose paragraph, interactive example when a browser example
  exists, public Python contract, "When To Use", "Use When" comparison table,
  validation guidance, and concrete limitations.
- Prefer one model family per page. Avoid aggregate pages that duplicate
  individual model pages unless the aggregate page adds a real decision table
  or workflow that cannot live in the index.
- Do not list undocumented public models. If a user-guide index, router table,
  comparison table, or `sidebars.ts` entry names a public model class or model
  family, it must link to a real maintained guide page for that model or
  family. If the page should not exist, remove the listing instead of leaving a
  dangling or aggregate-only mention.
- Every model guide page linked from a user-guide index or sidebar must exist
  in the current tree and have a clear Python example, model-selection guidance,
  validation guidance, and limitations unless the page is explicitly a router/index page.
- Use `## Interactive Example` for embedded browser examples. Explain the
  browser run as a quick shape and behavior check, not as benchmark evidence.
  Do not document raw Wasm or JavaScript payloads in user model guides unless
  the page is specifically about the browser API.
- Use `## Python Example` for the primary Python example. Examples should be
  complete enough to show required inputs and output shape, but should not
  expose internal Rust, PyO3, artifact, or benchmark-harness details.
- Keep "Use When" sections. They should help users decide between adjacent
  CartoBoost models and serious baselines.
- Avoid process labels such as "new", "cleanup", "scaffold", or "future work"
  in public docs. If a public class is intentionally not production-ready,
  document it under limitations without making the page about implementation
  backlog.

## Testing instructions
- Cross-check examples against current Python, CLI, and Rust contracts.
- Update the specific contract page when implementation behavior changes, not only the README.
- Before finishing navigation or index changes, search for newly introduced
  model class names and verify each listed model resolves to a real guide page
  or is removed from the listing.
- Run `npm run typecheck` and `npm run build` after navigation changes, renamed pages, public API docs changes, or Docusaurus component edits.
- Search docs before finishing terminology updates for old non-taxi terminology in `docs`, `README.md`, `docusaurus.config.ts`, and `sidebars.ts`.

## PR instructions
- Identify which public contract or guide changed.
- Mention any code or test changes that keep docs in sync.
- If generated docs assets are refreshed, state the generating command and why the committed outputs changed.
