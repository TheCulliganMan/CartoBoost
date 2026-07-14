# Geo-Causal Experiment Models

CartoBoost geo-causal tools estimate intervention effects on geographic panels.
They are for questions such as marketing lift, policy rollout impact, store
openings, and network changes. They do not forecast future demand and should not
be reported as forecast accuracy evidence.

## Models

| Model | Use |
| --- | --- |
| `SyntheticDIDEstimator` | Estimate post-intervention treatment effect from treated geos and weighted controls. |
| `GeoLiftEstimator` | Alias for GeoLift-style experiment design helpers. |
| `GeoExperimentDesigner` | Choose candidate test geos, check balance, estimate detectable lift, and run placebos. |
| `SpatialPlaceboTester` | Run deterministic placebo assignments and spillover diagnostics. |
| `InvariantRiskEncoder` | Representation supplement for held-out-region diagnostics; not a causal estimator. |

All fitting and causal scoring behavior is implemented in Rust under
`crates/cartoboost-geo-causal`. Python wraps the native routines and only adds
data coercion, `.plot()`, and ergonomics.

Representation supplements such as `InvariantRiskEncoder`,
`DomainAdversarialGeoEncoder`, `CounterfactualRepresentationNet`, and
`TreatmentEffectRepresentationHead` can reduce held-out-region prediction error
in domain-shift diagnostics, but they do not identify causal effects by
themselves. Use them only alongside an identified design such as
`SyntheticDIDEstimator` or `GeoExperimentDesigner`, and keep the estimator's
assumptions, placebo checks, and spillover diagnostics in the causal report.

## Panel Contract

`GeoCausalPanel` requires:

| Field | Meaning |
| --- | --- |
| `unit_id` | Stable geographic unit such as pickup zone, store catchment, district, or network node. |
| `time` | Comparable ordered timestamp label. ISO strings sort correctly and are recommended. |
| `outcome` | Numeric outcome such as trips, revenue, visits, policy incidents, or throughput. |
| `treatment` | Boolean flag for treated unit-period rows. |
| `covariates` | Optional numeric controls retained in the panel contract. |
| `latitude`, `longitude` | Optional coordinates for distance diagnostics. |
| `region_id` | Optional stable region identifier when coordinates are not the primary geography. |
| `spatial_weights` | Optional weighted adjacency used for spillover diagnostics. |

Missing required inputs fail clearly. CartoBoost does not synthesize controls,
drop invalid geos, or replace failed experiment designs with weaker defaults.

## Synthetic DID

`SyntheticDIDEstimator` splits rows into pre and post periods using
`intervention_time`, partitions units into treated and controls, builds control
unit weights from pre-period outcome balance, assigns deterministic pre-period
time weights, estimates the post-period treatment effect, and can run placebo
assignments.

```python
from cartoboost.geo_causal import GeoCausalPanel, SyntheticDIDEstimator

panel = GeoCausalPanel(
    rows,
    unit_col="pickup_zone",
    time_col="date",
    outcome_col="trips",
    treatment_col="campaign_live",
    covariate_cols=["avg_trip_distance", "pickup_hour_share"],
    latitude_col="lat",
    longitude_col="lon",
    spatial_weights=[("zone_101", "zone_102", 1.0)],
)

estimator = SyntheticDIDEstimator(intervention_time="2026-03-08", seed=11).fit(panel)
effect = estimator.estimate_effect()
placebos = estimator.placebo_test(n=100)
summary = estimator.summary()
```

Every summary includes assumptions. Interpret the effect only when those
assumptions are defensible for the study:

- No unmeasured post-intervention shocks differentially hit treated geos.
- Control geos represent the untreated counterfactual.
- Spillovers from treated to control geos are absent or small.
- The result is a causal estimate, not a forecast.

## GeoLift-Style Design

Use `GeoExperimentDesigner` or `GeoLiftEstimator` before launching a geographic
test. The designer ranks candidate test geos by pre-period balance, estimates
detectable lift from placebo dispersion, and reports spillover warnings.

```python
from cartoboost.geo_causal import GeoExperimentDesigner

design = (
    GeoExperimentDesigner(intervention_time="2026-03-08", seed=11)
    .fit(panel)
    .summary(candidate_count=2, placebo_n=200)
)
```

Use this for marketing lift when selecting treated media markets or taxi pickup
zones, policy rollout when choosing districts, store openings when choosing
candidate catchments, and network changes when selecting treated corridors or
nodes. The design helper is not proof that the intervention will work; it only
checks whether the historical panel can support a measurable experiment under
the stated assumptions.

## Spillover Diagnostics

`SpatialPlaceboTester.summary()` reports:

| Diagnostic | Meaning |
| --- | --- |
| `adjacent_treated_control_pairs` | Treated/control pairs connected by spatial weights. |
| `min_treated_control_distance` | Closest treated/control distance when coordinates are available. |
| `mean_treated_control_distance` | Average treated/control distance. |
| `treated_weight_exposure` | Total weighted adjacency exposure involving treated units. |
| `control_weight_exposure` | Total weighted adjacency exposure among controls. |
| `warnings` | Explicit spillover warnings that should appear in reports. |

Adjacent treated/control units are a warning, not an automatic correction. Move
geos, buffer controls, redefine spatial weights, or treat the estimate as
potentially contaminated.

## WASM And Model Lab

The browser/WASM surface exposes `runGeoCausalExperiment(request)`. The request
matches the native panel contract and returns a JSON-compatible summary with
effect, weights, placebos, assumptions, and warnings.

```js
const response = wasm.runGeoCausalExperiment({
  interventionTime: "2026-03-08",
  seed: 11,
  placeboN: 100,
  rows,
  spatialWeights: [
    {fromUnit: "pickup_zone_101", toUnit: "pickup_zone_102", weight: 1.0},
  ],
});
```

See `examples/model_lab_geo_causal_request.json` for a complete model-lab
request payload and `examples/06_geo_causal_lift.py` for the Python workflow.

## Reporting

Reports should include the exact intervention time, treated geos, control
weights, placebo distribution, spillover warnings, and assumptions. Do not
combine causal estimates with forecast benchmark tables unless the table clearly
separates intervention effects from prediction metrics.
