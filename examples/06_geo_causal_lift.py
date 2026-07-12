from __future__ import annotations

from cartoboost.preview.geo_causal import (
    GeoCausalPanel,
    GeoExperimentDesigner,
    SpatialPlaceboTester,
    SyntheticDIDEstimator,
)


def make_panel(effect: float = 4.0) -> GeoCausalPanel:
    rows = []
    geos = {
        "pickup_zone_101": (40.71, -74.00, 1.0),
        "pickup_zone_102": (40.73, -73.98, 1.0),
        "pickup_zone_201": (40.80, -73.94, 0.8),
        "pickup_zone_202": (40.65, -73.90, 1.2),
    }
    for geo, (lat, lon, geo_shift) in geos.items():
        for day in range(14):
            post = day >= 7
            baseline = 120.0 + day * 2.0 + geo_shift
            rows.append(
                {
                    "unit_id": geo,
                    "time": f"2026-03-{day + 1:02}",
                    "outcome": baseline + (effect if geo == "pickup_zone_101" and post else 0.0),
                    "treatment": geo == "pickup_zone_101" and post,
                    "latitude": lat,
                    "longitude": lon,
                    "region_id": geo,
                    "avg_trip_distance": 2.5 + geo_shift * 0.1,
                }
            )
    return GeoCausalPanel(
        rows,
        covariate_cols=["avg_trip_distance"],
        spatial_weights=[
            ("pickup_zone_101", "pickup_zone_102", 1.0),
            ("pickup_zone_201", "pickup_zone_202", 0.4),
        ],
    )


panel = make_panel()
estimator = SyntheticDIDEstimator(intervention_time="2026-03-08", seed=11).fit(panel)
placebos = estimator.placebo_test(n=8)
print("effect", round(estimator.estimate_effect(), 3))
print("placebo_mean", round(sum(placebos) / len(placebos), 3))
print("assumptions", estimator.summary()["assumptions"])

design = (
    GeoExperimentDesigner(intervention_time="2026-03-08", seed=11)
    .fit(panel)
    .summary(
        candidate_count=1,
        placebo_n=8,
    )
)
print("candidate_test_geos", design["candidate_test_geos"])
print("estimated_detectable_lift", round(design["estimated_detectable_lift"], 4))

spillover = SpatialPlaceboTester(intervention_time="2026-03-08", seed=11).fit(panel).summary()
print("spillover_warnings", spillover["warnings"])
