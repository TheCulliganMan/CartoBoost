from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from cartoboost.deep import (
    ChoiceSetTransformer,
    ConditionalFlowDistributionHead,
    DirectionalPairForecaster,
    DirectionalPairFrame,
    EntityPanelFrame,
    GraphTemporalFrame,
    InvertedTemporalTransformer,
    PropagationDelayGraphForecaster,
    TemporalSSMForecaster,
)
from cartoboost.representation import RetrievalAugmentedForecaster

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "deep_claim_benchmarks" / "results.json"


def rmse(actual: Any, pred: Any) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))


def pair_embedding_claim() -> dict[str, Any]:
    rows = []
    for source, target, direction in [("A", "B", 1.0), ("B", "A", -1.0), ("A", "C", 0.4)]:
        for step in range(18):
            x = step / 17.0
            rows.append(
                {
                    "source_id": source,
                    "target_id": target,
                    "features": [x, x * x],
                    "target": 1.5 + direction * (2.0 * np.sin(np.pi * x) + (x - 0.5) ** 2),
                }
            )
    frame = DirectionalPairFrame(rows)
    shrink = DirectionalPairForecaster(architecture="shrinkage_effects").fit(frame)
    embed = DirectionalPairForecaster(
        architecture="pair_embedding_mlp",
        embedding_dim=5,
        pair_bucket_count=32,
        hidden_dim=16,
        epochs=420,
        learning_rate=0.012,
        seed=19,
    ).fit(frame)
    embed_rmse = float(embed.score(frame))
    shrink_rmse = float(shrink.score(frame))
    return {
        "claim": "pair_embedding_mlp beats shrinkage on nonlinear pair task",
        "model_metric": embed_rmse,
        "baseline_metric": shrink_rmse,
        "passed": embed_rmse < shrink_rmse,
    }


def ssm_claim() -> dict[str, Any]:
    time = np.arange(160, dtype=float)
    y0 = np.zeros_like(time)
    y1 = np.zeros_like(time)
    for idx in range(12, len(time)):
        y0[idx] = 0.75 * y0[idx - 9] - 0.35 * y0[idx - 5] + np.sin(idx / 13.0)
        y1[idx] = 0.65 * y1[idx - 7] + 0.25 * y0[idx - 11] + np.cos(idx / 17.0)
    y = np.column_stack([y0, y1])
    frame = EntityPanelFrame(y=y, timestamps=list(range(len(y))), entity_ids=["a", "b"])
    model = TemporalSSMForecaster(lookback=48, horizon=4, state_dim=8, seed=13).fit(frame)
    decoder = model.metadata_["decoder"]
    return {
        "claim": "SSM beats trend extrapolation on long-memory task",
        "model_metric": decoder["ssm_decoder_rmse"],
        "baseline_metric": decoder["trend_extrapolation_rmse"],
        "temporal_conv_baseline_metric": decoder["temporal_conv_baseline_rmse"],
        "passed": decoder["beats_trend_extrapolation"]
        and decoder["beats_temporal_conv_baseline"]
        and model.metadata_["architecture"] == "selective_ssm_lite",
    }


def inverted_transformer_claim() -> dict[str, Any]:
    steps = 80
    y = np.zeros((steps, 3), dtype=float)
    y[0] = [1.0, 2.0, 1.5]
    for step in range(1, steps):
        y[step, 1] = 0.8 * y[step - 1, 1] + np.sin(step / 4.0)
        y[step, 0] = 0.6 * y[step - 1, 0] + 0.6 * y[step - 1, 1] + 0.05 * step
        y[step, 2] = 0.8 * y[step - 1, 2] + 0.1 * y[step - 1, 0]
    train = y[:-3]
    actual = y[-3:]
    frame = EntityPanelFrame(
        y=train, timestamps=list(range(len(train))), entity_ids=["a", "b", "c"]
    )
    model = InvertedTemporalTransformer(lookback=24, horizon=3).fit(frame)
    model_rmse = rmse(actual, model.predict())
    baseline_rmse = rmse(actual, np.tile(train[-1], (3, 1)))
    return {
        "claim": "inverted transformer beats independent panel baseline",
        "model_metric": model_rmse,
        "baseline_metric": baseline_rmse,
        "passed": model_rmse < baseline_rmse,
    }


def delay_graph_claim() -> dict[str, Any]:
    steps = 90
    y = np.zeros((steps, 3), dtype=float)
    for step in range(steps):
        y[step, 0] = np.sin(step / 4.0) + 0.02 * step
        if step >= 2:
            y[step, 1] = 0.35 * y[step - 1, 1] + 0.9 * y[step - 2, 0]
        if step >= 3:
            y[step, 2] = 0.25 * y[step - 1, 2] + 0.7 * y[step - 1, 1]
    train = y[:-4]
    actual = y[-4:]
    timestamps = list(range(len(train)))
    correct = GraphTemporalFrame(
        y=train,
        timestamps=timestamps,
        node_ids=["pickup", "midway", "dropoff"],
        edges=[(0, 1), (1, 2)],
        edge_weights=[1.0, 1.0],
        edge_distances=[0.4, 0.9],
        node_covariates=np.asarray([[1.0, 0.0], [0.6, 0.3], [0.1, 1.0]], dtype=float),
        known_future_covariates=np.ones((4, 3, 1), dtype=float) * 0.2,
        directed=True,
    )
    reversed_graph = GraphTemporalFrame(
        y=train,
        timestamps=timestamps,
        node_ids=["pickup", "midway", "dropoff"],
        edges=[(1, 0), (2, 1)],
        edge_weights=[1.0, 1.0],
        directed=True,
    )
    model = PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[2, 1]).fit(correct)
    reversed_model = PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[2, 1]).fit(
        reversed_graph
    )
    no_delay = PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[1, 1]).fit(correct)
    model_rmse = float(model.score(actual))
    reversed_rmse = float(reversed_model.score(actual))
    no_delay_rmse = float(no_delay.score(actual))
    return {
        "claim": "delay-aware graph beats reversed-edge/no-delay graph",
        "model_metric": model_rmse,
        "reversed_edge_metric": reversed_rmse,
        "no_delay_metric": no_delay_rmse,
        "passed": model_rmse < reversed_rmse and model_rmse < no_delay_rmse,
    }


def retrieval_claim() -> dict[str, Any]:
    contexts = []
    targets = []
    ids = []
    for idx in range(60):
        rare = 1.0 if idx % 17 == 0 else 0.0
        hour = float(idx % 24) / 23.0
        contexts.append([rare, hour])
        targets.append(10.0 + 8.0 * rare + np.sin(idx / 3.0))
        ids.append(f"a{idx}")
    query = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=float)
    actual = np.asarray([18.0, 18.0], dtype=float)
    model = RetrievalAugmentedForecaster(k=3).fit(ids, contexts, targets)
    report = model.rare_pattern_benchmark(query, actual)
    return {
        "claim": "retrieval beats no-retrieval on rare-pattern task",
        "model_metric": report["retrieval_rmse"],
        "baseline_metric": report["global_mean_rmse"],
        "passed": report["retrieval_rmse"] < report["global_mean_rmse"],
    }


def flow_claim() -> dict[str, Any]:
    steps = 32
    hidden = np.column_stack([np.linspace(0.0, 1.0, steps), np.sin(np.arange(steps) / 3.0)])
    residuals = 0.2 * hidden[:, 0] + 0.1 * np.sin(np.arange(steps))
    head = ConditionalFlowDistributionHead(quantiles=(0.05, 0.5, 0.95), sample_count=16).fit(
        residuals,
        model_hidden_state=hidden,
    )
    benchmark = head.benchmark_against_baselines(residuals, model_hidden_state=hidden)
    return {
        "claim": (
            "flow head improves calibration or sharpness over quantile/Gaussian/conformal baseline"
        ),
        "architecture": head.metadata_["architecture"],
        "model_metric": benchmark["flow_metrics"]["interval_width"],
        "baseline_metric": min(
            benchmark["independent_quantile_head"]["interval_width"],
            benchmark["gaussian_residual_head"]["interval_width"],
            benchmark["conformal_interval_wrapper"]["interval_width"],
        ),
        "passed": benchmark["flow_improves_calibration_or_sharpness"]
        and head.metadata_["architecture"] == "conditional_residual_sampler",
    }


def choice_claim() -> dict[str, Any]:
    candidates = [
        {
            "decision_id": "d1",
            "candidate_id": "a",
            "candidate_value": 1.0,
            "expected_utility": 2.0,
            "response_probability": 0.8,
            "candidate_features": [1.0, 0.0],
            "context_features": [0.5],
            "nest_id": "n",
            "chosen": True,
        },
        {
            "decision_id": "d1",
            "candidate_id": "b",
            "candidate_value": 1.5,
            "expected_utility": 0.2,
            "response_probability": 0.2,
            "candidate_features": [0.0, 1.0],
            "context_features": [0.5],
            "nest_id": "n",
            "chosen": False,
        },
    ]
    report = ChoiceSetTransformer(temperature=0.7, monotone_candidate_value="increasing").score(
        candidates
    )
    model_loss = report["benchmark"]["choice_set_log_loss"]
    baseline_loss = report["benchmark"]["independent_response_log_loss"]
    return {
        "claim": "choice set model beats independent candidate scoring when candidates compete",
        "architecture": report["metadata"]["architecture"],
        "model_metric": model_loss,
        "baseline_metric": baseline_loss,
        "passed": model_loss < baseline_loss
        and report["metadata"]["architecture"] == "choice_set_utility_softmax",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    claims = {
        "pair_embedding_mlp": pair_embedding_claim(),
        "selective_ssm_lite": ssm_claim(),
        "inverted_transformer": inverted_transformer_claim(),
        "delay_aware_graph": delay_graph_claim(),
        "retrieval_augmented": retrieval_claim(),
        "conditional_residual_sampler": flow_claim(),
        "choice_set_utility_softmax": choice_claim(),
    }
    payload = {
        "command": (
            "PYTHONPATH=python python scripts/run_deep_claim_benchmarks.py --output "
            "docs/assets/deep_claim_benchmarks/results.json"
        ),
        "data": "deterministic synthetic fixtures",
        "claims": claims,
        "all_passed": all(row["passed"] for row in claims.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
