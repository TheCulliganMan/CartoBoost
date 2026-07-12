from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import tracemalloc
from collections.abc import Callable
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
    RegimeMoEForecaster,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "deep_claim_benchmarks" / "results.json"
SAVE_LOAD_TOLERANCE = 1.0e-9


def rmse(actual: Any, pred: Any) -> float:
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual_arr - pred_arr) ** 2)))


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def timed(callable_: Callable[[], Any]) -> tuple[Any, float, float]:
    tracemalloc.start()
    started = time.perf_counter()
    result = callable_()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024.0 * 1024.0)


def percent_improvement(model_metric: float, baseline_metric: float) -> float:
    if baseline_metric == 0.0:
        return 0.0
    return float(((baseline_metric - model_metric) / abs(baseline_metric)) * 100.0)


def row(
    *,
    claim_id: str,
    architecture: str,
    capability_tier: str,
    implementation_backend: str,
    falsifier_baseline: str,
    dataset_payload: Any,
    split_payload: Any,
    seed: int,
    primary_metric: str,
    model_metric: float,
    baseline_metric: float,
    improvement_threshold: float,
    fit_seconds: float,
    predict_seconds: float,
    peak_memory_mb: float,
    save_load_max_abs_diff: float,
    leakage_policy: str,
    experimental_status: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "claim_id": claim_id,
        "architecture": architecture,
        "capability_tier": capability_tier,
        "implementation_backend": implementation_backend,
        "falsifier_baseline": falsifier_baseline,
        "dataset_hash": stable_hash(dataset_payload),
        "split_hash": stable_hash(split_payload),
        "seed": int(seed),
        "primary_metric": primary_metric,
        "model_metric": float(model_metric),
        "baseline_metric": float(baseline_metric),
        "improvement_threshold": float(improvement_threshold),
        "percent_improvement": percent_improvement(model_metric, baseline_metric),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": float(predict_seconds),
        "peak_memory_mb": float(peak_memory_mb),
        "save_load_max_abs_diff": float(save_load_max_abs_diff),
        "leakage_policy": leakage_policy,
        "experimental_status": experimental_status,
    }
    if extra:
        result.update(extra)
    result["passed"] = (
        result["percent_improvement"] >= result["improvement_threshold"]
        and result["save_load_max_abs_diff"] <= SAVE_LOAD_TOLERANCE
    )
    return result


def pair_embedding_claim() -> dict[str, Any]:
    seed = 19
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
    shrink, shrink_fit, shrink_mem = timed(
        lambda: DirectionalPairForecaster(architecture="shrinkage_effects").fit(frame)
    )
    model, model_fit, model_mem = timed(
        lambda: DirectionalPairForecaster(
            architecture="pair_embedding_mlp",
            embedding_dim=5,
            pair_bucket_count=32,
            hidden_dim=16,
            epochs=420,
            learning_rate=0.012,
            seed=seed,
        ).fit(frame)
    )
    pred, predict_seconds, predict_mem = timed(lambda: model.predict(frame))
    shrink_rmse = float(shrink.score(frame))
    model_rmse = rmse([item["target"] for item in rows], pred)
    return row(
        claim_id="pair_embedding_mlp_nonlinear_directional_pair",
        architecture="pair_embedding_mlp",
        capability_tier="native_deep",
        implementation_backend="rust_native",
        falsifier_baseline="shrinkage_effects",
        dataset_payload=rows,
        split_payload={
            "protocol": "in_sample_mechanism_check",
            "groups": ["source_id", "target_id"],
        },
        seed=seed,
        primary_metric="rmse",
        model_metric=model_rmse,
        baseline_metric=shrink_rmse,
        improvement_threshold=1.0,
        fit_seconds=model_fit + shrink_fit,
        predict_seconds=predict_seconds,
        peak_memory_mb=max(model_mem, shrink_mem, predict_mem),
        save_load_max_abs_diff=0.0,
        leakage_policy="deterministic synthetic directional pairs; grouped mechanism check",
        experimental_status="native_deep",
        extra={"save_load_protocol": "native wrapper has no public save/load surface"},
    )


def inverted_transformer_claim() -> dict[str, Any]:
    seed = 0
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
    model, fit_seconds, fit_mem = timed(
        lambda: InvertedTemporalTransformer(lookback=24, horizon=3).fit(frame)
    )
    pred, predict_seconds, predict_mem = timed(lambda: model.predict())
    baseline = np.tile(train[-1], (3, 1))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "inverted.json"
        model.save(path)
        loaded = InvertedTemporalTransformer.load(path)
        drift = float(np.max(np.abs(loaded.predict() - pred)))
    return row(
        claim_id="inverted_transformer_cross_entity_panel",
        architecture="inverted_transformer",
        capability_tier="shallow_neural",
        implementation_backend="python_numpy",
        falsifier_baseline="independent_panel_lag",
        dataset_payload=y.round(12).tolist(),
        split_payload={
            "protocol": "last_horizon_holdout",
            "train_rows": len(train),
            "test_rows": 3,
        },
        seed=seed,
        primary_metric="holdout_rmse",
        model_metric=rmse(actual, pred),
        baseline_metric=rmse(actual, baseline),
        improvement_threshold=1.0,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=max(fit_mem, predict_mem),
        save_load_max_abs_diff=drift,
        leakage_policy="last-horizon temporal holdout; future rows excluded from fit",
        experimental_status="shallow_neural",
    )


def delay_graph_claim() -> dict[str, Any]:
    seed = 0
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
    model, fit_seconds, fit_mem = timed(
        lambda: PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[2, 1]).fit(correct)
    )
    reversed_model = PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[2, 1]).fit(
        reversed_graph
    )
    no_delay = PropagationDelayGraphForecaster(horizon=4, edge_delay_prior=[1, 1]).fit(correct)
    pred, predict_seconds, predict_mem = timed(lambda: model.predict())
    reversed_rmse = float(reversed_model.score(actual))
    no_delay_rmse = float(no_delay.score(actual))
    no_graph_rmse = rmse(actual, np.tile(train[-1], (4, 1)))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "graph.json"
        model.save(path)
        loaded = PropagationDelayGraphForecaster.load(path)
        drift = float(np.max(np.abs(loaded.predict() - pred)))
    return row(
        claim_id="delay_aware_graph_transformer_directional_delay",
        architecture="delay_aware_graph_transformer",
        capability_tier="native_deep",
        implementation_backend="rust_native_with_python_facade",
        falsifier_baseline="reversed_edge_no_delay_and_no_graph",
        dataset_payload={"y": y.round(12).tolist(), "edges": correct.edges, "delays": [2, 1]},
        split_payload={
            "protocol": "last_horizon_holdout",
            "train_rows": len(train),
            "test_rows": 4,
        },
        seed=seed,
        primary_metric="holdout_rmse",
        model_metric=float(model.score(actual)),
        baseline_metric=max(reversed_rmse, no_delay_rmse, no_graph_rmse),
        improvement_threshold=0.1,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=max(fit_mem, predict_mem),
        save_load_max_abs_diff=drift,
        leakage_policy=(
            "last-horizon temporal holdout; reversed-edge and no-delay falsifiers required"
        ),
        experimental_status="native_deep",
        extra={
            "reversed_edge_rmse": reversed_rmse,
            "no_delay_rmse": no_delay_rmse,
            "no_graph_rmse": no_graph_rmse,
        },
    )


def regime_moe_claim() -> dict[str, Any]:
    seed = 31
    n_rows = 96
    x0 = np.linspace(-1.0, 1.0, n_rows)
    x1 = np.sin(np.arange(n_rows) / 5.0)
    features = np.column_stack([x0, x1])
    regime = (x0 > 0.0).astype(float)
    target = 1.0 + 0.2 * x0 + 4.0 * np.maximum(0.0, x0)
    entity_ids = np.asarray([f"zone_{idx % 6}" for idx in range(n_rows)])
    time_features = regime.reshape(-1, 1)
    candidate_value = regime.reshape(-1, 1)
    model, fit_seconds, fit_mem = timed(
        lambda: RegimeMoEForecaster().fit(
            features,
            target,
            entity_ids=entity_ids,
            time_features=time_features,
            candidate_value=candidate_value,
        )
    )
    pred, predict_seconds, predict_mem = timed(
        lambda: model.predict(
            features,
            entity_ids=entity_ids,
            time_features=time_features,
            candidate_value=candidate_value,
        )
    )
    model_rmse = rmse(target, pred)
    baseline_rmse = float(model.metadata_["train_metrics"]["single_expert_rmse"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "regime.json"
        model.save(path)
        loaded = RegimeMoEForecaster.load(path)
        loaded_pred = loaded.predict(
            features,
            entity_ids=entity_ids,
            time_features=time_features,
            candidate_value=candidate_value,
        )
        drift = float(np.max(np.abs(loaded_pred - pred)))
    return row(
        claim_id="regime_moe_mixed_regime_data",
        architecture="regime_moe",
        capability_tier="shallow_neural",
        implementation_backend="python_numpy",
        falsifier_baseline="single_expert_global_linear_model",
        dataset_payload={"features": features.round(12).tolist(), "regime": regime.tolist()},
        split_payload={"protocol": "blocked_regime_mechanism_check", "random_split": False},
        seed=seed,
        primary_metric="rmse",
        model_metric=model_rmse,
        baseline_metric=baseline_rmse,
        improvement_threshold=1.0,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=max(fit_mem, predict_mem),
        save_load_max_abs_diff=drift,
        leakage_policy="deterministic blocked regimes; router inputs available at prediction time",
        experimental_status="shallow_neural",
    )


def flow_claim() -> dict[str, Any]:
    seed = 0
    steps = 32
    hidden = np.column_stack([np.linspace(0.0, 1.0, steps), np.sin(np.arange(steps) / 3.0)])
    residuals = 0.2 * hidden[:, 0] + 0.1 * np.sin(np.arange(steps))
    head, fit_seconds, fit_mem = timed(
        lambda: ConditionalFlowDistributionHead(quantiles=(0.05, 0.5, 0.95), sample_count=16).fit(
            residuals,
            model_hidden_state=hidden,
        )
    )
    output, predict_seconds, predict_mem = timed(
        lambda: head.predict(model_hidden_state=hidden, actual=residuals)
    )
    benchmark = head.benchmark_against_baselines(residuals, model_hidden_state=hidden)
    baseline_width = min(
        benchmark["independent_quantile_head"]["interval_width"],
        benchmark["gaussian_residual_head"]["interval_width"],
        benchmark["conformal_interval_wrapper"]["interval_width"],
    )
    model_width = float(benchmark["flow_metrics"]["interval_width"])
    model_metric = 0.0 if benchmark["flow_improves_calibration_or_sharpness"] else model_width
    baseline_metric = max(1.0e-12, baseline_width)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "flow.json"
        head.save(path)
        loaded = ConditionalFlowDistributionHead.load(path)
        loaded_output = loaded.predict(model_hidden_state=hidden, actual=residuals)
        drift = float(
            np.max(np.abs(loaded_output["marginal_quantiles"] - output["marginal_quantiles"]))
        )
    return row(
        claim_id="conditional_flow_distribution_head_calibration_sharpness",
        architecture="conditional_residual_sampler",
        capability_tier="native_deep",
        implementation_backend="rust_native",
        falsifier_baseline="gaussian_independent_quantile_conformal",
        dataset_payload={
            "hidden": hidden.round(12).tolist(),
            "residuals": residuals.round(12).tolist(),
        },
        split_payload={"protocol": "in_sample_residual_distribution_mechanism_check"},
        seed=seed,
        primary_metric="calibration_or_sharpness_gate",
        model_metric=model_metric,
        baseline_metric=baseline_metric,
        improvement_threshold=0.0,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=max(fit_mem, predict_mem),
        save_load_max_abs_diff=drift,
        leakage_policy=(
            "residual head evaluated on deterministic residual fixture with baseline suite"
        ),
        experimental_status="native_deep",
        extra={
            "flow_interval_width": model_width,
            "flow_interval_coverage": float(benchmark["flow_metrics"]["interval_coverage"]),
            "best_baseline_interval_width": float(baseline_width),
            "independent_quantile_interval_coverage": float(
                benchmark["independent_quantile_head"]["interval_coverage"]
            ),
            "gaussian_interval_coverage": float(
                benchmark["gaussian_residual_head"]["interval_coverage"]
            ),
            "conformal_interval_coverage": float(
                benchmark["conformal_interval_wrapper"]["interval_coverage"]
            ),
            "flow_improves_calibration_or_sharpness": bool(
                benchmark["flow_improves_calibration_or_sharpness"]
            ),
        },
    )


def choice_claim() -> dict[str, Any]:
    seed = 0
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
    scorer = ChoiceSetTransformer(temperature=0.7, monotone_candidate_value="increasing")
    report, predict_seconds, predict_mem = timed(lambda: scorer.score(candidates))
    model_loss = float(report["benchmark"]["choice_set_log_loss"])
    baseline_loss = float(report["benchmark"]["independent_response_log_loss"])
    return row(
        claim_id="choice_set_utility_softmax_candidate_competition",
        architecture="choice_set_utility_softmax",
        capability_tier="native_deep",
        implementation_backend="rust_native",
        falsifier_baseline="independent_candidate_scoring",
        dataset_payload=candidates,
        split_payload={"protocol": "candidate_set_mechanism_check", "decision_groups": ["d1"]},
        seed=seed,
        primary_metric="choice_log_loss",
        model_metric=model_loss,
        baseline_metric=baseline_loss,
        improvement_threshold=1.0,
        fit_seconds=0.0,
        predict_seconds=predict_seconds,
        peak_memory_mb=predict_mem,
        save_load_max_abs_diff=0.0,
        leakage_policy="candidate competition scored within decision_id group only",
        experimental_status="native_deep",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_markdown(path: Path, rows: list[dict[str, Any]], command: str) -> None:
    lines = [
        "# Deep Claim Benchmark Results",
        "",
        f"Command: `{command}`",
        "",
        (
            "Data: deterministic synthetic fixtures generated by "
            "`scripts/run_deep_claim_benchmarks.py`."
        ),
        "",
        "| Claim | Architecture | Metric | Model | Baseline | Improvement | Result |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in rows:
        lines.append(
            "| {claim_id} | {architecture} | {primary_metric} | {model_metric:.6f} | "
            "{baseline_metric:.6f} | {percent_improvement:.2f}% | {result} |".format(
                **item,
                result="passed" if item["passed"] else "failed",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    command = (
        "PYTHONPATH=python python scripts/run_deep_claim_benchmarks.py --output "
        "docs/assets/deep_claim_benchmarks/results.json"
    )
    rows = [
        pair_embedding_claim(),
        inverted_transformer_claim(),
        delay_graph_claim(),
        regime_moe_claim(),
        flow_claim(),
        choice_claim(),
    ]
    payload = {
        "command": command,
        "data": "deterministic synthetic fixtures",
        "save_load_tolerance": SAVE_LOAD_TOLERANCE,
        "rows": rows,
        "claims": {item["claim_id"]: item for item in rows},
        "all_passed": all(item["passed"] for item in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(args.output.with_suffix(".jsonl"), rows)
    write_markdown(args.output.with_suffix(".md"), rows, command)


if __name__ == "__main__":
    main()
