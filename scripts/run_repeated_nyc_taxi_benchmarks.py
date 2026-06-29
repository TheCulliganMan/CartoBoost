#!/usr/bin/env python3
"""Run repeated NYC taxi benchmarks and summarize CartoBoost speed ratios."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.runners.significance import normal_mean_ci, paired_bootstrap_ci  # noqa: E402

BENCHMARK_SCRIPT = ROOT / "scripts" / "run_nyc_taxi_quality_benchmarks.py"
DEFAULT_RUN_DIR = ROOT / "target" / "nyc_taxi_repeated"
DEFAULT_SUMMARY_JSON = ROOT / "docs" / "assets" / "nyc_taxi_benchmarks" / "repeated_results.json"
DEFAULT_SUMMARY_MD = ROOT / "docs" / "assets" / "nyc_taxi_benchmarks" / "repeated_results.md"
DEFAULT_SEEDS_CONFIG = ROOT / "benchmarks" / "configs" / "seeds.json"
EXTERNAL_REGRESSION_BASELINES = {
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "random_forest",
    "extra_trees",
    "ridge",
    "mean",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", default="1")
    parser.add_argument("--sample-size", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds-config",
        type=Path,
        default=DEFAULT_SEEDS_CONFIG,
        help="Benchmark seed config. Used for repeated runs when available.",
    )
    parser.add_argument("--tasks", default="")
    parser.add_argument(
        "--models",
        default=(
            "cartoboost,lightgbm,xgboost,catboost,hist_gradient_boosting,"
            "random_forest,extra_trees,ridge,mean"
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--cartoboost-n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--cartoboost-max-depth", type=int, default=5)
    parser.add_argument(
        "--cartoboost-splitters",
        default="axis_histogram:512,periodic:24,periodic:7,sparse_set",
    )
    parser.add_argument("--cartoboost-min-samples-leaf", type=int, default=1)
    parser.add_argument("--cartoboost-constant-l2", type=float, default=0.0)
    parser.add_argument(
        "--cartoboost-leaf-predictor", default="constant", choices=["constant", "linear"]
    )
    parser.add_argument("--cartoboost-init", default="constant", choices=["constant", "linear"])
    parser.add_argument("--cartoboost-calibration", default="none", choices=["none", "affine"])
    parser.add_argument(
        "--xgboost-tree-method",
        default="hist",
        choices=["auto", "exact", "approx", "hist"],
    )
    parser.add_argument("--xgboost-max-bin", type=int, default=256)
    parser.add_argument("--xgboost-subsample", type=float, default=1.0)
    parser.add_argument("--xgboost-colsample-bytree", type=float, default=1.0)
    parser.add_argument("--zone-treatment", default="target_mean", choices=["raw", "target_mean"])
    parser.add_argument("--zone-target-smoothing", type=float, default=20.0)
    parser.add_argument("--model-workers", type=int, default=1)
    parser.add_argument("--n-threads", type=int, default=1)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument(
        "--profile-fit",
        action="store_true",
        help="Set CARTOBOOST_PROFILE_FIT=1 for each benchmark subprocess.",
    )
    return parser.parse_args()


def benchmark_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--models",
        args.models,
        "--n-estimators",
        str(args.n_estimators),
        "--cartoboost-n-estimators",
        str(args.cartoboost_n_estimators),
        "--learning-rate",
        str(args.learning_rate),
        "--max-depth",
        str(args.max_depth),
        "--cartoboost-max-depth",
        str(args.cartoboost_max_depth),
        "--cartoboost-splitters",
        args.cartoboost_splitters,
        "--cartoboost-min-samples-leaf",
        str(args.cartoboost_min_samples_leaf),
        "--cartoboost-constant-l2",
        str(args.cartoboost_constant_l2),
        "--cartoboost-leaf-predictor",
        args.cartoboost_leaf_predictor,
        "--cartoboost-init",
        args.cartoboost_init,
        "--cartoboost-calibration",
        args.cartoboost_calibration,
        "--xgboost-tree-method",
        args.xgboost_tree_method,
        "--xgboost-max-bin",
        str(args.xgboost_max_bin),
        "--xgboost-subsample",
        str(args.xgboost_subsample),
        "--xgboost-colsample-bytree",
        str(args.xgboost_colsample_bytree),
        "--zone-treatment",
        args.zone_treatment,
        "--zone-target-smoothing",
        str(args.zone_target_smoothing),
        "--model-workers",
        str(args.model_workers),
        "--n-threads",
        str(args.n_threads),
        "--seed",
        str(args.seed),
    ]
    if args.tasks:
        command.extend(["--tasks", args.tasks])
    if args.synthetic_smoke:
        command.append("--synthetic-smoke")
    else:
        command.extend(
            [
                "--year",
                str(args.year),
                "--months",
                args.months,
                "--sample-size",
                str(args.sample_size),
            ]
        )
    if args.no_download:
        command.append("--no-download")
    if args.no_plots:
        command.append("--no-plots")
    return command


def benchmark_seeds(args: argparse.Namespace) -> list[int]:
    if args.runs == 1:
        return [int(args.seed)]
    if args.seeds_config.exists():
        config = json.loads(args.seeds_config.read_text(encoding="utf-8"))
        seeds = [int(seed) for seed in config.get("benchmark_seeds", [])]
        if seeds:
            return seeds[: args.runs]
    return [int(args.seed) + run_index - 1 for run_index in range(1, args.runs + 1)]


def run_once(args: argparse.Namespace, run_index: int, seed: int) -> dict[str, Any]:
    output_dir = args.run_dir / f"run_{run_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    env = None
    if args.profile_fit:
        env = {**os.environ, "CARTOBOOST_PROFILE_FIT": "1"}
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = seed
    subprocess.run(benchmark_command(run_args, output_dir), cwd=ROOT, check=True, env=env)
    return json.loads((output_dir / "results.json").read_text(encoding="utf-8"))


def collect_ratios(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, float]]] = {}
    for run_index, result in enumerate(results, start=1):
        for task_name, task in result["tasks"].items():
            for split_name, split in task["splits"].items():
                models = split["models"]
                cartoboost = models.get("cartoboost")
                if cartoboost is None or cartoboost["status"] != "ok":
                    continue
                xgboost = models.get("xgboost")
                if xgboost is None or xgboost["status"] != "ok":
                    continue
                cartoboost_reference = models.get("cartoboost_reference")
                carto_timing = cartoboost["timing"]
                xgb_timing = xgboost["timing"]
                carto_metrics = cartoboost["metrics"]
                xgb_metrics = xgboost["metrics"]
                reference_metrics = (
                    cartoboost_reference["metrics"]
                    if cartoboost_reference is not None and cartoboost_reference["status"] == "ok"
                    else None
                )
                key = f"{task_name}/{split_name}"
                row = {
                    "run": float(run_index),
                    "cartoboost_train_seconds": float(carto_timing["train_seconds"]),
                    "cartoboost_predict_rows_per_second": float(
                        carto_timing["predict_rows_per_second"]
                    ),
                    "xgboost_train_seconds": float(xgb_timing["train_seconds"]),
                    "xgboost_predict_rows_per_second": float(xgb_timing["predict_rows_per_second"]),
                    "train_ratio_vs_xgboost": float(carto_timing["train_seconds"])
                    / float(xgb_timing["train_seconds"]),
                    "predict_rps_ratio_vs_xgboost": float(carto_timing["predict_rows_per_second"])
                    / float(xgb_timing["predict_rows_per_second"]),
                    "rmse_delta_vs_xgboost": float(carto_metrics["rmse"])
                    - float(xgb_metrics["rmse"]),
                    "r2_delta_vs_xgboost": float(carto_metrics["r2"]) - float(xgb_metrics["r2"]),
                }
                if reference_metrics is not None:
                    row["rmse_delta_vs_cartoboost_reference"] = float(
                        carto_metrics["rmse"]
                    ) - float(reference_metrics["rmse"])
                    row["r2_delta_vs_cartoboost_reference"] = float(carto_metrics["r2"]) - float(
                        reference_metrics["r2"]
                    )
                rows.setdefault(key, []).append(row)
    return {
        key: {
            "runs": values,
            "median_train_ratio_vs_xgboost": statistics.median(
                item["train_ratio_vs_xgboost"] for item in values
            ),
            "min_train_ratio_vs_xgboost": min(item["train_ratio_vs_xgboost"] for item in values),
            "max_train_ratio_vs_xgboost": max(item["train_ratio_vs_xgboost"] for item in values),
            "median_predict_rps_ratio_vs_xgboost": statistics.median(
                item["predict_rps_ratio_vs_xgboost"] for item in values
            ),
            "min_predict_rps_ratio_vs_xgboost": min(
                item["predict_rps_ratio_vs_xgboost"] for item in values
            ),
            "max_predict_rps_ratio_vs_xgboost": max(
                item["predict_rps_ratio_vs_xgboost"] for item in values
            ),
            "median_rmse_delta_vs_xgboost": statistics.median(
                item["rmse_delta_vs_xgboost"] for item in values
            ),
            "median_r2_delta_vs_xgboost": statistics.median(
                item["r2_delta_vs_xgboost"] for item in values
            ),
            **reference_delta_summary(values),
        }
        for key, values in sorted(rows.items())
    }


def collect_quality(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    win_counts: dict[str, dict[str, int]] = {}
    best_external_rows: dict[str, list[dict[str, float | str]]] = {}
    for result in results:
        for task_name, task in result["tasks"].items():
            for split_name, split in task["splits"].items():
                key = f"{task_name}/{split_name}"
                ok_models = {
                    model_name: model
                    for model_name, model in split["models"].items()
                    if model.get("status") == "ok"
                }
                if not ok_models:
                    continue
                cartoboost = ok_models.get("cartoboost")
                external_candidates = {
                    model_name: model
                    for model_name, model in ok_models.items()
                    if model_name in EXTERNAL_REGRESSION_BASELINES
                }
                if cartoboost is not None and external_candidates:
                    best_external_name, best_external = min(
                        external_candidates.items(),
                        key=lambda item: float(item[1]["metrics"]["rmse"]),
                    )
                    best_external_rows.setdefault(key, []).append(
                        {
                            "best_external_model": best_external_name,
                            "cartoboost_rmse": float(cartoboost["metrics"]["rmse"]),
                            "external_rmse": float(best_external["metrics"]["rmse"]),
                            "cartoboost_r2": float(cartoboost["metrics"]["r2"]),
                            "external_r2": float(best_external["metrics"]["r2"]),
                            "cartoboost_wape": float(cartoboost["metrics"].get("wape", 0.0)),
                            "external_wape": float(best_external["metrics"].get("wape", 0.0)),
                        }
                    )
                best_rmse = min(float(model["metrics"]["rmse"]) for model in ok_models.values())
                for model_name, model in ok_models.items():
                    model_rows = rows.setdefault(key, {}).setdefault(
                        model_name,
                        {
                            "metrics": {"rmse": [], "mae": [], "r2": [], "wape": []},
                            "timing": {
                                "train_seconds": [],
                                "predict_seconds": [],
                                "predict_rows_per_second": [],
                            },
                        },
                    )
                    for metric in ["rmse", "mae", "r2", "wape"]:
                        if metric in model["metrics"]:
                            model_rows["metrics"][metric].append(float(model["metrics"][metric]))
                    for timing_name in [
                        "train_seconds",
                        "predict_seconds",
                        "predict_rows_per_second",
                    ]:
                        model_rows["timing"][timing_name].append(
                            float(model["timing"][timing_name])
                        )
                    if float(model["metrics"]["rmse"]) == best_rmse:
                        win_counts.setdefault(key, {})[model_name] = (
                            win_counts.setdefault(key, {}).get(model_name, 0) + 1
                        )

    summary: dict[str, Any] = {}
    for key, model_rows in sorted(rows.items()):
        summary[key] = {"models": {}, "paired_deltas": {}}
        if key in best_external_rows:
            comparison_rows = best_external_rows[key]
            model_counts: dict[str, int] = {}
            for row in comparison_rows:
                model_name = str(row["best_external_model"])
                model_counts[model_name] = model_counts.get(model_name, 0) + 1
            rmse_delta = paired_bootstrap_ci(
                [float(row["cartoboost_rmse"]) for row in comparison_rows],
                [float(row["external_rmse"]) for row in comparison_rows],
                seed=11,
            )
            r2_delta = paired_bootstrap_ci(
                [float(row["cartoboost_r2"]) for row in comparison_rows],
                [float(row["external_r2"]) for row in comparison_rows],
                seed=11,
            )
            wape_delta = paired_bootstrap_ci(
                [float(row["cartoboost_wape"]) for row in comparison_rows],
                [float(row["external_wape"]) for row in comparison_rows],
                seed=11,
            )
            summary[key]["primary_vs_best_external"] = {
                "n": len(comparison_rows),
                "best_external_model_counts": dict(sorted(model_counts.items())),
                "rmse_delta_mean": rmse_delta[0],
                "rmse_delta_ci95_low": rmse_delta[1],
                "rmse_delta_ci95_high": rmse_delta[2],
                "r2_delta_mean": r2_delta[0],
                "r2_delta_ci95_low": r2_delta[1],
                "r2_delta_ci95_high": r2_delta[2],
                "wape_delta_mean": wape_delta[0],
                "wape_delta_ci95_low": wape_delta[1],
                "wape_delta_ci95_high": wape_delta[2],
            }
        for model_name, payload in sorted(model_rows.items()):
            summary[key]["models"][model_name] = {
                "metrics": {
                    metric: ci_payload(values)
                    for metric, values in sorted(payload["metrics"].items())
                    if values
                },
                "timing": {
                    timing_name: median_payload(values)
                    for timing_name, values in sorted(payload["timing"].items())
                },
                "rmse_wins_or_ties": win_counts.get(key, {}).get(model_name, 0),
            }
        for baseline in ["lightgbm", "xgboost"]:
            if baseline not in model_rows:
                continue
            baseline_rmse = model_rows[baseline]["metrics"]["rmse"]
            baseline_r2 = model_rows[baseline]["metrics"]["r2"]
            for challenger in sorted(name for name in model_rows if name.startswith("cartoboost")):
                challenger_rmse = model_rows[challenger]["metrics"]["rmse"]
                challenger_r2 = model_rows[challenger]["metrics"]["r2"]
                if len(challenger_rmse) != len(baseline_rmse):
                    continue
                rmse_delta = paired_bootstrap_ci(challenger_rmse, baseline_rmse, seed=11)
                r2_delta = paired_bootstrap_ci(challenger_r2, baseline_r2, seed=11)
                summary[key]["paired_deltas"][f"{challenger}_vs_{baseline}"] = {
                    "rmse_delta_mean": rmse_delta[0],
                    "rmse_delta_ci95_low": rmse_delta[1],
                    "rmse_delta_ci95_high": rmse_delta[2],
                    "r2_delta_mean": r2_delta[0],
                    "r2_delta_ci95_low": r2_delta[1],
                    "r2_delta_ci95_high": r2_delta[2],
                }
    return summary


def ci_payload(values: list[float]) -> dict[str, float | int]:
    center, low, high = normal_mean_ci(values)
    return {"n": len(values), "mean": center, "ci95_low": low, "ci95_high": high}


def median_payload(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def reference_delta_summary(values: list[dict[str, float]]) -> dict[str, float | None]:
    rmse = [
        item["rmse_delta_vs_cartoboost_reference"]
        for item in values
        if "rmse_delta_vs_cartoboost_reference" in item
    ]
    r2 = [
        item["r2_delta_vs_cartoboost_reference"]
        for item in values
        if "r2_delta_vs_cartoboost_reference" in item
    ]
    return {
        "median_rmse_delta_vs_cartoboost_reference": statistics.median(rmse) if rmse else None,
        "median_r2_delta_vs_cartoboost_reference": statistics.median(r2) if r2 else None,
    }


def format_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def format_interval(low: float, high: float) -> str:
    return f"{low:.6f} to {high:.6f}"


def same_quality(row: dict[str, Any], tolerance: float = 1e-9) -> bool:
    rmse_ref = row["median_rmse_delta_vs_cartoboost_reference"]
    r2_ref = row["median_r2_delta_vs_cartoboost_reference"]
    if rmse_ref is None or r2_ref is None:
        return False
    return abs(rmse_ref) <= tolerance and abs(r2_ref) <= tolerance


def beats_xgboost_quality(row: dict[str, Any]) -> bool:
    return row["median_rmse_delta_vs_xgboost"] < 0.0 and row["median_r2_delta_vs_xgboost"] >= 0.0


def passes_xgboost_gate(row: dict[str, Any]) -> bool:
    return (
        row["median_train_ratio_vs_xgboost"] <= 1.0
        and row["median_predict_rps_ratio_vs_xgboost"] >= 1.0
        and beats_xgboost_quality(row)
    )


def repeated_comparability_audit(results: list[dict[str, Any]]) -> dict[str, Any]:
    child_audits = [result.get("comparability_audit", {}) for result in results]
    completed_external = sorted(
        set.intersection(
            *[
                set(audit.get("completed_external_baselines", []))
                for audit in child_audits
                if audit.get("completed_external_baselines")
            ]
        )
        if child_audits
        else set()
    )
    skipped_external = sorted(
        {
            model
            for audit in child_audits
            for model in audit.get("skipped_requested_external_baselines", [])
        }
    )
    completed_cartoboost = sorted(
        set.intersection(
            *[
                set(audit.get("completed_cartoboost_family_models", []))
                for audit in child_audits
                if audit.get("completed_cartoboost_family_models")
            ]
        )
        if child_audits
        else set()
    )
    return {
        "run_count": len(results),
        "same_outer_splits": all(audit.get("same_outer_splits") is True for audit in child_audits),
        "selection_uses_outer_test_labels": any(
            audit.get("selection_uses_outer_test_labels") is True for audit in child_audits
        ),
        "same_feature_access_policy": all(
            audit.get("same_feature_access_policy") is True for audit in child_audits
        ),
        "completed_external_baselines_all_runs": completed_external,
        "skipped_requested_external_baselines_any_run": skipped_external,
        "completed_cartoboost_family_models_all_runs": completed_cartoboost,
        "cartoboost_external_comparison_rows": sum(
            int(audit.get("cartoboost_external_comparison_count", 0)) for audit in child_audits
        ),
    }


def write_markdown(summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# Repeated NYC Taxi Benchmark",
        "",
        (
            "This report reruns the maintained NYC taxi benchmark and summarizes quality "
            "confidence intervals, paired baseline deltas, and speed ratios."
        ),
        "",
        f"- runs: {summary['runs']}",
        f"- seeds: {', '.join(str(seed) for seed in summary['seeds'])}",
        f"- command arguments: `{' '.join(summary.get('command_argv', []))}`",
        f"- run artifacts: `{summary.get('run_dir', '')}`",
        f"- models: {', '.join(summary.get('model_roster', []))}",
        f"- sample size: {summary.get('sample_size', 'synthetic_smoke')}",
        (
            f"- baseline estimators: {summary['model_config']['baseline_n_estimators']}; "
            f"CartoBoost candidate estimators: {summary['model_config']['cartoboost_n_estimators']}"
        ),
        (
            f"- baseline max depth: {summary['model_config']['baseline_max_depth']}; "
            f"CartoBoost candidate max depth: {summary['model_config']['cartoboost_max_depth']}"
        ),
        (
            f"- CartoBoost splitters: {summary['model_config']['cartoboost_splitters']}; "
            f"XGBoost tree_method: {summary['model_config']['xgboost_tree_method']}"
        ),
        f"- zone treatment: {summary['model_config'].get('zone_treatment', 'raw')}",
        "- quality summary reports every model that finished for each task/split.",
        "",
    ]
    artifacts = summary.get("output_artifacts", {})
    if artifacts:
        lines.extend(
            [
                "## Output Artifacts",
                "",
                "| Artifact | Size bytes |",
                "| --- | ---: |",
            ]
        )
        for name, metadata in sorted(artifacts.items()):
            lines.append(f"| `{name}` | {metadata['size_bytes']} |")
        lines.append("")
    comparability = summary.get("comparability_audit", {})
    if comparability:
        lines.extend(
            [
                "## Comparability Audit",
                "",
                "| Check | Value |",
                "| --- | --- |",
                f"| Run count | {comparability['run_count']} |",
                f"| Same outer splits in every child run | {comparability['same_outer_splits']} |",
                (
                    "| Any outer test-label selection | "
                    f"{comparability['selection_uses_outer_test_labels']} |"
                ),
                (
                    "| Same feature-access policy in every child run | "
                    f"{comparability['same_feature_access_policy']} |"
                ),
                (
                    "| Completed external baselines in all runs | "
                    f"`{', '.join(comparability['completed_external_baselines_all_runs'])}` |"
                ),
                (
                    "| Skipped requested external baselines in any run | "
                    "`"
                    f"{', '.join(comparability['skipped_requested_external_baselines_any_run'])}"
                    "` |"
                ),
                (
                    "| Completed CartoBoost-family rows in all runs | "
                    f"`{', '.join(comparability['completed_cartoboost_family_models_all_runs'])}` |"
                ),
                (
                    "| CartoBoost/external comparison rows | "
                    f"{comparability['cartoboost_external_comparison_rows']} |"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Quality Summary",
            "",
            (
                "| task/split | model | RMSE mean | RMSE 95% CI | MAE mean | R2 mean | "
                "RMSE wins/ties | train median sec | predict rows/sec median |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, quality in summary["quality"].items():
        for model_name, model in quality["models"].items():
            rmse = model["metrics"]["rmse"]
            mae = model["metrics"]["mae"]
            r2 = model["metrics"]["r2"]
            train = model["timing"]["train_seconds"]
            throughput = model["timing"]["predict_rows_per_second"]
            lines.append(
                f"| {key} | {model_name} | {rmse['mean']:.6f} | "
                f"{format_interval(rmse['ci95_low'], rmse['ci95_high'])} | "
                f"{mae['mean']:.6f} | {r2['mean']:.6f} | "
                f"{model['rmse_wins_or_ties']} | {train['median']:.6f} | "
                f"{throughput['median']:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Paired Model Deltas",
            "",
            "Negative RMSE deltas favor the CartoBoost-family row. Positive R2 deltas favor it.",
            "",
            (
                "| task/split | comparison | RMSE delta mean | RMSE delta 95% CI | "
                "R2 delta mean | R2 delta 95% CI |"
            ),
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, quality in summary["quality"].items():
        for comparison, deltas in quality["paired_deltas"].items():
            rmse_ci = format_interval(
                deltas["rmse_delta_ci95_low"],
                deltas["rmse_delta_ci95_high"],
            )
            lines.append(
                f"| {key} | {comparison} | {deltas['rmse_delta_mean']:.6f} | "
                f"{rmse_ci} | "
                f"{deltas['r2_delta_mean']:.6f} | "
                f"{format_interval(deltas['r2_delta_ci95_low'], deltas['r2_delta_ci95_high'])} |"
            )
    lines.extend(
        [
            "",
            "## Speed Ratios",
            "",
            "| task/split | train ratio vs XGBoost median | train ratio min-max | "
            "predict rps ratio vs XGBoost median | predict rps ratio min-max | "
            "RMSE delta vs Carto ref | R2 delta vs Carto ref | "
            "RMSE delta vs XGB | R2 delta vs XGB | gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for key, row in summary["ratios"].items():
        train_median = row["median_train_ratio_vs_xgboost"]
        predict_median = row["median_predict_rps_ratio_vs_xgboost"]
        gate = "pass" if passes_xgboost_gate(row) else "miss"
        rmse_ref = row["median_rmse_delta_vs_cartoboost_reference"]
        r2_ref = row["median_r2_delta_vs_cartoboost_reference"]
        lines.append(
            f"| {key} | {train_median:.2f}x | "
            f"{row['min_train_ratio_vs_xgboost']:.2f}x-{row['max_train_ratio_vs_xgboost']:.2f}x | "
            f"{predict_median:.3f}x | "
            f"{row['min_predict_rps_ratio_vs_xgboost']:.3f}x-"
            f"{row['max_predict_rps_ratio_vs_xgboost']:.3f}x | "
            f"{format_delta(rmse_ref)} | "
            f"{format_delta(r2_ref)} | "
            f"{row['median_rmse_delta_vs_xgboost']:.6f} | "
            f"{row['median_r2_delta_vs_xgboost']:.6f} | {gate} |"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be positive")
    seeds = benchmark_seeds(args)
    if len(seeds) < args.runs:
        raise SystemExit(
            f"requested {args.runs} runs but only {len(seeds)} benchmark seeds are available"
        )
    args.run_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_once(args, run_index, seed)
        for run_index, seed in enumerate(seeds[: args.runs], start=1)
    ]
    ratios = collect_ratios(results)
    quality = collect_quality(results)
    summary = {
        "artifact_version": 1,
        "command_argv": list(sys.argv),
        "runs": args.runs,
        "seeds": seeds[: args.runs],
        "run_dir": str(args.run_dir),
        "model_roster": [part.strip() for part in args.models.split(",") if part.strip()],
        "tasks": [part.strip() for part in args.tasks.split(",") if part.strip()],
        "sample_size": None if args.synthetic_smoke else int(args.sample_size),
        "synthetic_smoke": bool(args.synthetic_smoke),
        "no_download": bool(args.no_download),
        "no_plots": bool(args.no_plots),
        "run_datasets": [
            {
                "run": run_index,
                "dataset": result.get("dataset", {}),
                "dataset_hash": result.get("dataset_hash"),
                "git_commit": result.get("git_commit"),
            }
            for run_index, result in enumerate(results, start=1)
        ],
        "target": {
            "train_ratio_vs_xgboost_max": 1.0,
            "predict_rps_ratio_vs_xgboost_min": 1.0,
            "rmse_delta_vs_xgboost_max_exclusive": 0.0,
            "r2_delta_vs_xgboost_min": 0.0,
            "quality_delta_vs_cartoboost_reference_abs_max": 1e-9,
        },
        "model_config": {
            "baseline_n_estimators": args.n_estimators,
            "cartoboost_n_estimators": args.cartoboost_n_estimators,
            "baseline_max_depth": args.max_depth,
            "cartoboost_max_depth": args.cartoboost_max_depth,
            "cartoboost_splitters": args.cartoboost_splitters,
            "cartoboost_min_samples_leaf": args.cartoboost_min_samples_leaf,
            "cartoboost_constant_l2": args.cartoboost_constant_l2,
            "cartoboost_leaf_predictor": args.cartoboost_leaf_predictor,
            "cartoboost_init": args.cartoboost_init,
            "cartoboost_calibration": args.cartoboost_calibration,
            "xgboost_tree_method": args.xgboost_tree_method,
            "xgboost_max_bin": args.xgboost_max_bin,
            "xgboost_subsample": args.xgboost_subsample,
            "xgboost_colsample_bytree": args.xgboost_colsample_bytree,
            "zone_treatment": args.zone_treatment,
            "zone_target_smoothing": args.zone_target_smoothing,
        },
        "ratios": ratios,
        "quality": quality,
        "comparability_audit": repeated_comparability_audit(results),
    }
    summary["all_gates_pass"] = all(passes_xgboost_gate(row) for row in ratios.values())
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown(summary, args.summary_md)
    for _ in range(5):
        manifest = output_artifact_manifest(args.summary_json, args.summary_md)
        if manifest == summary.get("output_artifacts"):
            break
        summary["output_artifacts"] = manifest
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        write_markdown(summary, args.summary_md)


def output_artifact_manifest(*paths: Path) -> dict[str, dict[str, int]]:
    artifacts: dict[str, dict[str, int]] = {}
    for path in paths:
        if path.exists():
            artifacts[str(path)] = {"size_bytes": int(path.stat().st_size)}
    return artifacts


if __name__ == "__main__":
    main()
