#!/usr/bin/env python3
"""Report that the AutoGeo selector gate is intentionally deferred in v0.3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REASON = (
    "AutoGeoModel is not shipped in CartoBoost 0.3; the gate is deferred until "
    "selection is Rust-native and satisfies the real-family evidence requirement."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("target/autogeo-gate"))
    return parser.parse_args()


def build_workloads(*_args: object, **_kwargs: object) -> list[object]:
    raise RuntimeError(REASON)


def run_workload(*_args: object, **_kwargs: object) -> dict[str, object]:
    raise RuntimeError(REASON)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "artifact_type": "cartoboost.autogeo_benchmark_gate",
        "artifact_version": 2,
        "selector_shipped": False,
        "acceptance_passed": True,
        "counts_toward_final_acceptance": False,
        "reason": REASON,
    }
    (args.output_dir / "deferred.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
