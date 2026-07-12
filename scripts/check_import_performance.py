#!/usr/bin/env python3
"""Check that importing the stable package stays below the v0.3 budget."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def measure_import_ms() -> float:
    code = (
        "import time; started=time.perf_counter(); import cartoboost; "
        "print((time.perf_counter()-started)*1000)"
    )
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT / "python"))
    output = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, env=env, text=True)
    return float(output.strip().splitlines()[-1])


def check(*, maximum_ms: float = 350.0) -> dict[str, Any]:
    measured_ms = measure_import_ms()
    return {
        "artifact_type": "cartoboost.import_performance_audit",
        "artifact_version": 1,
        "maximum_ms": maximum_ms,
        "measured_ms": measured_ms,
        "passed": measured_ms < maximum_ms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maximum-ms", type=float, default=350.0)
    args = parser.parse_args()
    report = check(maximum_ms=args.maximum_ms)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("import performance gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
