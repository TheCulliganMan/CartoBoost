from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "assets" / "deep_claim_benchmarks" / "results.json"
REQUIRED = {
    "pair_embedding_mlp",
    "selective_ssm_lite",
    "inverted_transformer",
    "delay_aware_graph",
    "retrieval_augmented",
    "conditional_residual_sampler",
    "choice_set_utility_softmax",
}


def main() -> int:
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    claims = payload.get("claims", {})
    errors = []
    for name in sorted(REQUIRED - set(claims)):
        errors.append(f"missing deep claim result: {name}")
    for name in sorted(REQUIRED & set(claims)):
        if not claims[name].get("passed"):
            errors.append(f"deep claim gate failed: {name}")
    if not payload.get("all_passed"):
        errors.append("all_passed is false")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
