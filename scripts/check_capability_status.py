from __future__ import annotations

import inspect
import sys

import cartoboost.deep as deep
import cartoboost.representation as representation
from cartoboost.capabilities import capability_table, validate_capability_table

EXCLUDED_EXPORTS = {
    "available_deep_backends",
    "backend_dispatch_report",
}


def exported_classes(module: object) -> set[str]:
    names = set(getattr(module, "__all__", [])) - EXCLUDED_EXPORTS
    result = set()
    for name in names:
        value = getattr(module, name, None)
        if inspect.isclass(value):
            result.add(name)
    return result


def main() -> int:
    rows = capability_table()
    errors = validate_capability_table()
    present = {str(row["class_name"]) for row in rows}
    required = exported_classes(deep) | exported_classes(representation)
    for class_name in sorted(required - present):
        errors.append(f"{class_name} has no capability status row")
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
