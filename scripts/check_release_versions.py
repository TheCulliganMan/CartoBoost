#!/usr/bin/env python3
"""Validate release tag and package version consistency."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
CHANGELOG = ROOT / "CHANGELOG.md"


def _toml_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^version\s*=\s*[\"']([^\"']+)[\"']", text)
    if match is None:
        raise ValueError(f"{path} does not declare a version")
    return match.group(1)


def _python_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'(?m)^__version__\s*=\s*["\']([^"\']+)["\']', text)
    if match is None:
        raise ValueError(f"{path} does not declare __version__")
    return match.group(1)


def _json_version(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["version"])


def _package_lock_versions(path: Path) -> dict[str, str]:
    """Return both npm lockfile version declarations.

    ``package-lock.json`` repeats the project version in the top-level
    metadata and in the root package entry.  Checking both catches a partial
    release bump that would otherwise leave npm metadata stale.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    versions: dict[str, str] = {}
    if "version" in payload:
        versions["package-lock.json:top-level"] = str(payload["version"])
    root_package = payload.get("packages", {}).get("")
    if isinstance(root_package, dict) and "version" in root_package:
        versions["package-lock.json:root-package"] = str(root_package["version"])
    if len(versions) != 2:
        raise ValueError(
            "package-lock.json must declare project version at top level and in packages['']"
        )
    return versions


def _cargo_lock_versions(path: Path) -> dict[str, str]:
    """Extract workspace crate versions from Cargo.lock."""

    text = path.read_text(encoding="utf-8")
    packages = {
        match.group("name"): match.group("version")
        for match in re.finditer(
            r'(?ms)^\[\[package\]\]\nname = "(?P<name>[^"]+)"\nversion = "(?P<version>[^"]+)"',
            text,
        )
    }
    expected_names: list[str] = []
    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        manifest_text = manifest.read_text(encoding="utf-8")
        match = re.search(r'(?m)^name\s*=\s*"([^"]+)"', manifest_text)
        if match is None:
            raise ValueError(f"{manifest} does not declare a package name")
        expected_names.append(match.group(1))
    missing = sorted(set(expected_names) - packages.keys())
    if missing:
        raise ValueError(f"Cargo.lock is missing workspace packages: {missing}")
    return {f"Cargo.lock:{name}": packages[name] for name in expected_names}


def _uv_lock_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'(?ms)^\[\[package\]\]\nname = "cartoboost"\nversion = "([^"]+)"', text)
    if match is None:
        raise ValueError("uv.lock is missing the cartoboost package entry")
    return match.group(1)


def declared_versions() -> dict[str, str]:
    versions = {
        "Cargo.toml": _toml_version(ROOT / "Cargo.toml"),
        "pyproject.toml": _toml_version(ROOT / "pyproject.toml"),
        "python/cartoboost/__init__.py": _python_version(ROOT / "python/cartoboost/__init__.py"),
        "package.json": _json_version(ROOT / "package.json"),
        "crates/cartoboost-py/pyproject.toml": _toml_version(
            ROOT / "crates/cartoboost-py/pyproject.toml"
        ),
    }
    for manifest in sorted((ROOT / "crates").glob("*/Cargo.toml")):
        versions[str(manifest.relative_to(ROOT))] = _toml_version(manifest)
    versions.update(_package_lock_versions(ROOT / "package-lock.json"))
    versions.update(_cargo_lock_versions(ROOT / "Cargo.lock"))
    versions["uv.lock:cartoboost"] = _uv_lock_version(ROOT / "uv.lock")
    return versions


def check(tag: str | None = None) -> dict[str, object]:
    versions = declared_versions()
    unique = sorted(set(versions.values()))
    if len(unique) != 1:
        raise SystemExit(f"package versions disagree: {versions}")
    version = unique[0]
    if not VERSION_RE.fullmatch(version):
        raise SystemExit(f"package version is not SemVer: {version!r}")
    changelog = CHANGELOG.read_text(encoding="utf-8")
    if not re.search(rf"(?m)^##\s+{re.escape(version)}(?:\s|$)", changelog):
        raise SystemExit(f"CHANGELOG.md is missing a release section for {version}")
    normalized_tag = None if tag is None else tag.removeprefix("v")
    if normalized_tag is not None and normalized_tag != version:
        raise SystemExit(f"tag {tag!r} does not match package version {version!r}")
    return {"version": version, "tag": tag, "files": len(versions), "passed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag such as v0.3.0")
    args = parser.parse_args()
    print(json.dumps(check(args.tag), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
