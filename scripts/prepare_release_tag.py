#!/usr/bin/env python3
"""Prepare CartoBoost release versions and create committed release tags."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
TOML_VERSION_RE = re.compile(
    r'(?m)^(?P<prefix>[ \t]*version[ \t]*=[ \t]*)"(?P<version>\d+\.\d+\.\d+)"'
)
PY_VERSION_RE = re.compile(r'(?m)^(?P<prefix>__version__[ \t]*=[ \t]*)"(?P<version>\d+\.\d+\.\d+)"')


STATIC_VERSIONED_FILES = [
    Path("Cargo.toml"),
    Path("pyproject.toml"),
    Path("python/cartoboost/__init__.py"),
]


def run_git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def require_clean_worktree() -> None:
    status = run_git(["status", "--porcelain"]).stdout.strip()
    if status:
        raise SystemExit("Worktree must be clean before preparing a release tag.")


def normalize_version(raw: str) -> str:
    version = raw.removeprefix("v")
    if not VERSION_RE.fullmatch(version):
        raise SystemExit(f"Expected a SemVer version like 0.2.33, got {raw!r}.")
    return version


def parse_current_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = TOML_VERSION_RE.search(text)
    if not match:
        raise SystemExit("Could not read project version from pyproject.toml.")
    return match.group("version")


def bump_version(current: str, bump: str) -> str:
    major, minor, patch = (int(part) for part in current.split("."))
    if bump == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump == "minor":
        minor += 1
        patch = 0
    elif bump == "patch":
        patch += 1
    else:
        raise SystemExit(f"Unknown bump level: {bump}")
    return f"{major}.{minor}.{patch}"


def replace_one(path: Path, pattern: re.Pattern[str], version: str) -> None:
    full_path = ROOT / path
    text = full_path.read_text(encoding="utf-8")
    updated, count = pattern.subn(
        lambda match: f'{match.group("prefix")}"{version}"',
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit(f"Could not update version in {path}.")
    full_path.write_text(updated, encoding="utf-8")


def crate_manifests() -> list[Path]:
    crates_dir = ROOT / "crates"
    return sorted(path.relative_to(ROOT) for path in crates_dir.glob("*/Cargo.toml"))


def package_names_from_crates() -> list[str]:
    package_names = []
    for path in crate_manifests():
        text = (ROOT / path).read_text(encoding="utf-8")
        match = re.search(r'(?m)^name[ \t]*=[ \t]*"([^"]+)"', text)
        if not match:
            raise SystemExit(f"Could not read package name from {path}.")
        package_names.append(match.group(1))
    return sorted(package_names)


def update_versions(version: str) -> None:
    for path in [
        *STATIC_VERSIONED_FILES,
        *crate_manifests(),
        Path("crates/cartoboost-py/pyproject.toml"),
    ]:
        pattern = PY_VERSION_RE if path.name == "__init__.py" else TOML_VERSION_RE
        replace_one(path, pattern, version)

    for json_path in [ROOT / "package.json", ROOT / "package-lock.json"]:
        package_data = json.loads(json_path.read_text(encoding="utf-8"))
        package_data["version"] = version
        if "packages" in package_data and "" in package_data["packages"]:
            package_data["packages"][""]["version"] = version
        json_path.write_text(json.dumps(package_data, indent=2) + "\n", encoding="utf-8")

    update_cargo_lock(version)
    update_uv_lock(version)


def update_cargo_lock(version: str) -> None:
    cargo_lock = ROOT / "Cargo.lock"
    text = cargo_lock.read_text(encoding="utf-8")
    for package in package_names_from_crates():
        pattern = re.compile(
            rf'(?ms)(\[\[package\]\]\nname = "{re.escape(package)}"\nversion = )"\d+\.\d+\.\d+"'
        )
        text, count = pattern.subn(lambda match: f'{match.group(1)}"{version}"', text, count=1)
        if count != 1:
            raise SystemExit(f"Could not update Cargo.lock entry for {package}.")
    cargo_lock.write_text(text, encoding="utf-8")


def update_uv_lock(version: str) -> None:
    uv_lock = ROOT / "uv.lock"
    text = uv_lock.read_text(encoding="utf-8")
    pattern = re.compile(r'(?ms)(\[\[package\]\]\nname = "cartoboost"\nversion = )"\d+\.\d+\.\d+"')
    text, count = pattern.subn(lambda match: f'{match.group(1)}"{version}"', text, count=1)
    if count != 1:
        raise SystemExit("Could not update uv.lock entry for cartoboost.")
    uv_lock.write_text(text, encoding="utf-8")


def tag_exists(tag: str) -> bool:
    local = run_git(["rev-parse", "-q", "--verify", f"refs/tags/{tag}"], check=False)
    if local.returncode == 0:
        return True
    remote = run_git(
        ["ls-remote", "--exit-code", "--tags", "origin", f"refs/tags/{tag}"], check=False
    )
    return remote.returncode == 0


def create_tag(version: str) -> None:
    tag = f"v{version}"
    if tag_exists(tag):
        raise SystemExit(f"Tag {tag} already exists locally or on origin.")
    run_git(["tag", "-a", tag, "-m", f"CartoBoost {version}"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="Release version, with or without a leading v.")
    group.add_argument(
        "--bump", choices=["patch", "minor", "major"], help="Bump from current version."
    )
    group.add_argument(
        "--tag-current",
        action="store_true",
        help="Create a local annotated tag for the currently committed project version.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow version preparation when unrelated worktree changes already exist.",
    )
    args = parser.parse_args()

    if args.tag_current:
        require_clean_worktree()
        version = parse_current_version()
        create_tag(version)
        print(f"Created local tag v{version}.")
        print(f"Push it with: git push origin v{version}")
        return

    if not args.allow_dirty:
        require_clean_worktree()

    current = parse_current_version()
    version = normalize_version(args.version) if args.version else bump_version(current, args.bump)
    update_versions(version)

    print(f"Prepared CartoBoost {version} version files.")
    print("Review and commit the version changes, then create and push the tag:")
    print("  git push origin main")
    print("  python scripts/prepare_release_tag.py --tag-current")
    print(f"  git push origin v{version}")


if __name__ == "__main__":
    main()
