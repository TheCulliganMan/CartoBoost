#!/usr/bin/env python3
"""Build current docs plus selected historical docs without committing snapshots."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs-versions.json"
WORKTREES = ROOT / "target" / "docs-worktrees"


def run(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def git_ref_exists(ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def patch_route_base(config: Path, version: str) -> None:
    text = config.read_text(encoding="utf-8")
    text = re.sub(
        r"routeBasePath:\s*'docs'",
        f"routeBasePath: 'docs/{version}'",
        text,
        count=1,
    )
    # Redirect targets in a historical config are authored for that config's
    # unversioned docs route. Once the route base is prefixed, those targets
    # must carry the same version prefix or Docusaurus rejects the build.
    text = re.sub(
        rf"(to:\s*'/docs/)(?!{re.escape(version)}/)",
        rf"\g<1>{version}/",
        text,
    )
    config.write_text(text, encoding="utf-8")


def patch_historical_links(worktree: Path, version: str) -> None:
    """Prefix absolute docs links in a detached historical site."""

    roots = [worktree / "docs", worktree / "src", worktree / "docusaurus.config.ts"]
    suffixes = {".md", ".mdx", ".ts", ".tsx", ".json"}
    for root in roots:
        paths = [root] if root.is_file() else root.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = path.read_text(encoding="utf-8")
            text = re.sub(
                rf"(?<![\w/])(/docs)(?!/{re.escape(version)}(?:/|[\s\"')?#]|$))",
                rf"/docs/{version}",
                text,
            )
            # These links intentionally leave the docs tree for the current
            # browser lab; preserve that destination in the historical build.
            text = re.sub(r"(?:\.\./)+modeling-lab", "/modeling-lab", text)
            path.write_text(text, encoding="utf-8")


def copy_path(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def copy_versioned_build(source: Path, target: Path, version: str) -> None:
    versioned_docs = source / "docs" / version
    if not versioned_docs.exists():
        raise SystemExit(f"Expected versioned docs output at {versioned_docs}.")
    copy_path(versioned_docs, target / "docs" / version)

    assets = source / "assets"
    if assets.exists():
        copy_path(assets, target / "assets")


@contextmanager
def without_historical_nav():
    """Build the current site before copying historical docs.

    Docusaurus validates navbar links while producing the current build. The
    historical pages do not exist until their detached worktrees are built,
    so temporarily suppress the version dropdown for this one build. The
    manifest is restored even when npm fails, leaving the source tree intact.
    """

    original = MANIFEST.read_bytes()
    try:
        MANIFEST.write_text("[]\n", encoding="utf-8")
        yield
    finally:
        MANIFEST.write_bytes(original)


def build_current() -> None:
    with without_historical_nav():
        run(["npm", "run", "build"], ROOT)


def build_version(version: str, ref: str) -> None:
    if not git_ref_exists(ref):
        raise SystemExit(
            f"Docs version {version} points to missing git ref {ref}. "
            "Create that release tag or update docs-versions.json."
        )

    worktree = WORKTREES / version
    if worktree.exists():
        shutil.rmtree(worktree)
    worktree.parent.mkdir(parents=True, exist_ok=True)

    run(["git", "worktree", "add", "--detach", str(worktree), ref], ROOT)
    try:
        patch_route_base(worktree / "docusaurus.config.ts", version)
        patch_historical_links(worktree, version)
        run(["npm", "ci"], worktree)
        run(["npm", "run", "build"], worktree)
        copy_versioned_build(worktree / "build", ROOT / "build", version)
    finally:
        run(["git", "worktree", "remove", "--force", str(worktree)], ROOT)


def main() -> None:
    versions = json.loads(MANIFEST.read_text(encoding="utf-8"))
    build_current()
    for entry in versions:
        build_version(entry["version"], entry["ref"])


if __name__ == "__main__":
    main()
