from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pytest

EXPECTED_PROOF_IMAGES = {
    "segmentation_diagonal_2d.png",
    "segmentation_gaussian_2d.png",
    "splitter_tests/phase_1_axis_threshold.png",
    "splitter_tests/phase_2_diagonal_2d.png",
    "splitter_tests/phase_3_gaussian_2d.png",
    "splitter_tests/phase_4_periodic_wraparound.png",
    "splitter_tests/phase_5_fuzzy_boundary.png",
    "splitter_tests/phase_6_linear_leaf.png",
    "splitter_tests/phase_7_sparse_set.png",
    "splitter_tests/phase_8_learning_rate_shrinkage.png",
    "splitter_tests/phase_9_fuzzy_periodic_wraparound.png",
}


def _assert_nonblank_segmentation(path: Path) -> None:
    assert path.exists()
    image = mpimg.imread(path)
    assert image.ndim in {2, 3}
    assert image.shape[0] >= 400
    assert image.shape[1] >= 400

    rgb = image[..., :3] if image.ndim == 3 else image
    assert float(np.std(rgb)) > 0.01

    flattened = rgb.reshape(-1, rgb.shape[-1]) if rgb.ndim == 3 else rgb.reshape(-1, 1)
    # Inspect all pixels: a fixed stride can miss thin boundaries and labels.
    unique_colors = np.unique(np.round(flattened, 3), axis=0)
    assert len(unique_colors) >= 16


def test_segmentation_proof_images_are_generated_and_nonblank(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "generate_segmentation_proofs.py"

    subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path)], check=True, cwd=repo_root
    )

    asset_dir = tmp_path
    actual = {
        path.relative_to(asset_dir).as_posix()
        for path in asset_dir.glob("**/*.png")
        if path.relative_to(asset_dir).as_posix() in EXPECTED_PROOF_IMAGES
    }
    assert actual == EXPECTED_PROOF_IMAGES
    for relative_path in sorted(EXPECTED_PROOF_IMAGES):
        _assert_nonblank_segmentation(asset_dir / relative_path)


def test_segmentation_models_use_the_declared_native_splitters(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2]))
    from scripts.generate_segmentation_proofs import train_diagonal, train_radial

    diagonal = train_diagonal()
    radial = train_radial()
    assert diagonal.splitters == ["diagonal_2d"]
    assert radial.splitters == ["gaussian_2d"]
    assert diagonal.predict([[-2.0, -2.0], [2.0, 2.0]]) == pytest.approx([-10.0, 10.0])
    assert radial.predict([[0.0, 0.0], [3.0, 3.0]]) == pytest.approx([10.0, -10.0])
