from __future__ import annotations

import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cargo", "run", "--quiet", "-p", "cartoboost-cli", "--", *args],
        cwd=_repo_root(),
        text=True,
        capture_output=True,
    )


def _write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_cli_rejects_unknown_splitter(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'splitter = "axis,spline"\n')

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "unknown splitter 'spline'" in result.stderr


def test_cli_rejects_unknown_leaf_predictor(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'leaf_predictor = "spline"\n')

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "unknown leaf_predictor 'spline'" in result.stderr


def test_cli_rejects_invalid_config_value(tmp_path: Path) -> None:
    config = _write_config(tmp_path, 'learning_rate = "fast"\n')

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "invalid config value for 'learning_rate'" in result.stderr


def test_cli_rejects_malformed_config_line(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "learning_rate 0.1\n")

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "invalid config line 1" in result.stderr


def test_cli_rejects_unknown_command() -> None:
    result = _run_cli("score")

    assert result.returncode != 0
    assert "unknown command 'score'" in result.stderr


def test_cli_rejects_unknown_option_for_command(tmp_path: Path) -> None:
    data = tmp_path / "train.csv"
    data.write_text("x,target\n0,1\n", encoding="utf-8")

    result = _run_cli("train", "--data", str(data), "--bogus", "1")

    assert result.returncode != 0
    assert "unknown option '--bogus' for command 'train'" in result.stderr


def test_cli_rejects_unknown_config_key(tmp_path: Path) -> None:
    config = _write_config(tmp_path, "unknown_setting = 1\n")

    result = _run_cli("inspect", "--config", str(config))

    assert result.returncode != 0
    assert "unknown config key 'unknown_setting'" in result.stderr


def test_cli_train_rejects_missing_target_column(tmp_path: Path) -> None:
    data = tmp_path / "train.csv"
    data.write_text("x,y\n0,1\n", encoding="utf-8")
    config = _write_config(tmp_path, 'target = "target"\n')

    result = _run_cli("train", "--data", str(data), "--config", str(config))

    assert result.returncode != 0
    assert "target column 'target' not found" in result.stderr


def test_cli_rejects_malformed_csv_row_width(tmp_path: Path) -> None:
    data = tmp_path / "train.csv"
    data.write_text("x,target\n0,1,extra\n", encoding="utf-8")

    result = _run_cli("train", "--data", str(data))

    assert result.returncode != 0
    assert "CSV row 2 has 3 columns but header has 2" in result.stderr


def test_cli_dense_predict_rejects_sparse_list_artifact(tmp_path: Path) -> None:
    model = tmp_path / "sparse-list-model.json"
    model.write_text(
        """
{
  "artifact_version": 1,
  "init_prediction": 0.0,
  "learning_rate": 1.0,
  "feature_count": 1,
  "trees": [
    {
      "root": {
        "Branch": {
          "split": {
            "SparseListContainsAny": {
              "sparse_feature": 0,
              "ids": [7],
              "missing_goes_left": false
            }
          },
          "left": {
            "Leaf": {
              "value": 10.0,
              "sample_weight_sum": 1.0,
              "training_loss": 0.0
            }
          },
          "right": {
            "Leaf": {
              "value": -1.0,
              "sample_weight_sum": 1.0,
              "training_loss": 0.0
            }
          },
          "gain": 0.0,
          "sample_weight_sum": 2.0
        }
      }
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )
    dense_input = tmp_path / "predict.csv"
    dense_input.write_text("x\n7\n", encoding="utf-8")

    result = _run_cli("predict", "--model", str(model), "--input", str(dense_input))

    assert result.returncode != 0
    assert "prediction requires sparse_sets" in result.stderr


def test_cli_rejects_missing_option_value_before_another_option():
    result = _run_cli("train", "--data", "--model-out", "model.json")
    assert result.returncode != 0
    assert "missing value for '--data'" in result.stderr


def test_cli_rejects_duplicate_options():
    result = _run_cli("inspect", "--data", "one.csv", "--data", "two.csv")
    assert result.returncode != 0
    assert "duplicate option '--data'" in result.stderr


def test_cli_rejects_unterminated_csv_quotes(tmp_path: Path):
    data = tmp_path / "train.csv"
    data.write_text('x,target\n"0,1\n', encoding="utf-8")
    result = _run_cli("inspect", "--data", str(data))
    assert result.returncode != 0
    assert "unterminated quoted CSV field" in result.stderr


def test_cli_json_output_escapes_control_characters(tmp_path: Path):
    import json

    config = _write_config(tmp_path, 'target = "taxi\tzone"\n')
    result = _run_cli("inspect", "--config", str(config))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["config"]["target"] == "taxi\tzone"
