"""Configuration, project-relative paths, and lightweight provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

# Data, configs, outputs and the MLflow store live in the project checkout, not inside the
# installed package. ELECTRICITY_DEMO_ROOT points an installed copy at a checkout; the default
# is the source tree this module was imported from (the editable install `uv sync` creates).
PROJECT_ROOT = Path(os.environ.get("ELECTRICITY_DEMO_ROOT") or Path(__file__).resolve().parents[2]).resolve()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_yaml(path: str | Path) -> dict:
    with project_path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return value


def positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def load_training_config(path: str | Path) -> dict:
    config = read_yaml(path)
    for field in ("seed", "dataset", "dataset_manifest", "features", "evaluation", "model", "search", "output_dir"):
        if field not in config:
            raise ValueError(f"Missing configuration field: {field}")
    positive_int(config["seed"], "seed", minimum=0)
    for field in ("lags", "rolling_windows"):
        values = config["features"][field]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"features.{field} must be a nonempty list of distinct integers.")
        for value in values:
            positive_int(value, f"features.{field}")
    if 24 not in config["features"]["lags"]:
        raise ValueError("Include lag 24 so the seasonal baseline uses identical target rows.")
    evaluation = config["evaluation"]
    if evaluation["horizon_hours"] != 1 or evaluation["observation_delay_hours"] != 0:
        raise ValueError("This project supports hourly one-step forecasts with immediately available observations only.")
    positive_int(evaluation["cv_splits"], "cv_splits", minimum=2)
    positive_int(evaluation["cv_validation_hours"], "cv_validation_hours")
    positive_int(evaluation["gap_hours"], "gap_hours", minimum=0)
    model_type = config["model"]["type"]
    if model_type not in {"ridge", "random_forest"}:
        raise ValueError("model.type must be ridge or random_forest.")
    if not isinstance(config["search"]["enabled"], bool):
        raise ValueError("search.enabled must be a boolean.")
    parameters = config["search"].get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("search.parameters must be a mapping.")
    if config["search"]["enabled"] and not parameters:
        raise ValueError("Enabled search needs a nonempty parameter grid.")
    for name, values in parameters.items():
        if not name.startswith("model__") or not isinstance(values, list) or not values:
            raise ValueError("Search entries must be nonempty model__parameter lists.")
    return config


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def git_state() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(PROJECT_ROOT), *args], text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=normal"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
