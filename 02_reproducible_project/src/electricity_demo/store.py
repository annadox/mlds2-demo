"""Location of the local MLflow store, shared by tracking, registry and inference.

Kept separate from tracking.py so that inference code does not import training code.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
from mlflow import MlflowClient

from .config import PROJECT_ROOT, project_path

DEFAULT_EXPERIMENT = "electricity-mt158-hourly"
LECTURE_EXPERIMENT = "electricity-mt158-lecture"
DEFAULT_STORE = PROJECT_ROOT / ".mlflow"


def configure_store(store_dir: str | Path = DEFAULT_STORE) -> MlflowClient:
    """Anchor both metadata and artifacts to one explicit local directory."""
    root = project_path(store_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient()
