"""Resolve a registry alias and forecast one hour from observed meter history.

Inference path: imports features, prepare and the MLflow store only, never training code.
"""

from __future__ import annotations

import argparse
import json

import mlflow
import pandas as pd

from .config import project_path, write_json
from .features import build_next_features
from .prepare import load_hourly
from .store import DEFAULT_EXPERIMENT, DEFAULT_STORE, configure_store


def predict_next(history_path, *, name=DEFAULT_EXPERIMENT, alias="champion", store_dir=DEFAULT_STORE) -> dict:
    client = configure_store(store_dir)
    selected = client.get_model_version_by_alias(name, alias)
    model_uri = f"models:/{name}/{selected.version}"
    model_info = mlflow.models.get_model_info(model_uri)
    history = load_hourly(project_path(history_path))
    features = build_next_features(history, model_info.metadata["feature_spec"])
    # Do not present a prediction inside the model's fitting period as a new forecast.
    if features.index[0] <= pd.Timestamp(model_info.metadata["development_last_target"]):
        raise ValueError("Forecast target must be after the model's training period.")
    model = mlflow.pyfunc.load_model(model_uri)
    predicted = float(model.predict(features)[0])
    return {
        "model_uri": model_uri, "requested_alias": alias, "source_run_id": selected.run_id,
        "prediction_origin": history["timestamp"].iloc[-1].isoformat(),
        "target_timestamp": features.index[0].isoformat(), "predicted_kw": predicted,
        "timestamp_policy": "source Portuguese wall clock",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE))
    parser.add_argument("--output")
    args = parser.parse_args()
    result = predict_next(args.history, name=args.name, alias=args.alias, store_dir=args.store_dir)
    if args.output:
        write_json(project_path(args.output), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
