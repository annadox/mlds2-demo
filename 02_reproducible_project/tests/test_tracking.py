import json
import pickle

import numpy as np
import pandas as pd
import pytest
import yaml

mlflow = pytest.importorskip("mlflow", reason="Install the tracking extra for MLflow integration tests.")

from electricity_demo.config import load_training_config
from electricity_demo.features import build_features
from electricity_demo.predict import predict_next
from electricity_demo.registry import register_run
from electricity_demo.tracking import configure_store, run_tracked_training


@pytest.mark.parametrize("model_type", ["ridge", "random_forest"])
def test_tracking_registry_and_history_only_prediction(training_config, hourly_frame, tmp_path, model_type):
    if model_type == "random_forest":
        config = load_training_config(training_config)
        config["model"] = {"type": "random_forest", "params": {"n_estimators": 5, "max_depth": 4}}
        config["search"] = {"enabled": False, "parameters": {}}
        training_config.write_text(yaml.safe_dump(config))
    store = tmp_path / "tracking"
    output = run_tracked_training(training_config, experiment="integration", store_dir=store)
    receipt = json.loads((output / "tracking.json").read_text())
    metrics = json.loads((output / "metrics.json").read_text())
    client = configure_store(store)
    run = client.get_run(receipt["run_id"])
    assert run.info.status == "FINISHED"
    assert run.data.metrics["cv_mae_kw"] == metrics["cv"]["model"]["mae_kw"]
    if model_type == "ridge":
        assert run.data.params["model.alpha"] == str(metrics["best_params"]["model__alpha"])
    else:
        assert run.data.params["model.max_depth"] == "4"
    assert not any(key.startswith("test_") for key in run.data.metrics)
    assert run.data.tags["model_roundtrip_verified"] == "true"
    assert {item.path for item in client.list_artifacts(run.info.run_id, "training")} >= {
        "training/manifest.json", "training/cv_results.csv", "training/forecast.png",
    }

    registered = register_run(run.info.run_id, name="integration-model", reason="CV comparison fixture",
                              alias="champion", store_dir=store)
    result = predict_next(output / "history.csv", name="integration-model", store_dir=store)
    assert result["model_uri"] == f"models:/integration-model/{registered.version}"
    assert result["source_run_id"] == run.info.run_id
    assert result["target_timestamp"] == hourly_frame["timestamp"].iloc[1000].isoformat()

    config = load_training_config(training_config)
    reference = build_features(hourly_frame, config["features"]).drop(columns="target").loc[
        [pd.Timestamp(result["target_timestamp"])]]
    with (output / "model.pkl").open("rb") as handle:
        original = pickle.load(handle)
    np.testing.assert_allclose([result["predicted_kw"]], original.predict(reference), rtol=1e-9, atol=1e-9)

    # The pyfunc signature rejects a structurally incomplete request.
    loaded = mlflow.pyfunc.load_model(receipt["model_uri"])
    with pytest.raises(mlflow.exceptions.MlflowException):
        loaded.predict(reference.drop(columns="lag_1h").astype(float))


def test_training_failure_is_recorded(training_config, tmp_path):
    store = tmp_path / "failed-tracking"
    config = load_training_config(training_config)
    from pathlib import Path
    Path(config["dataset"]).unlink()
    with pytest.raises(FileNotFoundError):
        run_tracked_training(training_config, experiment="failure", store_dir=store)
    client = configure_store(store)
    experiment = client.get_experiment_by_name("failure")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FAILED"
