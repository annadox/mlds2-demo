import json

import pickle
import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from electricity_demo.config import load_training_config
from electricity_demo.features import build_features
from electricity_demo.snapshots import corrected_snapshot
from electricity_demo.train import run_training


@pytest.mark.parametrize("model_type", ["ridge", "random_forest"])
def test_training_repeats_and_saved_pipeline_reproduces_predictions(training_config, hourly_frame, model_type):
    if model_type == "random_forest":
        config = yaml.safe_load(training_config.read_text(encoding="utf-8"))
        config["model"] = {"type": model_type, "params": {"n_estimators": 5, "max_depth": 4}}
        config["search"] = {"enabled": False, "parameters": {}}
        training_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    first = run_training(training_config, final_test=True)
    second = run_training(training_config, final_test=True)
    assert first != second
    first_predictions = pd.read_csv(first / "predictions.csv")
    second_predictions = pd.read_csv(second / "predictions.csv")
    np.testing.assert_allclose(first_predictions["predicted_kw"], second_predictions["predicted_kw"], rtol=1e-10, atol=1e-10)
    with (first / "model.pkl").open("rb") as handle:
        saved = pickle.load(handle)
    config = load_training_config(training_config)
    table = build_features(hourly_frame, config["features"])
    heldout = table.loc[table.index >= pd.Timestamp(config["evaluation"]["test_start"])].drop(columns="target")
    actual_saved = saved.predict(heldout)
    np.testing.assert_allclose(actual_saved, first_predictions.loc[first_predictions["split"].eq("test"), "predicted_kw"], rtol=1e-9, atol=1e-9)
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["environment"]["uv_lock_sha256"]
    assert manifest["dataset"]["sha256"]


def test_default_training_does_not_score_final_test(training_config):
    output = run_training(training_config)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["final_test_evaluated"] is False
    assert "test" not in metrics
    assert set(pd.read_csv(output / "predictions.csv")["split"]) == {"cv"}


def test_unsupported_model_type_is_rejected(training_config):
    config = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    config["model"]["type"] = "unsupported_model"
    training_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="ridge or random_forest"):
        load_training_config(training_config)


def test_simulated_correction_changes_only_training_period():
    dates = pd.date_range("2014-01-01T01:00:00", periods=8760, freq="1h")
    original = pd.DataFrame({"timestamp": dates, "load_kw": np.full(len(dates), 100.0)})
    altered, metadata = corrected_snapshot(original, {"snapshot": "v1", "simulated_corrections": []})
    final_test = original["timestamp"] >= pd.Timestamp("2014-11-01T01:00:00")
    assert_frame_equal(original.loc[final_test], altered.loc[final_test])
    assert (original["load_kw"] != altered["load_kw"]).sum() == 336
    assert metadata["snapshot"] == "v2"
    assert "not an upstream" in metadata["simulated_corrections"][0]["description"]
