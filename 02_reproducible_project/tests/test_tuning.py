import json

import numpy as np
import pytest
import yaml

optuna = pytest.importorskip("optuna", reason="Install the tuning extra for search integration tests.")
mlflow = pytest.importorskip("mlflow")

from electricity_demo.config import load_training_config
from electricity_demo.selection import log_baseline, select_candidate
from electricity_demo.predict import predict_next
from electricity_demo.registry import assign_alias, register_run
from electricity_demo.tracking import configure_store
from electricity_demo.tune import run_search


@pytest.mark.parametrize("family", ["ridge", "random_forest"])
def test_trials_fold_histories_selection_and_rollback(training_config, tmp_path, family):
    if family == "random_forest":
        config = load_training_config(training_config)
        config["model"] = {"type": family, "params": {"n_estimators": 4, "max_depth": 4}}
        config["search"] = {"enabled": False, "parameters": {}}
        training_config.write_text(yaml.safe_dump(config))
    search = {family: {"sampler": "grid", "n_trials": 2, "parameters": (
        {"alpha": {"kind": "categorical", "choices": [0.1, 1.0]}} if family == "ridge" else
        {"max_depth": {"kind": "categorical", "choices": [3, 6]}})}}
    if family == "random_forest":
        search[family].update(sampler="tpe", n_startup_trials=1,
                              parameters={"max_depth": {"kind": "int", "low": 3, "high": 6}})
    search_path = tmp_path / "search.yaml"
    search_path.write_text(yaml.safe_dump(search))
    store = tmp_path / "store"
    output = run_search(training_config, search_path=search_path, experiment="test-search", store_dir=store, batch_id="test")
    selection = json.loads((output / "selection.json").read_text())
    client = configure_store(store)
    parent = client.get_run(selection["parent_run_id"])
    children = client.search_runs([parent.info.experiment_id],
                                 filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'")
    assert len(children) == 2
    assert all(child.info.status == "FINISHED" for child in children)
    assert selection["selected_trial_run_id"] in {child.info.run_id for child in children}
    assert parent.data.tags["selected_trial_run_id"] == selection["selected_trial_run_id"]
    selected_child = client.get_run(selection["selected_trial_run_id"])
    # The packaged parent reuses the winning trial's CV results instead of re-running them.
    assert parent.data.metrics["cv_mae_kw"] == selected_child.data.metrics["cv_mae_kw"]
    for child in children:
        history = sorted(client.get_metric_history(child.info.run_id, "fold_mae_kw"), key=lambda metric: metric.step)
        assert [metric.step for metric in history] == [1, 2, 3]
        np.testing.assert_allclose(np.mean([m.value for m in history]), child.data.metrics["cv_mae_kw"])
        assert not any(key.startswith("test_") for key in child.data.metrics)
        assert len(child.inputs.dataset_inputs) == 2
        assert "model_uri" not in child.data.tags  # Only the selected refit is packaged.
    progress = sorted(client.get_metric_history(parent.info.run_id, "best_so_far_mae_kw"), key=lambda m: m.step)
    assert [m.step for m in progress] == [0, 1]
    assert progress[-1].value <= progress[0].value
    assert (output / "optimization.png").exists()
    assert (output / "hyperparameters.png").exists()
    study = optuna.load_study(study_name=selection["study_name"], storage=f"sqlite:///{store / 'optuna.db'}")
    assert len(study.trials) == 2

    # Demonstrate reference movement with real logged-model versions.
    first = register_run(parent.info.run_id, name="search-model", reason="Test selection", alias="champion", store_dir=store)
    second = register_run(parent.info.run_id, name="search-model", reason="Second version for alias test", store_dir=store)
    before = predict_next(output / "history.csv", name="search-model", store_dir=store)
    event = assign_alias("search-model", second.version, reason="Test move", store_dir=store)
    moved = predict_next(output / "history.csv", name="search-model", store_dir=store)
    assert str(event["previous_version"]) == str(first.version)
    assert before["model_uri"] != moved["model_uri"]
    assign_alias("search-model", first.version, reason="Test rollback", store_dir=store)
    restored = predict_next(output / "history.csv", name="search-model", store_dir=store)
    assert restored == before

    if family == "ridge":
        baseline = log_baseline(training_config, experiment="test-search", store_dir=store, batch_id="test")
        decision = select_candidate([parent.info.run_id], baseline["run_id"], store_dir=store)
        assert decision["selected_run_id"] == parent.info.run_id
        client.log_metric(parent.info.run_id, "cv_mae_kw", baseline["cv_mae_kw"] + 1)
        with pytest.raises(ValueError, match="No candidate improves"):
            select_candidate([parent.info.run_id], baseline["run_id"], store_dir=store)
        client.set_tag(parent.info.run_id, "comparison_group", "incompatible-data")
        with pytest.raises(ValueError, match="incompatible"):
            select_candidate([parent.info.run_id], baseline["run_id"], store_dir=store)


def test_invalid_trial_is_failed_in_both_systems(training_config, tmp_path):
    # An invalid estimator parameter triggers a real fit error.
    spec = {"ridge": {"sampler": "grid", "n_trials": 1,
                      "parameters": {"alpha": {"kind": "categorical", "choices": [-1.0]}}}}
    path = tmp_path / "bad-search.yaml"
    path.write_text(yaml.safe_dump(spec))
    store = tmp_path / "bad-store"
    with pytest.raises(ValueError):
        run_search(training_config, search_path=path, experiment="bad-search", store_dir=store)
    client = configure_store(store)
    experiment = client.get_experiment_by_name("bad-search")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 2
    assert {run.info.status for run in runs} == {"FAILED"}
    summaries = optuna.study.get_all_study_summaries(storage=f"sqlite:///{store / 'optuna.db'}")
    study = optuna.load_study(study_name=summaries[0].study_name, storage=f"sqlite:///{store / 'optuna.db'}")
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
