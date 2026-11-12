"""Generate comparable searches, figures and a registry handoff for the lecture."""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from .config import PROJECT_ROOT, load_training_config, project_path, write_json
from .evaluate import compare_predictions
from .lecture_plots import plot_diagnostics, plot_model_comparison, plot_tradeoff
from .predict import predict_next
from .registry import assign_alias, register_run
from .tracking import DEFAULT_STORE, configure_store, evaluation_metrics, experiment_id_for, log_datasets, log_fold_metrics
from .train import prediction_frame
from .tune import LECTURE_EXPERIMENT, load_development, run_search


def log_baseline(config_path, *, experiment=LECTURE_EXPERIMENT, store_dir=DEFAULT_STORE, batch_id):
    config = load_training_config(config_path)
    development, folds, identity = load_development(config)
    experiment_id = experiment_id_for(experiment, store_dir)
    with mlflow.start_run(experiment_id=experiment_id, run_name="seasonal-naive-reference") as run:
        mlflow.set_tags({"run_kind": "baseline", "comparison_group": identity, "lecture_batch": batch_id})
        mlflow.log_params({"model_type": "seasonal_naive", "seasonal_lag_hours": 24, **config["evaluation"]})
        log_datasets(development, folds, project_path(config["dataset"]))
        frames, scores = [], []
        for number, (_, validation) in enumerate(folds, 1):
            table = development.iloc[validation]
            predictions = prediction_frame(table, table["lag_24h"].to_numpy(), "cv", number)
            measured = compare_predictions(predictions)
            fold = {"fold": number, "validation_start": table.index[0].isoformat(),
                    "validation_end": table.index[-1].isoformat(), **measured}
            log_fold_metrics(fold)
            frames.append(predictions)
            scores.append(fold)
        measured = compare_predictions(pd.concat(frames, ignore_index=True))
        mlflow.log_metrics({**evaluation_metrics(measured, "cv"),
                            "cv_std_fold_mae_kw": float(np.std([f["model"]["mae_kw"] for f in scores]))})
        mlflow.log_dict({"folds": scores, "definition": "Same hour yesterday on identical forecast targets"}, "folds.json")
        return {"run_id": run.info.run_id, "label": "Same hour yesterday", "cv_mae_kw": measured["model"]["mae_kw"],
                "cv_rmse_kw": measured["model"]["rmse_kw"], "fold_mae_kw": [f["model"]["mae_kw"] for f in scores],
                "comparison_group": identity}


def select_candidate(run_ids, baseline_run_id, *, store_dir=DEFAULT_STORE):
    """Select only compatible completed model runs, using development CV."""
    if not run_ids or len(run_ids) != len(set(run_ids)):
        raise ValueError("Provide distinct candidate run IDs.")
    client = configure_store(store_dir)
    baseline = client.get_run(baseline_run_id)
    if baseline.info.status != "FINISHED" or baseline.data.tags.get("run_kind") != "baseline":
        raise ValueError("A completed baseline run is required.")
    if not baseline.data.tags.get("comparison_group") or not np.isfinite(baseline.data.metrics.get("cv_mae_kw", float("nan"))):
        raise ValueError("Baseline must have an evaluation identity and finite CV MAE.")
    candidates = [client.get_run(run_id) for run_id in run_ids]
    for run in candidates:
        if run.info.status != "FINISHED" or run.data.tags.get("model_roundtrip_verified") != "true":
            raise ValueError("Candidates must be completed, model-verified runs.")
        if run.data.tags.get("comparison_group") != baseline.data.tags.get("comparison_group"):
            raise ValueError("Cannot select across incompatible data, code or evaluation protocols.")
        if not np.isfinite(run.data.metrics.get("cv_mae_kw", float("nan"))):
            raise ValueError("Candidate is missing a finite development CV MAE.")
    winner = min(candidates, key=lambda run: run.data.metrics["cv_mae_kw"])
    if winner.data.metrics["cv_mae_kw"] >= baseline.data.metrics["cv_mae_kw"]:
        raise ValueError("No candidate improves on the seasonal baseline; retain the reference and investigate.")
    return {
        "selected_run_id": winner.info.run_id, "baseline_run_id": baseline_run_id,
        "comparison_group": winner.data.tags["comparison_group"],
        "rule": "Lowest development CV MAE among compatible completed model runs, with an improvement over the baseline.",
        "selected_cv_mae_kw": winner.data.metrics["cv_mae_kw"],
        "baseline_cv_mae_kw": baseline.data.metrics["cv_mae_kw"],
        "candidates": [{"run_id": run.info.run_id, "cv_mae_kw": run.data.metrics["cv_mae_kw"],
                        "cv_rmse_kw": run.data.metrics["cv_rmse_kw"],
                        "cv_worst_fold_mae_kw": run.data.metrics["cv_worst_fold_mae_kw"]} for run in candidates],
        "scope": "A teaching selection rule, not business approval or a statistically unbiased final evaluation.",
    }


def run_demo(*, store_dir=DEFAULT_STORE, experiment=LECTURE_EXPERIMENT, forest_trials=None,
             demonstrate_rollback=False):
    batch = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    evidence = PROJECT_ROOT / "outputs" / f"lecture-{batch}"
    evidence.mkdir(parents=True)
    baseline = log_baseline("configs/ridge.yaml", experiment=experiment, store_dir=store_dir, batch_id=batch)
    records, tables = [], []
    for model_type, label in [("ridge", "Ridge (grid)"), ("random_forest", "Random forest (Optuna)")]:
        output = run_search(f"configs/{model_type}.yaml", experiment=experiment, store_dir=store_dir,
                            n_trials=forest_trials if model_type == "random_forest" else None, batch_id=batch)
        receipt = json.loads((output / "tracking.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        records.append({"label": label, "output_dir": str(output.relative_to(PROJECT_ROOT)), **receipt,
                        "cv_mae_kw": metrics["cv"]["model"]["mae_kw"], "cv_rmse_kw": metrics["cv"]["model"]["rmse_kw"],
                        "fold_mae_kw": [fold["model"]["mae_kw"] for fold in metrics["cv"]["folds"]]})
        table = pd.read_csv(output / "trials.csv")
        table["model_family"] = model_type
        tables.append(table)
        plot_diagnostics(pd.read_csv(output / "predictions.csv"), output)
        client = configure_store(store_dir)
        for name in ("diagnostics.png", "error_by_hour.csv"):
            client.log_artifact(receipt["run_id"], str(output / name), artifact_path="diagnostics")
    decision = select_candidate([r["run_id"] for r in records], baseline["run_id"], store_dir=store_dir)
    winner = next(r for r in records if r["run_id"] == decision["selected_run_id"])
    plot_model_comparison(records + [baseline], evidence)
    all_trials = pd.concat(tables, ignore_index=True)
    all_trials.to_csv(evidence / "all_trials.csv", index=False)
    plot_tradeoff(all_trials, evidence)
    write_json(evidence / "selection_decision.json", decision)
    client = configure_store(store_dir)
    for path in evidence.iterdir():
        client.log_artifact(winner["run_id"], str(path), artifact_path="lecture")
    for record in records:
        version = register_run(record["run_id"], name=experiment, store_dir=store_dir,
                               reason="Selected model-family configuration from chronological development CV; see selection_decision.json.")
        record["registered_version"] = str(version.version)
    events = [assign_alias(experiment, winner["registered_version"], reason=decision["rule"], store_dir=store_dir)]
    history = PROJECT_ROOT / winner["output_dir"] / "history.csv"
    prediction = predict_next(history, name=experiment, store_dir=store_dir)
    rollback = None
    if demonstrate_rollback:
        other = next(r for r in records if r is not winner)
        try:
            events.append(assign_alias(experiment, other["registered_version"],
                                       reason="Illustrative local rollback to a comparison version; no production service exists.", store_dir=store_dir))
            rollback = predict_next(history, name=experiment, store_dir=store_dir)
        finally:
            events.append(assign_alias(experiment, winner["registered_version"],
                                       reason="Restore the validation-selected champion after the local rollback illustration.", store_dir=store_dir))
        restored = predict_next(history, name=experiment, store_dir=store_dir)
        assert restored["model_uri"] == prediction["model_uri"]
        np.testing.assert_allclose(restored["predicted_kw"], prediction["predicted_kw"], rtol=1e-9, atol=1e-9)
    result = {"batch_id": batch, "experiment": experiment, "evidence_dir": str(evidence.relative_to(PROJECT_ROOT)),
              "baseline": baseline, "models": records, "selection": decision,
              "prediction": prediction, "rollback_prediction": rollback, "alias_events": events}
    write_json(evidence / "evidence.json", result)
    write_json(PROJECT_ROOT / "outputs/lecture_latest.json", result)
    client.log_artifact(winner["run_id"], str(evidence / "evidence.json"), artifact_path="lecture")
    print(f"Lecture evidence: {evidence}\nSelected run: {winner['run_id']}\nModel: {prediction['model_uri']}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE))
    parser.add_argument("--experiment", default=LECTURE_EXPERIMENT)
    parser.add_argument("--forest-trials", type=int)
    parser.add_argument("--demonstrate-rollback", action="store_true")
    args = parser.parse_args()
    run_demo(store_dir=args.store_dir, experiment=args.experiment, forest_trials=args.forest_trials,
             demonstrate_rollback=args.demonstrate_rollback)


if __name__ == "__main__":
    main()
