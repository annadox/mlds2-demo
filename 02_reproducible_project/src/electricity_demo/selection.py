"""Model selection as code: a reference run and an explicit, rerunnable selection rule.

The UI is for investigating runs; this module makes the decision reproducible and reviewable.
"""

from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from .config import load_training_config
from .evaluate import compare_predictions
from .store import DEFAULT_STORE, LECTURE_EXPERIMENT, configure_store
from .tracking import evaluation_metrics, experiment_id_for, log_datasets, log_fold_metrics
from .train import load_development, prediction_frame, run_identity


def log_baseline(config_path, *, experiment=LECTURE_EXPERIMENT, store_dir=DEFAULT_STORE, batch_id):
    config = load_training_config(config_path)
    data = load_development(config)
    development, folds = data.development, data.folds
    identity = run_identity(config, data.dataset_hash)["comparison_group"]
    experiment_id = experiment_id_for(experiment, store_dir)
    with mlflow.start_run(experiment_id=experiment_id, run_name="seasonal-naive-reference") as run:
        mlflow.set_tags({"run_kind": "baseline", "comparison_group": identity, "lecture_batch": batch_id})
        mlflow.log_params({"model_type": "seasonal_naive", "seasonal_lag_hours": 24, **config["evaluation"]})
        log_datasets(development, folds, data.dataset_path)
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
