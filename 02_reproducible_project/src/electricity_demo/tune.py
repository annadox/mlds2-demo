"""Bounded Optuna searches with nested MLflow trials and shared temporal CV."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml

from .config import PROJECT_ROOT, load_training_config, positive_int, project_path, read_yaml, sha256_file, write_json
from .evaluate import compare_predictions
from .features import build_features
from .pipeline import build_pipeline
from .prepare import load_hourly
from .report import comparison_group
from .split import chronological_folds, split_development_test
from .tracking import DEFAULT_STORE, evaluation_metrics, experiment_id_for, log_datasets, log_fold_metrics, log_training_output
from .train import evaluate_candidate, run_training

LECTURE_EXPERIMENT = "electricity-mt158-lecture"


def load_development(config):
    path = project_path(config["dataset"])
    raw = load_hourly(path, project_path(config["dataset_manifest"]))
    development, _ = split_development_test(build_features(raw, config["features"]), config["evaluation"])
    folds = chronological_folds(development, config["evaluation"])
    source = {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in sorted((PROJECT_ROOT / "src/electricity_demo").glob("*.py"))}
    identity = comparison_group(config, sha256_file(path), sha256_file(PROJECT_ROOT / "uv.lock"), source)
    return development, folds, identity


def suggest_parameters(trial, definitions):
    params = {}
    for name, spec in definitions.items():
        if spec["kind"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif spec["kind"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["kind"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        else:
            raise ValueError(f"Unsupported search parameter kind: {spec['kind']}")
    return params


def run_search(config_path, *, search_path="configs/tuning.yaml", experiment=LECTURE_EXPERIMENT,
               store_dir=DEFAULT_STORE, n_trials=None, batch_id=None) -> Path:
    config = load_training_config(config_path)
    model_type = config["model"]["type"]
    search = copy.deepcopy(read_yaml(search_path)[model_type])
    budget = positive_int(n_trials if n_trials is not None else search["n_trials"], "n_trials")
    if search["sampler"] == "grid":
        if any(spec["kind"] != "categorical" for spec in search["parameters"].values()):
            raise ValueError("Grid searches require categorical parameter choices.")
        sampler = optuna.samplers.GridSampler({k: v["choices"] for k, v in search["parameters"].items()}, seed=config["seed"])
    elif search["sampler"] == "tpe":
        sampler = optuna.samplers.TPESampler(seed=config["seed"], n_startup_trials=search["n_startup_trials"])
    else:
        raise ValueError("sampler must be grid or tpe.")
    experiment_id = experiment_id_for(experiment, store_dir)
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_type}-{search['sampler']}-study") as parent:
        development, folds, identity = load_development(config)
        dataset_hash = sha256_file(project_path(config["dataset"]))
        base = build_pipeline(config)
        work = project_path(store_dir).resolve() / "studies" / parent.info.run_id
        work.mkdir(parents=True)
        mlflow.set_tags({"run_kind": "study", "model_family": model_type, "comparison_group": identity,
                         "lecture_batch": batch_id or parent.info.run_id, "pruning": "disabled: complete identical temporal windows"})
        mlflow.log_params({"study.sampler": search["sampler"], "study.budget": budget,
                           "study.seed": config["seed"], "study.jobs": 1, "study.optuna_version": optuna.__version__})
        mlflow.log_dict(search, "search_space.json")
        mlflow.log_dict(config, "original_training_config.json")
        study = optuna.create_study(
            study_name=f"{model_type}-{parent.info.run_id}", direction="minimize", sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            storage=f"sqlite:///{(project_path(store_dir).resolve() / 'optuna.db').as_posix()}",
        )
        study.set_user_attr("mlflow_parent_run_id", parent.info.run_id)
        study.set_user_attr("comparison_group", identity)
        for params in search.get("enqueue", []):
            study.enqueue_trial(params)
        records = []

        def objective(trial):
            chosen = suggest_parameters(trial, search["parameters"])
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_type}-trial-{trial.number:02d}", nested=True) as child:
                trial.set_user_attr("mlflow_run_id", child.info.run_id)
                mlflow.set_tags({"run_kind": "trial", "model_family": model_type, "comparison_group": identity,
                                 "lecture_batch": batch_id or parent.info.run_id})
                effective = {**base.named_steps["model"].get_params(), **chosen}
                mlflow.log_params({"trial_number": trial.number, "model_type": model_type,
                                  **{f"model.{k}": v for k, v in effective.items()},
                                  "horizon_hours": 1, "seed": config["seed"]})
                log_datasets(development, folds, project_path(config["dataset"]))
                elapsed = time.perf_counter()
                fold_scores = []

                def observe(fold):
                    log_fold_metrics(fold)
                    fold_scores.append(fold["model"]["mae_kw"])
                    trial.report(float(np.mean(fold_scores)), step=fold["fold"])

                fold_results, predictions = evaluate_candidate(
                    base, development, folds, {f"model__{k}": v for k, v in chosen.items()}, on_fold=observe,
                )
                measured = compare_predictions(pd.concat(predictions, ignore_index=True))
                scores = {**evaluation_metrics(measured, "cv"),
                          "cv_std_fold_mae_kw": float(np.std(fold_scores)),
                          "cv_worst_fold_mae_kw": max(fold_scores),
                          "cv_elapsed_seconds": time.perf_counter() - elapsed}
                mlflow.log_metrics(scores)
                mlflow.log_dict({"folds": fold_results, "metric_step": "chronological fold number"}, "folds.json")
                records.append({"trial": trial.number, "run_id": child.info.run_id,
                                **{f"model.{k}": v for k, v in chosen.items()}, **scores,
                                **{f"fold_{f['fold']}_mae_kw": f["model"]["mae_kw"] for f in fold_results}})
            value = float(np.mean(fold_scores))
            # Parent history uses trial number; child history uses fold number.
            mlflow.log_metrics({"trial_objective_mae_kw": value,
                                "best_so_far_mae_kw": min(r["cv_mae_kw"] for r in records)}, step=trial.number)
            return value

        study.optimize(objective, n_trials=budget, n_jobs=1)
        if sha256_file(project_path(config["dataset"])) != dataset_hash:
            raise ValueError("Dataset changed during tuning.")
        winner = study.best_trial
        selected = copy.deepcopy(config)
        selected["model"]["params"].update(winner.params)
        selected["search"] = {"enabled": False, "parameters": {}}
        selected_path = work / "selected.yaml"
        selected_path.write_text(yaml.safe_dump(selected, sort_keys=False))
        # Reuse Presenter 1's complete training/artifact path for the selected
        # configuration, and verify that its repeated CV matches the trial.
        destination = run_training(selected_path)
        final_metrics = json.loads((destination / "metrics.json").read_text())
        repeated_score = final_metrics["cv"]["mean_fold_mae_kw"]
        # Independent LSQR fits on this macOS numerical backend vary by a few
        # micro-kW even with identical inputs. This is a score-agreement check,
        # not bitwise reproducibility; retain both measurements in the record.
        agreement_rtol, agreement_atol = 1e-5, 1e-6
        np.testing.assert_allclose(repeated_score, winner.value,
                                   rtol=agreement_rtol, atol=agreement_atol)
        final_manifest = json.loads((destination / "manifest.json").read_text())
        if final_manifest["comparison_group"] != identity:
            raise ValueError("Code, environment or evaluation identity changed during tuning.")
        log_training_output(destination)
        mlflow.set_tags({"selected_trial_run_id": winner.user_attrs["mlflow_run_id"],
                         "selected_trial_number": str(winner.number), "optuna_study_name": study.study_name})
        mlflow.log_params({"study.completed_trials": len(study.trials)})
        trial_table = pd.DataFrame(records)
        trial_table.to_csv(destination / "trials.csv", index=False)
        study.trials_dataframe().to_csv(destination / "optuna_trials.csv", index=False)
        selection = {"study_name": study.study_name, "parent_run_id": parent.info.run_id,
                     "selected_trial": winner.number, "selected_trial_run_id": winner.user_attrs["mlflow_run_id"],
                     "selected_params": winner.params, "objective": "mean chronological-fold MAE in kW",
                     "value": winner.value, "n_trials": len(study.trials), "comparison_group": identity,
                     "repeat_cv_check": {"trial_mean_fold_mae_kw": winner.value,
                                         "repeated_mean_fold_mae_kw": repeated_score,
                                         "absolute_difference_kw": abs(repeated_score - winner.value),
                                         "rtol": agreement_rtol, "atol": agreement_atol, "passed": True},
                     "final_test_evaluated": False,
                     "qualification": "Selection CV, not an unbiased assessment of the whole search; budget is small."}
        write_json(destination / "selection.json", selection)
        from .lecture_plots import plot_search
        plot_search(trial_table, metrics=final_metrics, destination=destination, model_type=model_type)
        for name in ("trials.csv", "optuna_trials.csv", "selection.json", "optimization.png", "hyperparameters.png"):
            mlflow.log_artifact(str(destination / name), artifact_path="search")
        mlflow.set_tag("local_output_dir", str(destination.relative_to(PROJECT_ROOT)) if destination.is_relative_to(PROJECT_ROOT) else str(destination))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/random_forest.yaml")
    parser.add_argument("--search-config", default="configs/tuning.yaml")
    parser.add_argument("--experiment", default=LECTURE_EXPERIMENT)
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE))
    parser.add_argument("--trials", type=int)
    args = parser.parse_args()
    run_search(args.config, search_path=args.search_config, experiment=args.experiment,
               store_dir=args.store_dir, n_trials=args.trials)


if __name__ == "__main__":
    main()
