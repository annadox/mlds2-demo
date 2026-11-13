"""Bounded Optuna searches with nested MLflow trials and shared temporal CV.

Each trial evaluates one candidate with the same ``evaluate_candidate`` used by train.py. The parent
run reuses the winning trial's fold results and only repeats the final development refit.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml

from .config import load_training_config, positive_int, project_path, read_yaml, sha256_file, write_json
from .evaluate import compare_predictions
from .pipeline import build_pipeline
from .plots import plot_search
from .store import LECTURE_EXPERIMENT
from .tracking import DEFAULT_STORE, evaluation_metrics, experiment_id_for, log_datasets, log_fold_metrics, log_training_result
from .train import candidate_record, evaluate_candidate, finalize, load_development, run_identity, save_run



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
    started = time.perf_counter()
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
        data = load_development(config)
        identity = run_identity(config, data.dataset_hash)
        group = identity["comparison_group"]
        base = build_pipeline(config)
        work = project_path(store_dir).resolve() / "studies" / parent.info.run_id
        work.mkdir(parents=True)
        mlflow.set_tags({"run_kind": "study", "model_family": model_type, "comparison_group": group,
                         "lecture_batch": batch_id or parent.info.run_id,
                         "pruning": "none: every trial completes the same temporal windows"})
        mlflow.log_params({"study.sampler": search["sampler"], "study.budget": budget,
                           "study.seed": config["seed"], "study.jobs": 1, "study.optuna_version": optuna.__version__})
        mlflow.log_dict(search, "search_space.json")
        mlflow.log_dict(config, "original_training_config.json")
        study = optuna.create_study(
            study_name=f"{model_type}-{parent.info.run_id}", direction="minimize", sampler=sampler,
            storage=f"sqlite:///{(project_path(store_dir).resolve() / 'optuna.db').as_posix()}",
        )
        study.set_user_attr("mlflow_parent_run_id", parent.info.run_id)
        study.set_user_attr("comparison_group", group)
        for params in search.get("enqueue", []):
            study.enqueue_trial(params)
        records, candidates, evaluated = [], [], {}

        def objective(trial):
            chosen = suggest_parameters(trial, search["parameters"])
            pipeline_params = {f"model__{k}": v for k, v in chosen.items()}
            with mlflow.start_run(experiment_id=experiment_id, run_name=f"{model_type}-trial-{trial.number:02d}", nested=True) as child:
                trial.set_user_attr("mlflow_run_id", child.info.run_id)
                mlflow.set_tags({"run_kind": "trial", "model_family": model_type, "comparison_group": group,
                                 "lecture_batch": batch_id or parent.info.run_id})
                effective = {**base.named_steps["model"].get_params(), **chosen}
                mlflow.log_params({"trial_number": trial.number, "model_type": model_type,
                                  **{f"model.{k}": v for k, v in effective.items()},
                                  "horizon_hours": 1, "seed": config["seed"]})
                log_datasets(data.development, data.folds, data.dataset_path)
                trial_started = time.perf_counter()
                fold_results, predictions = evaluate_candidate(base, data.development, data.folds, pipeline_params,
                                                               on_fold=log_fold_metrics)
                elapsed = time.perf_counter() - trial_started
                fold_scores = [fold["model"]["mae_kw"] for fold in fold_results]
                measured = compare_predictions(pd.concat(predictions, ignore_index=True))
                scores = {**evaluation_metrics(measured, "cv"),
                          "cv_std_fold_mae_kw": float(np.std(fold_scores)),
                          "cv_worst_fold_mae_kw": max(fold_scores),
                          "cv_elapsed_seconds": elapsed}
                mlflow.log_metrics(scores)
                mlflow.log_dict({"folds": fold_results, "metric_step": "chronological fold number"}, "folds.json")
                records.append({"trial": trial.number, "run_id": child.info.run_id,
                                **{f"model.{k}": v for k, v in chosen.items()}, **scores,
                                **{f"fold_{f['fold']}_mae_kw": f["model"]["mae_kw"] for f in fold_results}})
                candidates.append(candidate_record(pipeline_params, fold_results, elapsed))
                evaluated[trial.number] = (fold_results, predictions)
            value = float(np.mean(fold_scores))
            # Parent history uses trial number; child history uses fold number.
            mlflow.log_metrics({"trial_objective_mae_kw": value,
                                "best_so_far_mae_kw": min(r["cv_mae_kw"] for r in records)}, step=trial.number)
            return value

        study.optimize(objective, n_trials=budget, n_jobs=1)
        if sha256_file(data.dataset_path) != data.dataset_hash:
            raise ValueError("Dataset changed during tuning.")
        winner = study.best_trial
        selected = copy.deepcopy(config)
        selected["model"]["params"].update(winner.params)
        selected["search"] = {"enabled": False, "parameters": {}}
        selected_path = work / "selected.yaml"
        selected_path.write_text(yaml.safe_dump(selected, sort_keys=False))
        # Reuse the winning trial's CV results; only the final development refit runs again.
        best_folds, best_predictions = evaluated[winner.number]
        result = finalize(selected_path, selected, data, identity, best_params={}, best_folds=best_folds,
                          best_predictions=best_predictions, candidates=candidates, final_test=False, started=started)
        destination = save_run(result)
        log_training_result(result, destination)
        mlflow.set_tags({"selected_trial_run_id": winner.user_attrs["mlflow_run_id"],
                         "selected_trial_number": str(winner.number), "optuna_study_name": study.study_name})
        mlflow.log_params({"study.completed_trials": len(study.trials)})
        trial_table = pd.DataFrame(records)
        trial_table.to_csv(destination / "trials.csv", index=False)
        study.trials_dataframe().to_csv(destination / "optuna_trials.csv", index=False)
        selection = {"study_name": study.study_name, "parent_run_id": parent.info.run_id,
                     "selected_trial": winner.number, "selected_trial_run_id": winner.user_attrs["mlflow_run_id"],
                     "selected_params": winner.params, "objective": "mean chronological-fold MAE in kW",
                     "value": winner.value, "n_trials": len(study.trials), "comparison_group": group,
                     "cv_source": "Winning trial's fold results; only the final development refit is repeated.",
                     "final_test_evaluated": False,
                     "qualification": "Selection CV, not an unbiased assessment of the whole search; budget is small."}
        write_json(destination / "selection.json", selection)
        plot_search(trial_table, metrics=result.metrics, destination=destination, model_type=model_type)
        for name in ("trials.csv", "optuna_trials.csv", "selection.json", "optimization.png", "hyperparameters.png"):
            mlflow.log_artifact(str(destination / name), artifact_path="search")
        mlflow.set_tag("local_output_dir", str(destination))
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
