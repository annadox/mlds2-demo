"""Chronological model selection, final refit and run artifacts.

The steps are separate functions so that other entry points reuse them instead of re-reading files:

    data = load_development(config)          # validated data, features, folds
    result = fit_and_evaluate(config_path)   # CV search + refit, in memory
    destination = save_run(result)           # artifacts, manifest, summary

``run_training`` combines the last two for the CLI, notebooks and tests. Nothing here imports MLflow;
``tracking.py`` wraps these functions.
"""

from __future__ import annotations

import argparse
import json
import platform
import pickle
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from .config import PROJECT_ROOT, git_state, load_training_config, project_path, sha256_file, write_json
from .evaluate import compare_predictions, plot_forecast
from .features import build_features
from .pipeline import build_pipeline
from .prepare import load_hourly
from .split import chronological_folds, split_development_test
from .report import MODEL_NAMES, comparison_group, console, update_comparison, write_run_summary


@dataclass
class DevelopmentData:
    """Validated observations, causal features and the chronological evaluation design."""
    raw: pd.DataFrame
    preparation: dict
    dataset_path: Path
    preparation_path: Path
    dataset_hash: str
    development: pd.DataFrame
    test: pd.DataFrame
    folds: list

    @property
    def feature_names(self) -> list[str]:
        return list(self.development.columns.drop("target"))


@dataclass
class TrainingResult:
    """Everything a training execution produced, before anything is written or logged."""
    config_path: Path
    config: dict
    data: DevelopmentData
    identity: dict
    pipeline: Pipeline
    metrics: dict
    search_results: pd.DataFrame
    predictions: pd.DataFrame
    plot_predictions: pd.DataFrame
    final_test: bool
    started: float
    manifest: dict | None = field(default=None)

    @property
    def feature_spec(self) -> dict:
        config = self.config
        return {
            "features": config["features"], "column_order": self.data.feature_names,
            "required_history_hours": max(config["features"]["lags"] + config["features"]["rolling_windows"]),
            "horizon_hours": 1,
            "availability": "Predict y_t after observing y_(t-1); new actual readings arrive every hour.",
            "model_fit": "All development targets only; fitted model remains fixed throughout final test.",
        }


def prediction_frame(table: pd.DataFrame, predicted, split: str, fold: int | None = None) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": table.index,
        "prediction_origin": table.index - pd.Timedelta(hours=1),
        "split": split,
        "fold": fold,
        "actual_kw": table["target"].to_numpy(),
        "predicted_kw": predicted,
        "baseline_kw": table["lag_24h"].to_numpy(),
    })


def load_development(config: dict) -> DevelopmentData:
    """Load the recorded dataset version and build features, development/test split and folds."""
    dataset_path = project_path(config["dataset"])
    preparation_path = project_path(config["dataset_manifest"])
    dataset_hash = sha256_file(dataset_path)
    raw = load_hourly(dataset_path, preparation_path)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    table = build_features(raw, config["features"])
    development, test = split_development_test(table, config["evaluation"])
    folds = chronological_folds(development, config["evaluation"])
    return DevelopmentData(raw, preparation, dataset_path, preparation_path, dataset_hash, development, test, folds)


def run_identity(config: dict, dataset_hash: str) -> dict:
    """Code, environment and evaluation identity; runs are only comparable within one group."""
    source = {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path)
              for path in sorted((PROJECT_ROOT / "src" / "electricity_demo").glob("*.py"))}
    lock = sha256_file(PROJECT_ROOT / "uv.lock")
    return {"source_sha256": source, "uv_lock_sha256": lock,
            "comparison_group": comparison_group(config, dataset_hash, lock, source)}


def evaluate_candidate(base, development: pd.DataFrame, folds: list, params: dict, on_fold=None):
    """Evaluate one configuration; shared by the grid search here and the Optuna search in tune.py.

    Each fitted pipeline only sees that fold's training rows. The optional
    observer records completed folds without controlling their evaluation.
    """
    feature_names = list(development.columns.drop("target"))
    X, y = development[feature_names], development["target"]
    fold_metrics, fold_predictions = [], []
    for fold, (train_idx, validation_idx) in enumerate(folds, start=1):
        started = time.perf_counter()
        estimator = clone(base).set_params(**params)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        validation = development.iloc[validation_idx]
        result = prediction_frame(validation, estimator.predict(validation[feature_names]), "cv", fold)
        measured = {
            "fold": fold,
            "train_start": development.index[train_idx[0]].isoformat(),
            "train_end": development.index[train_idx[-1]].isoformat(),
            "validation_start": validation.index[0].isoformat(),
            "validation_end": validation.index[-1].isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            **compare_predictions(result),
        }
        fold_predictions.append(result)
        fold_metrics.append(measured)
        if on_fold is not None:
            on_fold(measured)
    return fold_metrics, fold_predictions


def candidate_record(params: dict, fold_metrics: list, elapsed: float) -> dict:
    """One row of the candidate table, in the same format for grid and Optuna searches."""
    scores = [fold["model"]["mae_kw"] for fold in fold_metrics]
    return {
        "parameters": json.dumps(params, sort_keys=True),
        "mean_mae_kw": float(np.mean(scores)), "std_mae_kw": float(np.std(scores)),
        "mean_rmse_kw": float(np.mean([fold["model"]["rmse_kw"] for fold in fold_metrics])),
        "elapsed_seconds": elapsed,
        **{f"fold_{fold['fold']}_mae_kw": fold["model"]["mae_kw"] for fold in fold_metrics},
        **{f"fold_{fold['fold']}_rmse_kw": fold["model"]["rmse_kw"] for fold in fold_metrics},
    }


def finalize(config_path: str | Path, config: dict, data: DevelopmentData, identity: dict, *, best_params: dict,
             best_folds: list, best_predictions: list, candidates: list[dict], final_test: bool,
             started: float) -> TrainingResult:
    """Refit the selected configuration on all development rows and assemble its metrics."""
    feature_names = data.feature_names
    X, y = data.development[feature_names], data.development["target"]
    console.print("[dim]Fitting selected configuration on all development targets...[/dim]")
    fitted = clone(build_pipeline(config)).set_params(**best_params).fit(X, y)
    cv_predictions = pd.concat(best_predictions, ignore_index=True)
    metrics = {
        "units": "kW",
        "cv": {
            **compare_predictions(cv_predictions), "folds": best_folds,
            "mean_fold_mae_kw": float(np.mean([fold["model"]["mae_kw"] for fold in best_folds])),
            "std_fold_mae_kw": float(np.std([fold["model"]["mae_kw"] for fold in best_folds])),
            "note": "Selection CV for the chosen parameters; not an unbiased evaluation of the entire search.",
        },
        "best_params": best_params, "final_test_evaluated": final_test,
    }
    predictions, plot_predictions = [cv_predictions], cv_predictions
    if final_test:
        test_predictions = prediction_frame(data.test, fitted.predict(data.test[feature_names]), "test")
        metrics["test"] = compare_predictions(test_predictions)
        predictions.append(test_predictions)
        plot_predictions = test_predictions
    if sha256_file(data.dataset_path) != data.dataset_hash:
        raise ValueError("Dataset changed while training was running; no results were saved.")
    search_results = pd.DataFrame(candidates)
    search_results["rank"] = search_results["mean_mae_kw"].rank(method="min").astype(int)
    return TrainingResult(Path(config_path), config, data, identity, fitted, metrics, search_results,
                          pd.concat(predictions, ignore_index=True), plot_predictions, final_test, started)


def fit_and_evaluate(config_path: str | Path, final_test: bool = False) -> TrainingResult:
    """Grid-search the configured candidates with chronological CV, then refit the best one."""
    started = time.perf_counter()
    config = load_training_config(config_path)
    data = load_development(config)
    identity = run_identity(config, data.dataset_hash)
    grid = config["search"]["parameters"] if config["search"]["enabled"] else {}
    candidates = list(ParameterGrid(grid))
    base = build_pipeline(config)
    preparation = data.preparation
    console.rule(f"[bold cyan]{MODEL_NAMES[config['model']['type']]}")
    console.print(f"Customer [bold]{preparation.get('preparation', {}).get('customer', 'unspecified')}[/bold] | {preparation.get('snapshot', 'dataset')} | {len(data.raw):,} hourly readings")
    console.print(f"Development: {len(data.development):,} targets | {len(data.folds)} chronological folds | {len(candidates)} configuration(s)")
    console.print(f"Final test: {'enabled | ' + format(len(data.test), ',') + ' targets' if final_test else 'reserved, not evaluated'}")

    best = (float("inf"), {}, [], [])
    records = []
    for number, params in enumerate(candidates, 1):
        candidate_started = time.perf_counter()
        console.print(f"[dim]Configuration {number}/{len(candidates)} | fitting {len(data.folds)} chronological folds...[/dim]")
        fold_metrics, fold_predictions = evaluate_candidate(base, data.development, data.folds, params)
        for measured in fold_metrics:
            console.print(f"  Fold {measured['fold']}: MAE [bold]{measured['model']['mae_kw']:.3f} kW[/bold] | RMSE {measured['model']['rmse_kw']:.3f} kW | {measured['elapsed_seconds']:.1f}s")
        record = candidate_record(params, fold_metrics, time.perf_counter() - candidate_started)
        records.append(record)
        if record["mean_mae_kw"] < best[0]:
            best = (record["mean_mae_kw"], params, fold_predictions, fold_metrics)
    _, best_params, best_predictions, best_folds = best
    return finalize(config_path, config, data, identity, best_params=best_params, best_folds=best_folds,
                    best_predictions=best_predictions, candidates=records, final_test=final_test, started=started)


def save_run(result: TrainingResult, output_root: Path | None = None) -> Path:
    """Write the run folder: config, metrics, predictions, model, feature spec, history and manifest."""
    config, data = result.config, result.data
    now = datetime.now(timezone.utc)
    snapshot = data.preparation.get("snapshot", "data")
    run_id = f"{now:%Y-%m-%d_%H-%M-%SZ}_{config['model']['type']}_{snapshot}_{uuid.uuid4().hex[:6]}"
    destination = (output_root if output_root is not None else project_path(config["output_dir"])) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result.search_results.sort_values("rank").to_csv(destination / "cv_results.csv", index=False, float_format="%.8f")
    result.predictions.to_csv(destination / "predictions.csv", index=False, float_format="%.12g")
    write_json(destination / "metrics.json", result.metrics)
    model_name = MODEL_NAMES[config["model"]["type"]]
    plot_forecast(result.plot_predictions, destination / "forecast.png",
                  f"{model_name} · {'Final test' if result.final_test else 'Development CV'}")
    with (destination / "model.pkl").open("wb") as handle:
        pickle.dump(result.pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
    spec = result.feature_spec
    write_json(destination / "feature_spec.json", spec)
    development, test = data.development, data.test
    data.raw.loc[data.raw["timestamp"] <= development.index[-1]].tail(spec["required_history_hours"]).to_csv(
        destination / "history.csv", index=False)
    package_versions = {name: version(name) for name in ("electricity-demo", "pandas", "numpy", "scikit-learn", "matplotlib", "pyyaml", "rich", "dvc")}
    pointer_paths = [data.dataset_path.with_suffix(data.dataset_path.suffix + ".dvc"),
                     data.preparation_path.with_suffix(data.preparation_path.suffix + ".dvc")]
    manifest = {
        "run_id": run_id, "created_utc": now.isoformat(), "elapsed_seconds": time.perf_counter() - result.started,
        "model_type": config["model"]["type"], "model_config": config["model"],
        "comparison_group": result.identity["comparison_group"],
        "git": git_state(), "source_sha256": result.identity["source_sha256"],
        "dataset": {"path": config["dataset"], "sha256": data.dataset_hash, "rows": len(data.raw), "preparation": data.preparation},
        "dataset_pointer_sha256": {path.name: sha256_file(path) for path in pointer_paths if path.exists()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": package_versions,
                        "uv_lock_sha256": result.identity["uv_lock_sha256"]},
        "config_sha256": sha256_file(destination / "config.yaml"), "source_config_sha256": sha256_file(project_path(result.config_path)),
        "seed": config["seed"],
        "evaluation": {**config["evaluation"], "test_evaluated": result.final_test,
                       "development_first_target": development.index[0].isoformat(), "development_last_target": development.index[-1].isoformat(),
                       "test_first_target": test.index[0].isoformat(), "test_last_target": test.index[-1].isoformat()},
    }
    write_json(destination / "manifest.json", manifest)
    result.manifest = manifest
    write_run_summary(destination, config, result.metrics, manifest, result.search_results)
    return destination


def run_training(config_path: str | Path, final_test: bool = False, output_root: Path | None = None) -> Path:
    """Train, evaluate and save one run; the entry point shared by the CLI, notebooks and tests."""
    return save_run(fit_and_evaluate(config_path, final_test=final_test), output_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/ridge.yaml")
    parser.add_argument("--final-test", action="store_true", help="Explicitly evaluate the reserved final test period. Use after model selection.")
    args = parser.parse_args()
    destination = run_training(args.config, final_test=args.final_test)
    update_comparison(destination.parent)  # Cross-run report: a CLI convenience, not part of training.


if __name__ == "__main__":
    main()
