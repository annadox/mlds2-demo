"""One entry point for chronological model search, evaluation, and artifacts."""

from __future__ import annotations

import argparse
import json
import platform
import pickle
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid

from .config import PROJECT_ROOT, git_state, load_training_config, project_path, sha256_file, write_json
from .evaluate import compare_predictions, plot_forecast
from .features import build_features
from .pipeline import build_pipeline
from .prepare import load_hourly
from .split import chronological_folds, split_development_test
from .report import MODEL_NAMES, comparison_group, console, update_comparison, write_run_summary


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


def evaluate_candidate(base, development: pd.DataFrame, folds: list, params: dict, on_fold=None):
    """Evaluate one configuration; shared by the original grid and Optuna.

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


def run_training(config_path: str | Path, final_test: bool = False, output_root: Path | None = None) -> Path:
    started = time.perf_counter()
    config = load_training_config(config_path)
    dataset_path = project_path(config["dataset"])
    preparation_path = project_path(config["dataset_manifest"])
    dataset_hash = sha256_file(dataset_path)
    raw = load_hourly(dataset_path, preparation_path)
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    table = build_features(raw, config["features"])
    development, test = split_development_test(table, config["evaluation"])
    folds = chronological_folds(development, config["evaluation"])
    feature_names = list(development.columns.drop("target"))
    X, y = development[feature_names], development["target"]
    grid = config["search"]["parameters"] if config["search"]["enabled"] else {}
    candidates = list(ParameterGrid(grid))
    base = build_pipeline(config)
    model_name = MODEL_NAMES[config["model"]["type"]]
    source_hashes = {str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256_file(path) for path in sorted((PROJECT_ROOT / "src" / "electricity_demo").glob("*.py"))}
    lock_hash = sha256_file(PROJECT_ROOT / "uv.lock")
    console.rule(f"[bold cyan]{model_name}")
    console.print(f"Customer [bold]{preparation.get('preparation', {}).get('customer', 'unspecified')}[/bold] | {preparation.get('snapshot', 'dataset')} | {len(raw):,} hourly readings")
    console.print(f"Development: {len(development):,} targets | {len(folds)} chronological folds | {len(candidates)} configuration(s)")
    console.print(f"Final test: {'enabled | ' + format(len(test), ',') + ' targets' if final_test else 'reserved, not evaluated'}")

    best_score = float("inf")
    best_params = {}
    best_predictions = []
    best_folds = []
    search_records = []
    for candidate_number, params in enumerate(candidates, 1):
        candidate_started = time.perf_counter()
        console.print(f"[dim]Configuration {candidate_number}/{len(candidates)} | fitting {len(folds)} chronological folds...[/dim]")
        fold_metrics, fold_predictions = evaluate_candidate(base, development, folds, params)
        for measured in fold_metrics:
            console.print(f"  Fold {measured['fold']}: MAE [bold]{measured['model']['mae_kw']:.3f} kW[/bold] | RMSE {measured['model']['rmse_kw']:.3f} kW | {measured['elapsed_seconds']:.1f}s")
        scores = [fold["model"]["mae_kw"] for fold in fold_metrics]
        mean_score = float(np.mean(scores))
        search_records.append({
            "parameters": json.dumps(params, sort_keys=True),
            "mean_mae_kw": mean_score, "std_mae_kw": float(np.std(scores)),
            "mean_rmse_kw": float(np.mean([fold["model"]["rmse_kw"] for fold in fold_metrics])),
            "elapsed_seconds": time.perf_counter() - candidate_started,
            **{f"fold_{fold['fold']}_mae_kw": fold["model"]["mae_kw"] for fold in fold_metrics},
            **{f"fold_{fold['fold']}_rmse_kw": fold["model"]["rmse_kw"] for fold in fold_metrics},
        })
        if mean_score < best_score:
            best_score, best_params = mean_score, params
            best_predictions, best_folds = fold_predictions, fold_metrics

    console.print("[dim]Fitting selected configuration on all development targets...[/dim]")
    fitted = clone(base).set_params(**best_params).fit(X, y)
    cv_predictions = pd.concat(best_predictions, ignore_index=True)
    metrics = {
        "units": "kW",
        "cv": {
            **compare_predictions(cv_predictions), "folds": best_folds,
            "mean_fold_mae_kw": best_score,
            "std_fold_mae_kw": float(np.std([fold["model"]["mae_kw"] for fold in best_folds])),
            "note": "Selection CV for the chosen parameters; not an unbiased evaluation of the entire search.",
        },
        "best_params": best_params, "final_test_evaluated": final_test,
    }
    predictions = [cv_predictions]
    plot_predictions = cv_predictions
    if final_test:
        test_predictions = prediction_frame(test, fitted.predict(test[feature_names]), "test")
        metrics["test"] = compare_predictions(test_predictions)
        predictions.append(test_predictions)
        plot_predictions = test_predictions
    if sha256_file(dataset_path) != dataset_hash:
        raise ValueError("Dataset changed while training was running; no results were saved.")

    now = datetime.now(timezone.utc)
    snapshot = preparation.get("snapshot", "data")
    run_id = f"{now:%Y-%m-%d_%H-%M-%SZ}_{config['model']['type']}_{snapshot}_{uuid.uuid4().hex[:6]}"
    destination = (output_root if output_root is not None else project_path(config["output_dir"])) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    search_results = pd.DataFrame(search_records)
    search_results["rank"] = search_results["mean_mae_kw"].rank(method="min").astype(int)
    search_results.sort_values("rank").to_csv(destination / "cv_results.csv", index=False, float_format="%.8f")
    pd.concat(predictions, ignore_index=True).to_csv(destination / "predictions.csv", index=False, float_format="%.12g")
    write_json(destination / "metrics.json", metrics)
    plot_forecast(plot_predictions, destination / "forecast.png", f"{model_name} · {'Final test' if final_test else 'Development CV'}")
    with (destination / "model.pkl").open("wb") as handle:
        pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    history_hours = max(config["features"]["lags"] + config["features"]["rolling_windows"])
    write_json(destination / "feature_spec.json", {
        "features": config["features"], "column_order": feature_names,
        "required_history_hours": history_hours, "horizon_hours": 1,
        "availability": "Predict y_t after observing y_(t-1); new actual readings arrive every hour.",
        "model_fit": "All development targets only; fitted model remains fixed throughout final test.",
    })
    raw.loc[raw["timestamp"] <= development.index[-1]].tail(history_hours).to_csv(destination / "history.csv", index=False)
    package_versions = {name: version(name) for name in ("electricity-demo", "pandas", "numpy", "scikit-learn", "matplotlib", "pyyaml", "rich", "dvc")}
    pointer_paths = [dataset_path.with_suffix(dataset_path.suffix + ".dvc"), preparation_path.with_suffix(preparation_path.suffix + ".dvc")]
    manifest = {
        "run_id": run_id, "created_utc": now.isoformat(), "elapsed_seconds": time.perf_counter() - started,
        "model_type": config["model"]["type"], "model_config": config["model"],
        "comparison_group": comparison_group(config, dataset_hash, lock_hash, source_hashes),
        "git": git_state(), "source_sha256": source_hashes,
        "dataset": {"path": config["dataset"], "sha256": dataset_hash, "rows": len(raw), "preparation": preparation},
        "dataset_pointer_sha256": {path.name: sha256_file(path) for path in pointer_paths if path.exists()},
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "packages": package_versions, "uv_lock_sha256": lock_hash},
        "config_sha256": sha256_file(destination / "config.yaml"), "source_config_sha256": sha256_file(project_path(config_path)),
        "seed": config["seed"],
        "evaluation": {**config["evaluation"], "test_evaluated": final_test, "development_first_target": development.index[0].isoformat(), "development_last_target": development.index[-1].isoformat(), "test_first_target": test.index[0].isoformat(), "test_last_target": test.index[-1].isoformat()},
    }
    write_json(destination / "manifest.json", manifest)
    write_run_summary(destination, config, metrics, manifest, search_results)
    update_comparison(destination.parent)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ridge.yaml")
    parser.add_argument("--final-test", action="store_true", help="Explicitly evaluate the reserved final test period. Use after model selection.")
    args = parser.parse_args()
    run_training(args.config, final_test=args.final_test)


if __name__ == "__main__":
    main()
