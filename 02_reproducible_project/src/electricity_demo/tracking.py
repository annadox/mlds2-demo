"""Track the existing training execution in a local MLflow store."""

from __future__ import annotations

import argparse
import json
import pickle
from importlib.metadata import version
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature

from .config import PROJECT_ROOT, project_path, sha256_file, write_json
from .features import build_features
from .prepare import load_hourly
from .split import chronological_folds, split_development_test
from .train import run_training

DEFAULT_EXPERIMENT = "electricity-mt158-hourly"
DEFAULT_STORE = PROJECT_ROOT / ".mlflow"


def experiment_id_for(name: str, store_dir=DEFAULT_STORE) -> str:
    client = configure_store(store_dir)
    existing = client.get_experiment_by_name(name)
    return existing.experiment_id if existing else client.create_experiment(
        name, artifact_location=(project_path(store_dir).resolve() / "artifacts").as_uri(),
    )


def log_fold_metrics(fold: dict) -> None:
    """The UI's Step axis is the chronological fold number, not an epoch."""
    mlflow.log_metrics(evaluation_metrics(fold, "fold"), step=fold["fold"])
    if "elapsed_seconds" in fold:
        mlflow.log_metric("fold_elapsed_seconds", fold["elapsed_seconds"], step=fold["fold"])


def log_datasets(development, folds, source_path):
    validation_rows = np.concatenate([validation for _, validation in folds])
    datasets = []
    for context, table in [("development", development), ("validation", development.iloc[validation_rows])]:
        data = mlflow.data.from_pandas(
            table.astype("float64").reset_index(),
            source=Path(source_path).resolve().as_uri(),
            targets="target", name=f"electricity-{context}",
        )
        mlflow.log_input(data, context=context)
        datasets.append(data)
    return datasets


def configure_store(store_dir: str | Path = DEFAULT_STORE) -> MlflowClient:
    """Anchor both metadata and artifacts to one explicit local directory."""
    root = project_path(store_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient()


def evaluation_metrics(result: dict, prefix: str) -> dict:
    values = {
        f"{prefix}_mae_kw": result["model"]["mae_kw"],
        f"{prefix}_rmse_kw": result["model"]["rmse_kw"],
        f"{prefix}_baseline_mae_kw": result["seasonal_baseline"]["mae_kw"],
        f"{prefix}_baseline_rmse_kw": result["seasonal_baseline"]["rmse_kw"],
    }
    if result["mae_improvement_percent"] is not None:
        values[f"{prefix}_mae_improvement_percent"] = result["mae_improvement_percent"]
    return values


def run_tracked_training(
    config_path: str | Path,
    *,
    experiment: str = DEFAULT_EXPERIMENT,
    run_name: str | None = None,
    store_dir: str | Path = DEFAULT_STORE,
    final_test: bool = False,
) -> Path:
    experiment_id = experiment_id_for(experiment, store_dir)
    # One execution includes the existing CV/search and development refit.
    # Exceptions leave an explicitly FAILED MLflow run.
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        destination = run_training(config_path, final_test=final_test)
        mlflow.set_tag("run_kind", "training")
        log_training_output(destination)
    return destination


def log_training_output(destination: Path) -> None:
    """Package an existing training output in the current MLflow run."""
    if mlflow.active_run() is None:
        raise ValueError("An active MLflow run is required.")
    config = yaml.safe_load((destination / "config.yaml").read_text())
    manifest = json.loads((destination / "manifest.json").read_text())
    metrics = json.loads((destination / "metrics.json").read_text())
    final_test = metrics["final_test_evaluated"]
    feature_spec = json.loads((destination / "feature_spec.json").read_text())
    with (destination / "model.pkl").open("rb") as handle:
        fitted = pickle.load(handle)  # Produced by run_training in this process.

    data_path = project_path(config["dataset"])
    if sha256_file(data_path) != manifest["dataset"]["sha256"]:
        raise ValueError("Dataset changed after training; cannot construct a matching input example.")
    raw = load_hourly(data_path, project_path(config["dataset_manifest"]))
    development, _ = split_development_test(build_features(raw, config["features"]), config["evaluation"])
    folds = chronological_folds(development, config["evaluation"])
    _, validation_dataset = log_datasets(development, folds, data_path)
    # All model inputs are numeric. A double schema avoids fragile integer
    # width differences between pandas, CSV and JSON consumers.
    example = development[feature_spec["column_order"]].tail(5).astype("float64")
    expected = fitted.predict(example)
    preparation = manifest["dataset"]["preparation"]
    params = {
        "model_type": config["model"]["type"],
        "seed": config["seed"],
        "customer": preparation.get("preparation", {}).get("customer", "unspecified"),
        "required_history_hours": feature_spec["required_history_hours"],
        "lags": json.dumps(config["features"]["lags"]),
        "rolling_windows": json.dumps(config["features"]["rolling_windows"]),
        "search_enabled": config["search"]["enabled"],
        "search_parameters": json.dumps(config["search"]["parameters"], sort_keys=True),
        **config["evaluation"],
        **{f"model.{key}": value for key, value in fitted.named_steps["model"].get_params().items()},
    }
    mlflow.log_params(params)
    mlflow.set_tags({
        "local_run_id": destination.name,
        "comparison_group": manifest["comparison_group"],
        "dataset_sha256": manifest["dataset"]["sha256"],
        "snapshot": preparation.get("snapshot", "unspecified"),
        "source_commit": manifest["git"]["commit"] or "unknown",
        "source_dirty": str(manifest["git"]["dirty"]).lower(),
        "uv_lock_sha256": manifest["environment"]["uv_lock_sha256"],
        "evaluation_protocol": "one-step advancing observed history; selection CV",
        "final_test_evaluated": str(final_test).lower(),
        "timestamp_policy": "source Portuguese wall clock",
    })
    mlflow.log_metrics({
        **evaluation_metrics(metrics["cv"], "cv"),
        "cv_mean_fold_mae_kw": metrics["cv"]["mean_fold_mae_kw"],
        "cv_std_fold_mae_kw": metrics["cv"]["std_fold_mae_kw"],
        "cv_worst_fold_mae_kw": max(fold["model"]["mae_kw"] for fold in metrics["cv"]["folds"]),
        "cv_elapsed_seconds": sum(fold.get("elapsed_seconds", 0) for fold in metrics["cv"]["folds"]),
        "training_elapsed_seconds": manifest["elapsed_seconds"],
    })
    for fold in metrics["cv"]["folds"]:
        mlflow.log_metrics(evaluation_metrics(fold, f"fold_{fold['fold']}"))
        log_fold_metrics(fold)
    if final_test:
        mlflow.log_metrics(evaluation_metrics(metrics["test"], "test"))

    # Preserve the exact code/lock contents as well as their identities.
    source_files = {PROJECT_ROOT / name: digest for name, digest in manifest["source_sha256"].items()}
    source_files[PROJECT_ROOT / "uv.lock"] = manifest["environment"]["uv_lock_sha256"]
    for path, digest in source_files.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Source/environment changed during training: {path.name}")
        mlflow.log_artifact(str(path), artifact_path="source")
    mlflow.log_artifacts(str(destination), artifact_path="training")

    requirements = [f"{name}=={version(name)}" for name in (
        "mlflow", "scikit-learn", "numpy", "pandas", "scipy", "skops",
    )]
    model_info = mlflow.sklearn.log_model(
        sk_model=fitted,
        name="forecast_pipeline",
        serialization_format="skops",
        # This tree object was fitted by our own training function above.
        # Trust only the known type required by that forest, not arbitrary
        # types discovered in an externally supplied model file.
        skops_trusted_types=["sklearn.tree._tree.Tree"] if config["model"]["type"] == "random_forest" else [],
        input_example=example,
        signature=infer_signature(example, expected),
        pip_requirements=requirements,
        metadata={"feature_spec": feature_spec, "development_last_target": manifest["evaluation"]["development_last_target"],
                  "selection_metrics": "CV uses fold-specific refits; the saved model is refitted on all development data."},
    )
    reloaded = mlflow.pyfunc.load_model(model_info.model_uri)
    np.testing.assert_allclose(reloaded.predict(example), expected, rtol=1e-9, atol=1e-9)
    try:
        reloaded.predict(example.drop(columns=example.columns[0]))
    except mlflow.exceptions.MlflowException:
        schema_rejects_missing_feature = True
    else:
        raise AssertionError("Model signature accepted an incomplete feature frame.")
    checks = {
        "prediction_roundtrip_passed": True,
        "missing_feature_rejected": schema_rejects_missing_feature,
        "checked_example_rows": len(example),
        "rtol": 1e-9, "atol": 1e-9,
        "scope": "Serialization and input-contract smoke checks, not production approval.",
    }
    write_json(destination / "model_checks.json", checks)
    mlflow.log_artifact(str(destination / "model_checks.json"))
    # Explicitly name these selection estimates: they are not errors obtained
    # by predicting the CV rows with the final development-fitted model.
    mlflow.log_metrics({"selection_cv_mae_kw": metrics["cv"]["model"]["mae_kw"],
                        "selection_cv_rmse_kw": metrics["cv"]["model"]["rmse_kw"]},
                       model_id=model_info.model_id, dataset=validation_dataset)
    mlflow.set_tags({"model_uri": model_info.model_uri, "model_roundtrip_verified": "true"})
    receipt = {
        "run_id": mlflow.active_run().info.run_id, "experiment_id": mlflow.active_run().info.experiment_id,
        "model_uri": model_info.model_uri, "model_id": model_info.model_id,
        "tracking_uri": mlflow.get_tracking_uri(), "mlflow_version": mlflow.__version__,
        "comparison_group": manifest["comparison_group"], "model_roundtrip_verified": True,
    }
    write_json(destination / "tracking.json", receipt)
    mlflow.log_artifact(str(destination / "tracking.json"))
    print(f"MLflow run: {mlflow.active_run().info.run_id}\nLogged model: {model_info.model_uri}\nReceipt: {destination / 'tracking.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ridge.yaml")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE))
    parser.add_argument("--final-test", action="store_true")
    args = parser.parse_args()
    run_tracked_training(args.config, experiment=args.experiment, run_name=args.run_name,
                         store_dir=args.store_dir, final_test=args.final_test)


if __name__ == "__main__":
    main()
