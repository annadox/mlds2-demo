# MLflow lecture demo

This is the introductory two-run workflow. For the presentation, use the [enhanced demo and screenshot guide](MLFLOW_SCREENSHOT_GUIDE.md): it adds nested Optuna trials, native fold curves, parameter comparisons, dataset lineage and a recorded alias-change illustration. Install its `tuning` extra when using that workflow.

This extends Presenter 1's existing training function. The model, features, chronological folds and metrics come from the same code. The added commands manage experiment records and the fitted model's downstream identity.

The [recorded execution](DEMO_RESULTS_2026-11-12.md) contains the completed local run IDs, observed scores, registry versions and prediction. Use those references when capturing the current demo.

## 1. Setup and data

From the repository root, in macOS/Linux:

```bash
cd 02_reproducible_project
uv sync --locked --extra tracking
uv run --locked --extra tracking python -m electricity_demo.prepare
```

In PowerShell, use `Set-Location 02_reproducible_project` for the first command. All subsequent `uv run` commands are the same. Preparation downloads the public source once and keeps the original data outside Git. The current lock resolves MLflow 3.18.0.

The `tracking` extra is optional. Presenter 1 can continue using the original training command without it. Use `--extra tracking` consistently for this demo, because plain `uv run` can synchronize the environment without the optional packages.

## 2. The connection to Presenter 1

Presenter 1's command:

```bash
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml
```

Your tracked version:

```bash
uv run --locked --extra tracking python -m electricity_demo.tracking --config configs/ridge.yaml --run-name ridge-search
uv run --locked --extra tracking python -m electricity_demo.tracking --config configs/random_forest.yaml --run-name forest-depth-10
```

`tracking.run_tracked_training` opens an MLflow run and calls the original `train.run_training`. It reads that execution's output files, logs explicit validation metrics and effective fitted parameters, packages the pipeline and verifies a prediction round trip. One run contains the entire execution, including its CV folds, any parameter search and final development refit.

The local `outputs/<execution>/tracking.json` receipt contains the MLflow run ID, logged-model URI/ID, experiment ID and store URI. Keep it with the screenshots. Failed training or logging leaves the MLflow run marked FAILED.

## 3. Open the Tracking UI

Run from `02_reproducible_project` in a separate terminal:

```bash
uv run --locked --extra tracking mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db --host 127.0.0.1 --port 5050
```

Open http://127.0.0.1:5050 and select `electricity-mt158-hourly`. The trainer uses the absolute path to this same database, independent of its caller's working directory. Artifact paths are absolute local file URIs inside `.mlflow/artifacts`. This store is for this machine. A teammate should rerun the workflow locally rather than copy a database containing another machine's absolute paths.

Show these run-table columns:

- `model_type`, `model.alpha` or `model.max_depth`
- `cv_mae_kw`, `cv_rmse_kw`, `cv_baseline_mae_kw`
- `cv_std_fold_mae_kw`
- `comparison_group` and `snapshot` when checking comparability

Filter to FINISHED runs with identical `comparison_group`. Select the lowest development CV MAE after inspecting the baseline and fold metrics. Do not use the notebook's scores or final-test results to rank these candidates.

The Ridge YAML starts with alpha 1.0 and searches a grid. The logged `model.alpha` is the selected fitted alpha; the initial config and full candidate table remain in artifacts. `cv_rmse_kw` is pooled across validation predictions. The candidate-table column `mean_rmse_kw` is an average of fold RMSEs and need not have the same value.

## 4. Inspect a run and its model

Inside the selected run:

1. Check parameters and named `cv_*` metrics. No `test_*` metric should exist in a default execution.
2. Open `training/forecast.png`, `training/cv_results.csv`, `training/config.yaml` and `training/manifest.json`.
3. Inspect archived `source/` modules and `uv.lock` alongside the recorded content hashes. Source modifications are disclosed by the `source_dirty` tag.
4. Open the linked logged model `forecast_pipeline`. Inspect its signature, input example and pinned model environment.

The model includes fitted numerical scaling, calendar encoding and the estimator. Its input is a frame with 29 engineered numeric columns, represented as doubles in the MLflow signature. Raw observation history is handled by the shared feature helper. The model metadata records this feature specification and the last fitted target timestamp.

Training automatically reloads the logged model through `mlflow.pyfunc.load_model` and checks predictions against the local pipeline. A run receives `model_roundtrip_verified=true` only after this succeeds.

## 5. Register an explicitly selected run

Copy the actual run ID from its UI or local receipt. Replace `SELECTED_RUN_ID` below:

```bash
uv run --locked --extra tracking python -m electricity_demo.registry --run-id SELECTED_RUN_ID --name electricity-mt158-hourly --alias champion --reason "Lowest development CV MAE among the compatible Ridge and forest runs; baseline and fold results inspected."
```

Use that reason only when it accurately describes your selection. Registration requires a completed tracked run with a verified model round trip. It does not rerun training or score the test period.

To show another version, explicitly register another completed run under the same name and explain its role. Omit `--alias champion` for a comparison candidate. Each invocation creates a new version, including repeat invocations for the same run. Registering a model or changing an alias does not deploy or reload a service.

The registry uses numbered versions and the case-sensitive alias `champion`. The source run and logged model remain traceable. Avoid deprecated Production/Staging stages.

## 6. Presenter 3's minimal handoff

Use `history.csv` from the selected execution, replacing `SELECTED_OUTPUT_DIRECTORY`:

```bash
uv run --locked --extra tracking python -m electricity_demo.predict --history outputs/SELECTED_OUTPUT_DIRECTORY/history.csv --output outputs/next_hour_forecast.json
```

The command resolves `champion` to a numbered version once, loads that version, constructs the next feature row from observed history and returns:

- immutable numbered model URI and source run ID
- prediction origin and target timestamp
- next-hour predicted demand in kW

The saved UCI development history ends at `2014-11-01 00:00`. This example predicts the hour ending `2014-11-01 01:00`. It requires no future target value. The loader rejects insufficient or gapped history, and rejects a target within the model's fitting period.

This is a local historical demonstration of an inference interface. A sequence of later predictions must advance with newly observed history. This model does not make a whole future-day forecast from one origin.

## 7. Validation and screenshots

```bash
uv run --locked --offline --extra tracking python -m pytest -q
```

Integration checks cover the actual SQLite-backed MLflow path, correct CV logging, effective selected parameters, absence of default test metrics, model signature rejection of missing features, model loading, version/alias retrieval, history-only prediction and failed-run status. Feature tests compare inference construction with training-time features.

For the slides, capture the compatible run table, selected run details/plot, logged-model signature, registered versions/alias and prediction result. Use the same run/model/version identifiers across the screenshots. The slide-by-slide script is in `PRESENTER_2_SLIDE_PLAN.md`.

Do not tune against the final test. If a final test report is needed after freezing a configuration, the tracked command accepts `--final-test`, following the base training implementation's explicit opt-in behavior. Prior test results are already documented in this repository, so describe that history accurately.

## Official references

- [MLflow Tracking concepts](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow sklearn model logging and autologging](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
- [MLflow registry, versions and aliases](https://mlflow.org/docs/latest/ml/model-registry/workflow/)

The dependency lock and executable tests establish which API version this demonstration uses. The lecture's focus is the ML lifecycle, not tracking-server administration.
