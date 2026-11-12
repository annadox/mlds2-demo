# Screenshot and demonstration guide

Use the enhanced experiment `electricity-mt158-lecture`. The earlier `electricity-mt158-hourly` experiment remains the introductory two-run example. All screenshots should come from one completed enhanced-demo batch.

## Generate the evidence once

From the repository root:

```bash
cd 02_reproducible_project
uv sync --locked --extra tuning
uv run --locked --extra tuning python -m electricity_demo.lecture_demo --demonstrate-rollback
```

This creates a baseline, three Ridge grid trials and ten forest Optuna trials. Each model family has one parent run with a refitted/logged model. It compares compatible parent runs, records the selection, registers both versions and assigns the selected champion. The optional rollback illustration temporarily loads the other local version, then restores and verifies the selected champion. No prediction service is deployed.

The source subset must already be prepared. For a fresh checkout run `uv run --locked --extra tuning python -m electricity_demo.prepare` first. Execution time depends on the machine. This is a preparation command, not a live lecture requirement.

The latest receipt is `outputs/lecture_latest.json`. It contains the exact batch tag, run IDs, model/version references, prediction, alias-change records and evidence folder. Save that receipt with the slide assets.

Start the UI from this project directory:

```bash
uv run --locked --extra tuning mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. If the server is already running, use it instead of launching a second copy. The optional packages must remain installed while it is running.

## Common filter

Replace `BATCH_ID` with `batch_id` from the latest receipt:

```text
tags.lecture_batch = 'BATCH_ID' and attributes.status = 'FINISHED'
```

Use `tags.run_kind = 'study'` for family winners, `tags.run_kind = 'trial'` for individual candidates, and `tags.run_kind = 'baseline'` for the seasonal reference. The UI's nesting display may hide children until their parent is expanded. A search/filter alone does not always expand the tree.

Do not mix old runs whose `comparison_group` differs. Do not include trial and parent rows in a ranking as if they were independent modeling procedures: the parent packages a selected trial's configuration.

## 1. Experiment hierarchy

Expand both study parents to show their trial children. Keep the baseline visible. Capture the run-name column and `model_type`, `model.alpha`, `model.max_depth`, `model.min_samples_leaf`, `cv_mae_kw`.

Explain: experiment = common task; parent = search execution; child = candidate configuration. Folds are metric steps within a trial. Each run has a role.

## 2. Model and baseline comparison

Select the two study parents and the baseline, using IDs from the receipt. Show `cv_mae_kw`, `cv_rmse_kw`, `cv_std_fold_mae_kw`, and model family. Use MLflow's comparison view and a bar chart if available in the installed UI. Sort ascending by `cv_mae_kw`.

The observed bar chart is also saved as `model_comparison.png` in the evidence folder and as a `lecture/` artifact on the selected parent. This is a generated data figure, not a screenshot of MLflow.

## 3. Fold metric curves

Compare the same three runs and open the metric history for `fold_mae_kw`. Set the x-axis to **Step** rather than wall time. Its values are folds 1, 2 and 3. Repeat with `fold_rmse_kw` only if it adds an insight.

Within a single trial, overlay `fold_mae_kw` and `fold_baseline_mae_kw` to compare with the seasonal reference. The fold boundaries are in `folds.json` for trials/baseline and in `training/metrics.json` for parent models.

These are chronological evaluation windows. Do not call the plot training loss, epochs or a learning curve. The static fallback is `fold_comparison.png` in the evidence folder.

## 4. Hyperparameter relationships

Filter to forest trials in the current batch:

```text
tags.lecture_batch = 'BATCH_ID' and tags.run_kind = 'trial' and tags.model_family = 'random_forest' and attributes.status = 'FINISHED'
```

Select the trials. Configure a scatter plot with `model.max_depth` on x and `cv_mae_kw` on y. If the installed UI offers a parallel-coordinates chart, use depth, minimum leaf size, feature fraction and CV MAE. Keep the legend and axes readable.

Show that the original fixed forest is included as trial 0. Avoid claiming that ten evaluations establish robust parameter importance. A search budget is part of the experiment.

The parent `search/hyperparameters.png` and `search/trials.csv` provide portable evidence if a native chart layout is awkward.

## 5. Search progress

Open the forest parent. Plot `trial_objective_mae_kw` and `best_so_far_mae_kw` with **Step** on x. Here step means trial number, starting at 0. Both curves are observed metrics, not a smoothed or invented progression.

The search uses a seeded TPE sampler with four startup trials, ten total trials and sequential execution. Each trial completes all CV folds. A flat best-so-far curve is a legitimate result. Do not tune the demonstration just to force an attractive improvement curve.

Static fallback: parent artifact `search/optimization.png`.

## 6. Dataset lineage

Open a parent or trial's dataset inputs. Show `electricity-development` and `electricity-validation`, their contexts, digest, schema and source. Relate the raw source to the full data SHA in the run manifest.

These are overlapping roles in a rolling evaluation protocol, not two disjoint fixed partitions. The development dataset contains all model-selection rows. Validation contains the targets scored by the three folds. Fold boundaries explain which observations each fitted fold model could access.

## 7. Model interface and checks

Open the parent's linked `forecast_pipeline` model. Show the input signature and example, then `model_checks.json`. It records a successful prediction round trip and rejection of an input missing a required feature.

The logged model has `selection_cv_*` metrics associated with the validation dataset. State that these are estimates from fold refits with the selected parameters; the final pipeline was refitted on development data.

A schema checks names/types, while the shared history helper checks temporal availability and adequate consecutive observations. One does not replace the other.

## 8. Selection, versions and alias

Open the selected parent's `lecture/selection_decision.json`. It records candidate IDs, the common comparison group, baseline, selection rule and scores. The program rejects incompatible candidates and a winner that fails to improve on the baseline.

Then open the registered model named after the enhanced experiment. Show both versions, `champion`, source runs and the `selected_trial_run_id` version tag. Follow the chain: version, parent run, winning trial, data/configuration.

The actual version numbers are in the receipt. Repeating the command creates new versions; never assume the current pair is numbered 1 and 2.

## 9. Optional rollback and inference

Use `prediction`, `rollback_prediction` and `alias_events` from `evidence.json`. Show the same historical input producing predictions from different numbered versions and the selected champion being restored. The restored version and prediction are checked automatically.

Explain that an alias change influences a later load. It does not replace an already loaded model inside a running application. This local reference change is an illustration for Presenter 3, not a production rollback mechanism.

## Useful appendix figures

- `diagnostics/diagnostics.png`: observed/predicted scatter and out-of-fold error by hour for each model family.
- `evaluation_cost.png`: candidate CV evaluation time versus validation error on this machine; not serving latency.
- `training/forecast.png`: one validation week and its residuals.

Use a maximum of six or seven screenshots across the main section. Keep additional figures as backup for questions. The full slide plan identifies where each view fits.
