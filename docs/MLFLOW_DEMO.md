# MLflow demo: tracking, registry and model retrieval

How MLflow is added to the existing electricity forecasting project, how to regenerate the evidence used in Presenter 2's slides, and how to read what the MLflow UI shows. The recorded IDs and scores of the batch used in the slides are in [MLFLOW_ENHANCED_RESULTS.md](MLFLOW_ENHANCED_RESULTS.md).

MLflow wraps Presenter 1's training code; it does not replace it. Model, features, chronological folds and metrics come from the same `train` functions (`load_development`, `evaluate_candidate`, `fit_and_evaluate`, `save_run`); `train.py` itself never imports MLflow. MLflow records executions and gives the selected fitted pipeline a retrievable identity. No live training or service is needed on stage.

## Setup

From the repository root (macOS/Linux shown; in PowerShell `cd` works the same):

```bash
cd 02_reproducible_project
uv sync --locked --extra tuning
uv run --locked --extra tuning python -m electricity_demo.prepare   # only on a fresh checkout
```

Two optional extras exist: `tracking` (MLflow only) and `tuning` (MLflow + Optuna, needed for the lecture batch). Presenter 1's untracked commands work without either. Pass the same `--extra` on every `uv run`, otherwise uv re-syncs the environment without the optional packages. The lock resolves MLflow 3.18.0.

The store is local: SQLite at `.mlflow/mlflow.db`, artifacts under `.mlflow/artifacts`, both ignored by Git. Paths are absolute to this machine, so teammates should regenerate the workflow rather than copy the database.

## Lecture workflow (used for the slides)

```bash
uv run --locked --extra tuning python scripts/lecture_demo.py --demonstrate-rollback
```

In experiment `electricity-mt158-lecture` this creates, under one `lecture_batch` tag:

```text
seasonal-naive-reference            run_kind=baseline
ridge-grid-study                    run_kind=study  (refitted + logged model)
  ridge-trial-00 … ridge-trial-02   run_kind=trial  (alpha 0.1 / 1 / 10)
random_forest-tpe-study             run_kind=study  (refitted + logged model)
  random_forest-trial-00 … -09      run_kind=trial  (seeded TPE; trial 0 = original forest)
```

Search spaces are in `configs/tuning.yaml`. Trials record parameters, CV metrics, fold boundaries and dataset inputs. Only each family's selected configuration is refitted on the development period and packaged on its parent run, which links to its winning trial. The parent reuses the winning trial's fold results; only the final development refit runs again, so the parent's CV metrics equal the selected trial's. (The recorded 12 November batch predates this and re-ran CV once with an agreement check.)

The script then compares the compatible parent runs and the baseline with `selection.select_candidate` (the selection rule as code), writes `lecture/selection_decision.json` (rejecting unfinished, incompatible or baseline-inferior candidates), registers both families under `electricity-mt158-lecture`, assigns `champion` to the winner and, with `--demonstrate-rollback`, temporarily moves the alias to the other version and back, verifying the restored prediction. The receipt is `outputs/lecture_latest.json`; figures and receipts go to `outputs/lecture-<batch>/presentation_assets/`. A single family can be searched with `python -m electricity_demo.tune --config configs/random_forest.yaml`.

## Introductory two-run workflow

The smaller step-by-step path, in experiment `electricity-mt158-hourly`, uses the separate commands that the lecture workflow combines:

```bash
uv run --locked --extra tracking python -m electricity_demo.tracking --config configs/ridge.yaml --run-name ridge-search
uv run --locked --extra tracking python -m electricity_demo.tracking --config configs/random_forest.yaml --run-name forest-depth-10
uv run --locked --extra tracking python -m electricity_demo.registry --run-id SELECTED_RUN_ID --name electricity-mt158-hourly --alias champion --reason "Lowest development CV MAE among compatible runs; baseline and folds inspected."
uv run --locked --extra tracking python -m electricity_demo.predict --history outputs/SELECTED_OUTPUT_DIRECTORY/history.csv
```

`tracking.run_tracked_training` opens one MLflow run, calls `train.fit_and_evaluate` and `train.save_run`, then logs the in-memory result: validation metrics, effective fitted parameters (the selected alpha, not the YAML's starting value) and artifacts, packages the pipeline and verifies a reload. One run covers the whole execution, including its folds, any grid search and the final development refit. `outputs/<execution>/tracking.json` records the run ID, model URI and store. Failed training or logging leaves the run marked FAILED. Registration requires a finished run with a verified model round trip; each invocation creates a new version. Omit `--alias` for a comparison candidate.

## Opening the UI

```bash
uv run --locked --extra tuning mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db --host 127.0.0.1 --port 5050
```

Open http://127.0.0.1:5050 (on macOS port 5000 is taken by AirPlay/AirTunes). Reuse a running server rather than starting a second one.

Filter one batch and exclude preliminary attempts:

```text
tags.lecture_batch = 'BATCH_ID' and attributes.status = 'FINISHED'
```

Add `tags.run_kind = 'study'`, `'trial'` or `'baseline'` to select roles, and `tags.model_family = 'random_forest'` for forest trials. Expand parent runs to see children; a filter alone may not expand the tree. Never mix runs with different `comparison_group`, and do not rank parent and trial rows together as independent procedures.

## What each view shows

| View | What to show | How to read it |
| --- | --- | --- |
| Experiment / run table | Run name, `model_type`, `model.alpha` / `model.max_depth` / `model.min_samples_leaf`, `cv_mae_kw`, `cv_rmse_kw`, `cv_std_fold_mae_kw` | Experiment = common task; parent = search; child = candidate configuration |
| Compare runs | Two study parents plus baseline, sorted by `cv_mae_kw` | Metrics support a decision only when data, targets and protocol agree |
| `fold_mae_kw`, `fold_rmse_kw` (x = Step) | Steps 1–3 are chronological validation windows | Not epochs or a learning curve; fold std is descriptive, not a confidence interval |
| `trial_objective_mae_kw`, `best_so_far_mae_kw` on the forest parent | Step = trial number from 0 | Ten trials show search behaviour, not convergence; a flat curve is legitimate |
| Forest-trial scatter / parallel coordinates | Depth, leaf size, feature fraction vs CV MAE | Ten trials do not establish robust parameter importance |
| Dataset inputs | `electricity-development`, `electricity-validation`, digest, schema, source | Overlapping roles in rolling CV, not disjoint partitions; metadata identifies data but does not store it (DVC does) |
| Artifacts | `training/` config, manifest, metrics, `cv_results.csv`, forecast plot; archived `source/` and `uv.lock`; `source_dirty` tag | Provenance links a result to its inputs; an identity does not preserve the resource |
| Logged model `forecast_pipeline` | Signature, input example, environment, `model_checks.json` | Fitted scaling + calendar encoding + estimator; input is 29 engineered features, raw history goes through `features.build_next_features` |
| Registered model | Versions, `champion`, source runs, `selected_trial_run_id` tag | Registration is not approval or deployment; version numbers depend on registry history |

Details worth stating accurately:

- `cv_rmse_kw` is pooled over validation predictions; `mean_rmse_kw` in `cv_results.csv` averages fold RMSEs and can differ.
- Model-linked metrics are named `selection_cv_*`: they come from fold refits with the selected settings, while the logged pipeline is refitted on all development rows. They are optimistic selection evidence, not an unbiased estimate of the whole search.
- A signature checks column names and types. Temporal validity is enforced separately by the history helper, which rejects short or gapped history and targets inside the fitting period.
- Moving `champion` changes later alias resolution only; an already loaded model keeps its artifact. `predict` resolves the alias once and records the numbered version.
- Models use explicit `skops` serialization, trusting only `sklearn.tree._tree.Tree` beyond the defaults. MLflow cannot pin pip in its Conda spec because the uv environment has no pip; requirements and the lock are recorded. A fresh isolated serving environment has not been tested.
- No `test_*` metric appears by default. `--final-test` exists for a frozen configuration, but earlier test scores are already recorded in `02_reproducible_project/experiments/verification.md`, so describe that history honestly.
- Static fallbacks (generated from real results, not UI screenshots): `model_comparison.png`, `fold_comparison.png`, `search/optimization.png`, `search/hyperparameters.png`, `diagnostics/diagnostics.png`, `evaluation_cost.png` (local CV time, not serving latency).

## Deliberately out of scope

Kept out to stay within the lecture's focus on experiment and model management:

- Fold-level grandchild runs: fold curves and `folds.json` already show the evidence.
- Optuna pruning (`NopPruner`): temporal windows differ in difficulty and fits are small, so every trial completes all three folds.
- Statistical nested CV: MLflow nesting organizes records; it is not an outer evaluation loop.
- Study resume or distributed search: the Optuna study is persisted in SQLite for inspection only; each command starts a new study.
- Automatic feature/parameter importance, serving platforms, monitoring suites, feature stores, containers, CI/CD and cloud deployment. Serving and monitoring belong to Presenter 3 and are discussed conceptually.
- Deprecated registry stages (Staging/Production); aliases and versions are used instead.

## Tests

```bash
uv run --locked --offline --extra tuning python -m pytest -q
```

Integration tests use the real SQLite-backed MLflow path: CV logging, effective parameters, no default test metrics, grid/TPE trials, fold histories, dataset inputs, failed-run status, selection rejection, signature rejection of missing features, model reload, version/alias retrieval and history-only prediction.

## References

- [MLflow Tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [MLflow sklearn flavor](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
- [Model Registry, versions and aliases](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
- [Parent and child runs](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs/)
- [Dataset tracking](https://mlflow.org/docs/latest/ml/dataset/)
- [Optuna TPE sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
