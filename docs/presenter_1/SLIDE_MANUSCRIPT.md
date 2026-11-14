# Presenter 1: From Notebook to Reproducible ML

Slide copy, speaker notes and sources for [mlds2-part1.pptx](mlds2-part1.pptx): a section opener, ten main slides (23 minutes) and one hidden backup. Generated from the deck; edit the deck first, then this file. Every code excerpt is taken from the repository.

## Section opener

**Notes:** Section opener for the whole lecture and Part 1. Introduce the three parts in one sentence: reproducible training (Part 1), experiment tracking and model management (Part 2), using and monitoring the model (Part 3). Then move straight to slide 1.

## Slide 01: The notebook that worked once

**Time:** 3 min

### On slide

- Four real cells from 01_notebook/electricity_forecasting.ipynb.
- raw.loc[500:515, 'load_kw'] *= 1.2 / raw.to_csv('electricity.csv', index=False)
- Running it twice changes the data
- The cell overwrites its own input file.
- scaled = saved['scaler'].transform(new_features) / check = saved['model'].predict(scaled)
- Every consumer repeats preprocessing
- The model was saved apart from its fitted scaler.
- RandomForestRegressor(n_estimators=40, max_depth=10, ...) / train_test_split(X_scaled, y, test_size=0.2)
- Settings and randomness live in cells
- No config file, no seed: a rerun gives different numbers.
- with open('model_final.pkl', 'wb') as f: /     pickle.dump({'model': model, ...}, f)
- Results cannot be traced
- 'final' is the Ridge model, overwritten on every run.
- None of these is a modelling mistake a better metric would catch. They are software problems.

### Speaker notes

Timing 0:00-3:00.

This notebook trained a decent forecasting model, and it is the kind of notebook every one of us has written. These four cells are copied from it unchanged.

[Top left] The cell multiplies part of the data and writes it back to the same CSV. Run the notebook twice and you train on different data, with no record that it happened.

[Top right] To reuse the model, a consumer must remember to apply the saved scaler and the right column order before predicting. Every consumer re-implements preprocessing.

[Bottom left] Hyperparameters are typed into cells, and neither the split nor the forest has a seed. A rerun produces different scores.

[Bottom right] model_final.pkl is overwritten on every run. Here "final" is actually the Ridge model, and nothing records which code or data produced it.

You all know how to avoid leakage and choose metrics; this notebook also leaks, but that is not today's topic. Today's topic is that none of these problems is about modelling. They are software problems, and they block anyone else from using the work.

Transition: "The first fix is structural: where does the code live?"

### Sources (not spoken)

01_notebook/electricity_forecasting.ipynb, cells 7, 9, 11, 16 and 18.

## Slide 02: Library first: notebooks and scripts are thin clients

**Time:** 3.5 min

### On slide

- Move logic out of cells into a tested package; everything else calls it.
- PROJECT LAYOUT (REAL)
- 02_reproducible_project/ / ├─ src/electricity_demo/   # the library / │    prepare · features · split / │    pipeline · evaluate · train / ├─ configs/       ridge.yaml, random_forest.yaml / ├─ tests/         pytest on synthetic data / ├─ notebooks/     exploration.ipynb / ├─ scripts/       one-off orchestration / ├─ data/          DVC pointers, no CSVs in Git / └─ pyproject.toml · uv.lock · .python-version
- electricity_demo / library
- Notebook / explore, call run_training
- CLI / python -m electricity_demo.train
- Tests / pytest, in CI
- Tracking + inference / Parts 2 and 3
- If a notebook cell would be copied, it belongs in the library.

### Speaker notes

Timing 3:00-6:30.

[Left] This is the real layout of our reproducible project. The logic lives in one installable package under src: data preparation, features, evaluation splits, the model pipeline, evaluation and training, each a small module with one responsibility. Configuration, tests, the exploration notebook and data pointers each have their own place.

[Right] The important idea is the direction of the arrows. The library is the only place where the logic exists. The notebook imports it and calls the same run_training function the command line uses. The CLI is a few lines that parse arguments. Tests import it with synthetic data. And in Parts 2 and 3, tracking and inference are just more clients of the same functions.

Notebooks stay valuable for exploration and explanation; they just stop being the only copy of the code.

A word of caution against the opposite extreme: you do not need an abstract class hierarchy for two models. Small functions with clear inputs and outputs are enough.

Ask: "Which of the four notebook problems does this structure fix on its own?" Mostly the copied preprocessing and the hidden cell order.

Transition: "Let us start with preprocessing."

### Sources (not spoken)

02_reproducible_project/ tree; notebooks/exploration.ipynb calls train.run_training.

## Slide 03: Keep fitted preprocessing with the model

**Time:** 2 min

### On slide

- Save one fitted object that turns features into predictions.
- BEFORE: NOTEBOOK BUNDLE
- pickle.dump({'model': model, 'scaler': scaler, /              'columns': cols}, f) /   / # every consumer must remember: / X = new[saved['columns']] / scaled = saved['scaler'].transform(X) / saved['model'].predict(scaled)
- AFTER: SRC/ELECTRICITY_DEMO/PIPELINE.PY
- Pipeline([ /   ('preprocess', ColumnTransformer([ /      ('numeric', StandardScaler(), lag_cols), /      ('calendar', OneHotEncoder(...), /                   ['hour', 'weekday'])])), /   ('model', RandomForestRegressor(...)), / ]) /   / pipeline.predict(X_new)   # the only call
- Fitted inside each CV fold, saved as one object (model.pkl), logged as one MLflow model in Part 2.

### Speaker notes

Timing 6:30-8:30.

[Left] The notebook saved a dictionary: the model, the scaler and the column list. Every consumer has to know that recipe and apply it in the right order. Forget the scaler and you get confident nonsense, without any error.

[Right] A scikit-learn Pipeline puts the learned transformations and the estimator into one object. Fitting the pipeline fits the scaler and encoder on exactly the rows the model is trained on, which also means they are refitted inside every cross-validation fold. Predicting applies them automatically.

The consumer now makes one call. That single object is what we save as model.pkl, and in Part 2 it is what MLflow packages as a logged model.

This is not new for you as an ML technique. The point here is the interface: a model artifact should contain everything needed to go from its declared inputs to predictions.

Transition: "But our model's inputs are themselves computed features. Who builds those?"

### Sources (not spoken)

01_notebook cell 9 (bundle); src/electricity_demo/pipeline.py; https://scikit-learn.org/stable/modules/compose.html

## Slide 04: One feature definition for training and inference

**Time:** 2 min

### On slide

- Two implementations of the same feature drift apart silently.
- SRC/ELECTRICITY_DEMO/FEATURES.PY
- def feature_table(load, config): /     f = pd.DataFrame(index=load.index) /     for lag in config['lags']: /         f[f'lag_{lag}h'] = load.shift(lag) /     for w in config['rolling_windows']: /         f[f'mean_{w}h'] = (load.shift(1) /                            .rolling(w).mean()) /     f['hour'] = f.index.hour /     f['weekday'] = f.index.dayofweek /     return f
- feature_table(load, config)
- build_features / training rows + target
- build_next_features / next-hour row, no target
- test_next_hour_features_match_training_without_the_target
- Inference (Part 3) calls build_next_features; it never re-implements a feature.
- Training-serving skew starts with copied feature code. Share one function, and test that both paths agree.

### Speaker notes

Timing 8:30-10:30.

Our features are lags and rolling means of past load. They are computed before the pipeline, so the question is where that code lives at prediction time.

[Left] This is the single definition. Every feature is shifted by at least one hour, so a row only uses observations that were available when the forecast was made.

[Right] Training calls build_features, which adds the target and drops rows without enough history. Inference calls build_next_features, which appends one empty row for the next hour and runs the very same function. An earlier version of our project had two separate implementations; it worked, but only because a test compared them. Now there is one function, and the test still guards the contract.

This is the most common way production models break quietly: someone rewrites a feature in the serving code, it computes something slightly different, and nothing raises an error. Presenter 3 comes back to this as training-serving skew.

Transition: "Code is shared. Now, what does it need to run?"

### Sources (not spoken)

src/electricity_demo/features.py; tests/test_features.py.

## Slide 05: Dependencies: consumer vs training

**Time:** 2 min

### On slide

- Declare them in pyproject.toml, lock them in uv.lock, install them the same way everywhere.
- PYPROJECT.TOML (ABRIDGED)
- [project] / dependencies = [pandas, numpy, /                 scikit-learn, pyyaml] /   / [project.optional-dependencies] / tracking = [mlflow]          # registry, inference / tuning   = [mlflow, optuna] /   / [dependency-groups] / train = [matplotlib, rich, dvc]  # training only / dev   = [pytest, ...]
- uv lock
- Exact versions for every package, committed as uv.lock
- uv sync --locked
- Installs exactly the lock; fails if it is out of date
- uv run --locked …
- Every command runs in that same environment
- A lockfile does not pin the data, uncommitted code, OS and numerical libraries, or unseeded randomness.
- A consumer installs the package without training tools; a developer gets everything with uv sync.

### Speaker notes

Timing 10:30-12:30.

[Left] pyproject.toml declares three different audiences. The package's own dependencies are only what is needed to build features and predict: pandas, NumPy, scikit-learn and PyYAML. MLflow is an optional extra for whoever loads models from the registry. Training tools such as plotting, console output and DVC are a development group: a developer gets them with uv sync, but an application that installs the package does not drag them along.

[Right] uv lock resolves exact versions of every package, including indirect ones, and we commit that file. uv sync with --locked installs exactly that environment and refuses to continue if the lock no longer matches pyproject. uv run makes every command, tests included, use the same environment, on a laptop or in CI.

[Bottom right] The honest limit: a lockfile pins Python packages. It does not pin the data, code you have not committed, the operating system and numerical libraries underneath, or randomness you did not seed. Those need the other practices in this section.

Transition: "Packages are pinned. What about the values we used to type into cells?"

### Sources (not spoken)

02_reproducible_project/pyproject.toml, uv.lock; https://docs.astral.sh/uv/concepts/projects/dependencies/

## Slide 06: Configuration instead of hard-coded values

**Time:** 2 min

### On slide

- Change an experiment by changing a file, not by editing code.
- CONFIGS/RANDOM_FOREST.YAML (ABRIDGED)
- seed: 42 / dataset: data/prepared/hourly.csv / features: /   lags: [1, 2, …, 24, 168] /   rolling_windows: [24, 168] / evaluation: /   horizon_hours: 1 /   cv_splits: 3 /   cv_validation_hours: 336 / model: /   type: random_forest /   params: {n_estimators: 40, max_depth: 10}
- 1
- Validated when loaded
- config.load_training_config rejects unknown models, unsupported horizons and malformed search grids before any training starts.
- 2
- Saved with every run
- The resolved config.yaml and its hash go into the run folder and manifest.
- 3
- One place per decision
- Features, evaluation design and model settings are no longer scattered through cells.
- Configs are code: they are versioned in Git and reviewed like code.

### Speaker notes

Timing 12:30-14:30.

[Left] Every setting that the notebook typed into cells now lives in one YAML file: the seed, the dataset version, feature definitions, the evaluation design and the model settings. Running a different experiment means passing a different file.

[1] Configuration is validated when it is loaded. A typo in the model type or an unsupported forecast horizon fails immediately, with a clear message, instead of half an hour into training.

[2] The resolved configuration is copied into every run folder, and its hash is recorded, so a result can always be traced back to its settings.

[3] Each decision has one place. Nobody needs to search notebook cells to find which lags were used.

You do not need a configuration framework for this. Plain YAML plus a validation function is enough for a project of this size; tools like Hydra or pydantic help when configurations grow.

Transition: "The last input to pin is the data itself."

### Sources (not spoken)

configs/random_forest.yaml; config.load_training_config; train.save_run.

## Slide 07: Version control: code in Git, data by hash

**Time:** 1.5 min

### On slide

- Git holds everything small and reviewable; large data is identified, not committed.
- Raw UCI archive
- never modified
- prepare.py
- aggregate, validate
- hourly.csv + manifest.json
- source and output hashes
- .dvc pointer in Git
- bytes in DVC storage
- IN GIT
- src/ and tests/, configs/*.yaml, pyproject.toml and uv.lock, .dvc pointers, CI workflow
- NOT IN GIT (.GITIGNORE)
- Data files, fitted models, outputs/ run folders, .venv, MLflow store. They are reproduced or restored, and identified by hash.
- Workflow: branch, pull request, CI runs the tests, a teammate reviews, then merge. Training refuses data that does not match its manifest.

### Speaker notes

Timing 14:30-16:00.

Keep this short; the point is the principle.

[Top] The raw archive is never edited. A preparation step aggregates and validates it, and writes the prepared file together with a manifest of hashes. DVC stores the file contents; Git stores only a small pointer. Restoring an older data version means checking out an older pointer.

[Left and right] The rule for Git: commit what is small, textual and reviewable: code, tests, configs, pyproject and the lockfile, the data pointers and the CI workflow. Do not commit data, fitted models, run outputs, environments or the MLflow store; those are reproduced or restored, and identified by their hashes. The .gitignore makes that rule automatic.

[Bottom] Changes go through a branch and a pull request: CI runs the tests, a teammate reviews, then it merges. That is how a team keeps the training code correct. And training compares the data with its manifest and refuses a file that was changed by hand, unlike the notebook cell that rewrote its own CSV.

Every run records the commit and the data hash. That is what lets Part 2 say exactly which code and data produced a model.

Transition: "With code, environment, configuration and data under control, how do we know the code is still correct after the next change?"

### Sources (not spoken)

prepare.py; data/README.md; experiments/README.md; .gitignore files; .github/workflows/tests.yml; tests/test_data.py.

## Slide 08: Testing ML code: contracts, not accuracy

**Time:** 4 min

### On slide

- Tests check behaviour on small synthetic data. They run on every change.
- TESTS.YML (GITHUB ACTIONS)
- on: [push, pull_request] / steps: / - uses: astral-sh/setup-uv@v6 / - run: uv sync --locked /        --extra tuning / - run: uv run --locked /        --extra tuning /        python -m pytest -q
- Tests are not quality gates. "Is the model good enough to ship?" is a separate, explicit check (Part 2).
- 32 tests, about 40 seconds, no dataset download.
- Checks | Example test in our repository / Data contract | test_invalid_hourly_data_is_rejected / No future information | test_current_and_future_targets_cannot_change_current_or_past_features / Preprocessing fitted in-fold | test_scaler_is_fitted_only_on_fold_training_rows / Train / inference parity | test_next_hour_features_match_training_without_the_target / Save and reload | test_training_repeats_and_saved_pipeline_reproduces_predictions / Invalid input rejected | test_next_hour_features_reject_short_or_gapped_history / Failures are visible | test_training_failure_is_recorded

### Speaker notes

Timing 16:00-20:00.

Accuracy tells you whether a model is good. Tests tell you whether the code still does what it promises. In an ML project the most valuable tests check contracts.

[Table, top to bottom] Invalid data, such as gaps, duplicates or negative load, is rejected. Changing future values cannot change current features, which is a leakage test written as code. Preprocessing is fitted only on each fold's training rows. Inference builds exactly the features training built. A saved and reloaded pipeline reproduces the same predictions, within a stated tolerance. Bad history at prediction time fails loudly. And a failed training run is recorded as failed, not silently ignored.

All of them use tiny synthetic time series, so they run in seconds without the real dataset.

[Top right] That is what makes automation possible: GitHub Actions runs the whole suite on every push and pull request, using the same uv commands as your laptop.

[Bottom right] Do not confuse tests with model-quality checks. "Is this model better than the current one?" is a decision rule over evaluation results, which is part of the selection step in Part 2.

Ask: "Which of these tests would have caught the notebook problems from slide 1?"

Transition: "Now training can run without a person."

### Sources (not spoken)

tests/; .github/workflows/tests.yml.

## Slide 09: One headless command

**Time:** 2 min

### On slide

- The same function runs from a terminal, a notebook, CI or a scheduler.
- uv run --locked python -m electricity_demo.train --config configs/random_forest.yaml
- INPUTS
- config file / data version (hash) / uv.lock
- EXECUTION
- load_development / fit_and_evaluate / save_run
- OUTPUTS: OUTPUTS/<RUN-ID>/
- config.yaml · metrics.json / predictions.csv · model.pkl / feature_spec.json · manifest.json
- manifest.json: Git commit and dirty flag, data hash, lockfile hash, resolved config and split boundaries.
- No cell order, no manual steps: if it runs once, it runs again.

### Speaker notes

Timing 20:00-22:00.

[Top] This is the whole training procedure: one command, given one configuration file.

[Inputs to outputs] The inputs are explicit: the config, the data version identified by its hash, and the locked environment. Internally, training is three small functions: load and validate the data, run the chronological evaluation and refit the selected model in memory, then save the run. Because they are separate functions, a notebook, a test or the tracking code in Part 2 can call them without going through files.

Every run writes its own folder with a unique ID: the resolved configuration, metrics, predictions, the fitted pipeline, the feature specification, and a manifest recording the Git commit, whether the working tree had uncommitted changes, the data hash and the lockfile hash.

Nothing depends on cell order or on someone remembering a manual step. That is what lets CI or a scheduler retrain the model later.

Transition: "So after a few experiments, what do we have?"

### Sources (not spoken)

train.py (load_development, fit_and_evaluate, save_run); outputs/<run-id>/.

## Slide 10: A few experiments later: a folder of runs

**Time:** 1 min

### On slide

- We have reproducible runs. We do not yet have a way to manage them.
- 2026-11-13_22-13-12Z_ridge_v1_87e9e4/
- 2026-11-13_22-13-27Z_random_forest_v1_74ded3/
- 2026-11-13_22-13-46Z_ridge_v1_b21b7d/
- 2026-11-13_22-14-53Z_random_forest_v1_8db6c5/
- …  and comparison.md, written by our own report.py
- This works for one person on one laptop. How does a team compare hundreds of runs, and find the exact model an application should use?
- Part 2: experiment tracking and model management

### Speaker notes

Timing 22:00-23:00.

[Left] After a few experiments we have reproducible runs, each in its own folder, and a comparison report that we wrote ourselves.

[Right] For one person that is enough. For a team it is not: you cannot search someone else's laptop, and nobody knows which of these models an application should load.

Handoff to Presenter 2: "We can reproduce a training execution and evaluate it correctly. When we try more models and settings, how do we compare those executions and find the exact model we want to use?"

### Sources (not spoken)

02_reproducible_project/outputs/; report.py.

## Backup A: the evaluation protocol in code

**Time:** hidden backup

### On slide

- Covered by Part 1's code and tests; not part of the timed flow.
- Chronological folds
- Three expanding windows of 336 hours; no shuffling (split.py) / test_folds_respect_prediction_origins_and_exclude_final_test
- Final test opt-in
- Scored only with --final-test, after selection / test_default_training_does_not_score_final_test
- Supported scope
- One-hour horizon only; other horizons are rejected / test_unsupported_forecast_horizon_is_rejected
- Repeatability
- Same config, data and lock: predictions equal within 1e-9 / test_training_repeats_and_saved_pipeline_reproduces_predictions

### Speaker notes

Hidden backup, for questions about evaluation.

The audience already knows temporal validation and leakage, so the main flow does not teach them. If asked: the project evaluates one-hour-ahead forecasts on three expanding chronological windows, keeps the final two months reserved until --final-test is passed explicitly, rejects horizons it was not designed for, and tests that repeated runs reproduce the same predictions within a stated numerical tolerance.

### Sources (not spoken)

split.py, train.py; tests/test_splits.py, tests/test_training.py.
