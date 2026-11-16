# Presenter 2: Experiment Tracking and Model Management with MLflow

Slide copy, visuals, speaker notes and sources for [mlds2-lecture.pptx](mlds2-lecture.pptx): a section opener, twelve main slides (25 minutes) and four hidden backups (A–D). Generated from the deck; edit the deck first, then this file. Screenshots are authentic local MLflow 3.18.0 captures of batch `20261113T221340Z-4f68dd`; no training or registry change happens on stage.

Bracketed cues such as [1] or [Left] in the notes refer to the numbered markers and regions on the slide.

## Section opener

MLDS2 · From Notebook to Production · Part 2 of 3  
**Experiment Tracking and Model Management**  
From a reproducible training command to a model an application can retrieve, with MLflow  
25 minutes

**Notes:** Section opener. Do not stop here: move straight to slide 1 after Presenter 1's handoff question.

## Slide 01: We already built a tracker by hand

**Time:** 2 min

### On slide

- Part 1's run folders record everything. They do not scale to a team.
- OUR OUTPUTS/ FOLDER TODAY
- 2026-11-13_22-13-12Z_ridge_v1_87e9e4/
- 2026-11-13_22-13-27Z_random_forest_…/
- manifest.json  metrics.json  model.pkl
- comparison.md   (report.py, 200 lines)
- … one folder per run, one laptop
- WHAT BREAKS WHEN A TEAM USES IT
- Search runs across people and machines / Know which runs are comparable / Find last month's run and its exact code / Hand one model to another team
- Tracking keeps the same records: shared, searchable and linked to the models they produced.

### Visual

Our real outputs/ folder and hand-written comparison report versus what breaks when a team uses it.

### Speaker notes

Timing 0:00-2:00.

Presenter 1 ended with a reproducible command, and every run already leaves a folder: config, metrics, a manifest with the Git commit and data hash, the model. Plus a comparison report that report.py generates, about two hundred lines of our own code.

[Point left] So we have built a small tracker by hand. For one person on one laptop, it works.

[Point right] Now put five people on the project for a month. You cannot search runs on someone else's laptop. You must know which runs are even comparable. You need last month's run and the exact code behind it. And another team needs one model they can load without walking through our folders.

That is the gap experiment tracking fills: the same records, but in a shared, searchable store, linked to the models they produced. Teams use a tool for this rather than growing their own report scripts.

Ask: "Who has kept results in a spreadsheet or a folder of pickles?" Most hands go up.

Transition: "So where does a tool like MLflow fit into the lifecycle?"

### Sources (not spoken)

02_reproducible_project/outputs/; report.py; train.save_run.

## Slide 02: Where MLflow fits

**Time:** 1.5 min

### On slide

- Tracking and model management connect reproducible training to application use.
- PART 1
- Reproducible training
- Code · data · environment · evaluation
- MLFLOW TRACKING
- Experiment tracking
- Runs · parameters · metrics · artifacts
- MLFLOW MODELS
- Model management
- Logged models · registry versions · aliases
- PART 3
- Application use
- Serving · monitoring · drift
- Boundary: this section ends with a versioned model we can retrieve, not a deployed service.
- The same ideas apply to other trackers (e.g. Weights & Biases). We run MLflow locally; a team would share one tracking server.

### Visual

Four-step lifecycle cards with MLflow Tracking and MLflow Models highlighted; boundary callout.

### Speaker notes

Timing 2:00-3:30.

Experiment tracking means recording what each execution used, what it measured and what it produced.

[Point to the two green boxes] MLflow covers two parts of the lifecycle. Tracking is an API plus a UI for runs, metrics and files. Model management is a standard model package plus a registry with names and versions.

Presenter 1 owns the left box: correct evaluation and recoverable inputs. Presenter 3 owns the right box: using and monitoring the model. A tracker cannot fix a leaking split. It will faithfully record a misleading score.

[Point to the boundary] My section ends with a model we can identify and load, not with a running service.

MLflow is one concrete tool; Weights & Biases and managed ML platforms implement the same ideas. We run it on a laptop; a team would use a shared tracking server with durable artifact storage.

Transition: "Let us define the objects MLflow organizes."

### Sources (not spoken)

https://mlflow.org/docs/latest/ml/tracking/; https://mlflow.org/docs/latest/ml/model-registry/workflow/

## Slide 03: Experiments, runs and what a run records

**Time:** 1.5 min

### On slide

- An experiment groups related runs and logged models. A run is the record of one execution.
- Experiment: electricity-mt158-lecture
- Run / Ridge search
- Run / Forest search
- Run / Baseline
- One run can cover a single fit, several CV folds or a whole search. Choose its scope deliberately.
- Parameters
- Settings you choose
- alpha, max_depth, seed
- Metrics
- Outcomes you measure
- validation MAE, fold scores
- Artifacts
- Files you keep
- plots, configs, fitted model
- Tags + datasets
- Searchable context
- purpose, data digest, code version

### Visual

Experiment-to-runs hierarchy plus four cards: parameters, metrics, artifacts, tags and datasets.

### Speaker notes

Timing 3:30-5:00.

In everyday language every attempt is an "experiment". In MLflow, an experiment is a container for one task, here forecasting hourly load for one customer. It groups runs and, in MLflow 3, the logged models they produce.

A run is the record of one execution: an ID, a status, and whatever we log. Note the scope: one run can cover a single fit, three cross-validation folds, or a whole search. Decide that scope on purpose.

[Point to the four cards] Parameters are settings you choose: alpha, tree depth, seed. Metrics are measured outcomes: validation MAE, per-fold scores. Artifacts are files: plots, configs, the fitted model. Tags and dataset inputs add searchable context: purpose, data digest, code version. Parameters are fixed once logged; tags can be updated later.

Ask: "Where would you put a learning rate, an error score and a residual plot?"

Transition: "How are these records created from existing training code?"

### Sources (not spoken)

https://mlflow.org/docs/latest/ml/tracking/ (experiments group runs and models).

## Slide 04: Adding MLflow to existing training

**Time:** 2.5 min

### On slide

- Training computes the result. A thin tracking wrapper records it.
- mlflow.set_experiment("electricity-mt158-lecture") /   / with mlflow.start_run(run_name="forest-depth-10"): /     mae = evaluate_cv(pipeline, folds)   # compute /     pipeline.fit(X_dev, y_dev)           # compute /   /     mlflow.log_params({"max_depth": 10, "min_samples_leaf": 3}) /     mlflow.log_metric("cv_mae_kw", mae) /     mlflow.log_artifact("forecast.png") /     mlflow.sklearn.log_model(pipeline, name="forecast_pipeline", /                              input_example=X_dev.head())
- 1
- Open a run
- A context manager. An exception marks the run FAILED.
- 2
- Compute
- The same temporal CV and refit as before.
- 3
- Record
- Code version, data identity, config, environment, metrics, files.
- 4
- Package
- The fitted pipeline becomes a logged model.
- Separate concerns: train.py never imports mlflow. tracking.py wraps it, so training still runs in tests, notebooks and CI without a tracker.

### Visual

Annotated logging pattern with four callouts; separate-concerns strip (train.py has no MLflow import).

### Speaker notes

Timing 5:00-7:30.

This is the simplified pattern. The point is where the tracking code lives.

[1] Choose the experiment, then open a run as a context manager. If the code raises an exception, MLflow marks the run FAILED, so a half-finished attempt is never mistaken for a result.

[2] The green lines compute: the same chronological CV and final refit as before.

[3] These lines only record. In a real project, record enough for someone else to investigate the run: the code version, the data identity, the config, the environment, metrics and output files.

[4] The yellow call packages the fitted pipeline as a logged model; that is the bridge to model management.

[Bottom] In our repository, train.py never imports MLflow. A thin wrapper, tracking.py, calls the same training function and logs its result. So the training code still runs in unit tests, in a notebook or in CI without a tracking server. Keep tracking as an adapter around training, not woven through it.

Autologging can record supported libraries automatically; we log manually because only we know which score is the honest validation score.

Transition: "Here is where those records appear."

### Sources (not spoken)

train.fit_and_evaluate, train.save_run, tracking.run_tracked_training, tracking.log_training_result.

## Slide 05: Finding runs in the Tracking UI

**Time:** 2 min

### On slide

- Filter by tags, then expand a search into its candidate runs.
- 1
- 2
- 3
- 1
- Filter with tags
- tags.lecture_batch = '…' keeps one batch. Tags find work among hundreds of runs.
- 2
- Parent and child runs
- A parent is one search; each child is one candidate: 10 forest settings, 3 Ridge alphas.
- 3
- Settings beside outcomes
- Parameter and metric columns show what changed and what happened. A run name opens its record.
- Actual MLflow 3.18 Runs view of the recorded lecture batch.

### Visual

Actual Runs view with tag filter and expanded parent/child search runs, three markers.

### Speaker notes

Timing 7:30-9:30.

This is the real Runs view of our recorded batch.

[1] The search box filters on a tag: only runs from one batch. Tags are how you find work among hundreds of runs, by purpose, dataset, author or status.

[2] The search is organized as parent and child runs. The parent is one search for a model family; each child is one candidate configuration: ten forest settings proposed by Optuna, three Ridge alphas. This is the usual way to log a hyperparameter search. Nesting organizes records; it is not nested cross-validation.

[3] Parameters and metrics sit side by side, so you can see which settings changed and what happened. Clicking a run name opens its full record.

Do not read numbers aloud here. The point is navigation: experiment, runs, one run.

Pause and ask: "Which column tells you the tree depth of each candidate?"

Transition: "Tables are one view. MLflow can also chart the same records."

### Sources (not spoken)

assets/mlflow_screenshots/runs.png; tune.run_search.

## Slide 06: Comparing runs visually

**Time:** 2 min

### On slide

- Metrics logged with a step, and parameters per trial, become MLflow charts with no plotting code.
- 1
- Fold error per run
- Step = chronological fold. All runs worsen in fold 3; the baseline stays far above.
- 2
- Optuna search progress
- Step = trial. Solid: each trial's error. Dashed: best so far; most gain by trial 3, a small gain at trial 9.
- 3
- Settings vs error
- One line per forest trial. Shallow trees (depth 3–5) give the three worst errors; depth 10+ all score about 8.4.
- Ten trials show search behaviour, not robust parameter importance. Folds are time windows, not epochs.

### Visual

Three native MLflow charts: fold_mae_kw per run by step, the forest search's trial objective and best-so-far, and parallel coordinates of the forest trials.

### Speaker notes

Timing 9:30-11:30.

These are MLflow's own charts. We only logged metrics with a step and parameters per trial; the UI draws them, so nobody writes plotting code to compare runs.

[1] Fold error per run. We logged fold_mae_kw with step equal to the fold number. Forest and Ridge stay close and far below the seasonal baseline, and every method gets worse in the third, latest window. That is a reason to look closer, not proof of drift.

[2] The Optuna search on the forest parent run. Step is the trial number. The solid line is each trial's error; the dashed line is the best so far. Most of the gain is made by trial 3, with one small further improvement at trial 9; the spikes are the search exploring poor settings.

[3] Parallel coordinates across the ten forest trials: each line is one trial, coloured by validation error. The three worst trials all use shallow trees, depth 3 to 5. Every trial with depth 10 or more lands around 8.4, whatever its feature fraction, so depth is the one setting that clearly matters here.

[Bottom] Keep the claims modest. Ten trials illustrate search behaviour; they do not prove which parameter matters. And the x-axis in the first chart is time windows, not training epochs.

Transition: "Charts help us investigate. But the decision itself should be code."

### Sources (not spoken)

assets/mlflow_screenshots/folds.png, search.png, params.png; tracking.log_fold_metrics; tune.run_search.

## Slide 07: Selecting a model: compare, then decide in code

**Time:** 2 min

### On slide

- The UI is for investigating. The decision rule should be code you can rerun.
- Same baseline error and comparison_group = comparable runs
- Ranking metric: validation MAE (kW)
- RANK ONLY COMPARABLE RUNS
- Same data digest, same target hours, same protocol (comparison_group tag), status FINISHED.
- Select with code
- runs = mlflow.search_runs( /     experiment_names=["electricity-mt158-lecture"], /     filter_string="tags.run_kind = 'study' and " /                   "attributes.status = 'FINISHED'", /     order_by=["metrics.cv_mae_kw ASC"]) / best = runs.iloc[0] / assert best["tags.comparison_group"] == group / assert best["metrics.cv_mae_kw"] < baseline_mae / mlflow.register_model(best["tags.model_uri"], name)
- Rerunnable, reviewable, CI-ready. Simplified from selection.select_candidate.

### Visual

Actual Compare Runs rows (identical baseline and comparison_group, ranking metric) beside a selection-as-code panel using mlflow.search_runs.

### Speaker notes

Timing 11:30-13:30.

[Left] MLflow's Compare Runs view for the two search parents and the seasonal baseline. Validation MAE: forest 8.39, Ridge 9.66, baseline 30.12 kilowatts. The amber rows are a quick sanity check: the baseline error and the comparison_group tag are identical in every run. That tag hashes the data, features, evaluation design, code and lockfile, so equal values mean the runs are comparable. MLflow will happily put incomparable numbers side by side, so check data digest, comparison_group and status first.

[Right] The UI is for investigating. The decision itself should be code: search runs with a filter, order by the agreed metric, check comparability and the baseline, then register. Our demo does exactly this in selection.select_candidate. Because it is code, it gives the same answer every time, a colleague can review it, and it can run in CI after every training job instead of someone clicking through a UI.

These are selection scores: the forest search tried ten settings and Ridge three, so they are optimistic. Freeze the choice before any final test.

Transition: "To investigate the selected candidate, we need more than metrics."

### Sources (not spoken)

assets/mlflow_screenshots/compare.png; selection.select_candidate; https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html#mlflow.search_runs

## Slide 08: Provenance: where did this result come from?

**Time:** 2 min

### On slide

- A run links its result to code, data, environment and outputs.
- 1
- 2
- 3
- 4
- 1
- Source
- Script, branch and Git commit.
- 2
- Dataset
- Digest of the data the run used.
- 3
- Tags
- Data hash, lockfile hash, commit 64ec6b2 and source_dirty: false.
- 4
- Artifacts
- Config, predictions, plots, archived source and uv.lock.
- An ID identifies a resource; it does not keep it. Recovery still needs Git, DVC and uv.

### Visual

Actual run overview (source commit, dataset, tags incl. source_dirty: false) and artifact tree, four markers.

### Speaker notes

Timing 13:30-15:30.

This is one run's record, the selected forest search.

[1] Source: which script, branch and commit produced it.

[2] Datasets: the digest of the data it used.

[3] Tags: the data hash, the lockfile hash, the exact commit, and source_dirty: false. We committed before running this batch, so anyone can check out commit 64ec6b2 and get exactly this code. Had there been uncommitted changes, the tag would say true, and the commit alone would not describe the code. That is provenance: the origin of a result, recorded honestly.

[4] Artifacts hold the evidence itself: the configuration, predictions, plots, plus archived source files and the uv.lock.

Why it matters: when a result changes, provenance tells you whether the data, the settings or the code changed.

[Bottom] The limit: an ID or a hash identifies something; it does not keep it. A commit hash cannot recover uncommitted code, and a data digest cannot recover deleted data. That is why Presenter 1's Git, DVC and uv still matter. MLflow records evidence; it does not guarantee reproducibility.

Transition: "One artifact is special: the model. Three objects get confused here."

### Sources (not spoken)

assets/mlflow_screenshots/run_overview.png, run_artifacts.png; https://mlflow.org/docs/latest/ml/dataset/

## Slide 09: Run, logged model, registered version

**Time:** 1.5 min

### On slide

- Three different objects answer three different questions.
- Tracked run
- What was executed and measured?
- random_forest-tpe-study
- Logged model
- What exactly was fitted, and how do I load it?
- forecast_pipeline
- Registered version
- Which model does the team refer to?
- electricity-mt158-lecture/v2
- Registration creates a named, numbered reference. It is not approval and not a deployment.
- In our demo the registered model shares the experiment's name. They are still different objects.

### Visual

Three cards: tracked run, logged model, registered version, with real identifiers.

### Speaker notes

Timing 15:30-17:00.

Three objects that are easy to mix up.

[Left] A run answers: what was executed and measured?

[Middle] A logged model answers: what exactly was fitted, and how do I load it? In MLflow 3 it has its own ID and links back to the run that produced it.

[Right] A registered model version answers: which model does the team refer to? It is a numbered entry under a shared name.

Each arrow is a deliberate step. Logging creates the model package; registration gives a selected package a name and a number.

[Bottom] Registration is neither business approval nor deployment. Nothing is being served yet.

One honest detail: in our demo the registered model and the experiment have the same name. They are still different objects. One groups runs, the other groups model versions.

Transition: "Let us open the logged model."

### Sources (not spoken)

https://mlflow.org/docs/latest/ml/tracking/; https://mlflow.org/docs/latest/ml/model-registry/workflow/

## Slide 10: The model contract: what a consumer receives

**Time:** 3 min

### On slide

- Another developer gets a reference, an interface and an environment, and must know what is not inside.
- HANDED TO THE APPLICATION
- Reference
- models:/electricity-mt158-lecture/2
- Package
- Fitted scaling + encoding + estimator
- Input
- 29 engineered numeric columns
- Output
- Next-hour load in kW
- Environment
- requirements.txt (no training tools)
- 1
- 2
- NOT INSIDE THE MODEL: FEATURES BUILT FROM RAW METER HISTORY
- Option A · features inside the model / A custom pyfunc wraps the feature code; consumers send raw history.
- Option B · shared feature library (our demo) / Consumers call build_next_features, the same feature code training uses.

### Visual

Consumer contract card, actual logged-model files and signature, and the two options for feature construction.

### Speaker notes

Timing 17:00-20:00.

This slide answers the key question of the section: what does another developer or application actually receive?

[Left] A reference to one exact version, a package with the fitted preprocessing and estimator together, an input interface, an output with units, and the environment to load it. [Screenshot 1] Those are the real files of our logged model. [2] The signature lists twenty-nine named double columns. A signature checks names and types; it cannot check that features were computed correctly or were available at prediction time.

[Bottom] The important detail: the model does not receive raw meter readings. Building the twenty-nine lag and rolling features from raw history happens outside it. That is a design decision with two valid answers.

Option A: wrap the feature code inside the model as a custom pyfunc, so consumers send raw history. One artifact, but more model code to maintain.

Option B, our demo: publish the feature code as a shared library. Inference calls build_next_features, which runs the very same feature_table function as training, and a test checks that the two agree. The cost: consumers must install the matching library version.

Either way, write the contract down. For Presenter 3: serving this exact model means sending the twenty-nine features, not raw meter data.

Transition: "With many logged models, how does an application find the selected one, and who decides which one that is?"

### Sources (not spoken)

assets/mlflow_screenshots/model.png; tracking.log_training_result; predict.predict_next; features.build_next_features.

## Slide 11: Registry and aliases: the hand-off boundary

**Time:** 2.5 min

### On slide

- Data scientists publish versions; applications load a name the team controls.
- 1
- 2
- 3
- 1 Numbered versions never change  2 champion → version 2  3 lineage to run and logged model
- Consumers load by alias
- uri = "models:/electricity-mt158-lecture@champion" / model = mlflow.pyfunc.load_model(uri) / v = client.get_model_version_by_alias( /     "electricity-mt158-lecture", "champion") / log(v.version)   # record what was used
- Moving champion is a team decision
- Tests pass and the selection rule agrees / A second person reviews the change / Old and new version are recorded
- Moving an alias changes the next load. A running service keeps its model until it reloads.

### Visual

Actual registry versions table and version-2 lineage strip; load-by-alias code and the alias-change review rules.

### Speaker notes

Timing 20:00-22:30.

The registry is where data science work is handed to the people who run applications.

[1] One registered name groups numbered versions: version 1 is Ridge, version 2 the forest. A number never changes, and a higher number is only a later registration, not a better model.

[2] An alias is a movable name for one version. "champion" is our convention; MLflow does not judge quality. MLflow's old Staging and Production stages are deprecated; aliases and tags replace them.

[3] Each version links back to its source run and logged model, so anyone can see why it was selected.

[Right, top] Applications load by alias, so switching models needs no code change. Resolve and log the version number too, so every prediction stays traceable after the alias moves.

[Right, bottom] Because one alias move changes what production loads, treat it like a code change: the tests pass, the selection rule agrees, a second person reviews it, and the old and new versions are recorded. Our demo records every alias move; in the recorded run we moved champion to version 1 and back.

[Bottom] Moving the alias affects the next load. A batch job picks it up on its next run; a running service keeps its model until it reloads. That is where Presenter 3 continues.

Transition: "So when is a model ready to hand over?"

### Sources (not spoken)

assets/mlflow_screenshots/registry.png, version.png; registry.assign_alias; predict.predict_next; https://mlflow.org/docs/latest/ml/model-registry/workflow/

## Slide 12: From a notebook model to a model others can use

**Time:** 2.5 min

### On slide

- Hand a model over only when every item is true.
- Training runs headless from a committed revision
- Environment locked; data version identifiable
- Tests pass in CI: data contract, parity, reload
- Run tracked: settings, metrics, data identity, files
- Model logged with signature and dependencies
- Registered version with lineage; alias moved after review
- Input contract and feature code documented
- Known limits stated: nothing monitored yet
- We can identify and retrieve a selected model version. How does an application use it, and how do we know it still works?

### Visual

Eight-item readiness checklist tagged Part 1 / MLflow / Team / Part 3; handoff question.

### Speaker notes

Timing 22:30-25:00.

This is the difference between a model that works in a notebook and one another team can rely on.

[Left column] Presenter 1's practices: training runs as one command from a committed revision, the environment is locked and the data version identifiable, and tests that run automatically in CI cover the data contract, training-inference parity and model reload.

[Right column] What we added with MLflow: a tracked run, a logged model with signature and dependencies, and a registered version with lineage whose alias moves only after review. Then the team's part: a written input contract and the feature code consumers need. And an honest statement of limits: nothing is monitored yet.

MLflow covers three of these eight items. The rest is engineering practice; the tool does not provide it.

Handoff to Presenter 3: "We can identify and retrieve a selected model version. How does an application use it, and how do we know it still works?"

### Sources (not spoken)

Lecture goals; repository tests, CI workflow and registry.

## Backup A: forecasting evaluation details

**Time:** hidden backup

### On slide

- Development selection CV on identical target hours.
- Model | MAE (kW) | RMSE (kW) / Same hour yesterday | 30.124 | 44.606 / Ridge (alpha 10) | 9.663 | 13.501 / Random forest (depth 15) | 8.390 | 13.146
- Every method has its largest error in the last window. That motivates investigation; it does not establish drift. Fold spread is descriptive, not a confidence interval.

### Visual

Table of CV MAE/RMSE and native line chart of MAE per fold.

### Speaker notes

Hidden backup, for questions only.

The reference predicts the same hour's observed load from the previous day. All methods are scored on identical targets, with the same data snapshot, features and folds. Each new observation becomes available before the next one-hour forecast, so a two-week validation block holds advancing one-step forecasts, not one fixed-origin forecast.

These CV predictions come from fold-specific fits; the saved model is a later refit on all development data. That is why the model-linked metrics are called selection_cv_*.

Prompt: "What does the average score hide here?"

### Sources (not spoken)

evidence_snapshot.json; ../MLFLOW_ENHANCED_RESULTS.md.

## Backup B: recorded inference contract

**Time:** hidden backup

### On slide

- Input: at least 168 consecutive observed hours (timestamp, load_kw).
- Field | Recorded value / Prediction origin | 2014-11-01 00:00 (last observed hour) / Target | 2014-11-01 01:00 / Prediction | 131.984 kW / Resolved model | models:/electricity-mt158-lecture/2 (forest) / Clock | Published Portuguese wall clock, not UTC
- Shared features.build_next_features turns observed history into 29 inputs; no future value is supplied. Input validity can be checked immediately; accuracy only once the actual load arrives.

### Visual

Recorded inference record table.

### Speaker notes

Hidden backup, for questions only.

The application supplies a table of timestamp and observed load in kW, at least 168 consecutive hours. The shared helper builds the next hour's 29 features, and the loaded pipeline returns one prediction. We know exactly which model produced it: version 2, linked to its source run. When another observation arrives, the next forecast advances by one hour.

For monitoring: data arrival and input validity are known immediately; forecast error only after the actual load is observed.

### Sources (not spoken)

outputs/lecture-20261113T221340Z-4f68dd/evidence.json; predict.py; features.build_next_features.

## Backup C: validation error by target hour

**Time:** hidden backup

### On slide

- Selected forest, out-of-fold predictions from the three validation windows.
- Diagnostics suggest where to investigate. They do not establish drift, a cause or an alert policy.

### Visual

Native bar chart of forest out-of-fold MAE by target hour.

### Speaker notes

Hidden backup, for questions only.

Every bar comes from validation folds whose pipeline only saw its own training rows. Errors are larger around the evening hours. This tells us where to look; it does not establish drift, a root cause or a production alert policy. The validation-week forecast plot is in assets/random_forest_forecast.png if someone asks what the one-step sequence looks like.

### Sources (not spoken)

Mean absolute error by target hour over the out-of-fold rows of the selected forest's predictions.csv (outputs/2026-11-13_22-14-53Z_random_forest_v1_8db6c5); values are fixed in the deck.

## Backup D: what the demo shows, and what it does not

**Time:** hidden backup

### On slide

- Demonstrated
- Tracked and nested runs, fair comparison, provenance tags, packaged and reload-verified models, registry versions, alias change and retrieval
- Explained conceptually
- Serving, application reload, delayed labels and monitoring (Presenter 3)
- Not claimed
- Shared tracking server, access control, durable artifact store, a tested fresh serving environment

### Visual

Three rows: demonstrated, explained conceptually, not claimed.

### Speaker notes

Hidden backup, for questions only.

If asked about reproducibility across machines: the lock file and model requirements are recorded and local reload is tested, but a fresh, isolated serving environment was not exercised. A team setup would add a shared tracking server, access control and durable artifact storage. Presenter 3 covers the application-facing lifecycle.

Sources: docs/MLFLOW_DEMO.md and docs/MLFLOW_ENHANCED_RESULTS.md.

### Sources (not spoken)

../MLFLOW_DEMO.md; ../MLFLOW_ENHANCED_RESULTS.md.
