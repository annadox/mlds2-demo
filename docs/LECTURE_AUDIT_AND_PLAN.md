# Lecture repository audit and completion plan

Audit date: 12 November 2026. Lecture: Monday, 16 November 2026.
Audited revision: `0702955`; the working tree was clean before this report.

Implementation follow-up on 12 November: the data are now prepared, the numerical test tolerance is aligned, and tracking, registration and next-hour inference have been implemented and verified. See [the recorded results](DEMO_RESULTS_2026-11-12.md) and [the presenter slide plan](PRESENTER_2_SLIDE_PLAN.md). The audit below describes the earlier state and is retained as a historical assessment.

## Decision

Keep the existing forecasting implementation. It provides a credible foundation for Presenter 1 and a small MLflow extension. No confirmed temporal leakage was found in the packaged workflow. The immediate prerequisites are resolving a failing repeatability check and obtaining the actual data on this checkout. Then prioritize tracking, model packaging, registry retrieval, and one next-hour prediction.

The current user request is for an audit and concrete plan first. The attached context supplies objectives; its invitation to begin implementation does not supersede that request. The newer electricity forecasting context supersedes the older proposal's churn example and preparation schedule. The lecture remains a screenshot/code walkthrough, not a live deployment demonstration. This report introduces no implementation or dependency changes.

## What exists

| Component | Observed implementation | Lecture value |
| --- | --- | --- |
| `01_notebook/electricity_forecasting.ipynb` | Self-contained Ridge/forest workflow; deliberately leaking features, full-data scaling, unseeded random split, mutable CSV | Clearly labelled negative example for Presenter 1 |
| `prepare.py` and `configs/prepare.yaml` | Reads one UCI client from ZIP; hourly aggregation, validation, hashes and preparation manifest | Explicit data definition and reproducibility |
| `features.py`, `split.py`, `pipeline.py` | Causal lags/rolling means; chronological splits; fitted scaler, calendar encoder and estimator | Correct training foundation |
| `train.py`, two model YAML files | Manual candidate/fold loop; development selection; final refit; optional test | Natural entry point for tracked executions |
| `evaluate.py`, `report.py` | MAE/RMSE, seasonal reference, prediction CSVs, plots and local comparison | Reusable MLflow artifacts |
| `snapshots.py`, DVC metadata | Original and explicitly synthetic data versions, matching manifests | Existing optional provenance illustration |
| `notebooks/exploration.ipynb` | Imports package functions and calls the CLI's training function | Positive notebook example without duplicated training logic |
| `tests/` | 21 collected cases including temporal causality and model round trips | Useful existing correctness checks |
| MLflow, registry, inference CLI | Absent; MLflow is neither declared nor installed | Main remaining integration work |
| Slides, diagrams, paper | No slide deck, diagram assets or paper found in this repository | Lecture materials still need preparation; do not infer a paper architecture |

### Exact forecasting contract

- Client `MT_158`, univariate demand, 2014, 8,760 configured hourly intervals. This is not forecasting all 370 clients.
- Target: mean kW for the hour ending at timestamp `t`, forecast after observing `t-1`.
- Inputs: lags 1–24 and 168, shifted means over 24 and 168 hours, target hour and weekday: **29 feature columns** and **168 required historical readings**.
- A 168-hour warm-up leaves 7,128 development and 1,464 test targets.
- Development ends `2014-11-01 00:00`; test runs `2014-11-01 01:00` through `2015-01-01 00:00` inclusive.
- Three expanding folds, each with 336 validation hours; gap zero. The model is fixed within each validation block while actual observed history advances hourly.
- Ridge searches alpha `[0.1, 1.0, 10.0]`. Forest uses 40 trees, depth 10, minimum leaf size 3, seed 42 and one estimator job.
- Both use the same target timestamps and features; the seasonal reference is `y_(t-24)`.
- The selected estimator is refitted on all development targets; `--final-test` explicitly enables test scoring.

This is a sequence of one-hour forecasts, not a 336-hour or two-month forecast made from a single origin. Earlier validation/test observations entering later features are legitimate under the documented immediate-observation assumption. A nonzero gap is not automatically necessary just because lag windows overlap.

### Correctness that should be preserved

`features.py:15` shifts all lags; line 19 shifts before rolling. Computing those causal features before splitting does not itself introduce future information. `train.py:77` clones and fits a complete pipeline for every candidate/fold. `pipeline.py:13` puts learned scaling and calendar transformation inside that pipeline. The persisted object at `train.py:141` contains fitted preprocessing, not just the estimator.

`train.py:115` correctly labels CV as selection CV rather than an unbiased assessment of the entire hyperparameter search. The final test is excluded from fitting and is not scored by default. MAE/RMSE and baseline comparisons use identical rows and appropriate kW units.

The hourly mean and documented treatment of source zeros/DST are consistent with retaining the published source clock. UCI documents kW readings, pre-activation zeros, March zero intervals and October combined intervals. This supports the existing caution that these timestamps are not a physical UTC clock. The selected client's actual zero pattern remains unverified here because the data are absent. [Official UCI documentation](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014).

## Findings and priorities

### C1 — Critical verification gate: Ridge repeatability test fails locally

**Evidence:** `tests/test_training.py:28`; `experiments/README.md` documents a different tolerance. The completed default suite produced **20 passed, 1 failed in 16.34 seconds**. Repeated Ridge predictions differed by at most `6.209999980910652e-08` kW; 146/344 values exceeded the test's `rtol=atol=1e-10` threshold. Both runs selected alpha 0.1.

The same saved predictions **pass the documented `rtol=atol=1e-9` threshold**. The affected test also passed in 4.22 seconds with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`. This suggests sensitivity to numerical execution, but does not isolate a specific numerical library as the cause.

**Impact:** the current claim of an entirely passing suite cannot be reproduced under default settings. The observed difference is not evidence of materially different forecasting quality or leakage.

**Small fix:** agree on and use the documented numerical tolerance consistently; retain separate checks for selected parameters, metrics and loaded-model prediction agreement. If thread limits are part of the reproducibility contract, record them. Re-run the affected check and full suite; append a dated verification entry rather than rewriting historical results. **Before lecture: yes.**

### C2 — Critical execution prerequisite: actual dataset is unavailable in this checkout

**Evidence:** `data/raw/` and `data/prepared/` contain pointers only; `.dvc/config` has no remote and the checkout has no `.dvc/cache`. Calling `load_hourly` produces the documented `FileNotFoundError` directing the user to preparation or DVC pull. `experiments/verification.md:35` describes cached data on a prior machine, not this checkout.

**Impact:** real UCI training, current scores, DVC round trips, and authentic screenshots cannot yet be independently reproduced here. This is a setup prerequisite, not a defect in the download fallback.

**Small fix:** run the existing preparation command once, verify the resulting hash against snapshot v1, and train both existing configurations without test scoring. Alternatively obtain data through a teammate's durable DVC store. Do not require implementing a shared remote or generating synthetic v2 to unlock MLflow. **Before lecture: yes for original project screenshots.**

### I1 — Important: no MLflow implementation

**Evidence:** `pyproject.toml:10`, `train.py:42`; no MLflow imports, declared dependency, tracking store, model package or registry code exists.

**Impact:** local JSON/CSV artifacts support reproducibility but cannot demonstrate your Tracking UI, logged model, registry version or alias narrative.

**Small fix:** add an optional tracking dependency and a small tracked-training entry point that calls `run_training`. Preserve the current untracked command and training logic. Details below. **Before lecture: yes, highest implementation priority after prerequisites.**

### I2 — Important: the saved pipeline is not a history-to-forecast interface

**Evidence:** `features.py:10` validates a frame with observed `load_kw`, attaches a target, then drops incomplete rows. `train.py:143` saves 168 history hours, feature definitions and column order, but there is no inference entry point. The README reload example assumes the caller already has `feature_frame`.

**Impact:** the model includes scaling/encoding, but lag construction remains outside it. Calling the training feature builder on exactly 168 historical rows produces no forecast row. Appending a missing future target also violates current validation. Presenter 3 cannot obtain the next forecast simply by loading the pickle.

**Small fix:** add a narrow `build_next_features(history, feature_spec)` function, require consecutive observed history through `t-1`, produce one row for `t` without requiring `y_t`, and test agreement with the training builder on a known historical target. Keep the sklearn pipeline's interface as engineered features; document this explicitly. **Before lecture: yes for the connected inference example.**

### I3 — Important comparison limitation: the local report hides same-family experiments

**Evidence:** `report.py:114` keys results only by model type. A synthetic audit probe with Ridge scores 1 and 9 retained only the later score 9. The final-test preference at line 116 compares base `model_config`, not the search grid or selected `best_params`.

**Impact:** `comparison.md` is a latest-per-family summary, not an experiment ledger. It can hide alternative hyperparameters and, with changed search settings, retain an earlier test run as if the effective settings were identical.

**Small fix:** use the full MLflow run table for the lecture's candidate comparison and carry over the existing `comparison_group` tag. Do not expand this report unless it is needed on slides. If retained for selection, key by effective configuration/search identity and show the selected parameters. **Before lecture: avoid relying on it for multi-configuration selection; code fix can wait.**

### I4 — Important evidence limitation: final-test results are already known

**Evidence:** `experiments/verification.md:23` records earlier test scores for both models. This does not establish that anyone tuned on those scores, and the current code does not use them for selection.

**Impact:** do not claim that the team has never inspected the final test. Future run comparison must remain development-only; do not manufacture a new holdout merely to retell the history.

**Small fix:** describe the period as reserved from fitting and tuning under the intended protocol; identify previous results as historical, freeze the chosen configuration before any new test evaluation, and record the selection rationale. **Before lecture: yes, wording and workflow.**

### I5 — Important: screenshot and presentation evidence is missing

**Evidence:** no committed slide deck, screenshots, serving request/response or model handoff example was found. The existing verification document is historical evidence, not a replacement for artifacts available on this machine.

**Small fix:** capture four readable MLflow views and one inference result after a minimal verified run. Save a compact run manifest listing source run/model/version identifiers so screenshots tell one consistent story. **Before lecture: yes.**

### Improvements, not blockers

- `workflow.md:5` still asks for PyTorch although neither notebook imports it and the package no longer depends on it. Remove that instruction. Include `cd` and `.venv/bin/python` equivalents beside Windows-specific setup examples. Small documentation fix before the lecture.
- `workflow.md:20` makes DVC snapshot creation look mandatory for training. Separate the shortest setup from the optional v1/v2/v1 illustration. Preserve the implementation; reduce lecture time spent on it.
- The bad notebook's centered rolling mean includes current/future load (`electricity_forecasting.ipynb:139`); scaling precedes splitting (line 168); the unseeded shuffled split is at line 169; the data mutation at line 378 compounds if rerun. These are confirmed **intentional counterexamples**, not defects to silently repair. Annotate screenshot excerpts and never compare their scores directly with the corrected protocol.
- Existing hashes identify source/data/environment contents but do not themselves preserve dirty source files or remote data. For final evidence, retain a clean committed source revision, lockfile and recoverable data. Avoid introducing a new provenance framework.

### Optional: explicitly defer

Nested fold runs, more models, multi-step forecasting, custom MLflow pyfunc history wrappers, online server implementation, Evidently, Feast installation, containers, orchestration, cloud deployment and CI/CD expansion. Explain feature stores and monitoring with diagrams; retain DVC as a brief existing illustration if time permits.

## Smallest end-to-end architecture

```text
Intentional notebook mistakes
    -> prepare + features + split + sklearn Pipeline
    -> run_training: temporal CV, selection, fitted artifact
    -> tracked-training wrapper: one execution = one MLflow run
    -> validation-based candidate selection
    -> logged model -> registered version -> champion alias
    -> 168 observed hours -> next-hour features -> loaded model -> kW prediction
    -> conceptual data checks / delayed-error monitoring / investigation
```

Reuse all current training modules. Proposed additions, not yet implemented:

| Addition | Responsibility | Completion check |
| --- | --- | --- |
| `train_tracked.py` | Start tracking context; call existing training function; log saved results and fitted pipeline | One completed run; failures do not appear successful; untracked CLI still works |
| `tracking.py` if needed | Small shared logging helpers, tracking URI and artifact handling | Logged metrics match JSON; model round trip preserves predictions |
| `register.py` | Register an explicitly selected logged model and set `champion` | Named version, source lineage and alias retrieval verified |
| Addition to `features.py` | Build the next row from past observations alone | Matches training features; rejects inadequate/gapped history |
| `predict.py` | Resolve alias to a version, load model, forecast and write output | No fitting; no future target supplied; output records version and origin |
| `docs/` plus selected screenshot assets | Commands, input contract, evidence and presenter notes | One coherent screenshot narrative and offline slide backup |

Use a local SQLite-backed MLflow store and local artifact directory with project-anchored paths. Exclude generated databases/artifacts from Git. Pin a tested MLflow release in `uv.lock`; there is currently no installed release whose API compatibility can be asserted. Official documentation requires a database-backed store for the self-hosted registry. Use aliases and versions rather than deprecated stages. [MLflow registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/).

### Tracking contract

Use manual tracking for this custom CV loop. Avoid having autologging create confusing fit-level records or treating training scores as validation scores. One existing Ridge search can be one run: log its candidate table as an artifact and its selected alpha separately. Nested runs are unnecessary.

- **Parameters:** model family and effective fitted parameters; search space; seed; horizon/delay; lag and rolling definitions; CV size/count/gap; training/test boundaries and customer.
- **Metrics:** development MAE, pooled RMSE, fold MAE standard deviation, baseline MAE/RMSE and improvement; per-fold metrics with unambiguous names or steps. Keep mean-fold RMSE distinct from pooled RMSE. Test metrics only on explicit final evaluation.
- **Tags:** data SHA, snapshot, source revision/dirty status, comparison group, timestamp policy, execution identity and selection/evaluation purpose.
- **Artifacts:** existing config, manifest, metrics, candidate table, predictions, plot and feature specification; lockfile and a compact input example. Preserve history with its cutoff and role clearly stated.
- **Model:** log the fitted sklearn pipeline with signature and a genuine development input example. Include compatible serving dependencies; retain the returned model URI/ID rather than assuming a storage layout. Log effective parameters from the fitted model, not only the YAML's initial alpha.

MLflow distinguishes runs, logged models and registered model versions; model logging provides native sklearn and generic prediction interfaces. The model signature describes the engineered feature frame, not raw meter history. [Tracking concepts](https://mlflow.org/docs/latest/ml/tracking/), [sklearn flavor API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html).

### Presenter 3 handoff

Registered model proposal: `electricity-mt158-hourly`; alias `champion`. Provide both `models:/electricity-mt158-hourly@champion` and its resolved numbered version. Include the source run/model identifier, 29-column input schema, 168-hour history requirement, source-clock convention and output units.

The minimal demonstration forecasts the first hour after the saved development history. It consumes only observed data through `2014-11-01 00:00` and predicts `2014-11-01 01:00`. Actual future load is used only afterward for evaluation. A batch of retrospective one-step forecasts must be labelled as advancing observed history, not as a future-block forecast from one origin.

Registering or aliasing does not deploy a service. Resolving the alias once and recording the numbered version makes the shown prediction traceable even if the alias later changes. A model signature checks the interface; separate history checks establish temporal validity.

## Ownership and deadline plan

| When / priority | Owner | Bounded task and dependency | Required evidence |
| --- | --- | --- | --- |
| Thu 12 Nov — P0 | Presenter 1, coordinated with Presenter 2 | Align reproducibility checks; prepare actual v1 data; run existing configs | Passing suite under stated settings, verified data hash and CV artifacts |
| Fri 13 Nov — P0 | Presenter 2 | Add locked optional MLflow integration; wrap existing training | Metrics and artifacts in UI; logged pipeline reload matches local pipeline |
| Fri 13 Nov — P0 | Presenter 2 | Track Ridge search and forest; add one forest variant only after the two-run path works | Comparable development-only run table and candidate-selection rationale |
| Sat 14 Nov — P1 | Presenter 2 | Register selected candidate; demonstrate versions and `champion` | Version/alias/source lineage and equivalent loaded predictions |
| Sat 14 Nov — P1 | Presenter 3, shared feature review by Presenter 1 | Next-hour feature helper and minimal prediction CLI using the registered model | History-only request and traceable next-hour forecast |
| Sat 14 Nov — P0 | All; Presenter 2 owns MLflow evidence | Capture screenshots and complete slides using the same project identities | Readable run comparison, run/artifacts, signature, registry and forecast |
| Sun 15 Nov — P0 | All | Freeze examples, verify links/claims, rehearse 23 + 25 + 22 + 5 minutes | Unified deck and local PDF backup; no dependence on live services |
| Mon 16 Nov | All | Present the frozen material | No new architecture or exploratory model tuning |

Dependencies: setup/correctness -> tracked runs -> selection/registry -> inference -> final evidence capture. Slide writing can proceed alongside setup using verified concepts and explicit placeholders. Presenter 1 owns preparation/features/splits; Presenter 2 owns tracking/registration; Presenter 3 consumes the model without copying training or refitting preprocessing. Review only shared feature-interface edits together.

For your 25-minute section, use 11 slides: motivation (2 min), lifecycle/tool scope (2), experiment/run/metadata (2), short tracking code and manual-vs-auto choice (3), run comparison (3), validation-based selection (2), provenance/artifacts (2), model signature/dependencies (2), registry/versions (3), alias/retrieval (2), inference handoff/limitations (2). Capture original UI screenshots only after the metrics and identifiers have been checked.

If behind on Saturday, cut the third configuration, second registered version, DVC sensitivity walkthrough and online serving first. Keep two meaningful runs, one logged and retrievable model, one registered version/alias if feasible, one prediction and the monitoring explanation. Attributed official screenshots remain a fallback; do not mix them with local results as if they came from the same execution.

## Immediate next implementation step

First make the verification contract explicit and prepare data. The first MLflow change should then be a single optional tracked-training entry point wrapping `run_training`, not a rewrite of its CV loop. Complete its vertical path: **train one Ridge execution -> log its existing artifacts and fitted pipeline -> reload through MLflow -> verify equal predictions**. This supports the tracking-code, run-detail and model-artifact slides and establishes the interface for later registration.

Acceptance checks should exercise real behavior: unchanged untracked training, correct logged CV metrics, no default test metrics, MLflow model round trip, alias/version retrieval, and history-only feature equivalence. Use synthetic fixtures for fast integration checks and the actual UCI subset for lecture screenshots.

## Audit validation and limits

- Read both supplied documents, both notebooks, all package modules, configurations, tests, DVC metadata and repository documentation.
- `uv lock --check --offline` passed; `uv run --locked --offline` used the locked environment. Installed Python 3.12.1, NumPy 2.3.5, pandas 2.3.3, scikit-learn 1.9.1, DVC 3.67.1 and pytest 9.2.0; MLflow absent.
- The first test attempt was interrupted during slow native-module imports; it produced no test result. A subsequent complete default suite returned the 20/1 result above. The numerical comparison and isolated thread-limited retry are separate follow-up checks, not a claim that the default full suite passed.
- Verified missing-data failure directly. No large dataset download or actual UCI training was performed during this audit. Historical CV/test values in `verification.md` were not independently reproduced.
- Existing synthetic tests exercise model persistence and final-test opt-in; DVC create/restore, a fresh notebook execution, MLflow and serving were not verified end to end here.
- No implementation, configuration, lockfile, notebook, data pointer, Git history or remote changes were made. This report is the only new tracked-source candidate; normal ignored environment/test caches were generated.
