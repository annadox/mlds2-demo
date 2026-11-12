# Presenter 2: Experiment tracking and model management

25 minutes, 12 slides. Audience: Data Science master's students. Use prepared screenshots and selected code, with the enhanced `electricity-mt158-lecture` experiment. No live training is necessary. Times are relative to your section, approximately minute 23 of the full lecture.

## Connection to Presenter 1

Presenter 1 finishes with the training command, its output directory, and the chronological evaluation diagram. Their workflow already saves configurations, metrics and provenance; do not claim that it loses all experiment information.

Their handoff:

> We can reproduce a training execution and evaluate it correctly. When we try more models and settings, how do we compare those executions and find the exact model we want to use?

Your opening:

> I will use that same training and evaluation code. We will organize its executions, compare their validation results, and give the selected model a reference that the next application can load.

Keep the shared task fixed: MT_158 hourly electricity demand, one-hour-ahead prediction, 168 hours of observed history, chronological development CV, and a same-hour-yesterday baseline. Refer back to the evaluation diagram instead of teaching CV again.

## Slide sequence

| Slide | Time | Title | Main message and visual |
| --- | --- | --- | --- |
| 1 | 0:00–1:30 | Keeping track of experiments | Presenter 1's output folder: which settings, data and fitted model belong to this result? |
| 2 | 1:30–3:00 | MLflow in the shared workflow | Diagram: training → tracking → identified model → inference. Locate Tracking, Models and Registry. |
| 3 | 3:00–5:00 | Anatomy of an experiment record | Annotated run detail: parameters, metrics, tags, artifacts and dataset inputs. |
| 4 | 5:00–7:00 | Tracking our existing evaluation | Short code excerpt: evaluate the candidate, log fold metrics, return the selection objective. |
| 5 | 7:00–10:00 | Organizing a hyperparameter search | Expanded run tree plus forest trial-progress chart. Parent = study; child = candidate; metric step = fold. |
| 6 | 10:00–13:00 | Comparing models across time | Ridge, forest and baseline comparison plus fold MAE curves. Same target timestamps and protocol. |
| 7 | 13:00–15:00 | Making a selection we can explain | Hyperparameter plot and explicit selection rule. Inspect worst fold and complexity, not only one aggregate score. |
| 8 | 15:00–17:00 | Tracing a result to its inputs | Dataset lineage and compact provenance excerpt: data, code, config and environment. |
| 9 | 17:00–19:00 | The model has an interface | Logged pipeline signature, example and checks. Explain the 29 engineered input features. |
| 10 | 19:00–21:00 | Versions retain the experiment history | Registry screenshot: version → parent → selected trial → data and settings. |
| 11 | 21:00–23:00 | Resolving and changing a model reference | Champion alias, numbered version and saved alias-change illustration. |
| 12 | 23:00–25:00 | Handoff to inference | Historical input → next-hour forecast, with origin, target, units and resolved version. |

## Speaking notes and evidence

### 1–2. Establish the problem and scope

Use three short questions: “Which configuration?”, “Which data?”, “Which fitted model?” The problem is organizing an accumulating set of executions. MLflow supports recording, comparing, packaging and retrieving them; it does not make the evaluation protocol correct.

Mention briefly that other tracking tools and managed ML platforms support similar practices. Git, data preservation and dependency locking remain necessary. Keep the lecture about the lifecycle rather than a catalogue of MLflow features.

### 3. Anatomy of a record

Use actual values from the current batch. A parameter is an effective estimator setting; a metric is development CV MAE in kW; a tag describes purpose or comparison identity; an artifact is a configuration, plot or selection record. Dataset inputs identify the tables and roles used in evaluation.

A run's meaning depends on its role: a study parent, a trial child or the seasonal reference. The experiment groups the common forecasting task.

### 4. Connect directly to Presenter 1's implementation

Show a simplified excerpt clearly labelled “excerpt”:

```python
with mlflow.start_run(run_name=trial_name, nested=True):
    mlflow.log_params(parameters)
    folds, predictions = evaluate_candidate(
        pipeline, development, splits, parameters,
        on_fold=log_fold_metrics,
    )
    # Log the aggregate validation result and return the objective.
```

`evaluate_candidate` clones and fits the preprocessing/model pipeline inside each training fold. The original training command uses the same helper. Actual parameter names and surrounding context are in `tune.py`; avoid presenting this abbreviated snippet as a standalone command.

Manual tracking makes this custom temporal loop explicit. Autologging can reduce logging code, but cannot choose a valid forecasting protocol.

### 5. Explain the hierarchy and Optuna's role

Show the expanded run tree: three Ridge grid trials and ten forest TPE trials. Optuna chooses candidate settings; the shared code evaluates them; MLflow records the evidence. Ridge is an exhaustive small grid. Forest uses four startup trials followed by adaptive TPE suggestions, with a fixed seed and sequential execution. The original forest is included as trial 0.

Use the forest parent’s `trial_objective_mae_kw` and `best_so_far_mae_kw`, with the UI x-axis set to Step. Here steps are trial numbers. A flat best-so-far curve is valid evidence, not a broken search.

Folds are steps in child metric histories. Parent/child nesting is record organization; it is not statistically nested cross-validation. Each trial evaluates all three windows. Pruning is deliberately disabled because earlier chronological windows may be systematically easier, and these fits are small.

### 6. Compare like with like

Select only the two study parents and the baseline from one batch. Show CV MAE/RMSE, then `fold_mae_kw` with Step on x. These fold curves expose temporal variability; they are not epochs or training-loss curves.

All candidates share data, features, evaluation boundaries and comparison identity. These are advancing one-step forecasts using observations available at each origin, not fixed-origin forecasts of an entire validation block. The final saved pipeline is refitted on all development rows; the CV predictions came from fold-specific refits.

### 7. Make the decision interpretable

Show one forest parameter/error scatter or parallel-coordinates view. Ten trials illustrate exploration, not reliable parameter importance or proof of optimality. The selected run must be finished, have a verified model, match the comparison identity, and beat the seasonal reference. The rule minimizes development CV MAE.

Ask: “Would a tiny error improvement justify a much more expensive model?” The cost figure is available in the appendix, but measures local CV fit/predict time, not production serving latency. Fold standard deviation is descriptive variability, not a confidence interval.

Selection CV is optimistic evidence for the searched configuration. This demo does not evaluate the final test while tuning. Historical test inspection by the team means we should not call that test never seen.

### 8. Follow provenance without a wall of hashes

Show development/validation dataset inputs, then a compact manifest excerpt. The dataset contexts describe overlapping roles in rolling evaluation; they are not a fixed disjoint train/validation split. Fold boundaries establish availability.

Follow the chain: run → resolved configuration → dataset identity → code and lockfile → fitted model. Source/lock contents are archived. Hashes and MLflow dataset metadata do not preserve the actual dataset by themselves; data recovery must also be maintained.

### 9. Show what can actually be loaded

The fitted sklearn pipeline includes scaling/encoding and the estimator. Its signature accepts 29 engineered numeric features; it does not accept raw hourly history directly. Shared application code constructs lag/calendar features from 168 consecutive observed hours.

Show `model_checks.json`: five-row prediction agreement after reload and rejection of a missing feature. These are serialization and schema checks, not a certification of production readiness. Temporal correctness is checked by the history helper and forecasting protocol, not by the signature alone.

### 10. Preserve the selection history

Show both registered versions and the source-run links. Follow `selected_trial_run_id` to the winning candidate. Registration gives a logged model a named version; a higher version number does not mean a better model. Different estimator families can share the name because they implement the same task and interface.

Use versions, aliases and tags. Avoid the deprecated fixed Production/Staging stages. Registration does not deploy a service.

### 11. Explain movable versus fixed references

Show `champion`, its resolved numbered version, and the alias events in the evidence receipt. The demonstration temporarily loads the comparison version and restores the selected one; it verifies that the restored reference and prediction agree with the original.

An alias changes what a later lookup resolves to. It cannot replace an already loaded model object. The prediction helper resolves once, loads the numbered version, and records it. Presenter 3 can explain how an application reloads or deploys that model.

### 12. Give Presenter 3 a concrete contract

Input: 168 consecutive observed hours ending at the forecast origin. Output: one next-hour prediction in kW, with origin, target timestamp, source run and numbered model version. Use `prediction` from `outputs/lecture_latest.json` for the actual example.

Your closing:

> We can explain why this configuration was selected, trace its training evidence, and retrieve a specific fitted model. Presenter 3 will show how an application supplies historical observations, obtains predictions, and monitors the model as new data arrive.

Presenter 3 owns batch versus API consumption, serving details, feature-store context and monitoring. The current diagnostics are development evidence, not operational monitoring.

## Screenshot preparation and backup material

Follow [the screenshot guide](MLFLOW_SCREENSHOT_GUIDE.md). Keep six or seven principal screenshots, using crops and progressive reveals where one view supports multiple slides. Preserve the batch/run/version identifiers across captures. Additional graphs are backup material, not extra required slides.

1. Run detail and expanded hierarchy.
2. Trial optimization progress and parameter comparison.
3. Family/baseline comparison and fold curves.
4. Dataset lineage and provenance.
5. Logged model signature and checks.
6. Registry versions, champion and source lineage.
7. Prediction/alias-change receipt as a compact code or table visual.

Appendix: error by hour, observed/predicted scatter, local evaluation cost, validation-week forecast, selection limitations. Generated PNGs are available if UI charts are awkward; label them as analysis figures rather than MLflow screenshots.

## Ownership and references

Presenter 1 owns data preparation, features, evaluation and base training. Presenter 2 owns tracking, comparison, packaging and registration. Presenter 3 consumes the registered model through shared feature construction. Do not duplicate preprocessing in the inference section.

- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [Parent and child runs](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs/)
- [Dataset tracking](https://mlflow.org/docs/latest/ml/dataset/)
- [sklearn model logging](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html)
- [Model Registry workflow](https://mlflow.org/docs/latest/ml/model-registry/workflow/)
