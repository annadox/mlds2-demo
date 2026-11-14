# Shared handoffs and rehearsal

Prepared on Friday, 13 November 2026, for the lecture on Monday, 16 November. The model-consumption command below was rerun successfully on 13 November against the existing local registry. Team discussion and the joint rehearsal still need to happen. Presenter 1 is preparing their slides and Presenter 3 will upload their work separately; this guide specifies the handoff contract, not a completeness assessment of their sections.

## Shared contract

| Item | Agreement supported by the current code |
| --- | --- |
| Task | MT_158 hourly mean electricity load, measured in kW |
| Forecast horizon | One hour after the latest observed hourly reading |
| Evaluation | Three expanding chronological development-CV folds, each with 336 target hours |
| Feature availability | Newly observed history advances before each one-step prediction |
| Seasonal reference | Observed load at the target hour minus 24 hours |
| Input history | At least 168 consecutive observed hours, columns `timestamp` and `load_kw` |
| Model input | 29 engineered numeric features in the saved feature specification |
| Selected reference | `models:/electricity-mt158-lecture@champion` |
| Recorded exact version | `models:/electricity-mt158-lecture/2` |
| Timestamp convention | Published Portuguese wall clock, without claiming UTC |
| Interpretation | Historical local demonstration, with selection-CV evidence |

Presenter 1 owns preparation, features, temporal evaluation and base training. Presenter 2 owns experiment records, comparisons, packaging and registry selection. Presenter 3 consumes the shared feature helper and registered pipeline, and explains application use and monitoring.

## Presenter 1 to Presenter 2

Presenter 1 finishes on the `outputs/` folder with its seven run folders ([Part 1 deck](../presenter_1/README.md), slide 10); Presenter 2 opens on the same folder. There is no need to rerun training on stage.

**Presenter 1 closing:**

> We can reproduce a training execution and evaluate it correctly. When we try more models and settings, how do we compare those executions and find the exact model we want to use?

**Presenter 2 opening:**

> I will keep the same training and evaluation code. We will organize its executions, compare their validation results, and give the selected model a reference that the application can load.

The handoff should take about 20 seconds across the two speakers. Avoid restarting with another introduction to the dataset or another explanation of cross-validation.

## Presenter 2 to Presenter 3

**Presenter 2 closing:**

> We can identify and retrieve a selected model version. How does an application use it, and how do we know it still works?

**Presenter 3 opening:**

> I will load that selected version and provide observed meter history through the shared feature code. Then we can discuss how a batch job or service uses the forecast and what we need to monitor afterward.

Presenter 3 receives the exact model reference, the history example, `feature_spec.json`, and the prediction receipt. Use `features.build_next_features` and `predict.predict_next` rather than defining a second feature implementation.

### Verified consumption command

From `02_reproducible_project` on the current machine:

```bash
uv run --locked --offline --extra tuning python -m electricity_demo.predict \
  --name electricity-mt158-lecture \
  --alias champion \
  --history outputs/2026-11-13_22-14-53Z_random_forest_v1_8db6c5/history.csv
```

Expected identity and result from the current store:

```json
{
  "model_uri": "models:/electricity-mt158-lecture/2",
  "source_run_id": "e90482dae66c49d5a503ab2c20190d0e",
  "prediction_origin": "2014-11-01T00:00:00",
  "target_timestamp": "2014-11-01T01:00:00",
  "predicted_kw": 131.9840938586937
}
```

`--offline` assumes the documented dependencies are already installed. Keep `--extra tuning` on commands while using this environment so uv retains the optional packages. On another machine, use the setup and preparation instructions in the [demo guide](../MLFLOW_DEMO.md). The example history and the registry are generated local files, rather than contents available from a fresh Git checkout.

The recorded verification output is in `outputs/lecture-20261113T221340Z-4f68dd/handoff_prediction_verified.json`. Alias movement can change later resolution, so use a saved receipt in the slides and do not assume a future rerun will always have version number 2.

### Decisions to settle with Presenter 3

1. Choose the concrete consumption example: a small batch-style invocation is already verified. An API example can be explained with code or a diagram if it is not ready.
2. Show the prediction record with resolved version and source run, rather than only a floating-point value.
3. The registered model expects the 29 engineered feature columns, not raw meter history. `mlflow models serve` on this model therefore needs a request with those columns; an endpoint that accepts raw history needs the feature code in front of it or a custom pyfunc wrapper. Slide 9 of Part 2 presents both options.
4. Separate immediate input checks from delayed accuracy measurement. A target observation must arrive before computing forecast error.
5. Explain a response to a monitoring alert, including investigation, validation and a controlled reload or deployment. A distribution change alone need not trigger automatic retraining.
6. Describe feature stores conceptually unless an implementation already exists and materially helps the example.
7. Use the same running example. Parts 1 and 2 use the electricity model `models:/electricity-mt158-lecture@champion`; the Part 3 storyboard draft uses an illustrative churn model (`models:/churn@champion`). Either switch Part 3's examples to the electricity model or say explicitly at the handoff that Part 3 uses a separate illustrative example.

These are discussion points, not a claim that Presenter 3's section or a serving system has been implemented.

## Twenty-minute team alignment meeting

| Minutes | Topic | Concrete result |
| --- | --- | --- |
| 0–5 | Shared task and terminology | Agree on the contract table above |
| 5–10 | Handoffs | Read both transition pairs aloud |
| 10–15 | Presenter 3 consumption | Run the command or inspect the saved receipt together |
| 15–20 | Deck ownership and cuts | Identify repeated content and assign the remaining slide work |

## Whole-lecture rehearsal on Sunday, 15 November

| Full-lecture clock | Owner | Checkpoint |
| --- | --- | --- |
| 0:00–23:00 | Presenter 1 | Finish on training output and the agreed handoff |
| 23:00–48:00 | Presenter 2 | Finish model retrieval and the application-use handoff |
| 48:00–70:00 | Presenter 3 | Finish model use, monitoring and lifecycle response |
| 70:00–75:00 | All | Questions |

For Presenter 1, finish slide 2 by minute 6:30, slide 5 by 12:30, slide 8 by 20 and slide 10 by 23. For Presenter 2, finish slide 4 by minute 7:30, slide 8 by 15:30, slide 11 by 22:30 and slide 12 by 25:00. The notes allocate time for explanation, pointing at real UI elements and short audience prompts. Measure the actual delivery rather than treating a word count as a timing guarantee.

Run the rehearsal with the actual deck, display scale and slide order. One teammate should record the end time of each section. Note terminology mismatches and claims that depend on an unimplemented feature. Rehearse the transitions without a new introduction: the Part 2 opener stays on screen only during the handoff line (hide it if the combined deck has its own divider).

If Presenter 2 exceeds 25 minutes, shorten the optional audience prompts and comparison discussion. Keep the core definitions, actual registry view, evaluation caveat and alias/retrieval handoff. The four backup slides are hidden and stay outside the timed flow.

## Freeze on Sunday evening

The final preparation target is one deck with a PDF copy, captured screenshots, figure sources and an offline walkthrough. Preserve the batch IDs used in the visuals. Test readable text and labels on the actual projector or presentation screen.

On Monday, verify projection, fonts and the slide order. Use prepared evidence if local services or networking fail. Avoid changing model settings or regenerating the experiment just before presenting.

## Likely questions and concise answers

**Does MLflow make the workflow reproducible?**

It records evidence and packages artifacts. Repetition also requires recoverable data, exact code/configuration, a compatible environment and an evaluation protocol. Seeds alone cannot guarantee bitwise equality across numerical backends.

**Why does a time-series validation row use earlier observations from within the validation block?**

This is an advancing one-hour protocol. Each new actual observation becomes available before forecasting the next target. A fixed-origin multi-hour forecast would require a different availability and evaluation design.

**Are the nested runs nested cross-validation?**

The runs organize a study and its candidates. Statistical nested CV adds an outer evaluation loop. Our CV scores helped select settings and therefore remain selection evidence.

**Why not prune trials after the first fold?**

The time windows can differ systematically in difficulty, and fits are small. All candidates complete the same three windows. Intermediate scores are recorded, but the pruner does not stop trials.

**Why do the CV metrics belong to a pipeline refitted on all development data?**

They describe the selected configuration's fold-specific fits. The logged pipeline is the later development refit. Model-linked metrics explicitly use the name `selection_cv_*` and document that distinction.

**Is the last fold evidence of drift?**

It shows higher errors for all three methods in that historical window. Determining drift, its cause or an operational response requires more evidence and an explicit monitoring design.

**Does the signature prevent forecasting leakage?**

It checks required columns and types. Temporal availability depends on the history helper, feature definitions and evaluation protocol.

**Does moving champion update an already running model?**

It changes later alias resolution. An existing model object continues to use its loaded artifact until the application replaces or reloads it.

**Why is there no full feature store, cloud platform or automated retraining?**

The lecture focuses on practical data-science lifecycle decisions. The shared feature contract and identified model demonstrate the necessary handoff without requiring a larger infrastructure tutorial.

**Can I reproduce the exact version number on my laptop?**

Version numbering depends on that registry's history. Regenerate the workflow locally and use its receipt. Do not copy a database containing absolute artifact paths from another machine and assume the artifacts will resolve.

Sources for these answers: [MLflow demo guide](../MLFLOW_DEMO.md), [recorded results](../MLFLOW_ENHANCED_RESULTS.md), shared training and prediction modules.
