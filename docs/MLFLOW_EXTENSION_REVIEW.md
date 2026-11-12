# Review of the MLflow demo against the lecture goals

Reviewed and extended on 12 November 2026. The objective is a coherent 25-minute section about managing experiments and handing an identified model to inference. Screenshots should support a decision or explain a failure mode.

## Findings and implementation decisions

| Existing limitation or proposed addition | Relation to the lecture | Decision |
| --- | --- | --- |
| Fold 1/2/3 were separate scalar metric names | Makes temporal variability hard to see in the UI | Add `fold_mae_kw`, `fold_rmse_kw` and baseline histories with the fold number as `step` |
| Hyperparameter candidates were only in a CSV | Prevents direct UI comparison of candidate configurations | Add a study parent and trial child runs for each candidate |
| Optuna was absent | Useful example of automatically choosing experiments | Add a small seeded sequential TPE search, while preserving Ridge's exhaustive three-value grid |
| Forest used only its original fixed settings | Too few points to discuss parameter effects | Search depth, leaf size and feature fraction, including the original forest as trial 0 |
| Baseline only appeared as a metric inside model runs | Makes baseline comparison less visible | Add a first-class baseline run on identical forecast targets |
| Data lineage was confined to hashes in tags/manifests | Leaves MLflow's dataset view empty | Log development and validation dataset metadata with source, digest, schema and target |
| Only a forecast-week plot existed | Hides variation by fold and target hour | Add comparison, fold, optimization, parameter and diagnostic figures from observed predictions |
| A human reason was the only selection record | Does not demonstrate checks needed before handing off a model | Record an explicit decision and reject unfinished, incompatible or baseline-inferior candidate sets |
| Model signatures were visible but their effect was implicit | Students may think the signature validates temporal correctness | Record a missing-feature rejection plus model reload agreement, with limits stated |
| Alias assignment was static | Does not show version changes and retrieval semantics | Add an optional local rollback illustration with old/new version records and restored prediction verification |

These additions expose the existing evaluation clearly and strengthen the story from experiment to identified model. The base training command remains available. One small extraction, `evaluate_candidate`, lets the original grid and Optuna use the same fold fitting/evaluation code. Model packaging is shared through `log_training_output`.

## Intended hierarchy

```text
Experiment: electricity-mt158-lecture
  seasonal-naive-reference
  ridge-grid-study
    ridge-trial-00
    ridge-trial-01
    ridge-trial-02
  random_forest-tpe-study
    random_forest-trial-00
    ...
    random_forest-trial-09
```

Only each family's selected configuration is refitted and packaged on its parent run. Children record candidate parameters, CV metrics, fold boundaries and dataset inputs. The parent links to its winning child, and its registered version retains that trial reference.

The selected configuration passes through the original training command's CV/refit/artifact path once more. Its repeated CV score must agree with the selected trial. This small extra evaluation prevents the tuning implementation from silently disagreeing with the existing training workflow.

Full-data verification exposed micro-kW variation between independent Ridge LSQR fits on this machine (roughly 0.000012 kW across repeated checks). The score-agreement check therefore uses `rtol=1e-5, atol=1e-6` and records both scores, their difference and the tolerance in `selection.json`. This checks numerical agreement rather than bitwise identity; it does not change the trial ranking or hide either measurement. Model serialization still uses the stricter prediction round-trip check. A failed preliminary study is retained and excluded by the final batch filter.

## What each graph means

- **Fold curves:** x = chronological fold number, y = MAE/RMSE in kW. These are evaluations on different time windows, not epochs or a learning curve.
- **Trial progress:** x = trial number, y = objective and best score observed so far. Ten trials illustrate search behavior, not convergence to a global optimum.
- **Parameter plots / parallel coordinates:** observed parameter combinations versus CV error. Correlation among a small set of trials is not a causal attribution or robust importance estimate.
- **Model comparison:** each family’s selected configuration plus the seasonal reference, on identical validation targets. Selection CV remains optimistically selected evidence.
- **Evaluation cost:** CV fit/predict wall time on this local machine. It is not a production latency benchmark and is not used as the selection objective.
- **Error by hour:** absolute errors from out-of-fold predictions, aggregated by target hour. It can motivate investigation; it does not establish drift or its cause.

The standard deviation across three temporal folds is descriptive variability, not a confidence interval. The development and validation dataset inputs refer to the same rolling evaluation scheme: later development rows can be validation targets in one fold and training observations in a later fold. Their contexts do not imply disjoint fixed train/validation partitions.

## Deliberately deferred

**Fold grandchild runs:** individual fold curves and boundary artifacts provide the required evidence without adding another level of navigation. Add grandchildren only if fold-specific fitted models or large artifacts need separate management.

**Pruning:** each candidate is evaluated on all three chronological windows. Early folds may be systematically easier than later folds, and these small fits do not justify introducing early stopping. Optuna uses an explicit `NopPruner`; intermediate fold scores are recorded without pruning.

**Nested statistical CV:** MLflow nesting organizes records and does not implement an outer evaluation loop. The final test remains reserved under the intended protocol, and historical inspection is disclosed. We do not claim unbiased evaluation of the search from its selected CV score.

**Resume/distributed optimization:** the Optuna study is persisted in SQLite for inspection, but each command starts a new study. We do not claim reproducible distributed scheduling or exact sampler-state resume. Sequential execution and recorded seeds reduce variability.

**Feature attribution and automatic parameter importance:** few search trials do not justify confident importance claims. Use the observable parameter/error plots and prediction diagnostics.

**A full serving platform or monitoring suite:** belongs principally to Presenter 3, and would distract from your section. The model reference, schema check and history-only prediction are the concrete handoff. The diagnostics are development evidence, not operational monitoring.

## Production-oriented practices versus production readiness

The demo demonstrates data/code/environment identities, correct temporal evaluation, baseline comparisons, run organization, model serialization checks, schema checks, model lineage, explicit selection and version retrieval. A movable alias is not a deployment. The checks do not certify business value, security, service uptime or production monitoring.

Source hashes identify files and the runs archive their contents. MLflow dataset metadata identifies subsets and sources but does not store a complete recoverable dataset by itself. Retain the original prepared data/DVC recovery path.

The final model's linked metrics are named `selection_cv_*`: CV predictions were produced by fold-specific refits, whereas the logged pipeline is fitted on all development rows. Avoid describing the final fitted object as having directly predicted those out-of-fold rows.

## References

- [MLflow parent and child runs](https://mlflow.org/docs/latest/ml/traditional-ml/tutorials/hyperparameter-tuning/part1-child-runs/)
- [MLflow dataset tracking](https://mlflow.org/docs/latest/ml/dataset/)
- [Optuna study creation and persistence](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html)
- [Optuna TPE sampler and startup trials](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)

The implementation uses explicit MLflow calls inside the objective. It does not depend on the deprecated Optuna MLflow callback integration.
