# Validation results — 11 November 2026

## Current two-model workflow

Verified after simplifying the demo to Ridge regression and random forest, using the locked Python 3.12.1 environment.

- All **21 automated tests passed** in 11.60 seconds. Checks cover causal features, chronological splits, train-only preprocessing, repeatability and saved-pipeline predictions for both supported models, final-test opt-in and compatible-run reporting.
- `uv lock --offline` and `uv sync --locked --offline` succeeded. The project dependency declaration and lockfile no longer require PyTorch or its numerical/GPU dependencies.
- Both models were trained again on the actual v1 UCI data using the same explicit lag features and chronological development folds. Final-test scores were not computed during this verification.
- Both configurations explicitly list the previous 24 hourly readings and lag 168. This preserves the feature values previously generated through the sequence setting.
- Customer MT_158 has 8,760 hourly observations. The 168-hour warm-up leaves 7,128 development targets and 1,464 reserved test targets.

| Model | Current development CV MAE (kW) |
|---|---:|
| Random forest | 8.428 |
| Ridge regression | 9.663 |
| Same-hour-yesterday baseline | 30.124 |

The current development scores match the previously recorded scores. `outputs/validation.json` records the two new run directories, and `outputs/comparison.md` displays the active two-model comparison.

The self-contained notebook was also executed with the cached local CSV, including both training cells, comparison/scatter plots and reloading the saved random-forest bundle. Reloaded predictions matched the original forest predictions. Its intentionally mutable-data cell was skipped to preserve the local dataset. Saved notebook outputs were cleared so students cannot see stale results from removed model types. The notebook retains its deliberate leakage and unseeded split for the lecture, so its scores are not directly comparable to the packaged workflow.

## Previously recorded final-test results

These results come from the earlier verification with the same feature values and evaluation policy. They were **not rerun** during the two-model simplification. Use development CV for model selection.

| Model | Previously recorded test MAE (kW) | Previously recorded test RMSE (kW) |
|---|---:|---:|
| Random forest | 8.420 | 13.399 |
| Ridge regression | 9.648 | 13.416 |
| Same-hour-yesterday baseline | 32.460 | 47.658 |

## Data recovery and historical artifacts

The earlier v1/v2/v1 DVC restore verification reproduced restored-v1 predictions. The source/prepared data and snapshot pointers were not changed by the two-model simplification. Prepared v1 SHA-256 is `06e155b416659aa93e510a6d233a637baee5f65970830c5715ce2f796b2733ea`; synthetic v2 SHA-256 is `a9171f44ae9891b5145bb60fa30f5954bd7cc1fe621052f4519f185f30fa1b3a`. Both snapshots remain cached; v1 remains active.

Historical generated artifacts for retired model types were moved into `01_notebook/archive/` and `outputs/archive/` rather than deleted. Active reports include only Ridge and random forest. Historical run records stay unchanged; they are excluded from current comparisons.
