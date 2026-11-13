# Recorded MLflow evidence

The observed results behind Presenter 2's slides. The full lecture workflow ran on 12 November 2026 (Python 3.12.1, MLflow 3.18.0, real UCI data); its stored records were verified through the local MLflow HTTP API on 13 November. How to regenerate and read this evidence is in [MLFLOW_DEMO.md](MLFLOW_DEMO.md).

## Exact batch and selection

- Experiment and registered-model name: `electricity-mt158-lecture` (local experiment ID `2`).
- Batch: `20261112T181949Z-424799`.
- Receipt: `02_reproducible_project/outputs/lecture_latest.json`.
- Evidence directory: `02_reproducible_project/outputs/lecture-20261112T181949Z-424799`.
- Prepared MT_158 data SHA-256: `06e155b416659aa93e510a6d233a637baee5f65970830c5715ce2f796b2733ea` (snapshot v1).
- Comparison group: `3e2ed10133c2b0791a5440187f8104176d3547f4420ee863c45d2391bd19cacf`.

The batch contains **16 FINISHED runs**: one seasonal reference, two study parents and thirteen trial children. Ridge evaluates all three alpha values; forest evaluates ten sequential Optuna TPE suggestions, including the original depth-10 forest as trial 0. All candidates use the same three expanding chronological CV windows, each with 336 validation targets. No final-test metrics were logged.

| Selected family/reference | Effective configuration | CV MAE (kW) | CV RMSE (kW) | Version |
| --- | --- | ---: | ---: | --- |
| Same hour yesterday | Observed load at target minus 24 hours | 30.123865 | 44.605909 | No registered estimator |
| Ridge | Alpha 10.0, selected from 0.1 / 1 / 10 | 9.662940 | 13.501337 | 1 |
| Random forest | 40 trees, depth 15, minimum leaf size 7, feature fraction 0.6 | **8.390400** | **13.146258** | **2, champion** |

The selected forest improves on its original configuration's CV MAE of 8.427510 kW by about 0.44%, and beats the seasonal reference by about 72.1%. Most of the gain came from a reasonable baseline forest; the search also exposed substantially worse configurations. These are selection-CV estimates, not an unbiased assessment of the whole search.

## Fold behavior

| Chronological fold | Ridge MAE (kW) | Forest MAE (kW) | Seasonal reference MAE (kW) |
| --- | ---: | ---: | ---: |
| 1 | 8.393101 | 6.786903 | 27.058495 |
| 2 | 8.690168 | 7.950856 | 29.960440 |
| 3 | 11.905551 | 10.433439 | 33.352661 |

All three methods have larger errors in the last window. This motivates inspecting temporal variation; it does not establish drift or its cause.

## Lineage

| Record | Run ID | Selected child |
| --- | --- | --- |
| Seasonal reference | `512fec84a4294ab192b5aa879ae867e6` | — |
| Ridge study parent | `9b8e8f01b6514f158e04313528445dcd` | Trial 2: `b6075f92e9464412b818e91553ec5f36` |
| Forest study parent | `7d5d3f2547714e2dabf8333ad7123157` | Trial 9: `d3d94f35e3e04aa886e9f8fe2c533cd2` |

- Ridge logged model `m-1fab52319f9c4282bdf863d9c4bb1bf5`, output `outputs/2026-11-12_18-19-54Z_ridge_v1_e7f336`.
- Forest logged model `m-466a99dd9e524ab4a7a2911b9bc1099c`, output `outputs/2026-11-12_18-21-13Z_random_forest_v1_cf4f24`.

Filter: `tags.lecture_batch = '20261112T181949Z-424799' and attributes.status = 'FINISHED'`. The archived source discloses a dirty workspace at execution time; later commits do not change that provenance.

## Model retrieval and alias illustration

`champion` resolved to `models:/electricity-mt158-lecture/2`. Given 168 observed hours ending `2014-11-01T00:00:00`, it predicted **131.984094 kW** for `2014-11-01T01:00:00` (published Portuguese wall clock). The rollback illustration moved `champion` from version 2 to 1 and back; version 1 predicted 124.623627 kW for the same history, and restoring version 2 restored its original prediction. Events are in `evidence.json` and the runs' `registry_events/` artifacts. The consumption command and full receipt are in [the team handoff](presenter_2/TEAM_HANDOFF_AND_REHEARSAL.md).

## Figures and screenshots

- `presentation_assets/` in the evidence directory: 11 PNGs generated from real results (model comparison, folds, forest search and parameters, diagnostics, forecasts, evaluation cost) plus receipts and tables. Portable copies of the main ones are in `docs/presenter_2/assets/`.
- Native MLflow UI screenshots, captured 13 November: `docs/presenter_2/assets/mlflow_screenshots/` (also in `docs/presenter_2/mlflow-captures.zip`).

Generated outputs, models, data and SQLite stores are local and ignored by Git.

## Verification and limits

- The suite passed 29 tests after the batch and **32 tests** after later maintenance (whole-hour timestamp validation, comparison by effective selected hyperparameters). Those changes, and the 13 November refactor (single feature definition, in-memory training result, inference decoupled from training, split dependencies, CI), give new executions a different source/comparison identity: do not mix newly trained runs into the recorded batch. After the refactor the real-data forest still scores CV MAE 8.427510 kW and the champion still predicts 131.984094 kW from the saved history.
- The MLflow HTTP API returned all 16 FINISHED runs, three fold points per run, two dataset inputs per run, parent progress histories of lengths 3 and 10, no test metrics, correct registry-to-trial links and champion version 2 (`verification.json`).
- Independent Ridge LSQR refits differed by 0.000006282 kW in mean-fold CV MAE, within the recorded tolerance; the selected alpha did not change.
- MLflow emitted an upstream SQLAlchemy deprecation and duplicate dataset-source warnings; none were suppressed.

## Earlier introductory experiment

`electricity-mt158-hourly` (local experiment ID `1`) holds the first two-run version of the workflow. It is not used in the slides. Final runs are tagged `lecture_demo=true` (comparison group `902cf6baded85dd7e4b9c9437db105090b5a87d1266f07cac8d59f03b72c744b`); an earlier failed forest logging attempt also remains visible. Records: `outputs/lecture_runs.json`.

| Execution | Run ID | Configuration | CV MAE (kW) | CV RMSE (kW) | Version |
| --- | --- | --- | ---: | ---: | --- |
| `lecture-ridge-search` | `c9bb5f57b00f4a2c9e683023b3dad95f` | Alpha 10.0 | 9.662940 | 13.501337 | 1 |
| `lecture-forest-depth-10` | `f73b3e5a2dd14b7d92a91c989fc1dd91` | 40 trees, depth 10, leaf 3 | 8.427510 | 13.303743 | 2, champion |

Its champion predicted 132.628849 kW for the same target hour (`outputs/next_hour_forecast.json`).
