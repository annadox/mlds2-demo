# Recorded MLflow evidence

The observed results behind both lecture decks. The full lecture workflow ran on 13 November 2026 on committed code (`64ec6b2`, `source_dirty: false`) with Python 3.12.1, MLflow 3.18.0 and the real UCI data, into a fresh MLflow store. How to regenerate and read this evidence is in [MLFLOW_DEMO.md](MLFLOW_DEMO.md). The earlier 12 November store and run folders were archived (not deleted) under `02_reproducible_project/.cache/archive-2026-11-12/`, which Git ignores.

## Exact batch and selection

- Experiment and registered-model name: `electricity-mt158-lecture` (local experiment ID `1`).
- Batch: `20261113T221340Z-4f68dd`.
- Receipt: `02_reproducible_project/outputs/lecture_latest.json`.
- Evidence directory: `02_reproducible_project/outputs/lecture-20261113T221340Z-4f68dd`.
- Prepared MT_158 data SHA-256: `06e155b416659aa93e510a6d233a637baee5f65970830c5715ce2f796b2733ea` (snapshot v1).
- Comparison group: `0b86a7bd8ec082d8ba2209b70385b0c373b1facbb62cb47228d382d7396c8170`.

The batch contains **16 FINISHED runs**: one seasonal reference, two study parents and thirteen trial children. Ridge evaluates all three alpha values; the forest evaluates ten sequential Optuna TPE suggestions, including the original depth-10 forest as trial 0. All candidates use the same three expanding chronological CV windows of 336 validation targets. Each parent reuses its winning trial's CV results and only refits on the development period. No final-test metrics were logged.

| Selected family/reference | Effective configuration | CV MAE (kW) | CV RMSE (kW) | Version |
| --- | --- | ---: | ---: | --- |
| Same hour yesterday | Observed load at target minus 24 hours | 30.123865 | 44.605909 | No registered estimator |
| Ridge | Alpha 10.0, selected from 0.1 / 1 / 10 | 9.662932 | 13.501329 | 1 |
| Random forest | 40 trees, depth 15, minimum leaf size 7, feature fraction 0.6 | **8.390400** | **13.146258** | **2, champion** |

The selected forest improves on its original configuration's CV MAE of 8.427510 kW by about 0.44% and beats the seasonal reference by about 72.1%. These are selection-CV estimates, not an unbiased assessment of the whole search. The scores match the 12 November batch to the reported precision; Ridge differs by micro-kW because of its LSQR solver.

## Fold behavior

| Chronological fold | Ridge MAE (kW) | Forest MAE (kW) | Seasonal reference MAE (kW) |
| --- | ---: | ---: | ---: |
| 1 | 8.393101 | 6.786903 | 27.058495 |
| 2 | 8.690177 | 7.950856 | 29.960440 |
| 3 | 11.905519 | 10.433439 | 33.352661 |

All three methods have larger errors in the last window. This motivates inspecting temporal variation; it does not establish drift or its cause.

## Lineage

| Record | Run ID | Selected child |
| --- | --- | --- |
| Seasonal reference | `4d6b6826f83c42b5a26a3d9128aeb502` | — |
| Ridge study parent | `54657465dc34459a86bff855c19345ac` | Trial 2: `23c3d2e3c5764699a5201b825af03002` |
| Forest study parent | `e90482dae66c49d5a503ab2c20190d0e` | Trial 9: `59ae5bcfd2804f01839d458da1fe985e` |

- Ridge logged model `m-aa73150b27f6419b8fc76531ce118290`, output `outputs/2026-11-13_22-13-46Z_ridge_v1_b21b7d`.
- Forest logged model `m-cca3bfbc5a8b4a1d975c67f03324699e`, output `outputs/2026-11-13_22-14-53Z_random_forest_v1_8db6c5`.
- Part 1's plain training runs (no MLflow): `outputs/2026-11-13_22-13-12Z_ridge_v1_87e9e4` and `outputs/2026-11-13_22-13-27Z_random_forest_v1_74ded3`.

Filter: `tags.lecture_batch = '20261113T221340Z-4f68dd' and attributes.status = 'FINISHED'`.

## Model retrieval and alias illustration

`champion` resolved to `models:/electricity-mt158-lecture/2`. Given 168 observed hours ending `2014-11-01T00:00:00`, it predicted **131.984094 kW** for `2014-11-01T01:00:00` (published Portuguese wall clock). The rollback illustration moved `champion` from version 2 to 1 and back; version 1 predicted 124.623567 kW for the same history, and restoring version 2 restored its original prediction. Events are in `evidence.json` and the runs' `registry_events/` artifacts. The consumption command and receipt are in [the team handoff](presenter_2/TEAM_HANDOFF_AND_REHEARSAL.md).

## Figures and screenshots

- Native MLflow UI screenshots of this batch: `docs/presenter_2/assets/mlflow_screenshots/` (runs, compare, run overview, run artifacts, logged model, registry, version) and MLflow's own charts: `folds.png` (fold_mae_kw per run by step), `search.png` (forest trial objective and best so far by trial) and `params.png` (parallel coordinates of the forest trials). Regenerate with `scripts/capture_mlflow_screenshots.py` against a running UI.
- `docs/presenter_2/assets/all_trials.csv` (all 13 trials with settings and scores) and `random_forest_forecast.png` (one validation week of the selected forest, from its training run).

Generated outputs, models, data and SQLite stores are local and ignored by Git.

## Verification and limits

- The test suite passes (32 tests) on the committed code, and the same suite runs in GitHub Actions.
- MLflow emitted an upstream SQLAlchemy deprecation and duplicate dataset-source warnings; none were suppressed. It cannot pin pip in its generated Conda spec because the uv environment has no pip; requirements and the lock are recorded. A fresh isolated serving environment has not been tested.
