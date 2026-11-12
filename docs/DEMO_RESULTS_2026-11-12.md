# Recorded lecture demo results

Executed locally on 12 November 2026 with Python 3.12.1 and locked MLflow 3.18.0. These are observed results from the UCI subset, not illustrative scores. Source files and the lockfile were archived with the runs. The runs disclose a dirty Git workspace because the implementation has not been committed.

## Data and evaluation

- Client MT_158, 8,760 hourly mean-kW observations.
- Prepared SHA-256: `06e155b416659aa93e510a6d233a637baee5f65970830c5715ce2f796b2733ea`, matching the original v1 snapshot descriptor.
- 7,128 development targets and three expanding validation folds of 336 hours each.
- One-hour forecasting with observed history advancing hourly.
- Identical features and target timestamps for both runs.
- No final-test scoring was requested for these runs.

| Execution | Effective configuration | CV MAE (kW) | CV RMSE (kW) | Registry version |
| --- | --- | ---: | ---: | --- |
| `lecture-ridge-search` | Selected alpha 10.0 from 0.1, 1.0, 10.0 | 9.662940 | 13.501337 | 1, comparison candidate |
| `lecture-forest-depth-10` | 40 trees, depth 10, minimum leaf size 3 | 8.427510 | 13.303743 | 2, `champion` |

The same-hour-yesterday baseline has CV MAE **30.123865 kW** on the same target rows. The forest was selected on development CV MAE. It also has lower pooled CV RMSE. This selection is specific to this dataset, feature set and validation protocol.

## Exact evidence references

Experiment: `electricity-mt158-hourly` (local experiment ID `1`). Registered model: `electricity-mt158-hourly`.

| Field | Ridge | Random forest |
| --- | --- | --- |
| Run ID | `c9bb5f57b00f4a2c9e683023b3dad95f` | `f73b3e5a2dd14b7d92a91c989fc1dd91` |
| Logged model ID | `m-79fb946bb9fd4561b29ac0c023bed41f` | `m-ca7b88f3c4e441ca88c5aaf8c47ee75b` |
| Local output folder | `outputs/2026-11-12_17-42-08Z_ridge_v1_ec4a95` | `outputs/2026-11-12_17-42-27Z_random_forest_v1_1ea427` |

Both final runs have `lecture_demo=true` and comparison group `902cf6baded85dd7e4b9c9437db105090b5a87d1266f07cac8d59f03b72c744b`. Filter the run table with:

```text
tags.lecture_demo = 'true' and attributes.status = 'FINISHED'
```

Earlier development attempts remain visible in the store, including a failed forest logging attempt before its serialization configuration was corrected. Use the tagged final pair for the lecture screenshots. Run status is part of the experiment record.

Machine-readable records are in `02_reproducible_project/outputs/lecture_runs.json`; each execution also has `tracking.json`. Generated data, models and the tracking database are ignored by Git and are local to this machine.

## Verified model handoff

`models:/electricity-mt158-hourly@champion` resolved to **version 2**. Loading that version and supplying the saved 168 observed hours produced:

```json
{
  "model_uri": "models:/electricity-mt158-hourly/2",
  "source_run_id": "f73b3e5a2dd14b7d92a91c989fc1dd91",
  "prediction_origin": "2014-11-01T00:00:00",
  "target_timestamp": "2014-11-01T01:00:00",
  "predicted_kw": 132.62884861213448
}
```

Timestamps retain the published Portuguese wall-clock convention. This is a historical next-hour forecast from available observations, not a current electricity forecast. No unknown target was supplied to construct its features. This single inference did not score the final test period.

To reproduce it from the project directory:

```bash
uv run --locked --extra tracking python -m electricity_demo.predict --history outputs/2026-11-12_17-42-27Z_random_forest_v1_1ea427/history.csv
```

The full output is saved as `outputs/next_hour_forecast.json`.

## Validation

The completed suite after the final implementation passed **26 tests**. Tests exercise both model families through logging, reloading, registration, alias retrieval and history-only inference. They also check validation metrics, selected parameters, default test exclusion, feature causality, schema rejection and failed-run status.

MLflow uses explicit `skops` serialization. The internally fitted random forest requires the specific trusted type `sklearn.tree._tree.Tree`; no general acceptance of arbitrary deserialized types was added. Its model environment pins the installed skops version.

One upstream SQLAlchemy deprecation warning was emitted during the tests. No warning was suppressed. MLflow also warns that the generated Conda specification cannot pin pip because this uv-managed environment does not contain pip; application requirements and the project lock remain recorded. A fresh isolated serving-environment installation was not exercised.

The local MLflow server's root and health endpoints returned HTTP 200 at `http://127.0.0.1:5000`. Its HTTP API returned both tagged FINISHED runs, their training artifact listings, and `champion` pointing to version 2 with the correct forest source run. Browser automation was unavailable, so no UI screenshots or visual inspection are claimed. The slide plan specifies the views to capture manually.
