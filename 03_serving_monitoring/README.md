# Part 3: Serving and monitoring

Code and results for Part 3. Part 3 takes the model that Part 2 registered as `electricity-mt158-lecture@champion` and asks two questions: how does it produce forecasts, and how do we know it keeps working?

The answer to the second question is a **production replay**. The champion was fitted on development data that ends at `2014-11-01 00:00` and is frozen. November and December 2014 (1,464 hours) were never used for fitting or model selection. We replay them hour by hour, exactly as an hourly batch job would:

```text
last 168 observed hours -> features.build_next_features -> champion (version 2) -> forecast for the next hour
one hour later          -> compare the forecast with the real reading
```

Nothing is fitted or tuned on these hours. They are the team's reserved final-test period, used only after the model was chosen and frozen.

## Files

| File | Purpose |
| --- | --- |
| `replay.py` | Resolves the alias, replays Nov–Dec through the shared feature code, computes the monitoring figures |
| `charts.py` | Draws the four charts from the replay results |
| `results/replay_predictions.csv` | Every replayed hour: forecast, real reading, same-hour-yesterday baseline, errors |
| `results/replay_daily.csv` | Daily average error, daily load and alert flag |
| `results/replay_summary.json` | All reported numbers |
| `results/chart_*.png` | The four charts |

## Reproduce

From `02_reproducible_project`, after preparing the data (see its README):

```bash
uv sync --locked --extra tuning
uv run --locked --extra tuning python scripts/lecture_demo.py          # Part 2 batch: registers the champion
uv run --locked --extra tuning python ../03_serving_monitoring/replay.py
uv run --locked --extra tuning python ../03_serving_monitoring/charts.py
```

A single forecast is Part 2's command:

```bash
uv run --locked --extra tuning python -m electricity_demo.predict --history outputs/<forest run>/history.csv
```

Rerun on 20 November 2026 on commit `d05b50c` in a fresh MLflow store: the champion again resolved to version 2 and the replay predictions were identical to the original 14 November run.

## How each number is computed

All values are in `results/replay_summary.json` unless noted.

| Topic | Number | Meaning |
| --- | --- | --- |
| One forecast | 131.98 kW, 130.3 kW, 1.7 kW | First replayed hour (01:00 on 1 Nov): forecast, real reading, error |
| Two months replayed | 8.7 kW | Mean absolute error over the 1,464 replayed hours |
| | 8.4 kW | Validation (CV) MAE of the champion, from its MLflow run |
| | 32.5 kW | MAE of the same-hour-yesterday baseline on the same hours |
| | ≈10% | Live MAE divided by the mean live load (87.9 kW) |
| Daily error and alerts | 12.6 kW | Alert line = 1.5 × validation MAE (our demo choice, not a standard) |
| | 9 of 60 days | Full days whose average error exceeds the alert line (see correction below) |
| Drift vs model quality | 74 → 88 kW | Mean load of the development hours (Jan–Oct) vs the replayed hours (Nov–Dec) |
| | 8.4 → 8.7 kW | Validation MAE vs live MAE |
| | Oct ≈ 90 kW | Monthly mean load; October is part of the training data |
| Christmas alert | 47 vs 23 kW | Forecast vs real reading on 25 Dec at 13:00 (47.0 vs 22.6 kW) |
| Values outside the training range | 209.7 kW, 191.9 kW, 160 kW | Live peak (26 Nov 18:00), highest training value, forecast for that hour |
| | 174.1 kW, 5 hours | Highest forecast in the replay; live hours above the training maximum |
| Error by weekday | 12.5 vs 1.9 kW | Average daily error on Fridays vs Sundays |

**Alert-day count.** Nov–Dec has 60 full days (1 Nov has 23 replayed hours, 1 Jan only one); 9 of them cross the alert line, or 11 of 62 when the two partial days are included. An earlier hand-written chart title said 10 of 61; `charts.py` now computes the count.

## Scope

- One customer, one model, historical data. The hourly job, alerts, shadow mode and canary releases were explained conceptually and are not implemented here.
- The alert threshold is illustrative. A real team tunes it so alerts stay rare and meaningful.
- The holiday explanation for Christmas is a hypothesis: the model's inputs contain the hour and weekday, but no holiday calendar.
- Random forests average training targets, so their forecasts stay within the training range; this explains why the highest replayed forecast (174.1 kW) stays below the training maximum (191.9 kW).
