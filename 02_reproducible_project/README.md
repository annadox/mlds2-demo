# Reproducible electricity forecasting

A configuration-driven workflow for hourly electricity load forecasting, with versioned data, chronological evaluation and reproducible run artifacts.

## Setup

From the repository root (commands are identical in a macOS/Linux shell and PowerShell):

```bash
cd 02_reproducible_project
uv sync --locked
uv run --locked python -m electricity_demo.prepare
uv run --locked dvc add data/raw/LD2011_2014.txt.zip
uv run --locked python -m electricity_demo.snapshots create
uv run --locked python -m pytest -q
```

The `dvc add` and `snapshots create` steps set up the optional v1/v2 data-version illustration; training only needs `prepare`. Python 3.12.1 is pinned by `.python-version`; uv can install it if needed. `pyproject.toml` defines the installable `src/electricity_demo` package and `uv.lock` fixes resolved dependencies. The package itself depends only on pandas, NumPy, scikit-learn and PyYAML, which is what a consumer needs to build features and predict; MLflow is the optional `tracking` extra (plus Optuna in `tuning`). Training-only tools (matplotlib, rich, DVC) are the `train` dependency group, which `uv sync` installs by default in this repository together with `dev`. `uv.toml` keeps its disposable cache inside this project. Preparation downloads the ~249 MiB UCI archive once and reads two columns directly from the ZIP. Later commands work offline from the retained data/cache. For an existing checkout with an accessible DVC remote, use `uv run --locked dvc pull` instead of downloading/preparing again.

The `.dvc/` project configuration is included already, so do **not** rerun `dvc init`. The cache uses copies to keep source and cached files independent during data updates. DVC's auxiliary site cache stays under `.dvc/tmp/` to avoid requiring system-directory write access. Run DVC commands from this project directory.

## Train and compare

```bash
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml
uv run --locked python -m electricity_demo.train --config configs/random_forest.yaml
```

Both models use the same target timestamps, input features, chronological folds and metrics. Ridge searches three `alpha` values across three expanding folds. The forest uses a fixed configuration. Each candidate is fitted once per fold; the selected configuration is then refitted on the development period. Default training uses development CV only; it does not compute a final test score. Choose the configuration using the reported development CV, then explicitly evaluate that chosen configuration:

```bash
# Use the selected config; ridge.yaml is an example, not a preselected winner.
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml --final-test
```

The final test period is reserved for evaluation after model selection; use development CV for iterative tuning. Selection CV for chosen search parameters is labelled as such; it is not an unbiased estimate of the whole search process.

All configured relative paths resolve against this project, not the caller's current directory. No `PYTHONPATH` changes are needed after installation.

## Forecasting protocol

- One fixed customer: `MT_158`, hourly mean kW for the 8,760 intervals of 2014.
- Forecast `y_t` after `y_(t-1)` is observed. Features: the previous 24 hourly readings, lag 168, shifted means over 24/168 previous hours, hour and weekday. Both models have access to the same information. The lag list explicitly records all 24 recent hours and lag 168.
- Drop the first 168 target rows without sufficient history.
- Development: target hours before `2014-11-01 01:00`; final test: 1,464 hours from that point through `2015-01-01 00:00` inclusive.
- Three expanding development folds, each with a 336-hour validation block. No shuffled splits. Gap zero is appropriate for this one-step, immediately observed target definition. A different horizon or observation delay is rejected rather than silently evaluated incorrectly.
- Fit the scaler/calendar encoder and estimator together inside each training fold. A `Pipeline` cannot fix an already leaking feature; causal feature construction is tested separately.
- Fit the selected model once on all development targets. During testing, the model remains fixed while observed history advances each hour. Earlier actual test readings can enter later one-step features. This is not a full future-block forecast from one origin.
- Compare with `y_(t-24)` on identical target timestamps. Report MAE and RMSE in kW, not classification accuracy or MAPE around zeros.

Source clock and DST limitations are documented in [data/README.md](data/README.md). The source has a preprocessed Portuguese wall clock, not a uniform physical UTC series; that policy is preserved explicitly.

## Code structure

The package is a library of small, tested functions; notebooks, CLIs, tracking and inference are thin clients of it.

| Layer | Modules | Imports MLflow? |
|---|---|---|
| Core library | `prepare.py` (ingestion/validation), `features.py` (one causal feature definition: `feature_table`, used by `build_features` for training and `build_next_features` for inference), `split.py` (evaluation boundaries), `pipeline.py` (fitted preprocessing + estimator), `evaluate.py` (metrics/plots), `config.py` (validated configuration, paths, identities) | No |
| Training | `train.py`: `load_development` → `fit_and_evaluate` (in-memory `TrainingResult`) → `save_run` (artifacts, manifest); `run_training` combines them; `report.py` writes summaries and the cross-run comparison | No |
| Data versions | `snapshots.py` (DVC snapshot catalog) | No |
| MLflow adapters | `tracking.py` (log a `TrainingResult` as a run with a model), `tune.py` (nested trial runs; the parent reuses the winning trial's CV), `selection.py` (baseline run and the selection rule as code), `registry.py` (versions, aliases) | Yes (`tracking` / `tuning` extras) |
| Inference | `predict.py` (resolve alias → build features → predict), `store.py` (store location) | Yes, but never imports training code |

`scripts/lecture_demo.py` and `scripts/lecture_figures.py` orchestrate the Presenter 2 evidence batch; they are lecture scripts, not part of the package. See [../docs/MLFLOW_DEMO.md](../docs/MLFLOW_DEMO.md).

Paths in configs resolve against this project directory. An installed (non-editable) copy of the package can be pointed at a checkout with `ELECTRICITY_DEMO_ROOT=/path/to/02_reproducible_project`.

Each execution creates `outputs/<unique-run-id>/`:

| File | Purpose |
|---|---|
| `summary.md` | Readable metrics, fold dates, plots, configuration and artifact links |
| `config.yaml` | Resolved experiment inputs |
| `manifest.json` | Code revision/dirty state, data and lockfile hashes, environment and actual split boundaries |
| `metrics.json` | CV metrics, seasonal baseline, chosen parameters, optional final test metrics |
| `cv_results.csv` | Search candidates, fold scores and timings; explicit positive MAE column |
| `predictions.csv` | Timestamp, prediction origin, fold/split, actual, forecast and baseline |
| `forecast.png` | First forecast week for a readable presentation plot |
| `model.pkl` | Standard Python pickle containing the fitted `Pipeline`, including preprocessing |
| `feature_spec.json` | Feature definition, column order and required history |
| `history.csv` | Last required observed hours of the fitted development period |

Only load pickle models you trust. These artifacts can be consumed by an experiment-tracking system or a separate inference workflow.

## Reading results

The terminal prints progress followed by a table of MAE, RMSE, baseline error and relative MAE reduction. Start with `outputs/comparison.md` for the latest compatible runs, or open a run's `summary.md`. The comparison also has a CSV and chart; per-run plots show a forecast week and residuals, for each model. Raw predictions retain full precision.

Comparison reports combine only runs with identical data contents, feature and split definitions, source hashes and environment lock. Old experiments are retained in their original folders. Rows are sorted by development CV MAE; final-test metrics are displayed separately.

```bash
uv run --locked python -m electricity_demo.report
```

To reload a fitted model from this project's environment:

```python
import pickle
with open("outputs/<run-id>/model.pkl", "rb") as handle:
    pipeline = pickle.load(handle)
predictions = pipeline.predict(feature_frame)
```

## Notebook

Select the project environment as the kernel (`.venv/bin/python` on macOS/Linux, `.venv/Scripts/python.exe` on Windows) and open `notebooks/exploration.ipynb`. For a browser interface, install the optional group:

```bash
uv sync --locked --group notebook
uv run --locked --group notebook jupyter lab
```

The notebook calls project functions and the same training entry point used by the CLI. It does not implement a second training workflow.

## Data versioning

See [experiments/README.md](experiments/README.md) for v1 -> v2 -> v1 recovery, the exact simulated correction, and remote storage instructions. The raw ZIP remains untouched. Preparation is deterministic, but a URL can change: the recorded source hash and recoverable DVC object identify the actual bytes used. Retain the cache or push it to durable storage.

## Verification

```bash
uv run --locked python -m pytest -q
uv run --locked dvc status
```

Run the suite with `--extra tuning` to include the MLflow integration tests. The same suite runs on every push and pull request in GitHub Actions (`../.github/workflows/tests.yml`); it uses synthetic fixtures, so no dataset download is needed. Checks cover interval aggregation, invalid data/gaps, dataset-manifest consistency, temporal feature causality, fold availability, train-only scaler fitting, unsupported horizons, saved-model predictions, repeated-run tolerance, final-test opt-in and preservation of test readings during the simulated correction. Recorded results: [experiments/verification.md](experiments/verification.md).
