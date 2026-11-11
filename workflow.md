# Project workflow

## Notebook

Open `01_notebook/electricity_forecasting.ipynb` with a Python kernel containing pandas, scikit-learn, matplotlib and PyTorch. Run from the notebook's directory. The first execution downloads the source archive and creates `electricity.csv`; subsequent executions reuse that local file.

The notebook contains data loading, hourly aggregation, exploration, lag and rolling features, and separate training cells for Ridge and random forest. It reads and writes its own files independently of the packaged project.

`model_scores.csv` and the displayed table compare both models on the same evaluation rows. `predictions.csv` contains the corresponding forecasts. The notebook saves each model as a `.pkl` bundle, plus comparison and scatter plots. Its reconstruction cell reloads the random-forest bundle.

To rebuild the local CSV from the cached source, first preserve any local adjustments, then remove `electricity.csv` and `electricity_fixed.csv` and rerun the notebook.

## Packaged project setup

From the repository root:

```powershell
Set-Location 02_reproducible_project
uv sync --locked
uv run --locked python -m electricity_demo.prepare
uv run --locked dvc add data/raw/LD2011_2014.txt.zip
uv run --locked python -m electricity_demo.snapshots create
```

The source archive is downloaded once. Preparation records source and output hashes, the customer, aggregation rules, timestamp policy and units. Both prepared snapshots are retained in the local DVC cache.

## Model selection

```powershell
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml
uv run --locked python -m electricity_demo.train --config configs/random_forest.yaml
```

Compare development CV MAE on the same target timestamps. Ridge searches three regularization values; the forest uses a fixed configuration. Both models receive the same 24-hour history, seasonal features and calendar information. The seasonal reference predicts the same hour's load from the preceding day.

After selecting a configuration using development results, evaluate it on the reserved final test period:

```powershell
uv run --locked python -m electricity_demo.train --config configs/random_forest.yaml --final-test
```

Use the selected configuration in that command. Final test scores should not guide iterative parameter tuning.

## Dataset versions

```powershell
uv run --locked python -m electricity_demo.snapshots restore v1
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml

uv run --locked python -m electricity_demo.snapshots restore v2
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml

uv run --locked python -m electricity_demo.snapshots restore v1
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml
```

v2 is a synthetic sensitivity scenario applying a 25% adjustment to 336 training-period hours in June. The original archive and November/December test readings remain unchanged. Restoring v1 restores both the data and its matching preparation manifest. Repeated v1 predictions should agree within the documented numerical tolerance.

See [snapshot management](02_reproducible_project/experiments/README.md) for pointer recovery and remote-storage commands.

## Results and checks

Each run creates a separate `outputs/<run-id>/` folder. Open `outputs/comparison.md` to compare compatible model runs, or a run's `summary.md` for readable results and plots. Raw data is retained in `metrics.json`, `cv_results.csv` and `predictions.csv`. Forecast charts include an error panel. `manifest.json` records the data, environment, configuration and source identities. The fitted pipeline is saved in `model.pkl`, with feature definitions and the required history alongside it.

```powershell
uv run --locked python -m pytest -q
uv run --locked dvc status
```

Recorded validation results are in [experiments/verification.md](02_reproducible_project/experiments/verification.md). Retain the local DVC cache or push both versions to persistent storage before moving machines.
