# From Notebook to Production: Reproducibility, Experiment Tracking and Model Management in Modern ML Systems

Shared demonstration repository for a 75-minute lecture in Machine Learning for Data Science 2 (University of Ljubljana, 16 November 2026). The lecture follows the practical ML lifecycle from a data scientist's perspective: from exploratory notebook to reproducible training, tracked experiments, managed models and inference. It is tool-focused and practical, not an infrastructure or DevOps tutorial.

## Contents

| Part | Topic | Covers |
| --- | --- | --- |
| 1 | From notebook to reproducible ML | Notebook limitations, project structure, Git, uv and dependency locking, preprocessing pipelines, leakage prevention, chronological evaluation, repeatable training |
| 2 | Experiment tracking and model management with MLflow | Experiments and runs, parameters/metrics/artifacts, provenance, comparing runs, logged models, Model Registry, versions, aliases, retrieval |
| 3 | Serving and monitoring | Batch and online inference, serving, training-serving consistency, feature stores (conceptually), monitoring, drift, retraining |

## Shared demonstration

One-hour-ahead electricity demand forecasting for a single customer (`MT_158`) of the UCI [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset: quarter-hour kW readings aggregated to 8,760 hourly means for 2014. Ridge regression is the reference model and a random forest the main example; both are compared with a same-hour-yesterday baseline under three expanding chronological CV folds, with November–December reserved as a final test period.

| Component | Contents | Part |
| --- | --- | --- |
| [01_notebook/electricity_forecasting.ipynb](01_notebook/electricity_forecasting.ipynb) | Self-contained "before" notebook with deliberate mistakes | 1 |
| [02_reproducible_project](02_reproducible_project/README.md) | uv-locked package: data preparation, causal features, chronological CV, sklearn pipelines, per-run artifacts, DVC data versions | 1 |
| Same project, `--extra tuning` | MLflow tracking, Optuna trials as nested runs, model logging, registry and aliases | 2 |
| `electricity_demo.predict` | Loads `champion` from the registry and forecasts the next hour from observed history | 3 |
| [03_serving_monitoring](03_serving_monitoring/README.md) | Production replay of Nov–Dec 2014 through the champion, monitoring figures and charts | 3 |

The demo illustrates a subset of real-world practice; serving infrastructure, monitoring tools, feature stores and CI/CD are discussed conceptually.

## Quick start

```bash
cd 02_reproducible_project
uv sync --locked
uv run --locked python -m electricity_demo.prepare          # downloads the ~249 MiB UCI archive once
uv run --locked python -m electricity_demo.train --config configs/random_forest.yaml
uv run --locked python -m pytest -q          # also runs in GitHub Actions on every push
```

Training, evaluation protocol, artifacts and DVC data versions: [02_reproducible_project/README.md](02_reproducible_project/README.md). MLflow workflow: [docs/MLFLOW_DEMO.md](docs/MLFLOW_DEMO.md).

## The notebook

Open `01_notebook/electricity_forecasting.ipynb` with a kernel that has pandas, scikit-learn and matplotlib, with `01_notebook` as the working directory. The first run downloads the source archive and creates `electricity.csv`; later runs reuse it. Its CSV/ZIP/model/plot outputs are local and ignored by Git. To rebuild the CSV, delete `electricity.csv` and `electricity_fixed.csv` and rerun.

The notebook is an intentional counterexample for Part 1, not code to fix: a centered rolling mean that uses future load, scaling fitted before splitting, an unseeded shuffled split, and a cell that mutates and overwrites the data file (compounding on rerun). Its scores are not comparable with the packaged project.

## Documentation

| Document | Purpose |
| --- | --- |
| [02_reproducible_project/README.md](02_reproducible_project/README.md) | Setup, training, forecasting protocol, modules and run artifacts |
| [02_reproducible_project/data/README.md](02_reproducible_project/data/README.md) | Dataset source, aggregation, timestamp/DST and zero-value policies |
| [02_reproducible_project/experiments/README.md](02_reproducible_project/experiments/README.md) | DVC snapshots v1/v2, recovery and remote storage |
| [02_reproducible_project/experiments/verification.md](02_reproducible_project/experiments/verification.md) | Validation record of the base workflow, including earlier final-test scores |
| [docs/MLFLOW_DEMO.md](docs/MLFLOW_DEMO.md) | MLflow commands, run structure, how to read the UI, scope decisions |
| [docs/MLFLOW_ENHANCED_RESULTS.md](docs/MLFLOW_ENHANCED_RESULTS.md) | Recorded run IDs, scores, versions and the champion's prediction |
| [03_serving_monitoring/](03_serving_monitoring/README.md) | Part 3 replay code, results and how each reported number is computed |

Data, environments, MLflow stores and generated outputs stay outside Git; the project tracks source and prepared data through DVC pointers. The configured remote is `gs://mlds2-demo/dvc`; see [snapshot management](02_reproducible_project/experiments/README.md) for authentication and uploading/recovering data versions.

Dataset: Artur Trindade (2015), [ElectricityLoadDiagrams20112014](https://doi.org/10.24432/C58C86), UCI Machine Learning Repository, CC BY 4.0. Snapshot v2 is a synthetic sensitivity scenario, not an upstream revision.
