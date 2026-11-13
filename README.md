# From Notebook to Production: Reproducibility, Experiment Tracking and Model Management in Modern ML Systems

Shared demonstration repository for a 75-minute lecture in Machine Learning for Data Science 2 (University of Ljubljana, 16 November 2026). The lecture follows the practical ML lifecycle from a data scientist's perspective: from exploratory notebook to reproducible training, tracked experiments, managed models and inference. It is tool-focused and practical, not an infrastructure or DevOps tutorial.

## Lecture sections

| Presenter | Section | Time | Topics |
| --- | --- | --- | --- |
| 1 | From Notebook to Reproducible ML | 0–23 | Notebook limitations, project structure, Git, uv and dependency locking, preprocessing pipelines, leakage prevention, chronological evaluation, repeatable training |
| 2 | Experiment Tracking and Model Management with MLflow | 23–48 | Experiments and runs, parameters/metrics/artifacts, provenance, comparing runs, logged models, Model Registry, versions, aliases, retrieval |
| 3 | From Managed Model to Production | 48–70 | Batch and online inference, serving, model interfaces, training-serving consistency, feature stores (conceptually), monitoring, drift, retraining |
| All | Questions | 70–75 | |

Decks: [Part 1](docs/presenter_1/README.md) and [Part 2](docs/presenter_2/README.md). The shared task contract, presenter handoffs and rehearsal plan are in [docs/presenter_2/TEAM_HANDOFF_AND_REHEARSAL.md](docs/presenter_2/TEAM_HANDOFF_AND_REHEARSAL.md).

## Shared demonstration

One-hour-ahead electricity demand forecasting for a single customer (`MT_158`) of the UCI [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset: quarter-hour kW readings aggregated to 8,760 hourly means for 2014. Ridge regression is the reference model and a random forest the main example; both are compared with a same-hour-yesterday baseline under three expanding chronological CV folds, with November–December reserved as a final test period.

| Part | Contents | Used by |
| --- | --- | --- |
| [01_notebook/electricity_forecasting.ipynb](01_notebook/electricity_forecasting.ipynb) | Self-contained "before" notebook with deliberate mistakes | Presenter 1 |
| [02_reproducible_project](02_reproducible_project/README.md) | uv-locked package: data preparation, causal features, chronological CV, sklearn pipelines, per-run artifacts, DVC data versions | Presenter 1 |
| Same project, `--extra tuning` | MLflow tracking, Optuna trials as nested runs, model logging, registry and aliases | Presenter 2 |
| `electricity_demo.predict` | Loads `champion` from the registry and forecasts the next hour from observed history | Presenter 3 |

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

The notebook is an intentional counterexample for Presenter 1, not code to fix: a centered rolling mean that uses future load, scaling fitted before splitting, an unseeded shuffled split, and a cell that mutates and overwrites the data file (compounding on rerun). Its scores are not comparable with the packaged project.

## Documentation

| Document | Purpose |
| --- | --- |
| [02_reproducible_project/README.md](02_reproducible_project/README.md) | Setup, training, forecasting protocol, modules and run artifacts |
| [02_reproducible_project/data/README.md](02_reproducible_project/data/README.md) | Dataset source, aggregation, timestamp/DST and zero-value policies |
| [02_reproducible_project/experiments/README.md](02_reproducible_project/experiments/README.md) | DVC snapshots v1/v2, recovery and remote storage |
| [02_reproducible_project/experiments/verification.md](02_reproducible_project/experiments/verification.md) | Validation record of the base workflow, including earlier final-test scores |
| [docs/MLFLOW_DEMO.md](docs/MLFLOW_DEMO.md) | MLflow commands, run structure, how to read the UI, scope decisions |
| [docs/MLFLOW_ENHANCED_RESULTS.md](docs/MLFLOW_ENHANCED_RESULTS.md) | Recorded run IDs, scores, versions and prediction used in the slides |
| [docs/presenter_1/](docs/presenter_1/README.md) | Presenter 1 deck (PPTX/PDF) and slide manuscript with speaker notes |
| [docs/presenter_2/](docs/presenter_2/README.md) | Presenter 2 deck (PPTX/PDF), manuscript and speaker notes, screenshots, team handoff |

Data, environments, MLflow stores and generated outputs stay outside Git; the project tracks data through DVC pointers. Configure a persistent DVC remote before sharing data across machines.

Dataset: Artur Trindade (2015), [ElectricityLoadDiagrams20112014](https://doi.org/10.24432/C58C86), UCI Machine Learning Repository, CC BY 4.0. Snapshot v2 is a synthetic sensitivity scenario, not an upstream revision.
