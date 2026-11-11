# Electricity load forecasting

Hourly electricity load modelling using the Electricity Load Diagrams 2011-2014 dataset.

| Project | Contents |
|---|---|
| [01_notebook/electricity_forecasting.ipynb](01_notebook/electricity_forecasting.ipynb) | Self-contained exploration, Ridge and random-forest training, and local result files |
| [02_reproducible_project](02_reproducible_project/README.md) | Configuration-driven training, a locked Python environment, versioned data, chronological evaluation and per-run artifacts |

The lecture uses Ridge as a reference and random forest as the main model example.

Both projects use customer `MT_158`, with quarter-hour readings aggregated to hourly mean kW. Each has its own environment and data copies. Evaluation protocols differ, so scores should be interpreted within their respective protocols.

See [workflow.md](workflow.md) for setup, training, dataset version recovery and result inspection.

The notebook requires pandas, scikit-learn and matplotlib. Open it with a Python kernel whose working directory is `01_notebook`; package installation instructions are included in the notebook. Its CSV/ZIP/model/plot files are generated locally and ignored by Git.

The reproducible project uses DVC pointers to track source and prepared data. Dataset contents, environments, caches and generated models remain outside ordinary Git history. Configure a persistent DVC remote before sharing data across machines.

Dataset: Artur Trindade (2015), [ElectricityLoadDiagrams20112014](https://doi.org/10.24432/C58C86), UCI Machine Learning Repository, CC BY 4.0. The additional v2 snapshot is a synthetic sensitivity scenario, not an upstream dataset revision.
