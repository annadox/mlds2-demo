# Electricity data

Source: Artur Trindade (2015), *ElectricityLoadDiagrams20112014*, UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C58C86. License: CC BY 4.0.
Dataset documentation: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014.

The original source has 370 customers and quarter-hour kW readings. This project uses `MT_158` and the 8,760 hourly intervals of 2014. Customer choice is fixed before model comparison, not selected by holdout accuracy.

`raw/LD2011_2014.txt.zip` is the untouched source archive. `prepared/hourly.csv` contains exactly `timestamp,load_kw`. The four quarter-hour values ending at 00:15, 00:30, 00:45 and 01:00 become the average load for the hour ending at 01:00, in kW. A sum of kW readings is not used. Energy for those four readings would be their sum divided by four in kWh.

Timestamps preserve the source's Portuguese wall-clock representation; they are not UTC. UCI encodes the March DST hour using zero readings and combines the duplicated October hour. Those conventions are retained and documented, not silently converted into a timezone-aware physical elapsed-time series. Seasonal features therefore refer to positions on that published hourly clock. The source encoding is retained rather than reconstructed as an energy-settlement time series.

Source zeros are preserved: they can mean actual zero load, a customer not yet active, or a DST convention. Do not replace all zeros with missing values. Validation rejects missing/duplicate/negative/nonfinite readings and gaps in the prepared hourly grid. Initial feature rows without sufficient history are dropped explicitly.

`prepared/manifest.json` records the source archive hash, preparation-config hash, policies, units, customer and output hash. A dataset and manifest must match before training proceeds. DVC pointer files are stored in Git; dataset contents live in the local DVC cache or a separately configured remote. `dvc push` and `git push` are separate operations.

Snapshot v2 applies a synthetic +25% correction to 336 training-period hours in June. This is **not an upstream UCI correction**. November/December test readings remain unchanged. Both the data and matching manifest are restored together. See `../experiments/README.md` for the snapshot-management workflow.
