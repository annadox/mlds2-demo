# Dataset snapshot management

Run commands from `02_reproducible_project` after installing the locked environment and preparing the source data. Once both versions are cached, snapshot recovery works offline.

## Prepare the cache once

```powershell
uv run --locked python -m electricity_demo.prepare
uv run --locked dvc add data/raw/LD2011_2014.txt.zip
uv run --locked python -m electricity_demo.snapshots create
```

`create` records the two prepared files in DVC, saves their small pointer files under `experiments/snapshots/v1/` and `v2/`, and leaves v1 active. v2 is an explicitly simulated +25% correction to 336 hours of June training data, not an upstream correction. Its test readings are unchanged. The source ZIP is never modified. The checked-in snapshot descriptors include hashes and the correction description.

The catalog uses real DVC objects, not duplicate CSVs. `.dvcignore` excludes the catalog from DVC's stage discovery; Git still versions those small saved pointers. The restore helper only selects the saved pointers, runs `dvc checkout` and verifies content hashes. It does not regenerate historical data, switch branches, make commits, or stage Git files. It refuses to overwrite unrecorded edits. Missing cached data requires `dvc pull` from a configured remote or recreating the cache from the original source.

## Switch dataset versions

```powershell
uv run --locked python -m electricity_demo.snapshots restore v1
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml

uv run --locked python -m electricity_demo.snapshots restore v2
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml

uv run --locked python -m electricity_demo.snapshots restore v1
uv run --locked python -m electricity_demo.train --config configs/ridge.yaml
```

The content hash in `data/prepared/hourly.csv.dvc` identifies the selected dataset version. The code, environment and model config are unchanged. The two v1 runs should agree within a numerical tolerance (use `rtol=1e-9, atol=1e-9` for predictions/metrics in the same locked environment). Runtime, run IDs, artifact paths and timestamps will differ. A seed is not a promise of byte-identical model files or portability across hardware.

The equivalent manual steps are **from this directory**, while the current data is a known unmodified snapshot:

```powershell
Copy-Item -LiteralPath experiments/snapshots/v2/hourly.csv.dvc -Destination data/prepared/hourly.csv.dvc
Copy-Item -LiteralPath experiments/snapshots/v2/manifest.json.dvc -Destination data/prepared/manifest.json.dvc
uv run --locked dvc checkout data/prepared/hourly.csv.dvc data/prepared/manifest.json.dvc
```

Restore v1 afterwards using the helper. Do not use `--force` to bypass unknown local data edits.

## Sharing and Git history

Commit the source, lockfile, configuration, `.dvc` configuration, generated data `.gitignore` files, `.dvc` pointers, and small `experiments/snapshots/` catalog. Do not commit CSV/ZIP contents, the environment, run outputs or `.dvc/cache`.

Commit the dataset and preparation-manifest pointers together when recording a version in Git. To recover a historical state, restore both pointers from the selected revision and run `dvc checkout`. The saved catalog also allows selecting a dataset version independently of the code revision.

Local cache is ready for one-machine offline use, not durable shared storage. To share both catalogued data versions, configure a persistent remote, then upload each pointer state:

```powershell
# Replace the path with your own durable directory or supported shared storage.
uv run --locked dvc remote add -d storage D:/your-persistent-dvc-store
uv run --locked dvc push data/raw/LD2011_2014.txt.zip.dvc
uv run --locked python -m electricity_demo.snapshots restore v1
uv run --locked dvc push data/prepared/hourly.csv.dvc data/prepared/manifest.json.dvc
uv run --locked python -m electricity_demo.snapshots restore v2
uv run --locked dvc push data/prepared/hourly.csv.dvc data/prepared/manifest.json.dvc
uv run --locked python -m electricity_demo.snapshots restore v1
```

Push the Git metadata separately. On another machine, pull v1 normally. To restore v2 when it is not cached yet, select the v2 pointers using the manual `Copy-Item` commands above and run `dvc pull` for both prepared pointers before using the restore helper. Avoid garbage-collecting catalogued versions: DVC GC does not automatically recognize arbitrary saved pointer copies as historical Git references.
