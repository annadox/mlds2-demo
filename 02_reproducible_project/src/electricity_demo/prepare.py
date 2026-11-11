"""Download once and derive an explicitly defined hourly dataset from UCI."""

from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import project_path, read_yaml, sha256_file, write_json


def download_archive(url: str, destination: Path) -> None:
    if destination.exists():
        if not zipfile.is_zipfile(destination):
            raise ValueError(f"Existing source is not a ZIP archive: {destination}")
        print(f"Using cached source: {destination}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading UCI source to {destination} ...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "electricity-reproducibility-demo/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            downloaded = 0
            reported = 0
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded - reported >= 32 * 1024 * 1024:
                    print(f"  {downloaded / 1024**2:.0f} MiB downloaded", flush=True)
                    reported = downloaded
        if not zipfile.is_zipfile(partial):
            raise ValueError("Source download did not contain a ZIP archive.")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def aggregate_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Four interval-end kW readings -> one interval-end mean load in kW."""
    series = frame.set_index("timestamp")["load_kw"].sort_index()
    if series.index.has_duplicates or series.empty:
        raise ValueError("Quarter-hour timestamps must be nonempty and unique.")
    if not np.isfinite(series.to_numpy()).all() or (series < 0).any():
        raise ValueError("Quarter-hour load values must be finite and nonnegative.")
    if not (series.index == series.index.floor("15min")).all():
        raise ValueError("Source timestamps must lie on quarter-hour boundaries.")
    groups = series.resample("1h", closed="right", label="right")
    hourly = pd.DataFrame({"load_kw": groups.mean(), "readings": groups.count()})
    # Only complete hours are allowed. Partial first/last bins are excluded.
    hourly = hourly.loc[hourly["readings"].eq(4), ["load_kw"]]
    hourly.index.name = "timestamp"
    return hourly.reset_index()


def validate_hourly(frame: pd.DataFrame) -> None:
    if list(frame.columns) != ["timestamp", "load_kw"]:
        raise ValueError("Expected exactly timestamp and load_kw columns.")
    if len(frame) < 2:
        raise ValueError("Need at least two hourly observations.")
    dates = pd.DatetimeIndex(frame["timestamp"])
    if dates.tz is not None or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Timestamps must be unique, increasing source wall-clock values without a timezone.")
    if not (dates.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("Expected an uninterrupted hourly grid; missing rows must not silently change lag durations.")
    values = frame["load_kw"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Hourly load must be finite and nonnegative; zero is allowed.")


def load_hourly(path: Path, manifest_path: Path | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset {path}. Run python -m electricity_demo.prepare first, or dvc pull.")
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    validate_hourly(frame)
    if manifest_path is not None:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata["dataset_sha256"] != sha256_file(path):
            raise ValueError("Dataset differs from its preparation manifest. Restore a matching snapshot or prepare it again.")
        if metadata["rows"] != len(frame):
            raise ValueError("Dataset row count differs from its preparation manifest.")
    return frame


def prepare(config_path: str = "configs/prepare.yaml") -> dict:
    config = read_yaml(config_path)
    if (config["frequency"], config["aggregation"], config["timestamp_policy"], config["zero_policy"]) != (
        "1h", "mean", "source_portuguese_wall_clock", "preserve_source_values"
    ):
        raise ValueError("Unsupported preparation policy; this project preserves source clock/zeros and averages hourly kW.")
    archive_path = project_path(config["source_archive"])
    download_archive(config["source_url"], archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        member = config["source_member"]
        if member not in archive.namelist():
            raise ValueError(f"Archive has no expected member {member}.")
        with archive.open(member) as handle:
            timestamp_column = pd.read_csv(handle, sep=";", nrows=0).columns[0]
        with archive.open(member) as handle:
            # Reading two columns avoids materializing all 370 customers.
            frame = pd.read_csv(handle, sep=";", decimal=",", usecols=[timestamp_column, config["customer"]])
    frame.columns = ["timestamp", "load_kw"]
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="%Y-%m-%d %H:%M:%S")
    hourly = aggregate_hourly(frame)
    start, end = pd.Timestamp(config["start"]), pd.Timestamp(config["end"])
    if end <= start:
        raise ValueError("Preparation end must be after start.")
    hourly = hourly.loc[hourly["timestamp"].ge(start) & hourly["timestamp"].lt(end)].reset_index(drop=True)
    validate_hourly(hourly)
    if not pd.DatetimeIndex(hourly["timestamp"]).equals(pd.date_range(start, end, freq="1h", inclusive="left")):
        raise ValueError("Source does not cover every requested complete hour.")
    output = project_path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output, index=False, float_format="%.10f", date_format="%Y-%m-%dT%H:%M:%S", lineterminator="\n")
    metadata = {
        "source_url": config["source_url"],
        "source_doi": "https://doi.org/10.24432/C58C86",
        "source_creator": "Artur Trindade (2015)",
        "license": "CC BY 4.0",
        "source_archive_sha256": sha256_file(archive_path),
        "preparation_config_sha256": sha256_file(project_path(config_path)),
        "preparation": config,
        "dataset_sha256": sha256_file(output),
        "rows": len(hourly),
        "first_timestamp": hourly["timestamp"].iloc[0].isoformat(),
        "last_timestamp": hourly["timestamp"].iloc[-1].isoformat(),
        "units": "kW (mean of four quarter-hour values)",
        "intervals": "right-closed, labelled by interval end",
        "timezone_note": "Source Portuguese wall clock, NOT UTC; original DST zero/combined-hour conventions preserved.",
        "zero_note": "Zeros preserved, including source encoding of pre-activation/DST; no automatic zero-to-missing conversion.",
        "snapshot": "v1",
        "simulated_corrections": [],
    }
    write_json(project_path(config["manifest"]), metadata)
    print(f"Prepared {len(hourly):,} hourly rows for {config['customer']}; SHA-256 {metadata['dataset_sha256']}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prepare.yaml")
    args = parser.parse_args()
    prepare(args.config)


if __name__ == "__main__":
    main()
