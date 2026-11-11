"""Build and restore two local DVC snapshots without switching the code revision."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, project_path, sha256_file, write_json
from .prepare import load_hourly

DATASET = project_path("data/prepared/hourly.csv")
MANIFEST = project_path("data/prepared/manifest.json")
SNAPSHOTS = PROJECT_ROOT / "experiments" / "snapshots"
POINTERS = [DATASET.with_suffix(".csv.dvc"), MANIFEST.with_suffix(".json.dvc")]


def dvc(*args: str) -> None:
    # No commits, branch changes, or automatic Git staging are performed.
    subprocess.run([sys.executable, "-m", "dvc", *args], cwd=PROJECT_ROOT, check=True)


def corrected_snapshot(frame: pd.DataFrame, metadata: dict) -> tuple[pd.DataFrame, dict]:
    changed = frame.copy(deep=True)
    correction_start, correction_end = pd.Timestamp("2014-06-01T01:00:00"), pd.Timestamp("2014-06-15T01:00:00")
    mask = changed["timestamp"].ge(correction_start) & changed["timestamp"].lt(correction_end)
    if int(mask.sum()) != 336:
        raise ValueError("Snapshot scenario expects all 336 hours of the specified June training interval.")
    changed.loc[mask, "load_kw"] *= 1.25
    result = copy.deepcopy(metadata)
    result["snapshot"] = "v2"
    result["simulated_corrections"] = [{
        "description": "Synthetic sensitivity scenario: +25% adjustment to training-period meter readings; not an upstream UCI correction.",
        "start": correction_start.isoformat(), "end_exclusive": correction_end.isoformat(),
        "factor": 1.25, "rows": int(mask.sum()),
    }]
    return changed, result


def save_pointer_snapshot(name: str) -> None:
    destination = SNAPSHOTS / name
    destination.mkdir(parents=True, exist_ok=True)
    for pointer in POINTERS:
        shutil.copyfile(pointer, destination / pointer.name)
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    write_json(destination / "snapshot.json", {
        "name": name,
        "dataset_sha256": sha256_file(DATASET),
        "manifest_sha256": sha256_file(MANIFEST),
        "corrections": metadata["simulated_corrections"],
    })


def create_snapshots() -> None:
    frame = load_hourly(DATASET, MANIFEST)
    metadata = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if metadata.get("snapshot") != "v1" or metadata.get("simulated_corrections"):
        raise ValueError("Start from unmodified prepared v1 data. Restore v1 or rerun preparation first.")
    # Check any existing catalog before overwriting it with a different source dataset.
    if (SNAPSHOTS / "v1" / "snapshot.json").exists():
        existing = json.loads((SNAPSHOTS / "v1" / "snapshot.json").read_text(encoding="utf-8"))
        if existing["dataset_sha256"] != sha256_file(DATASET):
            raise ValueError("Existing snapshot catalog belongs to different data; preserve it before making a new scenario.")
    original_data, original_manifest = DATASET.read_bytes(), MANIFEST.read_bytes()
    dvc("add", "data/prepared/hourly.csv", "data/prepared/manifest.json")
    save_pointer_snapshot("v1")
    try:
        changed, changed_metadata = corrected_snapshot(frame, metadata)
        changed.to_csv(DATASET, index=False, float_format="%.10f", date_format="%Y-%m-%dT%H:%M:%S", lineterminator="\n")
        changed_metadata["dataset_sha256"] = sha256_file(DATASET)
        write_json(MANIFEST, changed_metadata)
        dvc("add", "data/prepared/hourly.csv", "data/prepared/manifest.json")
        save_pointer_snapshot("v2")
    finally:
        # Restore v1 even if snapshot creation fails.
        DATASET.write_bytes(original_data)
        MANIFEST.write_bytes(original_manifest)
        for pointer in POINTERS:
            shutil.copyfile(SNAPSHOTS / "v1" / pointer.name, pointer)
    print("Cached v1 and v2; current workspace is v1. Both versions are available offline.")


def restore_snapshot(name: str) -> None:
    source = SNAPSHOTS / name
    descriptor_path = source / "snapshot.json"
    if not descriptor_path.exists():
        raise FileNotFoundError("Snapshot catalog missing. Run python -m electricity_demo.snapshots create after preparation.")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    # Refuse to overwrite a manually modified dataset or mismatched data/manifest pair.
    current = load_hourly(DATASET, MANIFEST) if DATASET.exists() and MANIFEST.exists() else None
    if current is not None:
        known_hashes = {
            json.loads(path.read_text(encoding="utf-8"))["dataset_sha256"]
            for path in SNAPSHOTS.glob("*/snapshot.json")
        }
        if sha256_file(DATASET) not in known_hashes:
            raise ValueError("Workspace data has unrecorded edits; preserve them before restoring a recorded snapshot.")
    for pointer in POINTERS:
        shutil.copyfile(source / pointer.name, pointer)
    dvc("checkout", "data/prepared/hourly.csv.dvc", "data/prepared/manifest.json.dvc")
    load_hourly(DATASET, MANIFEST)
    if sha256_file(DATASET) != descriptor["dataset_sha256"] or sha256_file(MANIFEST) != descriptor["manifest_sha256"]:
        raise ValueError("Restored content does not match the snapshot descriptor.")
    print(f"Restored {name}: SHA-256 {descriptor['dataset_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("create", help="Cache v1/v2 and save their DVC pointer catalogs; leaves v1 active.")
    restore = subparsers.add_parser("restore", help="Restore a catalogued DVC pointer and its cached dataset.")
    restore.add_argument("name", choices=["v1", "v2"])
    args = parser.parse_args()
    if args.command == "create":
        create_snapshots()
    else:
        restore_snapshot(args.name)


if __name__ == "__main__":
    main()
