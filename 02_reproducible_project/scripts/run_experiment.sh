#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/run_experiment.sh [options]

Run one configured experiment: environment, data preparation/validation,
causal features, chronological CV, refit, evaluation and saved reports.

Options:
  --config PATH          Training config (default: configs/ridge.yaml).
  --prepare-config PATH  Preparation config (default: configs/prepare.yaml).
  --reprepare            Rebuild data from the raw archive, replacing the selected
                         prepared snapshot with original, uncorrected data.
  --final-test           Also evaluate the reserved final test period.
  -h, --help             Show this help.

Config paths are relative to 02_reproducible_project, or absolute.
Existing data and its manifest are validated and reused unless --reprepare is set.
Use development CV for tuning; use --final-test after selecting a configuration.
EOF
}

config="configs/ridge.yaml"
prepare_config="configs/prepare.yaml"
reprepare=0
final_test=0
while (($#)); do
    case "$1" in
        --config|--prepare-config)
            if (($# < 2)) || [[ -z "$2" || "$2" == --* ]]; then
                printf 'Missing path after %s\n' "$1" >&2
                exit 2
            fi
            if [[ "$1" == --config ]]; then
                config="$2"
            else
                prepare_config="$2"
            fi
            shift 2
            ;;
        --reprepare) reprepare=1; shift ;;
        --final-test) final_test=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_dir"
if ! command -v uv >/dev/null 2>&1; then
    printf 'uv is required. Install it and rerun this script.\n' >&2
    exit 1
fi

printf '\n[1/3] Installing the locked environment\n'
# Retain optional MLflow/Optuna packages if another demo installed them already.
uv sync --locked --inexact

uv run --no-sync python - "$config" "$prepare_config" "$reprepare" "$final_test" <<'PY'
import json
import sys

from electricity_demo.config import load_training_config, project_path, read_yaml
from electricity_demo.prepare import load_hourly, prepare
from electricity_demo.report import update_comparison
from electricity_demo.train import run_training

config_path, prepare_config_path, reprepare, final_test = sys.argv[1:]
config = load_training_config(config_path)
dataset = project_path(config["dataset"])
manifest = project_path(config["dataset_manifest"])

print("\n[2/3] Preparing or validating the configured dataset", flush=True)
if reprepare == "1" or not (dataset.exists() or manifest.exists()):
    preparation = read_yaml(prepare_config_path)
    if (project_path(preparation["output"]).resolve() != dataset.resolve()
            or project_path(preparation["manifest"]).resolve() != manifest.resolve()):
        raise ValueError("Preparation output/manifest must match the training config's dataset/dataset_manifest.")
    prepare(prepare_config_path)
elif not (dataset.exists() and manifest.exists()):
    raise FileNotFoundError("Dataset/manifest pair is incomplete. Restore both with dvc pull, or use --reprepare.")

frame = load_hourly(dataset, manifest)
metadata = json.loads(manifest.read_text(encoding="utf-8"))
print(f"Using {metadata.get('snapshot', 'dataset')}: {len(frame):,} validated hourly rows", flush=True)

print("\n[3/3] Building features, training and evaluating", flush=True)
destination = run_training(config_path, final_test=final_test == "1")
update_comparison(destination.parent)
print(f"\nExperiment complete: {destination}")
print(f"Summary: {destination / 'summary.md'}")
print(f"Metrics: {destination / 'metrics.json'}")
PY
