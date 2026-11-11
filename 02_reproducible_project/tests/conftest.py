import numpy as np
import pandas as pd
import pytest
import yaml

from electricity_demo.config import sha256_file, write_json


@pytest.fixture
def hourly_frame():
    index = pd.date_range("2014-01-01T01:00:00", periods=1200, freq="1h")
    hours = np.arange(len(index))
    values = 70 + 0.03 * hours + 12 * np.sin(2 * np.pi * hours / 24) + 5 * np.cos(2 * np.pi * hours / 168)
    return pd.DataFrame({"timestamp": index, "load_kw": values})


@pytest.fixture
def training_config(tmp_path, hourly_frame):
    dataset = tmp_path / "hourly.csv"
    manifest = tmp_path / "manifest.json"
    hourly_frame.to_csv(dataset, index=False)
    write_json(manifest, {"dataset_sha256": sha256_file(dataset), "rows": len(hourly_frame), "snapshot": "test-fixture"})
    config = {
        "seed": 42,
        "dataset": str(dataset),
        "dataset_manifest": str(manifest),
        "features": {"lags": [1, 24, 168], "rolling_windows": [24, 168]},
        "evaluation": {
            "horizon_hours": 1, "observation_delay_hours": 0,
            "test_start": hourly_frame["timestamp"].iloc[1000].isoformat(),
            "test_end": (hourly_frame["timestamp"].iloc[-1] + pd.Timedelta(hours=1)).isoformat(),
            "cv_splits": 3, "cv_validation_hours": 48, "gap_hours": 0,
        },
        "model": {"type": "ridge", "params": {"alpha": 1.0}},
        "search": {"enabled": True, "parameters": {"model__alpha": [0.1, 1.0]}},
        "output_dir": str(tmp_path / "outputs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path
