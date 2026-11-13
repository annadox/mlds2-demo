from pathlib import Path
import zipfile

import pandas as pd
import pytest
import yaml

from electricity_demo.config import read_yaml
from electricity_demo.prepare import aggregate_hourly, load_hourly, prepare, validate_hourly


def test_hourly_mean_and_interval_end_are_explicit():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2014-01-01T00:15:00", periods=8, freq="15min"),
        "load_kw": [4, 8, 12, 16, 8, 12, 16, 20],
    })
    hourly = aggregate_hourly(frame)
    assert hourly["load_kw"].tolist() == [10, 14]
    assert hourly["timestamp"].tolist() == [pd.Timestamp("2014-01-01T01:00:00"), pd.Timestamp("2014-01-01T02:00:00")]


@pytest.mark.parametrize("problem", ["duplicate", "gap", "negative", "nonfinite", "unsorted", "off_hour"])
def test_invalid_hourly_data_is_rejected(hourly_frame, problem):
    frame = hourly_frame.copy()
    if problem == "duplicate":
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
    elif problem == "gap":
        frame = frame.drop(index=5)
    elif problem == "negative":
        frame.loc[1, "load_kw"] = -1
    elif problem == "nonfinite":
        frame.loc[1, "load_kw"] = float("inf")
    elif problem == "off_hour":
        frame["timestamp"] += pd.Timedelta(minutes=30)
    else:
        frame = frame.iloc[::-1]
    with pytest.raises(ValueError):
        validate_hourly(frame)


def test_zeros_are_allowed(hourly_frame):
    hourly_frame.loc[0, "load_kw"] = 0
    validate_hourly(hourly_frame)


def test_unrecorded_data_change_is_rejected(training_config):
    config = read_yaml(training_config)
    frame = pd.read_csv(config["dataset"])
    frame.loc[0, "load_kw"] += 10
    frame.to_csv(config["dataset"], index=False)
    with pytest.raises(ValueError, match="differs from its preparation manifest"):
        load_hourly(Path(config["dataset"]), Path(config["dataset_manifest"]))


def test_preparation_reads_uci_header_decimal_comma_and_records_source(tmp_path):
    source = tmp_path / "source.zip"
    lines = ['"";"MT_001";"MT_158"']
    for date in pd.date_range("2014-01-01T00:15:00", periods=96, freq="15min"):
        lines.append(f'"{date:%Y-%m-%d %H:%M:%S}";99,0;10,5')
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("LD2011_2014.txt", "\n".join(lines))
    config = {
        "source_url": "https://example.invalid/not-downloaded",
        "source_archive": str(source), "source_member": "LD2011_2014.txt",
        "customer": "MT_158", "start": "2014-01-01T01:00:00", "end": "2014-01-02T01:00:00",
        "frequency": "1h", "aggregation": "mean",
        "timestamp_policy": "source_portuguese_wall_clock", "zero_policy": "preserve_source_values",
        "output": str(tmp_path / "hourly.csv"), "manifest": str(tmp_path / "manifest.json"),
    }
    config_path = tmp_path / "prepare.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    metadata = prepare(str(config_path))
    data = load_hourly(Path(config["output"]), Path(config["manifest"]))
    assert len(data) == 24
    assert data["load_kw"].eq(10.5).all()
    assert metadata["source_archive_sha256"]
