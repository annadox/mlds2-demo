"""One causal feature definition, used for training rows and for the next-hour inference row."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .prepare import validate_hourly


def feature_table(load: pd.Series, config: dict) -> pd.DataFrame:
    """Features for every timestamp in ``load``; each row uses only earlier observations."""
    features = pd.DataFrame(index=load.index)
    for lag in sorted(config["lags"]):
        features[f"lag_{lag}h"] = load.shift(lag)
    for window in config["rolling_windows"]:
        # Excludes y_t: at prediction time the latest observed value is y_(t-1).
        features[f"mean_{window}h"] = load.shift(1).rolling(window, min_periods=window).mean()
    features["hour"] = features.index.hour
    features["weekday"] = features.index.dayofweek
    return features


def build_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Training/evaluation rows: features plus the observed target, without incomplete history."""
    validate_hourly(frame)
    load = frame.set_index("timestamp")["load_kw"]
    features = feature_table(load, config)
    features["target"] = load
    return features.dropna()


def numeric_columns(config: dict) -> list[str]:
    return [f"lag_{lag}h" for lag in sorted(config["lags"])] + [f"mean_{window}h" for window in config["rolling_windows"]]


def build_next_features(history: pd.DataFrame, feature_spec: dict) -> pd.DataFrame:
    """The next-hour inference row, computed by the same feature_table as training; no future target."""
    validate_hourly(history)
    if feature_spec["horizon_hours"] != 1:
        raise ValueError("Only next-hour forecasts are supported.")
    config = feature_spec["features"]
    required = max(config["lags"] + config["rolling_windows"])
    if len(history) < required:
        raise ValueError(f"Need at least {required} consecutive observed hours.")
    load = history.set_index("timestamp")["load_kw"].astype("float64")
    target_time = load.index[-1] + pd.Timedelta(hours=1)
    # The unknown target is a placeholder: every feature is shifted by at least one hour.
    extended = pd.concat([load, pd.Series([np.nan], index=pd.DatetimeIndex([target_time], name=load.index.name))])
    row = feature_table(extended, config).iloc[[-1]]
    if set(row.columns) != set(feature_spec["column_order"]):
        raise ValueError("Feature specification does not match the supported feature definitions.")
    row = row[feature_spec["column_order"]].astype("float64")
    if row.isna().to_numpy().any():
        raise ValueError("History is too short for the configured features.")
    return row
