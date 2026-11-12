"""Causal features for a new one-step forecast after each observed hour."""

from __future__ import annotations

import pandas as pd

from .prepare import validate_hourly


def build_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    validate_hourly(frame)
    data = frame.set_index("timestamp")
    load = data["load_kw"]
    features = pd.DataFrame(index=data.index)
    for lag in sorted(config["lags"]):
        features[f"lag_{lag}h"] = load.shift(lag)
    for window in config["rolling_windows"]:
        # Excludes y_t: at prediction time the latest observed value is y_(t-1).
        features[f"mean_{window}h"] = load.shift(1).rolling(window, min_periods=window).mean()
    features["hour"] = features.index.hour
    features["weekday"] = features.index.dayofweek
    features["target"] = load
    return features.dropna()


def numeric_columns(config: dict) -> list[str]:
    return [f"lag_{lag}h" for lag in sorted(config["lags"])] + [f"mean_{window}h" for window in config["rolling_windows"]]


def build_next_features(history: pd.DataFrame, feature_spec: dict) -> pd.DataFrame:
    """Build one next-hour row using observed history only, with no future target."""
    validate_hourly(history)
    if feature_spec["horizon_hours"] != 1:
        raise ValueError("Only next-hour forecasts are supported.")
    config = feature_spec["features"]
    required = max(config["lags"] + config["rolling_windows"])
    if len(history) < required:
        raise ValueError(f"Need at least {required} consecutive observed hours.")
    target_time = history["timestamp"].iloc[-1] + pd.Timedelta(hours=1)
    load = history["load_kw"]
    values = {f"lag_{lag}h": load.iloc[-lag] for lag in sorted(config["lags"])}
    values.update({f"mean_{window}h": load.iloc[-window:].mean() for window in config["rolling_windows"]})
    values.update(hour=target_time.hour, weekday=target_time.dayofweek)
    if set(values) != set(feature_spec["column_order"]):
        raise ValueError("Feature specification does not match the supported feature definitions.")
    return pd.DataFrame([values], index=pd.DatetimeIndex([target_time], name="timestamp"))[
        feature_spec["column_order"]
    ].astype("float64")
