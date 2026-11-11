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
