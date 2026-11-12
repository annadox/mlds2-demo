import pandas as pd
from pandas.testing import assert_frame_equal

from electricity_demo.features import build_features

FEATURES = {"lags": [1, 24, 168], "rolling_windows": [24, 168]}


def test_feature_values_use_only_completed_hours(hourly_frame):
    table = build_features(hourly_frame, FEATURES)
    row = table.iloc[0]
    assert table.index[0] == hourly_frame["timestamp"].iloc[168]
    assert row["lag_1h"] == hourly_frame["load_kw"].iloc[167]
    assert row["lag_24h"] == hourly_frame["load_kw"].iloc[144]
    assert row["lag_168h"] == hourly_frame["load_kw"].iloc[0]
    assert abs(row["mean_24h"] - hourly_frame["load_kw"].iloc[144:168].mean()) < 1e-12


def test_current_and_future_targets_cannot_change_current_or_past_features(hourly_frame):
    original = build_features(hourly_frame, FEATURES)
    changed = hourly_frame.copy()
    cutoff = changed["timestamp"].iloc[600]
    changed.loc[changed["timestamp"] >= cutoff, "load_kw"] *= 100
    altered = build_features(changed, FEATURES)
    assert_frame_equal(original.loc[:cutoff].drop(columns="target"), altered.loc[:cutoff].drop(columns="target"))


def test_full_day_lags_are_explicit_historical_values(hourly_frame):
    config = {"lags": list(range(1, 25)) + [168], "rolling_windows": [24]}
    table = build_features(hourly_frame, config)
    for lag in range(1, 25):
        assert table.iloc[0][f"lag_{lag}h"] == hourly_frame["load_kw"].iloc[168 - lag]


def test_next_hour_features_match_training_without_the_target(hourly_frame):
    from electricity_demo.features import build_next_features
    config = {"lags": list(range(1, 25)) + [168], "rolling_windows": [24, 168]}
    table = build_features(hourly_frame, config).drop(columns="target")
    spec = {"features": config, "column_order": list(table.columns), "horizon_hours": 1}
    history = hourly_frame.iloc[432:600]
    actual = build_next_features(history, spec)
    expected = table.loc[[hourly_frame["timestamp"].iloc[600]]].astype(float)
    assert_frame_equal(actual, expected, check_freq=False, rtol=1e-12, atol=1e-12)


def test_next_hour_features_reject_short_or_gapped_history(hourly_frame):
    import pytest
    from electricity_demo.features import build_next_features
    table = build_features(hourly_frame, FEATURES).drop(columns="target")
    spec = {"features": FEATURES, "column_order": list(table.columns), "horizon_hours": 1}
    with pytest.raises(ValueError, match="168"):
        build_next_features(hourly_frame.iloc[:167], spec)
    with pytest.raises(ValueError, match="uninterrupted"):
        build_next_features(hourly_frame.iloc[:200].drop(index=50), spec)
