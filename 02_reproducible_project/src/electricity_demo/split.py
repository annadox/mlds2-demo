"""Chronological development/test boundaries and expanding-window CV."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def split_development_test(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    start, end = pd.Timestamp(config["test_start"]), pd.Timestamp(config["test_end"])
    if start >= end:
        raise ValueError("test_start must be earlier than test_end.")
    development = table.loc[table.index < start]
    test = table.loc[(table.index >= start) & (table.index < end)]
    if development.empty or test.empty:
        raise ValueError("Both development and final test periods must contain observations.")
    expected = pd.date_range(start, end, freq="1h", inclusive="left")
    if not test.index.equals(expected):
        raise ValueError("Final test period must have complete hourly coverage at the configured boundaries.")
    if development.index[-1] >= test.index[0]:
        raise ValueError("Development and test periods overlap.")
    return development, test


def chronological_folds(development: pd.DataFrame, config: dict) -> list[tuple]:
    splitter = TimeSeriesSplit(
        n_splits=config["cv_splits"],
        test_size=config["cv_validation_hours"],
        gap=config["gap_hours"],
    )
    folds = list(splitter.split(development))
    for train_idx, validation_idx in folds:
        last_training_target = development.index[train_idx[-1]]
        first_prediction_origin = development.index[validation_idx[0]] - pd.Timedelta(hours=1)
        if last_training_target > first_prediction_origin:
            raise ValueError("Training includes targets unavailable at the first validation prediction origin.")
    return folds
