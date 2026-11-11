import numpy as np
import pandas as pd
import pytest
import yaml

from electricity_demo.config import load_training_config
from electricity_demo.features import build_features, numeric_columns
from electricity_demo.pipeline import build_pipeline
from electricity_demo.split import chronological_folds, split_development_test


def test_folds_respect_prediction_origins_and_exclude_final_test(hourly_frame, training_config):
    config = load_training_config(training_config)
    table = build_features(hourly_frame, config["features"])
    development, test = split_development_test(table, config["evaluation"])
    for training, validation in chronological_folds(development, config["evaluation"]):
        assert development.index[training[-1]] <= development.index[validation[0]] - pd.Timedelta(hours=1)
        assert development.index[validation[-1]] < test.index[0]
        assert len(validation) == 48
    assert development.index.intersection(test.index).empty


def test_scaler_is_fitted_only_on_fold_training_rows(hourly_frame, training_config):
    config = load_training_config(training_config)
    table = build_features(hourly_frame, config["features"])
    development, _ = split_development_test(table, config["evaluation"])
    training, _ = chronological_folds(development, config["evaluation"])[0]
    estimator = build_pipeline(config)
    estimator.fit(development.iloc[training].drop(columns="target"), development.iloc[training]["target"])
    scaler_mean = estimator.named_steps["preprocess"].named_transformers_["numeric"].mean_
    columns = numeric_columns(config["features"])
    np.testing.assert_allclose(scaler_mean, development.iloc[training][columns].mean().to_numpy())
    assert not np.allclose(scaler_mean, development[columns].mean().to_numpy())


def test_unsupported_forecast_horizon_is_rejected(training_config):
    config = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    config["evaluation"]["horizon_hours"] = 24
    training_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="one-step"):
        load_training_config(training_config)
