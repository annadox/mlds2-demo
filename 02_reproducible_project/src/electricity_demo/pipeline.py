"""Learn preprocessing inside each CV fold, together with the estimator."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import numeric_columns


def build_pipeline(config: dict) -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_columns(config["features"])),
        ("calendar", OneHotEncoder(
            categories=[list(range(24)), list(range(7))],
            handle_unknown="ignore", sparse_output=False,
        ), ["hour", "weekday"]),
    ])
    params = config["model"]["params"]
    if config["model"]["type"] == "ridge":
        model = Ridge(solver="lsqr", **params)
    elif config["model"]["type"] == "random_forest":
        model = RandomForestRegressor(random_state=config["seed"], n_jobs=1, **params)
    else:
        raise ValueError("model.type must be ridge or random_forest.")
    return Pipeline([("preprocess", preprocess), ("model", model)])
