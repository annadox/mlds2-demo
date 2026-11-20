"""Production replay for Part 3: run the registered champion hour by hour over Nov-Dec 2014.

The model was fitted on development data that ends at 2014-11-01 00:00 and is frozen.
For every later hour we do exactly what an hourly batch job would do:

    last 168 observed hours -> build_next_features -> champion model -> forecast
    one hour later          -> compare the forecast with the real reading

Nothing is fitted or tuned on these hours. Run from 02_reproducible_project after the
Part 2 MLflow batch (scripts/lecture_demo.py) has registered the champion:

    uv run --locked --extra tuning python ../03_serving_monitoring/replay.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import pandas as pd

from electricity_demo.config import project_path
from electricity_demo.features import build_next_features
from electricity_demo.prepare import load_hourly
from electricity_demo.store import DEFAULT_STORE, configure_store

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="electricity-mt158-lecture", help="registered model name")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--data", default="data/prepared/hourly.csv", help="relative to 02_reproducible_project")
    parser.add_argument("--output-dir", default=str(HERE / "results"))
    parser.add_argument("--alert-factor", type=float, default=1.5,
                        help="daily alert line = factor x validation MAE (a demo choice, not a standard)")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Resolve the alias once, exactly like electricity_demo.predict.
    client = configure_store(DEFAULT_STORE)
    version = client.get_model_version_by_alias(args.name, args.alias)
    uri = f"models:/{args.name}/{version.version}"
    info = mlflow.models.get_model_info(uri)
    spec = info.metadata["feature_spec"]
    last_train = pd.Timestamp(info.metadata["development_last_target"])
    model = mlflow.pyfunc.load_model(uri)
    run_metrics = client.get_run(version.run_id).data.metrics
    cv_mae = run_metrics.get("cv_mae_kw", run_metrics.get("selection_cv_mae_kw"))

    # 2. Replay every hour after the fitting period.
    data = load_hourly(project_path(args.data)).reset_index(drop=True)
    need = spec["required_history_hours"]
    rows = []
    for i in data.index[data["timestamp"] > last_train]:
        history = data.iloc[i - need:i]  # only readings observed before the target hour
        features = build_next_features(history, spec)
        assert features.index[0] == data.loc[i, "timestamp"]
        rows.append({
            "timestamp": data.loc[i, "timestamp"],
            "forecast_kw": float(model.predict(features)[0]),
            "actual_kw": float(data.loc[i, "load_kw"]),
            "baseline_kw": float(data.loc[i - 24, "load_kw"]),  # same hour yesterday
        })
    rep = pd.DataFrame(rows)
    rep["abs_error_kw"] = (rep["forecast_kw"] - rep["actual_kw"]).abs()
    rep["baseline_abs_error_kw"] = (rep["baseline_kw"] - rep["actual_kw"]).abs()
    rep.to_csv(out / "replay_predictions.csv", index=False)

    # 3. Daily monitoring view with an alert line.
    threshold = args.alert_factor * cv_mae
    rep["day"] = rep["timestamp"].dt.floor("D")
    daily = (rep.groupby("day")
             .agg(mae_kw=("abs_error_kw", "mean"), mean_load_kw=("actual_kw", "mean"),
                  baseline_mae_kw=("baseline_abs_error_kw", "mean"), hours=("abs_error_kw", "size"))
             .reset_index())
    daily["alert"] = daily["mae_kw"] > threshold
    daily.to_csv(out / "replay_daily.csv", index=False)
    full_days = daily[daily["hours"] == 24]

    # 4. Drift and data-quality figures.
    dev = data.loc[data["timestamp"] <= last_train, "load_kw"]
    live = rep["actual_kw"]
    dev_max = float(dev.max())
    peak = rep.loc[rep["actual_kw"].idxmax()]
    xmas = rep[rep["timestamp"] == pd.Timestamp("2014-12-25 13:00")].iloc[0]
    by_weekday = full_days.groupby(full_days["day"].dt.day_name())["mae_kw"].mean()
    monthly = data[data["timestamp"] < "2015-01-01"].groupby(data["timestamp"].dt.month)["load_kw"].mean()

    summary = {
        "model_uri": uri, "alias": args.alias, "source_run_id": version.run_id,
        "last_training_target": str(last_train),
        "live_period": [str(rep["timestamp"].min()), str(rep["timestamp"].max())],
        "live_hours": int(len(rep)),
        "first_forecast": {"target": str(rep.loc[0, "timestamp"]), "forecast_kw": round(rep.loc[0, "forecast_kw"], 3),
                           "actual_kw": round(rep.loc[0, "actual_kw"], 3)},
        "validation_cv_mae_kw": cv_mae,
        "live_mae_kw": float(rep["abs_error_kw"].mean()),
        "live_baseline_mae_kw": float(rep["baseline_abs_error_kw"].mean()),
        "live_relative_mae_pct": float(100 * rep["abs_error_kw"].mean() / live.mean()),
        "alert_threshold_kw": threshold,
        "alert_days_full": [str(d.date()) for d in full_days.loc[full_days["alert"], "day"]],
        "n_alert_days_full": int(full_days["alert"].sum()), "n_full_days": int(len(full_days)),
        "worst_days": [[str(r.day.date()), round(r.mae_kw, 2)] for r in full_days.nlargest(5, "mae_kw").itertuples()],
        "mae_by_weekday_kw": {k: round(v, 2) for k, v in by_weekday.items()},
        "dev_mean_load_kw": float(dev.mean()), "live_mean_load_kw": float(live.mean()),
        "monthly_mean_load_kw": {int(k): round(float(v), 1) for k, v in monthly.items()},
        "dev_max_load_kw": dev_max,
        "live_hours_above_dev_max": int((live > dev_max).sum()),
        "live_peak": {"timestamp": str(peak["timestamp"]), "actual_kw": round(peak["actual_kw"], 1),
                      "forecast_kw": round(peak["forecast_kw"], 1)},
        "highest_live_forecast_kw": round(float(rep["forecast_kw"].max()), 1),
        "christmas_13h": {"forecast_kw": round(xmas["forecast_kw"], 1), "actual_kw": round(xmas["actual_kw"], 1)},
    }
    (out / "replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
