"""Named regression metrics, seasonal comparison, and forecast plots."""

import os
from pathlib import Path

from .config import PROJECT_ROOT

# Keep the disposable plot/font cache inside the project.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter, DayLocator
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def regression_metrics(actual, predicted) -> dict:
    return {"mae_kw": float(mean_absolute_error(actual, predicted)), "rmse_kw": float(root_mean_squared_error(actual, predicted))}


def compare_predictions(predictions: pd.DataFrame) -> dict:
    result = {
        "model": regression_metrics(predictions["actual_kw"], predictions["predicted_kw"]),
        "seasonal_baseline": regression_metrics(predictions["actual_kw"], predictions["baseline_kw"]),
        "rows": len(predictions),
    }
    baseline_mae = result["seasonal_baseline"]["mae_kw"]
    result["mae_improvement_percent"] = 100 * (1 - result["model"]["mae_kw"] / baseline_mae) if baseline_mae > 0 else None
    return result


def plot_forecast(predictions: pd.DataFrame, destination: Path, title: str) -> None:
    display = predictions.iloc[:168]
    fig = Figure(figsize=(12, 5.8), facecolor="white")
    FigureCanvasAgg(fig)
    ax, residual = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    fig.suptitle(title, x=0.09, y=0.98, ha="left", fontsize=17, fontweight="bold", color="#172b4d")
    ax.plot(display["timestamp"], display["actual_kw"], label="Observed", color="#172b4d", linewidth=1.9)
    ax.plot(display["timestamp"], display["predicted_kw"], label="One-hour forecast", color="#007f86", linewidth=1.5)
    ax.plot(display["timestamp"], display["baseline_kw"], label="Same hour yesterday", color="#b8833f", alpha=0.6, linewidth=1, linestyle="--")
    ax.set_ylabel("Average load (kW)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncols=3, frameon=False, fontsize=10)
    error = display["predicted_kw"] - display["actual_kw"]
    residual.axhline(0, color="#172b4d", linewidth=0.8)
    residual.fill_between(display["timestamp"], error, 0, color="#007f86", alpha=0.2)
    residual.plot(display["timestamp"], error, color="#007f86", linewidth=0.8)
    residual.set(ylabel="Error (kW)", xlabel="Hour ending · source Portuguese wall clock")
    residual.xaxis.set_major_locator(DayLocator())
    residual.xaxis.set_major_formatter(DateFormatter("%a\n%d %b"))
    for axis in (ax, residual):
        axis.grid(axis="y", alpha=0.15)
        axis.spines[["top", "right"]].set_visible(False)
        axis.margins(x=0.01)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.85)
    fig.savefig(destination, dpi=160)
