"""Static, exportable figures from observed CV results for lecture evidence."""

from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import Figure, FigureCanvasAgg

COLORS = ["#007f86", "#7558a3", "#b8833f"]


def figure(width=11, height=5):
    result = Figure(figsize=(width, height), facecolor="white", layout="constrained")
    FigureCanvasAgg(result)
    return result


def save(fig, path):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=11)
    fig.savefig(path, dpi=180)


def plot_search(trials: pd.DataFrame, *, metrics: dict, destination: Path, model_type: str):
    ordered = trials.sort_values("trial")
    fig = figure()
    ax = fig.subplots()
    ax.plot(ordered["trial"], ordered["cv_mae_kw"], "o-", color=COLORS[0], label="Trial CV MAE")
    ax.plot(ordered["trial"], ordered["cv_mae_kw"].cummin(), "o--", color=COLORS[1], label="Best so far")
    ax.set(title=f"{model_type.replace('_', ' ').title()}: validation results across trials",
           xlabel="Trial number (0-based)", ylabel="Mean absolute error (kW)", xticks=ordered["trial"])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    save(fig, destination / "optimization.png")

    parameters = [name for name in trials if name.startswith("model.")]
    fig = figure(width=max(7, 4 * len(parameters)), height=4.6)
    axes = fig.subplots(1, len(parameters), squeeze=False)[0]
    for ax, name in zip(axes, parameters):
        ax.scatter(trials[name], trials["cv_mae_kw"], c=trials["trial"], cmap="viridis", s=75, edgecolors="white")
        ax.set(xlabel=name.removeprefix("model."), ylabel="CV MAE (kW)")
        if name.endswith("alpha") and (trials[name] > 0).all():
            ax.set_xscale("log")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Observed parameter settings and validation error", fontsize=16)
    save(fig, destination / "hyperparameters.png")


def plot_model_comparison(records, destination: Path):
    fig = figure()
    ax = fig.subplots()
    names = [record["label"] for record in records]
    scores = [record["cv_mae_kw"] for record in records]
    bars = ax.barh(names, scores, color=COLORS[:len(records)], height=.6)
    ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=12)
    ax.invert_yaxis()
    ax.set(xlabel="Development CV MAE (kW), lower is better", title="Models evaluated on identical forecast targets",
           xlim=(0, max(scores) * 1.2))
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    save(fig, destination / "model_comparison.png")

    fig = figure()
    ax = fig.subplots()
    for record, color in zip(records, COLORS):
        ax.plot(range(1, len(record["fold_mae_kw"]) + 1), record["fold_mae_kw"], "o-", color=color,
                label=record["label"], linewidth=2, markersize=7)
    ax.set(xlabel="Chronological validation fold", ylabel="MAE (kW)",
           title="A single average hides variation across evaluation windows",
           xticks=range(1, len(records[0]["fold_mae_kw"]) + 1))
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=.2)
    save(fig, destination / "fold_comparison.png")


def plot_diagnostics(predictions: pd.DataFrame, destination: Path):
    cv = predictions.loc[predictions["split"].eq("cv")].copy()
    cv["timestamp"] = pd.to_datetime(cv["timestamp"])
    cv["absolute_error"] = (cv["predicted_kw"] - cv["actual_kw"]).abs()
    hourly = cv.groupby(cv["timestamp"].dt.hour)["absolute_error"].agg(["mean", "count"])
    hourly.to_csv(destination / "error_by_hour.csv", index_label="hour")
    fig = figure(width=12, height=4.8)
    left, right = fig.subplots(1, 2)
    left.scatter(cv["actual_kw"], cv["predicted_kw"], s=10, alpha=.35, color=COLORS[0])
    low = min(cv["actual_kw"].min(), cv["predicted_kw"].min())
    high = max(cv["actual_kw"].max(), cv["predicted_kw"].max())
    left.plot([low, high], [low, high], "--", color="#555555")
    left.set(xlabel="Observed load (kW)", ylabel="Predicted load (kW)", title="Out-of-fold predictions")
    right.bar(hourly.index, hourly["mean"], color=COLORS[0])
    right.set(xlabel="Target hour on source clock", ylabel="Mean absolute error (kW)",
              title="Errors by hour across CV windows", xticks=range(0, 24, 3))
    save(fig, destination / "diagnostics.png")


def plot_tradeoff(trials: pd.DataFrame, destination: Path):
    fig = figure()
    ax = fig.subplots()
    for i, (family, group) in enumerate(trials.groupby("model_family", sort=False)):
        ax.scatter(group["cv_elapsed_seconds"], group["cv_mae_kw"], label=family.replace("_", " "),
                   color=COLORS[i], s=65, alpha=.8)
    ax.set(xlabel="CV evaluation time (seconds on this machine)", ylabel="CV MAE (kW)",
           title="Validation error and local evaluation cost")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save(fig, destination / "evaluation_cost.png")
