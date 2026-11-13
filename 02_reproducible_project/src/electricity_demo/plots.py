"""Shared figure helpers and the search-progress plots logged by tune.py."""

from pathlib import Path

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
