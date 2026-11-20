"""Draw the four Part 3 charts from the replay results.

Run from 02_reproducible_project after replay.py:

    uv run --locked --extra tuning python ../03_serving_monitoring/charts.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from electricity_demo.config import project_path

HERE = Path(__file__).resolve().parent
TEAL, AMBER, RED, GRAY, NAVY, LIGHT = "#0E7C66", "#E8A33D", "#C2453A", "#5B7083", "#1B2A35", "#BFDDD3"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY, "axes.labelcolor": NAVY, "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.titleweight": "bold", "axes.titlesize": 15, "axes.titlecolor": NAVY, "axes.titlelocation": "left",
})


def week_chart(rep, start, end, title, path, shade=None, label=None):
    w = rep[(rep.timestamp >= start) & (rep.timestamp < end)]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    if shade:
        ax.axvspan(pd.Timestamp(shade[0]), pd.Timestamp(shade[1]), color=AMBER, alpha=0.15, lw=0)
        mid = pd.Timestamp(shade[0]) + (pd.Timestamp(shade[1]) - pd.Timestamp(shade[0])) / 2
        ax.text(mid, 205, label, ha="center", color="#8A5A10", fontsize=12, fontweight="bold")
    ax.plot(w.timestamp, w.actual_kw, color=NAVY, lw=2, label="Real reading")
    ax.plot(w.timestamp, w.forecast_kw, color=TEAL, lw=2, ls="--", label="Forecast (champion, v2)")
    ax.set_ylabel("Load (kW)")
    ax.set_ylim(0, 220)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=str(HERE / "results"))
    parser.add_argument("--data", default="data/prepared/hourly.csv", help="relative to 02_reproducible_project")
    args = parser.parse_args()
    res = Path(args.results)
    summary = json.loads((res / "replay_summary.json").read_text())
    rep = pd.read_csv(res / "replay_predictions.csv", parse_dates=["timestamp"])
    daily = pd.read_csv(res / "replay_daily.csv", parse_dates=["day"])
    daily = daily[daily.hours == 24]
    cv, th = summary["validation_cv_mae_kw"], summary["alert_threshold_kw"]

    week_chart(rep, "2014-11-03", "2014-11-10", "Replay: forecast vs real load, 3–9 November 2014",
               res / "chart_normal_week.png")
    week_chart(rep, "2014-12-20", "2014-12-28", "Replay: forecast vs real load, 20–27 December 2014",
               res / "chart_christmas_week.png", shade=("2014-12-24", "2014-12-27"), label="24–26 Dec")

    # Daily error with the alert line
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.bar(daily.day, daily.mae_kw, width=0.8, color=[RED if a else LIGHT for a in daily.alert], edgecolor="none")
    ax.axhline(cv, color=TEAL, lw=2)
    ax.axhline(th, color=RED, lw=1.5, ls="--")
    end = daily.day.iloc[-1] + pd.Timedelta(days=1)
    ax.text(end, cv, f" validation\n error {cv:.1f}", va="center", color=TEAL, fontsize=11)
    ax.text(end, th, f" alert line\n {th:.1f}", va="center", color=RED, fontsize=11)
    for day, lab in [("2014-12-25", "25 Dec"), ("2014-11-15", "15 Nov")]:
        v = daily.loc[daily.day == day, "mae_kw"].iloc[0]
        ax.annotate(lab, (pd.Timestamp(day), v), xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=11, color=RED)
    ax.set_ylabel("Average error per day (kW)")
    ax.set_xlim(daily.day.min() - pd.Timedelta(days=1), daily.day.max() + pd.Timedelta(days=5))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_title(f"Replay: forecast error for each day, Nov–Dec 2014 ({int(daily.alert.sum())} alert days in red)")
    fig.tight_layout()
    fig.savefig(res / "chart_daily_error.png", dpi=200)
    plt.close(fig)

    # Monthly average load: training vs live
    data = pd.read_csv(project_path(args.data), parse_dates=["timestamp"])
    m = data[data.timestamp < "2015-01-01"].groupby(data.timestamp.dt.month).load_kw.mean()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    bars = ax.bar(names, m.values, color=[TEAL] * 10 + [AMBER] * 2)
    for b, v in zip(bars, m.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=11, color=NAVY)
    dev_mean, live_mean = summary["dev_mean_load_kw"], summary["live_mean_load_kw"]
    ax.axhline(dev_mean, xmax=10 / 12, color=TEAL, ls="--", lw=1.5)
    ax.text(-0.4, 103, f"dashed line = training avg {dev_mean:.0f} kW", color=TEAL, fontsize=11)
    ax.text(10.5, 101, f"live avg {live_mean:.0f} kW", color="#8A5A10", fontsize=11, ha="center", fontweight="bold")
    ax.set_ylabel("Average load (kW)")
    ax.set_ylim(0, 110)
    ax.set_title("Average load per month, 2014: training (Jan–Oct) vs live (Nov–Dec)")
    fig.tight_layout()
    fig.savefig(res / "chart_monthly_load.png", dpi=200)
    plt.close(fig)
    print("charts written to", res)


if __name__ == "__main__":
    main()
