"""Readable run summaries and comparisons of compatible experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from .config import project_path
from .evaluate import FigureCanvasAgg, Figure

MODEL_NAMES = {"ridge": "Ridge regression", "random_forest": "Random forest"}
console = Console(highlight=False)


def markdown_table(headers, rows) -> str:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(cell, headers)) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(lines)


def comparison_group(config: dict, dataset_hash: str, lock_hash: str, source_hashes: dict) -> str:
    identity = {"dataset": dataset_hash, "features": config["features"], "evaluation": config["evaluation"], "lock": lock_hash, "source": source_hashes}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def metric_rows(metrics: dict) -> list[list[str]]:
    rows = []
    for key, label in [("cv", "Development CV"), ("test", "Final test")]:
        if key in metrics:
            result = metrics[key]
            improvement = result["mae_improvement_percent"]
            rows.append([
                label, f"{result['model']['mae_kw']:.3f}", f"{result['model']['rmse_kw']:.3f}",
                f"{result['seasonal_baseline']['mae_kw']:.3f}",
                f"{improvement:+.1f}%" if improvement is not None else "—",
            ])
    return rows


METRIC_HEADERS = ["Evaluation", "MAE (kW)", "RMSE (kW)", "Baseline MAE (kW)", "MAE reduction"]


def fitted_parameters(manifest: dict, metrics: dict) -> dict:
    """Resolve the estimator settings after any training-time grid search."""
    return {
        **manifest["model_config"].get("params", {}),
        **{name.removeprefix("model__"): value for name, value in metrics.get("best_params", {}).items()},
    }


def write_run_summary(destination: Path, config: dict, metrics: dict, manifest: dict, search_results: pd.DataFrame) -> None:
    model_name = MODEL_NAMES[config["model"]["type"]]
    title = f"{model_name} · {manifest['dataset']['preparation'].get('snapshot', 'dataset')}"
    console.print()
    table = Table(title=title.replace("·", "|"), title_justify="left", show_header=True, header_style="bold cyan", padding=(0, 1))
    for i, header in enumerate(METRIC_HEADERS):
        table.add_column(header, justify="left" if i == 0 else "right")
    for row in metric_rows(metrics):
        table.add_row(*row)
    console.print(table)
    console.print(f"[dim]Completed in {manifest['elapsed_seconds']:.1f}s[/dim]")
    console.print(f"Summary: {destination / 'summary.md'}", soft_wrap=True)
    console.print(f"Model:   {destination / 'model.pkl'}", soft_wrap=True)
    folds = metrics["cv"]["folds"]
    fold_rows = [[
        fold["fold"], fold["train_end"].replace("T", " "),
        fold["validation_start"].replace("T", " "), fold["validation_end"].replace("T", " "),
        f"{fold['model']['mae_kw']:.3f}", f"{fold['model']['rmse_kw']:.3f}",
    ] for fold in folds]
    content = [
        f"# {title}",
        f"Customer **{manifest['dataset']['preparation'].get('preparation', {}).get('customer', 'unspecified')}** · one-hour-ahead forecasting · hourly mean kW.",
        "## Results", markdown_table(METRIC_HEADERS, metric_rows(metrics)),
        "MAE reduction is relative to the same-hour-yesterday baseline on identical timestamps. Lower MAE/RMSE is better.",
        f"Development fold MAE: **{metrics['cv']['mean_fold_mae_kw']:.3f} ± {metrics['cv']['std_fold_mae_kw']:.3f} kW** (mean ± standard deviation).",
        "Development CV is used for model selection. The final test is reported only when explicitly requested.",
        "## Forecast", "![First forecast week](forecast.png)",
        "The lower panel shows forecast minus observed load. Earlier observations provide history for subsequent hourly forecasts.",
        "## Chronological validation", markdown_table(["Fold", "Training through", "Validation from", "Validation through", "MAE (kW)", "RMSE (kW)"], fold_rows),
        "## Configuration", "```yaml\n" + (destination / "config.yaml").read_text(encoding="utf-8").strip() + "\n```",
        "## Run details", markdown_table(["Field", "Value"], [
            ["Completed (UTC)", manifest["created_utc"]], ["Elapsed", f"{manifest['elapsed_seconds']:.1f} seconds"],
            ["Python", manifest["environment"]["python"]], ["Dataset SHA-256", manifest["dataset"]["sha256"]],
            ["Saved model", "[model.pkl](model.pkl) · fitted preprocessing and estimator"],
        ]),
    ]
    if len(search_results) > 1:
        content.extend(["## Parameter search", markdown_table(["Parameters", "CV MAE (kW)", "CV std (kW)"], [
            [row["parameters"], f"{row['mean_mae_kw']:.3f}", f"{row['std_mae_kw']:.3f}"] for _, row in search_results.iterrows()
        ])])
    content.extend(["## Files", "[Metrics](metrics.json) · [Predictions](predictions.csv) · [CV results](cv_results.csv) · [Run manifest](manifest.json) · [Feature specification](feature_spec.json)"])
    (destination / "summary.md").write_text("\n\n".join(content) + "\n", encoding="utf-8")


def update_comparison(output_root: Path) -> pd.DataFrame:
    runs = []
    for path in output_root.glob("*/manifest.json"):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "comparison_group" not in manifest or manifest.get("model_type") not in MODEL_NAMES:
            continue
        metrics_path = path.parent / "metrics.json"
        if metrics_path.exists():
            runs.append((manifest, json.loads(metrics_path.read_text(encoding="utf-8")), path.parent))
    if not runs:
        return pd.DataFrame()
    runs.sort(key=lambda entry: entry[0]["created_utc"])
    group = runs[-1][0]["comparison_group"]
    # Keep the latest run of each model, with final-test runs preferred over CV-only
    # runs of the exact same model settings so a notebook exploration cannot hide results.
    latest = {}
    for manifest, metrics, folder in runs:
        if manifest["comparison_group"] != group:
            continue
        key = manifest["model_type"]
        previous = latest.get(key)
        if (previous is not None and previous[1].get("final_test_evaluated")
                and not metrics.get("final_test_evaluated")
                and fitted_parameters(previous[0], previous[1]) == fitted_parameters(manifest, metrics)):
            continue
        latest[key] = (manifest, metrics, folder)
    rows = []
    for manifest, metrics, folder in latest.values():
        rows.append({
            "model": MODEL_NAMES[manifest["model_type"]], "model_type": manifest["model_type"],
            "cv_mae_kw": metrics["cv"]["model"]["mae_kw"], "cv_rmse_kw": metrics["cv"]["model"]["rmse_kw"],
            "cv_std_mae_kw": metrics["cv"]["std_fold_mae_kw"],
            "test_mae_kw": metrics.get("test", {}).get("model", {}).get("mae_kw"),
            "test_rmse_kw": metrics.get("test", {}).get("model", {}).get("rmse_kw"),
            "cv_baseline_mae_kw": metrics["cv"]["seasonal_baseline"]["mae_kw"],
            "test_baseline_mae_kw": metrics.get("test", {}).get("seasonal_baseline", {}).get("mae_kw"),
            "elapsed_seconds": manifest["elapsed_seconds"], "run": folder.name,
        })
    frame = pd.DataFrame(rows).sort_values("cv_mae_kw").reset_index(drop=True)
    frame.to_csv(output_root / "comparison.csv", index=False, float_format="%.6f")
    def number(value):
        return f"{value:.3f}" if pd.notna(value) else "—"
    table_rows = [[
        f"[{row['model']}]({row['run']}/summary.md)", number(row["cv_mae_kw"]), number(row["cv_std_mae_kw"]),
        number(row["test_mae_kw"]), number(row["test_rmse_kw"]), f"{row['elapsed_seconds']:.1f}",
    ] for _, row in frame.iterrows()]
    newest = runs[-1][0]
    text = [
        "# Model comparison",
        f"Dataset **{newest['dataset']['preparation'].get('snapshot', 'dataset')}** · {newest['dataset']['rows']:,} hourly observations · same features and chronological folds.",
        markdown_table(["Model / report", "CV MAE (kW)", "CV std (kW)", "Test MAE (kW)", "Test RMSE (kW)", "Time (s)"], table_rows),
        "Sorted by development CV MAE. Lower is better. A dash means the final test was not evaluated. Test metrics are for reporting, not iterative parameter selection.",
        "![MAE comparison](comparison.png)",
        f"Same-hour-yesterday baseline: **{frame.iloc[0]['cv_baseline_mae_kw']:.3f} kW** development CV MAE.",
        "Only runs with identical dataset contents, features, split settings, source files and environment lock are combined. Historical or incompatible runs remain in their own folders.",
        f"Dataset SHA-256: `{newest['dataset']['sha256']}`.",
    ]
    (output_root / "comparison.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    fig = Figure(figsize=(11, max(3.6, len(frame) * 0.6 + 1.8)), facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, squeeze=False)[0]
    for ax, scope, title in zip(axes, ["cv", "test"], ["Development CV", "Final test"]):
        available = frame.dropna(subset=[f"{scope}_mae_kw"])
        names = available["model"].tolist()
        values = available[f"{scope}_mae_kw"].tolist()
        if values:
            names.append("Yesterday baseline")
            values.append(float(available.iloc[0][f"{scope}_baseline_mae_kw"]))
            bars = ax.barh(names, values, color=["#007f86"] * (len(values) - 1) + ["#b7bdc8"], height=0.6)
            ax.bar_label(bars, fmt="%.2f", padding=5, fontsize=10)
            ax.set_xlim(0, max(values) * 1.2)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "Not evaluated", ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])
        ax.set(title=title, xlabel="Mean absolute error (kW)")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)
    fig.tight_layout(w_pad=3)
    fig.savefig(output_root / "comparison.png", dpi=160)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    root = project_path(args.outputs)
    results = update_comparison(root)
    if results.empty:
        console.print("No compatible completed runs found.")
    else:
        console.print(f"Comparison: {root / 'comparison.md'}")


if __name__ == "__main__":
    main()
