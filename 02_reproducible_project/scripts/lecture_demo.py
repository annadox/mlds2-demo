"""Generate the complete Presenter 2 evidence batch: baseline, searches, selection, registry and alias change.

Lecture orchestration, deliberately outside the reusable package. Run from 02_reproducible_project:

    uv run --locked --extra tuning python scripts/lecture_demo.py --demonstrate-rollback
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from electricity_demo.config import PROJECT_ROOT, write_json
from electricity_demo.predict import predict_next
from electricity_demo.registry import assign_alias, register_run
from electricity_demo.selection import log_baseline, select_candidate
from electricity_demo.store import DEFAULT_STORE, LECTURE_EXPERIMENT, configure_store
from electricity_demo.tune import run_search


def run_demo(*, store_dir=DEFAULT_STORE, experiment=LECTURE_EXPERIMENT, forest_trials=None,
             demonstrate_rollback=False):
    batch = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    evidence = PROJECT_ROOT / "outputs" / f"lecture-{batch}"
    evidence.mkdir(parents=True)
    baseline = log_baseline("configs/ridge.yaml", experiment=experiment, store_dir=store_dir, batch_id=batch)
    records, tables = [], []
    for model_type, label in [("ridge", "Ridge (grid)"), ("random_forest", "Random forest (Optuna)")]:
        output = run_search(f"configs/{model_type}.yaml", experiment=experiment, store_dir=store_dir,
                            n_trials=forest_trials if model_type == "random_forest" else None, batch_id=batch)
        receipt = json.loads((output / "tracking.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        records.append({"label": label, "output_dir": str(output.relative_to(PROJECT_ROOT) if output.is_relative_to(PROJECT_ROOT) else output), **receipt,
                        "cv_mae_kw": metrics["cv"]["model"]["mae_kw"], "cv_rmse_kw": metrics["cv"]["model"]["rmse_kw"],
                        "fold_mae_kw": [fold["model"]["mae_kw"] for fold in metrics["cv"]["folds"]]})
        table = pd.read_csv(output / "trials.csv")
        table["model_family"] = model_type
        tables.append(table)
    decision = select_candidate([r["run_id"] for r in records], baseline["run_id"], store_dir=store_dir)
    winner = next(r for r in records if r["run_id"] == decision["selected_run_id"])
    all_trials = pd.concat(tables, ignore_index=True)
    all_trials.to_csv(evidence / "all_trials.csv", index=False)
    write_json(evidence / "selection_decision.json", decision)
    client = configure_store(store_dir)
    for path in evidence.iterdir():
        client.log_artifact(winner["run_id"], str(path), artifact_path="lecture")
    for record in records:
        version = register_run(record["run_id"], name=experiment, store_dir=store_dir,
                               reason="Selected model-family configuration from chronological development CV; see selection_decision.json.")
        record["registered_version"] = str(version.version)
    events = [assign_alias(experiment, winner["registered_version"], reason=decision["rule"], store_dir=store_dir)]
    history = PROJECT_ROOT / winner["output_dir"] / "history.csv"
    prediction = predict_next(history, name=experiment, store_dir=store_dir)
    rollback = None
    if demonstrate_rollback:
        other = next(r for r in records if r is not winner)
        try:
            events.append(assign_alias(experiment, other["registered_version"],
                                       reason="Illustrative local rollback to a comparison version; no production service exists.", store_dir=store_dir))
            rollback = predict_next(history, name=experiment, store_dir=store_dir)
        finally:
            events.append(assign_alias(experiment, winner["registered_version"],
                                       reason="Restore the validation-selected champion after the local rollback illustration.", store_dir=store_dir))
        restored = predict_next(history, name=experiment, store_dir=store_dir)
        assert restored["model_uri"] == prediction["model_uri"]
        np.testing.assert_allclose(restored["predicted_kw"], prediction["predicted_kw"], rtol=1e-9, atol=1e-9)
    result = {"batch_id": batch, "experiment": experiment, "evidence_dir": str(evidence.relative_to(PROJECT_ROOT)),
              "baseline": baseline, "models": records, "selection": decision,
              "prediction": prediction, "rollback_prediction": rollback, "alias_events": events}
    write_json(evidence / "evidence.json", result)
    write_json(PROJECT_ROOT / "outputs/lecture_latest.json", result)
    client.log_artifact(winner["run_id"], str(evidence / "evidence.json"), artifact_path="lecture")
    print(f"Lecture evidence: {evidence}\nSelected run: {winner['run_id']}\nModel: {prediction['model_uri']}")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE))
    parser.add_argument("--experiment", default=LECTURE_EXPERIMENT)
    parser.add_argument("--forest-trials", type=int)
    parser.add_argument("--demonstrate-rollback", action="store_true")
    args = parser.parse_args()
    run_demo(store_dir=args.store_dir, experiment=args.experiment, forest_trials=args.forest_trials,
             demonstrate_rollback=args.demonstrate_rollback)


if __name__ == "__main__":
    main()
