"""Register an explicitly selected run and optionally assign a model alias."""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime, timezone

import mlflow

from .tracking import DEFAULT_EXPERIMENT, DEFAULT_STORE, configure_store
from .config import project_path, write_json


def assign_alias(name, version, *, alias="champion", reason, store_dir=DEFAULT_STORE):
    """Move a local alias and retain an explicit old/new event for the demo."""
    if not reason.strip():
        raise ValueError("Record a reason for changing the alias.")
    client = configure_store(store_dir)
    selected = client.get_model_version(name, str(version))
    source = client.get_run(selected.run_id)
    if source.info.status != "FINISHED" or source.data.tags.get("model_roundtrip_verified") != "true":
        raise ValueError("Alias target must come from a completed, round-trip-verified run.")
    previous = client.get_registered_model(name).aliases.get(alias)
    client.set_registered_model_alias(name, alias, str(version))
    event = {"name": name, "alias": alias, "previous_version": previous, "new_version": str(version),
             "source_run_id": selected.run_id, "reason": reason,
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "scope": "Local model-reference change; no running service is deployed or reloaded."}
    path = project_path(store_dir) / "alias_events" / f"{uuid.uuid4().hex}.json"
    write_json(path, event)
    client.log_artifact(selected.run_id, str(path), artifact_path="registry_events")
    return event


def register_run(run_id: str, *, name: str = DEFAULT_EXPERIMENT, reason: str,
                 alias: str | None = None, store_dir=DEFAULT_STORE):
    if not reason.strip():
        raise ValueError("Record the validation-based selection reason.")
    client = configure_store(store_dir)
    run = client.get_run(run_id)
    if run.info.status != "FINISHED" or run.data.tags.get("model_roundtrip_verified") != "true":
        raise ValueError("Select a completed tracked run with a verified model round trip.")
    registered = mlflow.register_model(run.data.tags["model_uri"], name)
    client.set_model_version_tag(name, registered.version, "selection_reason", reason)
    client.set_model_version_tag(name, registered.version, "comparison_group", run.data.tags["comparison_group"])
    client.set_model_version_tag(name, registered.version, "cv_mae_kw", str(run.data.metrics["cv_mae_kw"]))
    if "selected_trial_run_id" in run.data.tags:
        client.set_model_version_tag(name, registered.version, "selected_trial_run_id", run.data.tags["selected_trial_run_id"])
    if alias:
        assign_alias(name, registered.version, alias=alias, reason=reason, store_dir=store_dir)
    print(f"Registered: models:/{name}/{registered.version}")
    if alias:
        print(f"Alias: models:/{name}@{alias}")
    return registered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--alias")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE))
    args = parser.parse_args()
    register_run(args.run_id, name=args.name, reason=args.reason, alias=args.alias, store_dir=args.store_dir)


if __name__ == "__main__":
    main()
