# Presenter 2: Experiment Tracking and Model Management with MLflow

25 minutes (lecture clock 23:00–48:00). The canonical presentation is [mlds2-lecture.pptx](mlds2-lecture.pptx), with [mlds2-lecture.pdf](mlds2-lecture.pdf) as the offline copy (it includes the backups): a section opener, eleven main slides and four hidden backups (A–D). The section teaches MLflow and experiment/model-management concepts; the electricity demo is supporting evidence, not the topic. The design follows the team storyboard (lifecycle navigation strip, cards, serif titles).

The deck before the 13 November redesign is kept as `mlds2-lecture-previous.pptx` / `.pdf`; delete it once the team has accepted the new version.

## Contents

| Material | Purpose |
| --- | --- |
| [SLIDE_MANUSCRIPT.md](SLIDE_MANUSCRIPT.md) | Slide copy, visuals, speaker notes and sources for every slide (generated from the deck) |
| [TEAM_HANDOFF_AND_REHEARSAL.md](TEAM_HANDOFF_AND_REHEARSAL.md) | Shared task contract, presenter transitions, Presenter 3 consumption command, rehearsal plan, likely questions |
| `evidence_snapshot.json` | Exact batch IDs, scores, lineage, prediction and figure hashes used in the slides |
| `assets/mlflow_screenshots/` | Native MLflow UI captures from 13 November (also zipped in `mlflow-captures.zip`); the slides use crops of these |
| `assets/*.png`, `assets/all_trials.csv` | Analysis figures and trial table generated from the recorded batch |

Recorded IDs and scores: [../MLFLOW_ENHANCED_RESULTS.md](../MLFLOW_ENHANCED_RESULTS.md). Regenerating the evidence and reading the UI: [../MLFLOW_DEMO.md](../MLFLOW_DEMO.md).

## Slide flow

| Slide | Time | Teaching objective | Main visual |
| --- | --- | --- | --- |
| 1. We already built a tracker by hand | 0–2 | Why a team needs tracking: our own run folders don't scale | Real `outputs/` folder vs team needs |
| 2. Where MLflow fits | 2–3.5 | Tracking and model management in the lifecycle; section boundary | Four lifecycle cards |
| 3. Experiments, runs and what a run records | 3.5–5.5 | Experiment vs run; parameters, metrics, artifacts, tags | Hierarchy plus four cards |
| 4. Adding MLflow to existing training | 5.5–8 | Computing vs recording; what to record; tracking as an adapter around training | Annotated code pattern |
| 5. Finding runs in the Tracking UI | 8–10 | Tag filters, parent/child search runs | `hierarchy.jpg` |
| 6. Selecting a model: compare, then decide in code | 10–12.5 | Comparable runs; a rerunnable selection rule | `run-comparison.jpg` plus `search_runs` code |
| 7. Provenance | 12.5–14.5 | Source, data, tags, artifacts; identity is not recovery | `run-inspection.jpg`, `run-artifacts.jpg` |
| 8. Run, logged model, registered version | 14.5–16 | Three objects, three questions | Three cards |
| 9. The model contract: what a consumer receives | 16–19 | Reference, package, input, output, environment; feature code inside vs outside the model | Contract card, `model-signature.jpg`, two options |
| 10. Registry and aliases: the hand-off boundary | 19–22 | Versions, lineage, load by alias, who may move the alias, reload semantics | `registry-teaching.jpg`, `registry-lineage.jpg`, code |
| 11. From a notebook model to a model others can use | 22–25 | Readiness checklist across Parts 1–3; handoff | Eight-item checklist plus question |
| A–D (hidden) | — | Forecasting evaluation, inference contract, error by hour, demo scope | For questions only |

Checkpoints: slide 4 done by minute 8:00, slide 7 by 14:30, slide 10 by 22:00, slide 11 by 25:00. Use saved screenshots on stage; no live training or registry change is needed. If running over, shorten the audience prompts on slides 3 and 5; keep selection as code, the model contract, the alias semantics and the readiness checklist.

## Notes

- Presenter 1's deck is in [../presenter_1/](../presenter_1/README.md); it ends on the same `outputs/` folder that slide 1 opens with. Do not re-introduce the dataset or cross-validation. Close with: "We can identify and retrieve a selected model version. How does an application use it, and how do we know it still works?"
- The slides use batch `20261112T181949Z-424799`. Later code fixes give new runs a different comparison identity; do not mix them into the recorded batch.
- The registered model shares its name with the experiment (`electricity-mt158-lecture`); slide 8 says so explicitly.
- The logged model's input is 29 engineered features, not raw meter history (slide 9). Serving it directly means sending those features; raw history goes through `features.build_next_features` first.
- The forest champion predicts **131.984094 kW** for the hour ending `2014-11-01 01:00`, from 168 observed hours on the published Portuguese clock.
- To open MLflow locally use `http://127.0.0.1:5050` with `--extra tuning` (see the demo guide).
