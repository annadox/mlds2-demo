# Presenter 1: From Notebook to Reproducible ML

23 minutes (lecture clock 0:00–23:00). The presentation is [mlds2-part1.pptx](mlds2-part1.pptx), with [mlds2-part1.pdf](mlds2-part1.pdf) as the offline copy (it includes the backup): a section opener, ten main slides and one hidden backup. It uses the same storyboard design as Part 2, and every code excerpt is taken from this repository.

The audience already knows modelling, cross-validation and leakage. This section teaches the engineering that lets someone else run, test and reuse the training code. [SLIDE_MANUSCRIPT.md](SLIDE_MANUSCRIPT.md) has the slide text and speaker notes for every slide.

## Slide flow

| Slide | Time | Teaching objective | Repository evidence |
| --- | --- | --- | --- |
| 1. The notebook that worked once | 0–3 | Notebook problems are software problems | `01_notebook` cells: data mutation, scaler bundle, unseeded split, overwritten model |
| 2. Library first: notebooks and scripts are thin clients | 3–6.5 | One package; notebook, CLI, tests and tracking call it | `src/electricity_demo/` layout |
| 3. Keep fitted preprocessing with the model | 6.5–8.5 | One fitted object from features to predictions | `pipeline.py` vs notebook bundle |
| 4. One feature definition for training and inference | 8.5–10.5 | Avoid training-serving skew | `features.feature_table`, parity test |
| 5. Dependencies: consumer vs training | 10.5–12.5 | Declare, lock, separate runtime from training tools; limits of a lockfile | `pyproject.toml`, `uv.lock` |
| 6. Configuration instead of hard-coded values | 12.5–14.5 | Validated, versioned config saved with every run | `configs/*.yaml`, `config.load_training_config` |
| 7. Version control: code in Git, data by hash | 14.5–16 | What goes in Git vs not; pull requests with CI and review; immutable raw data, hash manifest, DVC pointer | `.gitignore` files, `prepare.py`, `data/README.md` |
| 8. Testing ML code: contracts, not accuracy | 16–20 | What to test; CI; tests vs quality gates | `tests/`, `.github/workflows/tests.yml` |
| 9. One headless command | 20–22 | Same function for terminal, notebook, CI, scheduler | `train.py`, run folder, manifest |
| 10. A few experiments later: a folder of runs | 22–23 | Hand-off: reproducible runs, but no way to manage them | `outputs/`, `report.py` |
| A (hidden) | — | Evaluation protocol in code, for questions | `split.py`, tests |

Checkpoints: slide 2 done by minute 6:30, slide 5 by 12:30, slide 8 by 20, slide 10 by 23. If running over, shorten slide 7 and the audience prompt on slide 2; keep the testing slide intact.

## Handoff to Part 2

Close with: "We can reproduce a training execution and evaluate it correctly. When we try more models and settings, how do we compare those executions and find the exact model we want to use?" Part 2 opens on the same `outputs/` folder ("We already built a tracker by hand"), so do not show MLflow here.

The commands shown on the slides are in [../../02_reproducible_project/README.md](../../02_reproducible_project/README.md).
