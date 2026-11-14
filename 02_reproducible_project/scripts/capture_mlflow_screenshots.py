"""Capture the MLflow UI views used in the Presenter 2 slides from a running local MLflow UI.

Start the UI first (from 02_reproducible_project):
    uv run --locked --extra tuning mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db --host 127.0.0.1 --port 5050
Then, using the locally installed Google Chrome:
    uv run --with playwright python scripts/capture_mlflow_screenshots.py              # all views
    uv run --with playwright python scripts/capture_mlflow_screenshots.py compare 1500 # one view, custom height
    SHOT_WIDTH=820 uv run --with playwright python scripts/capture_mlflow_screenshots.py folds search params 640  # chart views for the slides
Screenshots go to docs/presenter_2/assets/mlflow_screenshots/. Run IDs come from outputs/lecture_latest.json.
"""

import json, os, sys, urllib.parse
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SHOTS = PROJECT.parent / "docs" / "presenter_2" / "assets" / "mlflow_screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)
from playwright.sync_api import sync_playwright
B = "http://127.0.0.1:5050"
E = json.load(open(PROJECT / "outputs" / "lecture_latest.json"))
batch = E["batch_id"]; ridge, forest = E["models"][0]["run_id"], E["models"][1]["run_id"]; base = E["baseline"]["run_id"]
forest_model = E["models"][1]["model_id"]
EXP = json.load(open(PROJECT / E["models"][1]["output_dir"] / "tracking.json"))["experiment_id"]
cols = ["params.`model.alpha`", "params.`model.max_depth`", "params.`model.min_samples_leaf`", "params.`model.max_features`", "metrics.`cv_mae_kw`"]
state = {"searchFilter": f"tags.lecture_batch = '{batch}'", "orderByKey": "attributes.start_time", "orderByAsc": False, "startTime": "ALL",
         "lifecycleFilter": "Active", "datasetsFilter": [], "modelVersionFilter": "All Runs", "selectedColumns": cols, "columnOrder": [],
         "columnWidths": {}, "runsExpanded": {forest: True, ridge: True}, "runsPinned": [], "runsHidden": [], "runsVisibilityMap": {},
         "runsHiddenMode": "FIRST_10_RUNS", "viewMaximized": False, "runListHidden": False, "isAccordionReordered": False,
         "useGroupedValuesInCharts": True, "hideEmptyCharts": True, "groupBy": None, "groupsExpanded": {}, "autoRefreshEnabled": False,
         "globalLineChartConfig": {"xAxisKey": "step", "lineSmoothness": 0, "selectedXAxisMetricKey": ""}}
init = {
    f'MLflowLocalStorage-1.1-ExperimentPage-["{EXP}"]-ReactComponentState': json.dumps(state),
    "mlflow.workflowType_v1": '"machine_learning"', "_mlflow_dark_mode_toggle_enabled": "true", "databricks-dark-mode-pref": "dark",
    "mlflow.assistant.fab.autoOpenedOnFirstLoad_v1": "true", "mlflow.assistant.panelOpen_v1": "false",
}
script = "(() => { const v = %s; for (const k in v) localStorage.setItem(k, v[k]); })();" % json.dumps(init)
q = urllib.parse.quote
pages = {
  "runs": f"/#/experiments/{EXP}/runs?workflowType=machine_learning&searchFilter={q(state['searchFilter'])}&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs",
  "compare": f"/#/compare-runs?runs={q(json.dumps([forest, ridge, base]))}&experiments={q(json.dumps([EXP]))}",
  "run_overview": f"/#/experiments/{EXP}/runs/{forest}",
  "run_artifacts": f"/#/experiments/{EXP}/runs/{forest}/artifacts",
  "model": f"/#/experiments/{EXP}/models/{forest_model}/artifacts",
  "registry": "/#/models/electricity-mt158-lecture",
  "version": "/#/models/electricity-mt158-lecture/versions/2",
}
trials_filter = f"tags.lecture_batch = '{batch}' and tags.run_kind = 'trial' and tags.model_family = 'random_forest'"
parents_filter = f"tags.lecture_batch = '{batch}' and tags.run_kind != 'trial'"
forest_parent_filter = f"tags.lecture_batch = '{batch}' and tags.run_kind = 'study' and tags.model_family = 'random_forest'"
chart_views = {  # name: (run filter, chart type or None for an existing chart, params, metrics, chart to open full screen)
    "folds": (parents_filter, None, [], [], "fold_mae_kw"),
    "search": (forest_parent_filter, "Line chart", [], ["trial_objective_mae_kw", "best_so_far_mae_kw"], None),
    "params": (trials_filter, "Parallel coordinates", ["model.max_depth", "model.min_samples_leaf", "model.max_features"], ["cv_mae_kw"], None),
}
for name, (flt, *_) in chart_views.items():
    pages[name] = (f"/#/experiments/{EXP}/runs?workflowType=machine_learning&searchFilter={q(flt)}"
                   "&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active"
                   "&modelVersionFilter=All+Runs&compareRunsMode=CHART")


def chart_state(flt):
    st = dict(state, searchFilter=flt, runsExpanded={forest: True, ridge: True}, runsHiddenMode="SHOW_ALL")
    return "(() => { localStorage.setItem(%s, %s); })();" % (
        json.dumps(f'MLflowLocalStorage-1.1-ExperimentPage-["{EXP}"]-ReactComponentState'), json.dumps(json.dumps(st)))


def pick(pg, box, values):
    for v in values:
        box.click(); pg.wait_for_timeout(400); pg.keyboard.type(v); pg.wait_for_timeout(700)
        opts = pg.get_by_role("option", name=v, exact=True)
        visible = [opts.nth(i) for i in range(opts.count()) if opts.nth(i).is_visible()]
        if visible:
            visible[0].click()
        else:
            pg.keyboard.press("Enter")  # virtualised list: Enter selects the highlighted match
        pg.wait_for_timeout(400)
        pg.keyboard.press("Escape"); pg.wait_for_timeout(300)


only = [a for a in sys.argv[1:] if not a.isdigit()] or list(pages)
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome", headless=True)
    ctx = b.new_context(viewport={"width": int(os.environ.get("SHOT_WIDTH", 1440)), "height": int(sys.argv[-1]) if sys.argv[-1].isdigit() else 900}, device_scale_factor=2, color_scheme="dark")
    ctx.add_init_script(script)
    pg = ctx.new_page()
    for name in only:
        if name in chart_views:
            pg.goto(B + "/", wait_until="load"); pg.evaluate(chart_state(chart_views[name][0]))
        pg.goto(B + pages[name], wait_until="load"); pg.wait_for_timeout(5000)
        if name in chart_views:
            flt, kind, params, metrics, existing = chart_views[name]
            if kind:  # build the chart in MLflow's "Add chart" dialog and add it to the chart grid
                pg.get_by_role("button", name="Add chart").first.click(); pg.wait_for_timeout(1000)
                pg.get_by_role("menuitem", name=kind).first.click(); pg.wait_for_timeout(1500)
                dlg = pg.locator("[role=dialog]").first
                if kind == "Line chart":
                    y_metric = dlg.locator("input[role=combobox][type=search]").nth(1)
                    y_metric.click(); pg.wait_for_timeout(300); pg.keyboard.press("Backspace")  # drop the default metric
                    pick(pg, y_metric, metrics)
                else:
                    boxes = dlg.locator("[role=combobox], input")
                    n = boxes.count()
                    pick(pg, boxes.nth(n - 2), params)
                    pick(pg, boxes.nth(n - 1), metrics)
                pg.wait_for_timeout(800)
                dlg.get_by_role("button", name="Add chart").click(); pg.wait_for_timeout(2500)
                # Added charts are titled after their metrics; open the new one (not the default per-metric chart).
                pg.get_by_placeholder("Search metric charts").first.fill(metrics[0]); pg.wait_for_timeout(2000)
                buttons = pg.locator("[data-component-id=fullscreen_button_chartcard]")
                titles = buttons.evaluate_all("bs => bs.map(b => { let c = b; for (let i = 0; i < 3; i++) c = c.parentElement; return c.innerText.split('\\n')[0]; })")
                default = metrics[0]
                pick_index = next(i for i, t in enumerate(titles) if t != default) if kind == "Line chart" else len(titles) - 1
                buttons.nth(pick_index).click()
            else:  # open an existing metric chart full screen
                pg.get_by_placeholder("Search metric charts").first.fill(existing); pg.wait_for_timeout(1000)
                vp = pg.viewport_size  # charts are virtualised: render the whole grid, then restore the window
                pg.set_viewport_size({"width": vp["width"], "height": 2400}); pg.wait_for_timeout(2000)
                buttons = pg.locator("[data-component-id=fullscreen_button_chartcard]")
                titles = buttons.evaluate_all("bs => bs.map(b => { let c = b; for (let i = 0; i < 3; i++) c = c.parentElement; return c.innerText.split('\\n')[0]; })")
                if existing not in titles:
                    raise SystemExit(f"chart {existing!r} not found; visible charts: {titles}")
                buttons.nth(titles.index(existing)).click(); pg.wait_for_timeout(800)
                pg.set_viewport_size(vp)
            pg.wait_for_timeout(2500)
            pg.mouse.move(5, 700); pg.wait_for_timeout(800)
        if name == "run_artifacts":
            try: pg.get_by_text("training", exact=True).first.click(); pg.wait_for_timeout(1200)
            except Exception as e: print("expand", e)
        if name == "model":
            try: pg.get_by_text("MLmodel", exact=True).first.click(); pg.wait_for_timeout(1500)
            except Exception as e: print("mlmodel", e)
        if name == "model":
            el = pg.get_by_text("signature", exact=False).last
            try:
                el.scroll_into_view_if_needed(); pg.evaluate("""() => { const n=[...document.querySelectorAll('*')].find(e=>e.textContent.trim()==='signature'||e.textContent.trim()==='signature:'); if(n){ let p=n; while(p && p.scrollHeight<=p.clientHeight+2) p=p.parentElement; if(p) p.scrollTop = n.getBoundingClientRect().top - p.getBoundingClientRect().top + p.scrollTop - 120; } }"""); pg.wait_for_timeout(800)
            except Exception as e: print("scroll", e)
        if name == "compare":
            pg.get_by_text("Visualizations", exact=True).first.click(); pg.wait_for_timeout(800)
            pg.get_by_text("Parameters", exact=True).first.click(); pg.wait_for_timeout(800)
            pg.mouse.move(200, 1000); pg.wait_for_timeout(500)
        pg.screenshot(path=str(SHOTS / f"{name}.png"))
        print(name, pg.url[:110])
    b.close()
