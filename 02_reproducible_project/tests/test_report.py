import pytest

from electricity_demo.config import write_json
from electricity_demo.report import update_comparison


def make_run(root, name, group, model, score, *, params=None, best_params=None, final_test=False):
    folder = root / name
    folder.mkdir()
    write_json(folder / "manifest.json", {
        "comparison_group": group, "created_utc": name, "model_type": model,
        "model_config": {"type": model, "params": params or {}}, "elapsed_seconds": 1.0,
        "dataset": {"rows": 1000, "sha256": group, "preparation": {"snapshot": "v1"}},
    })
    write_json(folder / "metrics.json", {
        "final_test_evaluated": final_test,
        "best_params": best_params or {},
        "cv": {"model": {"mae_kw": score, "rmse_kw": score + 1}, "std_fold_mae_kw": 0.1, "seasonal_baseline": {"mae_kw": 10}},
    })


def test_comparison_excludes_other_data_protocols_and_historical_duplicates(tmp_path):
    make_run(tmp_path, "01", "old-data", "random_forest", 0.1)
    make_run(tmp_path, "02", "same-data", "ridge", 5)
    make_run(tmp_path, "03", "same-data", "ridge", 4)
    make_run(tmp_path, "04", "same-data", "random_forest", 3)
    # Historical models outside the demo must not change the active comparison group.
    make_run(tmp_path, "05", "archived-data", "archived_model", 0.01)
    result = update_comparison(tmp_path)
    assert result["model_type"].tolist() == ["random_forest", "ridge"]
    assert result["cv_mae_kw"].tolist() == [3, 4]
    assert (tmp_path / "comparison.md").exists()
    assert (tmp_path / "comparison.png").exists()


@pytest.mark.parametrize("new_alpha, expected_run", [(10.0, "02"), (0.1, "01")])
def test_test_result_preference_uses_fitted_settings(tmp_path, new_alpha, expected_run):
    make_run(tmp_path, "01", "same", "ridge", 9,
             params={"alpha": 1.0}, best_params={"model__alpha": 0.1}, final_test=True)
    make_run(tmp_path, "02", "same", "ridge", 4,
             params={"alpha": 1.0}, best_params={"model__alpha": new_alpha})
    result = update_comparison(tmp_path)
    assert result["run"].tolist() == [expected_run]
