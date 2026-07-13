import json
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.gui.panels.results_explorer_panel import ResultsExplorerPanel


def test_results_explorer_selects_full_row_and_emits_validation_request(tmp_path):
    QApplication.instance() or QApplication([])
    state = AppState(tmp_path / "results.sqlite")
    exp_id = state.database.create_experiment(state.config, {})
    # Minimal direct record insertion through database connection for GUI selection behavior.
    run_id = "run-1"
    payload = {
        "best_objective": 1.23,
        "feasible": True,
        "total_constraint_violation": 0.0,
        "runtime_seconds": 0.1,
        "decoded_controls": {},
        "objective_components": {},
        "termination_reason": "test",
        "metadata": {},
    }
    with state.database.connect() as con:
        con.execute(
            "INSERT INTO runs(id,experiment_id,algorithm,run_index,seed_json,result_json,arrays_path) VALUES(?,?,?,?,?,?,?)",
            (run_id, exp_id, "CALO", 0, json.dumps({"algorithm_seed": 1}), json.dumps(payload), ""),
        )
    panel = ResultsExplorerPanel(state)
    panel.refresh_experiments()
    assert panel.table.currentRow() == 0
    assert panel.review_button.isEnabled()
    assert "run-1" in panel.details.toPlainText()
    captured = []
    panel.validation_requested.connect(lambda experiment_id, selected_run: captured.append((experiment_id, selected_run)))
    panel._confirm_review()
    assert captured == [(exp_id, run_id)]
