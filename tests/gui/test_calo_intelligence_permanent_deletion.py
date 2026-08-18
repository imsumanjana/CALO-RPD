from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest
import torch

from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.policy_schema import (
    CALO_RUNTIME_ARCHITECTURE,
    POLICY_ACTION_SCHEMA,
    POLICY_STATE_DIM,
    POLICY_STATE_SCHEMA,
    TRAINING_ENVIRONMENT_VERSION,
)
from tests.gui.test_phase6_ribbon_workspace import _training_plan, _window


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None, reason="PyQt6 is not installed"
)


def test_stale_active_assessed_model_is_hidden_as_obsolete_then_can_be_deleted(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    state, window = _window(qtbot, tmp_path, monkeypatch)
    panel = window.pages_by_key["calo_intelligence"]
    campaign = window.training_model_library.default_directory / "old-incompatible-policy"
    campaign.mkdir(parents=True)
    candidate = campaign / "old-policy.candidate.pt"
    network = CALOPolicyNetwork(input_dim=POLICY_STATE_DIM, hidden_dim=16)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": 16},
            "metadata": {
                "calo_core": "v4.1",
                "state_dimension": POLICY_STATE_DIM,
                "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
                "state_schema_version": POLICY_STATE_SCHEMA,
                "action_schema_version": POLICY_ACTION_SCHEMA,
                "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
            },
        },
        candidate,
    )
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    plan_payload = json.loads(_training_plan(tmp_path).read_text(encoding="utf-8"))
    plan_payload["campaign_id"] = "old-incompatible-policy"
    (campaign / "training_plan.json").write_text(json.dumps(plan_payload), encoding="utf-8")
    (campaign / "training_status.json").write_text(
        json.dumps({"state": "completed"}), encoding="utf-8"
    )
    (campaign / "training_manifest.json").write_text(
        json.dumps(
            {
                "state": "completed_unqualified",
                "ensemble_candidate": {"path": candidate.name, "sha256": candidate_sha256},
            }
        ),
        encoding="utf-8",
    )
    registered = state.policy_registry.register(candidate, name="old-incompatible-policy")
    state.database.add_policy_qualification(
        qualification_id="old-assessment",
        policy_id=registered.id,
        config={"retained": True},
        metrics={"score": 50.0},
        passed=False,
        grade="U",
        score=50.0,
        qualification_status="assessed",
    )
    state.database.set_active_policy(registered.id)
    window.training_model_library.changed.emit()

    assert panel.show_obsolete_models.isChecked() is False
    assert all(
        panel.policy_table.item(row, 1).text() != "old-incompatible-policy"
        for row in range(panel.policy_table.rowCount())
    )
    panel.show_obsolete_models.setChecked(True)

    row = next(
        index
        for index in range(panel.policy_table.rowCount())
        if panel.policy_table.item(index, 1).text() == "old-incompatible-policy"
    )
    panel.policy_table.selectRow(row)
    panel._policy_selection_changed()

    assert panel.policy_table.horizontalHeaderItem(0).text() == "Use status"
    assert panel.policy_table.item(row, 0).text() == "Not governing"
    assert panel.policy_table.item(row, 4).text() == "Not compatible"
    assert panel.policy_table.item(row, 6).text() == "Not compatible"
    assert panel.policy_activate_button.text() == "Not currently governing"
    assert panel.policy_activate_button.isEnabled() is False
    assert "not compatible" in panel.qualification_workflow_status.text().lower()
    assert panel.policy_delete_button.isEnabled() is True
    assert panel.path.text() == ""
    assert "not compatible" in panel.policy_gate_status.text().lower()
    assert panel.apply_policy_button.isEnabled() is False

    warnings = []

    def confirm(_parent, title, text, *_args, **_kwargs):
        warnings.append((title, text))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "warning", confirm)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    panel.delete_selected_model_files()

    assert warnings
    assert "cannot be undone" in warnings[0][1].lower()
    assert "historical provenance" in warnings[0][1].lower()
    assert campaign.exists() is False
    archived = state.policy_registry.get(registered.id)
    assert archived.active is False
    assert archived.archived is True
    assert archived.usable is False
    assert len(state.database.list_policy_qualifications(registered.id)) == 1
    assert state.governing_policy_status().ready is False
