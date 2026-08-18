from __future__ import annotations

import importlib.util
import json

import pytest

from tests.gui.test_phase6_ribbon_workspace import _training_plan, _window


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None, reason="PyQt6 is not installed"
)


def _write_campaign(directory, plan_source, *, campaign_id: str, status_text: str) -> None:
    directory.mkdir(parents=True)
    plan = json.loads(plan_source.read_text(encoding="utf-8"))
    plan["campaign_id"] = campaign_id
    (directory / "training_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (directory / "training_status.json").write_text(status_text, encoding="utf-8")


def _policy_row(panel, name: str) -> int:
    return next(
        row
        for row in range(panel.policy_table.rowCount())
        if panel.policy_table.item(row, 1).text() == name
    )


def test_interrupted_saved_training_opens_central_obsolete_library_and_can_be_deleted(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    panel = window.pages_by_key["calo_intelligence"]
    editor = window.context_pane.training
    campaign = window.training_model_library.default_directory / "interrupted-policy"
    _write_campaign(
        campaign,
        _training_plan(tmp_path),
        campaign_id="interrupted-policy",
        status_text=json.dumps({"state": "interrupted"}),
    )
    window.training_model_library.changed.emit()

    assert panel.show_obsolete_models.isChecked() is False
    assert all(
        panel.policy_table.item(row, 1).text() != "interrupted-policy"
        for row in range(panel.policy_table.rowCount())
    )

    picker_index = next(
        index
        for index in range(1, editor.library_picker.count())
        if isinstance(editor.library_picker.itemData(index), dict)
        and editor.library_picker.itemData(index).get("directory") == str(campaign.resolve())
    )
    editor.library_picker.setCurrentIndex(picker_index)
    assert editor.manage_saved_button.text() == "View / delete files"
    assert editor.manage_saved_button.isEnabled() is True

    qtbot.mouseClick(editor.manage_saved_button, Qt.MouseButton.LeftButton)

    assert panel.show_obsolete_models.isChecked() is True
    row = _policy_row(panel, "interrupted-policy")
    assert panel.policy_table.currentRow() == row
    assert panel.policy_table.item(row, 4).text() == "Interrupted training"
    assert panel.policy_table.item(row, 6).text() == "Not usable"
    assert panel.policy_import_button.isEnabled() is False
    assert panel.qualification_button.isEnabled() is False
    assert panel.policy_select_button.isEnabled() is False
    assert panel.policy_activate_button.isVisible() is False
    assert panel.policy_delete_button.isEnabled() is True

    warnings = []

    def confirm(_parent, title, text, *_args, **_kwargs):
        warnings.append((title, text))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "warning", confirm)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    panel.delete_selected_model_files()

    assert warnings
    assert "recovery point" in warnings[0][1].lower()
    assert "cannot be undone" in warnings[0][1].lower()
    assert campaign.exists() is False
    window.training_model_library.refresh()
    assert all(
        not isinstance(editor.library_picker.itemData(index), dict)
        or editor.library_picker.itemData(index).get("directory") != str(campaign.resolve())
        for index in range(editor.library_picker.count())
    )


def test_corrupt_saved_training_is_hidden_by_default_and_deletable_when_obsolete_is_shown(
    qtbot, tmp_path, monkeypatch
):
    from PyQt6.QtWidgets import QMessageBox

    _state, window = _window(qtbot, tmp_path, monkeypatch)
    panel = window.pages_by_key["calo_intelligence"]
    campaign = window.training_model_library.default_directory / "corrupt-policy"
    _write_campaign(
        campaign,
        _training_plan(tmp_path),
        campaign_id="corrupt-policy",
        status_text="{not-json",
    )
    window.training_model_library.changed.emit()

    assert all(
        panel.policy_table.item(row, 1).text() != "corrupt-policy"
        for row in range(panel.policy_table.rowCount())
    )
    panel.show_obsolete_models.setChecked(True)

    row = _policy_row(panel, "corrupt-policy")
    panel.policy_table.selectRow(row)
    panel._policy_selection_changed()
    assert panel.policy_table.item(row, 4).text() == "Corrupted training status"
    assert panel.policy_delete_button.isEnabled() is True

    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    panel.delete_selected_model_files()

    assert campaign.exists() is False


def test_registered_model_with_changed_file_integrity_can_be_permanently_deleted(
    qtbot, tmp_path, monkeypatch
):
    import torch
    from PyQt6.QtWidgets import QMessageBox

    from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
    from calo_rpd_studio.algorithms.calo.policy_schema import (
        CALO_RUNTIME_ARCHITECTURE,
        POLICY_ACTION_SCHEMA,
        POLICY_STATE_DIM,
        POLICY_STATE_SCHEMA,
        TRAINING_ENVIRONMENT_VERSION,
    )

    state, window = _window(qtbot, tmp_path, monkeypatch)
    panel = window.pages_by_key["calo_intelligence"]
    candidate = tmp_path / "registered-corrupt-model.pt"
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
    registered = state.policy_registry.register(candidate, name="registered-corrupt-model")
    with candidate.open("ab") as stream:
        stream.write(b"changed-after-registration")
    panel.refresh_policy_library()

    row = _policy_row(panel, "registered-corrupt-model")
    panel.policy_table.selectRow(row)
    panel._policy_selection_changed()
    assert panel.policy_delete_button.isEnabled() is True

    warnings = []

    def confirm(_parent, title, text, *_args, **_kwargs):
        warnings.append((title, text))
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "warning", confirm)
    monkeypatch.setattr(QMessageBox, "information", lambda *_args, **_kwargs: None)
    panel.delete_selected_model_files()

    assert warnings
    assert "cannot verify" in warnings[0][1].lower()
    assert "cannot be undone" in warnings[0][1].lower()
    assert candidate.exists() is False
    archived = state.policy_registry.get(registered.id)
    assert archived.active is False
    assert archived.archived is True
