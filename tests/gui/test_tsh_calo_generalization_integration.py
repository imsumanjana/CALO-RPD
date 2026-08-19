"""Rendered UI and CLI integration regressions for TSH-CALO learning-health accounting."""

from __future__ import annotations

import importlib.util
import json
import os

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None, reason="PyQt6 is not installed"
)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _new_model(monkeypatch):
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingLaunchModel

    monkeypatch.setattr(
        TrainingLaunchModel,
        "_current_source_commit",
        staticmethod(lambda: "a" * 40),
    )
    model = TrainingLaunchModel()
    model.create_plan(
        campaign_id="guarded-gui-plan",
        development_cases=["toy-development"],
        member_count=2,
        master_seed=101,
        population_size=4,
        max_evaluations=8,
        requested_device="cpu",
        allow_cpu_fallback=False,
        training={},
    )
    assert model.plan_error == ""
    assert model.plan_payload is not None
    return model


def test_new_gui_plan_enables_guard_but_exact_saved_identity_is_preserved(
    tmp_path, monkeypatch
):
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingLaunchModel

    model = _new_model(monkeypatch)
    assert model.plan_payload["generalization_guard"]["enabled"] is True

    legacy = dict(model.plan_payload)
    legacy.pop("generalization_guard")
    plan_path = tmp_path / "legacy-plan.json"
    plan_path.write_text(json.dumps(legacy), encoding="utf-8")

    resumed = TrainingLaunchModel()
    resumed.set_value("plan", str(plan_path))
    resumed.load_plan(preserve_identity=True)
    assert resumed.plan_error == ""
    assert "generalization_guard" not in resumed.plan_payload

    imported = TrainingLaunchModel()
    imported.set_value("plan", str(plan_path))
    imported.load_plan(preserve_identity=False)
    assert imported.plan_error == ""
    assert imported.plan_payload["generalization_guard"]["enabled"] is True


def test_rendered_training_panel_discloses_guard_and_uses_counted_completion_total(
    qtbot, monkeypatch
):
    from PyQt6.QtCore import QObject, pyqtSignal

    from calo_rpd_studio.algorithms.calo.tsh_calo_evaluation_accounting import (
        plan_training_evaluation_accounting,
    )
    from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
        TSHCALOTrainingCampaignPlan,
    )
    from calo_rpd_studio.gui.panels.independent_training_panel import (
        IndependentTrainingPanel,
        TRAINING_EVENT_SCHEMA,
    )

    class TaskStatus(QObject):
        cancel_requested = pyqtSignal()

        def __init__(self):
            super().__init__()
            self.busy = False
            self.detail = ""
            self.progress = 0

        def update(self, *, progress=None, detail=None, **_kwargs):
            if progress is not None:
                self.progress = int(progress)
            if detail is not None:
                self.detail = str(detail)

        def begin(self, *_args, **_kwargs):
            self.busy = True

        def cancel(self, *_args, **_kwargs):
            self.cancel_requested.emit()

        def rearm_cancel(self, *_args, **_kwargs):
            pass

    class State:
        def __init__(self):
            self.task_status = TaskStatus()
            self.policy_training_active = False

        def begin_policy_training(self, *_args, **_kwargs):
            self.policy_training_active = True

        def end_policy_training(self, *_args, **_kwargs):
            self.policy_training_active = False

    model = _new_model(monkeypatch)
    plan = TSHCALOTrainingCampaignPlan.from_dict(model.plan_payload)
    accounting = plan_training_evaluation_accounting(plan)
    state = State()
    panel = IndependentTrainingPanel(state, model)
    qtbot.addWidget(panel)

    summary = panel.plan_summary.text()
    preview = panel.command_preview.toPlainText()
    assert "Learning-health check: enabled" in summary
    assert str(accounting.generalization_guard_candidate_evaluations) in summary
    assert str(accounting.total_counted_candidate_evaluations) in preview

    panel._apply_training_progress(
        {
            "schema_version": TRAINING_EVENT_SCHEMA,
            "event": "campaign_completed",
            "total_candidate_evaluations": accounting.training_candidate_evaluations,
            "total_counted_candidate_evaluations": (
                accounting.total_counted_candidate_evaluations
            ),
            "progress_percent": 100,
        }
    )
    assert str(accounting.total_counted_candidate_evaluations) in state.task_status.detail
    assert (
        panel.last_progress_event["total_candidate_evaluations"]
        == accounting.total_counted_candidate_evaluations
    )


def test_saved_campaign_library_exposes_latest_cumulative_counted_work(tmp_path, qtbot):
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingModelLibrary

    class Settings:
        @staticmethod
        def value(_key, default=None):
            return default

        @staticmethod
        def set_value(_key, _value):
            pass

        @staticmethod
        def sync():
            pass

    root = tmp_path / "models"
    campaign = root / "campaign"
    segment = campaign / "extensions" / "segment-000001"
    segment.mkdir(parents=True)
    (campaign / "training_plan.json").write_text(
        json.dumps({"campaign_id": "saved-guarded"}), encoding="utf-8"
    )
    (campaign / "training_status.json").write_text(
        json.dumps({"state": "completed", "progress": {}}), encoding="utf-8"
    )
    (campaign / "training_manifest.json").write_text(
        json.dumps(
            {
                "state": "completed_unqualified",
                "evaluation_accounting": {
                    "training_candidate_evaluations": 16,
                    "generalization_guard_candidate_evaluations": 96,
                    "total_counted_candidate_evaluations": 112,
                },
            }
        ),
        encoding="utf-8",
    )
    (segment / "training_status.json").write_text(
        json.dumps({"state": "completed", "progress": {}}), encoding="utf-8"
    )
    (segment / "training_manifest.json").write_text(
        json.dumps(
            {
                "state": "completed_unqualified_extension",
                "evaluation_accounting": {
                    "segment": {
                        "training_candidate_evaluations": 16,
                        "generalization_guard_candidate_evaluations": 96,
                        "total_counted_candidate_evaluations": 112,
                    },
                    "cumulative": {
                        "training_candidate_evaluations": 32,
                        "generalization_guard_candidate_evaluations": 192,
                        "total_counted_candidate_evaluations": 224,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    # The qtbot fixture supplies the QApplication required by the library's QObject base.
    assert qtbot is not None
    library = TrainingModelLibrary(Settings(), default_directory=root)
    row = next(item for item in library.saved_campaigns() if item["campaign_id"] == "saved-guarded")
    assert row["training_evaluations"] == 32
    assert row["generalization_guard_evaluations"] == 192
    assert row["total_counted_evaluations"] == 224


def test_cli_machine_event_discloses_training_guard_and_total(monkeypatch):
    from calo_rpd_studio.algorithms.calo.tsh_calo_evaluation_accounting import (
        plan_training_evaluation_accounting,
    )
    from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
        TSHCALOTrainingCampaignPlan,
    )
    from calo_rpd_studio.scripts.train_tsh_calo import (
        TRAINING_EVENT_PREFIX,
        _accounted_output_text,
    )

    model = _new_model(monkeypatch)
    plan = TSHCALOTrainingCampaignPlan.from_dict(model.plan_payload)
    accounting = plan_training_evaluation_accounting(plan)
    source = TRAINING_EVENT_PREFIX + json.dumps(
        {
            "schema_version": "tsh-calo-training-progress-event-v1",
            "event": "process_started",
            "total_candidate_evaluations": accounting.training_candidate_evaluations,
        }
    )
    transformed = _accounted_output_text(source, plan)
    payload = json.loads(transformed[len(TRAINING_EVENT_PREFIX) :])
    assert payload["total_training_candidate_evaluations"] == 16
    assert (
        payload["total_generalization_guard_candidate_evaluations"]
        == accounting.generalization_guard_candidate_evaluations
    )
    assert (
        payload["total_counted_candidate_evaluations"]
        == accounting.total_counted_candidate_evaluations
    )
    assert payload["legacy_candidate_evaluation_fields_are_training_only"] is True
