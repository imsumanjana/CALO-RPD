from __future__ import annotations

import importlib.util
import os

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None,
    reason="PyQt6 is not installed",
)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qt_application():
    from PyQt6.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    yield application


def test_experiment_workspace_has_only_two_scientist_compute_modes(qt_application, tmp_path):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.experiment_manager_panel import ExperimentManagerPanel

    state = AppState(tmp_path / "scientist-experiment.sqlite")
    panel = ExperimentManagerPanel(state, ExperimentManager(state))
    try:
        assert panel.execution_backend.count() == 2
        assert panel.execution_backend.findData("cuda_preferred") >= 0
        assert panel.execution_backend.findData("cpu_only") >= 0
        mode_text = " ".join(
            panel.execution_backend.itemText(index)
            for index in range(panel.execution_backend.count())
        ).lower()
        assert not any(token in mode_text for token in ("backend", "%", "utilization"))
        assert not hasattr(panel, "cuda_share")
        assert not hasattr(panel, "cpu_share")
        assert panel.cuda_vram_budget.parentWidget() is None
    finally:
        panel.close()
        qt_application.processEvents()


def test_policy_workspace_omits_internal_training_and_validation_modes(qt_application, tmp_path):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.calo_intelligence_panel import CALOIntelligencePanel

    state = AppState(tmp_path / "scientist-policy.sqlite")
    panel = CALOIntelligencePanel(state, ExperimentManager(state))
    try:
        assert panel.new_training_button.text() == "Train policy"
        assert panel.policy_import_button.text() == "Import policy"
        assert not hasattr(panel, "no_ai_mode")
        assert not hasattr(panel, "allow_unqualified")
        assert not hasattr(panel, "metadata")
        assert not hasattr(panel, "cuda_rollout_share")
        assert not hasattr(panel, "resume_task_by_id")
    finally:
        panel.close()
        qt_application.processEvents()


def test_failed_study_application_does_not_mutate_live_configuration(
    qt_application, monkeypatch, tmp_path
):
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels import dashboard_panel

    state = AppState(tmp_path / "transactional-study.sqlite")
    original = state.config
    original_payload = original.to_dict()
    panel = dashboard_panel.DashboardPanel(state)
    monkeypatch.setattr(
        dashboard_panel.CaseLoader,
        "load",
        staticmethod(lambda _name: (_ for _ in ()).throw(ValueError("invalid case"))),
    )
    monkeypatch.setattr(
        dashboard_panel.QMessageBox,
        "critical",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    try:
        panel._apply_study_protocol()
        assert state.config is original
        assert state.config.to_dict() == original_payload
    finally:
        panel.close()
        qt_application.processEvents()


def test_dashboard_readiness_uses_scientist_facing_resource_language(qt_application, tmp_path):
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.dashboard_panel import DashboardPanel

    state = AppState(tmp_path / "scientist-readiness.sqlite")
    panel = DashboardPanel(state)
    try:
        visible_labels = " ".join(
            (*panel.compute_labels.keys(), *panel.training_labels.keys())
        ).lower()
        assert "available system memory" in visible_labels
        assert "available accelerator memory" in visible_labels
        assert "recoverable checkpoint" in visible_labels
        assert not any(
            token in visible_labels
            for token in ("utilization", "backend", "microbatch", "schema", "worker budget")
        )
    finally:
        panel.close()
        qt_application.processEvents()


def _visible_interface_text(panel) -> str:
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QComboBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QTabWidget,
        QWidget,
    )

    fragments: list[str] = []
    for widget in panel.findChildren(QWidget):
        if not widget.isVisibleTo(panel):
            continue
        fragments.extend((widget.toolTip(), widget.statusTip(), widget.whatsThis()))
        if isinstance(widget, (QLabel, QAbstractButton)):
            fragments.append(widget.text())
        if isinstance(widget, QGroupBox):
            fragments.append(widget.title())
        if isinstance(widget, QLineEdit):
            fragments.append(widget.placeholderText())
        if isinstance(widget, QComboBox):
            fragments.extend(widget.itemText(index) for index in range(widget.count()))
        if isinstance(widget, QTabWidget):
            fragments.extend(widget.tabText(index) for index in range(widget.count()))
    return " ".join(fragment for fragment in fragments if fragment).lower()


def test_normal_scientist_panels_hide_venue_and_engineering_language(qt_application, tmp_path):
    from calo_rpd_studio.app.experiment_manager import ExperimentManager
    from calo_rpd_studio.app.state_manager import AppState
    from calo_rpd_studio.gui.panels.benchmark_campaign_panel import BenchmarkCampaignPanel
    from calo_rpd_studio.gui.panels.calo_intelligence_panel import CALOIntelligencePanel
    from calo_rpd_studio.gui.panels.dashboard_panel import DashboardPanel
    from calo_rpd_studio.gui.panels.experiment_manager_panel import ExperimentManagerPanel
    from calo_rpd_studio.gui.panels.portfolio_manager_panel import PortfolioManagerPanel

    state = AppState(tmp_path / "scientist-visible-text.sqlite")
    manager = ExperimentManager(state)
    panels = (
        DashboardPanel(state),
        ExperimentManagerPanel(state, manager),
        PortfolioManagerPanel(state),
        CALOIntelligencePanel(state, manager),
        BenchmarkCampaignPanel(state, manager),
    )
    try:
        for panel in panels:
            panel.show()
        qt_application.processEvents()
        visible_text = " ".join(_visible_interface_text(panel) for panel in panels)
        forbidden = (
            "journal",
            "transactions",
            "q1",
            "q2",
            "q3",
            "development",
            "developer",
            "backend",
            "schema",
            "xpu",
            "safe-80",
            "worker budget",
            "microbatch",
            "utilization",
            "phase 4",
            "phase 6",
            "legacy",
            "production-candidate",
            "development freeze",
            "source-bound",
            "feature flag",
            "post-freeze",
            "a-e/f-off",
            "runtime abi",
            "software freeze",
            "frozen calo",
            "checksum",
            "sha-256",
        )
        hits = {token for token in forbidden if token in visible_text}
        assert not hits, {
            token: visible_text[
                max(0, visible_text.find(token) - 120) : visible_text.find(token) + 180
            ]
            for token in hits
        }
    finally:
        for panel in panels:
            panel.close()
        qt_application.processEvents()
