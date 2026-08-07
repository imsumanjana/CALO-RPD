"""Rendered contracts for compact tabbed scientist workspaces."""

import pytest

pytest.importorskip("PyQt6")

from calo_rpd_studio.app.state_manager import AppState
from calo_rpd_studio.gui.panels.orpd_formulation_panel import ORPDFormulationPanel
from calo_rpd_studio.gui.panels.portfolio_manager_panel import PortfolioManagerPanel
from calo_rpd_studio.gui.panels.robust_scenarios_panel import RobustScenariosPanel


@pytest.mark.parametrize(
    ("panel_type", "expected_tabs"),
    (
        (
            ORPDFormulationPanel,
            ("Objective", "Control variables", "Constraint policy"),
        ),
        (
            RobustScenariosPanel,
            ("Scenario generator", "Robust objective"),
        ),
        (
            PortfolioManagerPanel,
            ("Definition", "Requested outputs", "Reuse and validation", "Derived plan"),
        ),
    ),
)
def test_configuration_workspaces_use_accessible_section_tabs(
    tmp_path,
    qapp,
    panel_type,
    expected_tabs,
):
    state = AppState(tmp_path / f"{panel_type.__name__}.sqlite")
    panel = panel_type(state)
    try:
        tabs = panel.section_tabs
        assert tabs.accessibleName()
        assert tuple(tabs.tabText(index) for index in range(tabs.count())) == expected_tabs
        assert all(tabs.tabToolTip(index) for index in range(tabs.count()))
        assert all(tabs.widget(index).accessibleName() for index in range(tabs.count()))
    finally:
        panel.close()


def test_portfolio_requested_outputs_fill_the_visible_tree_width(tmp_path, qapp):
    from PyQt6.QtWidgets import QHeaderView

    state = AppState(tmp_path / "portfolio-output-width.sqlite")
    panel = PortfolioManagerPanel(state)
    try:
        panel.resize(1400, 900)
        panel.show()
        panel.section_tabs.setCurrentIndex(1)
        qapp.processEvents()

        tree = panel.outputs
        header = tree.header()
        assert tree.objectName() == "PortfolioRequestedOutputs"
        assert tree.accessibleName()
        assert header.sectionResizeMode(0) == QHeaderView.ResizeMode.Fixed
        assert header.sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
        assert header.sectionResizeMode(2) == QHeaderView.ResizeMode.ResizeToContents
        assert abs(header.length() - tree.viewport().width()) <= 24

        metrics = tree.fontMetrics()
        for column in range(tree.columnCount()):
            required = metrics.horizontalAdvance(tree.headerItem().text(column)) + 32
            assert header.sectionSize(column) >= required
    finally:
        panel.close()
