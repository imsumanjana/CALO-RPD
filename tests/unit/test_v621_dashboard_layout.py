from __future__ import annotations

import ast
from pathlib import Path


def _dashboard_source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "calo_rpd_studio" / "gui" / "panels" / "dashboard_panel.py").read_text(
        encoding="utf-8"
    )


def test_dashboard_source_is_valid_python():
    ast.parse(_dashboard_source())


def test_dashboard_has_one_page_scroll_and_two_focused_detail_tabs():
    source = _dashboard_source()
    assert "self.dashboard_scroll = QScrollArea()" in source
    assert (
        "self.dashboard_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)"
        in source
    )
    assert "self.dashboard_tabs = QTabWidget()" in source
    assert 'self.dashboard_tabs.addTab(readiness_tab, "System Readiness")' in source
    assert 'self.dashboard_tabs.addTab(context_tab, "Scientific Context")' in source
    assert source.count("self.dashboard_tabs.addTab(") == 2
    assert "self.legacy_study_setup = study_tab" in source
    assert "self.activity_drawer = DisclosurePanel(" in source
    assert "def _scrollable_tab(" not in source


def test_dashboard_sections_are_not_stacked_directly_on_root_layout():
    source = _dashboard_source()
    assert "self.layout_root.addWidget(readiness)" not in source
    assert "self.layout_root.addWidget(training_queue)" not in source
    assert "self.layout_root.addWidget(context)" not in source
    assert "self.dashboard_body_layout.addWidget(self.dashboard_tabs, 1)" in source


def test_dashboard_prioritizes_next_action_readiness_and_recent_work():
    source = _dashboard_source()
    tree = ast.parse(source)
    metric_titles = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "MetricCard"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert '"Next required action"' in source
    assert {"Data", "Compute", "Policy", "Validation", "Storage"} <= metric_titles
    assert '"Recent work and evidence"' in source


def test_dashboard_device_table_keeps_usable_height_inside_system_tab():
    source = _dashboard_source()
    assert "self.device_table.setMinimumHeight(280)" in source
    assert "self.dashboard_tabs.setMinimumHeight(500)" in source


def test_dashboard_uses_scientist_facing_memory_and_compute_labels():
    source = _dashboard_source()
    assert '"Available memory"' in source
    assert '"Compute choice"' in source
    assert '"Backend"' not in source
    assert "80% of what is free at that moment" in source
