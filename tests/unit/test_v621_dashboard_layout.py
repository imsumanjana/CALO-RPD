from __future__ import annotations

import ast
from pathlib import Path


def _dashboard_source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "calo_rpd_studio" / "gui" / "panels" / "dashboard_panel.py").read_text(encoding="utf-8")


def test_dashboard_source_is_valid_python():
    ast.parse(_dashboard_source())


def test_dashboard_has_page_scroll_and_three_scrollable_tabs():
    source = _dashboard_source()
    assert 'self.dashboard_scroll = QScrollArea()' in source
    assert 'self.dashboard_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)' in source
    assert 'self.dashboard_tabs = QTabWidget()' in source
    assert 'self.dashboard_tabs.addTab(_scrollable_tab(readiness_tab), "System Readiness")' in source
    assert 'self.dashboard_tabs.addTab(_scrollable_tab(training_tab), "Training Queue")' in source
    assert 'self.dashboard_tabs.addTab(_scrollable_tab(context_tab), "Scientific Context")' in source
    assert source.count('self.dashboard_tabs.addTab(') == 3


def test_dashboard_sections_are_not_stacked_directly_on_root_layout():
    source = _dashboard_source()
    assert 'self.layout_root.addWidget(readiness)' not in source
    assert 'self.layout_root.addWidget(training_queue)' not in source
    assert 'self.layout_root.addWidget(context)' not in source
    assert 'self.dashboard_body_layout.addWidget(self.dashboard_tabs, 1)' in source


def test_dashboard_device_table_keeps_usable_height_inside_system_tab():
    source = _dashboard_source()
    assert 'self.device_table.setMinimumHeight(280)' in source
    assert 'self.dashboard_tabs.setMinimumHeight(500)' in source
