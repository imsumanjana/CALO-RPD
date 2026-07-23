from __future__ import annotations

import pytest

PyQt6 = pytest.importorskip("PyQt6")

from calo_rpd_studio.app.workspaces import WORKSPACE_KEYS


def test_v620_canonical_policy_first_workspace_order():
    assert WORKSPACE_KEYS[:3] == ("dashboard", "calo_intelligence", "power_system")


def test_v620_gui_dependency_available_for_full_target_suite():
    # Marker test: target-machine release qualification must execute this module without skip.
    assert PyQt6 is not None
