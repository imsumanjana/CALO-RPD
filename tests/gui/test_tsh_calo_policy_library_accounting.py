"""Policy-library counted-work presentation contracts."""

from __future__ import annotations

import importlib.util
import os

import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("PyQt6") is None, reason="PyQt6 is not installed"
)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_completed_campaign_row_prefers_exact_counted_total():
    from calo_rpd_studio.gui.panels.calo_intelligence_panel import (
        _campaign_accounting,
        _row_counted_evaluations,
    )

    row = {
        "training_evaluations": 32,
        "generalization_guard_evaluations": 192,
        "total_counted_evaluations": 224,
    }
    assert _campaign_accounting(row) == {
        "training_candidate_evaluations": 32,
        "generalization_guard_candidate_evaluations": 192,
        "total_counted_candidate_evaluations": 224,
    }
    assert _row_counted_evaluations(row, object()) == 224


def test_registered_policy_without_manifest_fails_closed_instead_of_underreporting():
    from calo_rpd_studio.gui.panels.calo_intelligence_panel import (
        _row_counted_evaluations,
    )

    class Policy:
        id = "policy-1"

    class Registry:
        @staticmethod
        def counted_evaluation_count(_policy_id):
            return None

    assert _row_counted_evaluations(Policy(), Registry()) is None


def test_policy_registry_metadata_requires_exact_additive_total():
    from calo_rpd_studio.algorithms.calo.policy_registry import (
        _metadata_evaluation_accounting,
    )

    assert _metadata_evaluation_accounting(
        {
            "training_candidate_evaluations": 32,
            "generalization_guard_candidate_evaluations": 192,
            "total_counted_candidate_evaluations": 224,
        }
    ) == {
        "training_candidate_evaluations": 32,
        "generalization_guard_candidate_evaluations": 192,
        "total_counted_candidate_evaluations": 224,
    }
    assert (
        _metadata_evaluation_accounting(
            {
                "training_candidate_evaluations": 32,
                "generalization_guard_candidate_evaluations": 192,
                "total_counted_candidate_evaluations": 223,
            }
        )
        is None
    )
