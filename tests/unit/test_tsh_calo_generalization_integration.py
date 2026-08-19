"""Cross-layer regressions for TSH-CALO guard accounting and ensemble invariants."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_evaluation_accounting import (
    augment_extension_plan,
    plan_training_evaluation_accounting,
    synchronize_training_progress,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_generalization_guard import (
    candidate_generalization_status,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import (
    _validate_ensemble_generalization_homogeneity_rows,
)


class _Guard:
    enabled = True

    @staticmethod
    def validation_evaluations_per_case(population_size: int) -> int:
        return 2 * int(population_size)


def _plan(*, guarded: bool = True):
    return SimpleNamespace(
        development_cases=("toy-development",),
        members=(
            SimpleNamespace(episodes=(object(),)),
            SimpleNamespace(episodes=(object(),)),
        ),
        population_size=4,
        max_evaluations=8,
        generalization_guard=_Guard() if guarded else None,
    )


def _evidence(count: int = 8) -> dict:
    return {"candidate_evaluations": count}


def _complete_member_slot() -> dict:
    return {
        "baseline_monitor_evidence": _evidence(),
        "baseline_final_evidence": _evidence(),
        "monitor_evidence": [_evidence()],
        "result": {"final_evidence": _evidence()},
    }


def test_segment_accounting_includes_every_guard_bundle():
    accounting = plan_training_evaluation_accounting(_plan())
    assert accounting.training_candidate_evaluations == 16
    assert accounting.generalization_guard_candidate_evaluations == 64
    assert accounting.total_counted_candidate_evaluations == 80

    legacy = plan_training_evaluation_accounting(_plan(guarded=False))
    assert legacy.training_candidate_evaluations == 16
    assert legacy.generalization_guard_candidate_evaluations == 0
    assert legacy.total_counted_candidate_evaluations == 16


def test_progress_uses_only_durable_guard_evidence_and_preserves_legacy_fields():
    status = {
        "state": "running",
        "progress": {"committed_candidate_evaluations": 8},
        "generalization_guard": {
            "members": {
                "0": _complete_member_slot(),
                "1": {
                    "baseline_monitor_evidence": None,
                    "baseline_final_evidence": None,
                    "monitor_evidence": [],
                    "result": None,
                },
            }
        },
    }
    progress = synchronize_training_progress(_plan(), status)
    assert progress["committed_candidate_evaluations"] == 8
    assert progress["total_candidate_evaluations"] == 16
    assert progress["committed_generalization_guard_candidate_evaluations"] == 32
    assert progress["committed_total_candidate_evaluations"] == 40
    assert progress["total_counted_candidate_evaluations"] == 80
    assert progress["progress_percent"] == 50

    status["state"] = "completed"
    status["progress"]["committed_candidate_evaluations"] = 16
    status["generalization_guard"]["members"]["1"] = _complete_member_slot()
    progress = synchronize_training_progress(_plan(), status)
    assert progress["committed_total_candidate_evaluations"] == 80
    assert progress["progress_percent"] == 100


def test_mixed_or_differently_guarded_ensemble_fails_before_promotion():
    mixed = {
        "source_kind": "independent_policy_training_ensemble",
        "members": [
            {
                "training_provenance": {
                    "generalization_guard_sha256": "a" * 64,
                    "generalization_guard": {"status": "passed"},
                }
            },
            {"training_provenance": {}},
        ],
    }
    allowed, reason = candidate_generalization_status(mixed)
    assert allowed is False
    assert "mixes guarded and legacy" in reason

    different = {
        "source_kind": "independent_policy_training_ensemble",
        "members": [
            {
                "training_provenance": {
                    "generalization_guard_sha256": "a" * 64,
                    "generalization_guard": {"status": "passed"},
                }
            },
            {
                "training_provenance": {
                    "generalization_guard_sha256": "b" * 64,
                    "generalization_guard": {"status": "passed"},
                }
            },
        ],
    }
    allowed, reason = candidate_generalization_status(different)
    assert allowed is False
    assert "different generalization-guard designs" in reason


def test_artifact_boundary_rejects_mixed_guard_rows():
    with pytest.raises(ValueError, match="mix guarded and legacy"):
        _validate_ensemble_generalization_homogeneity_rows(
            [
                {
                    "generalization_guard_sha256": "a" * 64,
                    "generalization_guard": {"status": "passed"},
                },
                {},
            ]
        )
    with pytest.raises(ValueError, match="share one generalization-guard design"):
        _validate_ensemble_generalization_homogeneity_rows(
            [
                {
                    "generalization_guard_sha256": "a" * 64,
                    "generalization_guard": {"status": "passed"},
                },
                {
                    "generalization_guard_sha256": "b" * 64,
                    "generalization_guard": {"status": "passed"},
                },
            ]
        )


def test_extension_contract_discloses_segment_and_cumulative_counted_work():
    payload = {
        "segment_number": 1,
        "prior_cumulative_candidate_evaluations": 16,
    }
    augment_extension_plan(_plan(), payload)
    assert payload["segment_training_candidate_evaluations"] == 16
    assert payload["segment_generalization_guard_candidate_evaluations"] == 64
    assert payload["segment_total_counted_candidate_evaluations"] == 80
    assert payload["prior_cumulative_total_counted_candidate_evaluations"] == 80
    assert payload["next_cumulative_total_counted_candidate_evaluations"] == 160
    assert payload["legacy_candidate_evaluation_fields_are_training_only"] is True
