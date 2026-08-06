from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from calo_rpd_studio.algorithms.calo.policy_qualification import (
    _convergence_auc,
    _eval_to_feasible,
    _paired_evidence,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import (
    TSHCALOQualificationPlan,
    _case_evidence,
)
from calo_rpd_studio.benchmarking.freeze import create_freeze_manifest, verify_freeze_manifest
from calo_rpd_studio.scripts.audit_broad_exceptions import audit_broad_exceptions
from calo_rpd_studio.scripts.verify_active_version import verify_active_version
from calo_rpd_studio.statistics import paired as paired_module
from calo_rpd_studio.statistics.paired import (
    PAIRED_ANALYSIS_SCHEMA_VERSION,
    PairIntegrityError,
    exact_keyed_pairs,
    exploratory_pair_status,
    matched_pairs_rank_biserial,
    relative_objective_improvement,
    wilcoxon_signed_rank_evidence,
)
from calo_rpd_studio.statistics.wilcoxon import wilcoxon_signed_rank


ROOT = Path(__file__).resolve().parents[2]


def _row(case: str, run_index: int, objective: float, label: str = "") -> dict:
    return {
        "case": case,
        "run_index": run_index,
        "objective": objective,
        "label": label,
    }


def test_rank_biserial_uses_signed_rank_mass_and_average_ties():
    assert matched_pairs_rank_biserial([-1.0, -2.0, 3.0]) == pytest.approx(0.0)
    assert matched_pairs_rank_biserial([1.0, -1.0, 0.0]) == pytest.approx(0.0)
    assert matched_pairs_rank_biserial([1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_relative_improvement_is_bounded_symmetric_and_positive_is_better():
    assert relative_objective_improvement(9.0, 10.0) == pytest.approx(0.1)
    assert relative_objective_improvement(10.0, 9.0) == pytest.approx(-0.1)
    assert relative_objective_improvement(1e-9, 0.0) == pytest.approx(-1.0)


def test_exact_keyed_pairs_align_reordered_inputs_and_reject_integrity_failures():
    candidate = [_row("case30", 1, 9.0), _row("case30", 0, 8.0)]
    comparator = [_row("case30", 0, 10.0), _row("case30", 1, 11.0)]
    pairs = exact_keyed_pairs(
        candidate,
        comparator,
        expected_keys=(("case30", 0), ("case30", 1)),
    )
    assert [pair.key for pair in pairs] == [("case30", 0), ("case30", 1)]
    with pytest.raises(PairIntegrityError, match="Duplicate"):
        exact_keyed_pairs(candidate + [candidate[0]], comparator)
    with pytest.raises(PairIntegrityError, match="record sets differ"):
        exact_keyed_pairs(candidate[:-1], comparator)
    with pytest.raises(PairIntegrityError, match="preregistered"):
        exact_keyed_pairs(candidate, comparator, expected_keys=(("case30", 0),))


def test_incomplete_exploratory_pair_status_can_never_qualify():
    status = exploratory_pair_status([_row("case30", 0, 1.0)], [])
    assert status["status"] == "incomplete_pairs"
    assert status["qualifying"] is False


def test_unequal_wilcoxon_arrays_fail_before_statistical_execution():
    with pytest.raises(PairIntegrityError, match="equal lengths"):
        wilcoxon_signed_rank([1.0, 2.0], [1.0])


def test_declared_wilcoxon_failure_does_not_switch_test_family(monkeypatch):
    def invalid_test(*_args, **_kwargs):
        raise ValueError("declared test unavailable")

    monkeypatch.setattr(paired_module, "_scipy_wilcoxon", invalid_test)
    evidence = wilcoxon_signed_rank_evidence([1.0, 2.0], alternative="greater")
    assert evidence["status"] == "declared_test_invalid"
    assert evidence["p_value"] is None
    assert evidence["fallback_used"] is False
    assert evidence["statistical_test"] == "scipy.stats.wilcoxon"


def test_policy_qualification_uses_exact_pairs_and_positive_effect_orientation():
    candidate = [_row("case30", index, 9.0 - index) for index in range(3)]
    comparator = [_row("case30", index, 10.0 - index) for index in reversed(range(3))]
    evidence = _paired_evidence(candidate, comparator)
    assert evidence["analysis_schema_version"] == PAIRED_ANALYSIS_SCHEMA_VERSION
    assert evidence["expected_pairs"] == evidence["n_pairs"] == 3
    assert evidence["median_relative_improvement"] > 0.0
    assert evidence["rank_biserial"] == pytest.approx(1.0)


def _campaign_record(label: str, run_index: int, objective: float) -> dict:
    anytime = {
        str(fraction): {"feasible": True, "objective": objective}
        for fraction in (0.25, 0.5, 0.75, 1.0)
    }
    return {
        **_row("case30", run_index, objective, label),
        "feasible": True,
        "evaluations": 100,
        "independent_validation": {"passed": True},
        "anytime": anytime,
    }


def test_both_qualification_engines_share_v12_improvement_and_effect_definitions():
    plan = TSHCALOQualificationPlan(
        qualification_run_id="phase1-contract",
        source_commit="a" * 40,
        candidate_path="candidate.pt",
        candidate_sha256="b" * 64,
        development_cases=("case30",),
        runs=3,
        master_seed=7,
        population_size=10,
        max_evaluations=100,
        bootstrap_resamples=1_000,
    )
    records = []
    for index in range(3):
        records.append(_campaign_record("candidate", index, 9.0 - index))
        records.append(_campaign_record("baseline", index, 10.0 - index))
    tsh = _case_evidence(plan, "case30", records, analysis_seed=19)
    generic = _paired_evidence(
        [item for item in records if item["label"] == "candidate"],
        [item for item in records if item["label"] == "baseline"],
    )
    assert tsh["analysis_schema_version"] == generic["analysis_schema_version"]
    assert tsh["median_relative_objective_improvement"] == pytest.approx(
        generic["median_relative_improvement"]
    )
    assert tsh["paired_rank_biserial"] == pytest.approx(generic["rank_biserial"])


def test_convergence_separates_feasibility_delay_from_post_feasible_auc():
    first = SimpleNamespace(
        evaluations=3,
        metadata={
            "convergence_evaluations": [1, 2, 3],
            "best_feasible_objective_history": [float("nan"), 10.0, 8.0],
            "first_feasible_evaluation": 2,
        },
    )
    delayed = SimpleNamespace(
        evaluations=13,
        metadata={
            "convergence_evaluations": [11, 12, 13],
            "best_feasible_objective_history": [float("nan"), 10.0, 8.0],
            "first_feasible_evaluation": 12,
        },
    )
    assert _convergence_auc(first) == pytest.approx(_convergence_auc(delayed))
    assert _eval_to_feasible(first) == 2.0
    assert _eval_to_feasible(delayed) == 12.0


def test_freeze_diagnostics_report_all_categories_in_text_and_json(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    changed = repo / "changed.txt"
    missing = repo / "missing.txt"
    changed.write_text("before", encoding="utf-8")
    missing.write_text("before", encoding="utf-8")
    manifest = create_freeze_manifest(
        tmp_path / "freeze.json",
        project_root=repo,
        relative_paths=("changed.txt", "missing.txt"),
    )
    changed.write_text("after", encoding="utf-8")
    missing.unlink()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["note"] = "invalidate manifest hash"
    payload["files"]["../outside.txt"] = {"sha256": "0" * 64, "size_bytes": 0}
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    verification = verify_freeze_manifest(manifest, project_root=repo)
    machine = verification.to_dict()
    assert verification.passed is False
    assert all(
        token in verification.message for token in ("invalid=", "missing=", "changed=", "extra=")
    )
    assert machine["manifest_valid"] is False
    assert machine["missing_files"] == ["missing.txt"]
    assert machine["changed_files"] == ["changed.txt"]
    assert machine["extra_files"] == ["../outside.txt"]


def test_active_v12_identity_and_historical_correction_records_are_consistent():
    assert verify_active_version(ROOT)["passed"] is True
    correction = json.loads(
        (
            ROOT / "docs/implementation/HISTORICAL_STATISTICAL_EVIDENCE_CORRECTIONS_V12.json"
        ).read_text(encoding="utf-8")
    )
    record = correction["records"][0]
    assert record["historical_record_mutated"] is False
    assert record["decision_after_correction_review"] == "negative_decision_preserved"
    assert record["promotion_or_activation_authority"] is False


def test_broad_exception_inventory_has_no_scientific_priority_handlers():
    report = audit_broad_exceptions(ROOT)
    assert report["passed"] is True
    assert report["scientific_priority_handlers"] == []
