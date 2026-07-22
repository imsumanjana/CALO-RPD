from __future__ import annotations

import ast
import functools
import json
from pathlib import Path

import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo.competitive_training import (
    TrainingSessionStatus,
    _rank_key,
    compare_champion_metrics,
    validation_bundle_fingerprint,
)
from calo_rpd_studio.algorithms.calo.optimizer import CALOOptimizer
from calo_rpd_studio.algorithms.calo.policy_qualification import PolicyQualificationConfig, _grade
from calo_rpd_studio.algorithms.calo.training import TrainingConfig, _curriculum_stage
from calo_rpd_studio.power_system.newton_raphson import _jacobian


def _champion_metrics(*, objective: float, latency: float, stable_fp: str = "bundle") -> dict:
    return {
        "valid": True,
        "eligible": True,
        "validation_bundle_fingerprint": stable_fp,
        "feasible_episode_rate": 1.0,
        "median_final_feasible_objective": objective,
        "median_constraint_violation": 0.0,
        "convergence_auc": objective + 1.0,
        "objective_iqr": 0.1,
        "median_validation_return": -objective,
        "policy_inference_ms": latency,
    }


def test_champion_quality_is_hardware_neutral_and_global_rank_is_order_independent():
    incumbent = _champion_metrics(objective=5.0, latency=0.1)
    candidate = _champion_metrics(objective=4.8, latency=999.0)
    decision = compare_champion_metrics(candidate, incumbent)
    assert decision.superior is True
    assert "median_final_feasible_objective" in decision.reason

    candidates = [
        ("B03", _champion_metrics(objective=4.9, latency=0.01)),
        ("B01", _champion_metrics(objective=4.7, latency=500.0)),
        ("B02", _champion_metrics(objective=4.8, latency=0.02)),
    ]
    winners = {
        min(order, key=lambda row: _rank_key(row[1], stable_id=row[0]))[0]
        for order in (candidates, list(reversed(candidates)), [candidates[1], candidates[2], candidates[0]])
    }
    assert winners == {"B01"}


def test_validation_bundle_fingerprint_changes_with_scientific_evidence_bundle():
    a = TrainingConfig(champion_validation_seed=17, champion_validation_episodes=5, champion_validation_horizon=32)
    b = TrainingConfig(champion_validation_seed=18, champion_validation_episodes=5, champion_validation_horizon=32)
    c = TrainingConfig(champion_validation_seed=17, champion_validation_episodes=6, champion_validation_horizon=32)
    assert validation_bundle_fingerprint(a) != validation_bundle_fingerprint(b)
    assert validation_bundle_fingerprint(a) != validation_bundle_fingerprint(c)


def test_infinite_curriculum_is_independent_of_hidden_session_epochs():
    milestones = (5, 10, 16, 20)
    # Session-duration/epochs is intentionally absent from the stage calculation in v5.8.
    for completed_epoch in (0, 4, 5, 9, 10, 16, 25, 1000):
        expected = _curriculum_stage(completed_epoch, None, False, milestones=milestones)
        assert expected == _curriculum_stage(completed_epoch, 24, False, milestones=milestones)
        assert expected == _curriculum_stage(completed_epoch, 1_000_000, False, milestones=milestones)


def test_partial_callable_scientific_state_changes_exact_resume_fingerprint():
    def transform(case, *, scale=1.0):
        return case, scale

    a = functools.partial(transform, scale=0.95)
    b = functools.partial(transform, scale=1.05)
    ca = CALOOptimizer._compatibility_jsonable(a)
    cb = CALOOptimizer._compatibility_jsonable(b)
    assert json.dumps(ca, sort_keys=True) != json.dumps(cb, sort_keys=True)


def test_sparse_newton_jacobian_stays_sparse_when_scipy_is_available():
    scipy_sparse = pytest.importorskip("scipy.sparse")
    ybus = scipy_sparse.csr_matrix(
        np.array(
            [
                [10 - 30j, -10 + 30j, 0j],
                [-10 + 30j, 20 - 60j, -10 + 30j],
                [0j, -10 + 30j, 10 - 30j],
            ],
            dtype=complex,
        )
    )
    voltage = np.array([1 + 0j, 1 + 0j, 1 + 0j], dtype=complex)
    jac = _jacobian(ybus, voltage, np.array([1, 2]), np.array([2]))
    assert scipy_sparse.issparse(jac)


def test_formal_superiority_qualification_fails_without_favorable_significant_evidence():
    cfg = PolicyQualificationConfig(
        runs=30,
        minimum_promotion_runs=30,
        require_independent_validation=True,
        qualification_mode="superiority",
    )
    aggregate = {
        "feasible_probability": 1.0,
        "independent_validation_probability": 1.0,
        "median_auc": 1.0,
    }
    case_summary = {"case30": {"median_objective": 100.0}}
    paired = {
        "candidate_vs_no_ai": {
            "n_pairs": 30,
            "median_difference": 0.0,
            "win_rate": 0.0,
            "rank_biserial": 0.0,
            "holm_p": 1.0,
        }
    }
    passed, grade, _score, reasons = _grade(
        aggregate,
        None,
        aggregate,
        cfg,
        paired,
        {"candidate": case_summary, "no_ai": case_summary, "reference": {}},
    )
    assert passed is False
    assert grade == "U"
    assert any("superiority" in reason.lower() or "holm" in reason.lower() for reason in reasons)



def test_legacy_resume_migration_is_explicit_and_authenticates_new_copy(tmp_path, monkeypatch):
    import hashlib
    import torch
    from calo_rpd_studio.ai import model_io

    monkeypatch.setattr(model_io, "_TRUST_DIR", tmp_path / ".trust")
    monkeypatch.setattr(model_io, "_TRUST_KEY", tmp_path / ".trust" / "resume_trust.key")
    source = tmp_path / "legacy.resume.pt"
    payload = {
        "model_state_dict": {"w": torch.tensor([1.0])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "next_epoch": 10,
        "extra": {},
    }
    torch.save(payload, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_suffix(source.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")

    with pytest.raises(PermissionError):
        model_io.migrate_legacy_local_resume(source, explicit_trust=False)
    migrated = model_io.migrate_legacy_local_resume(source, explicit_trust=True)
    assert migrated != source
    assert source.is_file()
    loaded = model_io.load_trusted_resume(migrated, map_location="cpu")
    assert loaded["next_epoch"] == 10
    migration = loaded["extra"]["legacy_resume_migration"]
    assert migration["source_sha256"] == digest



def test_exact_resume_history_payload_is_bounded(tmp_path, monkeypatch):
    import torch
    from calo_rpd_studio.ai import model_io
    from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
    from calo_rpd_studio.algorithms.calo.training import save_training_resume

    monkeypatch.setattr(model_io, "_TRUST_DIR", tmp_path / ".trust")
    monkeypatch.setattr(model_io, "_TRUST_KEY", tmp_path / ".trust" / "resume_trust.key")
    cfg = TrainingConfig(hidden_dim=8, resume_history_limit=3)
    net = CALOPolicyNetwork(32, 8)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = np.random.default_rng(7)
    path = tmp_path / "bounded.resume.pt"
    save_training_resume(
        path,
        network=net,
        optimizer=opt,
        next_epoch=1000,
        history=[{"epoch": i} for i in range(1000)],
        rng=rng,
        historical_pretraining={},
        config=cfg,
    )
    payload = model_io.load_trusted_resume(path, map_location="cpu")
    assert [row["epoch"] for row in payload["history"]] == [997, 998, 999]

def test_no_silent_broad_exception_pass_continue_or_return_in_source_tree():
    root = Path(__file__).resolve().parents[2] / "calo_rpd_studio"
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"})
            if broad and len(node.body) == 1 and isinstance(node.body[0], (ast.Pass, ast.Continue, ast.Return)):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}:{type(node.body[0]).__name__}")
    assert offenders == []

def test_training_session_status_has_safe_stop_distinct_from_completion():
    assert TrainingSessionStatus.SAFE_STOPPED != TrainingSessionStatus.COMPLETED
    assert TrainingSessionStatus.SAFE_STOPPED_DEGRADED != TrainingSessionStatus.COMPLETED
