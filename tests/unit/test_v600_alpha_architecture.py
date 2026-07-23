from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from calo_rpd_studio.app.workspaces import (
    WORKSPACE_KEYS,
    migrate_legacy_workspace_index,
    workspace_index_for_key,
    workspace_key_for_index,
)
from calo_rpd_studio.compute.topology import (
    ComputeDevice,
    ComputeTopologySnapshot,
    SafeResourceBudgetEngine,
)
from calo_rpd_studio.algorithms.calo.policy_readiness import evaluate_governing_policy
from calo_rpd_studio.algorithms.calo.training import TrainingConfig
from calo_rpd_studio.algorithms.calo.competitive_training import _plan_branch_devices


def _topology(*, devices=(), logical=16, physical=8, ram_gib=32, ram_used=25.0):
    return ComputeTopologySnapshot(
        cpu_name="Test CPU",
        physical_cores=physical,
        logical_threads=logical,
        ram_total_bytes=ram_gib * 1024**3,
        ram_used_percent=ram_used,
        devices=tuple(devices),
        platform_name="test",
        fingerprint="topology-test",
    )


def _device(runtime_id: str, backend: str, *, runtime="primary", full_branch=True, memory=20.0):
    return ComputeDevice(
        physical_id=runtime_id,
        os_label=f"Physical {runtime_id}",
        runtime_id=runtime_id,
        backend=backend,
        runtime=runtime,
        name=runtime_id,
        memory_total_bytes=8 * 1024**3,
        memory_used_percent=memory,
        utilization_percent=10.0,
        telemetry="test",
        ppo_learner=runtime == "primary",
        policy_actor=True,
        orpd_evaluator=True,
        full_training_branch=full_branch,
    )


def test_v600_workspace_order_and_legacy_migration_are_key_based():
    assert WORKSPACE_KEYS[:3] == ("dashboard", "calo_intelligence", "power_system")
    assert workspace_index_for_key("calo_intelligence") == 1
    assert workspace_key_for_index(1) == "calo_intelligence"
    # v5.9 index 5 was CALO Intelligence and index 1 was Power System.
    assert migrate_legacy_workspace_index(5) == "calo_intelligence"
    assert migrate_legacy_workspace_index(1) == "power_system"
    # A schema-2 positional fallback uses the current v6 ordering.
    assert migrate_legacy_workspace_index(1, source_schema=2) == "calo_intelligence"


def test_safe80_profile_calculates_global_budget_and_validated_branch_slots():
    topology = _topology(
        devices=(
            _device("cuda:0", "cuda"),
            _device("xpu:0", "xpu"),
        )
    )
    profile = SafeResourceBudgetEngine(allocation_limit_fraction=0.80).calculate(topology)
    assert profile.ready is True
    assert profile.reserve_percent == 20
    assert profile.safe_cpu_worker_budget == 12
    assert profile.accelerator_branch_slots == 2
    assert profile.safe_parallel_branches == 2


def test_safe80_sidecar_xpu_is_not_counted_as_full_branch_in_alpha():
    topology = _topology(
        devices=(
            _device("cuda:0", "cuda"),
            _device("xpu:0", "xpu", runtime="sidecar", full_branch=False),
        )
    )
    profile = SafeResourceBudgetEngine().calculate(topology)
    assert profile.accelerator_branch_slots == 1
    assert profile.safe_parallel_branches == 1




def test_safe80_accelerator_memory_headroom_is_required_for_branch_admission():
    # 8 GiB device at 70% used has only 0.8 GiB left before the 80% protection ceiling,
    # below the default 2 GiB estimated branch working set.
    topology = _topology(devices=(_device("cuda:0", "cuda", memory=70.0),))
    profile = SafeResourceBudgetEngine().calculate(topology)
    assert profile.accelerator_branch_slots == 0
    # Accelerator hardware exists but lacks protected headroom, so automatic CPU spillover is blocked.
    assert profile.safe_parallel_branches == 0
    assert profile.ready is False


def test_safe80_memory_pressure_can_block_training_capacity():
    profile = SafeResourceBudgetEngine().calculate(_topology(devices=(), ram_used=95.0))
    assert profile.ready is False
    assert profile.safe_parallel_branches == 0
    assert any("RAM" in reason for reason in profile.reasons)


@dataclass
class _Policy:
    id: str = "p1"
    name: str = "Policy"
    checkpoint_path: str = "policy.pt"
    sha256: str = "abc"
    architecture_version: str = ""
    state_schema_version: str = ""
    action_schema_version: str = ""
    training_environment_version: str = ""
    qualification_status: str = "candidate"
    grade: str = "U"
    active: bool = False
    archived: bool = False
    metadata: dict = None
    usable: bool = True
    runtime_compatible: bool = True


class _Registry:
    def __init__(self, records, checksum="abc"):
        self.records = records
        self.checksum = checksum

    def list(self, include_archived=False):
        return list(self.records)

    def inspect_checkpoint(self, path):
        return {"sha256": self.checksum}


def test_governing_policy_is_fail_closed_until_qualified_active_integrity_verified():
    assert evaluate_governing_policy(_Registry([])).ready is False
    candidate = _Policy(active=True, qualification_status="candidate")
    assert evaluate_governing_policy(_Registry([candidate])).state == "unqualified"
    qualified = _Policy(active=True, qualification_status="qualified", grade="A")
    assert evaluate_governing_policy(_Registry([qualified])).ready is True
    assert evaluate_governing_policy(_Registry([qualified], checksum="changed")).state == "checksum_mismatch"


def test_competitive_planner_beta_separates_total_branches_from_safe_concurrency():
    # v6.1 supersedes the alpha rule that total scientific branch count itself must fit the
    # simultaneous ceiling. On a CPU-only test runtime, two scientific branches may queue behind
    # one protected CPU execution slot.
    config = TrainingConfig(ppo_device="cpu", parallel_concurrency=1)
    assert _plan_branch_devices(config, 2) == ["cpu"]


def test_competitive_planner_never_silently_falls_back_from_explicit_cuda(monkeypatch):
    import calo_rpd_studio.algorithms.calo.competitive_training as competitive

    monkeypatch.setattr(competitive.torch.cuda, "is_available", lambda: False)
    config = TrainingConfig(ppo_device="cuda", safe_parallel_branches=1)
    with pytest.raises(RuntimeError, match="fallback is disabled"):
        _plan_branch_devices(config, 1)
