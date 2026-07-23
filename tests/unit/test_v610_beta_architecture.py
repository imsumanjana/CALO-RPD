from __future__ import annotations

from pathlib import Path

import pytest

from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy_parallel
from calo_rpd_studio.algorithms.calo.heterogeneous_training import HeterogeneousTrainingConfig
from calo_rpd_studio.compute.topology import (
    ComputeDevice,
    ComputeTopologySnapshot,
    SafeResourceBudgetEngine,
)
from calo_rpd_studio.compute.training_resources import build_training_resource_plan


def _device(
    runtime_id: str,
    backend: str,
    *,
    runtime: str = "primary",
    full: bool = True,
    memory_percent: float = 10.0,
    total_gib: int = 8,
    actor: bool = True,
    evaluator: bool = True,
):
    return ComputeDevice(
        physical_id=f"physical:{runtime_id}:{runtime}",
        os_label=f"OS {runtime_id}",
        runtime_id=runtime_id,
        backend=backend,
        runtime=runtime,
        name=f"Test {runtime_id} {runtime}",
        memory_total_bytes=total_gib * 1024**3,
        memory_used_percent=memory_percent,
        utilization_percent=5.0,
        telemetry="test",
        ppo_learner=runtime == "primary",
        policy_actor=actor,
        orpd_evaluator=evaluator,
        full_training_branch=full,
        capability_status="validated" if full else "auxiliary",
        capability_detail="test capability",
    )


def _topology(*devices, logical: int = 16, ram_used: float = 20.0):
    return ComputeTopologySnapshot(
        cpu_name="Test CPU",
        physical_cores=max(1, logical // 2),
        logical_threads=logical,
        ram_total_bytes=32 * 1024**3,
        ram_used_percent=ram_used,
        devices=tuple(devices),
        platform_name="test",
        fingerprint="topology-v610-test",
    )


def _plan(config: TrainingConfig, total: int, topology: ComputeTopologySnapshot):
    profile = SafeResourceBudgetEngine(allocation_limit_fraction=0.80).calculate(topology)
    return build_training_resource_plan(config, total, topology=topology, profile=profile)


def test_beta2_total_branches_are_separate_from_simultaneous_concurrency():
    topology = _topology(_device("cuda:0", "cuda"), _device("xpu:0", "xpu"))
    cfg = TrainingConfig(ppo_device="auto", parallel_concurrency=2)
    plan = _plan(cfg, 4, topology)
    assert plan.total_branches == 4
    assert plan.simultaneous_branches == 2
    assert plan.queued_branches == 2
    assert [slot.primary_device for slot in plan.slots] == ["cuda:0", "xpu:0"]


def test_beta3_one_global_cpu_budget_is_partitioned_not_multiplied():
    topology = _topology(_device("cuda:0", "cuda"), _device("xpu:0", "xpu"), logical=16)
    cfg = TrainingConfig(ppo_device="auto", parallel_concurrency=2, safe_global_cpu_workers=12)
    plan = _plan(cfg, 6, topology)
    assert plan.global_cpu_worker_budget == 12
    assert plan.cpu_support_reserve_per_branch == 2
    assert plan.cpu_rollout_worker_budget_total == 8
    assert sum(slot.cpu_worker_budget for slot in plan.slots) == 8
    assert sum(slot.cpu_worker_budget for slot in plan.slots) + 4 == 12
    assert plan.queued_branches == 4


def test_beta3_no_automatic_accelerator_to_cpu_spillover():
    topology = _topology(_device("cuda:0", "cuda"))
    cfg = TrainingConfig(ppo_device="auto", parallel_concurrency=2)
    with pytest.raises(RuntimeError, match="simultaneous"):
        _plan(cfg, 4, topology)


def test_beta3_explicit_cpu_is_a_deliberate_primary_not_fallback():
    topology = _topology()
    cfg = TrainingConfig(ppo_device="cpu", parallel_concurrency=1)
    plan = _plan(cfg, 4, topology)
    assert [slot.primary_device for slot in plan.slots] == ["cpu"]
    assert plan.queued_branches == 3


def test_beta4_sidecar_xpu_is_auxiliary_not_a_fake_full_branch():
    topology = _topology(
        _device("cuda:0", "cuda"),
        _device("xpu-sidecar:0", "xpu", runtime="sidecar", full=False),
    )
    cfg = HeterogeneousTrainingConfig(
        ppo_device="auto",
        parallel_concurrency=1,
        cuda_rollout_share=80,
        xpu_rollout_share=10,
        cpu_rollout_share=10,
    )
    plan = _plan(cfg, 4, topology)
    assert plan.simultaneous_branches == 1
    assert plan.queued_branches == 3
    assert plan.slots[0].primary_device == "cuda:0"
    assert plan.slots[0].uses_auxiliary_xpu
    assert plan.slots[0].auxiliary_xpu_runtime == "sidecar"


def test_beta4_direct_xpu_can_be_a_full_branch_when_capability_validated():
    topology = _topology(_device("xpu:0", "xpu", runtime="primary", full=True))
    cfg = TrainingConfig(ppo_device="xpu", parallel_concurrency=1)
    plan = _plan(cfg, 3, topology)
    assert plan.slots[0].primary_device == "xpu:0"
    assert plan.queued_branches == 2


def test_beta4_busy_first_accelerator_is_not_selected_using_another_devices_headroom():
    topology = _topology(
        _device("cuda:0", "cuda", memory_percent=70.0),  # < 2 GiB Safe-80 headroom
        _device("xpu:0", "xpu", memory_percent=10.0),
    )
    cfg = TrainingConfig(ppo_device="auto", parallel_concurrency=1)
    plan = _plan(cfg, 2, topology)
    assert plan.slots[0].primary_device == "xpu:0"



def test_beta3_strict_actor_lane_binding_never_redistributes_missing_cuda_to_cpu():
    from calo_rpd_studio.algorithms.calo.heterogeneous_training import plan_training_lanes

    with pytest.raises(RuntimeError, match="CPU redistribution is disabled"):
        plan_training_lanes(4, cuda_share=100, xpu_share=0, cpu_share=0, cuda_available=False, xpu_available=False, xpu_sidecar_available=False, strict_unavailable=True)


def test_beta1_global_training_exclusive_lock_is_wired_application_wide():
    root = Path(__file__).resolve().parents[2]
    main = (root / "calo_rpd_studio/app/main_window.py").read_text(encoding="utf-8")
    workflow = (root / "calo_rpd_studio/app/workflow_manager.py").read_text(encoding="utf-8")
    state = (root / "calo_rpd_studio/app/state_manager.py").read_text(encoding="utf-8")
    assert "policy_training_changed.connect(self._on_policy_training_changed)" in main
    assert "page.setEnabled(not active or key == \"dashboard\")" in main
    assert "Global Training Exclusive Lock" in workflow
    assert "def begin_policy_training" in state and "def end_policy_training" in state
    assert "Compute topology cannot be refreshed while policy training is active" in state


def _tiny_queue_config(**updates):
    values = dict(
        epochs=1,
        episodes_per_epoch=1,
        horizon=2,
        population_size=4,
        ppo_epochs=1,
        minibatch_size=4,
        hidden_dim=16,
        seed=321,
        rollout_workers=1,
        ppo_device="cpu",
        parallel_runs=2,
        parallel_concurrency=1,
        parallel_same_seed_branches=1,
        parallel_incremental_branches=1,
        champion_validation_horizon=2,
        champion_validation_episodes=1,
        champion_min_feasible_rate=0.0,
        branch_queue_quantum_epochs=10,
    )
    values.update(updates)
    return TrainingConfig(**values)


def test_beta2_queue_scheduler_runs_more_scientific_branches_than_concurrency(tmp_path):
    output = tmp_path / "queued_base.pt"
    _, history = train_policy_parallel(_tiny_queue_config(), output, parallel_runs=2)
    import json

    manifest = json.loads(output.with_suffix(".branches.json").read_text(encoding="utf-8"))
    session = manifest["session"]
    assert session["requested_branches"] == 2
    assert session["parallel_concurrency"] == 1
    assert session["started_branches"] == 2
    assert session["successful_branches"] == 2
    assert session["queued_branch_scheduler"] is True
    assert manifest["common_resume_epoch"] == 1
    assert len(manifest["branches"]) == 2
    assert history


def test_beta2_indefinite_queue_rotates_all_branches_before_safe_stop(tmp_path):
    """A one-slot indefinite session must time-slice rather than starving queued branches."""
    output = tmp_path / "queued_indefinite_base.pt"
    cfg = _tiny_queue_config(training_mode="indefinite", epochs=1)
    observed = {"epochs": [0, 0]}

    def session_state(payload):
        observed["epochs"] = list(payload.get("epochs", observed["epochs"]))

    def should_cancel():
        values = observed["epochs"]
        return len(values) == 2 and min(values) >= 10

    result = train_policy_parallel(
        cfg,
        output,
        parallel_runs=2,
        cancel_callback=should_cancel,
        session_state_callback=session_state,
    )
    import json

    manifest = json.loads(output.with_suffix(".branches.json").read_text(encoding="utf-8"))
    assert result.status.value in {"SAFE_STOPPED", "SAFE_STOPPED_DEGRADED"}
    assert manifest["common_resume_epoch"] >= 10
    assert all(int(row["resume_epoch"]) >= 10 for row in manifest["branches"])
    assert manifest["session"]["parallel_concurrency"] == 1


def test_beta3_dashboard_safe_parallel_cpu_capacity_matches_scheduler_minimum_budget():
    # 4 logical threads under Safe-80 -> 3 protected worker-equivalents, exactly one branch
    # (2 host-support equivalents + 1 rollout worker). Dashboard and planner must agree.
    topology = _topology(logical=4)
    profile = SafeResourceBudgetEngine(allocation_limit_fraction=0.80).calculate(topology)
    assert profile.safe_cpu_worker_budget == 3
    assert profile.cpu_support_branch_capacity == 1
    assert profile.safe_parallel_branches == 1
    cfg = TrainingConfig(ppo_device="cpu", parallel_concurrency=1)
    plan = build_training_resource_plan(cfg, 3, topology=topology, profile=profile)
    assert plan.simultaneous_branches == 1
    assert plan.global_cpu_worker_budget == 3
    assert plan.cpu_support_reserve_per_branch == 2
    assert plan.cpu_rollout_worker_budget_total == 1
