from __future__ import annotations

from pathlib import Path

from calo_rpd_studio.app.session_recovery import SessionRecoveryJournal
from calo_rpd_studio.app.workspaces import (
    WORKSPACE_SCHEMA_VERSION,
    migrate_workspace_ui,
)
from calo_rpd_studio.compute.governor import AdaptiveComputeGovernor, GovernorConfig, ProtectionState
from calo_rpd_studio.compute.provenance import ComputeProvenanceRecorder
from calo_rpd_studio.compute.resource_scheduler import DeviceSnapshot, ResourceSnapshot
from calo_rpd_studio.compute.scientific_equivalence import BranchScientificIdentity, scheduling_equivalent
from calo_rpd_studio.compute.soak import HardwareSoakRunner, SoakConfig
from calo_rpd_studio.compute.topology import ComputeProtectionProfile
from calo_rpd_studio.validation.gui_contract import validate_gui_contract


def _profile() -> ComputeProtectionProfile:
    return ComputeProtectionProfile(
        profile_name="Safe 80%",
        allocation_limit_fraction=0.8,
        reserve_fraction=0.2,
        ready=True,
        status="READY",
        safe_cpu_worker_budget=8,
        safe_ram_ceiling_bytes=8_000_000_000,
        safe_parallel_branches=2,
        accelerator_branch_slots=2,
        cpu_support_branch_capacity=2,
        ram_branch_capacity=4,
        estimated_branch_ram_bytes=1_000_000_000,
        estimated_branch_accelerator_bytes=1_000_000_000,
        topology_fingerprint="topo",
        profile_fingerprint="profile",
        reasons=(),
    )


def test_governor_hysteresis_and_red_safe_stop():
    governor = AdaptiveComputeGovernor(
        _profile(), config=GovernorConfig(amber_confirm_samples=1, red_confirm_samples=2, green_recovery_samples=1)
    )
    green = ResourceSnapshot(cpu_percent=20, system_memory_percent=20, sampled_at_monotonic=1)
    amber = ResourceSnapshot(cpu_percent=82, system_memory_percent=20, sampled_at_monotonic=2)
    red = ResourceSnapshot(cpu_percent=97, system_memory_percent=20, sampled_at_monotonic=3)
    assert governor.evaluate_snapshot(green).state is ProtectionState.GREEN
    assert governor.evaluate_snapshot(amber).state is ProtectionState.AMBER
    first_red = governor.evaluate_snapshot(red)
    assert first_red.allow_new_admission is False
    assert first_red.request_safe_stop is False
    second_red = governor.evaluate_snapshot(red)
    assert second_red.state is ProtectionState.RED
    assert second_red.request_safe_stop is True


def test_governor_uses_actual_temperature_only_when_present():
    governor = AdaptiveComputeGovernor(_profile(), config=GovernorConfig(red_confirm_samples=1))
    missing = ResourceSnapshot(cpu_percent=10, system_memory_percent=10, cpu_temperature_c=None, sampled_at_monotonic=1)
    assert not any("temperature" in r.lower() for r in governor.evaluate_snapshot(missing).reasons)
    hot = ResourceSnapshot(cpu_percent=10, system_memory_percent=10, cpu_temperature_c=99, sampled_at_monotonic=2)
    decision = governor.evaluate_snapshot(hot)
    assert decision.state is ProtectionState.RED
    assert any("temperature" in r.lower() for r in decision.reasons)


def test_governor_detects_device_power_and_temperature():
    governor = AdaptiveComputeGovernor(_profile(), config=GovernorConfig(red_confirm_samples=1))
    device = DeviceSnapshot(
        "cuda:0", "cuda", 0, "GPU", True, 50, 50, "nvidia-smi", "primary",
        temperature_c=89, power_w=99, power_limit_w=100,
    )
    decision = governor.evaluate_snapshot(ResourceSnapshot(20, (device,), 20, sampled_at_monotonic=1))
    assert decision.request_safe_stop


def test_workspace_migration_v59_v61_and_unknown():
    v59, report59 = migrate_workspace_ui({"workspace_index": 5, "workspace_schema_version": 1})
    assert v59["workspace_key"] == "calo_intelligence"
    assert v59["workspace_schema_version"] == WORKSPACE_SCHEMA_VERSION
    assert report59.migrated
    v61, report61 = migrate_workspace_ui({"workspace_key": "power_system", "workspace_schema_version": 2})
    assert v61["workspace_key"] == "power_system"
    assert report61.migrated
    bad, report_bad = migrate_workspace_ui({"workspace_key": "does_not_exist", "workspace_schema_version": 2})
    assert bad["workspace_key"] == "dashboard"
    assert report_bad.warning


def test_session_recovery_is_integrity_sealed_and_clean_shutdown_is_not_recoverable(tmp_path: Path):
    journal = SessionRecoveryJournal(tmp_path)
    journal.begin(workspace_ui={"workspace_key": "dashboard"})
    assert journal.previous_unclean() is not None
    payload = journal.mark_clean(workspace_ui={"workspace_key": "dashboard"})
    assert SessionRecoveryJournal.verify_payload(payload)
    assert journal.previous_unclean() is None


def test_compute_provenance_hash_chain_detects_tampering(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    recorder = ComputeProvenanceRecorder(path, session_id="x", metadata={"a": 1})
    recorder.append("STEP", {"b": 2})
    assert ComputeProvenanceRecorder.verify(path)["ok"]
    text = path.read_text(encoding="utf-8").replace('"b": 2', '"b": 3')
    path.write_text(text, encoding="utf-8")
    assert not ComputeProvenanceRecorder.verify(path)["ok"]


def test_scheduling_equivalence_ignores_wall_clock_order():
    a = [
        BranchScientificIdentity("B01", 11, "cfg", "epoch:100"),
        BranchScientificIdentity("B02", 12, "cfg", "epoch:100"),
    ]
    b = list(reversed(a))
    ok, details = scheduling_equivalent(a, b)
    assert ok, details
    changed = [a[0], BranchScientificIdentity("B02", 99, "cfg", "epoch:100")]
    assert not scheduling_equivalent(a, changed)[0]


def test_short_cpu_soak_is_protocol_validation_not_false_physical_qualification(tmp_path: Path):
    result = HardwareSoakRunner(
        SoakConfig(
            duration_seconds=0.08,
            sample_interval_seconds=0.02,
            backend="cpu",
            minimum_physical_qualification_seconds=1.0,
            workload_matrix_size=16,
        ),
        output_dir=tmp_path,
    ).run()
    assert result.exercised_backend == "cpu"
    assert not result.physical_qualified
    assert Path(result.provenance_path).is_file()


def test_dependency_light_gui_contract():
    root = Path(__file__).resolve().parents[2]
    result = validate_gui_contract(root)
    assert result["ok"], result
