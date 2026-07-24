from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import inspect
import threading

import numpy as np
import pytest
import torch

from calo_rpd_studio.accelerated.torch_power_flow import (
    build_batched_admittance,
    build_dense_admittance,
    solve_newton_raphson_torch,
    _sbus as torch_sbus,
)
from calo_rpd_studio.ai.model_io import checkpoint_sha256
from calo_rpd_studio.algorithms.calo.ai_controller import (
    PolicyInferenceError,
    _PolicyInferenceBroker,
)
from calo_rpd_studio.algorithms.calo.device_resident_synthetic import (
    DeviceResidentCurriculumProblem,
    DeviceSyntheticEvaluation,
    SyntheticCrossEpisodeBatchBroker,
)
from calo_rpd_studio.algorithms.calo.policy_qualification import _paired_evidence
from calo_rpd_studio.algorithms.calo.run_checkpoint import (
    load_exact_run_checkpoint,
    save_exact_run_checkpoint,
)
from calo_rpd_studio.orpd.constraints import _normalized_above
from calo_rpd_studio.orpd.mixed_variable_handler import stepped_values
from calo_rpd_studio.power_system.ac_power_flow import _sbus as cpu_sbus, _types as cpu_types
from calo_rpd_studio.power_system.case_identity import (
    canonical_protected_holdout_checksums,
    protected_holdout_identity,
)
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.newton_raphson import solve_newton_raphson
from calo_rpd_studio.power_system.ybus import build_ybus
from calo_rpd_studio.results.database import ResultDatabase


def test_c04_torch_newton_matches_cpu_backtracking_on_stressed_start(toy_case):
    ref, pv, pq = cpu_types(toy_case)
    v0 = np.asarray([1.04 + 0j, 1.01 + 0j, 0.2 * np.exp(0.8j)], dtype=complex)
    cpu = solve_newton_raphson(
        build_ybus(toy_case).ybus,
        cpu_sbus(toy_case),
        v0,
        ref,
        pv,
        pq,
        1e-8,
        30,
    )
    ybus, _yf, _yt = build_dense_admittance(toy_case, "cpu", torch.float64)
    accelerated = solve_newton_raphson_torch(
        ybus,
        torch_sbus(toy_case, "cpu", torch.float64),
        torch.as_tensor(v0, dtype=torch.complex128),
        ref,
        pv,
        pq,
        tolerance=1e-8,
        max_iterations=30,
        minimum_damping=1.0 / 32.0,
    )
    assert cpu.converged and accelerated[0]
    assert accelerated[2] == cpu.iterations
    assert np.asarray(accelerated[4]) == pytest.approx(cpu.mismatch_history, rel=1e-11, abs=1e-12)
    assert np.max(np.abs(accelerated[1].detach().cpu().numpy() - cpu.voltage)) < 1e-11


def test_c10_stepped_values_never_exceed_declared_upper_bound():
    values = stepped_values(0.9, 1.1, 0.03)
    assert values[-1] == pytest.approx(1.08)
    assert all(value <= 1.1 + 1e-14 for value in values)
    assert stepped_values(0.9, 1.1, 0.1)[-1] == pytest.approx(1.1)


def test_h01_fixed_voltage_span_uses_stable_absolute_scale():
    violation = _normalized_above(
        np.asarray([1.000001]),
        np.asarray([1.0]),
        np.asarray([0.0]),
        absolute_tolerance=1e-7,
    )[0]
    assert violation == pytest.approx(1e-6, rel=1e-6)
    assert violation < 1e-3


def test_h05_single_and_batched_torch_reject_same_near_zero_active_impedance(toy_case):
    broken = toy_case.clone()
    broken.branch[0, 2] = 1e-14
    broken.branch[0, 3] = 0.0
    with pytest.raises(ValueError, match="zero/near-zero impedance"):
        build_dense_admittance(broken, "cpu", torch.float64)
    with pytest.raises(ValueError, match="zero/near-zero impedance"):
        build_batched_admittance([broken], "cpu", torch.float64)


def test_h15_near_zero_comparator_does_not_explode_relative_qualification_evidence():
    candidate = [{"case": "tiny", "run_index": i, "objective": 1e-9} for i in range(4)]
    comparator = [{"case": "tiny", "run_index": i, "objective": 0.0} for i in range(4)]
    evidence = _paired_evidence(candidate, comparator)
    assert evidence["median_relative_difference"] == pytest.approx(1e-9)
    assert max(abs(v) for v in evidence["paired_relative_differences"]) < 1e-6


def test_h21_h22_checkpoint_mutations_are_single_transaction_operations():
    delete_source = inspect.getsource(ResultDatabase.delete_policy_checkpoint)
    update_source = inspect.getsource(ResultDatabase.update_policy_checkpoint_qualification)
    assert 'BEGIN IMMEDIATE' in delete_source
    assert 'get_policy_checkpoint(' not in delete_source
    assert 'BEGIN IMMEDIATE' in update_source
    assert 'get_policy_checkpoint(' not in update_source


def test_h22_concurrent_metadata_updates_are_not_lost(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    lineage = db.create_policy_lineage("lineage")
    db.add_policy_checkpoint(
        checkpoint_id="cp",
        lineage_id=lineage,
        cumulative_epoch=1,
        phase_index=1,
        checkpoint_path="policy.pt",
        resume_path="",
        sha256="abc",
        is_latest=False,
    )
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(
                db.update_policy_checkpoint_qualification,
                "cp",
                qualification_status="candidate",
                grade="U",
                metadata_updates={f"k{i}": i},
            )
            for i in range(16)
        ]
        [future.result() for future in futures]
    metadata = db.get_policy_checkpoint("cp")["metadata"]
    assert all(metadata[f"k{i}"] == i for i in range(16))


def test_h28_comparison_applies_gui_before_fairness_gate_and_plan_build():
    source = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "gui"
        / "panels"
        / "experiment_manager_panel.py"
    ).read_text(encoding="utf-8")
    section = source[source.index("    def start_comparison"):source.index("    def start_calo", source.index("    def start_comparison"))]
    assert section.index("self.apply()") < section.index("if not self.fairness_passed")
    assert section.index("self.apply()") < section.index("labels_for_mode")


def test_h30_results_explorer_guards_json_and_null_run_id_fallback():
    source = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "gui"
        / "panels"
        / "results_explorer_panel.py"
    ).read_text(encoding="utf-8")
    assert "def _safe_json_object" in source
    assert 'row.get("run_id") or row["id"]' in source
    refresh_section = source[source.index("    def refresh("):source.index("    def show_selected")]
    assert "_safe_json_object" in refresh_section
    assert 'json.loads(row["result_json"])' not in refresh_section


def test_m18_policy_broker_close_releases_inflight_waiter_immediately():
    entered = threading.Event()
    release = threading.Event()

    class StallingPolicy(torch.nn.Module):
        def forward(self, x):
            entered.set()
            release.wait(timeout=3.0)
            batch = x.shape[0]
            return (
                torch.zeros((batch, 4)),
                torch.zeros((batch, 6)),
                torch.ones((batch, 6)),
                torch.ones((batch, 6)),
                torch.zeros((batch,)),
            )

    broker = _PolicyInferenceBroker(
        StallingPolicy(), torch.device("cpu"), request_timeout_s=5.0
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(broker.infer, np.zeros(32, dtype=np.float32))
        assert entered.wait(timeout=1.0)
        closer = pool.submit(broker.close)
        with pytest.raises(PolicyInferenceError, match="closed during request"):
            future.result(timeout=0.5)
        release.set()
        closer.result(timeout=2.0)


def test_v64_n01_synthetic_broker_close_releases_inflight_waiter(monkeypatch):
    import calo_rpd_studio.algorithms.calo.device_resident_synthetic as module

    entered = threading.Event()
    release = threading.Event()

    def stalled(_requests):
        entered.set()
        release.wait(timeout=3.0)
        return [[]]

    monkeypatch.setattr(module, "evaluate_device_resident_curriculum_batch", stalled)
    broker = SyntheticCrossEpisodeBatchBroker(
        device="cpu", batch_window_ms=0.0, request_timeout_s=5.0
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(broker.submit, object(), np.zeros((1, 2)))
        assert entered.wait(timeout=1.0)
        closer = pool.submit(broker.close)
        with pytest.raises(RuntimeError, match="Synthetic device-resident evaluation failed"):
            future.result(timeout=0.5)
        release.set()
        closer.result(timeout=2.0)


def test_m30_out_of_order_checkpoint_registration_cannot_replace_newer_latest(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    lineage = db.create_policy_lineage("lineage")
    common = dict(
        lineage_id=lineage,
        phase_index=1,
        checkpoint_path="policy.pt",
        resume_path="",
        sha256="abc",
        is_latest=True,
    )
    db.add_policy_checkpoint(checkpoint_id="epoch20", cumulative_epoch=20, **common)
    db.add_policy_checkpoint(checkpoint_id="epoch10", cumulative_epoch=10, **common)
    rows = {row["id"]: row for row in db.list_policy_checkpoints(lineage)}
    assert bool(rows["epoch20"]["is_latest"]) is True
    assert bool(rows["epoch10"]["is_latest"]) is False


def test_m32_exact_resume_is_self_authenticating_even_without_external_sidecar(tmp_path):
    path = tmp_path / "run.resume.pt"
    returned = save_exact_run_checkpoint(path, {"value": 42})
    assert returned == checkpoint_sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    assert sidecar.is_file()
    sidecar.unlink()
    loaded = load_exact_run_checkpoint(path)
    assert loaded["value"] == 42


def test_m45_checkpoint_hash_streams_without_path_read_bytes(tmp_path, monkeypatch):
    path = tmp_path / "large.bin"
    path.write_bytes(b"abc" * 1024 * 1024)

    def forbidden(_self):
        raise AssertionError("Path.read_bytes must not be used for checkpoint hashing")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    digest = checkpoint_sha256(path, chunk_size=64 * 1024)
    assert len(digest) == 64


def test_v64_n02_stage_b_parity_rejects_truncated_accelerated_result():
    wrapped = object.__new__(DeviceResidentCurriculumProblem)
    wrapped.reference = SimpleNamespace(
        evaluate=lambda _x: DeviceSyntheticEvaluation(
            value=1.0,
            feasible=True,
            violation=0.0,
            metadata={"constraint_components": {"a": 0.0}},
        )
    )
    wrapped._parity_max_error = 0.0
    wrapped._parity_verified = False
    wrapped.parity_tolerance = 1e-9
    with pytest.raises(RuntimeError, match="result length differs"):
        wrapped._verify_reference_parity(np.zeros((2, 1)), [wrapped.reference.evaluate(None)])


def test_v64_n03_renamed_protected_case_is_detected_by_canonical_checksum(monkeypatch):
    class FakeCase:
        def __init__(self, n_bus, digest):
            self.n_bus = n_bus
            self._digest = digest

        def checksum(self):
            return self._digest

    def fake_load(cls, source):
        name = str(source)
        if name == "case118":
            return FakeCase(118, "118-canonical")
        if name == "case300":
            return FakeCase(300, "300-canonical")
        if name.endswith("renamed_training_case.json"):
            return FakeCase(118, "118-canonical")
        return FakeCase(30, "other")

    monkeypatch.setattr(CaseLoader, "load", classmethod(fake_load))
    canonical_protected_holdout_checksums.cache_clear()
    try:
        assert protected_holdout_identity("renamed_training_case.json") == "case118"
    finally:
        canonical_protected_holdout_checksums.cache_clear()
