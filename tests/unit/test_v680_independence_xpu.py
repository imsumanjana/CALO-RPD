from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from calo_bootstrap.launcher import accelerator_repair_required
from calo_rpd_studio.compute.resource_scheduler import ResourceSnapshot
from calo_rpd_studio.compute.topology import ComputeTopologyService
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_policy_development_validation_is_independent_of_portfolio_run_minimum():
    path = _root() / "calo_rpd_studio" / "data" / "examples" / "policy_development_active_loss.yaml"
    config = ExperimentConfig.load(path)
    assert config.runs == 1
    assert config.portfolio.required_runs() >= 30

    # The full Comparison/Portfolio contract correctly rejects one benchmark run.
    with pytest.raises(ValueError, match="portfolio-required minimum"):
        config.validate()

    # CALO Intelligence consumes only the scientific formulation and must remain independent.
    config.validate_policy_development()


def test_policy_training_paths_use_independent_scientific_validation():
    root = _root()
    for relative in (
        "calo_rpd_studio/gui/panels/calo_intelligence_panel.py",
        "calo_rpd_studio/algorithms/calo/heterogeneous_training.py",
        "calo_rpd_studio/algorithms/calo/training.py",
        "calo_rpd_studio/algorithms/calo/competitive_training.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "validate_policy_development" in text, relative

    panel_text = (root / "calo_rpd_studio/gui/panels/calo_intelligence_panel.py").read_text(
        encoding="utf-8"
    )
    assert "self.state.config_changed.connect(lambda config: self.load_from_config(config))" not in panel_text
    assert "PolicyQualifier(intelligence_template" in panel_text


def test_mixed_cuda_intel_host_requires_xpu_repair_even_when_cuda_is_ready():
    report = SimpleNamespace(
        nvidia=SimpleNamespace(detected=True),
        intel=SimpleNamespace(detected=True),
        torch=SimpleNamespace(
            cuda_available=True,
            gpu_test_passed=True,
            xpu_available=False,
            xpu_test_passed=False,
        ),
        xpu_sidecar=SimpleNamespace(xpu_available=False, gpu_test_passed=False),
    )
    assert accelerator_repair_required(report) is True

    report.xpu_sidecar.xpu_available = True
    report.xpu_sidecar.gpu_test_passed = True
    assert accelerator_repair_required(report) is False


def test_configured_xpu_interpreter_can_recover_without_trusting_state(monkeypatch, tmp_path):
    import calo_rpd_studio.compute.resource_scheduler as rs

    candidate = tmp_path / ("python.exe" if __import__("os").name == "nt" else "python")
    candidate.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(rs, "_xpu_interpreter_candidates", lambda: (candidate,))
    monkeypatch.setattr(rs, "_probe_xpu_interpreter", lambda path: path == candidate)
    monkeypatch.setattr(rs, "_XPU_INTERPRETER_CACHE", ("", 0.0))

    resolved = rs.configured_xpu_interpreter(force_refresh=True)
    assert Path(resolved) == candidate.resolve()


def test_topology_keeps_unmapped_intel_gpu_visible_as_repairable_xpu(monkeypatch):
    class Monitor:
        def sample(self):
            return ResourceSnapshot(cpu_percent=10.0, devices=(), system_memory_percent=20.0)

    adapters = [
        {
            "Name": "Intel(R) RaptorLake-S Mobile Graphics Controller",
            "PNPDeviceID": r"PCI\\VEN_8086&DEV_A7A0",
            "DriverVersion": "test-driver",
        },
        {
            "Name": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "PNPDeviceID": r"PCI\\VEN_10DE&DEV_28E0",
        },
    ]
    monkeypatch.setattr(ComputeTopologyService, "_windows_adapters", staticmethod(lambda: adapters))

    topology = ComputeTopologyService(Monitor()).scan()
    xpu = [device for device in topology.devices if device.backend == "xpu"]
    assert len(xpu) == 1
    assert xpu[0].runtime == "unavailable"
    assert xpu[0].runtime_id == ""
    assert xpu[0].policy_actor is False
    assert xpu[0].full_training_branch is False
    assert "runtime unavailable" in xpu[0].capability_status.lower()
