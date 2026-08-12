from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from calo_bootstrap.launcher import accelerator_repair_required
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_policy_development_validation_is_independent_of_portfolio_run_minimum():
    path = _root() / "calo_rpd_studio" / "data" / "examples" / "policy_training_active_loss.yaml"
    config = ExperimentConfig.load(path)
    assert config.runs == 1
    assert config.portfolio.required_runs() >= 30

    with pytest.raises(ValueError, match="portfolio-required minimum"):
        config.validate()

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
    assert (
        "self.state.config_changed.connect(lambda config: self.load_from_config(config))"
        not in panel_text
    )
    assert "PolicyQualifier(intelligence_template" in panel_text


def test_detected_nvidia_without_verified_cuda_requires_repair():
    report = SimpleNamespace(
        nvidia=SimpleNamespace(detected=True),
        torch=SimpleNamespace(cuda_available=False, gpu_test_passed=False),
    )
    assert accelerator_repair_required(report) is True

    report.torch.cuda_available = True
    report.torch.gpu_test_passed = True
    assert accelerator_repair_required(report) is False


def test_cpu_only_host_does_not_require_accelerator_repair():
    report = SimpleNamespace(
        nvidia=SimpleNamespace(detected=False),
        torch=SimpleNamespace(cuda_available=False, gpu_test_passed=False),
    )
    assert accelerator_repair_required(report) is False


def test_active_governing_policy_is_automatically_bound_to_new_experiments(monkeypatch, tmp_path):
    from calo_rpd_studio.app.state_manager import AppState

    state = AppState(tmp_path / "policy-binding.sqlite")
    parameters = state.config.algorithm_parameters.setdefault("CALO", {})
    parameters.update(
        {
            "use_ai": True,
            "deterministic_policy": False,
            "policy_id": "stale",
            "policy_sha256": "stale-sha",
            "strict_policy_binding": True,
        }
    )
    state.config.algorithm_parameters["TSH-CALO"] = {"deterministic_policy": False}
    unavailable = SimpleNamespace(ready=False)
    assert state.synchronize_governing_policy_binding(unavailable) is True
    assert "policy_id" not in state.config.algorithm_parameters["CALO"]
    assert state.config.algorithm_parameters["CALO"]["strict_policy_binding"] is False

    ready = SimpleNamespace(ready=True, policy_id="governing-policy", algorithm_id="TSH-CALO")

    def bind(policy_id, config, *, deterministic, allow_unqualified, algorithm_id):
        assert policy_id == "governing-policy"
        assert deterministic is False
        assert allow_unqualified is False
        assert algorithm_id == "TSH-CALO"
        config.algorithm_parameters["TSH-CALO"].update(
            {
                "policy_id": policy_id,
                "policy_sha256": "verified-sha",
                "strict_policy_binding": True,
                "allow_cpu_fallback": False,
                "baseline_fallback_permitted": False,
            }
        )
        return dict(config.algorithm_parameters["TSH-CALO"])

    monkeypatch.setattr(state.policy_registry, "bind_to_experiment_config", bind)
    assert state.synchronize_governing_policy_binding(ready) is True
    bound = state.config.algorithm_parameters["TSH-CALO"]
    assert bound["policy_id"] == "governing-policy"
    assert bound["policy_sha256"] == "verified-sha"
    assert bound["strict_policy_binding"] is True
    assert state.config.algorithm_parameters["CALO"]["use_ai"] is False
