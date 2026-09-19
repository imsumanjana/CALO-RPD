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
        "calo_rpd_studio/algorithms/calo/heterogeneous_training.py",
        "calo_rpd_studio/algorithms/calo/training.py",
        "calo_rpd_studio/algorithms/calo/competitive_training.py",
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "validate_policy_development" in text, relative

    # The current policy library no longer owns training. Exercise the separate
    # public boundary below instead of requiring retired combined-UI source text.


def test_policy_library_and_training_have_separate_public_entry_points():
    from calo_rpd_studio.gui.panels.calo_intelligence_panel import CALOIntelligencePanel
    from calo_rpd_studio.gui.panels.independent_training_panel import IndependentTrainingPanel

    assert not hasattr(CALOIntelligencePanel, "start_training")
    assert callable(CALOIntelligencePanel.qualify_selected_policy)
    assert callable(IndependentTrainingPanel.start_training)


def test_new_training_plan_does_not_validate_or_consume_an_experiment(monkeypatch):
    from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
        IndependentTSHCALOTrainingCampaign,
    )
    from calo_rpd_studio.gui.panels.independent_training_panel import TrainingLaunchModel

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Plan preparation must not consume an experiment or run training")

    monkeypatch.setattr(ExperimentConfig, "validate", forbidden)
    monkeypatch.setattr(ExperimentConfig, "validate_policy_development", forbidden)
    monkeypatch.setattr(IndependentTSHCALOTrainingCampaign, "start", forbidden)
    monkeypatch.setattr(IndependentTSHCALOTrainingCampaign, "resume", forbidden)
    monkeypatch.setattr(IndependentTSHCALOTrainingCampaign, "_execute", forbidden)
    monkeypatch.setattr(
        TrainingLaunchModel, "_current_source_commit", staticmethod(lambda: "a" * 40)
    )
    model = TrainingLaunchModel()
    model.create_plan(
        campaign_id="independent-plan-contract",
        development_cases=["case30"],
        member_count=3,
        master_seed=17,
        population_size=10,
        max_evaluations=100,
        requested_device="cpu",
        allow_cpu_fallback=False,
        training={},
    )
    assert model.plan_payload is not None, model.plan_error
    assert tuple(model.plan_payload["development_cases"]) == ("case30",)
    assert model.plan_payload["max_evaluations"] == 100
    assert "portfolio" not in model.plan_payload
    assert "study" not in model.plan_payload


@pytest.mark.parametrize(
    ("training_active", "busy", "ready"),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_training_start_rejects_busy_or_unchecked_inputs(monkeypatch, training_active, busy, ready):
    from calo_rpd_studio.gui.panels import independent_training_panel as panel

    warnings = []
    launched = []
    monkeypatch.setattr(panel.QMessageBox, "warning", lambda *_args: warnings.append(True))
    view = SimpleNamespace(
        process=None,
        state=SimpleNamespace(
            policy_training_active=training_active,
            task_status=SimpleNamespace(busy=busy),
        ),
        _validated_fingerprint="current" if ready else "obsolete",
        model=SimpleNamespace(fingerprint=lambda: "current"),
        start_button=SimpleNamespace(setEnabled=lambda _value: None),
        _start_process=lambda *_args: launched.append(True),
    )
    panel.IndependentTrainingPanel.start_training(view)
    assert warnings
    assert launched == []


def test_training_start_decline_does_not_launch_a_process(monkeypatch, tmp_path):
    from calo_rpd_studio.gui.panels import independent_training_panel as panel

    questions = []
    launched = []

    def decline(*_args):
        questions.append(True)
        return panel.QMessageBox.StandardButton.No

    monkeypatch.setattr(panel.QMessageBox, "question", decline)
    view = SimpleNamespace(
        process=None,
        state=SimpleNamespace(
            policy_training_active=False, task_status=SimpleNamespace(busy=False)
        ),
        _validated_fingerprint="current",
        model=SimpleNamespace(
            fingerprint=lambda: "current",
            missing=lambda **_kwargs: [],
            values={"output": str(tmp_path / "new-training")},
        ),
        resume=SimpleNamespace(isChecked=lambda: False),
        _extension_mode=False,
        _start_process=lambda *_args: launched.append(True),
    )
    panel.IndependentTrainingPanel.start_training(view)
    assert questions
    assert launched == []


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
