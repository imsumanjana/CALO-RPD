from __future__ import annotations

import pytest
from calo_rpd_studio.algorithms.calo.ai_controller import AIController
from calo_rpd_studio.algorithms.registry import SPECS
from calo_rpd_studio.app.state_manager import AppState


def test_missing_policy_never_constructs_an_untrained_fallback(tmp_path):
    with pytest.raises(RuntimeError, match="fail-closed"):
        AIController(None, seed=7, device="cpu")
    with pytest.raises(FileNotFoundError, match="explicitly imported/trained"):
        AIController(tmp_path / "missing.pt", seed=7, device="cpu")


def test_policy_intent_remains_strict_until_explicit_binding():
    assert SPECS["CALO"].default_parameters["use_ai"] is True
    assert SPECS["CALO"].default_parameters["strict_policy_binding"] is True


def test_empty_registry_startup_does_not_invent_a_policy(tmp_path):
    state = AppState(tmp_path / "no-policy.sqlite")
    assert state.policy_registry.list() == []
    assert state.governing_policy_status().ready is False
    assert state.policy_training_active is False
    assert state.config.algorithm_parameters["CALO"]["use_ai"] is False
