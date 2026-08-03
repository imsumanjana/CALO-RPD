from pathlib import Path

import pytest
import torch

from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
    HeterogeneousTrainingConfig,
    _state_dict_sha256,
    plan_training_lanes,
    train_policy_heterogeneous,
)
from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork


def test_weighted_plan_uses_9_3_for_twelve_episodes():
    plan = plan_training_lanes(
        12,
        cuda_share=75,
        cpu_share=25,
        cuda_available=True,
    )
    assert plan.episode_counts == {"cuda": 9, "cpu": 3}
    assert plan.total_episodes == 12


def test_unavailable_cuda_is_redistributed_to_cpu():
    plan = plan_training_lanes(
        10,
        cuda_share=80,
        cpu_share=20,
        cuda_available=False,
    )
    assert plan.episode_counts == {"cuda": 0, "cpu": 10}
    assert len(plan.warnings) == 1


def test_rollout_shares_must_total_one_hundred():
    with pytest.raises(ValueError, match="exactly 100"):
        plan_training_lanes(
            10,
            cuda_share=60,
            cpu_share=20,
            cuda_available=True,
        )


def test_policy_snapshot_hash_detects_changes():
    network = CALOPolicyNetwork()
    first = _state_dict_sha256(network.state_dict())
    with torch.no_grad():
        next(network.parameters()).add_(1.0)
    second = _state_dict_sha256(network.state_dict())
    assert first != second


def test_cpu_fallback_heterogeneous_training_smoke(tmp_path: Path):
    output = tmp_path / "candidate.pt"
    config = HeterogeneousTrainingConfig(
        epochs=1,
        episodes_per_epoch=2,
        horizon=2,
        population_size=4,
        ppo_epochs=1,
        minibatch_size=4,
        rollout_workers=1,
        ppo_device="cpu",
        use_historical_trajectories=False,
    )
    path, history = train_policy_heterogeneous(config, output)
    assert Path(path).is_file()
    assert history[-1]["episode_allocation"]["cpu"] == 2
    payload = torch.load(path, map_location="cpu", weights_only=False)
    execution = payload["metadata"]["execution"]
    assert execution["architecture"].startswith("same-policy synchronous")
    assert payload["metadata"]["candidate_checkpoint"] is True
