from pathlib import Path

import torch

from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
    HeterogeneousTrainingConfig,
    _state_dict_sha256,
    plan_training_lanes,
    train_policy_heterogeneous,
)
from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork


def test_weighted_plan_uses_6_4_2_for_twelve_episodes():
    plan = plan_training_lanes(
        12,
        cuda_share=50,
        xpu_share=30,
        cpu_share=20,
        cuda_available=True,
        xpu_available=True,
        xpu_sidecar_available=False,
    )
    assert plan.episode_counts == {"cuda": 6, "xpu": 4, "cpu": 2}
    assert plan.total_episodes == 12
    assert plan.xpu_runtime == "primary"


def test_unavailable_accelerators_are_redistributed_to_cpu():
    plan = plan_training_lanes(
        10,
        cuda_share=50,
        xpu_share=30,
        cpu_share=20,
        cuda_available=False,
        xpu_available=False,
        xpu_sidecar_available=False,
    )
    assert plan.episode_counts == {"cuda": 0, "xpu": 0, "cpu": 10}
    assert len(plan.warnings) == 2


def test_rollout_shares_must_total_one_hundred():
    try:
        plan_training_lanes(
            10,
            cuda_share=50,
            xpu_share=20,
            cpu_share=20,
            cuda_available=True,
            xpu_available=True,
        )
    except ValueError as exc:
        assert "exactly 100" in str(exc)
    else:
        raise AssertionError("Invalid shares were accepted")


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
