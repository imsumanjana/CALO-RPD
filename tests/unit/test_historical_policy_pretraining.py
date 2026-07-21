from __future__ import annotations

import json

import torch

from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy
from calo_rpd_studio.learning.experience_repository import REPOSITORY_SCHEMA_VERSION


def test_historical_policy_pretraining_is_separate_from_fresh_ppo(tmp_path):
    payload = {
        "schema_version": REPOSITORY_SCHEMA_VERSION,
        "created_at": "2026-01-01T00:00:00+00:00",
        "source_database": "test",
        "selection_policy": {
            "experiment_role": "train",
            "learning_eligible": True,
            "verified_only": True,
            "test_and_validation_experiments_excluded": True,
        },
        "summary": {},
        "policy_trajectories": [
            {
                "experiment_id": "e",
                "run_id": "r",
                "problem": {"case_name": "development", "dimension": 4},
                "validation_status": "verified",
                "transitions": [
                    {
                        "state": [0.0] * 24,
                        "regime": 1,
                        "operator": 2,
                        "parameter": [0.5] * 6,
                        "reward": 1.0,
                    },
                    {
                        "state": [0.1] * 24,
                        "regime": 2,
                        "operator": 4,
                        "parameter": [0.6] * 6,
                        "reward": 0.5,
                    },
                ],
            }
        ],
        "cross_algorithm_solutions": [],
        "parameter_priors": {},
    }
    from calo_rpd_studio.learning.experience_repository import _repository_checksum

    payload["repository_sha256"] = _repository_checksum(payload)
    repository_path = tmp_path / "experience.json"
    repository_path.write_text(json.dumps(payload), encoding="utf-8")

    path, _history = train_policy(
        TrainingConfig(
            epochs=1,
            episodes_per_epoch=1,
            horizon=2,
            population_size=4,
            ppo_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
            seed=31,
            historical_repository=str(repository_path),
            use_historical_trajectories=True,
            historical_pretraining_epochs=1,
        ),
        tmp_path / "policy.pt",
    )
    metadata = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
    historical = metadata["historical_pretraining"]
    assert historical["enabled"] is True
    assert historical["samples"] == 2
    assert historical["epochs"] == 1
    assert metadata["training_method"] == "PPO"
    assert (
        "old trajectories used only for offline pretraining" in metadata["historical_data_policy"]
    )
