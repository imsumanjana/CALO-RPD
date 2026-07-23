from __future__ import annotations

import json
from pathlib import Path

import pytest

from calo_rpd_studio.algorithms.calo.competitive_training import (
    build_branch_seed_plan,
    compare_champion_metrics,
)
from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy_parallel


def _tiny_config(**updates):
    values = dict(
        epochs=1,
        episodes_per_epoch=1,
        horizon=2,
        population_size=4,
        ppo_epochs=1,
        minibatch_size=4,
        hidden_dim=16,
        seed=100,
        rollout_workers=1,
        ppo_device="cpu",
        parallel_runs=2,
        parallel_same_seed_branches=1,
        parallel_incremental_branches=1,
        champion_validation_horizon=2,
        champion_validation_episodes=1,
        champion_min_feasible_rate=0.0,
    )
    values.update(updates)
    return TrainingConfig(**values)


def test_seed_plan_supports_same_increment_decrement_and_custom():
    cfg = TrainingConfig(
        seed=100,
        parallel_runs=7,
        parallel_same_seed_branches=2,
        parallel_incremental_branches=2,
        parallel_decremental_branches=2,
        parallel_custom_seeds=(575,),
    )
    plan = build_branch_seed_plan(cfg)
    assert [(item.seed, item.strategy) for item in plan] == [
        (100, "same"),
        (100, "same"),
        (101, "incremental"),
        (102, "incremental"),
        (99, "decremental"),
        (98, "decremental"),
        (575, "custom"),
    ]


def test_multi_metric_comparator_prioritizes_scientific_quality_over_runtime():
    incumbent = {
        "valid": True,
        "feasible_episode_rate": 1.0,
        "mean_final_feasible_ratio": 0.9,
        "median_final_feasible_objective": 5.0,
        "mean_final_feasible_objective": 5.1,
        "best_final_feasible_objective": 4.9,
        "worst_final_feasible_objective": 5.4,
        "convergence_auc": 6.0,
        "median_constraint_violation": 0.0,
        "median_steps_to_first_feasibility": 5.0,
        "mean_validation_return": 1.0,
        "median_validation_return": 1.0,
        "worst_validation_return": 0.5,
        "objective_iqr": 0.2,
        "policy_inference_ms": 0.2,
    }
    candidate = dict(incumbent)
    candidate.update(
        median_final_feasible_objective=4.7,
        mean_final_feasible_objective=4.8,
        best_final_feasible_objective=4.6,
        convergence_auc=5.6,
        objective_iqr=0.15,
        policy_inference_ms=0.4,  # slower, but scientifically better
    )
    decision = compare_champion_metrics(candidate, incumbent)
    assert decision.superior
    assert decision.wins > decision.losses


def test_multi_metric_comparator_rejects_material_feasibility_regression():
    incumbent = {"valid": True, "feasible_episode_rate": 1.0}
    candidate = {"valid": True, "feasible_episode_rate": 0.8}
    decision = compare_champion_metrics(candidate, incumbent)
    assert not decision.superior
    assert "feasible" in decision.reason


def test_competitive_training_creates_separate_branch_resume_states_and_one_base(tmp_path):
    output = tmp_path / "base.pt"
    path, history = train_policy_parallel(_tiny_config(), output, parallel_runs=2)
    manifest = json.loads(Path(path).with_suffix(".branches.json").read_text(encoding="utf-8"))
    assert manifest["session"]["method"] == "competitive independent PPO branches with protected queued scheduling and exact-resume indefinite rotation; no parameter averaging"
    assert manifest["session"]["queued_branch_scheduler"] is True
    assert len(manifest["branches"]) == 2
    # v5.9 separates synthetic Training Champion screening from deployable scientific Base
    # promotion. Without an exact real-ORPD development bundle, the selected artifact is
    # provisional and must not overwrite the logical deployable Base alias.
    assert manifest["base_artifact_path"] == ""
    assert Path(manifest["provisional_artifact_path"]).is_file()
    assert not Path(path).is_file()
    assert history
    for branch in manifest["branches"]:
        assert Path(branch["resume_path"]).is_file()
        assert Path(branch["resume_path"] + ".sha256").is_file()


def test_exact_competitive_resume_restores_same_branches_and_advances_epoch(tmp_path):
    output = tmp_path / "base.pt"
    train_policy_parallel(_tiny_config(), output, parallel_runs=2)
    first = json.loads(output.with_suffix(".branches.json").read_text(encoding="utf-8"))
    assert first["common_resume_epoch"] == 1
    first_generation_paths = [Path(row["resume_path"]) for row in first["branches"]]
    first_generation_bytes = [path.read_bytes() for path in first_generation_paths]

    resumed = _tiny_config(
        seed=999,
        parallel_runs=1,
        parallel_same_seed_branches=1,
        parallel_incremental_branches=0,
        parallel_start_mode="exact_resume",
    )
    train_policy_parallel(resumed, output, parallel_runs=1)
    second = json.loads(output.with_suffix(".branches.json").read_text(encoding="utf-8"))
    assert second["common_resume_epoch"] == 2
    assert [(b["branch_id"], b["seed"]) for b in second["branches"]] == [
        ("B01", 100),
        ("B02", 101),
    ]
    # v5.8 exact-resume continuation publishes a new immutable generation; the previous coherent
    # branch set remains byte-for-byte intact until/after the new authoritative manifest commit.
    assert [path.read_bytes() for path in first_generation_paths] == first_generation_bytes
    assert {str(Path(row["resume_path"]).parent) for row in first["branches"]} != {
        str(Path(row["resume_path"]).parent) for row in second["branches"]
    }


def test_base_guided_fork_starts_fresh_branches_without_mutating_parent(tmp_path):
    parent = tmp_path / "parent.pt"
    train_policy_parallel(
        _tiny_config(parallel_runs=1, parallel_same_seed_branches=1, parallel_incremental_branches=0),
        parent,
        parallel_runs=1,
    )
    parent_manifest = json.loads(parent.with_suffix(".branches.json").read_text(encoding="utf-8"))
    parent_artifact = Path(
        parent_manifest["base_artifact_path"] or parent_manifest["provisional_artifact_path"]
    )
    parent_bytes = parent_artifact.read_bytes()

    child = tmp_path / "child.pt"
    child_cfg = _tiny_config(
        seed=700,
        parallel_runs=2,
        parallel_same_seed_branches=1,
        parallel_incremental_branches=1,
        parallel_start_mode="base_guided_fork",
        base_model_checkpoint=str(parent_artifact),
    )
    train_policy_parallel(child_cfg, child, parallel_runs=2)
    child_manifest = json.loads(child.with_suffix(".branches.json").read_text(encoding="utf-8"))
    assert child_manifest["session"]["start_mode"] == "base_guided_fork"
    assert len(child_manifest["branches"]) == 2
    assert parent_artifact.read_bytes() == parent_bytes
    assert all(Path(row["resume_path"]).is_file() for row in child_manifest["branches"])


def test_safe_stop_commits_exact_branch_resume_states_and_cleans_scratch(tmp_path):
    output = tmp_path / "infinite.pt"
    scratch = tmp_path / "scratch"
    cfg = _tiny_config(
        training_mode="indefinite",
        max_session_epochs=100,
        parallel_runs=2,
        parallel_same_seed_branches=1,
        parallel_incremental_branches=1,
        training_scratch_dir=str(scratch),
        safe_snapshot_interval_epochs=10,
    )
    # Immediate Safe Stop is deterministic and should commit the common exact epoch-0 state rather
    # than leaving worker-specific partial trajectories or permanent intermediate snapshots.
    result = train_policy_parallel(cfg, output, parallel_runs=2, cancel_callback=lambda: True)
    path, _history = result
    assert str(getattr(result.status, "value", result.status)).startswith("SAFE_STOPPED")
    manifest = json.loads(Path(path).with_suffix(".branches.json").read_text(encoding="utf-8"))
    assert manifest["session"]["cancelled_safe_stop"] is True
    assert manifest["common_resume_epoch"] == 0
    assert {row["resume_epoch"] for row in manifest["branches"]} == {0}
    assert all(Path(row["resume_path"]).is_file() for row in manifest["branches"])
    # Session scratch is always deleted after permanent exact-state commit.
    assert not any(scratch.iterdir()) if scratch.exists() else True


def test_legacy_curriculum_encoding_conversion_is_schema_driven():
    from calo_rpd_studio.algorithms.calo.training import _stage_floor_from_history

    assert _stage_floor_from_history(
        [{"curriculum_stage": 1}], {"_resume_format": "calo_policy_training_resume_v41"}
    ) == 0
    assert _stage_floor_from_history(
        [{"curriculum_stage": 4}], {"_resume_format": "calo_policy_training_resume_v41"}
    ) == 3
    assert _stage_floor_from_history(
        [{"curriculum_stage": 5}], {"_resume_format": "calo_policy_training_resume_v41"}
    ) == 4
    assert _stage_floor_from_history(
        [{"curriculum_stage": 1}], {"_resume_format": "calo_policy_training_resume_v56"}
    ) == 1


def test_standalone_training_restores_caller_global_rng_state(tmp_path):
    import random
    import numpy as np
    import torch

    from calo_rpd_studio.algorithms.calo.training import train_policy

    random.seed(9101)
    np.random.seed(9102)
    torch.manual_seed(9103)
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    train_policy(
        TrainingConfig(
            epochs=1,
            episodes_per_epoch=1,
            horizon=2,
            population_size=4,
            ppo_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
            seed=17,
            rollout_workers=1,
            ppo_device="cpu",
        ),
        tmp_path / "rng-policy.pt",
    )

    assert random.getstate() == py_state
    after_np = np.random.get_state()
    assert after_np[0] == np_state[0]
    assert np.array_equal(after_np[1], np_state[1])
    assert after_np[2:] == np_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)
