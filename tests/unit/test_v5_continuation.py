from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.algorithms.calo.optimizer import CALOOptimizer
from calo_rpd_studio.algorithms.calo.policy_lineage import PolicyLineageManager
from calo_rpd_studio.algorithms.calo.training import TrainingConfig, _resolve_training_target
from calo_rpd_studio.continuation.experiment_evolution import (
    ExperimentEvolutionService,
    ExtensionProtocol,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.problem import Evaluation
from calo_rpd_studio.results.database import ResultDatabase


class CheckpointSphere:
    dimension = 4

    def __init__(self):
        self.case = SimpleNamespace(checksum=lambda: "checkpoint-sphere", name="checkpoint-sphere")
        self.decoder = SimpleNamespace(variables=[])
        self.physical_calls = 0

    def evaluate(self, x):
        self.physical_calls += 1
        vector = np.asarray(x, dtype=float)
        value = float(np.sum((vector - 0.25) ** 2))
        return Evaluation(value, True, 0.0, {}, {})

    def evaluate_population(self, population):
        return [self.evaluate(row) for row in population]

    def solution_state(self, x):
        return {"normalized_decision_vector": np.asarray(x).tolist(), "scenarios": []}


def test_training_target_modes_are_cumulative_additional_and_indefinite():
    config = TrainingConfig(epochs=100)
    config.training_mode = "cumulative"
    assert _resolve_training_target(config, 40) == (100, "cumulative")
    config.training_mode = "additional"
    assert _resolve_training_target(config, 40) == (140, "additional")
    config.training_mode = "indefinite"
    assert _resolve_training_target(config, 40) == (None, "indefinite")


def test_exact_policy_training_resume_matches_uninterrupted_same_target(tmp_path):
    import torch
    from calo_rpd_studio.algorithms.calo.training import TrainingCancelled, train_policy

    def make_config():
        return TrainingConfig(
            epochs=2,
            episodes_per_epoch=1,
            horizon=2,
            population_size=4,
            ppo_epochs=1,
            minibatch_size=4,
            hidden_dim=16,
            seed=31,
            rollout_workers=1,
            ppo_device="cpu",
            checkpoint_interval_epochs=1,
        )

    full_path, full_history = train_policy(make_config(), tmp_path / "full.pt")
    full_payload = torch.load(full_path, map_location="cpu", weights_only=False)

    stop = {"requested": False}

    def progress(_percent, message):
        if "PPO update" in str(message):
            stop["requested"] = True

    with pytest.raises(TrainingCancelled):
        train_policy(
            make_config(),
            tmp_path / "resumed.pt",
            progress_callback=progress,
            cancel_callback=lambda: stop["requested"],
        )

    resumed_config = make_config()
    resumed_config.resume_checkpoint = str((tmp_path / "resumed.pt").with_suffix(".resume.pt"))
    resumed_path, resumed_history = train_policy(resumed_config, tmp_path / "resumed.pt")
    resumed_payload = torch.load(resumed_path, map_location="cpu", weights_only=False)

    assert resumed_history == full_history
    assert resumed_payload["metadata"]["cumulative_epoch"] == 2
    for key, expected in full_payload["model_state_dict"].items():
        torch.testing.assert_close(
            resumed_payload["model_state_dict"][key], expected, rtol=0, atol=0
        )


def test_policy_lineage_tracks_latest_separately_from_best(tmp_path):
    import torch
    from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
    from calo_rpd_studio.algorithms.calo.policy_schema import POLICY_STATE_DIM

    db = ResultDatabase(tmp_path / "results.sqlite")
    manager = PolicyLineageManager(db)
    lineage = manager.create("P-test")
    model = CALOPolicyNetwork(POLICY_STATE_DIM, 96)
    paths = []
    for epoch in (10, 20):
        path = tmp_path / f"p_{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": 96},
                "metadata": {},
            },
            path,
        )
        paths.append(path)
    c10 = manager.register_checkpoint(lineage, paths[0], cumulative_epoch=10)
    c20 = manager.register_checkpoint(lineage, paths[1], cumulative_epoch=20)
    manager.mark_best(lineage, c10.id)
    assert manager.latest(lineage).id == c20.id
    assert manager.best(lineage).id == c10.id


def test_experiment_revisions_preserve_original_and_mark_manual_extension_exploratory(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig()
    config.runs = 3
    config.algorithms = ["CALO"]
    config.budget.max_evaluations = 100
    experiment_id = db.create_experiment(config, {}, campaign_status="completed")
    service = ExperimentEvolutionService(db)
    original = service.ensure_original_revision(experiment_id)
    db.update_experiment_revision(original["id"], status="completed")

    run_plan, run_config = service.extend_run_count(experiment_id, 5)
    assert run_plan.run_target == 5
    assert run_config.runs == 5
    assert run_plan.publication_eligible
    db.update_experiment_revision(run_plan.revision_id, status="completed")

    horizon_plan, horizon_config = service.extend_evaluation_horizon(
        experiment_id,
        200,
        protocol=ExtensionProtocol.MANUAL_EXPLORATORY,
        run_indices=(0,),
        algorithm_names=("CALO",),
    )
    assert horizon_config.budget.max_evaluations == 200
    assert not horizon_plan.publication_eligible
    revisions = db.list_experiment_revisions(experiment_id)
    assert [row["extension_mode"] for row in revisions] == [
        "original",
        "increase_run_count",
        "extend_evaluation_horizon",
    ]


def test_publication_eligible_horizon_subset_cannot_drop_algorithms(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig()
    config.algorithms = ["CALO", "TLBO"]
    config.runs = 4
    config.budget.max_evaluations = 100
    experiment_id = db.create_experiment(config, {}, campaign_status="completed")
    service = ExperimentEvolutionService(db)
    original = service.ensure_original_revision(experiment_id)
    db.update_experiment_revision(original["id"], status="completed")
    with pytest.raises(ValueError, match="every algorithm"):
        service.extend_evaluation_horizon(
            experiment_id,
            200,
            protocol=ExtensionProtocol.DETERMINISTIC_SUBSET,
            run_indices=(0, 2),
            algorithm_names=("CALO",),
        )


def test_calo_exact_horizon_resume_uses_only_additional_physical_calls(tmp_path):
    checkpoint = tmp_path / "calo_run.resume.pt"
    params = {
        "use_ai": False,
        "use_exact_evaluation_cache": False,
        "run_checkpoint_path": str(checkpoint),
        "checkpoint_interval_evaluations": 8,
    }
    first_problem = CheckpointSphere()
    first = CALOOptimizer(
        first_problem,
        OptimizerConfig(
            population_size=4, max_evaluations=20, max_iterations=100, parameters=params
        ),
        seed=7,
    ).run()
    assert first.evaluations == 20
    assert first_problem.physical_calls == 20
    assert checkpoint.is_file()
    assert checkpoint.with_suffix(checkpoint.suffix + ".sha256").is_file()

    resumed_params = dict(params)
    resumed_params["resume_run_checkpoint"] = str(checkpoint)
    second_problem = CheckpointSphere()
    second = CALOOptimizer(
        second_problem,
        OptimizerConfig(
            population_size=4, max_evaluations=36, max_iterations=100, parameters=resumed_params
        ),
        seed=7,
    ).run()
    assert second.evaluations == 36
    # Resume performs no hidden fresh-population solves: only the 16 additional requested FEs occur.
    assert second_problem.physical_calls == 16
    assert second.metadata["run_continuation"]["resumed_from"] == str(checkpoint)


def test_calo_interrupted_exact_resume_matches_uninterrupted_same_horizon(tmp_path):
    checkpoint = tmp_path / "interrupted.resume.pt"
    params = {
        "use_ai": False,
        "use_exact_evaluation_cache": False,
        "run_checkpoint_path": str(checkpoint),
        "checkpoint_interval_evaluations": 4,
    }

    interrupted_problem = CheckpointSphere()
    interrupted = CALOOptimizer(
        interrupted_problem,
        OptimizerConfig(
            population_size=4, max_evaluations=36, max_iterations=100, parameters=params
        ),
        seed=19,
        cancel_callback=lambda: interrupted_problem.physical_calls >= 20,
    ).run()
    assert interrupted.evaluations == 20
    assert checkpoint.is_file()

    resumed_parameters = dict(params)
    resumed_parameters["resume_run_checkpoint"] = str(checkpoint)
    resumed_problem = CheckpointSphere()
    resumed = CALOOptimizer(
        resumed_problem,
        OptimizerConfig(
            population_size=4, max_evaluations=36, max_iterations=100, parameters=resumed_parameters
        ),
        seed=19,
    ).run()
    assert resumed_problem.physical_calls == 16

    full_problem = CheckpointSphere()
    full_parameters = dict(params)
    full_parameters["run_checkpoint_path"] = ""
    uninterrupted = CALOOptimizer(
        full_problem,
        OptimizerConfig(
            population_size=4, max_evaluations=36, max_iterations=100, parameters=full_parameters
        ),
        seed=19,
    ).run()

    assert resumed.evaluations == uninterrupted.evaluations == 36
    assert resumed.best_objective == pytest.approx(uninterrupted.best_objective, abs=1e-15)
    np.testing.assert_allclose(resumed.best_vector, uninterrupted.best_vector, atol=0.0, rtol=0.0)
    assert (
        resumed.metadata["convergence_evaluations"]
        == uninterrupted.metadata["convergence_evaluations"]
    )
    np.testing.assert_allclose(
        resumed.metadata["incumbent_objective_history"],
        uninterrupted.metadata["incumbent_objective_history"],
        atol=0.0,
        rtol=0.0,
    )


def _fake_completed(*, algorithm="CALO", run_index=0, evaluations=10, objective=1.0):
    result = SimpleNamespace(
        algorithm=algorithm,
        seed=17,
        parameters={},
        best_vector=np.zeros(2, dtype=float),
        decoded_controls={},
        best_objective=float(objective),
        objective_components={},
        total_constraint_violation=0.0,
        feasible=True,
        evaluations=int(evaluations),
        iterations=1,
        convergence_history=[float(objective)],
        runtime_seconds=0.01,
        termination_reason="budget",
        metadata={},
    )
    seeds = SimpleNamespace(algorithm_seed=17, scenario_seed=23, ai_inference_seed=29)
    return SimpleNamespace(
        algorithm=algorithm, run_index=int(run_index), result=result, seeds=seeds
    )


def test_horizon_validation_is_attached_to_exact_evidence_only(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig()
    config.algorithms = ["CALO"]
    config.runs = 1
    config.budget.max_evaluations = 10
    experiment_id = db.create_experiment(config, {}, campaign_status="completed")
    run_id = db.add_run(
        experiment_id, _fake_completed(evaluations=10), "ten.npz", scientific_fingerprint="fp10"
    )
    db.add_validation(
        run_id, {"passed": True, "tag": "ten"}, evaluation_horizon=10, revision_id="r1"
    )
    db.snapshot_run_horizon(run_id, evaluation_horizon=10, revision_id="r1")
    db.update_run_result(
        run_id,
        _fake_completed(evaluations=20, objective=0.8),
        "twenty.npz",
        scientific_fingerprint="fp20",
    )

    # Revalidating preserved 10-FE evidence must not overwrite the current 20-FE validation state.
    db.add_validation(
        run_id, {"passed": False, "tag": "ten-recheck"}, evaluation_horizon=10, revision_id="r1"
    )
    assert db.get_run(run_id)["validation_status"] == "unverified"
    old = db.get_run_evidence_at_horizon(run_id, 10)
    current = db.get_run_evidence_at_horizon(run_id, 20)
    assert old["validation_status"] == "failed"
    assert len(old["validations"]) == 2
    assert current["validation_status"] == "unverified"
    assert db.available_run_horizons(run_id) == {10, 20}


def test_publication_horizon_can_branch_after_longer_exploratory_horizon(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig()
    config.algorithms = ["CALO", "TLBO"]
    config.runs = 2
    config.budget.max_evaluations = 100
    experiment_id = db.create_experiment(config, {}, campaign_status="completed")
    service = ExperimentEvolutionService(db)
    original = service.ensure_original_revision(experiment_id)
    db.update_experiment_revision(original["id"], status="completed")
    exploratory, _ = service.extend_evaluation_horizon(
        experiment_id,
        300,
        protocol=ExtensionProtocol.MANUAL_EXPLORATORY,
        run_indices=(0,),
        algorithm_names=("CALO",),
        execution_strategy="exact_continue",
    )
    db.update_experiment_revision(exploratory.revision_id, status="completed")

    # A post-hoc 300-FE exploratory branch must not prevent a later paired 200-FE primary revision.
    primary, cfg = service.extend_evaluation_horizon(
        experiment_id,
        200,
        protocol=ExtensionProtocol.ALL_PAIRED,
        execution_strategy="recompute_from_seed",
    )
    assert primary.publication_eligible is True
    assert primary.evaluation_target == 200
    assert cfg.extension_execution_strategy == "recompute_from_seed"
    row = db.get_experiment_revision(primary.revision_id)
    assert row["parent_revision_id"] == original["id"]


def test_run_count_extension_branches_from_primary_not_newer_exploratory_revision(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig()
    config.algorithms = ["CALO"]
    config.runs = 3
    config.budget.max_evaluations = 100
    experiment_id = db.create_experiment(config, {}, campaign_status="completed")
    service = ExperimentEvolutionService(db)
    original = service.ensure_original_revision(experiment_id)
    db.update_experiment_revision(original["id"], status="completed")
    exploratory, _ = service.extend_evaluation_horizon(
        experiment_id,
        300,
        protocol=ExtensionProtocol.MANUAL_EXPLORATORY,
        run_indices=(0,),
        algorithm_names=("CALO",),
        execution_strategy="exact_continue",
        source_horizon=100,
    )
    db.update_experiment_revision(exploratory.revision_id, status="completed")

    expanded, cfg = service.extend_run_count(experiment_id, 5)
    row = db.get_experiment_revision(expanded.revision_id)
    assert row["parent_revision_id"] == original["id"]
    assert expanded.evaluation_target == 100
    assert cfg.budget.max_evaluations == 100


def test_revision_scoped_run_checkpoint_paths_never_overwrite_prior_revision(tmp_path):
    from copy import deepcopy
    from calo_rpd_studio.continuation.runtime_binding import bind_exact_run_checkpoint

    base = ExperimentConfig()
    base.run_checkpoint_root = str(tmp_path / "checkpoints")
    item = SimpleNamespace(label="CALO", run_index=2)
    a = deepcopy(base)
    a.experiment_revision_id = "revision-A"
    b = deepcopy(base)
    b.experiment_revision_id = "revision-B"
    bind_exact_run_checkpoint(a, item)
    bind_exact_run_checkpoint(b, item)
    path_a = a.algorithm_parameters["CALO"]["run_checkpoint_path"]
    path_b = b.algorithm_parameters["CALO"]["run_checkpoint_path"]
    assert path_a != path_b
    assert "revision-A" in path_a and "revision-B" in path_b


def test_exact_policy_resume_blocks_scientific_hyperparameter_drift():
    from dataclasses import asdict
    from calo_rpd_studio.algorithms.calo.training import _validate_exact_resume_config

    original = TrainingConfig(epochs=100, learning_rate=3e-4)
    continued = TrainingConfig(epochs=1000, learning_rate=3e-4)
    continued.training_mode = "cumulative"
    # Increasing only the continuation target is valid.
    _validate_exact_resume_config(asdict(original), continued)
    changed = TrainingConfig(epochs=1000, learning_rate=1e-3)
    with pytest.raises(ValueError, match="learning_rate"):
        _validate_exact_resume_config(asdict(original), changed)


def test_exact_horizon_branch_records_explicit_source_and_parent_revision(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig()
    config.algorithms = ["CALO"]
    config.runs = 1
    config.budget.max_evaluations = 100
    experiment_id = db.create_experiment(config, {}, campaign_status="completed")
    service = ExperimentEvolutionService(db)
    original = service.ensure_original_revision(experiment_id)
    db.update_experiment_revision(original["id"], status="completed")
    first, _ = service.extend_evaluation_horizon(
        experiment_id,
        300,
        protocol=ExtensionProtocol.MANUAL_EXPLORATORY,
        run_indices=(0,),
        algorithm_names=("CALO",),
        execution_strategy="exact_continue",
        source_horizon=100,
    )
    db.update_experiment_revision(first.revision_id, status="completed")
    second, cfg = service.extend_evaluation_horizon(
        experiment_id,
        400,
        protocol=ExtensionProtocol.MANUAL_EXPLORATORY,
        run_indices=(0,),
        algorithm_names=("CALO",),
        execution_strategy="exact_continue",
        source_horizon=100,
    )
    row = db.get_experiment_revision(second.revision_id)
    assert cfg.extension_source_horizon == 100
    assert row["protocol"]["source_horizon"] == 100
    assert row["parent_revision_id"] == original["id"]


def test_v5_history_deletion_removes_revision_children_and_preserved_artifacts(tmp_path):
    db = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig()
    config.algorithms = ["CALO"]
    config.runs = 1
    config.budget.max_evaluations = 10
    experiment_id = db.create_experiment(config, {}, campaign_status="completed")
    revision = ExperimentEvolutionService(db).ensure_original_revision(experiment_id)
    db.update_experiment_revision(revision["id"], status="completed")
    old_array = tmp_path / "old.npz"
    old_array.write_bytes(b"old")
    new_array = tmp_path / "new.npz"
    new_array.write_bytes(b"new")
    checkpoint = tmp_path / "run.resume.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint.with_suffix(checkpoint.suffix + ".sha256").write_text("hash\n", encoding="utf-8")
    run_id = db.add_run(
        experiment_id,
        _fake_completed(evaluations=10),
        str(old_array),
        scientific_fingerprint="fp10",
    )
    db.snapshot_run_horizon(run_id, evaluation_horizon=10, revision_id=revision["id"])
    db.update_run_result(
        run_id, _fake_completed(evaluations=20), str(new_array), scientific_fingerprint="fp20"
    )
    db.add_run_segment(
        run_id=run_id,
        segment_index=0,
        start_evaluations=10,
        end_evaluations=20,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256="hash",
        metadata={"revision_id": revision["id"]},
    )
    summary = db.delete_experiment(experiment_id, compact=False)
    assert summary["experiments_deleted"] == 1
    assert db.get_experiment(experiment_id) is None
    assert db.get_run(run_id) is None
    assert db.list_run_horizon_snapshots(run_id) == []
    assert db.list_run_segments(run_id) == []
    assert not old_array.exists() and not new_array.exists() and not checkpoint.exists()
