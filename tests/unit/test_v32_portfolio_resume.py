from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from calo_rpd_studio.algorithms.result import OptimizerResult
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.portfolio.exporter import PortfolioExporter
from calo_rpd_studio.portfolio.fingerprint import experiment_fingerprint
from calo_rpd_studio.portfolio.models import (
    EvidenceProfile,
    PortfolioConfig,
    PortfolioKind,
    StorageProfile,
)
from calo_rpd_studio.portfolio.planner import PortfolioPlanner
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.results.result_store import ResultStore
from calo_rpd_studio.resume.models import ResumeStatus, ResumeTaskType
from calo_rpd_studio.resume.service import ResumeService


def test_portfolio_drives_minimum_repetitions_and_disables_invalid_single_run_statistics():
    config = ExperimentConfig(algorithms=["CALO", "TLBO"])
    single = PortfolioConfig(
        kind=PortfolioKind.SINGLE_RUN,
        evidence_profile=EvidenceProfile.DIAGNOSTIC,
        requested_outputs=["voltage_profile", "objective_boxplot"],
        storage_profile=StorageProfile.FULL_SINGLE_RUN,
    )
    plan = PortfolioPlanner.plan(config, single)
    assert plan.required_runs == 1
    assert plan.total_jobs == 2
    assert plan.disabled_outputs["objective_boxplot"] == "Requires repeated independent runs"

    journal = PortfolioConfig(
        kind=PortfolioKind.OVERALL_EXPERIMENT,
        evidence_profile=EvidenceProfile.JOURNAL,
        requested_outputs=["objective_boxplot", "wilcoxon_holm"],
    )
    plan = PortfolioPlanner.plan(config, journal)
    assert plan.required_runs == 30
    assert plan.total_jobs == 60


def test_scientific_fingerprint_ignores_portfolio_and_operational_changes():
    left = ExperimentConfig(algorithms=["CALO", "TLBO"])
    right = ExperimentConfig.from_dict(left.to_dict())
    right.output_directory = "another/location"
    right.parallel_workers = 12
    right.portfolio = PortfolioConfig(
        kind=PortfolioKind.SINGLE_RUN,
        evidence_profile=EvidenceProfile.DIAGNOSTIC,
        requested_outputs=["voltage_profile"],
    )
    right.portfolio_id = "different-portfolio"
    assert experiment_fingerprint(left) == experiment_fingerprint(right)


def test_resume_service_marks_unclean_running_records_interrupted(tmp_path):
    database = ResultDatabase(tmp_path / "resume.sqlite")
    service = ResumeService(database, tmp_path / "checkpoints")
    task_id = service.register(
        ResumeTaskType.POLICY_TRAINING,
        "Training",
        {"epoch": 2},
        total=10,
        status=ResumeStatus.RUNNING,
    )
    summary = service.recover_after_restart()
    assert summary["resume_tasks"] == 1
    row = database.get_resumable_task(task_id)
    assert row["status"] == ResumeStatus.INTERRUPTED.value


def _fake_completed():
    result = OptimizerResult(
        algorithm="CALO",
        seed=10,
        parameters={},
        best_vector=np.asarray([0.25, 0.75]),
        decoded_controls={"vg_1": 1.01, "tap_1": 1.0},
        best_objective=2.1,
        objective_components={"active_power_loss": 2.1},
        total_constraint_violation=0.0,
        feasible=True,
        evaluations=100,
        iterations=4,
        convergence_history=[3.0, 2.5, 2.1],
        runtime_seconds=0.2,
        final_population=np.asarray([[0.25, 0.75], [0.3, 0.7]]),
        termination_reason="budget",
        metadata={
            "convergence_evaluations": [20, 60, 100],
            "best_feasible_objective_history": [3.0, 2.5, 2.1],
            "best_constraint_violation_history": [0.02, 0.01, 0.0],
            "constraint_component_histories": {"bus_voltage": [0.02, 0.01, 0.0]},
            "first_feasible_evaluation": 100,
            "regime_history": ["feasibility", "transition", "objective"],
            "operator_usage_history": [{"teacher": 5}, {"teacher": 4, "memory": 1}],
            "operator_success_history": [{"teacher": 0.4}, {"teacher": 0.5, "memory": 0.2}],
            "solution_state": {
                "scenarios": [
                    {
                        "bus_numbers": [1, 2],
                        "vm_pu": [1.0, 0.99],
                        "branch_from_bus": [1],
                        "branch_to_bus": [2],
                        "loading_percent": [42.0],
                        "qg_mvar": [5.0],
                        "generator_bus": [1],
                        "converged": True,
                        "active_loss_mw": 2.1,
                    }
                ]
            },
        },
    )
    return SimpleNamespace(
        algorithm="CALO",
        run_index=0,
        seeds=SimpleNamespace(algorithm_seed=10, scenario_seed=11, ai_inference_seed=12),
        result=result,
    )


def test_portfolio_export_resumes_artifact_by_artifact(tmp_path):
    database = ResultDatabase(tmp_path / "results.sqlite")
    config = ExperimentConfig(algorithms=["CALO"])
    config.portfolio = PortfolioConfig(
        kind=PortfolioKind.SINGLE_RUN,
        evidence_profile=EvidenceProfile.DIAGNOSTIC,
        requested_outputs=[
            "objective_convergence",
            "constraint_convergence",
            "constraint_decomposition",
            "voltage_profile",
            "calo_regime_timeline",
            "calo_operator_usage",
        ],
        require_independent_validation=False,
        storage_profile=StorageProfile.FULL_SINGLE_RUN,
    )
    config.validate()
    experiment_id = database.create_experiment(config, {})
    completed = _fake_completed()
    store = ResultStore(
        tmp_path / "arrays",
        storage_profile="full_single_run",
        required_fields={"constraint_components"},
    )
    arrays_path = store.save_arrays(completed.result)
    database.add_run(experiment_id, completed, str(arrays_path))

    output = tmp_path / "portfolio"
    PortfolioExporter(database).export(experiment_id, output)
    manifest = json.loads((output / "portfolio_manifest.json").read_text())
    assert manifest["artifacts"]["objective_convergence"]["status"] == "completed"
    assert (output / "figures" / "objective_convergence.png").is_file()
    mtime = (output / "figures" / "objective_convergence.png").stat().st_mtime_ns
    PortfolioExporter(database).export(experiment_id, output)
    assert (output / "figures" / "objective_convergence.png").stat().st_mtime_ns == mtime


def test_shared_reused_trace_is_deleted_only_after_last_reference(tmp_path):
    database = ResultDatabase(tmp_path / "shared.sqlite")
    config = ExperimentConfig(algorithms=["CALO"])
    first = database.create_experiment(config, {})
    second = database.create_experiment(config, {})
    completed = _fake_completed()
    path = ResultStore(tmp_path / "arrays").save_arrays(completed.result)
    source = database.add_run(first, completed, str(path), scientific_fingerprint="same")
    clone = database.clone_run_to_experiment(source, second)
    summary = database.delete_run(clone, compact=False)
    assert path.is_file()
    assert summary["trace_files_shared"] == 1
    database.delete_run(source, compact=False)
    assert not path.exists()
