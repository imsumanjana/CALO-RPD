import json
from pathlib import Path

import pytest

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.evaluation_budget import BudgetPolicy


def test_config_enum_roundtrip(tmp_path):
    c = ExperimentConfig()
    c.budget.policy = BudgetPolicy.EQUAL_WALL_CLOCK
    c.budget.wall_clock_seconds = 2.5
    path = c.save(tmp_path / "config.yaml")
    loaded = ExperimentConfig.load(path)
    assert loaded.budget.policy is BudgetPolicy.EQUAL_WALL_CLOCK
    assert loaded.budget.wall_clock_seconds == 2.5


def test_plan_bound_result_contract_round_trip() -> None:
    config = ExperimentConfig(
        execution_plan_kind="individual_experiment",
        result_contract={
            "schema_version": "calo-rpd-individual-result-contract-v1",
            "owner": "individual_experiment",
            "requested_outputs": ["objective_convergence"],
            "required_fields": [
                "convergence",
                "decoded_controls",
                "final_metrics",
                "seed_provenance",
            ],
            "storage_profile": "full_single_run",
            "reuse_compatible_results": True,
            "reuse_verified_only": True,
        },
    )

    restored = ExperimentConfig.from_dict(config.to_dict())

    assert restored.execution_plan_kind == "individual_experiment"
    assert restored.result_contract == config.result_contract
    restored.validate()


def test_workspace_study_runtime_contract_replaces_legacy_profile_run_floor() -> None:
    config = ExperimentConfig(runs=6, execution_plan_kind="workspace")
    config.workspace_study_contract = {
        "schema_version": "calo-rpd-workspace-study-runtime-contract-v1",
        "portfolio_goal_id": "portfolio-goal-1",
        "portfolio_goal_sha256": "a" * 64,
        "recommendation_id": "study-recommendation-1",
        "recommendation_sha256": "b" * 64,
        "study_setup_id": "study-setup-1",
        "study_setup_sha256": "c" * 64,
        "hard_minimum_runs": 5,
    }

    restored = ExperimentConfig.from_dict(config.to_dict())

    assert restored.workspace_study_contract == config.workspace_study_contract
    restored.validate()
    restored.runs = 4
    with pytest.raises(ValueError, match="portfolio-required minimum of 5"):
        restored.validate()


def test_config_roundtrip_uses_automatic_compute_defaults(tmp_path):
    config = ExperimentConfig.from_dict(
        {
            "execution_backend": "cuda_preferred",
            "gpu_utilization_target": 72,
            "cpu_utilization_target": 48,
            "gpu_memory_limit": 88,
            "gpu_parallel_jobs": 2,
            "system_memory_limit": 79,
            "cuda_cpu_fallback_enabled": False,
        }
    )
    path = config.save(tmp_path / "hybrid.yaml")
    loaded = ExperimentConfig.load(path)
    assert loaded.execution_backend == "cuda_preferred"
    assert not hasattr(loaded, "gpu_utilization_target")
    assert not hasattr(loaded, "cpu_utilization_target")
    assert not hasattr(loaded, "gpu_memory_limit")
    assert not hasattr(loaded, "gpu_parallel_jobs")
    assert not hasattr(loaded, "system_memory_limit")
    assert loaded.cuda_cpu_fallback_enabled is False
    saved = path.read_text(encoding="utf-8")
    assert "gpu_utilization_target" not in saved
    assert "cpu_utilization_target" not in saved
    assert "cuda_task_share" not in saved


def test_experiment_roundtrip_retains_saved_calo_and_tsh_calo_runtime_settings(tmp_path):
    config = ExperimentConfig(
        algorithms=["CALO", "PSO"],
        algorithm_parameters={
            "CALO": {
                "calo_profile": "custom",
                "use_ai": False,
                "strict_policy_binding": False,
                "epsilon_quantile": 0.8,
                "memory_capacity": 300,
                "checkpoint_interval_evaluations": 750,
            },
            "TSH-CALO": {
                "deterministic_policy": True,
                "inference_device": "cpu",
                "bandit_exploration": 0.42,
                "allow_cpu_fallback": False,
                "baseline_fallback_permitted": False,
            },
            "PSO": {"inertia": 0.71, "c1": 1.49618, "c2": 1.49618},
        },
    )

    path = config.save(tmp_path / "calo-settings.yaml")
    loaded = ExperimentConfig.load(path)

    assert loaded.algorithms == ["CALO", "PSO"]
    assert loaded.algorithm_parameters["CALO"]["epsilon_quantile"] == 0.8
    assert loaded.algorithm_parameters["CALO"]["memory_capacity"] == 300
    assert loaded.algorithm_parameters["CALO"]["checkpoint_interval_evaluations"] == 750
    assert loaded.algorithm_parameters["TSH-CALO"]["deterministic_policy"] is True
    assert loaded.algorithm_parameters["TSH-CALO"]["inference_device"] == "cpu"
    assert loaded.algorithm_parameters["TSH-CALO"]["bandit_exploration"] == 0.42
    assert loaded.algorithm_parameters["PSO"]["inertia"] == 0.71


@pytest.mark.parametrize(
    "legacy_mode",
    [
        "adaptive_hybrid",
        "weighted_split",
        "throughput_auto",
        "cuda_priority",
        "cuda_only",
        "gpu_preferred",
    ],
)
def test_legacy_cuda_scheduler_modes_migrate_to_cuda_preferred(legacy_mode):
    loaded = ExperimentConfig.from_dict(
        {
            "execution_backend": legacy_mode,
            "cuda_task_share": 35,
            "cpu_task_share": 65,
        }
    )
    assert loaded.execution_backend == "cuda_preferred"
    assert not hasattr(loaded, "cuda_task_share")
    assert not hasattr(loaded, "cpu_task_share")


def test_historical_xpu_mode_is_readable_but_view_only():
    config = ExperimentConfig.from_dict({"execution_backend": "xpu_priority"})
    assert config.execution_backend == "xpu_priority"
    with pytest.raises(ValueError, match="view-only"):
        config.validate()


def test_schema_contains_only_current_compute_modes_and_no_xpu_fields():
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "data"
        / "schemas"
        / "experiment_config.schema.json"
    )
    properties = json.loads(schema_path.read_text(encoding="utf-8"))["properties"]
    assert properties["execution_backend"]["enum"] == ["cuda_preferred", "cpu_only"]
    assert "cuda_task_share" not in properties
    assert "cpu_task_share" not in properties
    assert "gpu_utilization_target" not in properties
    assert "cpu_utilization_target" not in properties
    assert not any("xpu" in name.lower() for name in properties)


def test_population_requires_two_learners_in_code_and_schema():
    config = ExperimentConfig(population_size=1)
    with pytest.raises(ValueError, match="at least 2"):
        config.validate()

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "data"
        / "schemas"
        / "experiment_config.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["population_size"]["minimum"] == 2


def test_schema_matches_loader_unknown_field_and_throughput_defaults():
    from calo_rpd_studio.scripts.generate_experiment_schema import build_schema

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "data"
        / "schemas"
        / "experiment_config.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    default_profile = ExperimentConfig().throughput_profile_path

    assert schema["additionalProperties"] is False
    assert schema == build_schema()
    assert schema["properties"]["throughput_profile_path"]["default"] == default_profile
    assert set(ExperimentConfig().to_dict()) <= set(schema["properties"])
    with pytest.raises(ValueError, match="Unknown experiment configuration field"):
        ExperimentConfig.from_dict({"unexpected_scientific_field": 1})


def test_schema_declares_reviewed_external_control_identifiers_as_optional_arrays():
    from calo_rpd_studio.scripts.generate_experiment_schema import build_schema

    variable_properties = build_schema()["properties"]["variables"]["properties"]
    for field in ("generator_voltage_buses", "transformer_branch_indices"):
        assert variable_properties[field]["type"] == ["array", "null"]
        assert variable_properties[field]["items"] == {"type": "integer", "minimum": 0}
        assert variable_properties[field]["uniqueItems"] is True

    with pytest.raises(TypeError, match="integer bus numbers"):
        ExperimentConfig.from_dict({"variables": {"generator_voltage_buses": [2.5]}})
    with pytest.raises(TypeError, match="integer indices"):
        ExperimentConfig.from_dict({"variables": {"transformer_branch_indices": [True]}})
