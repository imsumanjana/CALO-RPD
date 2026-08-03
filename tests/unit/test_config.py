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
