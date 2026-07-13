from calo_rpd_studio.experiments.execution_plan import (
    ABLATION_MODE,
    COMPARISON_MODE,
    build_execution_plan,
    labels_for_mode,
    planned_item_count,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def test_primary_comparison_plan_uses_selected_algorithms():
    config = ExperimentConfig(algorithms=["CALO", "TLBO", "PSO"], runs=4)
    plan = build_execution_plan(config, COMPARISON_MODE)
    assert planned_item_count(config, COMPARISON_MODE) == 12
    assert len(plan) == 12
    assert labels_for_mode(config, COMPARISON_MODE) == ("CALO", "TLBO", "PSO")
    assert [item.label for item in plan[:3]] == ["CALO", "TLBO", "PSO"]
    assert all(item.ablation_spec is None for item in plan)


def test_calo_ablation_plan_is_fixed_and_independent_of_primary_selection():
    config = ExperimentConfig(algorithms=["CALO", "TLBO", "PSO"], runs=5)
    labels = labels_for_mode(config, ABLATION_MODE)
    plan = build_execution_plan(config, ABLATION_MODE)
    assert len(labels) == 7
    assert planned_item_count(config, ABLATION_MODE) == 35
    assert len(plan) == 35
    assert "Legacy Gaussian MTLBO" in labels
    assert all(item.ablation_spec is not None for item in plan)
