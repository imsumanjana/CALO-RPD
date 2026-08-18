from dataclasses import replace

import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_parameter_study import (
    ParameterStudyFactor,
    ParameterStudyPlan,
    generate_parameter_design,
)


def _study(**changes) -> ParameterStudyPlan:
    base = ParameterStudyPlan(
        study_id="screening-1",
        base_execution_plan_sha256="a" * 64,
        development_cases=("case30", "case57"),
        factors=(
            ParameterStudyFactor("training.learning_rate", 1e-4, 1e-3),
            ParameterStudyFactor("training.ppo_epochs", 2, 8),
        ),
        design_method="latin_hypercube",
        design_points=8,
        independent_replicates=2,
        design_seed=17,
    )
    return replace(base, **changes)


def test_parameter_study_design_is_deterministic_and_blocked() -> None:
    plan = _study()
    first = generate_parameter_design(plan)
    second = generate_parameter_design(plan)
    assert first == second
    assert len(first["assignments"]) == 16
    assert {row["block_id"] for row in first["assignments"]} == {
        "replicate-01",
        "replicate-02",
    }
    assert first["protected_cases_opened"] is False


def test_parameter_study_rejects_protected_holdout() -> None:
    with pytest.raises(ValueError, match="Protected holdout"):
        _study(development_cases=("case30", "case118")).validate()


def test_sobol_requires_power_of_two_points() -> None:
    with pytest.raises(ValueError, match="power of two"):
        _study(design_method="sobol", design_points=6).validate()
