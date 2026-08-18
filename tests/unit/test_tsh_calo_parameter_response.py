import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_parameter_response import (
    LocalParameterResponsePlan,
    ParameterStudyObservation,
    summarize_robust_parameter_response,
)


def test_robust_response_requires_complete_case_replicate_blocks() -> None:
    rows = []
    for design in range(2):
        for replicate in range(2):
            for case in ("case30", "case57"):
                rows.append(
                    ParameterStudyObservation(
                        design_index=design,
                        replicate_index=replicate,
                        case_identity=case,
                        values={"training.learning_rate": 1e-4 * (design + 1)},
                        full_feasible=True,
                        final_objective=10.0 - design,
                        first_feasible_evaluations=40 + design,
                        convergence_auc=0.5,
                        runtime_seconds=1.0 + design,
                    )
                )
    result = summarize_robust_parameter_response(
        study_sha256="a" * 64,
        design_sha256="b" * 64,
        observations=rows,
        required_cases=("case30", "case57"),
        independent_replicates=2,
    )
    assert result["robust_order"] == [1, 0]
    assert result["automatic_parameter_change"] is False


def test_local_response_has_separate_fe_ledger_and_refuses_protected_state() -> None:
    plan = LocalParameterResponsePlan(
        analysis_id="local-1",
        source_run_id="run-1",
        policy_sha256="a" * 64,
        trajectory_row_sha256="b" * 64,
        rng_state_sha256="c" * 64,
        parameter="policy.exploration_sigma",
        candidate_values=(0.05, 0.10, 0.15),
        analysis_fe_budget_per_value=20,
        analysis_fe_ledger_id="analysis-ledger-1",
    )
    assert plan.to_dict()["official_experiment_fe_accounting"] is False
    with pytest.raises(ValueError, match="Protected holdout"):
        LocalParameterResponsePlan(
            analysis_id="local-2",
            source_run_id="run-1",
            policy_sha256="a" * 64,
            trajectory_row_sha256="b" * 64,
            rng_state_sha256="c" * 64,
            parameter="policy.exploration_sigma",
            candidate_values=(0.05, 0.10, 0.15),
            analysis_fe_budget_per_value=20,
            analysis_fe_ledger_id="analysis-ledger-2",
            protected_case=True,
        ).validate()


def test_local_response_rejects_training_hyperparameter() -> None:
    with pytest.raises(ValueError, match="adaptive policy parameters"):
        LocalParameterResponsePlan(
            analysis_id="local-training",
            source_run_id="run-1",
            policy_sha256="a" * 64,
            trajectory_row_sha256="b" * 64,
            rng_state_sha256="c" * 64,
            parameter="training.learning_rate",
            candidate_values=(1e-4, 2e-4, 3e-4),
            analysis_fe_budget_per_value=20,
            analysis_fe_ledger_id="analysis-ledger-training",
        ).validate()
