from __future__ import annotations

import numpy as np
import pytest

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.mathematical_reference import (
    MATHEMATICAL_REFERENCE_SCHEMA,
    NONCONVEX_RELAXATION_WARNING,
    SLSQPReferenceOptions,
    build_continuous_relaxation,
    solve_exhaustive_finite_lattice_reference,
    solve_slsqp_continuous_reference,
)
from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig
from calo_rpd_studio.scripts.run_mathematical_reference import (
    load_reference_problem,
    load_start_vector,
    write_reference_evidence,
)


def _all_discrete_problem(toy_case) -> ORPDProblem:
    return ORPDProblem(
        toy_case,
        ORPDProblemConfig(
            variables=ORPDVariableConfig(
                generator_voltages=False,
                transformer_taps=True,
                shunt_compensation=False,
                discrete_transformer_taps=True,
            )
        ),
    )


def test_continuous_relaxation_changes_only_lattice_semantics(toy_case):
    source = ORPDProblem(toy_case)
    assert source.config.variables.discrete_transformer_taps is True

    relaxed = build_continuous_relaxation(source)

    assert relaxed is not source
    assert relaxed.case.checksum() == source.case.checksum()
    assert relaxed.scenarios == source.scenarios
    assert relaxed.config.objective == source.config.objective
    assert relaxed.config.robust == source.config.robust
    assert relaxed.config.power_flow == source.config.power_flow
    assert relaxed.config.constraint_tolerances == source.config.constraint_tolerances
    assert relaxed.config.variables.discrete_transformer_taps is False
    assert relaxed.config.variables.discrete_shunts is False
    assert source.config.variables.discrete_transformer_taps is True
    assert any(variable.values for variable in source.decoder.variables)
    assert all(not variable.values for variable in relaxed.decoder.variables)


def test_slsqp_report_separates_local_relaxation_from_mixed_incumbent(toy_case):
    problem = ORPDProblem(toy_case)
    start = np.full(problem.dimension, 0.5)

    report = solve_slsqp_continuous_reference(
        problem,
        start,
        options=SLSQPReferenceOptions(max_iterations=4, function_tolerance=1e-7),
        run_independent_validation=False,
    )

    assert report.schema_version == MATHEMATICAL_REFERENCE_SCHEMA
    assert report.solver_backend == "SciPy"
    assert report.solver_algorithm == "SLSQP"
    assert report.reference_problem_fingerprint != report.source_problem_fingerprint
    assert report.reference_point.candidate_space == "continuous_relaxation"
    assert report.reference_point.claim == "local_nonconvex_continuous_relaxation_point_not_a_bound"
    assert report.mixed_variable_point is not None
    assert report.mixed_variable_point.candidate_space == "original_mixed_variable_formulation"
    assert report.mixed_variable_point.lattice_valid is True
    assert report.certified_lower_bound is None
    assert report.optimality_gap is None
    assert report.gap_claim_permitted is False
    assert report.exact_claim_scope is None
    assert report.warning == NONCONVEX_RELAXATION_WARNING
    assert report.accounting.common_evaluator_solver_calls > 0
    assert report.accounting.common_evaluator_validation_calls == 3
    assert report.accounting.independent_validation_requests == 0
    assert report.reference_point.independent_validation_status == "not_run"
    assert report.mixed_variable_point.independent_validation_status == "not_run"
    assert report.sha256() == report.sha256()
    assert report.to_dict()["certified_lower_bound"] is None


@pytest.mark.parametrize(
    "initial, message",
    [
        ([0.5], "shape"),
        ([float("nan")] * 3, "finite"),
        ([-0.1, 0.5, 0.5], r"\[0,1\]"),
    ],
)
def test_slsqp_rejects_ambiguous_or_invalid_start(toy_case, initial, message):
    problem = ORPDProblem(toy_case)
    assert problem.dimension == 3
    with pytest.raises(ValueError, match=message):
        solve_slsqp_continuous_reference(
            problem,
            initial,
            options=SLSQPReferenceOptions(max_iterations=1),
            run_independent_validation=False,
        )


def test_slsqp_options_reject_complex_step_that_common_evaluator_cannot_preserve():
    with pytest.raises(ValueError, match="2-point or 3-point"):
        SLSQPReferenceOptions(finite_difference_scheme="cs").validate()


def test_exhaustive_reference_is_exact_only_on_declared_all_discrete_lattice(toy_case):
    problem = _all_discrete_problem(toy_case)
    assert problem.dimension == 1
    declared = len(problem.decoder.variables[0].values)

    report = solve_exhaustive_finite_lattice_reference(
        problem,
        maximum_candidates=declared,
        run_independent_validation=False,
    )

    assert report.termination_success is True
    assert report.settings["declared_candidate_count"] == declared
    assert report.accounting.backend_objective_evaluations == declared
    assert report.accounting.common_evaluator_solver_calls == declared
    assert report.accounting.common_evaluator_validation_calls == 1
    assert report.reference_problem_fingerprint == report.source_problem_fingerprint
    assert report.reference_point.lattice_valid is True
    assert report.reference_point.feasible is True
    assert report.reference_point.claim == "exact_best_feasible_point_on_declared_finite_lattice"
    assert "complete declared finite lattice" in str(report.exact_claim_scope)
    assert report.certified_lower_bound is None
    assert report.optimality_gap is None
    assert report.gap_claim_permitted is False

    objectives = []
    for index in range(declared):
        representative = np.asarray([(index + 0.5) / declared])
        evaluation = problem.evaluate(representative)
        if evaluation.feasible:
            objectives.append(float(evaluation.value))
    assert report.reference_point.objective == pytest.approx(min(objectives), abs=1e-12)


def test_exhaustive_reference_rejects_continuous_or_oversized_tasks(toy_case):
    with pytest.raises(ValueError, match="rejects continuous controls"):
        solve_exhaustive_finite_lattice_reference(
            ORPDProblem(toy_case),
            run_independent_validation=False,
        )

    problem = _all_discrete_problem(toy_case)
    with pytest.raises(ValueError, match="exceeding ceiling"):
        solve_exhaustive_finite_lattice_reference(
            problem,
            maximum_candidates=1,
            run_independent_validation=False,
        )


def test_exhaustive_winner_can_be_independently_cross_validated(toy_case):
    problem = _all_discrete_problem(toy_case)
    report = solve_exhaustive_finite_lattice_reference(
        problem,
        maximum_candidates=100,
        run_independent_validation=True,
    )

    assert report.accounting.independent_validation_requests == 1
    assert report.reference_point.independent_validation_status == "passed"
    assert len(report.reference_point.independent_scenarios) == 1
    validation = report.reference_point.independent_scenarios[0]
    assert validation.available is True
    assert validation.passed is True
    assert validation.bus_type_mismatches == 0
    assert validation.q_limit_mismatches == 0


def test_reference_cli_inputs_are_hashed_strict_and_protected_cases_fail_closed(tmp_path):
    config = ExperimentConfig(case_name="case30")
    config_path = config.save(tmp_path / "task.json")
    problem, loaded, config_sha256 = load_reference_problem(config_path, scenario_seed=1701)

    assert loaded.case_name == "case30"
    assert problem.case.name == "case30"
    assert len(config_sha256) == 64

    start_path = tmp_path / "start.json"
    start_path.write_text("[" + ",".join(["0.5"] * problem.dimension) + "]\n", encoding="utf-8")
    start, start_sha256 = load_start_vector(start_path, dimension=problem.dimension)
    assert start.shape == (problem.dimension,)
    assert len(start_sha256) == 64

    start_path.write_text('{"normalized_vector": []}\n', encoding="utf-8")
    with pytest.raises(TypeError, match="JSON array"):
        load_start_vector(start_path, dimension=problem.dimension)

    protected = ExperimentConfig(case_name="case118")
    protected_path = protected.save(tmp_path / "protected.json")
    with pytest.raises(ValueError, match="refuses protected holdout"):
        load_reference_problem(protected_path, scenario_seed=1701)


def test_reference_evidence_writer_is_new_file_only(tmp_path):
    destination = tmp_path / "reference.json"
    payload = {"schema_version": "test", "value": 1}

    written = write_reference_evidence(destination, payload)

    assert written == destination
    assert destination.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_reference_evidence(destination, payload)
