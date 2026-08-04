"""Disclosed mathematical-reference adapters for ORPD evidence.

The adapters in this module are intentionally outside the stochastic optimizer registry.  They do
not receive an artificial black-box FE budget and they never turn a local nonconvex solve into an
optimality or lower-bound claim.  Every returned point is re-evaluated with the common ORPD
evaluator; optional PYPOWER checks use the repository's independent AC power-flow validator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import itertools
import json
import math
from typing import Any

import numpy as np

from calo_rpd_studio.orpd.decision_variables import VariableKind
from calo_rpd_studio.orpd.feasibility_rules import better
from calo_rpd_studio.orpd.formulation_fingerprint import scientific_problem_fingerprint
from calo_rpd_studio.orpd.problem import Evaluation, ORPDProblem
from calo_rpd_studio.power_system.independent_validator import validate_against_pypower


MATHEMATICAL_REFERENCE_SCHEMA = "calo-rpd-mathematical-reference-v1"
CONTINUOUS_RELAXATION_RELATION = (
    "same case, scenarios, objective, AC equations, constraints, tolerances and declared controls; "
    "only transformer-tap and shunt lattice snapping is removed"
)
NONCONVEX_RELAXATION_WARNING = (
    "SLSQP returns a local point for a nonconvex AC continuous relaxation. It is not a certified "
    "lower bound, global optimum, or mixed-variable feasible solution."
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


@dataclass(frozen=True, slots=True)
class SLSQPReferenceOptions:
    """Frozen, fully disclosed local-solver settings."""

    max_iterations: int = 200
    function_tolerance: float = 1e-9
    finite_difference_scheme: str = "2-point"
    failure_objective: float = 1e30
    failure_constraint_margin: float = 1e6

    def validate(self) -> None:
        if isinstance(self.max_iterations, bool) or int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not math.isfinite(float(self.function_tolerance)) or self.function_tolerance <= 0.0:
            raise ValueError("function_tolerance must be finite and positive")
        if self.finite_difference_scheme not in {"2-point", "3-point"}:
            raise ValueError("finite_difference_scheme must be 2-point or 3-point")
        if not math.isfinite(float(self.failure_objective)) or self.failure_objective <= 0.0:
            raise ValueError("failure_objective must be finite and positive")
        if (
            not math.isfinite(float(self.failure_constraint_margin))
            or self.failure_constraint_margin <= 0.0
        ):
            raise ValueError("failure_constraint_margin must be finite and positive")


@dataclass(frozen=True, slots=True)
class IndependentScenarioValidation:
    scenario: str
    available: bool
    passed: bool
    message: str
    max_vm_difference: float
    max_va_difference_deg: float
    loss_difference_mw: float
    bus_type_mismatches: int
    q_limit_mismatches: int
    max_q_difference_mvar: float


@dataclass(frozen=True, slots=True)
class ReferencePoint:
    candidate_space: str
    claim: str
    normalized_vector: tuple[float, ...]
    decoded_controls: dict[str, float]
    objective: float
    feasible: bool
    violation: float
    lattice_valid: bool | None
    common_evaluator_verified: bool
    independent_validation_status: str
    independent_scenarios: tuple[IndependentScenarioValidation, ...]


@dataclass(frozen=True, slots=True)
class SolverAccounting:
    backend_objective_evaluations: int
    backend_derivative_evaluations: int
    backend_iterations: int
    common_evaluator_solver_calls: int
    common_evaluator_cache_hits: int
    common_evaluator_validation_calls: int
    independent_validation_requests: int


@dataclass(frozen=True, slots=True)
class MathematicalReferenceReport:
    schema_version: str
    report_kind: str
    source_problem_fingerprint: str
    reference_problem_fingerprint: str
    formulation_relation: str
    solver_backend: str
    solver_backend_version: str
    solver_algorithm: str
    derivative_mode: str
    deterministic_start: tuple[float, ...] | None
    settings: dict[str, Any]
    termination_success: bool
    termination_status: int
    termination_message: str
    accounting: SolverAccounting
    reference_point: ReferencePoint
    mixed_variable_point: ReferencePoint | None
    certified_lower_bound: float | None
    optimality_gap: float | None
    gap_claim_permitted: bool
    exact_claim_scope: str | None
    warning: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class _EvaluationCache:
    def __init__(self, problem: ORPDProblem):
        self.problem = problem
        self._values: dict[bytes, Evaluation] = {}
        self.calls = 0
        self.hits = 0

    def evaluate(self, normalized: np.ndarray) -> Evaluation:
        candidate = np.clip(np.asarray(normalized, dtype=np.float64), 0.0, 1.0)
        key = np.ascontiguousarray(candidate).tobytes()
        cached = self._values.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        result = self.problem.evaluate(candidate)
        self.calls += 1
        self._values[key] = result
        return result


def build_continuous_relaxation(problem: ORPDProblem) -> ORPDProblem:
    """Construct the exact declared-control relaxation without mutating ``problem``.

    This removes only tap and shunt snapping.  Because AC ORPD remains nonconvex, a local solution
    of this relaxation is not automatically a valid lower bound.
    """

    if not isinstance(problem, ORPDProblem):
        raise TypeError("continuous relaxation requires an ORPDProblem")
    variables = replace(
        problem.config.variables,
        discrete_transformer_taps=False,
        discrete_shunts=False,
    )
    config = replace(problem.config, variables=variables)
    return ORPDProblem(problem.case, config=config, scenarios=tuple(problem.scenarios))


def _validate_start(problem: ORPDProblem, initial: np.ndarray | list[float]) -> np.ndarray:
    candidate = np.asarray(initial, dtype=float)
    if candidate.shape != (problem.dimension,):
        raise ValueError(
            f"Expected deterministic start shape ({problem.dimension},), got {candidate.shape}"
        )
    if not np.all(np.isfinite(candidate)):
        raise ValueError("Deterministic start must contain only finite values")
    if np.any((candidate < 0.0) | (candidate > 1.0)):
        raise ValueError("Deterministic start must lie in normalized [0,1] bounds")
    return candidate.astype(np.float64, copy=True)


def _lattice_valid(problem: ORPDProblem, evaluation: Evaluation) -> bool:
    for variable in problem.decoder.variables:
        if variable.kind is not VariableKind.DISCRETE:
            continue
        value = float(evaluation.physical_controls[variable.name])
        if not any(
            math.isclose(value, float(item), rel_tol=0.0, abs_tol=1e-12) for item in variable.values
        ):
            return False
    return True


def _independent_validations(
    problem: ORPDProblem,
    normalized: np.ndarray,
    contexts,
) -> tuple[IndependentScenarioValidation, ...]:
    controlled, _physical = problem.decoder.decode(normalized)
    records: list[IndependentScenarioValidation] = []
    for scenario, context in zip(problem.scenarios, contexts.scenarios):
        formulation_case = scenario.apply(controlled, copy_base=True)
        result = validate_against_pypower(
            formulation_case,
            context.power_flow,
            power_flow_options=problem.config.power_flow,
        )
        records.append(
            IndependentScenarioValidation(
                scenario=str(scenario.name),
                available=bool(result.available),
                passed=bool(result.passed),
                message=str(result.message),
                max_vm_difference=float(result.max_vm_difference),
                max_va_difference_deg=float(result.max_va_difference_deg),
                loss_difference_mw=float(result.loss_difference_mw),
                bus_type_mismatches=int(result.bus_type_mismatches),
                q_limit_mismatches=int(result.q_limit_mismatches),
                max_q_difference_mvar=float(result.max_q_difference_mvar),
            )
        )
    return tuple(records)


def _reference_point(
    problem: ORPDProblem,
    normalized: np.ndarray,
    *,
    candidate_space: str,
    claim: str,
    require_lattice: bool,
    run_independent_validation: bool,
) -> ReferencePoint:
    evaluation, contexts = problem.evaluate_with_context(normalized)
    lattice_valid = _lattice_valid(problem, evaluation) if require_lattice else None
    independent = (
        _independent_validations(problem, normalized, contexts)
        if run_independent_validation
        else ()
    )
    if not run_independent_validation:
        independent_status = "not_run"
    elif not independent or not all(record.available for record in independent):
        independent_status = "unavailable"
    elif all(record.passed for record in independent):
        independent_status = "passed"
    else:
        independent_status = "failed"
    return ReferencePoint(
        candidate_space=candidate_space,
        claim=claim,
        normalized_vector=tuple(float(value) for value in normalized),
        decoded_controls={
            str(key): float(value) for key, value in evaluation.physical_controls.items()
        },
        objective=float(evaluation.value),
        feasible=bool(evaluation.feasible),
        violation=float(evaluation.violation),
        lattice_valid=lattice_valid,
        common_evaluator_verified=True,
        independent_validation_status=independent_status,
        independent_scenarios=independent,
    )


def solve_slsqp_continuous_reference(
    problem: ORPDProblem,
    initial: np.ndarray | list[float],
    *,
    options: SLSQPReferenceOptions | None = None,
    run_independent_validation: bool = True,
) -> MathematicalReferenceReport:
    """Solve a local continuous relaxation and separately audit its mixed-variable projection."""

    settings = options or SLSQPReferenceOptions()
    settings.validate()
    if problem.dimension < 1:
        raise ValueError("SLSQP continuous reference requires at least one declared control")
    start = _validate_start(problem, initial)
    relaxed = build_continuous_relaxation(problem)
    cache = _EvaluationCache(relaxed)
    tolerance = float(relaxed.config.constraint_tolerances.feasibility_total)

    def objective(candidate: np.ndarray) -> float:
        value = float(cache.evaluate(candidate).value)
        return value if math.isfinite(value) else float(settings.failure_objective)

    def feasible_margin(candidate: np.ndarray) -> float:
        violation = float(cache.evaluate(candidate).violation)
        if not math.isfinite(violation):
            return -float(settings.failure_constraint_margin)
        return tolerance - violation

    from scipy.optimize import minimize

    result = minimize(
        objective,
        start,
        method="SLSQP",
        jac=settings.finite_difference_scheme,
        bounds=[(0.0, 1.0)] * problem.dimension,
        constraints=({"type": "ineq", "fun": feasible_margin},),
        options={
            "maxiter": int(settings.max_iterations),
            "ftol": float(settings.function_tolerance),
            "disp": False,
        },
    )
    candidate = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    relaxed_point = _reference_point(
        relaxed,
        candidate,
        candidate_space="continuous_relaxation",
        claim="local_nonconvex_continuous_relaxation_point_not_a_bound",
        require_lattice=False,
        run_independent_validation=run_independent_validation,
    )
    mixed_evaluation = problem.evaluate(candidate)
    mixed_claim = (
        "feasible_mixed_variable_incumbent"
        if mixed_evaluation.feasible and _lattice_valid(problem, mixed_evaluation)
        else "projected_mixed_variable_candidate_not_feasible"
    )
    mixed_point = _reference_point(
        problem,
        candidate,
        candidate_space="original_mixed_variable_formulation",
        claim=mixed_claim,
        require_lattice=True,
        run_independent_validation=run_independent_validation,
    )
    independent_requests = len(relaxed_point.independent_scenarios) + len(
        mixed_point.independent_scenarios
    )
    try:
        scipy_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        scipy_version = "unavailable"
    return MathematicalReferenceReport(
        schema_version=MATHEMATICAL_REFERENCE_SCHEMA,
        report_kind="slsqp_continuous_relaxation_and_mixed_projection",
        source_problem_fingerprint=scientific_problem_fingerprint(problem),
        reference_problem_fingerprint=scientific_problem_fingerprint(relaxed),
        formulation_relation=CONTINUOUS_RELAXATION_RELATION,
        solver_backend="SciPy",
        solver_backend_version=scipy_version,
        solver_algorithm="SLSQP",
        derivative_mode=(
            f"SciPy {settings.finite_difference_scheme} numerical objective derivatives; "
            "constraint derivatives selected by the SLSQP interface"
        ),
        deterministic_start=tuple(float(value) for value in start),
        settings=asdict(settings),
        termination_success=bool(result.success),
        termination_status=int(result.status),
        termination_message=str(result.message),
        accounting=SolverAccounting(
            backend_objective_evaluations=int(getattr(result, "nfev", 0)),
            backend_derivative_evaluations=int(getattr(result, "njev", 0)),
            backend_iterations=int(getattr(result, "nit", 0)),
            common_evaluator_solver_calls=int(cache.calls),
            common_evaluator_cache_hits=int(cache.hits),
            # One relaxed validation plus the explicit mixed classification and its retained
            # validation evaluation. The classification call is intentionally not hidden.
            common_evaluator_validation_calls=3,
            independent_validation_requests=independent_requests,
        ),
        reference_point=relaxed_point,
        mixed_variable_point=mixed_point,
        certified_lower_bound=None,
        optimality_gap=None,
        gap_claim_permitted=False,
        exact_claim_scope=None,
        warning=NONCONVEX_RELAXATION_WARNING,
    )


def _discrete_representatives(variable) -> tuple[float, ...]:
    count = len(variable.values)
    if count < 1:
        raise ValueError(f"Discrete variable {variable.name!r} has no declared lattice")
    return tuple((index + 0.5) / count for index in range(count))


def solve_exhaustive_finite_lattice_reference(
    problem: ORPDProblem,
    *,
    maximum_candidates: int = 10_000,
    run_independent_validation: bool = True,
) -> MathematicalReferenceReport:
    """Exhaust an all-discrete declared lattice under an explicit size ceiling.

    Any continuous dimension is rejected.  A successful result is exact only for the declared
    finite lattice and the common deterministic evaluator; it is not a continuous-ORPD bound.
    """

    if isinstance(maximum_candidates, bool) or int(maximum_candidates) < 1:
        raise ValueError("maximum_candidates must be a positive integer")
    variables = tuple(problem.decoder.variables)
    continuous = [item.name for item in variables if item.kind is not VariableKind.DISCRETE]
    if continuous:
        raise ValueError(
            "Exact finite-lattice enumeration rejects continuous controls: " + ", ".join(continuous)
        )
    representatives = tuple(_discrete_representatives(variable) for variable in variables)
    candidate_count = math.prod(len(values) for values in representatives)
    if candidate_count > int(maximum_candidates):
        raise ValueError(
            f"Declared lattice has {candidate_count} candidates, exceeding ceiling "
            f"{int(maximum_candidates)}"
        )

    best_evaluation: Evaluation | None = None
    best_candidate: np.ndarray | None = None
    for raw in itertools.product(*representatives):
        candidate = np.asarray(raw, dtype=np.float64)
        evaluation = problem.evaluate(candidate)
        if better(evaluation, best_evaluation):
            best_evaluation = evaluation
            best_candidate = candidate.copy()
    assert best_evaluation is not None and best_candidate is not None
    lattice_valid = _lattice_valid(problem, best_evaluation)
    feasible_exact = bool(best_evaluation.feasible and lattice_valid)
    claim = (
        "exact_best_feasible_point_on_declared_finite_lattice"
        if feasible_exact
        else "exhaustive_finite_lattice_screen_found_no_feasible_evaluator_point"
    )
    point = _reference_point(
        problem,
        best_candidate,
        candidate_space="original_all_discrete_finite_lattice",
        claim=claim,
        require_lattice=True,
        run_independent_validation=run_independent_validation,
    )
    return MathematicalReferenceReport(
        schema_version=MATHEMATICAL_REFERENCE_SCHEMA,
        report_kind="exhaustive_all_discrete_finite_lattice",
        source_problem_fingerprint=scientific_problem_fingerprint(problem),
        reference_problem_fingerprint=scientific_problem_fingerprint(problem),
        formulation_relation="identical original all-discrete ORPD formulation",
        solver_backend="CALO-RPD disclosed enumeration adapter",
        solver_backend_version=MATHEMATICAL_REFERENCE_SCHEMA,
        solver_algorithm="deterministic Cartesian finite-lattice enumeration",
        derivative_mode="none",
        deterministic_start=None,
        settings={
            "maximum_candidates": int(maximum_candidates),
            "declared_candidate_count": int(candidate_count),
            "selection": "common feasibility-first ordering",
        },
        termination_success=True,
        termination_status=0,
        termination_message=f"evaluated all {candidate_count} declared lattice points",
        accounting=SolverAccounting(
            backend_objective_evaluations=int(candidate_count),
            backend_derivative_evaluations=0,
            backend_iterations=int(candidate_count),
            common_evaluator_solver_calls=int(candidate_count),
            common_evaluator_cache_hits=0,
            common_evaluator_validation_calls=1,
            independent_validation_requests=len(point.independent_scenarios),
        ),
        reference_point=point,
        mixed_variable_point=point,
        certified_lower_bound=None,
        optimality_gap=None,
        gap_claim_permitted=False,
        exact_claim_scope=(
            "global best feasible objective on the complete declared finite lattice under the "
            "common deterministic evaluator"
            if feasible_exact
            else None
        ),
        warning=(
            "Exactness is restricted to the declared all-discrete lattice and common evaluator. "
            "It is not a lower bound for any continuous or differently controlled ORPD task."
        ),
    )
