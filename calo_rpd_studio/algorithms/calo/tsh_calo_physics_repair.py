"""Optional, counted, proposal-only physics repair for approved TSH-CALO Change E."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Callable, Sequence

import numpy as np

from .transition_kernel import evaluate_candidates


class PhysicsRepairStatus(str, Enum):
    PROPOSED = "proposed"
    MASKED = "masked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PhysicsRepairContext:
    """Linearization retained by an already-counted converged evaluator call."""

    converged: bool
    available_from_counted_evaluation: bool
    source_evaluation_id: str
    ac_jacobian: object | None
    control_sensitivity: np.ndarray | None
    constraint_residual: np.ndarray | None
    condition_number: float | None


@dataclass(frozen=True, slots=True)
class PhysicsRepairConfig:
    enabled: bool = False
    trust_radius: float = 0.08
    maximum_condition_number: float = 1e10

    def validate(self) -> None:
        if not np.isfinite(self.trust_radius) or not 0.0 < self.trust_radius <= 1.0:
            raise ValueError("Physics-repair trust radius must be within (0, 1]")
        if not np.isfinite(self.maximum_condition_number) or self.maximum_condition_number <= 1.0:
            raise ValueError("Physics-repair condition limit must be finite and greater than one")


@dataclass(frozen=True, slots=True)
class PhysicsRepairProposal:
    status: PhysicsRepairStatus
    candidate: np.ndarray | None
    reason: str
    source_evaluation_id: str
    condition_number: float | None
    step_norm: float
    linear_algebra_seconds: float
    hidden_solver_calls: int = 0
    evaluator_calls: int = 0
    declares_feasibility: bool = False


@dataclass(frozen=True, slots=True)
class CountedPhysicsRepairEvaluation:
    evaluation: object
    requested_evaluations: int
    completed_evaluations: int
    source_evaluation_id: str


def _masked(context: PhysicsRepairContext | None, reason: str) -> PhysicsRepairProposal:
    return PhysicsRepairProposal(
        PhysicsRepairStatus.MASKED,
        None,
        reason,
        "" if context is None else str(context.source_evaluation_id),
        None if context is None else context.condition_number,
        0.0,
        0.0,
    )


def _dense_or_sparse_solve(matrix, rhs: np.ndarray) -> np.ndarray:
    try:
        from scipy.sparse import issparse
        from scipy.sparse.linalg import spsolve

        if issparse(matrix):
            return np.asarray(spsolve(matrix, rhs), dtype=float)
    except (ImportError, TypeError, ValueError, RuntimeError):
        pass
    return np.asarray(np.linalg.solve(np.asarray(matrix, dtype=float), rhs), dtype=float)


def _normalized_lattice(variable: object) -> np.ndarray:
    values = tuple(getattr(variable, "values", ()) or ())
    if not values:
        return np.asarray([], dtype=float)
    lower = float(variable.lower)
    upper = float(variable.upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError("Discrete repair variable has invalid declared bounds")
    return np.clip((np.asarray(values, dtype=float) - lower) / (upper - lower), 0.0, 1.0)


def _snap_declared_lattice(candidate: np.ndarray, variables: Sequence[object]) -> np.ndarray:
    snapped = np.asarray(candidate, dtype=float).copy()
    for index, variable in enumerate(variables):
        normalized = _normalized_lattice(variable)
        if not normalized.size:
            continue
        snapped[index] = float(normalized[np.argmin(np.abs(normalized - snapped[index]))])
    return snapped


def _trust_bounded_lattice_candidate(
    current: np.ndarray,
    proposed: np.ndarray,
    control_direction: np.ndarray,
    variables: Sequence[object],
    trust_radius: float,
) -> np.ndarray:
    """Admit whole lattice moves first, then spend the remaining norm on continuous controls."""

    snapped = _snap_declared_lattice(proposed, variables)
    candidate = np.asarray(current, dtype=float).copy()
    discrete: list[int] = []
    continuous: list[int] = []
    for index, variable in enumerate(variables):
        lattice = _normalized_lattice(variable)
        if not lattice.size:
            continuous.append(index)
            continue
        if abs(float(snapped[index]) - float(current[index])) > 1e-15:
            discrete.append(index)

    used_squared = 0.0
    ranked = sorted(
        discrete,
        key=lambda index: (
            -abs(float(control_direction[index]))
            / max(abs(float(snapped[index] - current[index])), 1e-15),
            index,
        ),
    )
    for index in ranked:
        delta = float(snapped[index] - current[index])
        cost = delta * delta
        if used_squared + cost <= trust_radius * trust_radius + 1e-15:
            candidate[index] = snapped[index]
            used_squared += cost

    remaining = float(np.sqrt(max(trust_radius * trust_radius - used_squared, 0.0)))
    if continuous and remaining > 0.0:
        desired = np.asarray(proposed[continuous] - current[continuous], dtype=float)
        norm = float(np.linalg.norm(desired))
        if norm > 0.0:
            candidate[continuous] = current[continuous] + desired * min(1.0, remaining / norm)
    return np.clip(candidate, 0.0, 1.0)


class PhysicsRepairOperator:
    """Map supplied converged linearization residuals to a bounded control proposal."""

    def __init__(self, config: PhysicsRepairConfig | None = None) -> None:
        self.config = config or PhysicsRepairConfig()
        self.config.validate()

    def propose(
        self,
        current: np.ndarray,
        context: PhysicsRepairContext | None,
        variables: Sequence[object],
    ) -> PhysicsRepairProposal:
        if not self.config.enabled:
            return _masked(context, "physics repair is disabled")
        if context is None or not context.available_from_counted_evaluation:
            return _masked(context, "counted converged Jacobian context is unavailable")
        if not context.converged:
            return _masked(context, "source power flow did not converge")
        if not str(context.source_evaluation_id):
            return _masked(context, "source evaluation identity is missing")
        if (
            context.ac_jacobian is None
            or context.control_sensitivity is None
            or context.constraint_residual is None
            or context.condition_number is None
        ):
            return _masked(
                context, "required Jacobian, sensitivity, residual, or condition is missing"
            )
        condition = float(context.condition_number)
        if not np.isfinite(condition) or condition > self.config.maximum_condition_number:
            return _masked(context, "converged Jacobian is unavailable or ill-conditioned")

        current = np.asarray(current, dtype=float)
        sensitivity = np.asarray(context.control_sensitivity, dtype=float)
        residual = np.asarray(context.constraint_residual, dtype=float)
        jacobian_shape = getattr(context.ac_jacobian, "shape", ())
        if current.ndim != 1 or len(variables) != len(current):
            return _masked(
                context, "repair decision vector and variable declarations are incompatible"
            )
        if (
            residual.ndim != 1
            or tuple(jacobian_shape) != (len(residual), len(residual))
            or sensitivity.shape != (len(residual), len(current))
        ):
            return _masked(context, "repair linearization shapes are incompatible")
        if (
            not np.all(np.isfinite(current))
            or not np.all(np.isfinite(sensitivity))
            or not np.all(np.isfinite(residual))
        ):
            return _masked(context, "repair linearization contains non-finite values")
        if np.any((current < 0.0) | (current > 1.0)):
            return _masked(context, "repair decision vector is outside normalized bounds")

        started = time.perf_counter()
        try:
            state_direction = _dense_or_sparse_solve(context.ac_jacobian, -residual)
            control_direction = np.linalg.lstsq(sensitivity, state_direction, rcond=None)[0]
            if not np.all(np.isfinite(control_direction)):
                raise FloatingPointError("non-finite control direction")
            norm = float(np.linalg.norm(control_direction))
            if norm <= 1e-15:
                return _masked(context, "physics repair produced a zero control direction")
            bounded_direction = control_direction * min(1.0, float(self.config.trust_radius) / norm)
            proposed = np.clip(current + bounded_direction, 0.0, 1.0)
            candidate = _trust_bounded_lattice_candidate(
                current,
                proposed,
                control_direction,
                variables,
                float(self.config.trust_radius),
            )
            step_norm = float(np.linalg.norm(candidate - current))
            if step_norm <= 1e-15:
                return _masked(
                    context,
                    "physics repair has no lattice-valid move inside the trust radius",
                )
            if step_norm > float(self.config.trust_radius) + 1e-12:
                raise FloatingPointError("lattice-valid repair escaped its trust radius")
            elapsed = time.perf_counter() - started
            return PhysicsRepairProposal(
                PhysicsRepairStatus.PROPOSED,
                candidate,
                "bounded Jacobian-informed proposal; trusted evaluation required",
                str(context.source_evaluation_id),
                condition,
                step_norm,
                float(elapsed),
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError, TypeError) as exc:
            elapsed = time.perf_counter() - started
            return PhysicsRepairProposal(
                PhysicsRepairStatus.FAILED,
                None,
                f"physics repair linear algebra failed: {type(exc).__name__}",
                str(context.source_evaluation_id),
                condition,
                0.0,
                float(elapsed),
            )


def physics_repair_context_from_counted_evaluation(
    evaluation_context,
) -> PhysicsRepairContext | None:
    """Adapt the selected ephemeral ORPD linearization without a solver or evaluator call."""

    try:
        selected = evaluation_context.primary_converged_scenario()
        linearization = selected.control_linearization
    except (AttributeError, ValueError):
        return None
    if linearization is None or linearization.condition_number is None:
        return None
    return PhysicsRepairContext(
        converged=bool(getattr(selected.power_flow, "converged", False)),
        available_from_counted_evaluation=True,
        source_evaluation_id=str(linearization.source_evaluation_id),
        ac_jacobian=linearization.jacobian,
        control_sensitivity=np.asarray(linearization.control_sensitivity, dtype=float),
        constraint_residual=np.asarray(linearization.constraint_residual, dtype=float),
        condition_number=float(linearization.condition_number),
    )


def physics_repair_context_is_usable(
    context: PhysicsRepairContext | None,
    *,
    maximum_condition_number: float,
) -> bool:
    """Conservatively gate operator exposure before the policy can select Change E."""

    if (
        context is None
        or not context.converged
        or not context.available_from_counted_evaluation
        or not context.source_evaluation_id
        or context.ac_jacobian is None
        or context.control_sensitivity is None
        or context.constraint_residual is None
        or context.condition_number is None
    ):
        return False
    sensitivity = np.asarray(context.control_sensitivity, dtype=float)
    residual = np.asarray(context.constraint_residual, dtype=float)
    return bool(
        np.isfinite(float(context.condition_number))
        and float(context.condition_number) <= float(maximum_condition_number)
        and sensitivity.ndim == 2
        and residual.ndim == 1
        and sensitivity.shape[0] == residual.size
        and np.all(np.isfinite(sensitivity))
        and np.all(np.isfinite(residual))
        and float(np.linalg.norm(sensitivity)) > 1e-15
        and float(np.linalg.norm(residual)) > 1e-15
    )


def evaluate_physics_repair_proposal(
    proposal: PhysicsRepairProposal,
    evaluator: Callable[[np.ndarray], Sequence[object]],
    *,
    remaining_evaluations: int,
) -> CountedPhysicsRepairEvaluation:
    """Require the same trusted evaluator and one explicit FE for a repair proposal."""

    if proposal.status is not PhysicsRepairStatus.PROPOSED or proposal.candidate is None:
        raise ValueError("Only a completed physics-repair proposal can be evaluated")
    if int(remaining_evaluations) < 1:
        raise ValueError("Physics-repair evaluation would exceed the remaining FE budget")
    batch = evaluate_candidates(np.asarray(proposal.candidate)[None, :], evaluator)
    if not batch.complete:
        raise RuntimeError(
            "Trusted evaluator did not complete the counted physics-repair evaluation"
        )
    return CountedPhysicsRepairEvaluation(
        batch.evaluations[0],
        batch.requested,
        batch.completed,
        proposal.source_evaluation_id,
    )
