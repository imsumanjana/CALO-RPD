"""Build TSH-CALO topology context from already-counted ORPD scenario solves only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from calo_rpd_studio.power_system.case_model import BR_STATUS, GEN_STATUS, PD, QD

from .topology_context import (
    ScenarioDescriptor,
    TopologyAwarePolicyState,
    build_topology_graph_state,
)


@dataclass(frozen=True, slots=True)
class RuntimeTopologyPolicyContext:
    policy_state: TopologyAwarePolicyState
    reference_scenario: str
    scenario_names: tuple[str, ...]
    descriptor_source: str = "measured_from_already_counted_scenario_cases"


def _scenario_role(name: str) -> float:
    label = str(name).strip().lower()
    if label == "base":
        return 0.0
    if label.startswith("load"):
        return 1.0 / 3.0
    if label.startswith("renewable"):
        return 2.0 / 3.0
    if "out" in label or "conting" in label:
        return 1.0
    return 0.5


def _measured_ratio(current: float, baseline: float) -> float:
    """Return a finite relative magnitude while treating two structural zeros as unchanged."""

    scale = abs(float(baseline))
    observed = abs(float(current))
    if scale <= 1e-12:
        return 1.0 if observed <= 1e-12 else 1.0 + observed
    return observed / scale


def measured_scenario_descriptors(problem, evaluation_context) -> tuple[ScenarioDescriptor, ...]:
    """Encode observed stress without inferring any unmeasured hidden scenario parameter."""

    base = problem.case
    base_p = float(np.sum(np.maximum(base.bus[:, PD], 0.0)))
    base_q = float(np.sum(np.abs(base.bus[:, QD])))
    base_online = int(np.count_nonzero(base.branch[:, BR_STATUS] > 0.0)) + int(
        np.count_nonzero(base.gen[:, GEN_STATUS] > 0.0)
    )
    descriptors: list[ScenarioDescriptor] = []
    for item in evaluation_context.scenarios:
        solved = item.power_flow.case
        p_ratio = _measured_ratio(np.sum(np.maximum(solved.bus[:, PD], 0.0)), base_p)
        q_ratio = _measured_ratio(np.sum(np.abs(solved.bus[:, QD])), base_q)
        load_stress = 0.5 * (p_ratio + q_ratio)
        renewable_stress = (
            max(0.0, 1.0 - p_ratio) if str(item.name).lower().startswith("renewable") else 0.0
        )
        online = int(np.count_nonzero(solved.branch[:, BR_STATUS] > 0.0)) + int(
            np.count_nonzero(solved.gen[:, GEN_STATUS] > 0.0)
        )
        contingency_stress = max(0.0, float(base_online - online) / max(base_online, 1))
        descriptors.append(
            ScenarioDescriptor(
                load_stress=load_stress,
                renewable_stress=renewable_stress,
                contingency_stress=contingency_stress,
                aggregation_role=_scenario_role(item.name),
                weight=float(item.weight),
                is_ood=0.0,
                observed=1.0,
            )
        )
    if not descriptors:
        raise ValueError("TSH-CALO topology context requires counted scenario records")
    return tuple(descriptors)


def build_runtime_topology_policy_context(
    aggregate_features,
    problem,
    evaluation_context,
) -> RuntimeTopologyPolicyContext:
    reference = evaluation_context.primary_converged_scenario()
    descriptors = measured_scenario_descriptors(problem, evaluation_context)
    topology = build_topology_graph_state(
        problem.case,
        problem.decoder,
        evaluation_context.normalized_controls,
        reference.power_flow,
        descriptors,
    )
    state = TopologyAwarePolicyState(np.asarray(aggregate_features, dtype=float), topology)
    state.validate()
    return RuntimeTopologyPolicyContext(
        state,
        str(reference.name),
        tuple(str(item.name) for item in evaluation_context.scenarios),
    )
