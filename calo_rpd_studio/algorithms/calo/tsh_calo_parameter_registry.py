"""Scientist-facing registry of TSH-CALO parameters used by parameter studies.

The registry is descriptive.  It does not change optimizer or training values.  Every entry states
where a parameter acts and what must be repeated when it changes so study tooling can keep unlike
scientific questions separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ParameterKind = Literal["integer", "continuous", "binary"]
ParameterDomain = Literal[
    "training_dynamics",
    "policy_architecture",
    "search_environment",
    "resource_design",
    "reward_definition",
    "adaptive_search_action",
]
StudyScale = Literal["linear", "log", "discrete"]


@dataclass(frozen=True, slots=True)
class TSHCALOParameterDefinition:
    key: str
    label: str
    domain: ParameterDomain
    kind: ParameterKind
    scale: StudyScale
    requires_retraining: bool
    changes_policy_contract: bool = False
    changes_learning_objective: bool = False
    scientist_tunable: bool = True
    local_response_allowed: bool = False
    description: str = ""


_DEFINITIONS = (
    TSHCALOParameterDefinition(
        "population_size", "Population size", "search_environment", "integer", "discrete", True,
        description="Number of learners evaluated in each fixed-population generation.",
    ),
    TSHCALOParameterDefinition(
        "max_evaluations", "Training evaluation budget", "resource_design", "integer", "discrete", True,
        description="Exact counted function-evaluation budget used by each training episode.",
    ),
    TSHCALOParameterDefinition(
        "ensemble_members", "Ensemble members", "policy_architecture", "integer", "discrete", True,
        changes_policy_contract=True,
        description="Number of independently trained members retained in the policy ensemble.",
    ),
    TSHCALOParameterDefinition(
        "deterministic_policy", "Deterministic training policy", "training_dynamics", "binary", "discrete", True,
    ),
    TSHCALOParameterDefinition(
        "environment_deterministic", "Deterministic training environment", "search_environment", "binary", "discrete", True,
    ),
    TSHCALOParameterDefinition(
        "training.hidden_dim", "Hidden width", "policy_architecture", "integer", "discrete", True,
        changes_policy_contract=True,
    ),
    TSHCALOParameterDefinition(
        "training.graph_steps", "Graph propagation steps", "policy_architecture", "integer", "discrete", True,
        changes_policy_contract=True,
    ),
    TSHCALOParameterDefinition(
        "training.learning_rate", "Learning rate", "training_dynamics", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "training.ppo_epochs", "Policy update epochs", "training_dynamics", "integer", "discrete", True,
    ),
    TSHCALOParameterDefinition(
        "training.clip_ratio", "Policy clipping ratio", "training_dynamics", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "training.value_weight", "Value-loss weight", "training_dynamics", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "training.entropy_weight", "Entropy weight", "training_dynamics", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "training.gradient_norm", "Gradient norm limit", "training_dynamics", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "training.discount_factor", "Discount factor", "training_dynamics", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "training.gae_lambda", "Advantage smoothing", "training_dynamics", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "environment.feasible_archive_capacity", "Feasible archive capacity", "search_environment", "integer", "discrete", True,
    ),
    TSHCALOParameterDefinition(
        "environment.boundary_archive_capacity", "Boundary archive capacity", "search_environment", "integer", "discrete", True,
    ),
    TSHCALOParameterDefinition(
        "environment.memory_capacity", "Success-memory capacity", "search_environment", "integer", "discrete", True,
    ),
    TSHCALOParameterDefinition(
        "environment.memory_decay", "Success-memory decay", "search_environment", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "environment.credit_decay", "Operator-credit decay", "search_environment", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "environment.credit_floor", "Operator-credit floor", "search_environment", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "environment.group_credit_decay", "Group-credit decay", "search_environment", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "environment.max_learning_lane_fraction", "Maximum learning-lane fraction", "search_environment", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "environment.precision_start_radius", "Initial precision radius", "search_environment", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "environment.precision_min_radius", "Minimum precision radius", "search_environment", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "environment.precision_max_radius", "Maximum precision radius", "search_environment", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "environment.epsilon_quantile", "Feasibility quantile", "search_environment", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "environment.epsilon_control_fraction", "Feasibility-control fraction", "search_environment", "continuous", "linear", True,
    ),
    TSHCALOParameterDefinition(
        "environment.epsilon_exponent", "Feasibility-control exponent", "search_environment", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "environment.stagnation_window", "Stagnation window", "search_environment", "integer", "discrete", True,
    ),
    TSHCALOParameterDefinition(
        "environment.memory_evidence_batches", "Memory evidence window", "search_environment", "integer", "discrete", True,
    ),
    TSHCALOParameterDefinition(
        "environment.recovery_diversity_threshold", "Recovery diversity threshold", "search_environment", "continuous", "log", True,
    ),
    TSHCALOParameterDefinition(
        "environment.recovery_fraction", "Recovery fraction ceiling", "search_environment", "continuous", "linear", True,
        description="Upper bound on the fraction of learners eligible for policy-directed recovery.",
    ),
    TSHCALOParameterDefinition(
        "policy.attraction", "Policy attraction", "adaptive_search_action", "continuous", "linear", False,
        scientist_tunable=False, local_response_allowed=True,
        description="Effective attraction selected by the policy for a control group.",
    ),
    TSHCALOParameterDefinition(
        "policy.differential", "Policy differential influence", "adaptive_search_action", "continuous", "linear", False,
        scientist_tunable=False, local_response_allowed=True,
        description="Effective differential influence selected by the policy for a control group.",
    ),
    TSHCALOParameterDefinition(
        "policy.exploration_sigma", "Policy exploration scale", "adaptive_search_action", "continuous", "linear", False,
        scientist_tunable=False, local_response_allowed=True,
        description="Effective exploration scale selected by the policy for a control group.",
    ),
    TSHCALOParameterDefinition(
        "policy.memory_weight", "Policy memory weight", "adaptive_search_action", "continuous", "linear", False,
        scientist_tunable=False, local_response_allowed=True,
        description="Effective success-memory weight selected by the policy for a control group.",
    ),
    TSHCALOParameterDefinition(
        "policy.diversity_weight", "Policy diversity weight", "adaptive_search_action", "continuous", "linear", False,
        scientist_tunable=False, local_response_allowed=True,
        description="Effective diversity weight selected by the policy for a control group.",
    ),
    TSHCALOParameterDefinition(
        "policy.recovery_fraction", "Policy recovery fraction", "adaptive_search_action", "continuous", "linear", False,
        scientist_tunable=False, local_response_allowed=True,
        description="Effective recovery fraction proposed by the policy before the configured ceiling is applied.",
    ),
    TSHCALOParameterDefinition(
        "reward.objective_weight", "Objective-improvement reward weight", "reward_definition", "continuous", "linear", True,
        changes_learning_objective=True, scientist_tunable=False,
    ),
    TSHCALOParameterDefinition(
        "reward.constraint_weight", "Constraint-improvement reward weight", "reward_definition", "continuous", "linear", True,
        changes_learning_objective=True, scientist_tunable=False,
    ),
    TSHCALOParameterDefinition(
        "reward.feasibility_weight", "Feasibility reward weight", "reward_definition", "continuous", "linear", True,
        changes_learning_objective=True, scientist_tunable=False,
    ),
    TSHCALOParameterDefinition(
        "reward.diversity_weight", "Diversity-recovery reward weight", "reward_definition", "continuous", "linear", True,
        changes_learning_objective=True, scientist_tunable=False,
    ),
    TSHCALOParameterDefinition(
        "reward.overhead_weight", "Overhead penalty weight", "reward_definition", "continuous", "linear", True,
        changes_learning_objective=True, scientist_tunable=False,
    ),
)


TSH_CALO_PARAMETER_REGISTRY = {item.key: item for item in _DEFINITIONS}


def parameter_definition(key: str) -> TSHCALOParameterDefinition:
    try:
        return TSH_CALO_PARAMETER_REGISTRY[str(key)]
    except KeyError as exc:
        raise KeyError(f"Unknown TSH-CALO study parameter: {key}") from exc


def registered_parameter_keys(*, scientist_tunable_only: bool = False) -> tuple[str, ...]:
    items = _DEFINITIONS
    if scientist_tunable_only:
        items = tuple(item for item in items if item.scientist_tunable)
    return tuple(item.key for item in items)
