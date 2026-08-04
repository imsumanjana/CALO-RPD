"""Versioned public contracts for the approved TSH-CALO candidate architecture."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


TSH_CALO_ALGORITHM_ID = "TSH-CALO"
TSH_CALO_ALGORITHM_VERSION = "tsh-calo-v1.1.0-counted-physics-candidate"
TSH_CALO_POLICY_ARCHITECTURE = "tsh-calo-policy-v1"
TSH_CALO_STATE_SCHEMA = "tsh-calo-state-v1-aggregate32-topology"
TSH_CALO_ACTION_SCHEMA = "tsh-calo-action-v1-hierarchical-4r-3g-4c-7o-6p"
TSH_CALO_TRAINING_ENVIRONMENT = "tsh-calo-training-v5-batched-device-context-safe80"


class ControlGroup(IntEnum):
    GENERATOR_VOLTAGE = 0
    TRANSFORMER_TAP = 1
    SHUNT = 2


N_CONTROL_GROUPS = len(ControlGroup)
N_SEARCH_REGIMES = 4
N_LEARNER_CONTEXTS = 4
N_OPERATORS = 7
N_BOUNDED_PARAMETERS = 6


@dataclass(frozen=True, slots=True)
class TSHCALOFeatureFlags:
    """Explicit component flags; optional E and experimental F are never inferred."""

    graph_context: bool = True
    hierarchical_actions: bool = True
    uncertainty_shield: bool = True
    contextual_bandit_residual: bool = True
    physics_repair: bool = False
    population_schedule: bool = False
    allow_experimental_components: bool = False

    def validate(self) -> None:
        if not self.graph_context or not self.hierarchical_actions:
            raise ValueError("TSH-CALO requires approved graph context and hierarchical actions")
        if not self.uncertainty_shield or not self.contextual_bandit_residual:
            raise ValueError("TSH-CALO requires the approved uncertainty and bandit shield")
        if self.population_schedule and not self.allow_experimental_components:
            raise ValueError(
                "TSH-CALO population scheduling is experimental and requires an explicit "
                "allow_experimental_components flag"
            )


DEFAULT_TSH_CALO_FEATURES = TSHCALOFeatureFlags()
DEFAULT_TSH_CALO_FEATURES.validate()
