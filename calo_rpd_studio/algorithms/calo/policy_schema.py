"""Versioned CALO policy/runtime schemas and v5.9 policy-state construction.

The policy schema is deliberately independent from the optimizer implementation.  A checkpoint
must declare the state/action semantics it was trained against; legacy checkpoints remain readable
but are explicitly classified as legacy rather than silently treated as native v5.9 policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .cognitive_state import STATE_DIM as LEGACY_STATE_DIM

CALO_RUNTIME_ARCHITECTURE = "calo-v5.9"
LEGACY_STATE_SCHEMA = "calo-state-v2-24"
POLICY_STATE_SCHEMA = "calo-state-v5.9-32"
POLICY_ACTION_SCHEMA = "calo-action-v5.9-raw-global-4r-6o-6p"
TRAINING_ENVIRONMENT_VERSION = "calo-training-v5.9-exact-controller"
POLICY_STATE_DIM = 32


@dataclass(slots=True)
class PolicyRuntimeContext:
    """Compact v5.9 runtime features unavailable in the historical 24-D state."""

    hpem_occupancy: float = 0.0
    memory_consensus: float = 0.0
    memory_readiness: float = 0.0
    success_memory_density: float = 0.0
    learning_lane_fraction: float = 0.0
    precision_active: float = 0.0
    precision_radius: float = 0.0
    variable_group_concentration: float = 0.0

    def vector(self) -> np.ndarray:
        values = np.asarray(
            [
                self.hpem_occupancy,
                self.memory_consensus,
                self.memory_readiness,
                self.success_memory_density,
                self.learning_lane_fraction,
                self.precision_active,
                self.precision_radius,
                self.variable_group_concentration,
            ],
            dtype=np.float32,
        )
        return np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def build_policy_vector(
    state, context: PolicyRuntimeContext | None = None, *, input_dim: int = POLICY_STATE_DIM
) -> np.ndarray:
    """Build exactly the state vector declared by a checkpoint.

    ``input_dim == 24`` is retained solely for legacy checkpoint compatibility.  Native v5.9
    checkpoints consume the same historical cognitive vector plus eight bounded features that make
    HPEM, dual-lane readiness, precision, and variable-group learning observable to the policy.
    """

    base = state.vector() if hasattr(state, "vector") else np.asarray(state, dtype=np.float32)
    base = np.asarray(base, dtype=np.float32).reshape(-1)
    if input_dim == LEGACY_STATE_DIM:
        if base.size != LEGACY_STATE_DIM:
            raise ValueError(
                f"Legacy CALO policy requires {LEGACY_STATE_DIM} state features, received {base.size}"
            )
        return base
    if input_dim != POLICY_STATE_DIM:
        raise ValueError(
            f"Unsupported CALO policy input dimension {input_dim}; expected {LEGACY_STATE_DIM} or {POLICY_STATE_DIM}"
        )
    if base.size != LEGACY_STATE_DIM:
        raise ValueError(
            f"CALO v5.9 policy requires the {LEGACY_STATE_DIM}-feature cognitive base, received {base.size}"
        )
    extra = (context or PolicyRuntimeContext()).vector()
    return np.concatenate((base, extra), dtype=np.float32)


def infer_checkpoint_schema(payload: dict) -> dict[str, str | int | bool]:
    """Return conservative compatibility metadata for old and new checkpoints."""

    architecture = dict(payload.get("architecture", {}) or {})
    metadata = dict(payload.get("metadata", {}) or {})
    input_dim = int(
        architecture.get("input_dim", metadata.get("state_dimension", LEGACY_STATE_DIM))
    )
    state_schema = str(metadata.get("state_schema_version", "") or "")
    action_schema = str(metadata.get("action_schema_version", "") or "")
    runtime_arch = str(metadata.get("runtime_architecture_version", "") or "")
    training_env = str(metadata.get("training_environment_version", "") or "")
    if not state_schema:
        state_schema = POLICY_STATE_SCHEMA if input_dim == POLICY_STATE_DIM else LEGACY_STATE_SCHEMA
    if not action_schema:
        # The head dimensions are compatible, but legacy checkpoints did not declare semantics.
        action_schema = (
            POLICY_ACTION_SCHEMA if input_dim == POLICY_STATE_DIM else "calo-action-legacy-4r-6o-6p"
        )
    native = (
        input_dim == POLICY_STATE_DIM
        and state_schema == POLICY_STATE_SCHEMA
        and action_schema == POLICY_ACTION_SCHEMA
        and runtime_arch == CALO_RUNTIME_ARCHITECTURE
        and training_env == TRAINING_ENVIRONMENT_VERSION
    )
    return {
        "input_dim": input_dim,
        "state_schema_version": state_schema,
        "action_schema_version": action_schema,
        "runtime_architecture_version": runtime_arch or "legacy",
        "training_environment_version": training_env or "legacy",
        "native_v59": bool(native),
        # Compatibility-only inspector alias for legacy callers/tests. A true value means the
        # checkpoint is native to the CURRENT 32-D ABI, not that v4.1 execution semantics are enabled.
        "native_v41": bool(native),
    }


def variable_group_concentration(probabilities) -> float:
    """Map a variable-group probability vector to [0,1] concentration (1 = one group dominates)."""

    values = np.asarray(probabilities, dtype=float).reshape(-1)
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    total = float(values.sum())
    if total <= 0.0 or values.size <= 1:
        return 0.0
    p = values / total
    positive = p > 0.0
    entropy = -float(np.sum(p[positive] * np.log(p[positive])))
    max_entropy = math.log(values.size)
    return float(np.clip(1.0 - entropy / max(max_entropy, 1e-12), 0.0, 1.0))
