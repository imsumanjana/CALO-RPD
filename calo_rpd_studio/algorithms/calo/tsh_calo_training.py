"""Independent PPO update and exact-resume boundary for TSH-CALO policy training.

This module accepts already-built policy states and rollout rewards. It does not create, start,
modify, or inspect a power-system experiment, and it can only export an unqualified candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from calo_rpd_studio.ai.model_io import checkpoint_sha256, durable_torch_save, load_checkpoint
from calo_rpd_studio.power_system.case_identity import protected_holdout_matches

from .topology_context import TopologyAwarePolicyState
from .tsh_calo_policy import (
    GroupActionMask,
    PreparedTopologyAwarePolicyState,
    TSHCALOPolicyNetwork,
    assign_group_conditioned_learner_operators,
    hierarchical_action,
)
from .tsh_calo_policy_artifact import (
    IndependentTrainingProvenance,
    TSHCALOCandidateArtifact,
    save_tsh_calo_candidate,
)
from .tsh_calo_schema import (
    N_BOUNDED_PARAMETERS,
    N_CONTROL_GROUPS,
    TSH_CALO_ACTION_SCHEMA,
    TSH_CALO_ALGORITHM_ID,
    TSH_CALO_ALGORITHM_VERSION,
    TSH_CALO_STATE_SCHEMA,
    TSH_CALO_TRAINING_ENVIRONMENT,
    TSHCALOFeatureFlags,
)
from .tsh_calo_training_resources import (
    TSHCALOTrainingDeviceGuard,
    TSHCALOTrainingResourceEnvelope,
    estimate_tsh_calo_training_working_set,
    validate_tsh_calo_training_device_provenance,
)
from .tsh_calo_training_receipt import load_tsh_calo_training_episode_receipt
from .transition_kernel import TransitionResult


_LOG = logging.getLogger(__name__)


def _valid_sha256(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _build_and_admit_training_network(config: "TSHCALOTrainingConfig"):
    """Build the policy shape and apply the same Safe-80 admission used by training."""

    config.validate()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config.seed))
        network = TSHCALOPolicyNetwork(config.hidden_dim, config.graph_steps)
    estimate = estimate_tsh_calo_training_working_set(network, config.resource_envelope)
    guard = TSHCALOTrainingDeviceGuard.admit(
        estimate,
        requested_device=config.device,
        allow_cpu_fallback=config.allow_cpu_fallback,
    )
    return network, estimate, guard


def preflight_tsh_calo_training_resources(config: "TSHCALOTrainingConfig") -> dict:
    """Check current training admission without starting a campaign or retaining a device lease."""

    _network, estimate, guard = _build_and_admit_training_network(config)
    try:
        return {
            "memory_estimate": estimate.to_dict(),
            "memory_admission": guard.admission.to_dict(),
        }
    finally:
        guard.close()


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingConfig:
    training_run_id: str
    development_cases: tuple[str, ...]
    seed_manifest_sha256: str
    resource_envelope: TSHCALOTrainingResourceEnvelope
    development_freeze_commit: str = ""
    development_freeze_sha256: str = ""
    phase4_acceptance_sha256: str = ""
    seed: int = 0
    hidden_dim: int = 64
    graph_steps: int = 2
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    clip_ratio: float = 0.20
    value_weight: float = 0.50
    entropy_weight: float = 0.01
    gradient_norm: float = 0.50
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    device: str = "auto"
    allow_cpu_fallback: bool = True
    generalization_guard_sha256: str = ""
    feature_flags: TSHCALOFeatureFlags = field(default_factory=TSHCALOFeatureFlags)

    def validate(self) -> None:
        if not self.training_run_id.strip():
            raise ValueError("TSH-CALO training requires an independent training_run_id")
        if not self.development_cases:
            raise ValueError("TSH-CALO training requires development cases")
        leaked = protected_holdout_matches(self.development_cases)
        if leaked:
            raise ValueError(
                "Protected holdout cases cannot enter TSH-CALO training: " + ", ".join(leaked)
            )
        if not _valid_sha256(self.seed_manifest_sha256):
            raise ValueError("TSH-CALO seed manifest SHA-256 is invalid")
        if (
            self.development_freeze_commit
            or self.development_freeze_sha256
            or self.phase4_acceptance_sha256
        ):
            normalized_commit = str(self.development_freeze_commit).strip().lower()
            if (
                len(normalized_commit) != 40
                or any(character not in "0123456789abcdef" for character in normalized_commit)
                or not _valid_sha256(self.development_freeze_sha256)
                or not _valid_sha256(self.phase4_acceptance_sha256)
            ):
                raise ValueError(
                    "TSH-CALO post-development training requires an exact freeze commit, freeze "
                    "payload SHA-256, and Phase 4 acceptance SHA-256"
                )
        if self.hidden_dim < 8 or self.graph_steps < 1:
            raise ValueError("TSH-CALO policy architecture is invalid")
        if self.learning_rate <= 0.0 or self.ppo_epochs < 1:
            raise ValueError("TSH-CALO PPO learning rate and epoch count must be positive")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("TSH-CALO PPO clip ratio must be within (0, 1)")
        if self.value_weight < 0.0 or self.entropy_weight < 0.0 or self.gradient_norm <= 0.0:
            raise ValueError("TSH-CALO PPO loss and gradient weights are invalid")
        if not 0.0 <= self.discount_factor <= 1.0 or not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("TSH-CALO discount and GAE factors must be within [0, 1]")
        if str(self.device).lower() not in {"auto", "cpu", "cuda"} and not str(
            self.device
        ).lower().startswith("cuda:"):
            raise ValueError("TSH-CALO training device must be auto, cpu, cuda, or cuda:<index>")
        if self.generalization_guard_sha256 and not _valid_sha256(
            self.generalization_guard_sha256
        ):
            raise ValueError("TSH-CALO generalization-guard configuration SHA-256 is invalid")
        self.resource_envelope.validate()
        self.feature_flags.validate()

    def scientific_design_hash(self) -> str:
        self.validate()
        payload = asdict(self)
        payload.pop("device", None)
        payload.pop("allow_cpu_fallback", None)
        # Preserve pre-guard exact-resume hashes when no guard was declared. A configured guard is
        # scientific training authority and therefore participates in the design identity.
        if not self.generalization_guard_sha256:
            payload.pop("generalization_guard_sha256", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingAction:
    regime: int
    group_operators: np.ndarray
    group_parameters: np.ndarray
    learner_groups: np.ndarray
    learner_contexts: np.ndarray
    learner_operators: np.ndarray
    action_mask: GroupActionMask

    def validate(self) -> None:
        self.action_mask.validate()
        groups = np.asarray(self.group_operators, dtype=int)
        parameters = np.asarray(self.group_parameters, dtype=float)
        learner_groups = np.asarray(self.learner_groups, dtype=int)
        contexts = np.asarray(self.learner_contexts, dtype=int)
        operators = np.asarray(self.learner_operators, dtype=int)
        if not 0 <= int(self.regime) < 4:
            raise ValueError("TSH-CALO training action regime is invalid")
        if groups.shape != (N_CONTROL_GROUPS,):
            raise ValueError("TSH-CALO training group action shape is invalid")
        if parameters.shape != (N_CONTROL_GROUPS, N_BOUNDED_PARAMETERS):
            raise ValueError("TSH-CALO training parameter action shape is invalid")
        if contexts.shape != learner_groups.shape or operators.shape != learner_groups.shape:
            raise ValueError("TSH-CALO learner action vectors must align")
        if not np.all(np.isfinite(parameters)) or np.any((parameters < 0.0) | (parameters > 1.0)):
            raise ValueError("TSH-CALO training parameters must be finite and bounded")
        available = np.asarray(self.action_mask.available_groups, dtype=bool)
        allowed = np.asarray(self.action_mask.allowed, dtype=bool)
        if np.any(groups[~available] != -1):
            raise ValueError("Unavailable TSH-CALO groups require the -1 action sentinel")
        for group in np.flatnonzero(available):
            operator = int(groups[group])
            if operator < 0 or operator >= allowed.shape[1] or not allowed[group, operator]:
                raise ValueError("TSH-CALO group action violates its declared mask")
        if learner_groups.size:
            if np.any((learner_groups < 0) | (learner_groups >= N_CONTROL_GROUPS)):
                raise ValueError("TSH-CALO learner group is invalid")
            if np.any((contexts < 0) | (contexts >= 4)):
                raise ValueError("TSH-CALO learner context is invalid")
            if np.any((operators < 0) | (operators >= allowed.shape[1])):
                raise ValueError("TSH-CALO learner operator is invalid")
            if np.any(~available[learner_groups]) or np.any(~allowed[learner_groups, operators]):
                raise ValueError("TSH-CALO learner action violates its declared mask")


@dataclass(frozen=True, slots=True)
class _PreparedTSHCALOTrainingAction:
    """Validated action tensors retained on the learner device across PPO epochs."""

    regime: torch.Tensor
    group_operators: torch.Tensor
    group_parameters: torch.Tensor
    learner_groups: torch.Tensor
    learner_contexts: torch.Tensor
    learner_operators: torch.Tensor
    allowed: torch.Tensor
    available_groups: torch.Tensor


@dataclass(frozen=True, slots=True)
class TSHCALORolloutBatch:
    states: tuple[TopologyAwarePolicyState, ...]
    actions: tuple[TSHCALOTrainingAction, ...]
    old_log_probabilities: np.ndarray
    old_values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray

    def validate(self) -> None:
        count = len(self.states)
        if count < 1 or len(self.actions) != count:
            raise ValueError("TSH-CALO rollout states and actions must be non-empty and aligned")
        for state in self.states:
            state.validate()
        for action in self.actions:
            action.validate()
        for name, values in (
            ("old_log_probabilities", self.old_log_probabilities),
            ("old_values", self.old_values),
            ("advantages", self.advantages),
            ("returns", self.returns),
        ):
            vector = np.asarray(values, dtype=float)
            if vector.shape != (count,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"TSH-CALO rollout {name} must be a finite aligned vector")


@dataclass(frozen=True, slots=True)
class PendingTSHCALORolloutStep:
    state: TopologyAwarePolicyState
    action: TSHCALOTrainingAction
    log_probability: float
    value: float


class IndependentTSHCALORolloutCollector:
    """Collect PPO data only from the versioned canonical transition reward authority."""

    SCHEMA_VERSION = "tsh-calo-independent-rollout-v1"

    def __init__(self, trainer: "IndependentTSHCALOTrainer") -> None:
        self.trainer = trainer
        self._pending: PendingTSHCALORolloutStep | None = None
        self.states: list[TopologyAwarePolicyState] = []
        self.actions: list[TSHCALOTrainingAction] = []
        self.log_probabilities: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.terminals: list[bool] = []

    def sample(
        self,
        state: TopologyAwarePolicyState,
        action_mask: GroupActionMask,
        learner_groups,
        learner_contexts,
        *,
        deterministic: bool = False,
    ) -> PendingTSHCALORolloutStep:
        if self._pending is not None:
            raise RuntimeError("TSH-CALO rollout has an uncommitted sampled action")
        if len(self.states) >= self.trainer.config.resource_envelope.rollout_capacity:
            raise RuntimeError("TSH-CALO rollout reached its frozen resource-envelope capacity")
        action, log_probability, value = self.trainer.sample_action(
            state,
            action_mask,
            learner_groups,
            learner_contexts,
            deterministic=deterministic,
        )
        pending = PendingTSHCALORolloutStep(
            state,
            action,
            float(log_probability),
            float(value),
        )
        self._pending = pending
        return pending

    def commit(self, transition: TransitionResult, *, terminal: bool = False) -> None:
        if self._pending is None:
            raise RuntimeError("TSH-CALO rollout cannot commit without a sampled action")
        if not isinstance(transition, TransitionResult):
            raise TypeError("TSH-CALO rollout rewards must come from the canonical transition")
        reward = float(transition.reward.total)
        if not np.isfinite(reward):
            raise ValueError("TSH-CALO canonical transition reward must be finite")
        pending = self._pending
        self.states.append(pending.state)
        self.actions.append(pending.action)
        self.log_probabilities.append(pending.log_probability)
        self.values.append(pending.value)
        self.rewards.append(reward)
        self.terminals.append(bool(terminal))
        self._pending = None

    def discard_pending(self) -> None:
        """Explicitly discard an unevaluated action after a failed/cancelled environment step."""

        if self._pending is None:
            raise RuntimeError("TSH-CALO rollout has no pending action to discard")
        self._pending = None

    def build_batch(self, *, bootstrap_value: float = 0.0) -> TSHCALORolloutBatch:
        if self._pending is not None:
            raise RuntimeError("TSH-CALO rollout cannot finalize an uncommitted action")
        if not self.states:
            raise ValueError("TSH-CALO rollout requires at least one canonical transition")
        bootstrap = float(bootstrap_value)
        if not np.isfinite(bootstrap):
            raise ValueError("TSH-CALO rollout bootstrap value must be finite")
        rewards = np.asarray(self.rewards, dtype=float)
        values = np.asarray(self.values, dtype=float)
        terminals = np.asarray(self.terminals, dtype=bool)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        next_value = bootstrap
        gamma = float(self.trainer.config.discount_factor)
        gae_lambda = float(self.trainer.config.gae_lambda)
        for index in range(len(rewards) - 1, -1, -1):
            continuation = 0.0 if terminals[index] else 1.0
            delta = rewards[index] + gamma * next_value * continuation - values[index]
            gae = delta + gamma * gae_lambda * continuation * gae
            advantages[index] = gae
            next_value = values[index]
        batch = TSHCALORolloutBatch(
            tuple(self.states),
            tuple(self.actions),
            np.asarray(self.log_probabilities, dtype=float),
            values,
            advantages,
            advantages + values,
        )
        batch.validate()
        return batch

    def state_dict(self) -> dict:
        if self._pending is not None:
            raise RuntimeError("TSH-CALO rollout cannot checkpoint an uncommitted action")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "scientific_design_hash": self.trainer.config.scientific_design_hash(),
            "states": tuple(self.states),
            "actions": tuple(self.actions),
            "log_probabilities": np.asarray(self.log_probabilities, dtype=float),
            "values": np.asarray(self.values, dtype=float),
            "rewards": np.asarray(self.rewards, dtype=float),
            "terminals": np.asarray(self.terminals, dtype=bool),
        }

    @classmethod
    def from_state_dict(
        cls,
        trainer: "IndependentTSHCALOTrainer",
        payload: dict,
    ) -> "IndependentTSHCALORolloutCollector":
        if str(payload.get("schema_version", "")) != cls.SCHEMA_VERSION:
            raise ValueError("TSH-CALO rollout checkpoint schema is incompatible")
        if (
            str(payload.get("scientific_design_hash", ""))
            != trainer.config.scientific_design_hash()
        ):
            raise ValueError("TSH-CALO rollout checkpoint scientific design changed")
        collector = cls(trainer)
        collector.states = list(payload.get("states", ()))
        collector.actions = list(payload.get("actions", ()))
        collector.log_probabilities = np.asarray(
            payload.get("log_probabilities", []), dtype=float
        ).tolist()
        collector.values = np.asarray(payload.get("values", []), dtype=float).tolist()
        collector.rewards = np.asarray(payload.get("rewards", []), dtype=float).tolist()
        collector.terminals = np.asarray(payload.get("terminals", []), dtype=bool).tolist()
        count = len(collector.states)
        if not (
            len(collector.actions)
            == len(collector.log_probabilities)
            == len(collector.values)
            == len(collector.rewards)
            == len(collector.terminals)
            == count
        ):
            raise ValueError("TSH-CALO rollout checkpoint arrays are not aligned")
        if count:
            collector.build_batch(bootstrap_value=0.0)
        return collector


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


class IndependentTSHCALOTrainer:
    """PPO learner with no registry, activation, GUI, or experiment-workflow authority."""

    RESUME_FORMAT = "tsh_calo_independent_training_resume_v3"

    def __init__(self, config: TSHCALOTrainingConfig) -> None:
        self._closed = False
        self.config = config
        network, self.memory_estimate, self.device_guard = _build_and_admit_training_network(config)
        self.device = torch.device(self.device_guard.admission.selected_device)
        try:
            self.network = network.to(self.device)
        except torch.cuda.OutOfMemoryError:
            if self.device.type != "cuda" or not config.allow_cpu_fallback:
                self.device_guard.close()
                raise
            torch.cuda.empty_cache()
            self.device_guard = self.device_guard.fallback_after_cuda_oom(self.memory_estimate)
            self.device = torch.device("cpu")
            self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.numpy_rng = np.random.default_rng(config.seed)
        self.torch_generator = torch.Generator(device=self.device).manual_seed(int(config.seed))
        self.update_steps = 0
        self.training_episode_receipts: list[dict] = []

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.device_guard.close()

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("TSH-CALO trainer is closed and no longer owns its admitted device")

    def __enter__(self) -> "IndependentTSHCALOTrainer":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            _LOG.debug("Unable to release TSH-CALO training resources", exc_info=True)

    def device_provenance(self) -> dict:
        return {
            "memory_estimate": self.memory_estimate.to_dict(),
            "memory_admission": self.device_guard.admission.to_dict(),
            "computation_semantics": (
                "NVIDIA GPU computes; VRAM is admitted storage"
                if self.device.type == "cuda"
                else "CPU computes; system RAM is admitted storage"
            ),
            "ppo_host_synchronization": (
                "one packed metrics transfer after the complete configured PPO epoch block"
                if self.device.type == "cuda"
                else "not_applicable_cpu_execution"
            ),
            "per_ppo_epoch_cpu_metric_transfer": False,
        }

    def estimate_value(self, state: TopologyAwarePolicyState, *, population_size: int) -> float:
        self._assert_open()
        self.config.resource_envelope.validate_state(state, population_size=int(population_size))
        self.network.eval()
        with torch.no_grad():
            value = float(self.network(state).value.detach().cpu())
        if not np.isfinite(value):
            raise RuntimeError("TSH-CALO bootstrap value is non-finite")
        return value

    def record_training_episode_receipt(self, payload: dict) -> None:
        self._assert_open()
        receipt = load_tsh_calo_training_episode_receipt(payload)
        if receipt.training_run_id != self.config.training_run_id:
            raise ValueError("TSH-CALO episode receipt belongs to another training run")
        if receipt.training_design_sha256 != self.config.scientific_design_hash():
            raise ValueError("TSH-CALO episode receipt training design changed")
        if receipt.case_identity not in self.config.development_cases:
            raise ValueError("TSH-CALO episode receipt case is undeclared")
        if receipt.ppo_update_count < 1 or receipt.ppo_update_count > self.update_steps:
            raise ValueError("TSH-CALO episode receipt PPO update count is inconsistent")
        if any(
            item.get("receipt_sha256") == receipt.receipt_sha256
            for item in self.training_episode_receipts
        ):
            raise ValueError("TSH-CALO episode receipt was already recorded")
        if any(
            item.get("session_id") == receipt.session_id for item in self.training_episode_receipts
        ):
            raise ValueError("TSH-CALO training session ID was already recorded")
        self.training_episode_receipts.append(receipt.to_dict())

    def _log_probability_entropy_value(
        self,
        state: TopologyAwarePolicyState | PreparedTopologyAwarePolicyState,
        action: TSHCALOTrainingAction | _PreparedTSHCALOTrainingAction,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(state, PreparedTopologyAwarePolicyState):
            output = self.network.forward_prepared(state)
        else:
            output = self.network(state)
        if isinstance(action, _PreparedTSHCALOTrainingAction):
            prepared_action = action
        else:
            prepared_action = self._prepare_action(action)
        allowed = prepared_action.allowed
        available = prepared_action.available_groups
        regime_distribution = torch.distributions.Categorical(logits=output.regime_logits)
        log_probability = regime_distribution.log_prob(prepared_action.regime)
        entropy = regime_distribution.entropy()
        masked_group_logits = output.group_operator_logits.masked_fill(~allowed, -torch.inf)
        safe_group_logits = torch.where(
            available[:, None], masked_group_logits, torch.zeros_like(masked_group_logits)
        )
        group_probabilities = torch.where(
            available[:, None],
            torch.softmax(safe_group_logits, dim=-1),
            torch.zeros_like(masked_group_logits),
        )
        group_operators = prepared_action.group_operators
        safe_group_operators = group_operators.clamp_min(0)
        fallback = torch.zeros_like(group_probabilities)
        fallback[:, 0] = 1.0
        safe_probabilities = torch.where(available[:, None], group_probabilities, fallback)
        # Preserve the original fixed group addition order while removing Python reads of CUDA
        # availability flags. This keeps floating-point/replay semantics and avoids host syncs.
        for group in range(N_CONTROL_GROUPS):
            distribution = torch.distributions.Categorical(probs=safe_probabilities[group])
            active = available[group].to(dtype=output.regime_logits.dtype)
            log_probability = log_probability + active * distribution.log_prob(
                safe_group_operators[group]
            )
            entropy = entropy + active * distribution.entropy()
        beta_distribution = torch.distributions.Beta(output.group_alpha, output.group_beta)
        parameters = prepared_action.group_parameters.to(dtype=output.group_alpha.dtype).clamp(
            1e-6, 1.0 - 1e-6
        )
        log_probability = log_probability + beta_distribution.log_prob(parameters)[available].sum()
        entropy = entropy + beta_distribution.entropy()[available].sum()
        learner_groups = prepared_action.learner_groups
        learner_contexts = prepared_action.learner_contexts
        learner_operators = prepared_action.learner_operators
        if learner_groups.numel():
            combined_logits = (
                torch.log(group_probabilities[learner_groups].clamp_min(1e-12))
                + output.context_operator_logits[learner_contexts]
            )
            learner_allowed = allowed[learner_groups]
            learner_distribution = torch.distributions.Categorical(
                logits=combined_logits.masked_fill(~learner_allowed, -torch.inf)
            )
            log_probability = (
                log_probability + learner_distribution.log_prob(learner_operators).sum()
            )
            entropy = entropy + learner_distribution.entropy().sum()
        return log_probability, entropy, output.value

    def _prepare_action(self, action: TSHCALOTrainingAction) -> _PreparedTSHCALOTrainingAction:
        action.validate()
        mask = action.action_mask.to(self.device)
        return _PreparedTSHCALOTrainingAction(
            regime=torch.as_tensor(action.regime, dtype=torch.long, device=self.device),
            group_operators=torch.as_tensor(
                action.group_operators, dtype=torch.long, device=self.device
            ),
            group_parameters=torch.as_tensor(
                action.group_parameters, dtype=torch.float32, device=self.device
            ),
            learner_groups=torch.as_tensor(
                action.learner_groups, dtype=torch.long, device=self.device
            ),
            learner_contexts=torch.as_tensor(
                action.learner_contexts, dtype=torch.long, device=self.device
            ),
            learner_operators=torch.as_tensor(
                action.learner_operators, dtype=torch.long, device=self.device
            ),
            allowed=mask.allowed,
            available_groups=mask.available_groups,
        )

    @torch.no_grad()
    def sample_action(
        self,
        state: TopologyAwarePolicyState,
        action_mask: GroupActionMask,
        learner_groups,
        learner_contexts,
        *,
        deterministic: bool = False,
    ) -> tuple[TSHCALOTrainingAction, float, float]:
        self._assert_open()
        population_size = len(np.asarray(learner_groups).reshape(-1))
        self.config.resource_envelope.validate_state(state, population_size=population_size)
        self.network.eval()
        output = self.network(state)
        hierarchical = hierarchical_action(
            output,
            action_mask,
            deterministic=deterministic,
            generator=None if deterministic else self.torch_generator,
        )
        operators = assign_group_conditioned_learner_operators(
            hierarchical,
            torch.as_tensor(learner_groups, dtype=torch.long, device=self.device),
            torch.as_tensor(learner_contexts, dtype=torch.long, device=self.device),
            deterministic=deterministic,
            generator=None if deterministic else self.torch_generator,
            action_is_validated=True,
        )
        prepared_action = _PreparedTSHCALOTrainingAction(
            regime=torch.as_tensor(hierarchical.regime, dtype=torch.long, device=self.device),
            group_operators=hierarchical.group_operators,
            group_parameters=hierarchical.group_parameters,
            learner_groups=torch.as_tensor(learner_groups, dtype=torch.long, device=self.device),
            learner_contexts=torch.as_tensor(
                learner_contexts, dtype=torch.long, device=self.device
            ),
            learner_operators=operators,
            allowed=hierarchical.action_mask.allowed,
            available_groups=hierarchical.action_mask.available_groups,
        )
        log_probability, _entropy, value = self._log_probability_entropy_value(
            state, prepared_action
        )
        packed = (
            torch.cat(
                (
                    hierarchical.group_operators.reshape(-1).to(torch.float64),
                    hierarchical.group_parameters.reshape(-1).to(torch.float64),
                    operators.reshape(-1).to(torch.float64),
                    hierarchical.action_mask.allowed.reshape(-1).to(torch.float64),
                    hierarchical.action_mask.available_groups.reshape(-1).to(torch.float64),
                    log_probability.reshape(1).to(torch.float64),
                    value.reshape(1).to(torch.float64),
                )
            )
            .detach()
            .to("cpu")
            .numpy()
        )
        cursor = 0
        group_operator_count = N_CONTROL_GROUPS
        group_operators = packed[cursor : cursor + group_operator_count].astype(np.int64)
        cursor += group_operator_count
        group_parameter_count = N_CONTROL_GROUPS * N_BOUNDED_PARAMETERS
        group_parameters = packed[cursor : cursor + group_parameter_count].reshape(
            N_CONTROL_GROUPS, N_BOUNDED_PARAMETERS
        )
        cursor += group_parameter_count
        learner_operator_count = int(operators.numel())
        learner_operators = packed[cursor : cursor + learner_operator_count].astype(np.int64)
        cursor += learner_operator_count
        allowed_count = N_CONTROL_GROUPS * int(hierarchical.action_mask.allowed.shape[1])
        allowed = (
            packed[cursor : cursor + allowed_count]
            .reshape(N_CONTROL_GROUPS, int(hierarchical.action_mask.allowed.shape[1]))
            .astype(bool)
        )
        cursor += allowed_count
        available = packed[cursor : cursor + N_CONTROL_GROUPS].astype(bool)
        cursor += N_CONTROL_GROUPS
        log_probability_value = float(packed[cursor])
        value_scalar = float(packed[cursor + 1])
        action = TSHCALOTrainingAction(
            regime=hierarchical.regime,
            group_operators=group_operators,
            group_parameters=group_parameters,
            learner_groups=np.asarray(learner_groups, dtype=int),
            learner_contexts=np.asarray(learner_contexts, dtype=int),
            learner_operators=learner_operators,
            action_mask=GroupActionMask(
                torch.from_numpy(allowed),
                torch.from_numpy(available),
            ),
        )
        action.validate()
        return action, log_probability_value, value_scalar

    def update(self, batch: TSHCALORolloutBatch) -> dict[str, float]:
        self._assert_open()
        batch.validate()
        if len(batch.states) > self.config.resource_envelope.rollout_capacity:
            raise MemoryError("TSH-CALO PPO batch exceeds its frozen rollout capacity")
        for state, action in zip(batch.states, batch.actions):
            self.config.resource_envelope.validate_state(
                state, population_size=len(action.learner_groups)
            )
        prepared_states = tuple(self.network.prepare_state(state) for state in batch.states)
        prepared_actions = tuple(self._prepare_action(action) for action in batch.actions)
        old_log_probabilities = torch.as_tensor(
            batch.old_log_probabilities, dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(
                unbiased=False
            ).clamp_min(1e-8)
        final_metric_tensors: tuple[torch.Tensor, ...] | None = None
        self.network.train()
        for _ in range(self.config.ppo_epochs):
            evaluated = [
                self._log_probability_entropy_value(state, action)
                for state, action in zip(prepared_states, prepared_actions)
            ]
            log_probabilities = torch.stack([item[0] for item in evaluated])
            entropies = torch.stack([item[1] for item in evaluated])
            values = torch.stack([item[2] for item in evaluated])
            ratio = torch.exp(log_probabilities - old_log_probabilities)
            unclipped = ratio * advantages
            clipped = (
                torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                * advantages
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns)
            entropy = entropies.mean()
            loss = (
                policy_loss
                + self.config.value_weight * value_loss
                - self.config.entropy_weight * entropy
            )
            finite_loss = torch.isfinite(loss)
            if self.device.type == "cuda":
                # Enqueue the validation on the device. It raises when the CUDA stream reaches the
                # assertion, without forcing a Python scalar read after every PPO epoch.
                torch._assert_async(finite_loss, "TSH-CALO PPO update produced a non-finite loss")
            elif not bool(finite_loss):
                raise RuntimeError("TSH-CALO PPO update produced a non-finite loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.gradient_norm
            )
            self.optimizer.step()
            final_metric_tensors = (
                loss.detach(),
                policy_loss.detach(),
                value_loss.detach(),
                entropy.detach(),
                torch.as_tensor(gradient_norm).detach(),
            )
        if final_metric_tensors is None:
            raise RuntimeError("TSH-CALO PPO update completed without an epoch")
        metric_values = torch.stack(final_metric_tensors).to(device="cpu", dtype=torch.float64)
        metrics = dict(
            zip(
                ("loss", "policy_loss", "value_loss", "entropy", "gradient_norm"),
                (float(value) for value in metric_values.tolist()),
                strict=True,
            )
        )
        self.update_steps += 1
        self.network.eval()
        return metrics

    def resume_state_dict(self) -> dict:
        self._assert_open()
        training_config = asdict(self.config)
        if not self.config.generalization_guard_sha256:
            training_config.pop("generalization_guard_sha256", None)
        return {
            "format": self.RESUME_FORMAT,
            "algorithm_id": TSH_CALO_ALGORITHM_ID,
            "algorithm_version": TSH_CALO_ALGORITHM_VERSION,
            "state_schema_version": TSH_CALO_STATE_SCHEMA,
            "action_schema_version": TSH_CALO_ACTION_SCHEMA,
            "training_environment_version": TSH_CALO_TRAINING_ENVIRONMENT,
            "scientific_design_hash": self.config.scientific_design_hash(),
            "training_config": training_config,
            "training_device_provenance": self.device_provenance(),
            "model_state_dict": {
                name: tensor.detach().cpu() for name, tensor in self.network.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "numpy_generator_state": self.numpy_rng.bit_generator.state,
            "torch_generator_state": self.torch_generator.get_state().cpu(),
            "update_steps": self.update_steps,
            "training_episode_receipts": tuple(self.training_episode_receipts),
        }

    def save_resume(self, path: str | Path) -> str:
        target = Path(path).expanduser().resolve()
        payload = self.resume_state_dict()
        durable_torch_save(payload, target)
        return str(checkpoint_sha256(target))

    @classmethod
    def from_resume_state_dict(
        cls,
        payload: dict,
        *,
        expected_config: TSHCALOTrainingConfig,
    ) -> "IndependentTSHCALOTrainer":
        if str(payload.get("format", "")) != cls.RESUME_FORMAT:
            raise ValueError("TSH-CALO training resume format is incompatible")
        expected_config.validate()
        if (
            str(payload.get("scientific_design_hash", ""))
            != expected_config.scientific_design_hash()
        ):
            raise ValueError("TSH-CALO exact resume scientific design changed")
        for key, expected in (
            ("algorithm_id", TSH_CALO_ALGORITHM_ID),
            ("algorithm_version", TSH_CALO_ALGORITHM_VERSION),
            ("state_schema_version", TSH_CALO_STATE_SCHEMA),
            ("action_schema_version", TSH_CALO_ACTION_SCHEMA),
            ("training_environment_version", TSH_CALO_TRAINING_ENVIRONMENT),
        ):
            if str(payload.get(key, "")) != expected:
                raise ValueError(f"TSH-CALO training resume {key} is incompatible")
        validate_tsh_calo_training_device_provenance(
            dict(payload.get("training_device_provenance", {}) or {})
        )
        trainer = cls(expected_config)
        saved_device = str(
            dict(payload.get("training_device_provenance", {}))
            .get("memory_admission", {})
            .get("selected_device", "")
        )
        if saved_device != trainer.device_guard.admission.selected_device:
            trainer.close()
            raise RuntimeError(
                "TSH-CALO exact resume requires the same admitted computation device"
            )
        saved_estimate = dict(payload.get("training_device_provenance", {})).get(
            "memory_estimate", {}
        )
        if saved_estimate != trainer.memory_estimate.to_dict():
            trainer.close()
            raise ValueError("TSH-CALO exact resume memory estimate changed")
        trainer.network.load_state_dict(payload["model_state_dict"], strict=True)
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(trainer.optimizer, trainer.device)
        trainer.numpy_rng.bit_generator.state = payload["numpy_generator_state"]
        trainer.torch_generator.set_state(payload["torch_generator_state"].cpu())
        trainer.update_steps = int(payload.get("update_steps", 0))
        receipts = list(payload.get("training_episode_receipts", ()))
        trainer.training_episode_receipts = []
        for receipt in receipts:
            trainer.record_training_episode_receipt(receipt)
        trainer.network.eval()
        return trainer

    @classmethod
    def load_resume(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_config: TSHCALOTrainingConfig,
    ) -> "IndependentTSHCALOTrainer":
        payload = load_checkpoint(path, expected_sha256=expected_sha256, map_location="cpu")
        return cls.from_resume_state_dict(payload, expected_config=expected_config)

    def export_unqualified_candidate(
        self,
        path: str | Path,
        *,
        source_commit: str,
        execution_source_commit: str | None = None,
        generalization_guard: dict | None = None,
    ) -> TSHCALOCandidateArtifact:
        self._assert_open()
        normalized_source = str(source_commit).strip().lower()
        legacy_authority = (
            self.config.development_freeze_commit,
            self.config.development_freeze_sha256,
            self.config.phase4_acceptance_sha256,
        )
        if any(legacy_authority):
            if normalized_source != str(self.config.development_freeze_commit).strip().lower():
                raise ValueError(
                    "TSH-CALO candidate source does not match the retained development freeze"
                )
            if not _valid_sha256(self.config.development_freeze_sha256):
                raise ValueError("TSH-CALO candidate lacks a development-freeze payload SHA-256")
            if not _valid_sha256(self.config.phase4_acceptance_sha256):
                raise ValueError("TSH-CALO candidate lacks a Phase 4 acceptance receipt SHA-256")
        if self.update_steps < 1:
            raise ValueError("TSH-CALO candidate export requires at least one completed PPO update")
        if not self.training_episode_receipts:
            raise ValueError(
                "TSH-CALO candidate export requires a completed counted training episode receipt"
            )
        guard_payload = dict(generalization_guard or {})
        if self.config.generalization_guard_sha256:
            if not guard_payload:
                raise ValueError(
                    "TSH-CALO candidate export requires the configured generalization-guard evidence"
                )
            from .tsh_calo_generalization_guard import validate_generalization_guard_provenance

            validate_generalization_guard_provenance(
                guard_payload,
                training_episode_receipts=tuple(self.training_episode_receipts),
                expected_training_design_sha256=self.config.scientific_design_hash(),
            )
            if (
                guard_payload.get("guard_design_sha256")
                != self.config.generalization_guard_sha256
            ):
                raise ValueError(
                    "TSH-CALO candidate generalization evidence uses another guard configuration"
                )
            if guard_payload.get("promotion_allowed") is not True:
                raise ValueError(
                    "TSH-CALO candidate export is blocked by the generalization guard"
                )
        elif guard_payload:
            raise ValueError(
                "TSH-CALO candidate cannot attach undeclared generalization-guard evidence"
            )
        provenance = IndependentTrainingProvenance(
            training_run_id=self.config.training_run_id,
            training_design_sha256=self.config.scientific_design_hash(),
            source_commit=str(source_commit),
            execution_source_commit=str(execution_source_commit or source_commit),
            development_freeze_commit=str(self.config.development_freeze_commit),
            development_freeze_sha256=str(self.config.development_freeze_sha256),
            phase4_acceptance_sha256=str(self.config.phase4_acceptance_sha256),
            initialization_policy_sha256="",
            generalization_guard_sha256=self.config.generalization_guard_sha256,
            development_cases=tuple(self.config.development_cases),
            seed_manifest_sha256=self.config.seed_manifest_sha256,
            training_device_provenance=self.device_provenance(),
            training_episode_receipts=tuple(self.training_episode_receipts),
            generalization_guard=guard_payload or None,
        )
        return save_tsh_calo_candidate(
            path,
            self.network,
            provenance,
            feature_flags=self.config.feature_flags,
        )
