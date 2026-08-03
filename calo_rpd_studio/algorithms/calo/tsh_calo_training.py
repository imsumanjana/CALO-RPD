"""Independent PPO update and exact-resume boundary for TSH-CALO policy training.

This module accepts already-built policy states and rollout rewards. It does not create, start,
modify, or inspect a power-system experiment, and it can only export an unqualified candidate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from calo_rpd_studio.ai.model_io import checkpoint_sha256, durable_torch_save, load_checkpoint
from calo_rpd_studio.power_system.case_identity import protected_holdout_matches

from .topology_context import TopologyAwarePolicyState
from .tsh_calo_policy import (
    GroupActionMask,
    TSHCALOPolicyNetwork,
    assign_group_conditioned_learner_operators,
    hierarchical_action,
    masked_group_operator_probabilities,
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


def _valid_sha256(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingConfig:
    training_run_id: str
    development_cases: tuple[str, ...]
    seed_manifest_sha256: str
    seed: int = 0
    hidden_dim: int = 64
    graph_steps: int = 2
    learning_rate: float = 3e-4
    ppo_epochs: int = 4
    clip_ratio: float = 0.20
    value_weight: float = 0.50
    entropy_weight: float = 0.01
    gradient_norm: float = 0.50
    device: str = "auto"
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
        if self.hidden_dim < 8 or self.graph_steps < 1:
            raise ValueError("TSH-CALO policy architecture is invalid")
        if self.learning_rate <= 0.0 or self.ppo_epochs < 1:
            raise ValueError("TSH-CALO PPO learning rate and epoch count must be positive")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("TSH-CALO PPO clip ratio must be within (0, 1)")
        if self.value_weight < 0.0 or self.entropy_weight < 0.0 or self.gradient_norm <= 0.0:
            raise ValueError("TSH-CALO PPO loss and gradient weights are invalid")
        if str(self.device).lower() not in {"auto", "cpu", "cuda"} and not str(
            self.device
        ).lower().startswith("cuda:"):
            raise ValueError("TSH-CALO training device must be auto, cpu, cuda, or cuda:<index>")
        self.feature_flags.validate()

    def scientific_design_hash(self) -> str:
        self.validate()
        payload = asdict(self)
        payload.pop("device", None)
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


def _resolve_device(requested: str) -> torch.device:
    choice = str(requested).strip().lower()
    if choice == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if choice.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA TSH-CALO training was requested but CUDA is unavailable")
        return torch.device(choice if ":" in choice else "cuda:0")
    return torch.device("cpu")


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in tuple(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


class IndependentTSHCALOTrainer:
    """PPO learner with no registry, activation, GUI, or experiment-workflow authority."""

    RESUME_FORMAT = "tsh_calo_independent_training_resume_v1"

    def __init__(self, config: TSHCALOTrainingConfig) -> None:
        config.validate()
        self.config = config
        self.device = _resolve_device(config.device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.seed))
            network = TSHCALOPolicyNetwork(config.hidden_dim, config.graph_steps)
        self.network = network.to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.numpy_rng = np.random.default_rng(config.seed)
        self.torch_generator = torch.Generator(device=self.device).manual_seed(int(config.seed))
        self.update_steps = 0

    def _log_probability_entropy_value(
        self, state: TopologyAwarePolicyState, action: TSHCALOTrainingAction
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action.validate()
        output = self.network(state)
        mask = action.action_mask.to(self.device)
        regime_distribution = torch.distributions.Categorical(logits=output.regime_logits)
        log_probability = regime_distribution.log_prob(
            torch.tensor(action.regime, dtype=torch.long, device=self.device)
        )
        entropy = regime_distribution.entropy()
        group_probabilities = masked_group_operator_probabilities(
            output.group_operator_logits, mask
        )
        available = mask.available_groups
        group_operators = torch.as_tensor(
            action.group_operators, dtype=torch.long, device=self.device
        )
        for group in torch.nonzero(available, as_tuple=False).flatten().tolist():
            distribution = torch.distributions.Categorical(probs=group_probabilities[group])
            log_probability = log_probability + distribution.log_prob(group_operators[group])
            entropy = entropy + distribution.entropy()
        beta_distribution = torch.distributions.Beta(output.group_alpha, output.group_beta)
        parameters = torch.as_tensor(
            action.group_parameters, dtype=output.group_alpha.dtype, device=self.device
        ).clamp(1e-6, 1.0 - 1e-6)
        if bool(available.any()):
            log_probability = (
                log_probability + beta_distribution.log_prob(parameters)[available].sum()
            )
            entropy = entropy + beta_distribution.entropy()[available].sum()
        learner_groups = torch.as_tensor(
            action.learner_groups, dtype=torch.long, device=self.device
        )
        learner_contexts = torch.as_tensor(
            action.learner_contexts, dtype=torch.long, device=self.device
        )
        learner_operators = torch.as_tensor(
            action.learner_operators, dtype=torch.long, device=self.device
        )
        if learner_groups.numel():
            combined_logits = (
                torch.log(group_probabilities[learner_groups].clamp_min(1e-12))
                + output.context_operator_logits[learner_contexts]
            )
            allowed = mask.allowed[learner_groups]
            learner_distribution = torch.distributions.Categorical(
                logits=combined_logits.masked_fill(~allowed, -torch.inf)
            )
            log_probability = (
                log_probability + learner_distribution.log_prob(learner_operators).sum()
            )
            entropy = entropy + learner_distribution.entropy().sum()
        return log_probability, entropy, output.value

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
        )
        action = TSHCALOTrainingAction(
            regime=hierarchical.regime,
            group_operators=hierarchical.group_operators.detach().cpu().numpy(),
            group_parameters=hierarchical.group_parameters.detach().cpu().numpy(),
            learner_groups=np.asarray(learner_groups, dtype=int),
            learner_contexts=np.asarray(learner_contexts, dtype=int),
            learner_operators=operators.detach().cpu().numpy(),
            action_mask=GroupActionMask(
                hierarchical.action_mask.allowed.detach().cpu(),
                hierarchical.action_mask.available_groups.detach().cpu(),
            ),
        )
        log_probability, _entropy, value = self._log_probability_entropy_value(state, action)
        return action, float(log_probability.item()), float(value.item())

    def update(self, batch: TSHCALORolloutBatch) -> dict[str, float]:
        batch.validate()
        old_log_probabilities = torch.as_tensor(
            batch.old_log_probabilities, dtype=torch.float32, device=self.device
        )
        returns = torch.as_tensor(batch.returns, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch.advantages, dtype=torch.float32, device=self.device)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / advantages.std(
                unbiased=False
            ).clamp_min(1e-8)
        metrics: dict[str, float] = {}
        self.network.train()
        for _ in range(self.config.ppo_epochs):
            evaluated = [
                self._log_probability_entropy_value(state, action)
                for state, action in zip(batch.states, batch.actions)
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
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("TSH-CALO PPO update produced a non-finite loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.gradient_norm
            )
            self.optimizer.step()
            metrics = {
                "loss": float(loss.detach().cpu()),
                "policy_loss": float(policy_loss.detach().cpu()),
                "value_loss": float(value_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            }
        self.update_steps += 1
        self.network.eval()
        return metrics

    def save_resume(self, path: str | Path) -> str:
        target = Path(path).expanduser().resolve()
        payload = {
            "format": self.RESUME_FORMAT,
            "algorithm_id": TSH_CALO_ALGORITHM_ID,
            "algorithm_version": TSH_CALO_ALGORITHM_VERSION,
            "state_schema_version": TSH_CALO_STATE_SCHEMA,
            "action_schema_version": TSH_CALO_ACTION_SCHEMA,
            "training_environment_version": TSH_CALO_TRAINING_ENVIRONMENT,
            "scientific_design_hash": self.config.scientific_design_hash(),
            "training_config": asdict(self.config),
            "model_state_dict": {
                name: tensor.detach().cpu() for name, tensor in self.network.state_dict().items()
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "numpy_generator_state": self.numpy_rng.bit_generator.state,
            "torch_generator_state": self.torch_generator.get_state().cpu(),
            "update_steps": self.update_steps,
        }
        durable_torch_save(payload, target)
        return checkpoint_sha256(target)

    @classmethod
    def load_resume(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        expected_config: TSHCALOTrainingConfig,
    ) -> "IndependentTSHCALOTrainer":
        payload = load_checkpoint(path, expected_sha256=expected_sha256, map_location="cpu")
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
        trainer = cls(expected_config)
        trainer.network.load_state_dict(payload["model_state_dict"], strict=True)
        trainer.optimizer.load_state_dict(payload["optimizer_state_dict"])
        _optimizer_to_device(trainer.optimizer, trainer.device)
        trainer.numpy_rng.bit_generator.state = payload["numpy_generator_state"]
        trainer.torch_generator.set_state(payload["torch_generator_state"].cpu())
        trainer.update_steps = int(payload.get("update_steps", 0))
        trainer.network.eval()
        return trainer

    def export_unqualified_candidate(
        self,
        path: str | Path,
        *,
        source_commit: str,
    ) -> TSHCALOCandidateArtifact:
        if self.update_steps < 1:
            raise ValueError("TSH-CALO candidate export requires at least one completed PPO update")
        provenance = IndependentTrainingProvenance(
            training_run_id=self.config.training_run_id,
            training_design_sha256=self.config.scientific_design_hash(),
            source_commit=str(source_commit),
            development_cases=tuple(self.config.development_cases),
            seed_manifest_sha256=self.config.seed_manifest_sha256,
        )
        return save_tsh_calo_candidate(
            path,
            self.network,
            provenance,
            feature_flags=self.config.feature_flags,
        )
