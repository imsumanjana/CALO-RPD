"""Uncertainty, OOD, contextual-bandit, safety, and fallback authority for Change D."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch

from .transition_kernel import REGIME_OPERATOR_PRIORS
from .tsh_calo_policy import (
    HierarchicalPolicyAction,
    TSHCALOPolicyOutput,
)
from .tsh_calo_schema import (
    N_CONTROL_GROUPS,
    N_LEARNER_CONTEXTS,
    N_OPERATORS,
)


def _normalise_masked(values: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    positive = torch.where(allowed, torch.clamp(values, min=0.0), torch.zeros_like(values))
    total = positive.sum(dim=-1, keepdim=True)
    if bool((total <= 0.0).any()):
        raise ValueError("Safety shield received an action row with no positive allowed mass")
    return positive / total


@dataclass(frozen=True, slots=True)
class EnsembleDisagreement:
    regime: float
    groups: torch.Tensor
    contexts: torch.Tensor

    def validate(self) -> None:
        if self.groups.shape != (N_CONTROL_GROUPS,):
            raise ValueError("Ensemble group disagreement does not match the policy ABI")
        if self.contexts.shape != (N_LEARNER_CONTEXTS,):
            raise ValueError("Ensemble context disagreement does not match the policy ABI")
        values = torch.cat(
            [
                self.groups.reshape(-1),
                self.contexts.reshape(-1),
                self.groups.new_tensor([self.regime]),
            ]
        )
        if not bool(torch.isfinite(values).all()) or bool((values < 0.0).any()):
            raise ValueError("Ensemble disagreement must be finite and non-negative")


@dataclass(slots=True)
class EnsemblePolicyResult:
    mean_output: TSHCALOPolicyOutput
    disagreement: EnsembleDisagreement


def aggregate_policy_ensemble(outputs: list[TSHCALOPolicyOutput]) -> EnsemblePolicyResult:
    """Combine independently trained heads and retain probability-space disagreement."""

    if len(outputs) < 2:
        raise ValueError("TSH-CALO epistemic uncertainty requires at least two policy members")
    for output in outputs:
        output.validate()
    regime_probabilities = torch.stack(
        [torch.softmax(item.regime_logits, dim=-1) for item in outputs]
    )
    group_probabilities = torch.stack(
        [torch.softmax(item.group_operator_logits, dim=-1) for item in outputs]
    )
    context_probabilities = torch.stack(
        [torch.softmax(item.context_operator_logits, dim=-1) for item in outputs]
    )
    mean_regime = regime_probabilities.mean(dim=0)
    mean_groups = group_probabilities.mean(dim=0)
    mean_contexts = context_probabilities.mean(dim=0)
    eps = torch.finfo(mean_regime.dtype).tiny
    mean_output = TSHCALOPolicyOutput(
        regime_logits=torch.log(mean_regime.clamp_min(eps)),
        group_operator_logits=torch.log(mean_groups.clamp_min(eps)),
        context_operator_logits=torch.log(mean_contexts.clamp_min(eps)),
        group_alpha=torch.stack([item.group_alpha for item in outputs]).mean(dim=0),
        group_beta=torch.stack([item.group_beta for item in outputs]).mean(dim=0),
        value=torch.stack([item.value for item in outputs]).mean(dim=0),
    )
    disagreement = EnsembleDisagreement(
        regime=float(regime_probabilities.var(dim=0, unbiased=False).mean().item()),
        groups=group_probabilities.var(dim=0, unbiased=False).mean(dim=-1),
        contexts=context_probabilities.var(dim=0, unbiased=False).mean(dim=-1),
    )
    mean_output.validate()
    disagreement.validate()
    return EnsemblePolicyResult(mean_output, disagreement)


def topology_ood_signature(state) -> np.ndarray:
    """Return an order-invariant raw-state signature without learned or protected-test fitting."""

    state.validate()

    def moments(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape[0] == 0:
            return np.zeros(2 * values.shape[1], dtype=float)
        return np.concatenate([values.mean(axis=0), values.std(axis=0)])

    graph = state.topology
    signature = np.concatenate(
        [
            np.asarray(state.aggregate_features, dtype=float),
            moments(graph.node_features),
            moments(graph.edge_features),
            moments(graph.control_features),
            moments(graph.scenario_features),
            np.asarray(
                [
                    np.log1p(len(graph.node_features)),
                    np.log1p(len(graph.edge_features)),
                    np.log1p(len(graph.control_features)),
                    np.log1p(len(graph.scenario_features)),
                ],
                dtype=float,
            ),
        ]
    )
    if not np.all(np.isfinite(signature)):
        raise ValueError("TSH-CALO OOD signature must be finite")
    return signature


@dataclass(frozen=True, slots=True)
class OODCalibration:
    """Frozen development-only calibration; fitting is deliberately outside runtime."""

    mean: np.ndarray
    scale: np.ndarray
    attenuation_start: float = 2.0
    minimum_neural_weight: float = 0.0

    def validate(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        scale = np.asarray(self.scale, dtype=float)
        if mean.ndim != 1 or scale.shape != mean.shape or len(mean) == 0:
            raise ValueError("OOD calibration mean/scale must be aligned non-empty vectors")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("OOD calibration must be finite")
        if np.any(scale <= 0.0):
            raise ValueError("OOD calibration scales must be strictly positive")
        if not np.isfinite(self.attenuation_start) or self.attenuation_start <= 0.0:
            raise ValueError("OOD attenuation_start must be finite and positive")
        if not 0.0 <= self.minimum_neural_weight <= 1.0:
            raise ValueError("OOD minimum neural weight must be within [0, 1]")

    def score_and_attenuation(self, signature: np.ndarray) -> tuple[float, float]:
        self.validate()
        signature = np.asarray(signature, dtype=float)
        if signature.shape != np.asarray(self.mean).shape or not np.all(np.isfinite(signature)):
            raise ValueError("OOD signature is incompatible with the frozen calibration")
        score = float(
            np.sqrt(
                np.mean(np.square((signature - np.asarray(self.mean)) / np.asarray(self.scale)))
            )
        )
        excess = max(score - float(self.attenuation_start), 0.0)
        attenuation = max(float(self.minimum_neural_weight), 1.0 / (1.0 + excess))
        return score, float(np.clip(attenuation, 0.0, 1.0))


class SlidingWindowContextualBandit:
    """Fixed-memory deterministic reward residual with exact checkpoint state."""

    SCHEMA_VERSION = "tsh-calo-bandit-v1"

    def __init__(self, window_size: int = 32, exploration: float = 0.35) -> None:
        self.window_size = int(window_size)
        self.exploration = float(exploration)
        if self.window_size < 2 or not np.isfinite(self.exploration) or self.exploration < 0.0:
            raise ValueError("Bandit window must be >= 2 and exploration must be non-negative")
        shape = (N_CONTROL_GROUPS, N_LEARNER_CONTEXTS, N_OPERATORS)
        self.rewards = np.zeros(shape + (self.window_size,), dtype=np.float64)
        self.cursor = np.zeros(shape, dtype=np.int64)
        self.count = np.zeros(shape, dtype=np.int64)

    def update(self, group: int, context: int, operator: int, reward: float) -> None:
        group, context, operator = int(group), int(context), int(operator)
        if not (0 <= group < N_CONTROL_GROUPS):
            raise ValueError("Bandit update contains an unknown control group")
        if not (0 <= context < N_LEARNER_CONTEXTS):
            raise ValueError("Bandit update contains an unknown learner context")
        if not (0 <= operator < N_OPERATORS):
            raise ValueError("Bandit update contains an unknown operator")
        if not np.isfinite(reward):
            raise ValueError("Bandit rewards must be finite")
        position = int(self.cursor[group, context, operator])
        self.rewards[group, context, operator, position] = float(reward)
        self.cursor[group, context, operator] = (position + 1) % self.window_size
        self.count[group, context, operator] = min(
            int(self.count[group, context, operator]) + 1, self.window_size
        )

    def probabilities(
        self, group: int, context: int, allowed: np.ndarray | None = None
    ) -> np.ndarray:
        group, context = int(group), int(context)
        if not (0 <= group < N_CONTROL_GROUPS and 0 <= context < N_LEARNER_CONTEXTS):
            raise ValueError("Bandit query contains an unknown group/context")
        counts = self.count[group, context].astype(float)
        means = np.zeros(N_OPERATORS, dtype=float)
        for operator in range(N_OPERATORS):
            count = int(counts[operator])
            if count:
                means[operator] = float(np.mean(self.rewards[group, context, operator, :count]))
        total = float(counts.sum())
        bonus = self.exploration * np.sqrt(np.log1p(total + 1.0) / (counts + 1.0))
        logits = means + bonus
        allowed_mask = (
            np.ones(N_OPERATORS, dtype=bool) if allowed is None else np.asarray(allowed, dtype=bool)
        )
        if allowed_mask.shape != (N_OPERATORS,) or not np.any(allowed_mask):
            raise ValueError("Bandit action mask must permit at least one operator")
        logits = logits - np.max(logits[allowed_mask])
        values = np.where(allowed_mask, np.exp(logits), 0.0)
        return values / values.sum()

    def state_dict(self) -> dict:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "window_size": self.window_size,
            "exploration": self.exploration,
            "rewards": self.rewards.copy(),
            "cursor": self.cursor.copy(),
            "count": self.count.copy(),
        }

    @classmethod
    def from_state_dict(cls, payload: dict) -> "SlidingWindowContextualBandit":
        if str(payload.get("schema_version", "")) != cls.SCHEMA_VERSION:
            raise ValueError("Contextual-bandit checkpoint schema is incompatible")
        bandit = cls(int(payload["window_size"]), float(payload["exploration"]))
        for name, target in (
            ("rewards", bandit.rewards),
            ("cursor", bandit.cursor),
            ("count", bandit.count),
        ):
            value = np.asarray(payload[name], dtype=target.dtype)
            if value.shape != target.shape:
                raise ValueError(f"Contextual-bandit checkpoint {name} shape is incompatible")
            target[...] = value
        if np.any(bandit.count < 0) or np.any(bandit.count > bandit.window_size):
            raise ValueError("Contextual-bandit checkpoint counts are invalid")
        if np.any(bandit.cursor < 0) or np.any(bandit.cursor >= bandit.window_size):
            raise ValueError("Contextual-bandit checkpoint cursors are invalid")
        if not np.all(np.isfinite(bandit.rewards)):
            raise ValueError("Contextual-bandit checkpoint rewards are non-finite")
        return bandit


@dataclass(frozen=True, slots=True)
class SafetyEnvelope:
    remaining_evaluations: int
    candidate_count: int
    mixed_variable_lattice_valid: bool = True

    def validate(self) -> None:
        if self.remaining_evaluations < 0 or self.candidate_count < 0:
            raise ValueError("Safety envelope evaluation counts cannot be negative")
        if self.candidate_count > self.remaining_evaluations:
            raise ValueError(
                "TSH-CALO safety shield rejected candidate batch: requested evaluations exceed "
                "the remaining FE budget"
            )
        if not self.mixed_variable_lattice_valid:
            raise ValueError("TSH-CALO safety shield rejected an invalid mixed-variable lattice")


@dataclass(frozen=True, slots=True)
class ShieldConfig:
    neural_weight: float = 0.55
    rule_weight: float = 0.20
    bandit_weight: float = 0.20
    exploration_weight: float = 0.05
    disagreement_scale: float = 0.02

    def validate(self) -> None:
        weights = np.asarray(
            [self.neural_weight, self.rule_weight, self.bandit_weight, self.exploration_weight]
        )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("Shield mixture weights must be finite, non-negative, and non-zero")
        if self.exploration_weight <= 0.0:
            raise ValueError("Shield exploration floor must be strictly positive")
        if not np.isfinite(self.disagreement_scale) or self.disagreement_scale <= 0.0:
            raise ValueError("Shield disagreement scale must be finite and positive")


@dataclass(slots=True)
class ShieldTrace:
    schema_version: str
    uncertainty: torch.Tensor
    ood_score: float
    ood_attenuation: float
    mixture_weights: torch.Tensor
    action_mask: torch.Tensor
    intervention_reasons: tuple[str, ...]


@dataclass(slots=True)
class ShieldDecision:
    probabilities: torch.Tensor
    trace: ShieldTrace


class UncertaintySafetyShield:
    TRACE_SCHEMA = "tsh-calo-shield-trace-v1"

    def __init__(self, config: ShieldConfig | None = None) -> None:
        self.config = config or ShieldConfig()
        self.config.validate()

    def resolve(
        self,
        *,
        action: HierarchicalPolicyAction,
        disagreement: EnsembleDisagreement,
        ood_score: float,
        ood_attenuation: float,
        learner_groups: torch.Tensor,
        learner_contexts: torch.Tensor,
        bandit: SlidingWindowContextualBandit,
        safety: SafetyEnvelope,
    ) -> ShieldDecision:
        action.validate()
        disagreement.validate()
        safety.validate()
        if not np.isfinite(ood_score) or not 0.0 <= ood_attenuation <= 1.0:
            raise ValueError("OOD score/attenuation is invalid")
        groups = torch.as_tensor(
            learner_groups, dtype=torch.long, device=action.group_operator_probabilities.device
        )
        contexts = torch.as_tensor(learner_contexts, dtype=torch.long, device=groups.device)
        if groups.ndim != 1 or contexts.shape != groups.shape:
            raise ValueError("Shield learner groups and contexts must be aligned vectors")
        if len(groups) != safety.candidate_count:
            raise ValueError("Shield candidate_count must match learner assignments")
        if groups.numel() and (
            int(groups.min()) < 0
            or int(groups.max()) >= N_CONTROL_GROUPS
            or int(contexts.min()) < 0
            or int(contexts.max()) >= N_LEARNER_CONTEXTS
        ):
            raise ValueError("Shield learner group/context identity is invalid")
        mask = action.action_mask.to(groups.device)
        if groups.numel() and bool((~mask.available_groups[groups]).any()):
            raise ValueError("Shield cannot assign learners to an unavailable control group")

        group_probabilities = action.group_operator_probabilities[groups]
        neural_logits = (
            torch.log(group_probabilities.clamp_min(1e-12))
            + action.context_operator_logits[contexts]
        )
        allowed = mask.allowed[groups]
        neural = torch.softmax(neural_logits.masked_fill(~allowed, -torch.inf), dim=-1)
        frozen_rule_prior = np.pad(REGIME_OPERATOR_PRIORS[int(action.regime)], (0, N_OPERATORS - 6))
        rule_prior = torch.as_tensor(
            frozen_rule_prior,
            device=groups.device,
            dtype=neural.dtype,
        ).expand(len(groups), -1)
        rule = _normalise_masked(rule_prior, allowed)
        bandit_rows = [
            bandit.probabilities(int(group), int(context), allowed[index].detach().cpu().numpy())
            for index, (group, context) in enumerate(zip(groups.tolist(), contexts.tolist()))
        ]
        bandit_probability = torch.as_tensor(
            np.asarray(bandit_rows), device=groups.device, dtype=neural.dtype
        )
        exploration = _normalise_masked(torch.ones_like(neural), allowed)

        uncertainty = torch.maximum(
            disagreement.groups.to(groups.device)[groups],
            disagreement.contexts.to(groups.device)[contexts],
        ).to(neural.dtype)
        uncertainty_attenuation = 1.0 / (1.0 + uncertainty / float(self.config.disagreement_scale))
        neural_weight = (
            float(self.config.neural_weight) * uncertainty_attenuation * float(ood_attenuation)
        )
        weights = torch.stack(
            [
                neural_weight,
                torch.full_like(neural_weight, float(self.config.rule_weight)),
                torch.full_like(neural_weight, float(self.config.bandit_weight)),
                torch.full_like(neural_weight, float(self.config.exploration_weight)),
            ],
            dim=-1,
        )
        weights = weights / weights.sum(dim=-1, keepdim=True)
        probabilities = (
            weights[:, 0, None] * neural
            + weights[:, 1, None] * rule
            + weights[:, 2, None] * bandit_probability
            + weights[:, 3, None] * exploration
        )
        probabilities = _normalise_masked(probabilities, allowed)
        reasons: list[str] = []
        if bool((uncertainty > 0.0).any()):
            reasons.append("ensemble_disagreement_attenuated_neural_weight")
        if ood_attenuation < 1.0:
            reasons.append("ood_attenuated_neural_weight")
        if bool((~allowed).any()):
            reasons.append("invalid_operators_masked")
        trace = ShieldTrace(
            self.TRACE_SCHEMA,
            uncertainty.detach().clone(),
            float(ood_score),
            float(ood_attenuation),
            weights.detach().clone(),
            allowed.detach().clone(),
            tuple(reasons),
        )
        return ShieldDecision(probabilities, trace)

    @staticmethod
    def sample(
        decision: ShieldDecision,
        *,
        deterministic: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if deterministic:
            return torch.argmax(decision.probabilities, dim=-1)
        return torch.multinomial(
            decision.probabilities, 1, replacement=True, generator=generator
        ).squeeze(-1)


class FallbackDisposition(str, Enum):
    EXECUTE_POLICY = "execute_policy"
    EXPLICIT_BASELINE = "explicit_baseline_fallback"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class PolicyFallbackDecision:
    disposition: FallbackDisposition
    algorithm_identity: str
    reason: str


def resolve_policy_fallback(
    *,
    policy_usable: bool,
    rejection_reason: str,
    baseline_fallback_permitted: bool,
    tsh_algorithm_identity: str,
    frozen_baseline_identity: str,
) -> PolicyFallbackDecision:
    """Never reinterpret an unavailable/rejected policy or silently label fallback as TSH-CALO."""

    if policy_usable:
        return PolicyFallbackDecision(
            FallbackDisposition.EXECUTE_POLICY,
            str(tsh_algorithm_identity),
            "compatible qualified active immutable policy accepted",
        )
    reason = str(rejection_reason or "policy unavailable or rejected")
    if baseline_fallback_permitted:
        return PolicyFallbackDecision(
            FallbackDisposition.EXPLICIT_BASELINE,
            str(frozen_baseline_identity),
            reason,
        )
    return PolicyFallbackDecision(FallbackDisposition.BLOCK, "", reason)
