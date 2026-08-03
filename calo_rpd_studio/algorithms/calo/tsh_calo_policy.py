"""Topology-aware hierarchical policy and action contract for TSH-CALO Change C."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .policy_schema import POLICY_STATE_DIM
from .topology_context import TopologyAwarePolicyState, TopologyContextEncoder
from .tsh_calo_schema import (
    N_BOUNDED_PARAMETERS,
    N_CONTROL_GROUPS,
    N_LEARNER_CONTEXTS,
    N_OPERATORS,
    N_SEARCH_REGIMES,
    TSH_CALO_ACTION_SCHEMA,
)


@dataclass(frozen=True, slots=True)
class GroupActionMask:
    """Declared operator availability for the three physical control groups."""

    allowed: torch.Tensor
    available_groups: torch.Tensor

    def validate(self) -> None:
        allowed = torch.as_tensor(self.allowed, dtype=torch.bool)
        available = torch.as_tensor(self.available_groups, dtype=torch.bool)
        if allowed.shape != (N_CONTROL_GROUPS, N_OPERATORS):
            raise ValueError(
                f"TSH-CALO operator mask must have shape ({N_CONTROL_GROUPS}, {N_OPERATORS})"
            )
        if available.shape != (N_CONTROL_GROUPS,):
            raise ValueError(f"TSH-CALO group availability must have shape ({N_CONTROL_GROUPS},)")
        invalid_available = available & (~allowed.any(dim=1))
        if bool(invalid_available.any()):
            raise ValueError("Every available control group must permit at least one operator")
        if bool(((~available) & allowed.any(dim=1)).any()):
            raise ValueError("Unavailable control groups cannot expose executable operators")

    def to(self, device: torch.device) -> "GroupActionMask":
        return GroupActionMask(
            torch.as_tensor(self.allowed, dtype=torch.bool, device=device),
            torch.as_tensor(self.available_groups, dtype=torch.bool, device=device),
        )

    @classmethod
    def from_control_groups(
        cls,
        control_groups,
        *,
        mixed_variable_enabled: bool = True,
        diversity_recovery_enabled: bool = True,
        compatibility: torch.Tensor | None = None,
    ) -> "GroupActionMask":
        groups = torch.as_tensor(control_groups, dtype=torch.long)
        available = torch.zeros(N_CONTROL_GROUPS, dtype=torch.bool)
        if groups.numel():
            if int(groups.min()) < 0 or int(groups.max()) >= N_CONTROL_GROUPS:
                raise ValueError("control_groups contains an unknown group")
            available[groups.unique()] = True
        allowed = available[:, None].expand(-1, N_OPERATORS).clone()
        if not mixed_variable_enabled:
            allowed[:, 4] = False
        if not diversity_recovery_enabled:
            allowed[:, 5] = False
        if compatibility is not None:
            compatibility = torch.as_tensor(compatibility, dtype=torch.bool)
            if compatibility.shape != allowed.shape:
                raise ValueError("operator-group compatibility shape does not match the action ABI")
            allowed &= compatibility
        mask = cls(allowed, available)
        mask.validate()
        return mask


@dataclass(slots=True)
class TSHCALOPolicyOutput:
    regime_logits: torch.Tensor
    group_operator_logits: torch.Tensor
    context_operator_logits: torch.Tensor
    group_alpha: torch.Tensor
    group_beta: torch.Tensor
    value: torch.Tensor

    def validate(self) -> None:
        if self.regime_logits.shape != (N_SEARCH_REGIMES,):
            raise ValueError("TSH-CALO regime logits do not match the action ABI")
        expected_operators = (N_CONTROL_GROUPS, N_OPERATORS)
        expected_parameters = (N_CONTROL_GROUPS, N_BOUNDED_PARAMETERS)
        if self.group_operator_logits.shape != expected_operators:
            raise ValueError("TSH-CALO group operator logits do not match the action ABI")
        if self.context_operator_logits.shape != (N_LEARNER_CONTEXTS, N_OPERATORS):
            raise ValueError("TSH-CALO learner-context logits do not match the action ABI")
        if (
            self.group_alpha.shape != expected_parameters
            or self.group_beta.shape != expected_parameters
        ):
            raise ValueError("TSH-CALO bounded group parameters do not match the action ABI")
        for name, tensor in (
            ("regime_logits", self.regime_logits),
            ("group_operator_logits", self.group_operator_logits),
            ("context_operator_logits", self.context_operator_logits),
            ("group_alpha", self.group_alpha),
            ("group_beta", self.group_beta),
            ("value", self.value),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"TSH-CALO {name} must be finite")
        if bool((self.group_alpha <= 1.0).any()) or bool((self.group_beta <= 1.0).any()):
            raise ValueError("TSH-CALO beta parameters must remain greater than one")


@dataclass(slots=True)
class HierarchicalPolicyAction:
    schema_version: str
    regime: int
    regime_probabilities: torch.Tensor
    group_operators: torch.Tensor
    group_operator_probabilities: torch.Tensor
    context_operator_logits: torch.Tensor
    group_parameters: torch.Tensor
    action_mask: GroupActionMask

    def validate(self) -> None:
        self.action_mask.validate()
        if self.schema_version != TSH_CALO_ACTION_SCHEMA:
            raise ValueError("Hierarchical action schema is incompatible with TSH-CALO")
        if self.regime < 0 or self.regime >= N_SEARCH_REGIMES:
            raise ValueError("Hierarchical action contains an unknown search regime")
        if self.regime_probabilities.shape != (N_SEARCH_REGIMES,):
            raise ValueError("Hierarchical regime probabilities do not match the action ABI")
        if self.group_operators.shape != (N_CONTROL_GROUPS,):
            raise ValueError("Hierarchical group operators do not match the action ABI")
        if self.group_operator_probabilities.shape != (N_CONTROL_GROUPS, N_OPERATORS):
            raise ValueError("Hierarchical operator probabilities do not match the action ABI")
        if self.context_operator_logits.shape != (N_LEARNER_CONTEXTS, N_OPERATORS):
            raise ValueError("Hierarchical learner-context logits do not match the action ABI")
        if self.group_parameters.shape != (N_CONTROL_GROUPS, N_BOUNDED_PARAMETERS):
            raise ValueError("Hierarchical bounded parameters do not match the action ABI")
        available = self.action_mask.available_groups.to(self.group_operators.device)
        if bool((self.group_operators[~available] != -1).any()):
            raise ValueError("Unavailable groups must use the explicit -1 operator sentinel")
        if bool(((self.group_parameters < 0.0) | (self.group_parameters > 1.0)).any()):
            raise ValueError("Hierarchical bounded parameters must remain within [0, 1]")
        if not bool(torch.isfinite(self.group_parameters).all()):
            raise ValueError("Hierarchical bounded parameters must be finite")


def masked_group_operator_probabilities(
    logits: torch.Tensor, mask: GroupActionMask
) -> torch.Tensor:
    mask.validate()
    if logits.shape != (N_CONTROL_GROUPS, N_OPERATORS):
        raise ValueError("Group operator logits do not match the TSH-CALO action ABI")
    mask = mask.to(logits.device)
    probabilities = torch.zeros_like(logits)
    for group in range(N_CONTROL_GROUPS):
        if not bool(mask.available_groups[group]):
            continue
        group_logits = logits[group].masked_fill(~mask.allowed[group], -torch.inf)
        probabilities[group] = torch.softmax(group_logits, dim=-1)
    return probabilities


def hierarchical_action(
    output: TSHCALOPolicyOutput,
    mask: GroupActionMask,
    *,
    deterministic: bool,
    generator: torch.Generator | None = None,
) -> HierarchicalPolicyAction:
    output.validate()
    regime_probabilities = torch.softmax(output.regime_logits, dim=-1)
    group_probabilities = masked_group_operator_probabilities(output.group_operator_logits, mask)
    device_mask = mask.to(output.regime_logits.device)
    if deterministic:
        regime = int(torch.argmax(regime_probabilities).item())
        group_operators = torch.full(
            (N_CONTROL_GROUPS,), -1, dtype=torch.long, device=output.regime_logits.device
        )
        available = device_mask.available_groups
        group_operators[available] = torch.argmax(group_probabilities[available], dim=-1)
        parameters = output.group_alpha / (output.group_alpha + output.group_beta)
    else:
        regime = int(torch.multinomial(regime_probabilities, 1, generator=generator).item())
        group_operators = torch.full(
            (N_CONTROL_GROUPS,), -1, dtype=torch.long, device=output.regime_logits.device
        )
        for group in range(N_CONTROL_GROUPS):
            if bool(device_mask.available_groups[group]):
                group_operators[group] = torch.multinomial(
                    group_probabilities[group], 1, generator=generator
                )
        # Sampling uses the active tensor device. The generator state is checkpointable and makes
        # stochastic action replay exact; no CPU-only random side channel is introduced.
        alpha_sample = torch._standard_gamma(output.group_alpha, generator=generator)
        beta_sample = torch._standard_gamma(output.group_beta, generator=generator)
        parameters = alpha_sample / (alpha_sample + beta_sample).clamp_min(1e-12)
    action = HierarchicalPolicyAction(
        TSH_CALO_ACTION_SCHEMA,
        regime,
        regime_probabilities,
        group_operators,
        group_probabilities,
        output.context_operator_logits,
        parameters,
        device_mask,
    )
    action.validate()
    return action


def assign_group_conditioned_learner_operators(
    action: HierarchicalPolicyAction,
    learner_groups: torch.Tensor,
    learner_contexts: torch.Tensor,
    *,
    deterministic: bool,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Independently sample each learner from its physical control-group distribution."""

    action.validate()
    groups = torch.as_tensor(
        learner_groups, dtype=torch.long, device=action.group_operator_probabilities.device
    )
    if groups.ndim != 1:
        raise ValueError("learner_groups must be a one-dimensional group-index vector")
    contexts = torch.as_tensor(
        learner_contexts, dtype=torch.long, device=action.group_operator_probabilities.device
    )
    if contexts.shape != groups.shape:
        raise ValueError("learner_contexts must align exactly with learner_groups")
    if groups.numel() and (int(groups.min()) < 0 or int(groups.max()) >= N_CONTROL_GROUPS):
        raise ValueError("learner_groups contains an unknown control group")
    if contexts.numel() and (int(contexts.min()) < 0 or int(contexts.max()) >= N_LEARNER_CONTEXTS):
        raise ValueError("learner_contexts contains an unknown learner context")
    available = action.action_mask.available_groups.to(groups.device)
    if groups.numel() and bool((~available[groups]).any()):
        raise ValueError("A learner cannot be assigned to an unavailable control group")
    group_probabilities = action.group_operator_probabilities[groups]
    combined_logits = (
        torch.log(group_probabilities.clamp_min(1e-12)) + action.context_operator_logits[contexts]
    )
    allowed = action.action_mask.allowed.to(groups.device)[groups]
    probabilities = torch.softmax(combined_logits.masked_fill(~allowed, -torch.inf), dim=-1)
    if deterministic:
        return torch.argmax(probabilities, dim=-1)
    return torch.multinomial(probabilities, 1, replacement=True, generator=generator).squeeze(-1)


class TSHCALOPolicyNetwork(nn.Module):
    """Aggregate-plus-topology actor critic with global and group-conditioned heads."""

    def __init__(self, hidden_dim: int = 64, graph_steps: int = 2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim < 8:
            raise ValueError("TSH-CALO policy hidden_dim must be at least 8")
        self.topology_encoder = TopologyContextEncoder(hidden_dim, graph_steps)
        self.aggregate_encoder = nn.Sequential(
            nn.Linear(POLICY_STATE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.global_fusion = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.group_fusion = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Tanh())
        self.regime_head = nn.Linear(hidden_dim, N_SEARCH_REGIMES)
        self.group_operator_head = nn.Linear(hidden_dim, N_OPERATORS)
        self.context_operator_head = nn.Linear(hidden_dim, N_LEARNER_CONTEXTS * N_OPERATORS)
        self.group_alpha_head = nn.Linear(hidden_dim, N_BOUNDED_PARAMETERS)
        self.group_beta_head = nn.Linear(hidden_dim, N_BOUNDED_PARAMETERS)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: TopologyAwarePolicyState) -> TSHCALOPolicyOutput:
        state.validate()
        parameter = next(self.parameters())
        aggregate = torch.as_tensor(
            state.aggregate_features, device=parameter.device, dtype=parameter.dtype
        )
        aggregate_embedding = self.aggregate_encoder(aggregate)
        topology = self.topology_encoder(state.topology)
        shared = self.global_fusion(
            torch.cat([aggregate_embedding, topology.graph_embedding], dim=-1)
        )
        group_context = self.group_fusion(
            torch.cat([shared.expand(N_CONTROL_GROUPS, -1), topology.group_embeddings], dim=-1)
        )
        alpha = torch.nn.functional.softplus(self.group_alpha_head(group_context)) + 1.1
        beta = torch.nn.functional.softplus(self.group_beta_head(group_context)) + 1.1
        output = TSHCALOPolicyOutput(
            regime_logits=self.regime_head(shared),
            group_operator_logits=self.group_operator_head(group_context),
            context_operator_logits=self.context_operator_head(shared).reshape(
                N_LEARNER_CONTEXTS, N_OPERATORS
            ),
            group_alpha=alpha,
            group_beta=beta,
            value=self.value_head(shared).squeeze(-1),
        )
        output.validate()
        return output
