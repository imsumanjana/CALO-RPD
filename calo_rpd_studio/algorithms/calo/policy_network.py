"""Hierarchical actor-critic policy network used by legacy and native CALO policies."""

from __future__ import annotations

import torch
from torch import nn

from .cognitive_state import STATE_DIM


class CALOPolicyNetwork(nn.Module):
    """Outputs search regime, operator distribution, Beta parameters, and state value."""

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        hidden_dim: int = 96,
        n_regimes: int = 4,
        n_operators: int = 6,
        n_parameters: int = 6,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_regimes = int(n_regimes)
        self.n_operators = int(n_operators)
        self.n_parameters = int(n_parameters)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.regime_head = nn.Linear(self.hidden_dim, self.n_regimes)
        self.operator_head = nn.Linear(self.hidden_dim, self.n_operators)
        self.alpha_head = nn.Linear(self.hidden_dim, self.n_parameters)
        self.beta_head = nn.Linear(self.hidden_dim, self.n_parameters)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        h = self.backbone(x)
        # Alpha and beta above one avoid singular densities at 0/1 and provide stable bounded actions.
        alpha = torch.nn.functional.softplus(self.alpha_head(h)) + 1.1
        beta = torch.nn.functional.softplus(self.beta_head(h)) + 1.1
        return (
            self.regime_head(h),
            self.operator_head(h),
            alpha,
            beta,
            self.value_head(h).squeeze(-1),
        )
