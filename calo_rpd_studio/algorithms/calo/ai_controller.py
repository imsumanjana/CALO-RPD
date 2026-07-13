"""Hierarchical CALO policy loading and reproducible inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch

from .cognitive_state import STATE_DIM, REGIME_NAMES, rule_based_regime_prior
from .policy_network import CALOPolicyNetwork

PARAMETER_NAMES = (
    "attraction",
    "differential",
    "exploration_sigma",
    "memory_weight",
    "diversity_weight",
    "recovery_fraction",
)
PARAMETER_LOW = np.asarray([0.15, 0.05, 0.005, 0.05, 0.05, 0.05], dtype=float)
PARAMETER_HIGH = np.asarray([1.40, 0.95, 0.30, 1.00, 0.45, 0.45], dtype=float)


@dataclass(slots=True)
class PolicyDecision:
    regime: int
    regime_probabilities: np.ndarray
    operator_probabilities: np.ndarray
    parameters: dict[str, float]
    value_estimate: float


class AIController:
    def __init__(
        self,
        checkpoint=None,
        seed=0,
        deterministic=False,
        input_dim=STATE_DIM,
        device: str = "auto",
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.deterministic = bool(deterministic)
        requested = str(device or "auto").lower()
        xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else ("xpu:0" if xpu_available else "cpu")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA policy inference was requested, but this PyTorch installation cannot access a CUDA GPU."
            )
        if requested.startswith("xpu") and not xpu_available:
            raise RuntimeError(
                "Intel XPU policy inference was requested, but this PyTorch runtime cannot access an XPU device."
            )
        self.device = torch.device(requested)
        self.network = CALOPolicyNetwork(input_dim=input_dim).to(self.device)
        self.metadata: dict = {}
        self.checkpoint_path = ""
        self.checksum = ""
        if checkpoint and Path(checkpoint).exists():
            self.load(checkpoint)
        else:
            torch.manual_seed(2026)
            for parameter in self.network.parameters():
                if parameter.ndim > 1:
                    torch.nn.init.xavier_uniform_(parameter)
                else:
                    torch.nn.init.zeros_(parameter)
        self.network.eval()

    def load(self, path) -> None:
        path = Path(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", payload)
        if "regime_head.weight" not in state_dict or "alpha_head.weight" not in state_dict:
            raise RuntimeError(
                "This checkpoint uses the earlier CALO policy architecture and is not compatible "
                "with CALO Core v2. Select or train a v1.2.x CALO policy checkpoint."
            )
        architecture = payload.get("architecture", {})
        input_dim = int(architecture.get("input_dim", STATE_DIM))
        hidden_dim = int(architecture.get("hidden_dim", 96))
        self.network = CALOPolicyNetwork(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.network.load_state_dict(state_dict)
        self.metadata = dict(payload.get("metadata", {}))
        self.checkpoint_path = str(path)
        self.checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        self.network.eval()

    def decide(self, state) -> PolicyDecision:
        vector = state.vector() if hasattr(state, "vector") else np.asarray(state, float)
        x = torch.tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            regime_logits, operator_logits, alpha, beta, critic_value = self.network(x)
            learned_regime = torch.softmax(regime_logits, dim=-1)[0].cpu().numpy()
            operator_probabilities = torch.softmax(operator_logits, dim=-1)[0].cpu().numpy()
            alpha_values = alpha[0].cpu().numpy()
            beta_values = beta[0].cpu().numpy()

        prior = rule_based_regime_prior(state) if hasattr(state, "feasible_ratio") else np.full(4, 0.25)
        regime_probabilities = 0.35 * learned_regime + 0.65 * prior
        regime_probabilities /= regime_probabilities.sum()
        regime = int(np.argmax(regime_probabilities)) if self.deterministic else int(
            self.rng.choice(len(regime_probabilities), p=regime_probabilities)
        )

        if self.deterministic:
            raw = alpha_values / (alpha_values + beta_values)
        else:
            raw = self.rng.beta(alpha_values, beta_values)
        values = PARAMETER_LOW + raw * (PARAMETER_HIGH - PARAMETER_LOW)
        return PolicyDecision(
            regime=regime,
            regime_probabilities=regime_probabilities,
            operator_probabilities=operator_probabilities / operator_probabilities.sum(),
            parameters={name: float(parameter_value) for name, parameter_value in zip(PARAMETER_NAMES, values)},
            value_estimate=float(critic_value.item()),
        )

    @staticmethod
    def regime_name(index: int) -> str:
        return REGIME_NAMES[int(index)]
