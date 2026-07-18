"""Hierarchical CALO policy loading and reproducible inference."""
from __future__ import annotations

from dataclasses import dataclass
import atexit
import hashlib
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .cognitive_state import STATE_DIM, REGIME_NAMES, rule_based_regime_prior
from .policy_network import CALOPolicyNetwork
from calo_rpd_studio.accelerated.runtime_context import get_cross_run_broker
from calo_rpd_studio.ai.model_io import load_checkpoint

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

# Comparative v3.1 workers keep one process per accelerator.  Read-only policy modules are cached
# inside that process and safely shared by concurrent CALO runs; each controller still owns its own
# NumPy action RNG, so run-level stochastic decisions remain seed-isolated.
_POLICY_CACHE_LOCK = threading.Lock()
_POLICY_NETWORK_CACHE: dict[tuple[str, int, int, str], tuple[CALOPolicyNetwork, dict, str]] = {}
_POLICY_BROKER_CACHE: dict[tuple[str, int, int, str], "_PolicyInferenceBroker"] = {}


@dataclass(slots=True)
class _PolicyInferenceRequest:
    vector: np.ndarray
    ready: threading.Event
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    error: BaseException | None = None


class _PolicyInferenceBroker:
    """Microbatch frozen CALO policy inference across simultaneous comparison runs."""

    def __init__(self, network: CALOPolicyNetwork, device: torch.device, *, window_ms=1.0, max_batch=1024):
        self.network = network
        self.device = device
        self.window = max(0.0001, float(window_ms) / 1000.0)
        self.max_batch = max(1, int(max_batch))
        self.queue: queue.Queue[_PolicyInferenceRequest | None] = queue.Queue()
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._run, name="CALO-PolicyInferenceBroker", daemon=True)
        self.thread.start()

    def infer(self, vector: np.ndarray):
        request = _PolicyInferenceRequest(np.asarray(vector, dtype=np.float32), threading.Event())
        self.queue.put(request)
        request.ready.wait()
        if request.error is not None:
            raise request.error
        return request.result

    def _run(self) -> None:
        while not self.closed.is_set():
            try:
                first = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                break
            requests = [first]
            deadline = time.perf_counter() + self.window
            while len(requests) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self.queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    self.closed.set()
                    break
                requests.append(item)
            try:
                states = np.stack([item.vector for item in requests], axis=0)
                x = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                with torch.inference_mode():
                    regime_logits, operator_logits, alpha, beta, critic = self.network(x)
                    learned_regime = torch.softmax(regime_logits, dim=-1).detach().cpu().numpy()
                    operator = torch.softmax(operator_logits, dim=-1).detach().cpu().numpy()
                    alpha_np = alpha.detach().cpu().numpy()
                    beta_np = beta.detach().cpu().numpy()
                    critic_np = critic.detach().cpu().numpy()
                for index, item in enumerate(requests):
                    item.result = (
                        learned_regime[index],
                        operator[index],
                        alpha_np[index],
                        beta_np[index],
                        float(critic_np[index]),
                    )
            except BaseException as exc:
                for item in requests:
                    item.error = exc
            finally:
                for item in requests:
                    item.ready.set()

    def close(self) -> None:
        if self.closed.is_set():
            return
        self.closed.set()
        self.queue.put(None)
        self.thread.join(timeout=5)


def _close_policy_brokers() -> None:
    for broker in list(_POLICY_BROKER_CACHE.values()):
        try:
            broker.close()
        except Exception:
            pass
    _POLICY_BROKER_CACHE.clear()


atexit.register(_close_policy_brokers)


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
        self.metadata: dict = {}
        self.checkpoint_path = ""
        self.checksum = ""
        self._inference_broker = None
        self.batched_inference = False
        if checkpoint and Path(checkpoint).exists():
            self.network = None
            self.load(checkpoint)
        else:
            # Fallback initialization is only for development without a checkpoint.  Serialize it
            # so concurrent run threads cannot race through PyTorch's process-global initializer.
            with _POLICY_CACHE_LOCK:
                self.network = CALOPolicyNetwork(input_dim=input_dim).to(self.device)
                torch.manual_seed(2026)
                for parameter in self.network.parameters():
                    if parameter.ndim > 1:
                        torch.nn.init.xavier_uniform_(parameter)
                    else:
                        torch.nn.init.zeros_(parameter)
        self.network.eval()

    def load(self, path) -> None:
        path = Path(path).resolve()
        stat = path.stat()
        cache_key = (str(path), int(stat.st_mtime_ns), int(stat.st_size), str(self.device))
        with _POLICY_CACHE_LOCK:
            cached = _POLICY_NETWORK_CACHE.get(cache_key)
            if cached is None:
                payload = load_checkpoint(path, map_location="cpu")
                state_dict = payload.get("model_state_dict", payload)
                if "regime_head.weight" not in state_dict or "alpha_head.weight" not in state_dict:
                    raise RuntimeError(
                        "This checkpoint uses the earlier CALO policy architecture and is not compatible "
                        "with CALO Core v2. Select or train a v1.2.x CALO policy checkpoint."
                    )
                architecture = payload.get("architecture", {})
                input_dim = int(architecture.get("input_dim", STATE_DIM))
                hidden_dim = int(architecture.get("hidden_dim", 96))
                network = CALOPolicyNetwork(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
                network.load_state_dict(state_dict)
                network.eval()
                metadata = dict(payload.get("metadata", {}))
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                cached = (network, metadata, checksum)
                _POLICY_NETWORK_CACHE[cache_key] = cached
            self.network, metadata, self.checksum = cached
            if get_cross_run_broker() is not None:
                broker = _POLICY_BROKER_CACHE.get(cache_key)
                if broker is None:
                    broker = _PolicyInferenceBroker(self.network, self.device)
                    _POLICY_BROKER_CACHE[cache_key] = broker
                self._inference_broker = broker
                self.batched_inference = True
        self.metadata = dict(metadata)
        self.checkpoint_path = str(path)
        self.network.eval()

    def decide(self, state) -> PolicyDecision:
        vector = state.vector() if hasattr(state, "vector") else np.asarray(state, float)
        if self._inference_broker is not None:
            learned_regime, operator_probabilities, alpha_values, beta_values, critic_scalar = (
                self._inference_broker.infer(vector)
            )
        else:
            x = torch.tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.inference_mode():
                regime_logits, operator_logits, alpha, beta, critic_value = self.network(x)
                learned_regime = torch.softmax(regime_logits, dim=-1)[0].cpu().numpy()
                operator_probabilities = torch.softmax(operator_logits, dim=-1)[0].cpu().numpy()
                alpha_values = alpha[0].cpu().numpy()
                beta_values = beta[0].cpu().numpy()
                critic_scalar = float(critic_value.item())

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
            value_estimate=float(critic_scalar),
        )

    @staticmethod
    def regime_name(index: int) -> str:
        return REGIME_NAMES[int(index)]
