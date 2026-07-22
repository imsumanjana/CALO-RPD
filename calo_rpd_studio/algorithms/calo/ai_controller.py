"""Hierarchical CALO policy loading and reproducible inference."""

from __future__ import annotations

import logging

from dataclasses import dataclass
import atexit
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

from .cognitive_state import STATE_DIM, REGIME_NAMES, rule_based_regime_prior
from .policy_network import CALOPolicyNetwork
from .policy_schema import build_policy_vector, infer_checkpoint_schema
from calo_rpd_studio.accelerated.runtime_context import get_cross_run_broker
from calo_rpd_studio.ai.model_io import load_checkpoint, verify_checkpoint_hash

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
_POLICY_CACHE_KEY = tuple[str, str, str, str, str]
_POLICY_NETWORK_CACHE: dict[
    _POLICY_CACHE_KEY, tuple[CALOPolicyNetwork, dict, str, dict]
] = {}
_POLICY_BROKER_CACHE: dict[_POLICY_CACHE_KEY, "_PolicyInferenceBroker"] = {}


_LOG = logging.getLogger(__name__)

class PolicyInferenceError(RuntimeError):
    """Base error for fail-closed CALO policy inference."""


class PolicyInferenceTimeout(PolicyInferenceError):
    """Raised when the shared policy-inference broker does not answer in time."""


@dataclass(slots=True)
class _PolicyInferenceRequest:
    vector: np.ndarray
    ready: threading.Event
    result: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None = None
    error: BaseException | None = None


class _PolicyInferenceBroker:
    """Microbatch frozen CALO policy inference across simultaneous comparison runs."""

    def __init__(
        self,
        network: CALOPolicyNetwork,
        device: torch.device,
        *,
        window_ms=1.0,
        max_batch=1024,
        request_timeout_s=30.0,
    ):
        self.network = network
        self.device = device
        self.window = max(0.0001, float(window_ms) / 1000.0)
        self.max_batch = max(1, int(max_batch))
        self.request_timeout_s = max(0.1, float(request_timeout_s))
        self.queue: queue.Queue[_PolicyInferenceRequest | None] = queue.Queue()
        self.closed = threading.Event()
        self._state_lock = threading.Lock()
        self._fatal_error: BaseException | None = None
        self._last_success_monotonic = time.monotonic()
        self.thread = threading.Thread(
            target=self._run, name="CALO-PolicyInferenceBroker", daemon=True
        )
        self.thread.start()

    def _set_fatal_error(self, exc: BaseException) -> None:
        with self._state_lock:
            if self._fatal_error is None:
                self._fatal_error = exc

    def _get_fatal_error(self) -> BaseException | None:
        with self._state_lock:
            return self._fatal_error

    def _fail_pending(self, exc: BaseException) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                return
            if item is None:
                continue
            item.error = exc
            item.ready.set()

    def infer(self, vector: np.ndarray):
        fatal_error = self._get_fatal_error()
        if fatal_error is not None:
            raise PolicyInferenceError("CALO policy-inference broker has failed") from fatal_error
        if self.closed.is_set() or not self.thread.is_alive():
            raise PolicyInferenceError("CALO policy-inference broker is not running")

        request = _PolicyInferenceRequest(np.asarray(vector, dtype=np.float32), threading.Event())
        self.queue.put(request)
        if not request.ready.wait(timeout=self.request_timeout_s):
            fatal_error = self._get_fatal_error()
            alive = self.thread.is_alive()
            age = max(0.0, time.monotonic() - self._last_success_monotonic)
            message = (
                "CALO policy-inference broker timed out after "
                f"{self.request_timeout_s:.3f}s (thread_alive={alive}, "
                f"queue_depth={self.queue.qsize()}, last_success_age_s={age:.3f}). "
                "Policy-assisted execution is fail-closed; no alternate policy or No-AI "
                "fallback was used."
            )
            if fatal_error is not None:
                raise PolicyInferenceTimeout(message) from fatal_error
            raise PolicyInferenceTimeout(message)
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise PolicyInferenceError(
                "CALO policy-inference broker completed a request without a result"
            )
        return request.result

    def _run(self) -> None:
        try:
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
                    self._last_success_monotonic = time.monotonic()
                except Exception as exc:
                    for item in requests:
                        item.error = exc
                finally:
                    for item in requests:
                        item.ready.set()
        except BaseException as exc:
            self._set_fatal_error(exc)
            self.closed.set()
            self._fail_pending(exc)

    def close(self) -> None:
        was_closed = self.closed.is_set()
        self.closed.set()
        if not was_closed:
            self.queue.put(None)
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=5)


def _close_policy_brokers() -> None:
    for broker in list(_POLICY_BROKER_CACHE.values()):
        try:
            broker.close()
        except Exception:
            _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)
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
        expected_checksum: str = "",
        expected_state_schema: str = "",
        expected_action_schema: str = "",
        allow_no_policy: bool = False,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.deterministic = bool(deterministic)
        requested = str(device or "auto").lower()
        xpu_available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        if requested == "auto":
            # CALO's current cognitive/control plane is NumPy/CPU resident. Sending this tiny policy
            # network to CUDA/XPU forces a synchronization + device-to-host copy every decision and
            # is slower/more fragile than CPU inference in the single-run path. Heavy ORPD evaluation
            # remains accelerator-resident. Explicit cuda/xpu remains available for controlled/batched studies.
            requested = "cpu"
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
        self.expected_checksum = str(expected_checksum or "").lower()
        self.expected_state_schema = str(expected_state_schema or "")
        self.expected_action_schema = str(expected_action_schema or "")
        self.input_dim = int(input_dim)
        self.schema: dict = {}
        self._inference_broker = None
        self.batched_inference = False
        self.network: CALOPolicyNetwork | None = None
        if checkpoint:
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(
                    f"CALO policy checkpoint not found: {path}. "
                    "Policy-assisted CALO requires an explicitly imported/trained and activated policy."
                )
            self.load(path)
        elif not allow_no_policy:
            raise RuntimeError(
                "No CALO policy checkpoint was supplied. Policy-assisted CALO is fail-closed: "
                "train or import a compatible policy and explicitly activate/bind it before evaluation."
            )

    def load(self, path) -> None:
        path = Path(path).resolve()
        # Policy identity is content-addressed.  Filesystem metadata such as mtime/size is not
        # sufficient because a file can be replaced while preserving both values.  Verify/hash the
        # actual artifact before cache lookup; on a cache miss load_checkpoint verifies the same
        # digest again before deserializing, closing the stale-cache/replace race.
        checksum = verify_checkpoint_hash(path, self.expected_checksum or None)
        cache_key: _POLICY_CACHE_KEY = (
            str(path),
            checksum.lower(),
            str(self.device),
            self.expected_state_schema,
            self.expected_action_schema,
        )
        with _POLICY_CACHE_LOCK:
            cached = _POLICY_NETWORK_CACHE.get(cache_key)
            if cached is None:
                payload = load_checkpoint(path, expected_sha256=checksum, map_location="cpu")
                state_dict = payload.get("model_state_dict", payload)
                if "regime_head.weight" not in state_dict or "alpha_head.weight" not in state_dict:
                    raise RuntimeError(
                        "This checkpoint uses an unsupported earlier CALO policy architecture. "
                        "Select a compatible hierarchical CALO checkpoint or train a native v4.1 candidate."
                    )
                architecture = payload.get("architecture", {})
                schema = infer_checkpoint_schema(payload)
                input_dim = int(schema["input_dim"])
                hidden_dim = int(architecture.get("hidden_dim", 96))
                network = CALOPolicyNetwork(input_dim=input_dim, hidden_dim=hidden_dim).to(
                    self.device
                )
                network.load_state_dict(state_dict)
                network.eval()
                metadata = dict(payload.get("metadata", {}))
                cached = (network, metadata, checksum, schema)
                _POLICY_NETWORK_CACHE[cache_key] = cached
            self.network, metadata, self.checksum, schema = cached
            self.schema = dict(schema)
            self.input_dim = int(self.schema.get("input_dim", STATE_DIM))
            if self.expected_checksum and self.checksum.lower() != self.expected_checksum:
                raise RuntimeError(
                    "CALO policy checksum does not match the experiment's immutable policy binding"
                )
            if (
                self.expected_state_schema
                and str(self.schema.get("state_schema_version", "")) != self.expected_state_schema
            ):
                raise RuntimeError("CALO policy state schema does not match the experiment binding")
            if (
                self.expected_action_schema
                and str(self.schema.get("action_schema_version", "")) != self.expected_action_schema
            ):
                raise RuntimeError(
                    "CALO policy action schema does not match the experiment binding"
                )
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

    def decide(self, state, runtime_context=None) -> PolicyDecision:
        if self.network is None:
            raise RuntimeError(
                "CALO neural-policy inference was requested without an active policy artifact."
            )
        vector = build_policy_vector(state, runtime_context, input_dim=self.input_dim)
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

        prior = (
            rule_based_regime_prior(state) if hasattr(state, "feasible_ratio") else np.full(4, 0.25)
        )
        regime_probabilities = 0.35 * learned_regime + 0.65 * prior
        regime_probabilities /= regime_probabilities.sum()
        regime = (
            int(np.argmax(regime_probabilities))
            if self.deterministic
            else int(self.rng.choice(len(regime_probabilities), p=regime_probabilities))
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
            parameters={
                name: float(parameter_value)
                for name, parameter_value in zip(PARAMETER_NAMES, values)
            },
            value_estimate=float(critic_scalar),
        )

    @staticmethod
    def regime_name(index: int) -> str:
        return REGIME_NAMES[int(index)]
