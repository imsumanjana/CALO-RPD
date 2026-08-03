"""Safe-80 device admission for independent TSH-CALO policy training.

Admission is based on a frozen, declared maximum rollout shape and memory that is free/available at
the admission instant.  The estimate is an engineering safety bound, not benchmark evidence.  A
CUDA lease prevents independent heavy owners from racing the same physical device.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
import threading
from typing import ClassVar

import psutil
import torch

from calo_rpd_studio.compute.device_lease import DeviceLeaseUnavailable, ExclusiveDeviceLease
from calo_rpd_studio.compute.memory_budget import calculate_available_memory_admission

from .policy_schema import POLICY_STATE_DIM
from .topology_context import (
    BRANCH_FEATURE_DIM,
    BUS_FEATURE_DIM,
    CONTROL_FEATURE_DIM,
    SCENARIO_FEATURE_DIM,
    TopologyAwarePolicyState,
)


TRAINING_MEMORY_ESTIMATOR_VERSION = "tsh-calo-training-memory-v1"
_FLOAT_BYTES = 4
_INDEX_BYTES = 8
_RUNTIME_OVERHEAD_FLOOR = 64 << 20
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingResourceEnvelope:
    """Declared maxima for one PPO rollout/update; every observed state is checked."""

    rollout_capacity: int
    maximum_population_size: int
    maximum_topology_nodes: int
    maximum_topology_edges: int
    maximum_topology_controls: int
    maximum_scenarios: int

    def validate(self) -> None:
        if self.rollout_capacity < 1:
            raise ValueError("TSH-CALO rollout capacity must be positive")
        if self.maximum_population_size < 2:
            raise ValueError("TSH-CALO maximum population size must be at least two")
        if self.maximum_topology_nodes < 1:
            raise ValueError("TSH-CALO maximum topology node count must be positive")
        if self.maximum_topology_edges < 0 or self.maximum_topology_controls < 0:
            raise ValueError("TSH-CALO topology edge/control maxima cannot be negative")
        if self.maximum_scenarios < 1:
            raise ValueError("TSH-CALO maximum scenario count must be positive")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    def validate_state(self, state: TopologyAwarePolicyState, *, population_size: int) -> None:
        self.validate()
        state.validate()
        topology = state.topology
        observed = {
            "population": int(population_size),
            "topology nodes": int(topology.node_features.shape[0]),
            "topology edges": int(topology.edge_index.shape[1]),
            "topology controls": int(topology.control_features.shape[0]),
            "scenarios": int(topology.scenario_features.shape[0]),
        }
        declared = {
            "population": self.maximum_population_size,
            "topology nodes": self.maximum_topology_nodes,
            "topology edges": self.maximum_topology_edges,
            "topology controls": self.maximum_topology_controls,
            "scenarios": self.maximum_scenarios,
        }
        exceeded = [
            f"{name}={observed[name]}>{declared[name]}"
            for name in observed
            if observed[name] > declared[name]
        ]
        if exceeded:
            raise MemoryError(
                "TSH-CALO state exceeds its frozen training resource envelope: "
                + ", ".join(exceeded)
            )


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingMemoryEstimate:
    estimator_version: str
    parameter_and_buffer_bytes: int
    optimizer_gradient_bytes: int
    retained_rollout_bytes: int
    autograd_activation_bytes: int
    runtime_overhead_floor_bytes: int
    estimated_working_set_bytes: int
    envelope: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingMemoryAdmission:
    requested_device: str
    selected_device: str
    computation_device: str
    estimated_working_set_bytes: int
    total_bytes: int
    available_bytes_at_admission: int
    baseline_reserved_bytes: int
    allowance_bytes: int
    process_ceiling_bytes: int
    allocator_fraction_of_total: float
    fallback_reason: str
    estimator_version: str

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_tsh_calo_training_working_set(
    network: torch.nn.Module,
    envelope: TSHCALOTrainingResourceEnvelope,
) -> TSHCALOTrainingMemoryEstimate:
    """Conservatively estimate a whole PPO update from declared maximum tensor shapes."""

    envelope.validate()
    parameters = sum(int(value.numel() * value.element_size()) for value in network.parameters())
    buffers = sum(int(value.numel() * value.element_size()) for value in network.buffers())
    model_bytes = parameters + buffers
    # Gradients plus Adam first/second moments. Parameters/buffers are accounted separately.
    optimizer_gradient = 3 * parameters
    n = envelope.maximum_topology_nodes
    e = envelope.maximum_topology_edges
    c = envelope.maximum_topology_controls
    s = envelope.maximum_scenarios
    p = envelope.maximum_population_size
    steps = int(getattr(network.topology_encoder, "message_passing_steps", 1))
    hidden = int(getattr(network, "hidden_dim", 64))

    raw_float_values = (
        POLICY_STATE_DIM
        + n * BUS_FEATURE_DIM
        + e * BRANCH_FEATURE_DIM
        + c * CONTROL_FEATURE_DIM
        + s * SCENARIO_FEATURE_DIM
    )
    raw_index_values = 2 * e + 3 * c
    action_values = p * 3 + 128
    retained_per_step = raw_float_values * 8 + raw_index_values * _INDEX_BYTES + action_values * 8
    retained_rollout = envelope.rollout_capacity * retained_per_step

    # Upper-bound the encoder/head autograd tensors retained across a PPO epoch. The multiplier
    # includes concatenation inputs, messages, index-add buffers, nonlinear outputs and gradients.
    dynamic_hidden_rows = 10 * n + (8 + 8 * steps) * e + 12 * c + 6 * s + 256
    activations_per_step = dynamic_hidden_rows * hidden * _FLOAT_BYTES
    autograd_activations = envelope.rollout_capacity * activations_per_step
    subtotal = model_bytes + optimizer_gradient + retained_rollout + autograd_activations
    # A 1.5 safety factor covers allocator fragmentation and distribution temporaries. The explicit
    # floor covers the Python/PyTorch runtime for very small policies without claiming measurement.
    estimated = max(_RUNTIME_OVERHEAD_FLOOR, int(math.ceil(1.5 * subtotal)))
    return TSHCALOTrainingMemoryEstimate(
        TRAINING_MEMORY_ESTIMATOR_VERSION,
        model_bytes,
        optimizer_gradient,
        retained_rollout,
        autograd_activations,
        _RUNTIME_OVERHEAD_FLOOR,
        estimated,
        envelope.to_dict(),
    )


def _cpu_admission(
    estimate: TSHCALOTrainingMemoryEstimate, reason: str, requested: str
) -> TSHCALOTrainingMemoryAdmission:
    memory = psutil.virtual_memory()
    budget = calculate_available_memory_admission(
        total_bytes=int(memory.total),
        available_bytes=int(memory.available),
        requested_fraction=0.80,
    )
    if estimate.estimated_working_set_bytes > budget.additional_allowance_bytes:
        raise MemoryError(
            "TSH-CALO training working set exceeds 80% of currently available CPU RAM"
        )
    return TSHCALOTrainingMemoryAdmission(
        requested,
        "cpu",
        "cpu",
        estimate.estimated_working_set_bytes,
        budget.total_bytes,
        budget.available_bytes_at_admission,
        budget.baseline_reserved_bytes,
        budget.additional_allowance_bytes,
        budget.process_ceiling_bytes,
        budget.allocator_fraction_of_total,
        reason,
        estimate.estimator_version,
    )


class TSHCALOTrainingDeviceGuard:
    """Own the admitted training device and its exclusive CUDA lease."""

    _local_lock: ClassVar[threading.Lock] = threading.Lock()
    _local_cuda_owners: ClassVar[set[str]] = set()

    def __init__(
        self,
        admission: TSHCALOTrainingMemoryAdmission,
        lease: ExclusiveDeviceLease | None = None,
        local_cuda_key: str = "",
    ) -> None:
        self.admission = admission
        self.lease = lease
        self.local_cuda_key = local_cuda_key
        self._closed = False

    @classmethod
    def _claim_local_cuda(cls, key: str) -> None:
        with cls._local_lock:
            if key in cls._local_cuda_owners:
                raise DeviceLeaseUnavailable(
                    "CUDA device already has an independent TSH-CALO trainer in this process"
                )
            cls._local_cuda_owners.add(key)

    @classmethod
    def _release_local_cuda(cls, key: str) -> None:
        if not key:
            return
        with cls._local_lock:
            cls._local_cuda_owners.discard(key)

    @classmethod
    def admit(
        cls,
        estimate: TSHCALOTrainingMemoryEstimate,
        *,
        requested_device: str,
        allow_cpu_fallback: bool,
    ) -> "TSHCALOTrainingDeviceGuard":
        requested = str(requested_device).strip().lower()
        if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
            raise ValueError("TSH-CALO training device must be auto, cpu, cuda, or cuda:<index>")
        if requested == "cpu":
            return cls(_cpu_admission(estimate, "explicit CPU training", requested))

        if not torch.cuda.is_available():
            if requested != "auto" and not allow_cpu_fallback:
                raise RuntimeError("CUDA TSH-CALO training was requested but CUDA is unavailable")
            return cls(_cpu_admission(estimate, "CUDA unavailable", requested))

        selected = torch.device("cuda:0" if requested in {"auto", "cuda"} else requested)
        local_key = str(selected)
        cls._claim_local_cuda(local_key)
        try:
            lease = ExclusiveDeviceLease(local_key)
        except BaseException:
            cls._release_local_cuda(local_key)
            raise
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
            baseline = int(torch.cuda.memory_reserved(selected))
            budget = calculate_available_memory_admission(
                total_bytes=int(total_bytes),
                available_bytes=int(free_bytes),
                requested_fraction=0.80,
                baseline_reserved_bytes=baseline,
            )
            if estimate.estimated_working_set_bytes > budget.additional_allowance_bytes:
                lease.close()
                cls._release_local_cuda(local_key)
                local_key = ""
                if not allow_cpu_fallback:
                    raise MemoryError(
                        "TSH-CALO training working set exceeds 80% of currently free VRAM"
                    )
                admission = _cpu_admission(
                    estimate,
                    "estimated working set exceeds 80% of currently free VRAM",
                    requested,
                )
                return cls(admission)
            index = selected.index if selected.index is not None else 0
            torch.cuda.set_per_process_memory_fraction(budget.allocator_fraction_of_total, index)
            admission = TSHCALOTrainingMemoryAdmission(
                requested,
                str(selected),
                "nvidia_gpu",
                estimate.estimated_working_set_bytes,
                budget.total_bytes,
                budget.available_bytes_at_admission,
                budget.baseline_reserved_bytes,
                budget.additional_allowance_bytes,
                budget.process_ceiling_bytes,
                budget.allocator_fraction_of_total,
                "",
                estimate.estimator_version,
            )
            return cls(admission, lease, local_key)
        except BaseException:
            lease.close()
            cls._release_local_cuda(local_key)
            raise

    def fallback_after_cuda_oom(
        self, estimate: TSHCALOTrainingMemoryEstimate
    ) -> "TSHCALOTrainingDeviceGuard":
        requested = self.admission.requested_device
        self.close()
        admission = _cpu_admission(
            estimate,
            "CUDA allocation failed after VRAM admission",
            requested,
        )
        return type(self)(admission)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.lease is not None:
            self.lease.close()
        self._release_local_cuda(self.local_cuda_key)

    def __enter__(self) -> "TSHCALOTrainingDeviceGuard":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            _LOG.debug("Unable to release the TSH-CALO training device guard", exc_info=True)


def validate_tsh_calo_training_device_provenance(payload: dict) -> None:
    """Validate the immutable Safe-80/computation record embedded in a candidate."""

    record = dict(payload or {})
    estimate = dict(record.get("memory_estimate", {}) or {})
    admission = dict(record.get("memory_admission", {}) or {})
    if str(estimate.get("estimator_version", "")) != TRAINING_MEMORY_ESTIMATOR_VERSION:
        raise ValueError("TSH-CALO training memory estimator version is incompatible")
    if str(admission.get("estimator_version", "")) != TRAINING_MEMORY_ESTIMATOR_VERSION:
        raise ValueError("TSH-CALO training admission estimator version is incompatible")
    estimated = int(estimate.get("estimated_working_set_bytes", 0))
    if estimated < 1 or int(admission.get("estimated_working_set_bytes", 0)) != estimated:
        raise ValueError("TSH-CALO training working-set provenance is inconsistent")
    selected = str(admission.get("selected_device", "")).lower()
    computation = str(admission.get("computation_device", "")).lower()
    requested = str(admission.get("requested_device", "")).lower()
    if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
        raise ValueError("TSH-CALO training provenance has an unsupported requested device")
    if selected == "cpu":
        expected_computation = "cpu"
        expected_semantics = "CPU computes; system RAM is admitted storage"
    elif selected.startswith("cuda:"):
        expected_computation = "nvidia_gpu"
        expected_semantics = "NVIDIA GPU computes; VRAM is admitted storage"
    else:
        raise ValueError("TSH-CALO training provenance identifies an unsupported device")
    if computation != expected_computation:
        raise ValueError("TSH-CALO training computation device is inconsistent")
    fallback_reason = str(admission.get("fallback_reason", ""))
    if (selected == "cpu" and not fallback_reason) or (
        selected.startswith("cuda:") and fallback_reason
    ):
        raise ValueError("TSH-CALO training fallback provenance is inconsistent")
    total = int(admission.get("total_bytes", 0))
    available = int(admission.get("available_bytes_at_admission", -1))
    allowance = int(admission.get("allowance_bytes", -1))
    baseline = int(admission.get("baseline_reserved_bytes", -1))
    ceiling = int(admission.get("process_ceiling_bytes", -1))
    if (
        total <= 0
        or available < 0
        or available > total
        or baseline < 0
        or allowance < estimated
        or allowance > math.floor(0.80 * available)
        or ceiling != baseline + allowance
        or ceiling > total
    ):
        raise ValueError("TSH-CALO training Safe-80 admission provenance is invalid")
    fraction = float(admission.get("allocator_fraction_of_total", -1.0))
    if not math.isclose(fraction, ceiling / total, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("TSH-CALO training allocator ceiling provenance is invalid")
    if str(record.get("computation_semantics", "")) != expected_semantics:
        raise ValueError("TSH-CALO training computation semantics are dishonest or incomplete")
