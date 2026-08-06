"""CUDA VRAM residency governor for CALO-RPD Studio.

The governor implements the v6.9 execution contract:

* cap new CALO-RPD CUDA allocations at a fraction of VRAM available at admission (80% by default);
* keep every active CUDA-eligible tensor on the device until a population request completes;
* recover from transient CUDA OOM by reducing only the active microbatch and retrying on CUDA;
* never silently fall back to CPU because a CUDA microbatch was too large; and
* expose truthful allocation/peak/retry telemetry without synchronising the hot loop.

The module deliberately does not call ``empty_cache`` during normal execution.  It is used only after
an actual OOM has unwound and temporary references have been released.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import logging
import threading
from typing import Any, Callable

from calo_rpd_studio.compute.memory_budget import calculate_available_memory_admission
from calo_rpd_studio.compute.device_lease import ExclusiveDeviceLease


_LOG = logging.getLogger(__name__)


class CudaCapacityExhausted(RuntimeError):
    """CUDA cannot execute the request even at the declared minimum microbatch."""

    def __init__(self, message: str, metadata: dict[str, Any]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


@dataclass(slots=True, frozen=True)
class VramResidencyPolicy:
    """Configuration for one CUDA process.

    ``budget_fraction`` applies to global VRAM that is free immediately before admission. It is a
    maximum additional allowance, not a request to reserve or fill that memory. The calculated
    process ceiling is frozen for the governor lifetime.
    """

    budget_fraction: float = 0.80
    oom_retry_count: int = 4
    minimum_microbatch: int = 1
    retain_outputs_on_device: bool = True

    def validate(self) -> None:
        if not 0.10 <= float(self.budget_fraction) <= 0.80:
            raise ValueError("CUDA VRAM budget fraction must be positive and no greater than 0.80")
        if int(self.oom_retry_count) < 0:
            raise ValueError("CUDA OOM retry count must be non-negative")
        if int(self.minimum_microbatch) <= 0:
            raise ValueError("CUDA minimum microbatch must be positive")


@dataclass(slots=True)
class VramResidencyStats:
    device: str
    enabled: bool
    budget_fraction: float
    total_bytes: int = 0
    free_bytes_at_start: int = 0
    baseline_reserved_bytes: int = 0
    additional_allowance_bytes: int = 0
    process_budget_bytes: int = 0
    allocator_fraction_of_total: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    oom_retries: int = 0
    microbatches: int = 0
    smallest_microbatch: int = 0
    largest_microbatch: int = 0
    cpu_fallbacks: int = 0
    staged_host_requests: int = 0
    execution_state: str = "not_started"
    last_fallback_reason: str = ""
    physical_device_id: str = ""
    lease_host_scope: str = ""
    lease_container_scope: str = ""
    request_count: int = 0
    last_request_candidates: int = 0
    last_request_oom_retries: int = 0
    last_request_microbatches: int = 0
    last_request_smallest_microbatch: int = 0
    last_request_largest_microbatch: int = 0
    last_request_execution_state: str = "not_started"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["device_residency_contract"] = (
            "all active CUDA-eligible tensors remain on CUDA; only packed completed results may be materialized"
        )
        payload["memory_admission_contract"] = (
            "additional allowance is frozen at admission_fraction * globally_free_vram_at_admission"
        )
        payload["oom_policy"] = "reduce_cuda_microbatch_then_raise_typed_capacity_exhaustion"
        payload["request_statistics"] = {
            "request_count": int(self.request_count),
            "candidates": int(self.last_request_candidates),
            "oom_retries": int(self.last_request_oom_retries),
            "microbatches": int(self.last_request_microbatches),
            "smallest_microbatch": int(self.last_request_smallest_microbatch),
            "largest_microbatch": int(self.last_request_largest_microbatch),
            "execution_state": str(self.last_request_execution_state),
        }
        payload["governor_lifetime_statistics"] = {
            "oom_retries": int(self.oom_retries),
            "microbatches": int(self.microbatches),
            "smallest_microbatch": int(self.smallest_microbatch),
            "largest_microbatch": int(self.largest_microbatch),
            "cpu_fallbacks": int(self.cpu_fallbacks),
            "staged_host_requests": int(self.staged_host_requests),
            "peak_allocated_bytes": int(self.peak_allocated_bytes),
            "peak_reserved_bytes": int(self.peak_reserved_bytes),
        }
        return payload


class VramResidencyGovernor:
    """Process-local CUDA memory governor with fail-closed CUDA retry semantics."""

    _fraction_lock = threading.Lock()
    _configured_fractions: dict[str, float] = {}

    def __init__(
        self,
        device: Any,
        policy: VramResidencyPolicy | None = None,
        *,
        physical_device_id: str = "",
        lease_host_scope: str = "",
        lease_container_scope: str = "",
        lease_wait_timeout_seconds: float | None = None,
        lease_cancel_callback: Callable[[], bool] | None = None,
    ) -> None:
        self.policy = policy or VramResidencyPolicy()
        self.policy.validate()
        self.device = device
        self.device_text = str(device)
        self._torch = None
        self._enabled = False
        self._device_lease = None
        self.physical_device_id = str(physical_device_id or "")
        self.lease_host_scope = str(lease_host_scope or "")
        self.lease_container_scope = str(lease_container_scope or "")
        self.lease_wait_timeout_seconds = lease_wait_timeout_seconds
        self.lease_cancel_callback = lease_cancel_callback
        self.stats = VramResidencyStats(
            device=self.device_text,
            enabled=False,
            budget_fraction=float(self.policy.budget_fraction),
            physical_device_id=self.physical_device_id,
            lease_host_scope=self.lease_host_scope,
            lease_container_scope=self.lease_container_scope,
        )
        self._configure()

    @property
    def enabled(self) -> bool:
        return bool(self._enabled)

    def _configure(self) -> None:
        try:
            import torch
        except ImportError:
            return
        self._torch = torch
        try:
            device = torch.device(self.device)
        except (TypeError, RuntimeError):
            return
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        index = torch.cuda.current_device() if device.index is None else int(device.index)
        canonical = f"cuda:{index}"
        self._device_lease = ExclusiveDeviceLease(
            canonical,
            physical_device_id=self.physical_device_id or canonical,
            host_scope=self.lease_host_scope,
            container_scope=self.lease_container_scope,
            wait=True,
            timeout_seconds=self.lease_wait_timeout_seconds,
            cancel_callback=self.lease_cancel_callback,
        )
        requested_fraction = float(self.policy.budget_fraction)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        try:
            baseline_reserved = int(torch.cuda.memory_reserved(index))
        except (AttributeError, RuntimeError, ValueError):
            baseline_reserved = 0
        admission = calculate_available_memory_admission(
            total_bytes=int(total_bytes),
            available_bytes=int(free_bytes),
            requested_fraction=requested_fraction,
            baseline_reserved_bytes=baseline_reserved,
        )
        allocator_fraction = float(admission.allocator_fraction_of_total)
        # PyTorch's setting is process-wide per device and is expressed as a fraction of *total*
        # memory. Convert the availability-based byte ceiling to that API representation. The
        # strictest previously installed ceiling is preserved so same-process components cannot
        # expand the admission after allocations begin.
        with self._fraction_lock:
            previous = self._configured_fractions.get(canonical)
            effective = (
                allocator_fraction if previous is None else min(float(previous), allocator_fraction)
            )
            if previous is None or effective < float(previous) - 1e-12:
                torch.cuda.set_per_process_memory_fraction(effective, index)
                self._configured_fractions[canonical] = effective
        self.device = torch.device(canonical)
        self.device_text = canonical
        self._enabled = True
        self.stats.device = canonical
        self.stats.enabled = True
        self.stats.budget_fraction = requested_fraction
        self.stats.total_bytes = int(total_bytes)
        self.stats.free_bytes_at_start = int(free_bytes)
        self.stats.baseline_reserved_bytes = baseline_reserved
        effective_ceiling = int(
            total_bytes * self._configured_fractions.get(canonical, allocator_fraction)
        )
        self.stats.process_budget_bytes = effective_ceiling
        self.stats.additional_allowance_bytes = max(0, effective_ceiling - baseline_reserved)
        self.stats.allocator_fraction_of_total = float(
            self._configured_fractions.get(canonical, allocator_fraction)
        )

    def close(self) -> None:
        lease = self._device_lease
        self._device_lease = None
        if lease is not None:
            lease.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            _LOG.debug("Unable to release the CUDA device lease during finalization", exc_info=True)

    def reset_peak_stats(self) -> None:
        if not self._enabled:
            return
        try:
            self._torch.cuda.reset_peak_memory_stats(self.device)
        except (RuntimeError, ValueError):
            pass

    def note_cpu_fallback(self, reason: str) -> None:
        """Record one explicit full-request CPU restart without mixing request/lifetime counts."""

        self.stats.cpu_fallbacks += 1
        self.stats.execution_state = "cpu_fallback"
        self.stats.last_fallback_reason = str(reason)
        self.stats.last_request_execution_state = "cpu_fallback"

    def _update_memory_stats(self) -> None:
        if not self._enabled:
            return
        torch = self._torch
        try:
            self.stats.peak_allocated_bytes = max(
                int(self.stats.peak_allocated_bytes),
                int(torch.cuda.max_memory_allocated(self.device)),
            )
            self.stats.peak_reserved_bytes = max(
                int(self.stats.peak_reserved_bytes),
                int(torch.cuda.max_memory_reserved(self.device)),
            )
        except (RuntimeError, ValueError):
            pass

    def is_cuda_oom(self, exc: BaseException) -> bool:
        if not self._enabled or self._torch is None:
            return False
        oom_type = getattr(self._torch, "OutOfMemoryError", ())
        if oom_type and isinstance(exc, oom_type):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()

    def run_microbatched(
        self,
        population: Any,
        evaluate_once: Callable[[Any], Any],
        concatenate: Callable[[list[Any], dict[str, Any]], Any],
        *,
        preferred_microbatch: int | None = None,
    ) -> Any:
        """Evaluate a population on CUDA with OOM backoff and no CPU fallback.

        ``evaluate_once`` must return a device-resident result.  Only completed result tensors are
        retained between microbatches; large Newton/Jacobian workspaces are released when the call
        returns.  ``concatenate`` joins those small device outputs without host materialisation.
        """

        total = int(population.shape[0])
        if total <= 0:
            raise ValueError("Population must contain at least one candidate")
        if not self._enabled:
            self.stats.execution_state = "cpu_direct"
            self.stats.request_count += 1
            self.stats.last_request_candidates = total
            self.stats.last_request_execution_state = "cpu_direct"
            return evaluate_once(population)

        self.stats.request_count += 1
        self.stats.last_request_candidates = total
        self.stats.last_request_oom_retries = 0
        self.stats.last_request_microbatches = 0
        self.stats.last_request_smallest_microbatch = 0
        self.stats.last_request_largest_microbatch = 0
        self.stats.last_request_execution_state = "running"

        initial = (
            total if preferred_microbatch is None else min(total, max(1, int(preferred_microbatch)))
        )
        minimum = min(initial, max(1, int(self.policy.minimum_microbatch)))
        chunk_size = initial
        offset = 0
        outputs: list[Any] = []
        completed_sizes: list[int] = []
        retries_for_request = 0
        started_from_host = str(getattr(population, "device", "cpu")).split(":", 1)[0] == "cpu"
        if started_from_host:
            self.stats.staged_host_requests += 1
        self.reset_peak_stats()

        while offset < total:
            requested = min(chunk_size, total - offset)
            local_size = requested
            local_retries = 0
            while True:
                try:
                    result = evaluate_once(population[offset : offset + local_size])
                    outputs.append(result)
                    completed_sizes.append(local_size)
                    self.stats.microbatches += 1
                    self.stats.smallest_microbatch = (
                        local_size
                        if self.stats.smallest_microbatch == 0
                        else min(self.stats.smallest_microbatch, local_size)
                    )
                    self.stats.largest_microbatch = max(self.stats.largest_microbatch, local_size)
                    offset += local_size
                    # Keep a successful reduced size for the rest of this request.  This avoids
                    # repeatedly provoking OOM and does not move work to CPU.
                    chunk_size = local_size
                    break
                except BaseException as exc:
                    if not self.is_cuda_oom(exc):
                        raise
                    local_retries += 1
                    retries_for_request += 1
                    self.stats.oom_retries += 1
                    if local_retries > int(self.policy.oom_retry_count) or local_size <= minimum:
                        self._update_memory_stats()
                        self.stats.execution_state = "cuda_capacity_exhausted"
                        self.stats.last_fallback_reason = "minimum_cuda_microbatch_exhausted"
                        self.stats.last_request_oom_retries = retries_for_request
                        self.stats.last_request_microbatches = len(outputs)
                        self.stats.last_request_smallest_microbatch = min(
                            completed_sizes, default=0
                        )
                        self.stats.last_request_largest_microbatch = max(completed_sizes, default=0)
                        self.stats.last_request_execution_state = "cuda_capacity_exhausted"
                        metadata = self.stats.to_dict()
                        metadata.update(
                            {
                                "request_candidates": total,
                                "request_oom_retries": retries_for_request,
                                "requested_initial_microbatch": initial,
                                "completed_cuda_microbatches_before_restart": len(outputs),
                                "input_staged_from_host": started_from_host,
                            }
                        )
                        raise CudaCapacityExhausted(
                            "CUDA VRAM allowance exhausted after adaptive microbatch retries",
                            metadata,
                        ) from exc
                    local_size = max(minimum, local_size // 2)
                    # The failed call has unwound.  Releasing cached free blocks here is an OOM-only
                    # recovery action, never part of the normal hot loop.
                    try:
                        self._torch.cuda.empty_cache()
                    except (RuntimeError, ValueError):
                        pass

        self._update_memory_stats()
        self.stats.execution_state = (
            "cuda_resident"
            if len(outputs) == 1 and retries_for_request == 0 and not started_from_host
            else "cuda_staged_host"
            if started_from_host
            else "cuda_microbatched"
        )
        self.stats.last_request_oom_retries = retries_for_request
        self.stats.last_request_microbatches = len(outputs)
        self.stats.last_request_smallest_microbatch = min(completed_sizes, default=0)
        self.stats.last_request_largest_microbatch = max(completed_sizes, default=0)
        self.stats.last_request_execution_state = self.stats.execution_state
        metadata = self.stats.to_dict()
        metadata.update(
            {
                "request_candidates": total,
                "request_oom_retries": retries_for_request,
                "requested_initial_microbatch": initial,
                "completed_microbatches": len(outputs),
                "host_materializations_during_compute": 0,
                "cpu_inner_loop_participation": False,
                "input_staged_from_host": started_from_host,
            }
        )
        return concatenate(outputs, metadata)
