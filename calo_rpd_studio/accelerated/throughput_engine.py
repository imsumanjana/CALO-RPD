"""Batched-throughput orchestration shared by comparison and policy training.

The v3.4 engine deliberately optimizes *candidate-scenario evaluations per second* rather than
trying to force an arbitrary Task Manager utilization percentage.  It provides:

* a thread-safe performance ledger;
* a cross-run evaluation broker that combines compatible population requests arriving within a
  short batching window;
* deterministic automatic microbatch calibration;
* measured-throughput device-share allocation; and
* JSON-serializable calibration profiles for provenance and reuse.

The broker never changes optimizer equations, random seeds, evaluation accounting, constraint
normalization, or feasibility rules.  It only changes how compatible evaluation requests are
packed before calling the same FP64 scientific evaluator.
"""

from __future__ import annotations

import logging

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import queue
import threading
import time
from typing import Any, Iterable

import numpy as np


_LOG = logging.getLogger(__name__)

@dataclass(slots=True)
class StageTiming:
    calls: int = 0
    seconds: float = 0.0
    items: int = 0

    @property
    def items_per_second(self) -> float:
        return 0.0 if self.seconds <= 0 else float(self.items / self.seconds)


class PerformanceLedger:
    """Low-overhead cumulative timing ledger.

    Timings are intentionally aggregated in memory.  No SQLite/file write occurs in the hot path.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: dict[str, StageTiming] = {}

    def add(self, stage: str, seconds: float, items: int = 0) -> None:
        with self._lock:
            record = self._records.setdefault(str(stage), StageTiming())
            record.calls += 1
            record.seconds += max(0.0, float(seconds))
            record.items += max(0, int(items))

    def snapshot(self) -> dict[str, dict[str, float | int]]:
        with self._lock:
            total = sum(record.seconds for record in self._records.values())
            return {
                stage: {
                    "calls": record.calls,
                    "seconds": record.seconds,
                    "items": record.items,
                    "items_per_second": record.items_per_second,
                    "share_percent": 0.0 if total <= 0 else 100.0 * record.seconds / total,
                }
                for stage, record in sorted(self._records.items())
            }


GLOBAL_LEDGER = PerformanceLedger()


class timed_stage:
    """Context manager for cumulative hot-path timing."""

    def __init__(self, stage: str, items: int = 0, ledger: PerformanceLedger | None = None):
        self.stage = stage
        self.items = int(items)
        self.ledger = ledger or GLOBAL_LEDGER
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.ledger.add(self.stage, time.perf_counter() - self.started, self.items)
        return False


@dataclass(frozen=True, slots=True)
class DeviceCalibration:
    device: str
    device_name: str
    batch_size: int
    evaluations_per_second: float
    latency_seconds: float
    candidate_count: int
    repetitions: int
    successful: bool = True
    note: str = ""


@dataclass(slots=True)
class ThroughputProfile:
    case_name: str
    scenario_count: int
    dimension: int
    created_at: float
    devices: dict[str, DeviceCalibration] = field(default_factory=dict)
    scientific_backend: str = "torch_fp64"
    precision: str = "float64"
    calibration_version: str = "3.3"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThroughputProfile":
        devices = {
            key: DeviceCalibration(**value)
            for key, value in dict(payload.get("devices", {})).items()
        }
        return cls(
            case_name=str(payload.get("case_name", "")),
            scenario_count=int(payload.get("scenario_count", 1)),
            dimension=int(payload.get("dimension", 0)),
            created_at=float(payload.get("created_at", 0.0)),
            devices=devices,
            scientific_backend=str(payload.get("scientific_backend", "torch_fp64")),
            precision=str(payload.get("precision", "float64")),
            calibration_version=str(payload.get("calibration_version", "3.3")),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ThroughputProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(slots=True)
class _BatchRequest:
    evaluator: Any
    candidates: Any
    ready: threading.Event = field(default_factory=threading.Event)
    result: list[Any] | None = None
    error: BaseException | None = None


class CrossRunBatchBroker:
    """Combine compatible synchronous population-evaluation requests across run threads.

    Optimizers remain synchronous: a caller blocks until its own evaluations are available.  The
    broker waits a very short configurable window, groups requests with the same scientific
    signature, evaluates the concatenated candidate matrix once, and splits the results back in the
    original order.  This is especially effective for IEEE-30 deterministic campaigns where many
    independent runs repeatedly request populations of 20--100 candidates.
    """

    def __init__(
        self,
        *,
        batch_window_ms: float = 4.0,
        max_candidates: int = 4096,
        ledger: PerformanceLedger | None = None,
    ) -> None:
        self.batch_window_seconds = max(0.0001, float(batch_window_ms) / 1000.0)
        self.max_candidates = max(1, int(max_candidates))
        self.ledger = ledger or GLOBAL_LEDGER
        self._queue: queue.Queue[_BatchRequest | None] = queue.Queue()
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="CALO-CrossRunBatchBroker", daemon=True
        )
        self._thread.start()

    def submit(self, evaluator: Any, candidates: Iterable) -> list[Any]:
        if self._closed.is_set():
            raise RuntimeError("Cross-run batch broker is closed")
        try:
            import torch

            if isinstance(candidates, torch.Tensor):
                array = candidates
                if array.ndim == 1:
                    array = array.unsqueeze(0)
            else:
                array = np.asarray(candidates, dtype=float)
                if array.ndim == 1:
                    array = array[None, :]
        except (ImportError, TypeError, ValueError, RuntimeError):
            array = np.asarray(candidates, dtype=float)
            if array.ndim == 1:
                array = array[None, :]

        if getattr(array, "ndim", None) != 2:
            raise ValueError(
                "Cross-run batching requires a two-dimensional candidate matrix; "
                f"received shape {getattr(array, 'shape', None)!r}."
            )
        if len(array) == 0:
            return []

        request = _BatchRequest(evaluator=evaluator, candidates=array)
        self._queue.put(request)
        while not request.ready.wait(timeout=0.25):
            if not self._thread.is_alive():
                self._closed.set()
                raise RuntimeError(
                    "Cross-run batch broker terminated unexpectedly before returning the evaluation."
                )
        if request.error is not None:
            raise request.error
        return list(request.result or [])

    @staticmethod
    def _signature(evaluator: Any, candidates: Any | None = None) -> str:
        signature = getattr(evaluator, "batch_signature", None)
        scientific = str(signature() if callable(signature) else signature)
        if candidates is None:
            return scientific
        try:
            import torch

            if isinstance(candidates, torch.Tensor):
                width = int(candidates.shape[1]) if candidates.ndim == 2 else -1
                representation = (
                    f"torch:{candidates.device}:{candidates.dtype}:{candidates.layout}:{width}"
                )
                return f"{scientific}|{representation}"
        except (ImportError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
            _LOG.debug("Torch batch-signature inspection unavailable; using NumPy signature: %s", exc)
        array = np.asarray(candidates)
        width = int(array.shape[1]) if array.ndim == 2 else -1
        return f"{scientific}|numpy:{array.dtype.str}:{width}"

    def _flush_group(self, requests: list[_BatchRequest]) -> None:
        if not requests:
            return
        evaluator = requests[0].evaluator
        offsets: list[tuple[_BatchRequest, int, int]] = []
        matrices: list[Any] = []
        candidate_count = 0
        for request in requests:
            size = len(request.candidates)
            offsets.append((request, candidate_count, candidate_count + size))
            matrices.append(request.candidates)
            candidate_count += size
        started = time.perf_counter()
        try:
            tensor_batch = False
            try:
                import torch

                tensor_batch = bool(matrices and isinstance(matrices[0], torch.Tensor))
            except (ImportError, RuntimeError, TypeError, AttributeError):
                tensor_batch = False
            if tensor_batch:
                import torch

                if not all(isinstance(matrix, torch.Tensor) for matrix in matrices):
                    raise TypeError(
                        "A cross-run batch mixed Torch and NumPy candidate matrices. "
                        "Requests must use one representation per compatibility group."
                    )
                reference = matrices[0]
                for matrix in matrices[1:]:
                    if matrix.device != reference.device or matrix.dtype != reference.dtype:
                        raise TypeError(
                            "A cross-run Torch batch mixed devices or dtypes: "
                            f"expected {reference.device}/{reference.dtype}, got "
                            f"{matrix.device}/{matrix.dtype}."
                        )
                    if matrix.ndim != 2 or matrix.shape[1] != reference.shape[1]:
                        raise ValueError(
                            "A cross-run Torch batch contained incompatible candidate shapes: "
                            f"expected (*, {reference.shape[1]}), got {tuple(matrix.shape)}."
                        )
                combined = torch.cat(matrices, dim=0)
                tensor_direct = getattr(evaluator, "_evaluate_population_tensor_direct", None)
                if callable(tensor_direct):
                    results = list(tensor_direct(combined).to_evaluations())
                else:
                    results = list(getattr(evaluator, "_evaluate_population_direct")(combined))
            else:
                reference_width = int(np.asarray(matrices[0]).shape[1])
                for matrix in matrices[1:]:
                    array = np.asarray(matrix)
                    if array.ndim != 2 or int(array.shape[1]) != reference_width:
                        raise ValueError(
                            "A cross-run NumPy batch contained incompatible candidate shapes: "
                            f"expected (*, {reference_width}), got {array.shape}."
                        )
                combined = np.concatenate(matrices, axis=0)
                direct = getattr(evaluator, "_evaluate_population_direct")
                results = list(direct(combined))
            if len(results) != candidate_count:
                raise RuntimeError(
                    f"Cross-run batch returned {len(results)} results for {candidate_count} candidates"
                )
            for request, start, end in offsets:
                request.result = results[start:end]
        except Exception as exc:  # propagate the same scientific failure to every requester
            for request, _start, _end in offsets:
                request.error = exc
        finally:
            # Use the precomputed request count.  `combined` may not exist when concatenation itself
            # fails, and accounting must never mask the original scientific/runtime exception.
            self.ledger.add(
                "cross_run_batch",
                time.perf_counter() - started,
                candidate_count,
            )
            for request, _start, _end in offsets:
                request.ready.set()

    def _run(self) -> None:
        backlog: list[_BatchRequest] = []
        while not self._closed.is_set():
            try:
                first = backlog.pop(0) if backlog else self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if first is None:
                break
            try:
                signature = self._signature(first.evaluator, first.candidates)
            except Exception as exc:
                first.error = exc
                first.ready.set()
                continue
            group = [first]
            count = len(first.candidates)
            deadline = time.perf_counter() + self.batch_window_seconds
            while count < self.max_candidates:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    request = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    self._closed.set()
                    break
                try:
                    request_signature = self._signature(request.evaluator, request.candidates)
                except Exception as exc:
                    request.error = exc
                    request.ready.set()
                    continue
                request_count = len(request.candidates)
                if request_signature == signature and count + request_count <= self.max_candidates:
                    group.append(request)
                    count += request_count
                else:
                    backlog.append(request)
            try:
                self._flush_group(group)
            except Exception as exc:
                # `_flush_group` is designed to report errors to every requester itself.  This
                # outer guard prevents any unforeseen bookkeeping bug from killing the daemon and
                # leaving GUI/experiment threads blocked forever.
                for request in group:
                    if request.error is None:
                        request.error = exc
                    request.ready.set()

        # Fail any late callers rather than leave them blocked during shutdown.
        pending = backlog
        while True:
            try:
                request = self._queue.get_nowait()
            except queue.Empty:
                break
            if request is not None:
                pending.append(request)
        for request in pending:
            request.error = RuntimeError("Cross-run batch broker stopped before evaluation")
            request.ready.set()

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        self._thread.join(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def largest_remainder_counts(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Integer allocation proportional to non-negative measured throughputs."""
    names = list(weights)
    positive = {name: max(0.0, float(weights[name])) for name in names}
    weight_sum = sum(positive.values())
    if total <= 0:
        return {name: 0 for name in names}
    if weight_sum <= 0:
        return {name: (total if index == len(names) - 1 else 0) for index, name in enumerate(names)}
    exact = {name: total * positive[name] / weight_sum for name in names}
    counts = {name: int(np.floor(exact[name])) for name in names}
    remaining = total - sum(counts.values())
    ranked = sorted(names, key=lambda name: (-(exact[name] - counts[name]), names.index(name)))
    for name in ranked[:remaining]:
        counts[name] += 1
    return counts


def measured_throughput_allocation(
    total_jobs: int,
    throughputs: dict[str, float],
    *,
    enabled: dict[str, bool] | None = None,
) -> dict[str, int]:
    """Allocate whole jobs according to measured device throughput."""
    enabled = enabled or {name: True for name in throughputs}
    weights = {
        name: (float(value) if enabled.get(name, False) else 0.0)
        for name, value in throughputs.items()
    }
    if sum(weights.values()) <= 0:
        weights["cpu"] = max(float(throughputs.get("cpu", 1.0)), 1.0)
    return largest_remainder_counts(int(total_jobs), weights)


def calibrate_evaluator(
    evaluator: Any,
    *,
    batch_sizes: Iterable[int] = (16, 32, 64, 128, 256),
    repetitions: int = 2,
    seed: int = 31_415,
    synchronize: bool = True,
) -> DeviceCalibration:
    """Measure a problem/device pair and choose the best stable candidate batch size.

    The calibration uses valid normalized decision vectors and the same evaluator.  Its evaluations
    are not part of an optimizer run and therefore are not included in any algorithm budget.
    """
    rng = np.random.default_rng(seed)
    dimension = int(evaluator.dimension)
    device = str(getattr(evaluator, "device", "cpu"))
    device_name = str(getattr(getattr(evaluator, "device_context", None), "name", device))
    best: DeviceCalibration | None = None
    original_batch_size = int(getattr(evaluator, "batch_size", 1))
    for requested in sorted({max(1, int(value)) for value in batch_sizes}):
        candidates = rng.random((requested, dimension))
        try:
            # The candidate count and the evaluator microbatch must both change; otherwise this
            # would only benchmark a different number of fixed-size chunks rather than calibrate
            # the actual device microbatch.
            if hasattr(evaluator, "batch_size"):
                evaluator.batch_size = int(requested)
            # Warm-up once to initialize libraries/context and populate invariant caches.
            evaluator._evaluate_population_direct(candidates[: min(len(candidates), 8)])
            started = time.perf_counter()
            processed = 0
            for _ in range(max(1, int(repetitions))):
                evaluator._evaluate_population_direct(candidates)
                processed += len(candidates)
            if synchronize:
                try:
                    import torch

                    if device.startswith("cuda"):
                        torch.cuda.synchronize(device)
                    elif device.startswith("xpu") and hasattr(torch, "xpu"):
                        torch.xpu.synchronize(device)
                except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
                    _LOG.warning("Accelerator synchronization failed during calibration on %s: %s", device, exc)
            seconds = max(time.perf_counter() - started, 1e-12)
            record = DeviceCalibration(
                device=device,
                device_name=device_name,
                batch_size=requested,
                evaluations_per_second=float(processed / seconds),
                latency_seconds=float(seconds / max(1, repetitions)),
                candidate_count=processed,
                repetitions=max(1, int(repetitions)),
            )
        except Exception as exc:
            record = DeviceCalibration(
                device=device,
                device_name=device_name,
                batch_size=requested,
                evaluations_per_second=0.0,
                latency_seconds=float("inf"),
                candidate_count=0,
                repetitions=max(1, int(repetitions)),
                successful=False,
                note=str(exc),
            )
        if record.successful and (
            best is None or record.evaluations_per_second > best.evaluations_per_second
        ):
            best = record
    if hasattr(evaluator, "batch_size"):
        evaluator.batch_size = original_batch_size
    if best is None:
        return DeviceCalibration(
            device=device,
            device_name=device_name,
            batch_size=max(1, int(next(iter(batch_sizes), 64))),
            evaluations_per_second=0.0,
            latency_seconds=float("inf"),
            candidate_count=0,
            repetitions=max(1, int(repetitions)),
            successful=False,
            note="No calibration batch completed successfully",
        )
    return best
