"""Device-resident synthetic curriculum kernels for CALO policy training.

v6.4 Stage B keeps the stochastic CALO controller and scientific transition semantics on the
trusted reference path, while moving the synthetic curriculum objective/constraint population
kernel to persistent PyTorch tensors on the admitted accelerator. Multiple simultaneous episode
requests are cross-episode microbatched by :class:`SyntheticCrossEpisodeBatchBroker` so CUDA/XPU
receives materially larger FP64 batches instead of one tiny population at a time.

The first population request for every generated curriculum problem is checked against the NumPy
reference implementation. A mismatch fails closed by default. This startup parity gate preserves
scientific trust while avoiding a permanent double-computation tax after the problem/device pair
has been certified.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import logging
import queue
import threading
import time
from typing import Iterable, Sequence

import numpy as np
import torch

_LOG = logging.getLogger(__name__)
_MAX_DIMENSION = 20
_MAX_LEVELS = 17
_STATIC_STACK_CACHE_MAX_ENTRIES = 32
_STATIC_STACK_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_STATIC_STACK_CACHE_LOCK = threading.RLock()


@dataclass(slots=True)
class DeviceSyntheticEvaluation:
    value: float
    feasible: bool
    violation: float
    metadata: dict


@dataclass(slots=True)
class _BatchRequest:
    problem: "DeviceResidentCurriculumProblem"
    population: np.ndarray
    event: threading.Event
    result: list[DeviceSyntheticEvaluation] | None = None
    error: BaseException | None = None


class DeviceResidentCurriculumProblem:
    """Accelerator-resident mirror of one generated :class:`CurriculumProblem`.

    The wrapper does not consume RNG state and therefore can replace the evaluator without
    changing problem generation or subsequent CALO controller random streams.
    """

    def __init__(
        self,
        reference,
        *,
        device: str,
        broker: "SyntheticCrossEpisodeBatchBroker | None" = None,
        parity_tolerance: float = 1e-9,
        require_startup_parity: bool = True,
        parity_recheck_interval: int = 16,
    ) -> None:
        self.reference = reference
        self.stage = int(reference.stage)
        self.dimension = int(reference.dimension)
        self.variables = reference.variables
        self.decoder = reference.decoder
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.broker = broker
        self.parity_tolerance = float(parity_tolerance)
        self.require_startup_parity = bool(require_startup_parity)
        self.parity_recheck_interval = max(0, int(parity_recheck_interval))
        self._evaluation_calls = 0
        self._parity_verified = False
        self._parity_max_error = 0.0

        if self.dimension > _MAX_DIMENSION:
            raise ValueError(
                f"Synthetic curriculum dimension {self.dimension} exceeds Stage-B maximum "
                f"{_MAX_DIMENSION}."
            )

        # Fixed-size padded tensors allow heterogeneous curriculum dimensions to share one GPU
        # microbatch without altering the original generated task.
        shift = np.zeros(_MAX_DIMENSION, dtype=np.float64)
        shift[: self.dimension] = np.asarray(reference.shift, dtype=np.float64)
        rotation = np.zeros((_MAX_DIMENSION, _MAX_DIMENSION), dtype=np.float64)
        rotation[: self.dimension, : self.dimension] = np.asarray(
            reference.rotation, dtype=np.float64
        )
        centres = np.zeros((4, _MAX_DIMENSION), dtype=np.float64)
        centres[:, : self.dimension] = np.asarray(reference.constraint_centres, dtype=np.float64)
        normals = np.zeros((4, _MAX_DIMENSION), dtype=np.float64)
        normals[:, : self.dimension] = np.asarray(reference.constraint_normals, dtype=np.float64)
        dimension_mask = np.zeros(_MAX_DIMENSION, dtype=np.float64)
        dimension_mask[: self.dimension] = 1.0

        level_values = np.zeros((_MAX_DIMENSION, _MAX_LEVELS), dtype=np.float64)
        level_mask = np.zeros((_MAX_DIMENSION, _MAX_LEVELS), dtype=bool)
        discrete_mask = np.zeros(_MAX_DIMENSION, dtype=bool)
        for index, variable in enumerate(reference.variables):
            values = tuple(float(v) for v in getattr(variable, "values", ()) or ())
            if values:
                discrete_mask[index] = True
                if len(values) > _MAX_LEVELS:
                    raise ValueError("Synthetic discrete lattice exceeds supported level count")
                level_values[index, : len(values)] = np.asarray(values, dtype=np.float64)
                level_mask[index, : len(values)] = True

        self.shift = torch.as_tensor(shift, dtype=self.dtype, device=self.device)
        self.rotation = torch.as_tensor(rotation, dtype=self.dtype, device=self.device)
        self.constraint_centres = torch.as_tensor(centres, dtype=self.dtype, device=self.device)
        self.constraint_normals = torch.as_tensor(normals, dtype=self.dtype, device=self.device)
        self.dimension_mask = torch.as_tensor(dimension_mask, dtype=self.dtype, device=self.device)
        self.discrete_mask = torch.as_tensor(discrete_mask, dtype=torch.bool, device=self.device)
        self.level_values = torch.as_tensor(level_values, dtype=self.dtype, device=self.device)
        self.level_mask = torch.as_tensor(level_mask, dtype=torch.bool, device=self.device)
        self.narrowness = float(reference.narrowness)
        digest = hashlib.sha256()
        digest.update(str(self.device).encode("utf-8"))
        digest.update(np.asarray([self.stage, self.dimension, self.narrowness], dtype=np.float64).tobytes())
        for tensor in (shift, rotation, centres, normals, dimension_mask, discrete_mask, level_values, level_mask):
            digest.update(np.ascontiguousarray(tensor).tobytes())
        self.static_fingerprint = digest.hexdigest()

    @property
    def parity_verified(self) -> bool:
        return bool(self._parity_verified)

    @property
    def parity_max_error(self) -> float:
        return float(self._parity_max_error)

    def evaluate(self, x) -> DeviceSyntheticEvaluation:
        return self.evaluate_population(np.asarray(x, dtype=float)[None, :])[0]

    def evaluate_population(self, population: Iterable) -> list[DeviceSyntheticEvaluation]:
        candidates = np.asarray(population, dtype=np.float64)
        if candidates.ndim == 1:
            candidates = candidates[None, :]
        if candidates.ndim != 2 or candidates.shape[1] != self.dimension:
            raise ValueError(
                f"Synthetic population shape {candidates.shape} does not match dimension "
                f"{self.dimension}."
            )
        candidates = np.clip(candidates, 0.0, 1.0)
        if self.broker is None:
            result = evaluate_device_resident_curriculum_batch([(self, candidates)])[0]
        else:
            result = self.broker.submit(self, candidates)
        self._evaluation_calls += 1
        should_check = self.require_startup_parity and (
            (not self._parity_verified)
            or (
                self.parity_recheck_interval > 0
                and self._evaluation_calls % self.parity_recheck_interval == 0
            )
        )
        if should_check:
            self._verify_reference_parity(candidates, result)
        return result

    def _verify_reference_parity(
        self,
        population: np.ndarray,
        accelerated: Sequence[DeviceSyntheticEvaluation],
    ) -> None:
        reference = [self.reference.evaluate(x) for x in population]
        if len(reference) != len(accelerated):
            raise RuntimeError(
                "Stage-B synthetic accelerator parity failed: result length differs "
                f"(reference={len(reference)}, accelerated={len(accelerated)})."
            )
        max_error = 0.0
        for ref, got in zip(reference, accelerated):
            numeric_pairs = [
                (float(ref.value), float(got.value)),
                (float(ref.violation), float(got.violation)),
            ]
            ref_components = dict(ref.metadata.get("constraint_components", {}) or {})
            got_components = dict(got.metadata.get("constraint_components", {}) or {})
            if set(ref_components) != set(got_components):
                raise RuntimeError(
                    "Stage-B synthetic accelerator parity failed: constraint component schema differs."
                )
            numeric_pairs.extend(
                (float(ref_components[key]), float(got_components[key]))
                for key in sorted(ref_components)
            )
            for a, b in numeric_pairs:
                if np.isfinite(a) and np.isfinite(b):
                    max_error = max(max_error, abs(a - b))
                elif not (np.isinf(a) and np.isinf(b) and np.sign(a) == np.sign(b)):
                    max_error = float("inf")
            if bool(ref.feasible) != bool(got.feasible):
                raise RuntimeError(
                    "Stage-B synthetic accelerator parity failed: feasibility classification differs."
                )
        self._parity_max_error = float(max_error)
        scale = max(
            1.0,
            max(
                (abs(float(item.value)) for item in reference if np.isfinite(float(item.value))),
                default=1.0,
            ),
        )
        allowed = self.parity_tolerance * scale
        if not np.isfinite(max_error) or max_error > allowed:
            raise RuntimeError(
                "Stage-B synthetic accelerator parity failed before training admission: "
                f"max error {max_error:.6e} exceeds tolerance {allowed:.6e}."
            )
        self._parity_verified = True


def _stack_problem_static(problems: Sequence[DeviceResidentCurriculumProblem]):
    """Return bounded-LRU cached immutable stacked tensors for a microbatch problem set."""
    if not problems:
        raise ValueError("At least one problem is required for static tensor stacking")
    key = (
        str(problems[0].device),
        tuple(str(problem.static_fingerprint) for problem in problems),
    )
    with _STATIC_STACK_CACHE_LOCK:
        cached = _STATIC_STACK_CACHE.get(key)
        if cached is not None:
            _STATIC_STACK_CACHE.move_to_end(key)
            return cached
    device = problems[0].device
    cached = (
        torch.stack([problem.shift for problem in problems], dim=0),
        torch.stack([problem.rotation for problem in problems], dim=0),
        torch.stack([problem.constraint_centres for problem in problems], dim=0),
        torch.stack([problem.constraint_normals for problem in problems], dim=0),
        torch.stack([problem.dimension_mask for problem in problems], dim=0),
        torch.stack([problem.discrete_mask for problem in problems], dim=0),
        torch.stack([problem.level_values for problem in problems], dim=0),
        torch.stack([problem.level_mask for problem in problems], dim=0),
        torch.as_tensor([float(problem.dimension) for problem in problems], dtype=torch.float64, device=device),
        torch.as_tensor([float(problem.narrowness) for problem in problems], dtype=torch.float64, device=device),
        torch.as_tensor([int(problem.stage) for problem in problems], dtype=torch.long, device=device),
    )
    with _STATIC_STACK_CACHE_LOCK:
        _STATIC_STACK_CACHE[key] = cached
        _STATIC_STACK_CACHE.move_to_end(key)
        while len(_STATIC_STACK_CACHE) > _STATIC_STACK_CACHE_MAX_ENTRIES:
            _STATIC_STACK_CACHE.popitem(last=False)
    return cached


def evaluate_device_resident_curriculum_batch(
    requests: Sequence[tuple[DeviceResidentCurriculumProblem, np.ndarray]],
) -> list[list[DeviceSyntheticEvaluation]]:
    """Evaluate heterogeneous curriculum populations in one padded FP64 accelerator batch."""
    if not requests:
        return []
    device = requests[0][0].device
    if any(problem.device != device for problem, _ in requests):
        raise ValueError("All synthetic microbatch requests must target the same device")

    problems = [problem for problem, _ in requests]
    populations = [np.asarray(population, dtype=np.float64) for _, population in requests]
    counts = [int(len(population)) for population in populations]
    total = int(sum(counts))
    if total <= 0:
        return [[] for _ in requests]

    # Candidate tensor is copied once per microbatch, not once per problem/candidate.
    padded = np.zeros((total, _MAX_DIMENSION), dtype=np.float64)
    request_index = np.empty(total, dtype=np.int64)
    cursor = 0
    for index, (problem, population) in enumerate(requests):
        count = len(population)
        padded[cursor : cursor + count, : problem.dimension] = population
        request_index[cursor : cursor + count] = index
        cursor += count

    x = torch.as_tensor(padded, dtype=torch.float64, device=device)
    req = torch.as_tensor(request_index, dtype=torch.long, device=device)
    (
        shifts,
        rotations,
        centres,
        normals,
        dimension_masks,
        discrete_masks,
        level_values,
        level_masks,
        problem_dimensions,
        problem_narrowness,
        problem_stages,
    ) = _stack_problem_static(problems)

    shift = shifts.index_select(0, req)
    rotation = rotations.index_select(0, req)
    mask = dimension_masks.index_select(0, req)
    delta = x - shift
    y = torch.bmm(rotation, delta.unsqueeze(-1)).squeeze(-1)
    dimensions = problem_dimensions.index_select(0, req)
    rastrigin_terms = 25.0 * y.square() - 10.0 * torch.cos(2.0 * torch.pi * 5.0 * y)
    rastrigin = 10.0 * dimensions + torch.sum(rastrigin_terms * mask, dim=1)
    bowl = torch.sum(delta.square() * mask, dim=1)
    objective = 0.35 * rastrigin / torch.clamp(dimensions, min=1.0) + 0.65 * bowl

    centre = centres.index_select(0, req)
    normal = normals.index_select(0, req)
    projections = torch.sum((x.unsqueeze(1) - centre) * normal, dim=2)
    narrowness = problem_narrowness.index_select(0, req)
    limits = torch.stack(
        [narrowness, narrowness * 0.8, narrowness * 1.2, narrowness * 0.9], dim=1
    )
    raw = torch.relu(torch.abs(projections) - limits)

    stages = problem_stages.index_select(0, req)
    raw = torch.where((stages == 0).unsqueeze(1), torch.zeros_like(raw), raw)

    # Exact discrete-lattice distance using the same explicit linspace values created by the
    # NumPy reference problem. Invalid padded levels are masked to +inf before min-reduction.
    level_vals = level_values.index_select(0, req)
    level_valid = level_masks.index_select(0, req)
    discrete = discrete_masks.index_select(0, req)
    distances = torch.abs(x.unsqueeze(-1) - level_vals)
    distances = torch.where(level_valid, distances, torch.full_like(distances, float("inf")))
    nearest = torch.min(distances, dim=2).values
    lattice = torch.relu(nearest - 0.035)
    lattice = torch.where(discrete, lattice, torch.zeros_like(lattice))
    lattice_penalty = torch.sum(lattice * mask, dim=1) / torch.clamp(dimensions, min=1.0)
    raw[:, 1] = raw[:, 1] + torch.where(stages >= 2, lattice_penalty, torch.zeros_like(lattice_penalty))

    violation = torch.sum(raw, dim=1)
    feasible = violation <= 1e-12

    # One compact host materialisation after the complete heterogeneous microbatch.
    objective_cpu = objective.detach().cpu().numpy()
    violation_cpu = violation.detach().cpu().numpy()
    raw_cpu = raw.detach().cpu().numpy()
    feasible_cpu = feasible.detach().cpu().numpy()

    results: list[list[DeviceSyntheticEvaluation]] = [[] for _ in requests]
    cursor = 0
    for request_id, count in enumerate(counts):
        for offset in range(count):
            index = cursor + offset
            components = {
                "bus_voltage": float(raw_cpu[index, 0]),
                "generator_q": float(raw_cpu[index, 1]),
                "generator_p": float(raw_cpu[index, 2]),
                "branch_thermal": float(raw_cpu[index, 3]),
                "power_flow": 0.0,
            }
            results[request_id].append(
                DeviceSyntheticEvaluation(
                    value=float(objective_cpu[index]),
                    feasible=bool(feasible_cpu[index]),
                    violation=float(violation_cpu[index]),
                    metadata={"constraint_components": components},
                )
            )
        cursor += count
    return results


class SyntheticCrossEpisodeBatchBroker:
    """Persistent cross-episode microbatcher for synthetic curriculum evaluation.

    Submission and closure share one lifecycle lock. Every request has a bounded wait and is
    explicitly failed on shutdown or worker failure, eliminating close/enqueue races and indefinite
    scientific waits.
    """

    def __init__(
        self,
        *,
        device: str,
        batch_window_ms: float = 2.0,
        max_candidates: int = 4096,
        request_timeout_s: float = 30.0,
    ) -> None:
        self.device = str(device)
        self.batch_window_ms = max(0.0, float(batch_window_ms))
        self.max_candidates = max(1, int(max_candidates))
        self.request_timeout_s = max(0.1, float(request_timeout_s))
        self._queue: queue.Queue[_BatchRequest | None] = queue.Queue()
        self._state_lock = threading.RLock()
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._pending: dict[int, _BatchRequest] = {}
        self._thread = threading.Thread(target=self._run, name="calo-synthetic-broker", daemon=True)
        self._thread.start()
        self.batch_count = 0
        self.candidate_count = 0
        self.request_count = 0
        self.max_batch_candidates_observed = 0
        self.oversized_request_count = 0
        self.oversized_request_chunks = 0

    def _complete_request(
        self, request: _BatchRequest, *, result=None, error: BaseException | None = None
    ) -> None:
        with self._state_lock:
            if request.event.is_set():
                self._pending.pop(id(request), None)
                return
            if error is not None:
                request.error = error
            else:
                request.result = result
            self._pending.pop(id(request), None)
            request.event.set()

    def _fail_all_pending(self, exc: BaseException) -> None:
        with self._state_lock:
            requests = list(self._pending.values())
            self._pending.clear()
            for request in requests:
                if not request.event.is_set():
                    request.error = exc
                    request.event.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def submit(
        self,
        problem: DeviceResidentCurriculumProblem,
        population: np.ndarray,
    ) -> list[DeviceSyntheticEvaluation]:
        population = np.asarray(population, dtype=np.float64)
        if len(population) > self.max_candidates:
            # A single request must obey the same device-allocation ceiling as merged requests.
            # Split deterministically and preserve exact candidate order in the combined result.
            chunks = [
                population[offset : offset + self.max_candidates]
                for offset in range(0, len(population), self.max_candidates)
            ]
            with self._state_lock:
                self.oversized_request_count += 1
                self.oversized_request_chunks += len(chunks)
            combined: list[DeviceSyntheticEvaluation] = []
            for chunk in chunks:
                combined.extend(self.submit(problem, chunk))
            return combined
        request = _BatchRequest(
            problem=problem,
            population=population,
            event=threading.Event(),
        )
        with self._state_lock:
            if self._fatal_error is not None:
                raise RuntimeError("Synthetic cross-episode batch broker has failed") from self._fatal_error
            if self._closed or not self._thread.is_alive():
                raise RuntimeError("Synthetic cross-episode batch broker is closed")
            self._pending[id(request)] = request
            # Atomic with the closed-state check: close() cannot post its sentinel before this put.
            self._queue.put(request)
        if not request.event.wait(timeout=self.request_timeout_s):
            with self._state_lock:
                self._pending.pop(id(request), None)
                fatal = self._fatal_error
                closed = self._closed
            message = (
                "Synthetic device-resident evaluation timed out after "
                f"{self.request_timeout_s:.3f}s (thread_alive={self._thread.is_alive()}, "
                f"closed={closed}, queue_depth={self._queue.qsize()})."
            )
            if fatal is not None:
                raise RuntimeError(message) from fatal
            raise TimeoutError(message)
        if request.error is not None:
            raise RuntimeError("Synthetic device-resident evaluation failed") from request.error
        if request.result is None:
            raise RuntimeError("Synthetic device-resident broker completed without a result")
        return list(request.result)

    def _run(self) -> None:
        try:
            while True:
                first = self._queue.get()
                if first is None:
                    break
                batch = [first]
                total = len(first.population)
                deadline = time.monotonic() + self.batch_window_ms / 1000.0
                saw_sentinel = False
                while total < self.max_candidates:
                    timeout = max(0.0, deadline - time.monotonic())
                    if timeout <= 0.0:
                        break
                    try:
                        item = self._queue.get(timeout=timeout)
                    except queue.Empty:
                        break
                    if item is None:
                        saw_sentinel = True
                        break
                    if total + len(item.population) > self.max_candidates and batch:
                        # The lifecycle lock guarantees no close sentinel can be overtaken by a new
                        # submit; deferring this already-admitted request is therefore safe.
                        self._queue.put(item)
                        break
                    batch.append(item)
                    total += len(item.population)
                try:
                    results = evaluate_device_resident_curriculum_batch(
                        [(item.problem, item.population) for item in batch]
                    )
                    if len(results) != len(batch):
                        raise RuntimeError(
                            "Synthetic microbatch result count differs from admitted request count"
                        )
                    self.batch_count += 1
                    self.candidate_count += total
                    self.request_count += len(batch)
                    self.max_batch_candidates_observed = max(
                        self.max_batch_candidates_observed, total
                    )
                    for item, result in zip(batch, results):
                        self._complete_request(item, result=result)
                except BaseException as exc:
                    for item in batch:
                        self._complete_request(item, error=exc)
                if saw_sentinel:
                    break
        except BaseException as exc:
            with self._state_lock:
                if self._fatal_error is None:
                    self._fatal_error = exc
            self._fail_all_pending(exc)
        finally:
            with self._state_lock:
                self._closed = True
            self._fail_all_pending(RuntimeError("Synthetic cross-episode batch broker terminated"))

    def metrics(self) -> dict[str, float | int | str]:
        return {
            "device": self.device,
            "batch_count": int(self.batch_count),
            "candidate_count": int(self.candidate_count),
            "request_count": int(self.request_count),
            "max_batch_candidates": int(self.max_batch_candidates_observed),
            "oversized_request_count": int(self.oversized_request_count),
            "oversized_request_chunks": int(self.oversized_request_chunks),
            "static_tensor_cache_entries": int(len(_STATIC_STACK_CACHE)),
            "mean_candidates_per_batch": (
                float(self.candidate_count / self.batch_count) if self.batch_count else 0.0
            ),
        }

    def close(self) -> None:
        shutdown_error = RuntimeError("Synthetic cross-episode batch broker closed during request")
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            for request in list(self._pending.values()):
                if not request.event.is_set():
                    request.error = shutdown_error
                    request.event.set()
            self._pending.clear()
            self._queue.put(None)
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

