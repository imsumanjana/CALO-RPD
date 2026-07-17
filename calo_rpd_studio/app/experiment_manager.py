"""Qt-safe orchestration of sequential or process-parallel scientific experiments."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from copy import deepcopy
import multiprocessing as mp
import os
import queue
import threading
import time

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from calo_rpd_studio.compute.resource_scheduler import (
    ResourceMonitor,
    build_weighted_lane_plan,
    accelerator_admission_allowed,
    backend_allows_accelerators,
    cpu_admission_allowed,
    item_uses_calo_ai,
    prioritized_accelerators,
    weighted_worker_slots,
)
from calo_rpd_studio.compute.xpu_sidecar import execute_xpu_job
from calo_rpd_studio.experiments.calo_ablation import run_ablation
from calo_rpd_studio.experiments.execution_plan import (
    ABLATION_MODE,
    COMPARISON_MODE,
    PlannedItem,
    build_execution_plan,
)
from calo_rpd_studio.experiments.experiment_runner import failed_run_from_exception, run_single
from calo_rpd_studio.experiments.provenance import collect_provenance
from calo_rpd_studio.experiments.seed_manager import RunSeeds, SeedManager
from calo_rpd_studio.results.result_store import ResultStore


def _configure_child_numeric_threads() -> None:
    """Avoid BLAS/PyTorch oversubscription when many optimizer processes run together."""

    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
    except Exception:
        pass


def _config_for_item_device(config, mode: str, item: PlannedItem, compute_device: str):
    local_config = deepcopy(config)
    local_config.runtime_compute_device = str(compute_device)
    parameters = dict(local_config.algorithm_parameters)
    # v3 routes every canonical optimizer through the common torch FP64 scientific backend when
    # an accelerator lane is selected. CALO additionally places policy inference on that device.
    for algorithm_name in tuple(getattr(local_config, "algorithms", ())) + ("CALO", "TLBO"):
        values = dict(parameters.get(algorithm_name, {}))
        values["execution_device"] = str(compute_device)
        if str(getattr(local_config, "scientific_backend", "cpu_reference")) == "torch_fp64":
            values["optimizer_backend"] = "torch"
        if algorithm_name == "CALO":
            values["inference_device"] = str(compute_device)
        parameters[algorithm_name] = values
    local_config.algorithm_parameters = parameters
    return local_config


def _execute_process_job(
    config,
    mode: str,
    item: PlannedItem,
    seeds: RunSeeds,
    progress_queue,
    cancel_event,
    compute_device: str = "cpu",
):
    """Top-level, spawn-safe process worker.

    SQLite and result-array persistence deliberately remain in the parent ExperimentWorker so
    concurrent numerical workers never write to the same database connection.
    """

    _configure_child_numeric_threads()
    local_config = _config_for_item_device(config, mode, item, compute_device)
    last_emit = 0.0
    last_evaluations = -1
    evaluation_span = max(1, int(config.budget.max_evaluations))
    evaluation_step = max(1, evaluation_span // 100)

    def cancelled() -> bool:
        return bool(cancel_event.is_set())

    def emit_progress(payload: dict) -> None:
        nonlocal last_emit, last_evaluations
        now = time.monotonic()
        evaluations = int(payload.get("evaluations", 0))
        # Manager queues are intentionally throttled. This retains a responsive live trace while
        # preventing inter-process telemetry from becoming a new performance bottleneck.
        if (
            evaluations == 0
            or evaluations >= evaluation_span
            or evaluations - last_evaluations >= evaluation_step
            or now - last_emit >= 0.20
        ):
            data = dict(payload)
            data.update(
                {
                    "job_index": item.job_index,
                    "run_index": item.run_index + 1,
                    "algorithm": item.label,
                    "compute_device": str(compute_device),
                }
            )
            progress_queue.put(data)
            last_emit = now
            last_evaluations = evaluations

    try:
        if mode == COMPARISON_MODE:
            completed = run_single(
                local_config,
                item.label,
                item.run_index,
                seeds,
                emit_progress,
                cancelled,
            )
        elif mode == ABLATION_MODE:
            completed = run_ablation(
                local_config,
                item.ablation_spec,
                item.run_index,
                seeds,
                emit_progress,
                cancelled,
            )
        else:
            raise ValueError(f"Unsupported experiment mode: {mode}")
        completed.result.metadata["compute_device_assignment"] = str(compute_device)
        completed.result.metadata["execution_backend"] = str(local_config.execution_backend)
        return "completed", item, completed
    except Exception as exc:
        return "failed", item, failed_run_from_exception(
            item.label,
            item.run_index,
            seeds,
            exc,
        )


class ExperimentWorker(QThread):
    progress = pyqtSignal(dict)
    run_completed = pyqtSignal(str, str, int)
    run_failed = pyqtSignal(str, str, int)
    experiment_created = pyqtSignal(str)
    completed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, state, config, mode: str = COMPARISON_MODE) -> None:
        super().__init__()
        self.state = state
        self.config = config
        self.mode = mode
        self._cancel_event = threading.Event()
        self._process_cancel_event = None

    def cancel(self) -> None:
        self._cancel_event.set()
        event = self._process_cancel_event
        if event is not None:
            try:
                event.set()
            except Exception:
                pass

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _fraction_for_payload(self, payload: dict) -> float:
        if self.config.budget.policy.value == "equal_evaluations":
            evaluations = int(payload.get("evaluations", 0))
            return min(1.0, evaluations / max(int(self.config.budget.max_evaluations), 1))
        return 0.0

    def _emit_progress(
        self,
        payload: dict,
        item: PlannedItem,
        fractions: dict[int, float],
        completed_count: int,
        total_items: int,
        active_count: int = 1,
    ) -> None:
        data = dict(payload)
        fractions[item.job_index] = max(
            fractions.get(item.job_index, 0.0),
            self._fraction_for_payload(data),
        )
        data.update(
            {
                "algorithm": item.label,
                "job_index": item.job_index,
                "run_index": item.run_index + 1,
                "run_position": item.job_index + 1,
                "total_run_items": total_items,
                "completed_items": completed_count,
                "active_items": active_count,
                "overall_percent": int(100 * sum(fractions.values()) / max(total_items, 1)),
            }
        )
        self.progress.emit(data)

    def _persist_completed(self, experiment_id: str, store: ResultStore, item, completed) -> None:
        path = store.save_arrays(completed.result)
        run_id = self.state.database.add_run(experiment_id, completed, str(path))
        self.run_completed.emit(run_id, item.label, item.run_index + 1)

    def _persist_failure(self, experiment_id: str, item, failure) -> None:
        failure_id = self.state.database.add_failure(experiment_id, failure)
        self.run_failed.emit(failure_id, item.label, item.run_index + 1)

    def _run_sequential(self, experiment_id: str, store: ResultStore, plan, seeds) -> bool:
        """Run one job at a time on the highest-priority compatible device.

        Sequential execution still respects the accelerator priority order.  A verified secondary
        XPU runtime is used only when CUDA is unavailable for that job.
        """
        total_items = max(1, len(plan))
        fractions = {item.job_index: 0.0 for item in plan}
        completed_count = 0
        monitor = ResourceMonitor()

        class _ProgressRelay:
            def __init__(self, callback):
                self.callback = callback

            def put(self, payload):
                self.callback(payload)

        class _CancelRelay:
            def __init__(self, callback):
                self.callback = callback

            def is_set(self):
                return bool(self.callback())

        for item in plan:
            if self._cancelled():
                return False
            snapshot = monitor.sample()
            compute_device = "cpu"
            selected_device = None
            if backend_allows_accelerators(self.config.execution_backend) and item_uses_calo_ai(self.mode, item):
                accelerators = prioritized_accelerators(snapshot)
                if accelerators:
                    selected_device = accelerators[0]
                    compute_device = selected_device.device_id

            def emit_progress(payload: dict) -> None:
                data = dict(payload)
                data["compute_device"] = compute_device
                self._emit_progress(
                    data,
                    item,
                    fractions,
                    completed_count,
                    total_items,
                    1,
                )

            try:
                if selected_device is not None and selected_device.backend == "xpu" and selected_device.runtime == "sidecar":
                    outcome, _returned_item, payload = execute_xpu_job(
                        self.config,
                        self.mode,
                        item,
                        seeds[item.run_index],
                        _ProgressRelay(emit_progress),
                        _CancelRelay(self._cancelled),
                        compute_device,
                    )
                    if outcome == "completed":
                        completed = payload
                        self._persist_completed(experiment_id, store, item, completed)
                        phase = "run_completed"
                    else:
                        self._persist_failure(experiment_id, item, payload)
                        phase = "run_failed"
                else:
                    local_config = _config_for_item_device(
                        self.config, self.mode, item, compute_device
                    )
                    if self.mode == COMPARISON_MODE:
                        completed = run_single(
                            local_config,
                            item.label,
                            item.run_index,
                            seeds[item.run_index],
                            emit_progress,
                            self._cancelled,
                        )
                    else:
                        completed = run_ablation(
                            local_config,
                            item.ablation_spec,
                            item.run_index,
                            seeds[item.run_index],
                            emit_progress,
                            self._cancelled,
                        )
                    completed.result.metadata["compute_device_assignment"] = str(compute_device)
                    completed.result.metadata["execution_backend"] = str(local_config.execution_backend)
                    self._persist_completed(experiment_id, store, item, completed)
                    phase = "run_completed"
            except Exception as exc:
                failure = failed_run_from_exception(
                    item.label,
                    item.run_index,
                    seeds[item.run_index],
                    exc,
                )
                self._persist_failure(experiment_id, item, failure)
                phase = "run_failed"

            completed_count += 1
            fractions[item.job_index] = 1.0
            self.progress.emit(
                {
                    "algorithm": item.label,
                    "job_index": item.job_index,
                    "run_index": item.run_index + 1,
                    "overall_percent": int(100 * completed_count / total_items),
                    "run_position": item.job_index + 1,
                    "total_run_items": total_items,
                    "completed_items": completed_count,
                    "active_items": 0,
                    "phase": phase,
                    "compute_device": compute_device,
                }
            )
        return not self._cancelled()

    @staticmethod
    def _drain_progress_queue(progress_queue):
        messages = []
        while True:
            try:
                messages.append(progress_queue.get_nowait())
            except queue.Empty:
                break
            except Exception:
                break
        return messages

    def _run_parallel_weighted(self, experiment_id: str, store: ResultStore, plan, seeds) -> bool:
        """Run a deterministic weighted CUDA/XPU/CPU lane plan.

        Under the v3 PyTorch FP64 backend, the requested shares are applied to the complete
        optimizer plan because all primary algorithms use accelerator-compatible evaluator and
        optimizer kernels. Device thresholds remain safety gates, but they do not dynamically
        rewrite the precomputed lane assignment. The legacy CPU-reference backend remains CPU-only.
        """

        total_items = max(1, len(plan))
        requested_workers = max(1, int(self.config.parallel_workers))
        max_workers = min(requested_workers, total_items)
        fractions = {item.job_index: 0.0 for item in plan}
        completed_count = 0
        monitor = ResourceMonitor()
        initial_snapshot = monitor.sample()
        lane_by_job, allocation = build_weighted_lane_plan(
            plan,
            self.mode,
            cuda_available=bool(initial_snapshot.by_backend("cuda")),
            xpu_available=bool(initial_snapshot.by_backend("xpu")),
            cuda_share=int(self.config.cuda_task_share),
            xpu_share=int(self.config.xpu_task_share),
            cpu_share=int(self.config.cpu_task_share),
        )
        slots = weighted_worker_slots(max_workers, allocation)
        slots["cuda"] = min(slots["cuda"], max(1, int(self.config.gpu_parallel_jobs)))
        slots["xpu"] = min(slots["xpu"], max(1, int(self.config.xpu_parallel_jobs)))

        queues = {
            lane: [item for item in plan if lane_by_job.get(int(item.job_index), "cpu") == lane]
            for lane in ("cuda", "xpu", "cpu")
        }
        self.progress.emit(
            {
                "phase": "allocation_planned",
                "algorithm": "Weighted scheduler",
                "overall_percent": 0,
                "total_run_items": total_items,
                "completed_items": 0,
                "active_items": 0,
                "allocation_requested": allocation.requested_text,
                "allocation_effective": allocation.effective_text,
                "accelerator_eligible_jobs": allocation.accelerator_eligible_jobs,
                "cpu_only_jobs": allocation.cpu_only_jobs,
                "lane_slots": dict(slots),
            }
        )

        context = mp.get_context("spawn")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(key, "1")

        with context.Manager() as manager:
            cancel_event = manager.Event()
            progress_queue = manager.Queue()
            self._process_cancel_event = cancel_event
            xpu_executor = ThreadPoolExecutor(max_workers=max(1, int(self.config.xpu_parallel_jobs)))
            try:
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
                    pending: dict = {}
                    active_lane = {"cuda": 0, "xpu": 0, "cpu": 0}
                    active_by_device: dict[str, int] = {}

                    def submit_item(item: PlannedItem, lane: str, device_id: str, runtime: str = "primary") -> None:
                        if lane == "xpu" and runtime == "sidecar":
                            future = xpu_executor.submit(
                                execute_xpu_job, self.config, self.mode, item, seeds[item.run_index],
                                progress_queue, cancel_event, device_id,
                            )
                        else:
                            future = executor.submit(
                                _execute_process_job, self.config, self.mode, item, seeds[item.run_index],
                                progress_queue, cancel_event, device_id,
                            )
                        pending[future] = (item, lane, device_id)
                        active_lane[lane] += 1
                        if lane != "cpu":
                            active_by_device[device_id] = active_by_device.get(device_id, 0) + 1
                        self.progress.emit(
                            {
                                "algorithm": item.label,
                                "job_index": item.job_index,
                                "run_index": item.run_index + 1,
                                "overall_percent": int(100 * sum(fractions.values()) / total_items),
                                "run_position": item.job_index + 1,
                                "total_run_items": total_items,
                                "completed_items": completed_count,
                                "active_items": len(pending),
                                "phase": "job_started",
                                "compute_device": device_id,
                                "planned_lane": lane,
                            }
                        )

                    def select_device(snapshot, lane: str):
                        devices = list(snapshot.by_backend(lane))
                        devices.sort(key=lambda device: active_by_device.get(device.device_id, 0))
                        for device in devices:
                            if lane == "cuda":
                                target = self.config.gpu_utilization_target
                                memory_limit = self.config.gpu_memory_limit
                                max_jobs = self.config.gpu_parallel_jobs
                            else:
                                target = self.config.xpu_utilization_target
                                memory_limit = self.config.xpu_memory_limit
                                max_jobs = self.config.xpu_parallel_jobs
                            if accelerator_admission_allowed(
                                device, target, memory_limit, active_by_device.get(device.device_id, 0), max_jobs
                            ):
                                return device
                        return None

                    def admit_jobs() -> bool:
                        admitted_any = False
                        # CUDA and XPU lanes are considered first on every admission cycle.
                        for lane in ("cuda", "xpu", "cpu"):
                            while (
                                queues[lane]
                                and len(pending) < max_workers
                                and active_lane[lane] < slots.get(lane, 0)
                                and not self._cancelled()
                            ):
                                snapshot = monitor.sample()
                                if lane == "cpu":
                                    if not cpu_admission_allowed(
                                        snapshot, self.config.cpu_utilization_target, active_lane["cpu"],
                                        self.config.system_memory_limit,
                                    ):
                                        break
                                    item = queues[lane].pop(0)
                                    submit_item(item, lane, "cpu")
                                    admitted_any = True
                                    continue
                                device = select_device(snapshot, lane)
                                if device is None:
                                    break
                                item = queues[lane].pop(0)
                                submit_item(item, lane, device.device_id, device.runtime)
                                admitted_any = True
                        return admitted_any

                    admit_jobs()
                    while pending or any(queues.values()):
                        if self._cancelled():
                            cancel_event.set()

                        for payload in self._drain_progress_queue(progress_queue):
                            job_index = int(payload.get("job_index", -1))
                            item = next((entry for entry in plan if entry.job_index == job_index), None)
                            if item is not None:
                                self._emit_progress(payload, item, fractions, completed_count, total_items, len(pending))

                        if not pending:
                            if self._cancelled():
                                break
                            if not admit_jobs():
                                time.sleep(0.20)
                            continue

                        done, _ = wait(tuple(pending), timeout=0.15, return_when=FIRST_COMPLETED)
                        if not done:
                            admit_jobs()
                            continue

                        for future in done:
                            item, lane, device_id = pending.pop(future)
                            active_lane[lane] = max(0, active_lane[lane] - 1)
                            if lane != "cpu":
                                active_by_device[device_id] = max(0, active_by_device.get(device_id, 0) - 1)
                            try:
                                outcome, returned_item, payload = future.result()
                            except Exception as exc:
                                outcome = "failed"
                                returned_item = item
                                payload = failed_run_from_exception(
                                    item.label, item.run_index, seeds[item.run_index], exc
                                )
                            item = returned_item
                            if outcome == "completed":
                                payload.result.metadata["weighted_lane"] = lane
                                payload.result.metadata["weighted_allocation_requested"] = allocation.requested_text
                                payload.result.metadata["weighted_allocation_effective"] = allocation.effective_text
                                self._persist_completed(experiment_id, store, item, payload)
                                phase = "run_completed"
                            else:
                                self._persist_failure(experiment_id, item, payload)
                                phase = "run_failed"
                            completed_count += 1
                            fractions[item.job_index] = 1.0
                            self.progress.emit(
                                {
                                    "algorithm": item.label,
                                    "job_index": item.job_index,
                                    "run_index": item.run_index + 1,
                                    "overall_percent": int(100 * sum(fractions.values()) / total_items),
                                    "run_position": item.job_index + 1,
                                    "total_run_items": total_items,
                                    "completed_items": completed_count,
                                    "active_items": len(pending),
                                    "phase": phase,
                                    "compute_device": device_id,
                                    "planned_lane": lane,
                                }
                            )
                        if not self._cancelled():
                            admit_jobs()

                    for payload in self._drain_progress_queue(progress_queue):
                        job_index = int(payload.get("job_index", -1))
                        item = next((entry for entry in plan if entry.job_index == job_index), None)
                        if item is not None:
                            self._emit_progress(payload, item, fractions, completed_count, total_items, 0)
            finally:
                cancel_event.set()
                xpu_executor.shutdown(wait=True, cancel_futures=True)
                self._process_cancel_event = None

        return not self._cancelled()

    def _run_parallel(self, experiment_id: str, store: ResultStore, plan, seeds) -> bool:
        """Run independent jobs with accelerator-first heterogeneous admission control.

        Accelerator-capable CALO jobs are considered in strict priority order: CUDA, then Intel
        XPU, then CPU.  CPU-only baselines are admitted only after all compatible accelerator lanes
        have been considered for the current scheduling cycle.  Utilization and memory thresholds
        are soft admission limits; running jobs are never migrated between devices.
        """

        if str(self.config.execution_backend).lower() == "weighted_split":
            return self._run_parallel_weighted(experiment_id, store, plan, seeds)

        total_items = max(1, len(plan))
        requested_workers = max(1, int(self.config.parallel_workers))
        max_workers = min(requested_workers, total_items)
        fractions = {item.job_index: 0.0 for item in plan}
        completed_count = 0
        queued = list(plan)
        monitor = ResourceMonitor()
        accelerators_enabled = backend_allows_accelerators(self.config.execution_backend)

        context = mp.get_context("spawn")
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ.setdefault(key, "1")

        with context.Manager() as manager:
            cancel_event = manager.Event()
            progress_queue = manager.Queue()
            self._process_cancel_event = cancel_event
            xpu_executor = ThreadPoolExecutor(max_workers=max(1, int(self.config.xpu_parallel_jobs)))
            try:
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
                    pending: dict = {}
                    active_cpu_jobs = 0
                    active_by_device: dict[str, int] = {}

                    def submit_item(item: PlannedItem, device: str, runtime: str = "primary") -> None:
                        nonlocal active_cpu_jobs
                        if device.startswith("xpu") and runtime == "sidecar":
                            future = xpu_executor.submit(
                                execute_xpu_job,
                                self.config,
                                self.mode,
                                item,
                                seeds[item.run_index],
                                progress_queue,
                                cancel_event,
                                device,
                            )
                        else:
                            future = executor.submit(
                                _execute_process_job,
                                self.config,
                                self.mode,
                                item,
                                seeds[item.run_index],
                                progress_queue,
                                cancel_event,
                                device,
                            )
                        pending[future] = (item, device)
                        if device == "cpu":
                            active_cpu_jobs += 1
                        else:
                            active_by_device[device] = active_by_device.get(device, 0) + 1
                        self.progress.emit(
                            {
                                "algorithm": item.label,
                                "job_index": item.job_index,
                                "run_index": item.run_index + 1,
                                "overall_percent": int(100 * sum(fractions.values()) / total_items),
                                "run_position": item.job_index + 1,
                                "total_run_items": total_items,
                                "completed_items": completed_count,
                                "active_items": len(pending),
                                "phase": "job_started",
                                "compute_device": device,
                            }
                        )

                    def pop_first(predicate):
                        for index, candidate in enumerate(queued):
                            if predicate(candidate):
                                return queued.pop(index)
                        return None

                    def device_limits(device):
                        if device.backend == "cuda":
                            return (
                                self.config.gpu_utilization_target,
                                self.config.gpu_memory_limit,
                                self.config.gpu_parallel_jobs,
                            )
                        return (
                            self.config.xpu_utilization_target,
                            self.config.xpu_memory_limit,
                            self.config.xpu_parallel_jobs,
                        )

                    def admit_jobs() -> bool:
                        """Admit jobs while respecting CUDA -> XPU -> CPU priority."""
                        admitted_any = False
                        while queued and len(pending) < max_workers and not self._cancelled():
                            snapshot = monitor.sample()
                            admitted = False

                            # 1) Saturate compatible CUDA lanes first, then XPU lanes.  This selection
                            # happens before any new CPU job is considered in this scheduling cycle.
                            if accelerators_enabled:
                                for device in prioritized_accelerators(snapshot):
                                    target, memory_limit, max_jobs = device_limits(device)
                                    if not accelerator_admission_allowed(
                                        device,
                                        target,
                                        memory_limit,
                                        active_by_device.get(device.device_id, 0),
                                        max_jobs,
                                    ):
                                        continue
                                    item = pop_first(
                                        lambda candidate: item_uses_calo_ai(self.mode, candidate)
                                    )
                                    if item is None:
                                        break
                                    submit_item(item, device.device_id, device.runtime)
                                    admitted = True
                                    admitted_any = True
                                    break
                                if admitted:
                                    continue

                            # 2) Admit CPU-only work after accelerator-capable jobs had first refusal.
                            # Host RAM is a safety limit, not a compute tier.
                            if cpu_admission_allowed(
                                snapshot,
                                self.config.cpu_utilization_target,
                                active_cpu_jobs,
                                self.config.system_memory_limit,
                            ):
                                item = pop_first(
                                    lambda candidate: not item_uses_calo_ai(self.mode, candidate)
                                )
                                if item is None:
                                    # All remaining jobs are accelerator-capable but every compatible
                                    # accelerator is currently at a threshold/cap.  CPU fallback is
                                    # allowed only now.
                                    item = queued.pop(0) if queued else None
                                if item is not None:
                                    submit_item(item, "cpu", "primary")
                                    admitted = True
                                    admitted_any = True
                                    continue

                            if not admitted:
                                break
                        return admitted_any

                    admit_jobs()
                    while pending or queued:
                        if self._cancelled():
                            cancel_event.set()

                        for payload in self._drain_progress_queue(progress_queue):
                            job_index = int(payload.get("job_index", -1))
                            item = next((entry for entry in plan if entry.job_index == job_index), None)
                            if item is not None:
                                self._emit_progress(
                                    payload,
                                    item,
                                    fractions,
                                    completed_count,
                                    total_items,
                                    len(pending),
                                )

                        if not pending:
                            if self._cancelled():
                                break
                            if not admit_jobs():
                                time.sleep(0.20)
                            continue

                        done, _ = wait(tuple(pending), timeout=0.15, return_when=FIRST_COMPLETED)
                        if not done:
                            admit_jobs()
                            continue

                        for future in done:
                            item, device = pending.pop(future)
                            if device == "cpu":
                                active_cpu_jobs = max(0, active_cpu_jobs - 1)
                            else:
                                active_by_device[device] = max(
                                    0, active_by_device.get(device, 0) - 1
                                )
                            try:
                                outcome, returned_item, payload = future.result()
                            except Exception as exc:
                                outcome = "failed"
                                returned_item = item
                                payload = failed_run_from_exception(
                                    item.label,
                                    item.run_index,
                                    seeds[item.run_index],
                                    exc,
                                )

                            item = returned_item
                            if outcome == "completed":
                                self._persist_completed(experiment_id, store, item, payload)
                                phase = "run_completed"
                            else:
                                self._persist_failure(experiment_id, item, payload)
                                phase = "run_failed"

                            completed_count += 1
                            fractions[item.job_index] = 1.0
                            self.progress.emit(
                                {
                                    "algorithm": item.label,
                                    "job_index": item.job_index,
                                    "run_index": item.run_index + 1,
                                    "overall_percent": int(100 * sum(fractions.values()) / total_items),
                                    "run_position": item.job_index + 1,
                                    "total_run_items": total_items,
                                    "completed_items": completed_count,
                                    "active_items": len(pending),
                                    "phase": phase,
                                    "compute_device": device,
                                }
                            )

                        if not self._cancelled():
                            admit_jobs()

                    for payload in self._drain_progress_queue(progress_queue):
                        job_index = int(payload.get("job_index", -1))
                        item = next((entry for entry in plan if entry.job_index == job_index), None)
                        if item is not None:
                            self._emit_progress(
                                payload,
                                item,
                                fractions,
                                completed_count,
                                total_items,
                                0,
                            )
            finally:
                cancel_event.set()
                xpu_executor.shutdown(wait=True, cancel_futures=True)
                self._process_cancel_event = None

        return not self._cancelled()

    def run(self) -> None:
        try:
            experiment_id = self.state.database.create_experiment(
                self.config,
                collect_provenance(),
            )
            self.experiment_created.emit(experiment_id)
            store = ResultStore(self.config.output_directory)
            plan = build_execution_plan(self.config, self.mode)
            seeds = SeedManager(self.config.master_seed).generate(self.config.runs)

            if int(self.config.parallel_workers) > 1 and len(plan) > 1:
                finished = self._run_parallel(experiment_id, store, plan, seeds)
            else:
                finished = self._run_sequential(experiment_id, store, plan, seeds)

            if finished:
                self.completed.emit(experiment_id)
            else:
                self.cancelled.emit(experiment_id)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ExperimentManager(QObject):
    progress = pyqtSignal(dict)
    run_completed = pyqtSignal(str, str, int)
    run_failed = pyqtSignal(str, str, int)
    started = pyqtSignal(str)
    completed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)
    busy = pyqtSignal(str)

    def __init__(self, state) -> None:
        super().__init__()
        self.state = state
        self.worker: ExperimentWorker | None = None
        self._busy = False
        self._mode = COMPARISON_MODE
        self.state.task_status.cancel_requested.connect(self.cancel)

    @property
    def running(self) -> bool:
        return self._busy

    def start_comparison(self, config) -> bool:
        return self._start(config, COMPARISON_MODE)

    def start_calo_analysis(self, config) -> bool:
        return self._start(config, ABLATION_MODE)

    def _start(self, config, mode: str) -> bool:
        if self._busy or self.state.task_status.busy:
            self.busy.emit(
                "A scientific task is already running. Wait for it to finish or request safe cancellation before starting another run."
            )
            return False

        self._busy = True
        self._mode = mode
        if mode == COMPARISON_MODE:
            title = "Running primary algorithm comparison"
        else:
            title = "Running CALO ablation study"
        if str(config.execution_backend) == "cpu_only":
            backend = (
                f"CPU-only process pool with {config.parallel_workers} workers"
                if int(config.parallel_workers) > 1
                else "single CPU worker"
            )
        elif str(config.execution_backend) == "weighted_split":
            backend = (
                f"weighted CUDA/XPU/CPU scheduler ({config.cuda_task_share}/{config.xpu_task_share}/{config.cpu_task_share}) "
                f"with up to {config.parallel_workers} concurrent jobs"
            )
        else:
            backend = (
                f"adaptive CPU/GPU scheduler with up to {config.parallel_workers} concurrent jobs"
                if int(config.parallel_workers) > 1
                else "single adaptive CPU/GPU job"
            )
        self.state.task_status.begin(
            title,
            detail=f"Preparing reproducible experiment · {backend}",
            progress=0,
            cancellable=True,
        )
        try:
            worker = ExperimentWorker(self.state, config, mode)
            self.worker = worker
            worker.progress.connect(self._on_progress)
            worker.run_completed.connect(self._on_run_completed)
            worker.run_failed.connect(self._on_run_failed)
            worker.experiment_created.connect(self._created)
            worker.completed.connect(self._completed)
            worker.cancelled.connect(self._cancelled)
            worker.failed.connect(self._failed)
            worker.finished.connect(self._worker_finished)
            worker.start()
        except Exception as exc:
            self._busy = False
            self.worker = None
            self.state.task_status.fail(str(exc))
            raise
        return True

    def _on_progress(self, data: dict) -> None:
        self.progress.emit(data)
        if data.get("phase") == "allocation_planned":
            requested = str(data.get("allocation_requested", ""))
            effective = str(data.get("allocation_effective", ""))
            slots = dict(data.get("lane_slots", {}))
            self.state.task_status.update(
                int(data.get("overall_percent", 0)),
                f"Weighted device plan · requested {requested} · effective {effective} · concurrent slots {slots}",
            )
            return
        algorithm = str(data.get("algorithm", "optimizer"))
        completed_items = data.get("completed_items")
        total = data.get("total_run_items")
        evaluations = data.get("evaluations")
        active = data.get("active_items")
        compute_device = data.get("compute_device")
        detail = algorithm
        if completed_items is not None and total is not None:
            detail += f" · {completed_items}/{total} jobs completed"
        if active is not None and int(self.state.config.parallel_workers) > 1:
            detail += f" · {active} active"
        if compute_device:
            detail += f" · {compute_device}"
        if evaluations is not None:
            detail += f" · {evaluations} evaluations"
        self.state.task_status.update(int(data.get("overall_percent", -1)), detail)

    def _on_run_completed(self, run_id: str, algorithm: str, run_index: int) -> None:
        self.state.runs_changed.emit()
        self.run_completed.emit(run_id, algorithm, run_index)

    def _on_run_failed(self, failure_id: str, algorithm: str, run_index: int) -> None:
        self.state.runs_changed.emit()
        self.run_failed.emit(failure_id, algorithm, run_index)

    def _created(self, experiment_id: str) -> None:
        self.state.current_experiment_id = experiment_id
        self.started.emit(experiment_id)

    def _completed(self, experiment_id: str) -> None:
        self.state.runs_changed.emit()
        self.state.task_status.finish("Experiment completed; results and provenance were stored")
        self.completed.emit(experiment_id)

    def _cancelled(self, experiment_id: str) -> None:
        self.state.runs_changed.emit()
        self.state.task_status.cancelled("Experiment cancelled safely; completed runs were retained")
        self.cancelled.emit(experiment_id)

    def _failed(self, message: str) -> None:
        self.state.task_status.fail(message)
        self.failed.emit(message)

    def _worker_finished(self) -> None:
        worker = self.worker
        self.worker = None
        self._busy = False
        if worker is not None:
            worker.deleteLater()

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
