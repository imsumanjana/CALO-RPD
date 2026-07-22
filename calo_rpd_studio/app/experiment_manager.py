"""Qt-safe orchestration of sequential or process-parallel scientific experiments."""

from __future__ import annotations

import logging

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import asdict
import multiprocessing as mp
import os
import queue
from pathlib import Path
import json
import hashlib
import threading
import time

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from calo_rpd_studio.compute.resource_scheduler import (
    ResourceMonitor,
    build_weighted_lane_plan,
    build_throughput_lane_plan,
    accelerator_admission_allowed,
    backend_allows_accelerators,
    cpu_admission_allowed,
    item_uses_calo_ai,
    prioritized_accelerators,
    weighted_worker_slots,
    throughput_worker_slots,
)
from calo_rpd_studio.compute.xpu_sidecar import execute_xpu_job
from calo_rpd_studio.compute.persistent_accelerator_worker import PersistentAcceleratorPool
from calo_rpd_studio.compute.persistent_accelerator_sidecar import PersistentSidecarPool
from calo_rpd_studio.accelerated.throughput_engine import DeviceCalibration, ThroughputProfile
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
from calo_rpd_studio.portfolio.fingerprint import experiment_fingerprint, run_fingerprint
from calo_rpd_studio.resume.models import ResumeStatus, ResumeTaskType
from calo_rpd_studio.continuation.experiment_evolution import (
    ExperimentEvolutionService,
    ExtensionProtocol,
)
from calo_rpd_studio.continuation.runtime_binding import bind_exact_run_checkpoint


_LOG = logging.getLogger(__name__)

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
        _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)


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
    return bind_exact_run_checkpoint(local_config, item)


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
        if cancel_event.is_set() and int(completed.result.evaluations) < int(
            local_config.budget.max_evaluations
        ):
            return "interrupted", item, completed
        return "completed", item, completed
    except Exception as exc:
        _LOG.exception("Experiment job failed; returning a structured failed-run record")
        return (
            "failed",
            item,
            failed_run_from_exception(
                item.label,
                item.run_index,
                seeds,
                exc,
            ),
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
        self._pause_event = threading.Event()
        self._process_cancel_event = None
        self.campaign_id = str(getattr(config, "resume_campaign_id", "") or "")
        self.experiment_id = ""
        self.resume_task_id = ""
        self._task_by_job: dict[int, dict] = {}
        self._run_fingerprint_by_job: dict[int, str] = {}

    def pause(self) -> None:
        """Stop admitting new jobs and let active jobs finish at a reproducible boundary."""
        self._pause_event.set()

    def cancel(self) -> None:
        """Emergency cancellation; active jobs may restart from their last committed boundary."""
        self._cancel_event.set()
        event = self._process_cancel_event
        if event is not None:
            try:
                event.set()
            except Exception:
                _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _pause_requested(self) -> bool:
        return self._pause_event.is_set()

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

    def _sync_campaign_progress(self, message: str = "") -> None:
        if not self.campaign_id:
            return
        tasks = self.state.database.list_campaign_tasks(self.campaign_id)
        completed = sum(1 for row in tasks if row["status"] in {"completed", "reused"})
        total = len(tasks)
        self.state.database.update_campaign(
            self.campaign_id,
            completed_tasks=completed,
            message=message,
        )
        if self.resume_task_id:
            state = {
                "campaign_id": self.campaign_id,
                "experiment_id": self.experiment_id,
                "mode": self.mode,
            }
            self.state.resume_service.update(
                self.resume_task_id,
                current=completed,
                total=total,
                state=state,
            )

    def _mark_task_started(self, item) -> None:
        row = self._task_by_job.get(int(item.job_index))
        if row:
            self.state.database.update_campaign_task(
                row["id"], status="running", increment_attempts=True
            )
            self.state.database.append_task_event(
                row["id"], "started", {"job_index": item.job_index}
            )

    @staticmethod
    def _sha256_file(path: str) -> str:
        file_path = Path(str(path or ""))
        if not file_path.is_file():
            return ""
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _result_evaluations_from_row(row: dict | None) -> int:
        if not row:
            return 0
        try:
            payload = json.loads(str(row.get("result_json", "{}") or "{}"))
            return int(payload.get("evaluations", 0) or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0

    def _persist_completed(self, experiment_id: str, store: ResultStore, item, completed) -> None:
        path = store.save_arrays(completed.result)
        fingerprint = self._run_fingerprint_by_job.get(int(item.job_index), "")
        key = f"{item.label}:{int(item.run_index)}"
        extension_mode = str(getattr(self.config, "extension_mode", "") or "")
        existing_map = dict(getattr(self.config, "extension_existing_run_ids", {}) or {})
        revision_id = str(getattr(self.config, "experiment_revision_id", "") or "")
        publication_eligible = bool(getattr(self.config, "extension_publication_eligible", True))
        extension_strategy = str(
            getattr(self.config, "extension_execution_strategy", "exact_continue")
            or "exact_continue"
        )

        if extension_mode == "extend_evaluation_horizon" and key in existing_map:
            run_id = str(existing_map[key])
            previous = self.state.database.get_run(run_id)
            previous_evaluations = self._result_evaluations_from_row(previous)
            prior_segments = self.state.database.list_run_segments(run_id)
            previous_revision_id = (
                str(prior_segments[-1].get("metadata", {}).get("revision_id", "") or "")
                if prior_segments
                else ""
            )
            if previous_evaluations > 0:
                self.state.database.snapshot_run_horizon(
                    run_id,
                    evaluation_horizon=previous_evaluations,
                    revision_id=previous_revision_id,
                )
            self.state.database.update_run_result(
                run_id, completed, str(path), scientific_fingerprint=fingerprint
            )
            segment_index = len(prior_segments)
        else:
            run_id = self.state.database.add_run(
                experiment_id, completed, str(path), scientific_fingerprint=fingerprint
            )
            previous_evaluations = 0
            segment_index = 0

        continuation = dict(completed.result.metadata.get("run_continuation", {}) or {})
        checkpoint_path = str(continuation.get("checkpoint_path", "") or "")
        checkpoint_sha = str(continuation.get("checkpoint_sha256", "") or "") or self._sha256_file(
            checkpoint_path
        )
        source_horizon = int(getattr(self.config, "extension_source_horizon", 0) or 0)
        segment_start = (
            0
            if extension_mode == "extend_evaluation_horizon"
            and extension_strategy == "recompute_from_seed"
            else source_horizon
            if extension_mode == "extend_evaluation_horizon" and source_horizon > 0
            else int(previous_evaluations)
        )
        self.state.database.add_run_segment(
            run_id=run_id,
            segment_index=segment_index,
            start_evaluations=segment_start,
            end_evaluations=int(completed.result.evaluations),
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            status="completed",
            publication_eligible=publication_eligible,
            metadata={
                "revision_id": revision_id,
                "extension_mode": extension_mode or "original",
                "execution_strategy": (
                    extension_strategy
                    if extension_mode == "extend_evaluation_horizon"
                    else "original"
                ),
                "trajectory_semantics": (
                    "paired rerun from original seed under new horizon"
                    if extension_mode == "extend_evaluation_horizon"
                    and extension_strategy == "recompute_from_seed"
                    else "exact checkpoint continuation"
                    if extension_mode == "extend_evaluation_horizon"
                    else "original run"
                ),
                "prior_current_horizon": int(previous_evaluations),
                "source_horizon": int(source_horizon),
                "algorithm": item.label,
                "run_index": int(item.run_index),
            },
        )
        row = self._task_by_job.get(int(item.job_index))
        if row:
            self.state.database.update_campaign_task(
                row["id"],
                status="completed",
                run_id=run_id,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha,
            )
            self.state.database.append_task_event(
                row["id"],
                "completed",
                {
                    "run_id": run_id,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_sha256": checkpoint_sha,
                },
            )
        self._sync_campaign_progress(f"Completed {item.label} run {item.run_index + 1}")
        self.run_completed.emit(run_id, item.label, item.run_index + 1)

    def _persist_interrupted(self, item, completed) -> None:
        """Commit only the exact resume boundary for an interrupted active job.

        Partial numerical results are not promoted to completed-run evidence. CALO's terminal
        checkpoint can resume exactly; algorithms without such a checkpoint restart from their
        original paired seed when the campaign resumes.
        """
        row = self._task_by_job.get(int(item.job_index))
        if not row:
            return
        continuation = dict(completed.result.metadata.get("run_continuation", {}) or {})
        checkpoint_path = str(continuation.get("checkpoint_path", "") or "")
        checkpoint_sha = str(continuation.get("checkpoint_sha256", "") or "") or self._sha256_file(
            checkpoint_path
        )
        self.state.database.update_campaign_task(
            row["id"],
            status="interrupted",
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
        )
        self.state.database.append_task_event(
            row["id"],
            "interrupted",
            {
                "evaluations": int(completed.result.evaluations),
                "checkpoint_path": checkpoint_path,
                "checkpoint_sha256": checkpoint_sha,
                "exact_resume_available": bool(checkpoint_path),
            },
        )

    def _persist_failure(self, experiment_id: str, item, failure) -> None:
        failure_id = self.state.database.add_failure(experiment_id, failure)
        row = self._task_by_job.get(int(item.job_index))
        if row:
            self.state.database.update_campaign_task(
                row["id"], status="failed", failure_id=failure_id
            )
            self.state.database.append_task_event(
                row["id"], "failed", {"failure_id": failure_id, "message": failure.message}
            )
        self._sync_campaign_progress(f"Failed {item.label} run {item.run_index + 1}")
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
            if self._cancelled() or self._pause_requested():
                return False
            self._mark_task_started(item)
            snapshot = monitor.sample()
            compute_device = "cpu"
            selected_device = None
            if backend_allows_accelerators(self.config.execution_backend) and item_uses_calo_ai(
                self.mode, item
            ):
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
                if (
                    selected_device is not None
                    and selected_device.backend == "xpu"
                    and selected_device.runtime == "sidecar"
                ):
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
                        if self._cancelled() and int(completed.result.evaluations) < int(
                            self.config.budget.max_evaluations
                        ):
                            self._persist_interrupted(item, completed)
                            phase = "run_interrupted"
                        else:
                            self._persist_completed(experiment_id, store, item, completed)
                            phase = "run_completed"
                    elif outcome == "interrupted":
                        self._persist_interrupted(item, payload)
                        phase = "run_interrupted"
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
                    completed.result.metadata["execution_backend"] = str(
                        local_config.execution_backend
                    )
                    if self._cancelled() and int(completed.result.evaluations) < int(
                        local_config.budget.max_evaluations
                    ):
                        self._persist_interrupted(item, completed)
                        phase = "run_interrupted"
                    else:
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
        return not self._cancelled() and not self._pause_requested()

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
            xpu_executor = ThreadPoolExecutor(
                max_workers=max(1, int(self.config.xpu_parallel_jobs))
            )
            try:
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
                    pending: dict = {}
                    active_lane = {"cuda": 0, "xpu": 0, "cpu": 0}
                    active_by_device: dict[str, int] = {}

                    def submit_item(
                        item: PlannedItem, lane: str, device_id: str, runtime: str = "primary"
                    ) -> None:
                        self._mark_task_started(item)
                        if lane == "xpu" and runtime == "sidecar":
                            future = xpu_executor.submit(
                                execute_xpu_job,
                                self.config,
                                self.mode,
                                item,
                                seeds[item.run_index],
                                progress_queue,
                                cancel_event,
                                device_id,
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
                                device_id,
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
                                device,
                                target,
                                memory_limit,
                                active_by_device.get(device.device_id, 0),
                                max_jobs,
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
                                and not self._pause_requested()
                            ):
                                snapshot = monitor.sample()
                                if lane == "cpu":
                                    if not cpu_admission_allowed(
                                        snapshot,
                                        self.config.cpu_utilization_target,
                                        active_lane["cpu"],
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
                            item = next(
                                (entry for entry in plan if entry.job_index == job_index), None
                            )
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
                            if self._cancelled() or self._pause_requested():
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
                                active_by_device[device_id] = max(
                                    0, active_by_device.get(device_id, 0) - 1
                                )
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
                                payload.result.metadata["weighted_allocation_requested"] = (
                                    allocation.requested_text
                                )
                                payload.result.metadata["weighted_allocation_effective"] = (
                                    allocation.effective_text
                                )
                                self._persist_completed(experiment_id, store, item, payload)
                                phase = "run_completed"
                            elif outcome == "interrupted":
                                self._persist_interrupted(item, payload)
                                phase = "run_interrupted"
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
                                    "overall_percent": int(
                                        100 * sum(fractions.values()) / total_items
                                    ),
                                    "run_position": item.job_index + 1,
                                    "total_run_items": total_items,
                                    "completed_items": completed_count,
                                    "active_items": len(pending),
                                    "phase": phase,
                                    "compute_device": device_id,
                                    "planned_lane": lane,
                                }
                            )
                        if not self._cancelled() and not self._pause_requested():
                            admit_jobs()

                    for payload in self._drain_progress_queue(progress_queue):
                        job_index = int(payload.get("job_index", -1))
                        item = next((entry for entry in plan if entry.job_index == job_index), None)
                        if item is not None:
                            self._emit_progress(
                                payload, item, fractions, completed_count, total_items, 0
                            )
            finally:
                cancel_event.set()
                xpu_executor.shutdown(wait=True, cancel_futures=True)
                self._process_cancel_event = None

        return not self._cancelled() and not self._pause_requested()

    def _run_parallel_throughput(self, experiment_id: str, store: ResultStore, plan, seeds) -> bool:
        """Run the v3.1 persistent batched-throughput engine.

        One long-lived process owns each selected compute device.  Independent optimizer runs are
        executed as threads inside that process; compatible population requests are combined by a
        cross-run batch broker.  A short unbudgeted calibration selects the best stable microbatch
        and allocates complete jobs in proportion to measured candidate-evaluation throughput.
        """

        total_items = max(1, len(plan))
        fractions = {item.job_index: 0.0 for item in plan}
        completed_count = 0
        monitor = ResourceMonitor()
        snapshot = monitor.sample()
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
            pools = {}
            try:
                backend = str(self.config.execution_backend).lower()
                cuda_device = next(iter(snapshot.by_backend("cuda")), None)
                if backend == "cuda_only" and cuda_device is None:
                    raise RuntimeError(
                        "CUDA-only execution was requested, but no verified NVIDIA CUDA runtime is available."
                    )

                # GPU-maximum mode creates only one numerical lane: CUDA when available,
                # otherwise XPU, otherwise CPU. The host remains responsible for GUI,
                # orchestration, persistence, and independent validation only.
                gpu_max_lane = (
                    "cuda"
                    if cuda_device is not None
                    else (
                        "xpu" if next(iter(snapshot.by_backend("xpu")), None) is not None else "cpu"
                    )
                )
                if backend not in {"cuda_only", "gpu_preferred"} or (
                    backend == "gpu_preferred" and gpu_max_lane == "cpu"
                ):
                    pools["cpu"] = PersistentAcceleratorPool(
                        "cpu",
                        slots=max(1, int(self.config.parallel_workers)),
                        progress_queue=progress_queue,
                        cancel_event=cancel_event,
                        batch_window_ms=float(self.config.cross_run_batch_window_ms),
                        max_cross_run_batch=int(self.config.max_cross_run_batch),
                        cross_run_batching=bool(self.config.cross_run_batching),
                        context=context,
                    )

                if cuda_device is not None and (
                    backend != "gpu_preferred" or gpu_max_lane == "cuda"
                ):
                    pools["cuda"] = PersistentAcceleratorPool(
                        cuda_device.device_id,
                        slots=max(1, int(self.config.gpu_parallel_jobs)),
                        progress_queue=progress_queue,
                        cancel_event=cancel_event,
                        batch_window_ms=float(self.config.cross_run_batch_window_ms),
                        max_cross_run_batch=int(self.config.max_cross_run_batch),
                        cross_run_batching=bool(self.config.cross_run_batching),
                        context=context,
                    )

                xpu_device = next(iter(snapshot.by_backend("xpu")), None)
                if (
                    xpu_device is not None
                    and backend != "cuda_only"
                    and (backend != "gpu_preferred" or gpu_max_lane == "xpu")
                ):
                    if xpu_device.runtime == "sidecar":
                        from calo_rpd_studio.compute.resource_scheduler import (
                            configured_xpu_interpreter,
                        )

                        interpreter = configured_xpu_interpreter()
                        if interpreter:
                            pools["xpu"] = PersistentSidecarPool(
                                interpreter,
                                xpu_device.device_id,
                                slots=max(1, int(self.config.xpu_parallel_jobs)),
                                progress_queue=progress_queue,
                                batch_window_ms=float(self.config.cross_run_batch_window_ms),
                                max_cross_run_batch=int(self.config.max_cross_run_batch),
                                cross_run_batching=bool(self.config.cross_run_batching),
                            )
                    else:
                        pools["xpu"] = PersistentAcceleratorPool(
                            xpu_device.device_id,
                            slots=max(1, int(self.config.xpu_parallel_jobs)),
                            progress_queue=progress_queue,
                            cancel_event=cancel_event,
                            batch_window_ms=float(self.config.cross_run_batch_window_ms),
                            max_cross_run_batch=int(self.config.max_cross_run_batch),
                            cross_run_batching=bool(self.config.cross_run_batching),
                            context=context,
                        )

                # Automatic microbatch calibration is performed inside each persistent runtime, so
                # CUDA/XPU contexts and invariant tensors remain warm for the campaign itself.
                calibration_records: dict[str, DeviceCalibration] = {}
                batch_sizes = tuple(int(value) for value in self.config.calibration_batch_sizes)
                calibration_pending = set()
                if bool(self.config.automatic_batch_calibration):
                    for lane, pool in pools.items():
                        request_id = f"cal-{lane}-{time.time_ns()}"
                        calibration_pending.add(request_id)
                        pool.calibrate(
                            request_id,
                            self.config,
                            seeds[0].scenario_seed,
                            batch_sizes=batch_sizes,
                            repetitions=int(self.config.calibration_repetitions),
                        )
                    calibration_deadline = time.monotonic() + 600.0
                    while calibration_pending and time.monotonic() < calibration_deadline:
                        if self._cancelled():
                            cancel_event.set()
                            break
                        for lane, pool in pools.items():
                            for message in pool.poll():
                                kind = str(message.get("kind", ""))
                                if kind == "calibration":
                                    request_id = str(message.get("request_id", ""))
                                    calibration_pending.discard(request_id)
                                    record = message.get("record")
                                    if isinstance(record, DeviceCalibration):
                                        calibration_records[lane] = record
                                elif kind == "calibration_error":
                                    calibration_pending.discard(str(message.get("request_id", "")))
                        percent = int(
                            100 * (len(pools) - len(calibration_pending)) / max(len(pools), 1)
                        )
                        self.progress.emit(
                            {
                                "phase": "throughput_calibration",
                                "algorithm": "Throughput calibration",
                                "overall_percent": 0,
                                "calibration_percent": percent,
                                "total_run_items": total_items,
                                "completed_items": 0,
                                "active_items": 0,
                            }
                        )
                        time.sleep(0.05)

                # Load an existing profile only for lanes not successfully calibrated now.
                profile_path = str(self.config.throughput_profile_path)
                try:
                    previous = ThroughputProfile.load(profile_path)
                except Exception:
                    previous = None
                if previous is not None and previous.case_name == self.config.case_name:
                    for lane in pools:
                        if lane not in calibration_records:
                            candidates = [
                                record
                                for key, record in previous.devices.items()
                                if key.startswith(lane)
                            ]
                            if candidates:
                                calibration_records[lane] = max(
                                    candidates, key=lambda record: record.evaluations_per_second
                                )

                # A conservative fallback preserves execution when a calibration cannot run.
                for lane, pool in pools.items():
                    if lane not in calibration_records:
                        calibration_records[lane] = DeviceCalibration(
                            device=getattr(pool, "device", lane),
                            device_name=getattr(pool, "device", lane),
                            batch_size=int(self.config.tensor_batch_size),
                            evaluations_per_second=1.0 if lane == "cpu" else 0.0,
                            latency_seconds=float("inf"),
                            candidate_count=0,
                            repetitions=0,
                            successful=False,
                            note="Calibration unavailable; conservative fallback",
                        )

                profile = ThroughputProfile(
                    case_name=self.config.case_name,
                    scenario_count=max(1, int(getattr(self.config.scenarios, "count", 1)))
                    if self.config.scenarios.mode != "deterministic"
                    else 1,
                    dimension=0,
                    created_at=time.time(),
                    devices={record.device: record for record in calibration_records.values()},
                )
                try:
                    profile.save(profile_path)
                except Exception:
                    _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)

                lane_throughputs = {
                    lane: max(0.0, float(record.evaluations_per_second))
                    for lane, record in calibration_records.items()
                }
                backend = str(self.config.execution_backend).lower()
                if backend in {"cuda_priority", "cuda_only", "gpu_preferred"}:
                    if backend == "gpu_preferred":
                        effective_shares = {
                            "cuda": (100, 0, 0),
                            "xpu": (0, 100, 0),
                            "cpu": (0, 0, 100),
                        }[gpu_max_lane]
                    else:
                        effective_shares = (
                            int(self.config.cuda_task_share),
                            int(self.config.xpu_task_share),
                            int(self.config.cpu_task_share),
                        )
                    lane_by_job, allocation = build_weighted_lane_plan(
                        plan,
                        self.mode,
                        cuda_available="cuda" in pools,
                        xpu_available="xpu" in pools,
                        cuda_share=effective_shares[0],
                        xpu_share=effective_shares[1],
                        cpu_share=effective_shares[2],
                    )
                else:
                    lane_by_job, allocation = build_throughput_lane_plan(
                        plan,
                        self.mode,
                        lane_throughputs=lane_throughputs,
                        cuda_available="cuda" in pools,
                        xpu_available="xpu" in pools,
                    )
                slots = (
                    weighted_worker_slots(max(1, int(self.config.parallel_workers)), allocation)
                    if backend in {"cuda_priority", "cuda_only", "gpu_preferred"}
                    else throughput_worker_slots(
                        max(1, int(self.config.parallel_workers)), allocation
                    )
                )
                if "cuda" in pools:
                    slots["cuda"] = min(
                        slots["cuda"], int(self.config.gpu_parallel_jobs), pools["cuda"].slots
                    )
                if "xpu" in pools:
                    slots["xpu"] = min(
                        slots["xpu"], int(self.config.xpu_parallel_jobs), pools["xpu"].slots
                    )
                if "cpu" in pools:
                    slots["cpu"] = min(slots["cpu"], pools["cpu"].slots)
                else:
                    slots["cpu"] = 0
                for lane in ("cuda", "xpu", "cpu"):
                    jobs_for_lane = sum(1 for value in lane_by_job.values() if value == lane)
                    if jobs_for_lane > 0 and slots.get(lane, 0) == 0 and lane in pools:
                        slots[lane] = 1

                queues = {
                    lane: [
                        item for item in plan if lane_by_job.get(int(item.job_index), "cpu") == lane
                    ]
                    for lane in ("cuda", "xpu", "cpu")
                }
                self.progress.emit(
                    {
                        "phase": "throughput_allocation_planned",
                        "algorithm": "Batched Throughput Engine",
                        "overall_percent": 0,
                        "total_run_items": total_items,
                        "completed_items": 0,
                        "active_items": 0,
                        "allocation_effective": allocation.effective_text,
                        "measured_throughput": getattr(
                            allocation, "throughput_text", "fixed CUDA/XPU/CPU share"
                        ),
                        "lane_slots": dict(slots),
                        "calibrated_batch_sizes": {
                            lane: record.batch_size for lane, record in calibration_records.items()
                        },
                    }
                )

                active_by_lane = {lane: 0 for lane in ("cuda", "xpu", "cpu")}
                item_by_job_id = {}

                def active_total() -> int:
                    return sum(active_by_lane.values())

                def submit_available() -> bool:
                    admitted = False
                    current_snapshot = monitor.sample()
                    # CUDA-priority work stealing is limited to unstarted jobs. It preserves each
                    # run's seed and numerical protocol while preventing a fast NVIDIA lane from
                    # waiting behind slower XPU/CPU queues.
                    if (
                        str(self.config.execution_backend).lower() == "cuda_priority"
                        and bool(getattr(self.config, "cuda_priority_work_stealing", True))
                        and "cuda" in pools
                        and active_by_lane["cuda"] < slots.get("cuda", 0)
                    ):
                        while active_by_lane["cuda"] + len(queues["cuda"]) < slots.get("cuda", 0):
                            donor = "xpu" if queues["xpu"] else ("cpu" if queues["cpu"] else None)
                            if donor is None:
                                break
                            queues["cuda"].append(queues[donor].pop(0))
                    for lane in ("cuda", "xpu", "cpu"):
                        pool = pools.get(lane)
                        if pool is None:
                            continue
                        while (
                            queues[lane]
                            and active_by_lane[lane] < slots.get(lane, 0)
                            and pool.available_slots > 0
                            and not self._cancelled()
                            and not self._pause_requested()
                        ):
                            if lane == "cpu":
                                if (
                                    not cpu_admission_allowed(
                                        current_snapshot,
                                        self.config.cpu_utilization_target,
                                        active_by_lane["cpu"],
                                        self.config.system_memory_limit,
                                    )
                                    and active_total() > 0
                                ):
                                    break
                            else:
                                device = next(iter(current_snapshot.by_backend(lane)), None)
                                if device is not None:
                                    target = (
                                        self.config.gpu_utilization_target
                                        if lane == "cuda"
                                        else self.config.xpu_utilization_target
                                    )
                                    memory_limit = (
                                        self.config.gpu_memory_limit
                                        if lane == "cuda"
                                        else self.config.xpu_memory_limit
                                    )
                                    cap = (
                                        self.config.gpu_parallel_jobs
                                        if lane == "cuda"
                                        else self.config.xpu_parallel_jobs
                                    )
                                    if (
                                        not accelerator_admission_allowed(
                                            device, target, memory_limit, active_by_lane[lane], cap
                                        )
                                        and active_total() > 0
                                    ):
                                        break
                            item = queues[lane].pop(0)
                            job_id = f"job-{item.job_index}"
                            local = deepcopy(self.config)
                            local.tensor_batch_size = int(calibration_records[lane].batch_size)
                            self._mark_task_started(item)
                            pool.submit(job_id, local, self.mode, item, seeds[item.run_index])
                            active_by_lane[lane] += 1
                            item_by_job_id[job_id] = (item, lane)
                            admitted = True
                            self.progress.emit(
                                {
                                    "phase": "job_started",
                                    "algorithm": item.label,
                                    "job_index": item.job_index,
                                    "run_index": item.run_index + 1,
                                    "overall_percent": int(
                                        100 * sum(fractions.values()) / total_items
                                    ),
                                    "run_position": item.job_index + 1,
                                    "total_run_items": total_items,
                                    "completed_items": completed_count,
                                    "active_items": active_total(),
                                    "compute_device": getattr(pool, "device", lane),
                                    "planned_lane": lane,
                                    "calibrated_batch_size": int(
                                        calibration_records[lane].batch_size
                                    ),
                                }
                            )
                        current_snapshot = monitor.sample()
                    return admitted

                submit_available()
                while any(queues.values()) or active_total() > 0:
                    if self._cancelled():
                        cancel_event.set()
                        for pool in pools.values():
                            cancel = getattr(pool, "cancel", None)
                            if callable(cancel):
                                cancel()
                    if self._pause_requested() and active_total() == 0:
                        break

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
                                active_total(),
                            )

                    handled = False
                    for lane, pool in pools.items():
                        for message in pool.poll():
                            kind = str(message.get("kind", ""))
                            if kind in {"calibration", "calibration_error"}:
                                continue
                            job_id = str(message.get("job_id", ""))
                            item_lane = item_by_job_id.pop(job_id, None)
                            if item_lane is None:
                                continue
                            item, expected_lane = item_lane
                            active_by_lane[expected_lane] = max(
                                0, active_by_lane[expected_lane] - 1
                            )
                            if kind in {"completed", "interrupted"}:
                                completed = message["payload"]
                                completed.result.metadata.update(
                                    {
                                        "throughput_engine_version": "3.4",
                                        "device_resident_execution": bool(
                                            getattr(self.config, "device_resident_execution", True)
                                        ),
                                        "cuda_priority_work_stealing": bool(
                                            getattr(
                                                self.config, "cuda_priority_work_stealing", True
                                            )
                                        ),
                                        "throughput_lane": expected_lane,
                                        "throughput_calibration": asdict(
                                            calibration_records[expected_lane]
                                        ),
                                        "throughput_allocation": allocation.effective_text,
                                        "measured_lane_throughputs": dict(lane_throughputs),
                                        "numerical_device_residency": (
                                            "100% candidate evaluation on CUDA"
                                            if expected_lane == "cuda"
                                            else "100% candidate evaluation on XPU"
                                            if expected_lane == "xpu"
                                            else "CPU fallback because no compatible accelerator was available"
                                        ),
                                    }
                                )
                                if kind == "interrupted":
                                    self._persist_interrupted(item, completed)
                                    phase = "run_interrupted"
                                else:
                                    self._persist_completed(experiment_id, store, item, completed)
                                    phase = "run_completed"
                            else:
                                payload = message.get("payload")
                                if payload is None:
                                    payload = failed_run_from_exception(
                                        item.label,
                                        item.run_index,
                                        seeds[item.run_index],
                                        RuntimeError(
                                            str(message.get("message", "Persistent worker failure"))
                                        ),
                                    )
                                self._persist_failure(experiment_id, item, payload)
                                phase = "run_failed"
                            completed_count += 1
                            fractions[item.job_index] = 1.0
                            handled = True
                            self.progress.emit(
                                {
                                    "algorithm": item.label,
                                    "job_index": item.job_index,
                                    "run_index": item.run_index + 1,
                                    "overall_percent": int(
                                        100 * sum(fractions.values()) / total_items
                                    ),
                                    "run_position": item.job_index + 1,
                                    "total_run_items": total_items,
                                    "completed_items": completed_count,
                                    "active_items": active_total(),
                                    "phase": phase,
                                    "compute_device": getattr(pool, "device", expected_lane),
                                    "planned_lane": expected_lane,
                                }
                            )
                    if not self._cancelled() and not self._pause_requested():
                        submit_available()
                    if not handled:
                        time.sleep(0.05)

                for payload in self._drain_progress_queue(progress_queue):
                    job_index = int(payload.get("job_index", -1))
                    item = next((entry for entry in plan if entry.job_index == job_index), None)
                    if item is not None:
                        self._emit_progress(
                            payload, item, fractions, completed_count, total_items, 0
                        )
            finally:
                cancel_event.set()
                for pool in pools.values():
                    try:
                        pool.close()
                    except Exception:
                        _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)
                self._process_cancel_event = None

        return not self._cancelled() and not self._pause_requested()

    def _run_parallel(self, experiment_id: str, store: ResultStore, plan, seeds) -> bool:
        """Run independent jobs with accelerator-first heterogeneous admission control.

        Accelerator-capable CALO jobs are considered in strict priority order: CUDA, then Intel
        XPU, then CPU.  CPU-only baselines are admitted only after all compatible accelerator lanes
        have been considered for the current scheduling cycle.  Utilization and memory thresholds
        are soft admission limits; running jobs are never migrated between devices.
        """

        backend = str(self.config.execution_backend).lower()
        if backend in {"throughput_auto", "cuda_priority", "cuda_only", "gpu_preferred"}:
            if bool(self.config.throughput_engine_enabled) and bool(
                self.config.persistent_accelerator_workers
            ):
                return self._run_parallel_throughput(experiment_id, store, plan, seeds)
            return self._run_parallel_weighted(experiment_id, store, plan, seeds)
        if backend == "weighted_split":
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
            xpu_executor = ThreadPoolExecutor(
                max_workers=max(1, int(self.config.xpu_parallel_jobs))
            )
            try:
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
                    pending: dict = {}
                    active_cpu_jobs = 0
                    active_by_device: dict[str, int] = {}

                    def submit_item(
                        item: PlannedItem, device: str, runtime: str = "primary"
                    ) -> None:
                        self._mark_task_started(item)
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
                        while (
                            queued
                            and len(pending) < max_workers
                            and not self._cancelled()
                            and not self._pause_requested()
                        ):
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
                            item = next(
                                (entry for entry in plan if entry.job_index == job_index), None
                            )
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
                            if self._cancelled() or self._pause_requested():
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
                            elif outcome == "interrupted":
                                self._persist_interrupted(item, payload)
                                phase = "run_interrupted"
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
                                    "overall_percent": int(
                                        100 * sum(fractions.values()) / total_items
                                    ),
                                    "run_position": item.job_index + 1,
                                    "total_run_items": total_items,
                                    "completed_items": completed_count,
                                    "active_items": len(pending),
                                    "phase": phase,
                                    "compute_device": device,
                                }
                            )

                        if not self._cancelled() and not self._pause_requested():
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

        return not self._cancelled() and not self._pause_requested()

    def _prepare_campaign(self, full_plan, seeds):
        """Create/reopen a campaign, including scientifically explicit v5 extensions."""
        database = self.state.database
        if self.campaign_id:
            campaign = database.get_campaign(self.campaign_id)
            if campaign is None:
                raise KeyError(f"Unknown resume campaign: {self.campaign_id}")
            self.experiment_id = str(campaign["experiment_id"])
            self.resume_task_id = self.campaign_id
            rows = database.list_campaign_tasks(self.campaign_id)
            self._task_by_job = {int(row["job_index"]): row for row in rows}
            pending_statuses = {
                "planned",
                "queued",
                "running",
                "pausing",
                "paused",
                "interrupted",
                "failed",
            }
            plan = [
                item
                for item in full_plan
                if self._task_by_job.get(int(item.job_index), {}).get("status") in pending_statuses
            ]
            resume_map = dict(getattr(self.config, "extension_checkpoint_paths", {}) or {})
            for item in plan:
                row = self._task_by_job[int(item.job_index)]
                self._run_fingerprint_by_job[int(item.job_index)] = str(row["fingerprint"])
                checkpoint_path = str(row.get("checkpoint_path", "") or "")
                checkpoint_sha = str(row.get("checkpoint_sha256", "") or "")
                if checkpoint_path and str(item.label) == "CALO":
                    actual_sha = self._sha256_file(checkpoint_path)
                    if checkpoint_sha and actual_sha.lower() != checkpoint_sha.lower():
                        raise RuntimeError(
                            f"Stored CALO resume checkpoint checksum mismatch for run {item.run_index + 1}"
                        )
                    resume_map[f"{item.label}:{int(item.run_index)}"] = checkpoint_path
                database.update_campaign_task(row["id"], status="planned")
            self.config.extension_checkpoint_paths = resume_map
            database.update_campaign(self.campaign_id, status="running", message="Campaign resumed")
            self.state.resume_service.update(
                self.resume_task_id,
                status=ResumeStatus.RUNNING,
                state={
                    "campaign_id": self.campaign_id,
                    "experiment_id": self.experiment_id,
                    "mode": self.mode,
                },
                resumable=True,
            )
            self._sync_campaign_progress("Campaign resumed")
            return plan

        extension_experiment_id = str(getattr(self.config, "extension_experiment_id", "") or "")
        extension_mode = str(getattr(self.config, "extension_mode", "") or "")
        is_extension = bool(extension_experiment_id and extension_mode)
        if is_extension:
            experiment = database.get_experiment(extension_experiment_id)
            if experiment is None:
                raise KeyError(f"Unknown experiment extension target: {extension_experiment_id}")
            self.experiment_id = extension_experiment_id
        else:
            scientific_fp = experiment_fingerprint(self.config)
            self.experiment_id = database.create_experiment(
                self.config,
                collect_provenance(),
                scientific_fingerprint=scientific_fp,
                portfolio_id=str(getattr(self.config, "portfolio_id", "")),
                campaign_status="running",
            )

        # Every experiment receives a stable run-checkpoint root. Exact resumable algorithms write
        # atomic checkpoints there; the path is operational and excluded from scientific fingerprints.
        checkpoint_root = (
            Path(self.config.output_directory) / "checkpoints" / "runs" / self.experiment_id
        )
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.config.run_checkpoint_root = str(checkpoint_root)

        if "CALO" in self.config.algorithms:
            calo_parameters = dict(self.config.algorithm_parameters.get("CALO", {}))
            policy_id = str(calo_parameters.get("policy_id", "") or "")
            policy_checkpoint = str(calo_parameters.get("policy_checkpoint", "") or "")
            if bool(calo_parameters.get("use_ai", True)):
                if not bool(calo_parameters.get("strict_policy_binding", False)):
                    raise ValueError(
                        "CALO policy-assisted execution is fail-closed and requires an explicitly "
                        "activated immutable policy binding. Reapply CALO Intelligence before starting the experiment."
                    )
                if not policy_id or not policy_checkpoint:
                    raise ValueError(
                        "CALO policy binding is incomplete. Train or import a compatible policy, "
                        "qualify/activate it, then reapply CALO Intelligence before starting the experiment."
                    )
                policy = self.state.policy_registry.get(policy_id)
                if not is_extension and not policy.active:
                    raise ValueError(
                        "New policy-assisted CALO experiments must use the explicitly active policy. "
                        "Activate the intended policy in CALO Intelligence and reapply the configuration."
                    )
                if not is_extension and not policy.runtime_compatible:
                    raise ValueError(
                        "The selected CALO policy is not compatible with the current runtime schema. "
                        "Train/import and activate a compatible policy before starting the experiment."
                    )
                if not bool(
                    calo_parameters.get("allow_unqualified_policy", False)
                ) and policy.qualification_status not in {"qualified", "legacy_qualified"}:
                    raise ValueError(
                        f"CALO policy {policy.name!r} is not qualified for strict evaluation; "
                        "qualify it or explicitly enable research-only unqualified use."
                    )
                inspected = self.state.policy_registry.inspect_checkpoint(policy_checkpoint)
                expected_sha = str(calo_parameters.get("policy_sha256", "") or "").lower()
                if (
                    not expected_sha
                    or inspected["sha256"].lower() != expected_sha
                    or policy.sha256.lower() != expected_sha
                ):
                    raise RuntimeError(
                        "CALO policy artifact changed after configuration; experiment start is blocked"
                    )
                binding = {
                    key: value
                    for key, value in calo_parameters.items()
                    if key.startswith("policy_")
                    or key
                    in {"deterministic_policy", "strict_policy_binding", "allow_unqualified_policy"}
                }
                binding["policy_name"] = policy.name
                if is_extension:
                    previous_binding = database.get_experiment_policy_binding(self.experiment_id)
                    if (
                        previous_binding
                        and str(previous_binding.get("sha256", "")).lower() != expected_sha
                    ):
                        raise RuntimeError(
                            "Experiment extension is blocked because the selected CALO policy differs from the immutable policy bound to the original experiment."
                        )
                else:
                    database.bind_policy_to_experiment(self.experiment_id, binding)

        # Filter and validate horizon continuation before creating tasks. Exact continuation is
        # currently implemented by CALO's full optimizer-state checkpoint. Other optimizers must
        # not be silently replayed and called a continuation.
        candidate_plan = list(full_plan)
        if extension_mode == "extend_evaluation_horizon":
            selected_runs = set(
                int(i) for i in (getattr(self.config, "extension_run_indices", []) or [])
            )
            selected_algorithms = set(
                str(a) for a in (getattr(self.config, "extension_algorithm_names", []) or [])
            )
            if selected_runs:
                candidate_plan = [
                    item for item in candidate_plan if int(item.run_index) in selected_runs
                ]
            if selected_algorithms:
                candidate_plan = [
                    item for item in candidate_plan if str(item.label) in selected_algorithms
                ]
            if not candidate_plan:
                raise ValueError("No runs match the requested evaluation-horizon extension")
            resume_map = {}
            existing_map = {}
            strategy = str(
                getattr(self.config, "extension_execution_strategy", "exact_continue")
                or "exact_continue"
            )
            if strategy not in {"exact_continue", "recompute_from_seed"}:
                raise ValueError(f"Unsupported horizon-extension execution strategy: {strategy}")
            unsupported = sorted(
                {str(item.label) for item in candidate_plan if str(item.label) != "CALO"}
            )
            if strategy == "exact_continue" and unsupported:
                raise RuntimeError(
                    "Exact same-run optimizer-state continuation is currently implemented for CALO only. "
                    "Selected algorithms without complete exact checkpoints: "
                    + ", ".join(unsupported)
                    + ". Choose 'recompute from original paired seeds' for a scientifically valid multi-algorithm higher-horizon revision."
                )
            target_evaluations = int(self.config.budget.max_evaluations)
            for item in candidate_plan:
                existing = database.get_run_by_algorithm_index(
                    self.experiment_id, item.label, item.run_index
                )
                if existing is None:
                    raise RuntimeError(
                        f"Cannot extend missing {item.label} run {item.run_index + 1}"
                    )
                key = f"{item.label}:{int(item.run_index)}"
                previous_evaluations = self._result_evaluations_from_row(existing)
                available = database.available_run_horizons(str(existing["id"]))
                if target_evaluations in available:
                    raise ValueError(
                        f"{item.label} run {item.run_index + 1} already has preserved evidence at "
                        f"{target_evaluations} FE; refusing to overwrite/recompute the same horizon."
                    )
                checkpoint_path = ""
                checkpoint_sha = ""
                if strategy == "exact_continue":
                    source_horizon = int(getattr(self.config, "extension_source_horizon", 0) or 0)
                    if source_horizon not in available:
                        raise RuntimeError(
                            f"CALO run {item.run_index + 1} has no preserved evidence at the selected "
                            f"source horizon {source_horizon} FE."
                        )
                    if target_evaluations <= source_horizon:
                        raise ValueError(
                            f"Exact continuation target {target_evaluations} FE must exceed source horizon "
                            f"{source_horizon} FE for CALO run {item.run_index + 1}."
                        )
                    # Select the exact checkpoint belonging to the requested source horizon, not
                    # merely the most recently viewed/current branch of this logical run.
                    source_segments = [
                        segment
                        for segment in database.list_run_segments(str(existing["id"]))
                        if int(segment.get("end_evaluations", 0)) == source_horizon
                        and str(segment.get("checkpoint_path", "") or "")
                    ]
                    if source_segments:
                        checkpoint_path = str(source_segments[-1].get("checkpoint_path", "") or "")
                        checkpoint_sha = str(source_segments[-1].get("checkpoint_sha256", "") or "")
                    elif previous_evaluations == source_horizon:
                        payload = json.loads(str(existing.get("result_json", "{}") or "{}"))
                        continuation = dict(
                            (payload.get("metadata", {}) or {}).get("run_continuation", {}) or {}
                        )
                        checkpoint_path = str(continuation.get("checkpoint_path", "") or "")
                        checkpoint_sha = str(continuation.get("checkpoint_sha256", "") or "")
                    path = Path(checkpoint_path)
                    if not path.is_file():
                        raise RuntimeError(
                            f"Exact checkpoint is unavailable for CALO run {item.run_index + 1} at "
                            f"{source_horizon} FE; choose paired recomputation or restore that revision's checkpoint artifact."
                        )
                    actual_sha = self._sha256_file(checkpoint_path)
                    if checkpoint_sha and actual_sha.lower() != checkpoint_sha.lower():
                        raise RuntimeError(
                            f"Checkpoint checksum mismatch for CALO run {item.run_index + 1} at {source_horizon} FE"
                        )
                    resume_map[key] = checkpoint_path
                existing_map[key] = str(existing["id"])
            self.config.extension_checkpoint_paths = resume_map
            self.config.extension_existing_run_ids = existing_map

        # Register an immutable revision. Old evidence remains queryable at its original horizon.
        evolution = ExperimentEvolutionService(database)
        if is_extension:
            revision_id = str(getattr(self.config, "experiment_revision_id", "") or "")
            revision = database.get_experiment_revision(revision_id) if revision_id else None
            if revision is None:
                raise RuntimeError("Experiment extension has no valid revision record")
            database.update_experiment_revision(revision_id, status="running")
        else:
            original = evolution.ensure_original_revision(self.experiment_id)
            self.config.experiment_revision_id = str(original["id"])
            database.update_experiment_revision(str(original["id"]), status="running")

        self.campaign_id = database.create_campaign(
            self.experiment_id,
            str(getattr(self.config, "portfolio_id", "")),
            self.mode,
            self.config.to_dict(),
            len(candidate_plan),
        )
        self.resume_task_id = self.campaign_id
        self.state.resume_service.register(
            ResumeTaskType.EXPERIMENT,
            f"{self.config.name} ({self.mode})",
            {
                "campaign_id": self.campaign_id,
                "experiment_id": self.experiment_id,
                "mode": self.mode,
                "revision_id": str(getattr(self.config, "experiment_revision_id", "") or ""),
            },
            total=len(candidate_plan),
            task_id=self.resume_task_id,
            status=ResumeStatus.RUNNING,
        )
        required_outputs = list(getattr(self.config.portfolio, "requested_outputs", []))
        plan = []
        reused = 0
        for item in candidate_plan:
            seed = seeds[item.run_index]
            fp = run_fingerprint(self.config, item.label, item.run_index, seed)
            seed_payload = {
                "algorithm_seed": seed.algorithm_seed,
                "scenario_seed": seed.scenario_seed,
                "ai_inference_seed": seed.ai_inference_seed,
            }
            task_id = database.add_campaign_task(
                self.campaign_id,
                item.job_index,
                item.label,
                item.run_index,
                seed_payload,
                fp,
                required_outputs,
            )
            row = next(
                row
                for row in database.list_campaign_tasks(self.campaign_id)
                if row["id"] == task_id
            )
            self._task_by_job[int(item.job_index)] = row
            self._run_fingerprint_by_job[int(item.job_index)] = fp

            if extension_mode == "extend_evaluation_horizon":
                plan.append(item)
                continue

            # Run-count extensions reuse the original same-experiment rows without cloning; fresh
            # portfolio experiments may clone exact compatible evidence from another experiment.
            same_run = (
                database.get_run_by_algorithm_index(self.experiment_id, item.label, item.run_index)
                if is_extension
                else None
            )
            if same_run is not None:
                database.update_campaign_task(task_id, status="reused", run_id=str(same_run["id"]))
                database.append_task_event(task_id, "reused_existing", {"run_id": same_run["id"]})
                reused += 1
                continue
            reusable = None
            if bool(getattr(self.config, "reuse_compatible_results", True)) and not is_extension:
                reusable = database.find_reusable_run(
                    fp,
                    verified_only=bool(
                        getattr(self.config.portfolio, "require_independent_validation", False)
                    ),
                )
            if reusable is not None:
                cloned_run_id = database.clone_run_to_experiment(reusable["id"], self.experiment_id)
                database.update_campaign_task(task_id, status="reused", run_id=cloned_run_id)
                database.append_task_event(
                    task_id, "reused", {"source_run_id": reusable["id"], "run_id": cloned_run_id}
                )
                reused += 1
            else:
                plan.append(item)
        database.update_campaign(
            self.campaign_id,
            status="running",
            completed_tasks=reused,
            message=f"Prepared revision; reused {reused} existing compatible run(s)",
        )
        self._sync_campaign_progress(
            f"Prepared revision; reused {reused} existing compatible run(s)"
        )
        return plan

    def _pause_unfinished(self) -> None:
        if not self.campaign_id:
            return
        for row in self.state.database.list_campaign_tasks(self.campaign_id):
            if row["status"] in {"planned", "queued", "running", "pausing", "interrupted"}:
                self.state.database.update_campaign_task(row["id"], status="paused")
        self.state.database.update_campaign(
            self.campaign_id, status="paused", message="Paused safely; completed jobs retained"
        )
        revision_id = str(getattr(self.config, "experiment_revision_id", "") or "")
        if revision_id:
            self.state.database.update_experiment_revision(revision_id, status="paused")
        if self.resume_task_id:
            self.state.resume_service.update(
                self.resume_task_id, status=ResumeStatus.PAUSED, resumable=True
            )
        self._sync_campaign_progress("Paused safely")

    def run(self) -> None:
        try:
            self.config.validate()
            from calo_rpd_studio.portfolio.planner import PortfolioPlanner

            portfolio_plan = PortfolioPlanner.plan(self.config, self.config.portfolio)
            store = ResultStore(
                self.config.output_directory,
                storage_profile=portfolio_plan.storage_profile,
                required_fields=portfolio_plan.required_fields,
            )
            full_plan = build_execution_plan(self.config, self.mode)
            seeds = SeedManager(self.config.master_seed).generate(self.config.runs)
            plan = self._prepare_campaign(full_plan, seeds)
            self.experiment_created.emit(self.experiment_id)

            if not plan:
                self.state.database.update_campaign(
                    self.campaign_id,
                    status="completed",
                    message="All required jobs were already complete and reusable",
                )
                revision_id = str(getattr(self.config, "experiment_revision_id", "") or "")
                if revision_id:
                    self.state.database.update_experiment_revision(revision_id, status="completed")
                self.state.resume_service.update(
                    self.resume_task_id, status=ResumeStatus.COMPLETED, resumable=False
                )
                self.completed.emit(self.experiment_id)
                return

            if len(plan) > 1 and (
                int(self.config.parallel_workers) > 1
                or str(self.config.execution_backend).lower() == "throughput_auto"
            ):
                finished = self._run_parallel(self.experiment_id, store, plan, seeds)
            else:
                finished = self._run_sequential(self.experiment_id, store, plan, seeds)

            if finished:
                tasks = self.state.database.list_campaign_tasks(self.campaign_id)
                failures = [row for row in tasks if row["status"] == "failed"]
                status = "completed_with_failures" if failures else "completed"
                self.state.database.update_campaign(
                    self.campaign_id,
                    status=status,
                    message=(
                        f"Completed with {len(failures)} failed job(s)"
                        if failures
                        else "Portfolio numerical tasks completed"
                    ),
                )
                revision_id = str(getattr(self.config, "experiment_revision_id", "") or "")
                if revision_id:
                    self.state.database.update_experiment_revision(
                        revision_id, status=("failed" if failures else "completed")
                    )
                self.state.resume_service.update(
                    self.resume_task_id,
                    status=(ResumeStatus.FAILED if failures else ResumeStatus.COMPLETED),
                    resumable=bool(failures),
                )
                self.completed.emit(self.experiment_id)
            else:
                self._pause_unfinished()
                self.cancelled.emit(self.experiment_id)
        except Exception as exc:
            if self.campaign_id:
                self.state.database.update_campaign(
                    self.campaign_id, status="interrupted", message=f"{type(exc).__name__}: {exc}"
                )
            revision_id = str(getattr(self.config, "experiment_revision_id", "") or "")
            if revision_id:
                self.state.database.update_experiment_revision(revision_id, status="interrupted")
            if self.resume_task_id:
                self.state.resume_service.update(
                    self.resume_task_id, status=ResumeStatus.INTERRUPTED, resumable=True
                )
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

    def resume_campaign(self, campaign_id: str) -> bool:
        campaign = self.state.database.get_campaign(campaign_id)
        if campaign is None:
            self.failed.emit(f"Unknown resume campaign: {campaign_id}")
            return False
        from calo_rpd_studio.experiments.experiment_config import ExperimentConfig

        config = ExperimentConfig.from_dict(json.loads(campaign["config_json"]))
        config.resume_campaign_id = str(campaign_id)
        self.state.config = config
        self.state.current_experiment_id = str(campaign["experiment_id"] or "")
        self.state.update_config()
        return self._start(config, str(campaign["mode"]))

    def extend_run_count(self, experiment_id: str, new_total_runs: int) -> bool:
        """Append new independent paired runs to the same immutable experiment definition."""
        if self._busy or self.state.task_status.busy:
            self.busy.emit("A scientific task is already running.")
            return False
        try:
            _plan, config = ExperimentEvolutionService(self.state.database).extend_run_count(
                str(experiment_id), int(new_total_runs)
            )
            self.state.config = config
            self.state.current_experiment_id = str(experiment_id)
            self.state.update_config()
            return self._start(config, COMPARISON_MODE)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return False

    def extend_evaluation_horizon(
        self,
        experiment_id: str,
        new_evaluation_target: int,
        *,
        protocol: str = "manual_exploratory",
        run_indices=(),
        algorithm_names=(),
        execution_strategy: str = "exact_continue",
        source_horizon: int | None = None,
    ) -> bool:
        """Continue checkpoint-capable historical runs to a larger FE horizon.

        Publication eligibility is encoded by the protocol. The worker refuses to fake an exact
        continuation for algorithms that do not expose complete optimizer-state checkpoints.
        """
        if self._busy or self.state.task_status.busy:
            self.busy.emit("A scientific task is already running.")
            return False
        try:
            protocol_enum = ExtensionProtocol(str(protocol))
            _plan, config = ExperimentEvolutionService(
                self.state.database
            ).extend_evaluation_horizon(
                str(experiment_id),
                int(new_evaluation_target),
                protocol=protocol_enum,
                run_indices=tuple(int(i) for i in run_indices),
                algorithm_names=tuple(str(a) for a in algorithm_names),
                execution_strategy=str(execution_strategy),
                source_horizon=source_horizon,
            )
            self.state.config = config
            self.state.current_experiment_id = str(experiment_id)
            self.state.update_config()
            return self._start(config, COMPARISON_MODE)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return False

    def pause(self) -> None:
        """Request a safe pause. New jobs stop; active jobs finish and commit atomically."""
        worker = self.worker
        if worker is not None:
            self.state.task_status.update(
                detail="Safe pause requested; waiting for active jobs to finish"
            )
            worker.pause()

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
        elif str(config.execution_backend) == "throughput_auto":
            backend = (
                f"v3.4 persistent batched-throughput engine with automatic microbatch calibration "
                f"and up to {config.parallel_workers} concurrent runs"
            )
        elif str(config.execution_backend) in {"gpu_preferred", "cuda_priority", "cuda_only"}:
            if str(config.execution_backend) == "gpu_preferred":
                backend = (
                    "v3.4 GPU-maximum resident scheduler: 100% CUDA numerical work when CUDA is available; "
                    f"persistent workers and up to {config.parallel_workers} concurrent runs"
                )
            else:
                backend = (
                    f"v3.4 device-resident CUDA-priority scheduler ({config.cuda_task_share}/{config.xpu_task_share}/{config.cpu_task_share}) "
                    f"with persistent workers and up to {config.parallel_workers} concurrent runs"
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
        if data.get("phase") == "throughput_allocation_planned":
            effective = str(data.get("allocation_effective", ""))
            throughput = str(data.get("measured_throughput", ""))
            slots = dict(data.get("lane_slots", {}))
            batches = dict(data.get("calibrated_batch_sizes", {}))
            self.state.task_status.update(
                int(data.get("overall_percent", 0)),
                f"Batched evaluator-throughput plan · {effective} · {throughput} · slots {slots} · batches {batches} · optimizer control overhead is reported separately per run",
            )
            return
        if data.get("phase") == "throughput_calibration":
            self.state.task_status.update(
                int(data.get("overall_percent", 0)),
                f"Calibrating evaluator-only candidate-scenario throughput (optimizer control excluded) · {int(data.get('calibration_percent', 0))}%",
            )
            return
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
        self.state.task_status.cancelled(
            "Experiment cancelled safely; completed runs were retained"
        )
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
