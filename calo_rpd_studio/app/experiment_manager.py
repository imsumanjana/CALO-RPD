"""Qt-safe orchestration of sequential or process-parallel scientific experiments."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing as mp
import os
import queue
import threading
import time

from PyQt6.QtCore import QObject, QThread, pyqtSignal

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


def _execute_process_job(
    config,
    mode: str,
    item: PlannedItem,
    seeds: RunSeeds,
    progress_queue,
    cancel_event,
):
    """Top-level, spawn-safe process worker.

    SQLite and result-array persistence deliberately remain in the parent ExperimentWorker so
    concurrent numerical workers never write to the same database connection.
    """

    _configure_child_numeric_threads()
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
                }
            )
            progress_queue.put(data)
            last_emit = now
            last_evaluations = evaluations

    try:
        if mode == COMPARISON_MODE:
            completed = run_single(
                config,
                item.label,
                item.run_index,
                seeds,
                emit_progress,
                cancelled,
            )
        elif mode == ABLATION_MODE:
            completed = run_ablation(
                config,
                item.ablation_spec,
                item.run_index,
                seeds,
                emit_progress,
                cancelled,
            )
        else:
            raise ValueError(f"Unsupported experiment mode: {mode}")
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
        total_items = max(1, len(plan))
        fractions = {item.job_index: 0.0 for item in plan}
        completed_count = 0

        for item in plan:
            if self._cancelled():
                return False

            def emit_progress(payload: dict) -> None:
                self._emit_progress(
                    payload,
                    item,
                    fractions,
                    completed_count,
                    total_items,
                    1,
                )

            try:
                if self.mode == COMPARISON_MODE:
                    completed = run_single(
                        self.config,
                        item.label,
                        item.run_index,
                        seeds[item.run_index],
                        emit_progress,
                        self._cancelled,
                    )
                else:
                    completed = run_ablation(
                        self.config,
                        item.ablation_spec,
                        item.run_index,
                        seeds[item.run_index],
                        emit_progress,
                        self._cancelled,
                    )
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

    def _run_parallel(self, experiment_id: str, store: ResultStore, plan, seeds) -> bool:
        """Run independent optimizer jobs in separate spawn processes.

        This is throughput parallelism across independent run items. It accelerates large benchmark
        campaigns but makes per-run wall-clock timing subject to CPU contention. Strict publication
        timing comparisons should therefore use one worker.
        """

        total_items = max(1, len(plan))
        requested_workers = max(1, int(self.config.parallel_workers))
        max_workers = min(requested_workers, total_items)
        fractions = {item.job_index: 0.0 for item in plan}
        completed_count = 0
        next_submit = 0

        # Spawn is the safe cross-platform choice for a Qt application, especially on Windows.
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
            try:
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
                    pending = {}

                    def submit_next() -> bool:
                        nonlocal next_submit
                        if next_submit >= len(plan) or self._cancelled():
                            return False
                        item = plan[next_submit]
                        next_submit += 1
                        future = executor.submit(
                            _execute_process_job,
                            self.config,
                            self.mode,
                            item,
                            seeds[item.run_index],
                            progress_queue,
                            cancel_event,
                        )
                        pending[future] = item
                        return True

                    for _ in range(max_workers):
                        if not submit_next():
                            break

                    while pending:
                        if self._cancelled():
                            cancel_event.set()

                        for payload in self._drain_progress_queue(progress_queue):
                            job_index = int(payload.get("job_index", -1))
                            item = next((x for x in plan if x.job_index == job_index), None)
                            if item is not None:
                                self._emit_progress(
                                    payload,
                                    item,
                                    fractions,
                                    completed_count,
                                    total_items,
                                    len(pending),
                                )

                        done, _ = wait(tuple(pending), timeout=0.10, return_when=FIRST_COMPLETED)
                        if not done:
                            continue

                        for future in done:
                            item = pending.pop(future)
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
                                }
                            )

                            if not self._cancelled():
                                submit_next()

                    # Drain any final telemetry that arrived just before the last result.
                    for payload in self._drain_progress_queue(progress_queue):
                        job_index = int(payload.get("job_index", -1))
                        item = next((x for x in plan if x.job_index == job_index), None)
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
        backend = (
            f"CPU process pool with {config.parallel_workers} workers"
            if int(config.parallel_workers) > 1
            else "single CPU worker"
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
        algorithm = str(data.get("algorithm", "optimizer"))
        completed_items = data.get("completed_items")
        total = data.get("total_run_items")
        evaluations = data.get("evaluations")
        active = data.get("active_items")
        detail = algorithm
        if completed_items is not None and total is not None:
            detail += f" · {completed_items}/{total} jobs completed"
        if active is not None and int(self.state.config.parallel_workers) > 1:
            detail += f" · {active} active"
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
