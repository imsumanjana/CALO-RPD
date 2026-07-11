"""Qt-safe orchestration of comparative and CALO ablation experiments."""
from __future__ import annotations

import threading

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from calo_rpd_studio.experiments.calo_ablation import ABLATION_SPECS, run_ablation
from calo_rpd_studio.experiments.experiment_runner import failed_run_from_exception, run_single
from calo_rpd_studio.experiments.provenance import collect_provenance
from calo_rpd_studio.experiments.seed_manager import SeedManager
from calo_rpd_studio.results.result_store import ResultStore


class ExperimentWorker(QThread):
    progress = pyqtSignal(dict)
    run_completed = pyqtSignal(str, str)
    run_failed = pyqtSignal(str, str)
    experiment_created = pyqtSignal(str)
    completed = pyqtSignal(str)
    cancelled = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, state, config, mode: str = "comparison") -> None:
        super().__init__()
        self.state = state
        self.config = config
        self.mode = mode
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        try:
            experiment_id = self.state.database.create_experiment(
                self.config,
                collect_provenance(),
            )
            self.experiment_created.emit(experiment_id)
            store = ResultStore(self.config.output_directory)
            items = (
                [(name, None) for name in self.config.algorithms]
                if self.mode == "comparison"
                else [(spec.label, spec) for spec in ABLATION_SPECS]
            )
            seeds = SeedManager(self.config.master_seed).generate(self.config.runs)
            total_items = max(1, self.config.runs * len(items))
            item_index = 0

            for run_index in range(self.config.runs):
                for label, spec in items:
                    if self._cancelled():
                        self.cancelled.emit(experiment_id)
                        return

                    def emit_progress(payload: dict) -> None:
                        data = dict(payload)
                        evaluations = int(data.get("evaluations", 0))
                        if self.config.budget.policy.value == "equal_evaluations":
                            fraction = min(1.0, evaluations / max(self.config.budget.max_evaluations, 1))
                        else:
                            fraction = 0.0
                        data["overall_percent"] = int(100 * (item_index + fraction) / total_items)
                        data["run_position"] = item_index + 1
                        data["total_run_items"] = total_items
                        data["run_index"] = run_index + 1
                        self.progress.emit(data)

                    try:
                        if spec is None:
                            completed = run_single(
                                self.config,
                                label,
                                run_index,
                                seeds[run_index],
                                emit_progress,
                                self._cancelled,
                            )
                        else:
                            completed = run_ablation(
                                self.config,
                                spec,
                                run_index,
                                seeds[run_index],
                                emit_progress,
                                self._cancelled,
                            )
                        path = store.save_arrays(completed.result)
                        run_id = self.state.database.add_run(
                            experiment_id,
                            completed,
                            str(path),
                        )
                        item_index += 1
                        self.progress.emit(
                            {
                                "algorithm": label,
                                "overall_percent": int(100 * item_index / total_items),
                                "run_position": item_index,
                                "total_run_items": total_items,
                                "run_index": run_index + 1,
                                "phase": "run_completed",
                            }
                        )
                        self.run_completed.emit(run_id, label)
                    except Exception as exc:
                        failure = failed_run_from_exception(
                            label,
                            run_index,
                            seeds[run_index],
                            exc,
                        )
                        failure_id = self.state.database.add_failure(
                            experiment_id,
                            failure,
                        )
                        item_index += 1
                        self.progress.emit(
                            {
                                "algorithm": label,
                                "overall_percent": int(100 * item_index / total_items),
                                "run_position": item_index,
                                "total_run_items": total_items,
                                "run_index": run_index + 1,
                                "phase": "run_failed",
                            }
                        )
                        self.run_failed.emit(failure_id, label)
            self.completed.emit(experiment_id)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ExperimentManager(QObject):
    progress = pyqtSignal(dict)
    run_completed = pyqtSignal(str, str)
    run_failed = pyqtSignal(str, str)
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
        self._mode = "comparison"
        self.state.task_status.cancel_requested.connect(self.cancel)

    @property
    def running(self) -> bool:
        return self._busy

    def start_comparison(self, config) -> bool:
        return self._start(config, "comparison")

    def start_calo_analysis(self, config) -> bool:
        return self._start(config, "ablation")

    def _start(self, config, mode: str) -> bool:
        if self._busy or self.state.task_status.busy:
            self.busy.emit(
                "A scientific task is already running. Wait for it to finish or request safe cancellation before starting another run."
            )
            return False

        self._busy = True
        self._mode = mode
        title = "Running comparative experiment" if mode == "comparison" else "Running CALO analysis"
        self.state.task_status.begin(
            title,
            detail="Preparing reproducible experiment",
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
        position = data.get("run_position")
        total = data.get("total_run_items")
        evaluations = data.get("evaluations")
        detail = algorithm
        if position is not None and total is not None:
            detail += f" · run item {position}/{total}"
        if evaluations is not None:
            detail += f" · {evaluations} evaluations"
        self.state.task_status.update(int(data.get("overall_percent", -1)), detail)

    def _on_run_completed(self, run_id: str, algorithm: str) -> None:
        self.state.runs_changed.emit()
        self.run_completed.emit(run_id, algorithm)

    def _on_run_failed(self, failure_id: str, algorithm: str) -> None:
        self.state.runs_changed.emit()
        self.run_failed.emit(failure_id, algorithm)

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
            self.state.task_status.update(detail="Safe cancellation requested; finishing the active numerical step")
            self.worker.cancel()
