from __future__ import annotations
from types import SimpleNamespace

from calo_rpd_studio.app.experiment_manager import ExperimentManager, ExperimentWorker
from calo_rpd_studio.app.task_status import TaskStatus
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def test_worker_owns_nested_configuration_snapshot():
    config = ExperimentConfig()
    config.algorithms = ["TLBO"]
    config.algorithm_parameters = {"TLBO": {"example": [1, 2]}}
    worker = ExperimentWorker(SimpleNamespace(), config)
    config.runs = 77
    config.algorithm_parameters["TLBO"]["example"].append(3)
    assert worker.config.runs == 30
    assert worker.config.algorithm_parameters["TLBO"]["example"] == [1, 2]


def test_manager_snapshot_and_read_api_cannot_mutate_admitted_run(monkeypatch):
    state = SimpleNamespace(task_status=TaskStatus())
    manager = ExperimentManager(state)
    config = ExperimentConfig()
    config.algorithms = ["TLBO"]
    starts = []
    monkeypatch.setattr(ExperimentWorker, "start", lambda worker: starts.append(worker))
    assert manager.start_comparison(config)
    assert len(starts) == 1
    config.runs = 88
    exported = manager.active_config
    exported.algorithms.append("PSO")
    exported.budget.max_evaluations = 1
    assert manager.active_config.runs == 30
    assert manager.active_config.algorithms == ["TLBO"]
    assert manager.worker.config.algorithms == ["TLBO"]
    assert manager.worker.config.budget.max_evaluations == 5000
    assert manager.start_comparison(config) is False
    assert len(starts) == 1
    manager._worker_finished()
    state.task_status.finish("fixture complete")
    assert manager.active_config is None
    assert manager.running is False
