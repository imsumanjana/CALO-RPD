from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

import numpy as np

from calo_rpd_studio.accelerated.throughput_engine import (
    CrossRunBatchBroker,
    measured_throughput_allocation,
)
from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
    plan_training_lanes,
    plan_training_lanes_from_throughput,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


class _FakeEvaluator:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_lengths: list[int] = []
        self.lock = threading.Lock()

    def batch_signature(self) -> str:
        return "same-scientific-problem"

    def _evaluate_population_direct(self, candidates):
        array = np.asarray(candidates, dtype=float)
        with self.lock:
            self.calls += 1
            self.batch_lengths.append(len(array))
        return [float(row.sum()) for row in array]


def test_cross_run_broker_merges_compatible_population_requests():
    evaluator = _FakeEvaluator()
    barrier = threading.Barrier(3)
    with CrossRunBatchBroker(batch_window_ms=25.0, max_candidates=64) as broker:
        def submit(values):
            barrier.wait()
            return broker.submit(evaluator, values)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(submit, np.ones((4, 3)))
            second = pool.submit(submit, np.full((5, 3), 2.0))
            barrier.wait()
            assert first.result() == [3.0] * 4
            assert second.result() == [6.0] * 5

    assert evaluator.calls == 1
    assert evaluator.batch_lengths == [9]


def test_measured_throughput_allocation_uses_observed_capacity():
    allocation = measured_throughput_allocation(
        100,
        {"cuda": 700.0, "xpu": 200.0, "cpu": 100.0},
    )
    assert allocation == {"cuda": 70, "xpu": 20, "cpu": 10}


def test_policy_training_plan_can_be_auto_tuned_from_measured_throughput():
    base = plan_training_lanes(
        10,
        cuda_share=50,
        xpu_share=30,
        cpu_share=20,
        cuda_available=True,
        xpu_available=True,
        xpu_sidecar_available=False,
    )
    tuned = plan_training_lanes_from_throughput(
        10,
        {"cuda": 800.0, "xpu": 100.0, "cpu": 100.0},
        base_plan=base,
    )
    assert tuned.episode_counts == {"cuda": 8, "xpu": 1, "cpu": 1}
    assert any("auto-tuned" in item for item in tuned.warnings)


def test_v31_experiment_configuration_round_trip_preserves_throughput_fields():
    config = ExperimentConfig(
        execution_backend="throughput_auto",
        cross_run_batch_window_ms=6.5,
        max_cross_run_batch=8192,
        calibration_batch_sizes=[32, 64, 128],
        calibration_repetitions=3,
        telemetry_iteration_interval=20,
    )
    restored = ExperimentConfig.from_dict(config.to_dict())
    assert restored.execution_backend == "throughput_auto"
    assert restored.cross_run_batch_window_ms == 6.5
    assert restored.max_cross_run_batch == 8192
    assert restored.calibration_batch_sizes == [32, 64, 128]
    assert restored.calibration_repetitions == 3
    assert restored.telemetry_iteration_interval == 20


def test_v31_calo_policy_checkpoint_is_reused_inside_a_persistent_process():
    from pathlib import Path

    from calo_rpd_studio.algorithms.calo.ai_controller import AIController

    checkpoint = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "data"
        / "trained_models"
        / "calo_policy_v2.pt"
    )
    first = AIController(checkpoint, seed=1, device="cpu")
    second = AIController(checkpoint, seed=2, device="cpu")
    assert first.network is second.network
    assert first.checksum == second.checksum


def test_cross_run_broker_preserves_original_concatenation_error_and_recovers(monkeypatch):
    import calo_rpd_studio.accelerated.throughput_engine as throughput_engine

    evaluator = _FakeEvaluator()
    original = throughput_engine.np.concatenate

    def fail_concatenate(*_args, **_kwargs):
        raise ValueError("synthetic concatenation failure")

    with CrossRunBatchBroker(batch_window_ms=1.0, max_candidates=64) as broker:
        monkeypatch.setattr(throughput_engine.np, "concatenate", fail_concatenate)
        with pytest.raises(ValueError, match="synthetic concatenation failure"):
            broker.submit(evaluator, np.ones((2, 3)))
        assert broker._thread.is_alive()

        monkeypatch.setattr(throughput_engine.np, "concatenate", original)
        assert broker.submit(evaluator, np.ones((2, 3))) == [3.0, 3.0]


def test_cross_run_broker_does_not_merge_numpy_and_torch_requests():
    torch = pytest.importorskip("torch")
    evaluator = _FakeEvaluator()
    barrier = threading.Barrier(3)
    with CrossRunBatchBroker(batch_window_ms=25.0, max_candidates=64) as broker:
        def submit(values):
            barrier.wait()
            return broker.submit(evaluator, values)

        with ThreadPoolExecutor(max_workers=2) as pool:
            numpy_result = pool.submit(submit, np.ones((2, 3)))
            torch_result = pool.submit(submit, torch.ones((2, 3), dtype=torch.float64))
            barrier.wait()
            assert numpy_result.result() == [3.0, 3.0]
            assert torch_result.result() == [3.0, 3.0]

    assert evaluator.calls == 2


def test_cross_run_broker_rejects_non_matrix_candidates_without_blocking():
    evaluator = _FakeEvaluator()
    with CrossRunBatchBroker(batch_window_ms=1.0, max_candidates=64) as broker:
        with pytest.raises(ValueError, match="two-dimensional candidate matrix"):
            broker.submit(evaluator, np.ones((2, 3, 1)))
        assert broker.submit(evaluator, np.empty((0, 3))) == []
        assert broker._thread.is_alive()
