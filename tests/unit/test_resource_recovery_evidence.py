from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import calo_rpd_studio.scripts.validate_resource_recovery as recovery_module
from calo_rpd_studio.scripts.validate_resource_recovery import (
    _MIB,
    _bounded_pressure_bytes,
    _recovery_within_tolerance,
)


def test_pressure_probe_is_bounded_by_fraction_and_absolute_ceiling():
    assert _bounded_pressure_bytes(8 * 1024**3, 0.05, 256) == 256 * _MIB
    assert _bounded_pressure_bytes(256 * _MIB, 0.10, 128) == int(25.6 * _MIB)


@pytest.mark.parametrize(
    ("free_bytes", "fraction", "maximum_mib", "message"),
    [
        (0, 0.05, 256, "free bytes"),
        (1024**3, 0.0, 256, "pressure fraction"),
        (1024**3, 0.26, 256, "pressure fraction"),
        (1024**3, 0.05, 8, "between 16 and 512"),
        (1024**3, 0.05, 513, "between 16 and 512"),
    ],
)
def test_pressure_probe_rejects_unsafe_or_unusable_requests(
    free_bytes, fraction, maximum_mib, message
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _bounded_pressure_bytes(free_bytes, fraction, maximum_mib)


def test_recovery_tolerance_is_explicit_and_never_negative():
    before = 4 * 1024**3
    assert _recovery_within_tolerance(before, before - 63 * _MIB, 64)
    assert not _recovery_within_tolerance(before, before - 65 * _MIB, 64)
    with pytest.raises(ValueError, match="cannot be negative"):
        _recovery_within_tolerance(before, before, -1)


def test_staged_host_probe_injects_only_the_initial_upload_oom(monkeypatch):
    class FakeEvaluator:
        vram_governor = None

        @staticmethod
        def _move_full_request_to_device(population):
            return population

    class FakeProblem:
        dimension = 3

        def __init__(self):
            self._device_resident_evaluator = FakeEvaluator()

        def evaluate_population_tensor(self, population):
            with pytest.raises(torch.OutOfMemoryError, match="initial-upload"):
                self._device_resident_evaluator._move_full_request_to_device(population)
            return SimpleNamespace(
                objective=SimpleNamespace(device="cuda:0"),
                count=len(population),
                metadata={
                    "vram_residency": {
                        "execution_state": "cuda_staged_host",
                        "input_staged_from_host": True,
                        "full_request_residency_attempted": True,
                        "full_request_residency_admitted": False,
                        "host_staging_reason": "full_request_input_cuda_allocation_exhausted",
                        "cpu_inner_loop_participation": False,
                        "request_oom_retries": 0,
                    }
                },
            )

    monkeypatch.setattr(
        recovery_module,
        "_build_cuda_problem",
        lambda *_args, **_kwargs: FakeProblem(),
    )

    result = recovery_module._staged_host_probe(
        "cuda:0", case_name="case30", seed=17, candidates=8, batch_size=8
    )

    assert result["passed"] is True
    assert result["initial_upload_attempts"] == 1
    assert result["natural_hardware_oom_claimed"] is False
    assert np.asarray(result["residency"]["input_staged_from_host"]).item() is True
