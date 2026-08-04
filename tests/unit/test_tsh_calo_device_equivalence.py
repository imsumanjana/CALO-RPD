"""Candidate-bound CPU/CUDA policy equivalence evidence helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.calo.tsh_calo_inference import TSHCALOInferenceResult
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import (
    FallbackDisposition,
    PolicyFallbackDecision,
    ShieldTrace,
)
from calo_rpd_studio.scripts import validate_tsh_calo_device_equivalence as validator


def _result(offset: float = 0.0, operator: int = 2) -> TSHCALOInferenceResult:
    trace = ShieldTrace(
        "tsh-calo-shield-trace-v1",
        torch.tensor([0.1 + offset, 0.2, 0.3]),
        0.25 + offset,
        0.75,
        torch.tensor([[0.5 + offset, 0.2, 0.2, 0.1]]),
        torch.tensor([[True, True, False]]),
        ("invalid_operators_masked",),
    )
    return TSHCALOInferenceResult(
        PolicyFallbackDecision(FallbackDisposition.EXECUTE_POLICY, "TSH-CALO", "accepted"),
        1,
        torch.tensor([operator]),
        torch.tensor([[0.2 + offset, 0.8]]),
        torch.tensor([[0.4 + offset, 0.6]]),
        0.125 + offset,
        trace,
        {},
    )


def test_numeric_comparison_fails_closed_on_shape_nonfinite_and_tolerance():
    assert validator._numeric_comparison([1.0], [1.0 + 1e-7], rtol=1e-5, atol=1e-6)[
        "within_tolerance"
    ]
    assert not validator._numeric_comparison([1.0], [1.1], rtol=1e-5, atol=1e-6)["within_tolerance"]
    assert not validator._numeric_comparison([1.0], [1.0, 2.0], rtol=1e-5, atol=1e-6)[
        "within_tolerance"
    ]
    assert not validator._numeric_comparison([np.nan], [np.nan], rtol=1e-5, atol=1e-6)[
        "within_tolerance"
    ]


def test_policy_comparison_requires_exact_actions_and_bounded_numerics():
    matching = validator._compare_results(_result(), _result(1e-7), rtol=1e-5, atol=1e-6)
    action_mismatch = validator._compare_results(
        _result(), _result(1e-7, operator=3), rtol=1e-5, atol=1e-6
    )
    numeric_mismatch = validator._compare_results(_result(), _result(0.1), rtol=1e-5, atol=1e-6)

    assert matching["passed"] is True
    assert action_mismatch["passed"] is False
    assert numeric_mismatch["passed"] is False


def test_development_calibration_is_finite_and_deterministic(monkeypatch):
    signatures = {
        "left": np.asarray([0.0, 2.0, 4.0]),
        "right": np.asarray([2.0, 2.0, 8.0]),
    }
    monkeypatch.setattr(validator, "topology_ood_signature", signatures.__getitem__)

    first = validator._fit_development_calibration(["left", "right"])
    second = validator._fit_development_calibration(["left", "right"])

    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)
    assert first.attenuation_start == second.attenuation_start
    assert np.all(np.isfinite(first.scale))
    assert np.all(first.scale > 0.0)


def test_device_equivalence_evidence_refuses_overwrite(tmp_path):
    destination = tmp_path / "equivalence.json"
    validator._write_new_json(destination, {"passed": True})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        validator._write_new_json(destination, {"passed": False})
