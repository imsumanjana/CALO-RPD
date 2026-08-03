"""Qualification receipt integrity and protected-calibration invariants."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo.tsh_calo_qualification import (
    TSH_CALO_QUALIFICATION_RECEIPT_KEY,
    build_tsh_calo_qualification_receipt,
    calibration_from_receipt,
    load_tsh_calo_qualification_receipt,
    qualification_config,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import OODCalibration


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _receipt(*, cases=("case30", "case57")) -> dict:
    return build_tsh_calo_qualification_receipt(
        qualification_run_id="qualification-001",
        source_policy_sha256=_sha("policy"),
        source_commit="qualification-test",
        qualification_protocol_sha256=_sha("protocol"),
        seed_manifest_sha256=_sha("seeds"),
        evidence_artifact_sha256=_sha("synthetic-evidence-fixture"),
        development_cases=cases,
        ood_calibration=OODCalibration(np.arange(4.0), np.ones(4), 2.5, 0.1),
    )


def test_receipt_round_trip_freezes_exact_policy_calibration_and_provenance():
    config = qualification_config(_receipt())
    loaded = load_tsh_calo_qualification_receipt(config, expected_policy_sha256=_sha("policy"))
    calibration = calibration_from_receipt(loaded)

    assert loaded.schema_version == "tsh-calo-qualification-receipt-v1"
    assert loaded.development_cases == ("case30", "case57")
    assert loaded.receipt_sha256
    np.testing.assert_array_equal(calibration.mean, np.arange(4.0))
    np.testing.assert_array_equal(calibration.scale, np.ones(4))


def test_receipt_or_calibration_mutation_is_rejected():
    config = qualification_config(_receipt())
    config[TSH_CALO_QUALIFICATION_RECEIPT_KEY]["ood_calibration"]["mean"][0] = 99.0

    with pytest.raises(ValueError, match="calibration checksum"):
        load_tsh_calo_qualification_receipt(config)


def test_protected_holdout_cannot_enter_qualification_calibration():
    with pytest.raises(ValueError, match="Protected holdouts"):
        _receipt(cases=("case30", "case118"))
