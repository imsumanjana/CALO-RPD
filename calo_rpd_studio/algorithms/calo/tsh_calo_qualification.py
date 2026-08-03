"""Immutable qualification receipt contract for TSH-CALO activation and binding."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

import numpy as np

from calo_rpd_studio.power_system.case_identity import protected_holdout_matches

from .tsh_calo_shield import OODCalibration, ood_calibration_sha256


TSH_CALO_QUALIFICATION_RECEIPT_KEY = "tsh_calo_qualification_receipt"


def _is_sha256(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TSHCALOQualificationReceipt:
    """Integrity receipt only; scientific acceptance remains the qualifier's responsibility."""

    schema_version: str
    qualification_run_id: str
    source_policy_sha256: str
    source_commit: str
    qualification_protocol_sha256: str
    seed_manifest_sha256: str
    evidence_artifact_sha256: str
    development_cases: tuple[str, ...]
    ood_calibration: dict
    ood_calibration_sha256: str
    receipt_sha256: str = ""

    SCHEMA_VERSION = "tsh-calo-qualification-receipt-v1"

    def unsigned_payload(self) -> dict:
        payload = asdict(self)
        payload.pop("receipt_sha256", None)
        payload["development_cases"] = list(self.development_cases)
        return payload

    def validate(self, *, expected_policy_sha256: str = "") -> None:
        if self.schema_version != self.SCHEMA_VERSION:
            raise ValueError("TSH-CALO qualification receipt schema is incompatible")
        if not self.qualification_run_id.strip() or not self.source_commit.strip():
            raise ValueError("TSH-CALO qualification receipt requires run and source identities")
        for label, digest in (
            ("source policy", self.source_policy_sha256),
            ("qualification protocol", self.qualification_protocol_sha256),
            ("seed manifest", self.seed_manifest_sha256),
            ("evidence artifact", self.evidence_artifact_sha256),
            ("OOD calibration", self.ood_calibration_sha256),
            ("receipt", self.receipt_sha256),
        ):
            if not _is_sha256(digest):
                raise ValueError(f"TSH-CALO qualification {label} SHA-256 is invalid")
        if (
            expected_policy_sha256
            and self.source_policy_sha256 != str(expected_policy_sha256).lower()
        ):
            raise ValueError("TSH-CALO qualification receipt belongs to a different policy")
        if not self.development_cases:
            raise ValueError("TSH-CALO qualification receipt requires development cases")
        leaked = protected_holdout_matches(self.development_cases)
        if leaked:
            raise ValueError(
                "Protected holdouts cannot enter TSH-CALO qualification calibration: "
                + ", ".join(leaked)
            )
        calibration = calibration_from_receipt(self)
        if ood_calibration_sha256(calibration) != self.ood_calibration_sha256:
            raise ValueError("TSH-CALO qualification OOD calibration checksum mismatch")
        if _canonical_sha256(self.unsigned_payload()) != self.receipt_sha256:
            raise ValueError("TSH-CALO qualification receipt checksum mismatch")

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["development_cases"] = list(self.development_cases)
        return payload


def calibration_from_receipt(receipt: TSHCALOQualificationReceipt) -> OODCalibration:
    payload = dict(receipt.ood_calibration or {})
    calibration = OODCalibration(
        np.asarray(payload.get("mean", []), dtype=float),
        np.asarray(payload.get("scale", []), dtype=float),
        float(payload.get("attenuation_start", 2.0)),
        float(payload.get("minimum_neural_weight", 0.0)),
    )
    calibration.validate()
    return calibration


def build_tsh_calo_qualification_receipt(
    *,
    qualification_run_id: str,
    source_policy_sha256: str,
    source_commit: str,
    qualification_protocol_sha256: str,
    seed_manifest_sha256: str,
    evidence_artifact_sha256: str,
    development_cases,
    ood_calibration: OODCalibration,
) -> dict:
    """Freeze supplied qualification inputs without deciding whether their evidence passes."""

    calibration_sha = ood_calibration_sha256(ood_calibration)
    calibration_payload = {
        "mean": np.asarray(ood_calibration.mean, dtype=float).tolist(),
        "scale": np.asarray(ood_calibration.scale, dtype=float).tolist(),
        "attenuation_start": float(ood_calibration.attenuation_start),
        "minimum_neural_weight": float(ood_calibration.minimum_neural_weight),
    }
    provisional = TSHCALOQualificationReceipt(
        TSHCALOQualificationReceipt.SCHEMA_VERSION,
        str(qualification_run_id),
        str(source_policy_sha256).lower(),
        str(source_commit),
        str(qualification_protocol_sha256).lower(),
        str(seed_manifest_sha256).lower(),
        str(evidence_artifact_sha256).lower(),
        tuple(str(item) for item in development_cases),
        calibration_payload,
        calibration_sha,
    )
    receipt = replace(
        provisional,
        receipt_sha256=_canonical_sha256(provisional.unsigned_payload()),
    )
    receipt.validate(expected_policy_sha256=source_policy_sha256)
    return receipt.as_dict()


def qualification_config(receipt: dict) -> dict:
    loaded = load_tsh_calo_qualification_receipt(
        {TSH_CALO_QUALIFICATION_RECEIPT_KEY: dict(receipt)}
    )
    return {TSH_CALO_QUALIFICATION_RECEIPT_KEY: loaded.as_dict()}


def load_tsh_calo_qualification_receipt(
    config: dict,
    *,
    expected_policy_sha256: str = "",
) -> TSHCALOQualificationReceipt:
    payload = dict(config.get(TSH_CALO_QUALIFICATION_RECEIPT_KEY, {}) or {})
    if not payload:
        raise ValueError("TSH-CALO qualification is missing its immutable calibration receipt")
    payload["development_cases"] = tuple(payload.get("development_cases", ()))
    receipt = TSHCALOQualificationReceipt(**payload)
    receipt.validate(expected_policy_sha256=expected_policy_sha256)
    return receipt
