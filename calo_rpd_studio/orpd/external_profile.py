"""Reviewed ORPD-control profiles for independently sourced AC network cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from calo_rpd_studio.power_system.case_model import (
    BR_STATUS,
    BUS_I,
    BUS_TYPE,
    GEN_BUS,
    GEN_STATUS,
    PV,
    REF,
    TAP,
)
from calo_rpd_studio.power_system.pglib_import import PGLIB_IMPORT_SCHEMA

from .variable_decoder import ORPDVariableConfig, ShuntControlDefinition


REVIEWED_ORPD_PROFILE_SCHEMA = "calo-rpd-reviewed-external-orpd-profile-v1"
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PROFILE_KEYS = frozenset(
    {
        "schema_version",
        "profile_id",
        "profile_version",
        "review_status",
        "reviewed_by",
        "reviewed_at_utc",
        "review_evidence",
        "rationale",
        "source_asset_sha256",
        "physical_case_checksum",
        "generator_voltage_buses",
        "transformer_branch_indices",
        "shunt_controls",
        "transformer_minimum",
        "transformer_maximum",
        "transformer_step",
        "discrete_transformer_taps",
        "discrete_shunts",
    }
)
_SHUNT_KEYS = frozenset(
    {
        "bus_number",
        "minimum_mvar",
        "maximum_mvar",
        "step_mvar",
        "semantics",
        "source",
    }
)


class ReviewedORPDProfileError(ValueError):
    """The independent ORPD formulation declaration was rejected."""


@dataclass(frozen=True, slots=True)
class ReviewedORPDProfile:
    """Human-reviewed controls bound to one exact imported physical case."""

    profile_id: str
    profile_version: str
    review_status: str
    reviewed_by: str
    reviewed_at_utc: str
    review_evidence: str
    rationale: str
    source_asset_sha256: str
    physical_case_checksum: str
    generator_voltage_buses: tuple[int, ...]
    transformer_branch_indices: tuple[int, ...]
    shunt_controls: tuple[ShuntControlDefinition, ...]
    transformer_minimum: float = 0.90
    transformer_maximum: float = 1.10
    transformer_step: float = 0.0125
    discrete_transformer_taps: bool = True
    discrete_shunts: bool = True
    schema_version: str = REVIEWED_ORPD_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in self.generator_voltage_buses):
            raise ReviewedORPDProfileError(
                "generator_voltage_buses must contain only JSON integers"
            )
        if any(type(value) is not int for value in self.transformer_branch_indices):
            raise ReviewedORPDProfileError(
                "transformer_branch_indices must contain only JSON integers"
            )
        object.__setattr__(
            self, "generator_voltage_buses", tuple(int(v) for v in self.generator_voltage_buses)
        )
        object.__setattr__(
            self,
            "transformer_branch_indices",
            tuple(int(v) for v in self.transformer_branch_indices),
        )
        object.__setattr__(self, "shunt_controls", tuple(self.shunt_controls))
        self.validate()

    def validate(self) -> None:
        if self.schema_version != REVIEWED_ORPD_PROFILE_SCHEMA:
            raise ReviewedORPDProfileError("Unsupported reviewed ORPD profile schema")
        string_fields = {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "reviewed_by": self.reviewed_by,
            "review_evidence": self.review_evidence,
            "rationale": self.rationale,
            "reviewed_at_utc": self.reviewed_at_utc,
            "source_asset_sha256": self.source_asset_sha256,
            "physical_case_checksum": self.physical_case_checksum,
            "schema_version": self.schema_version,
        }
        for field, value in string_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ReviewedORPDProfileError(f"{field} must be non-empty")
        if (
            type(self.discrete_transformer_taps) is not bool
            or type(self.discrete_shunts) is not bool
        ):
            raise ReviewedORPDProfileError("Reviewed lattice flags must be JSON booleans")
        if self.review_status != "reviewed":
            raise ReviewedORPDProfileError(
                "External ORPD profile must have review_status='reviewed'"
            )
        try:
            reviewed = datetime.fromisoformat(self.reviewed_at_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ReviewedORPDProfileError("reviewed_at_utc must be an ISO-8601 timestamp") from exc
        if not self.reviewed_at_utc.endswith("Z") or reviewed.tzinfo != timezone.utc:
            raise ReviewedORPDProfileError("reviewed_at_utc must be an explicit UTC Z timestamp")
        if not _HEX64.fullmatch(self.source_asset_sha256):
            raise ReviewedORPDProfileError("source_asset_sha256 must be a lowercase SHA-256")
        if not _HEX64.fullmatch(self.physical_case_checksum):
            raise ReviewedORPDProfileError("physical_case_checksum must be a lowercase SHA-256")
        if len(set(self.generator_voltage_buses)) != len(self.generator_voltage_buses):
            raise ReviewedORPDProfileError("generator_voltage_buses must not contain duplicates")
        if len(set(self.transformer_branch_indices)) != len(self.transformer_branch_indices):
            raise ReviewedORPDProfileError("transformer_branch_indices must not contain duplicates")
        if any(bus < 0 for bus in self.generator_voltage_buses) or any(
            index < 0 for index in self.transformer_branch_indices
        ):
            raise ReviewedORPDProfileError("Reviewed control identifiers must be non-negative")
        seen_shunts: set[int] = set()
        for control in self.shunt_controls:
            if not isinstance(control, ShuntControlDefinition):
                raise ReviewedORPDProfileError(
                    "shunt_controls must contain ShuntControlDefinition objects"
                )
            control.validate()
            if control.bus_number in seen_shunts:
                raise ReviewedORPDProfileError(
                    f"Duplicate reviewed shunt control at bus {control.bus_number}"
                )
            seen_shunts.add(control.bus_number)
        # Reuse the production config's exact bound/lattice validation.
        ORPDVariableConfig(
            transformer_minimum=self.transformer_minimum,
            transformer_maximum=self.transformer_maximum,
            transformer_step=self.transformer_step,
            formulation_profile="review-validation-only",
        )

    def payload(self) -> dict:
        return asdict(self)

    def checksum(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_reviewed_orpd_profile(path: str | Path, *, expected_sha256: str) -> ReviewedORPDProfile:
    """Load a strict JSON profile only after verifying the exact serialized bytes."""

    if not _HEX64.fullmatch(expected_sha256):
        raise ReviewedORPDProfileError("expected_sha256 must be a lowercase SHA-256")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ReviewedORPDProfileError("Reviewed ORPD profile asset SHA-256 mismatch")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewedORPDProfileError("Reviewed ORPD profile must be valid UTF-8 JSON") from exc
    if not isinstance(data, dict) or set(data) != _PROFILE_KEYS:
        raise ReviewedORPDProfileError("Reviewed ORPD profile fields must exactly match the schema")
    shunt_data = data.get("shunt_controls")
    if not isinstance(shunt_data, list):
        raise ReviewedORPDProfileError("shunt_controls must be an array")
    shunts: list[ShuntControlDefinition] = []
    for item in shunt_data:
        if not isinstance(item, dict) or set(item) != _SHUNT_KEYS:
            raise ReviewedORPDProfileError(
                "Each reviewed shunt control must exactly match its schema"
            )
        if (
            type(item["bus_number"]) is not int
            or any(
                type(item[field]) not in {int, float}
                for field in ("minimum_mvar", "maximum_mvar", "step_mvar")
            )
            or not isinstance(item["semantics"], str)
            or not isinstance(item["source"], str)
        ):
            raise ReviewedORPDProfileError(
                "Reviewed shunt control values must use exact JSON scalar types"
            )
        shunts.append(ShuntControlDefinition(**item))
    for field in ("generator_voltage_buses", "transformer_branch_indices"):
        if not isinstance(data[field], list) or any(
            type(value) is not int for value in data[field]
        ):
            raise ReviewedORPDProfileError(f"{field} must be an array of JSON integers")
    for field in ("discrete_transformer_taps", "discrete_shunts"):
        if type(data[field]) is not bool:
            raise ReviewedORPDProfileError(f"{field} must be a JSON boolean")
    for field in ("transformer_minimum", "transformer_maximum", "transformer_step"):
        if type(data[field]) not in {int, float}:
            raise ReviewedORPDProfileError(f"{field} must be a JSON number")
    try:
        return ReviewedORPDProfile(
            profile_id=data["profile_id"],
            profile_version=data["profile_version"],
            review_status=data["review_status"],
            reviewed_by=data["reviewed_by"],
            reviewed_at_utc=data["reviewed_at_utc"],
            review_evidence=data["review_evidence"],
            rationale=data["rationale"],
            source_asset_sha256=data["source_asset_sha256"],
            physical_case_checksum=data["physical_case_checksum"],
            generator_voltage_buses=tuple(data["generator_voltage_buses"]),
            transformer_branch_indices=tuple(data["transformer_branch_indices"]),
            shunt_controls=tuple(shunts),
            transformer_minimum=data["transformer_minimum"],
            transformer_maximum=data["transformer_maximum"],
            transformer_step=data["transformer_step"],
            discrete_transformer_taps=data["discrete_transformer_taps"],
            discrete_shunts=data["discrete_shunts"],
            schema_version=data["schema_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ReviewedORPDProfileError):
            raise
        raise ReviewedORPDProfileError(f"Invalid reviewed ORPD profile: {exc}") from exc


def variable_config_from_reviewed_profile(
    case,
    profile: ReviewedORPDProfile,
    *,
    allow_protected_test: bool = False,
) -> ORPDVariableConfig:
    """Cross the explicit AC-case-to-ORPD boundary after exact provenance checks."""

    profile.validate()
    provenance = getattr(case, "source_provenance", None)
    if not isinstance(provenance, dict) or provenance.get("schema_version") != PGLIB_IMPORT_SCHEMA:
        raise ReviewedORPDProfileError("Reviewed external controls require a PGLib-imported case")
    if provenance.get("case_role") == "protected_test" and not allow_protected_test:
        raise ReviewedORPDProfileError(
            "Protected-test ORPD formulation requires explicit test-only access"
        )
    if provenance.get("asset_sha256") != profile.source_asset_sha256:
        raise ReviewedORPDProfileError("Profile source asset does not match imported case")
    if case.checksum() != profile.physical_case_checksum:
        raise ReviewedORPDProfileError("Profile physical checksum does not match imported case")

    bus_index = case.bus_index_map()
    online_generator_buses = {int(row[GEN_BUS]) for row in case.gen if float(row[GEN_STATUS]) > 0.0}
    invalid_voltage_buses = [
        bus
        for bus in profile.generator_voltage_buses
        if bus not in bus_index
        or bus not in online_generator_buses
        or int(case.bus[bus_index[bus], BUS_TYPE]) not in {REF, PV}
    ]
    if invalid_voltage_buses:
        raise ReviewedORPDProfileError(
            f"Invalid reviewed generator voltage controls: {invalid_voltage_buses}"
        )
    invalid_taps = [
        index
        for index in profile.transformer_branch_indices
        if index >= case.n_branch
        or float(case.branch[index, BR_STATUS]) <= 0.0
        or float(case.branch[index, TAP]) == 0.0
    ]
    if invalid_taps:
        raise ReviewedORPDProfileError(f"Invalid reviewed transformer controls: {invalid_taps}")
    case_buses = set(int(value) for value in case.bus[:, BUS_I])
    invalid_shunts = [
        control.bus_number
        for control in profile.shunt_controls
        if control.bus_number not in case_buses
    ]
    if invalid_shunts:
        raise ReviewedORPDProfileError(f"Invalid reviewed shunt controls: {invalid_shunts}")

    identity = (
        f"reviewed-external:{profile.profile_id}:{profile.profile_version}:{profile.checksum()}"
    )
    return ORPDVariableConfig(
        generator_voltages=bool(profile.generator_voltage_buses),
        transformer_taps=bool(profile.transformer_branch_indices),
        shunt_compensation=bool(profile.shunt_controls),
        discrete_transformer_taps=bool(profile.discrete_transformer_taps),
        discrete_shunts=bool(profile.discrete_shunts),
        transformer_minimum=float(profile.transformer_minimum),
        transformer_maximum=float(profile.transformer_maximum),
        transformer_step=float(profile.transformer_step),
        shunt_controls=profile.shunt_controls,
        formulation_profile=identity,
        generator_voltage_buses=profile.generator_voltage_buses,
        transformer_branch_indices=profile.transformer_branch_indices,
    )
