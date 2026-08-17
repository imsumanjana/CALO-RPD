"""Mode-specific result capture contracts for immutable execution plans.

Workspace studies derive their evidence/storage contract from Portfolio Manager.  A scientist-
launched individual experiment deliberately uses this direct contract instead, so its execution
never depends on an applied Workspace portfolio or study plan.
"""

from __future__ import annotations

from copy import deepcopy

from calo_rpd_studio.portfolio.catalog import OUTPUT_REQUIREMENTS
from calo_rpd_studio.portfolio.models import DEFAULT_SINGLE_RUN_OUTPUTS, StorageProfile


INDIVIDUAL_RESULT_CONTRACT_SCHEMA = "calo-rpd-individual-result-contract-v1"
_BASE_REQUIRED_FIELDS = frozenset({"decoded_controls", "final_metrics", "seed_provenance"})


def build_individual_result_contract(config, algorithm_names: tuple[str, ...]) -> dict:
    """Return the deterministic direct-output contract for one individual experiment plan."""

    names = tuple(str(name) for name in algorithm_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Individual result capture requires unique submitted algorithms")
    robust = str(getattr(config.scenarios, "mode", "deterministic")) != "deterministic"
    outputs: list[str] = []
    required_fields = set(_BASE_REQUIRED_FIELDS)
    for key in DEFAULT_SINGLE_RUN_OUTPUTS:
        requirement = OUTPUT_REQUIREMENTS[key]
        if requirement.requires_calo and "CALO" not in names:
            continue
        if requirement.robust_only and not robust:
            continue
        if len(names) < int(requirement.minimum_algorithms):
            continue
        outputs.append(key)
        required_fields.update(requirement.required_fields)
        if requirement.accelerator_records:
            required_fields.add("accelerator_telemetry")
    if robust:
        for key in ("contingency_matrix",):
            requirement = OUTPUT_REQUIREMENTS[key]
            outputs.append(key)
            required_fields.update(requirement.required_fields)
    contract = {
        "schema_version": INDIVIDUAL_RESULT_CONTRACT_SCHEMA,
        "owner": "individual_experiment",
        "requested_outputs": outputs,
        "required_fields": sorted(required_fields),
        "storage_profile": (
            StorageProfile.ROBUST_FULL.value if robust else StorageProfile.FULL_SINGLE_RUN.value
        ),
        "reuse_compatible_results": bool(getattr(config, "reuse_compatible_results", True)),
        # An individual run may reuse evidence only when that exact prior run was independently
        # verified. This is a direct experiment safety rule, not a Workspace Portfolio dependency.
        "reuse_verified_only": True,
    }
    validate_individual_result_contract(contract)
    return contract


def validate_individual_result_contract(payload: dict) -> dict:
    """Validate and return a defensive copy of an individual result contract."""

    contract = deepcopy(dict(payload or {}))
    if contract.get("schema_version") != INDIVIDUAL_RESULT_CONTRACT_SCHEMA:
        raise ValueError("Individual experiment result-contract schema is missing or unsupported")
    if contract.get("owner") != "individual_experiment":
        raise ValueError("Individual experiment result contract has the wrong owner")
    outputs = [str(value) for value in list(contract.get("requested_outputs", []))]
    if not outputs or len(outputs) != len(set(outputs)):
        raise ValueError("Individual experiment outputs must be non-empty and unique")
    unknown_outputs = sorted(set(outputs) - set(OUTPUT_REQUIREMENTS))
    if unknown_outputs:
        raise ValueError("Unknown individual experiment outputs: " + ", ".join(unknown_outputs))
    required_fields = [str(value) for value in list(contract.get("required_fields", []))]
    expected_fields = set(_BASE_REQUIRED_FIELDS)
    for output in outputs:
        requirement = OUTPUT_REQUIREMENTS[output]
        expected_fields.update(requirement.required_fields)
        if requirement.accelerator_records:
            expected_fields.add("accelerator_telemetry")
    if len(required_fields) != len(set(required_fields)) or not expected_fields.issubset(
        required_fields
    ):
        raise ValueError(
            "Individual experiment required result fields are incomplete or duplicated"
        )
    if str(contract.get("storage_profile", "")) not in {
        StorageProfile.FULL_SINGLE_RUN.value,
        StorageProfile.ROBUST_FULL.value,
    }:
        raise ValueError("Individual experiment storage profile is unsupported")
    for key in ("reuse_compatible_results", "reuse_verified_only"):
        if not isinstance(contract.get(key), bool):
            raise ValueError(f"Individual experiment result contract {key} must be boolean")
    contract["requested_outputs"] = outputs
    contract["required_fields"] = sorted(required_fields)
    return contract
