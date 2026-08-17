"""Immutable scientist-authored execution identities for Workspace and individual runs.

This module deliberately contains no optimizer or policy lifecycle behavior.  It freezes the
configuration already accepted by the ordinary experiment workflow and separates immutable
scientific/design hashes from mutable database lifecycle receipts.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
import uuid

from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.version import VERSION


ALGORITHM_STAGE_SCHEMA = "calo-rpd-algorithm-stage-v1"
WORKSPACE_PLAN_SCHEMA = "calo-rpd-workspace-study-plan-v2"
INDIVIDUAL_PLAN_SCHEMA = "calo-rpd-individual-experiment-plan-v2"
EXECUTION_CONTROLLER_SCHEMA = "calo-rpd-execution-controller-v1"


class ControllerKind(str, Enum):
    NONE = "none"
    WORKSPACE = "workspace"
    INDIVIDUAL_EXPERIMENT = "individual_experiment"


class ExecutionPlanKind(str, Enum):
    WORKSPACE = "workspace"
    INDIVIDUAL_EXPERIMENT = "individual_experiment"


class ExecutionLifecycle(str, Enum):
    DRAFT = "draft"
    AUDITED = "audited"
    STAGED = "staged"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    INTERRUPTED_RESUMABLE = "interrupted_resumable"
    COMPLETED = "completed"
    COMPLETED_WITH_FAILURES = "completed_with_failures"
    CANCELLED = "cancelled"
    FAILED_NON_RESUMABLE = "failed_non_resumable"
    DISCARDED_UNSTARTED = "discarded_unstarted"


TERMINAL_LIFECYCLES = frozenset(
    {
        ExecutionLifecycle.COMPLETED.value,
        ExecutionLifecycle.COMPLETED_WITH_FAILURES.value,
        ExecutionLifecycle.CANCELLED.value,
        ExecutionLifecycle.FAILED_NON_RESUMABLE.value,
        ExecutionLifecycle.DISCARDED_UNSTARTED.value,
    }
)
RESUMABLE_LIFECYCLES = frozenset(
    {ExecutionLifecycle.PAUSED.value, ExecutionLifecycle.INTERRUPTED_RESUMABLE.value}
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(payload: object) -> str:
    """Return the one accepted canonical representation for execution identities."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _selected_parameters(config_payload: dict, algorithm_names: tuple[str, ...]) -> dict:
    parameters = dict(config_payload.get("algorithm_parameters", {}) or {})
    return {name: deepcopy(dict(parameters.get(name, {}) or {})) for name in algorithm_names}


def _policy_binding_summary(config_payload: dict, algorithm_names: tuple[str, ...]) -> dict:
    """Freeze only an already-present TSH-CALO binding; never create or change one."""

    if "TSH-CALO" not in algorithm_names:
        return {}
    parameters = dict(config_payload.get("algorithm_parameters", {}) or {}).get("TSH-CALO", {})
    parameters = dict(parameters or {})
    permitted = {
        key: deepcopy(value)
        for key, value in parameters.items()
        if str(key).startswith("policy_")
        or key
        in {
            "use_ai",
            "strict_policy_binding",
            "allow_unqualified_policy",
            "deterministic_policy",
            "allow_cpu_fallback",
            "baseline_fallback_permitted",
        }
    }
    return permitted


def frozen_config_payload(
    config,
    algorithm_names: tuple[str, ...],
    *,
    plan_kind: ExecutionPlanKind | str,
) -> dict:
    """Clone a config into an execution-plan payload without trusting old runtime assignment."""

    kind = plan_kind.value if isinstance(plan_kind, ExecutionPlanKind) else str(plan_kind)
    if kind not in {item.value for item in ExecutionPlanKind}:
        raise ValueError(f"Unsupported execution-plan kind: {kind}")
    payload = deepcopy(config.to_dict())
    payload["algorithms"] = list(algorithm_names)
    payload["algorithm_parameters"] = _selected_parameters(payload, algorithm_names)
    payload["runtime_assigned_physical_device"] = ""
    payload["runtime_assigned_logical_device"] = "cpu"
    payload["runtime_compute_device"] = "cpu"
    payload["runtime_fallback_policy"] = "unresolved"
    payload["runtime_fallback_reason"] = ""
    payload["runtime_device_resolution"] = {}
    payload["resume_campaign_id"] = ""
    payload["run_checkpoint_root"] = ""
    payload["extension_checkpoint_paths"] = {}
    payload["execution_plan_id"] = ""
    payload["execution_plan_design_sha256"] = ""
    payload["algorithm_stage_id"] = ""
    payload["workspace_plan_cell_id"] = ""
    payload["execution_plan_kind"] = kind
    payload["result_contract"] = {}
    return payload


def frozen_individual_config_payload(config, algorithm_names: tuple[str, ...]) -> dict:
    """Freeze only scientist-selected individual inputs, excluding Workspace planning state."""

    payload = frozen_config_payload(
        config,
        algorithm_names,
        plan_kind=ExecutionPlanKind.INDIVIDUAL_EXPERIMENT,
    )
    payload["study_strength"] = "custom"
    payload["study_case_plan"] = [str(payload.get("case_name", "case30"))]
    payload["study_standardized_effect"] = None
    payload["study_target_power"] = None
    payload["study_family_alpha"] = 0.05
    payload["study_failure_allowance"] = 0.10
    payload["study_run_planning_method"] = "custom"
    payload.pop("portfolio", None)
    payload["portfolio_id"] = ""
    from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
    from calo_rpd_studio.experiments.result_contracts import build_individual_result_contract

    individual_config = ExperimentConfig.from_dict(payload)
    payload["result_contract"] = build_individual_result_contract(
        individual_config, algorithm_names
    )
    return payload


@dataclass(frozen=True, slots=True)
class AlgorithmStage:
    stage_id: str
    created_at: str
    algorithm_names: tuple[str, ...]
    algorithm_parameters: dict
    algorithm_parameter_sha256_by_name: dict
    policy_binding_summary: dict
    policy_binding_sha256: str
    source_provenance: dict
    content_sha256: str
    record_sha256: str
    schema_version: str = ALGORITHM_STAGE_SCHEMA

    @classmethod
    def create(cls, config) -> "AlgorithmStage":
        from calo_rpd_studio.algorithms.registry import POLICY_GATED_SPECS, SPECS

        names = tuple(str(name) for name in config.algorithms)
        if not names or len(set(names)) != len(names):
            raise ValueError("An algorithm stage must contain ordered, unique algorithm identities")
        registered = set(SPECS) | set(POLICY_GATED_SPECS)
        missing = sorted(set(names) - registered)
        if missing:
            raise ValueError("Algorithms are not registered for execution: " + ", ".join(missing))
        config_payload = config.to_dict()
        parameters = _selected_parameters(config_payload, names)
        policy = _policy_binding_summary(config_payload, names)
        parameter_hashes = {name: canonical_sha256(parameters.get(name, {})) for name in names}

        source_identity = resolve_source_identity(cwd=Path(__file__).resolve().parents[2])
        source_provenance = {
            "application_version": VERSION,
            "algorithm_registry_sha256": canonical_sha256({"algorithm_names": list(names)}),
            "experiment_config_schema": "strict-dataclass-fields-v1",
            **source_identity.to_dict(),
        }
        content = {
            "schema_version": ALGORITHM_STAGE_SCHEMA,
            "algorithm_names": list(names),
            "algorithm_parameters": parameters,
            "algorithm_parameter_sha256_by_name": parameter_hashes,
            "policy_binding_summary": policy,
            "policy_binding_sha256": canonical_sha256(policy),
            "source_provenance": source_provenance,
        }
        content_sha = canonical_sha256(content)
        stage_id = f"algorithm-stage-{uuid.uuid4().hex}"
        created_at = utc_now()
        record_sha = canonical_sha256(
            {
                "schema_version": ALGORITHM_STAGE_SCHEMA,
                "stage_id": stage_id,
                "created_at": created_at,
                "content_sha256": content_sha,
            }
        )
        return cls(
            stage_id=stage_id,
            created_at=created_at,
            algorithm_names=names,
            algorithm_parameters=parameters,
            algorithm_parameter_sha256_by_name=parameter_hashes,
            policy_binding_summary=policy,
            policy_binding_sha256=canonical_sha256(policy),
            source_provenance=source_provenance,
            content_sha256=content_sha,
            record_sha256=record_sha,
        )

    def content_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_names": list(self.algorithm_names),
            "algorithm_parameters": deepcopy(self.algorithm_parameters),
            "algorithm_parameter_sha256_by_name": deepcopy(
                self.algorithm_parameter_sha256_by_name
            ),
            "policy_binding_summary": deepcopy(self.policy_binding_summary),
            "policy_binding_sha256": self.policy_binding_sha256,
            "source_provenance": deepcopy(self.source_provenance),
        }


def _cell_payloads(plan_id: str, config_payload: dict) -> tuple[dict, ...]:
    case_names = tuple(
        dict.fromkeys(
            str(value)
            for value in (config_payload.get("study_case_plan", []) or [])
            if str(value).strip()
        )
    ) or (str(config_payload.get("case_name", "case30")),)
    cells: list[dict] = []
    for ordinal, case_name in enumerate(case_names):
        cell_config = deepcopy(config_payload)
        cell_config["case_name"] = case_name
        cell_design_sha = canonical_sha256(
            {
                "plan_id": plan_id,
                "ordinal": ordinal,
                "config": cell_config,
            }
        )
        cells.append(
            {
                "cell_id": f"workspace-cell-{cell_design_sha[:24]}",
                "ordinal": ordinal,
                "case_name": case_name,
                "config": cell_config,
                "design_sha256": cell_design_sha,
            }
        )
    return tuple(cells)


@dataclass(frozen=True, slots=True)
class WorkspaceStudyPlan:
    plan_id: str
    created_at: str
    algorithm_stage_id: str
    algorithm_stage_sha256: str
    study_algorithm_names: tuple[str, ...]
    config_payload: dict
    policy_binding: dict
    portfolio_fingerprint: str
    explicit_matrix: tuple[dict, ...]
    queue_task_count: int
    cells: tuple[dict, ...]
    design_sha256: str
    schema_version: str = WORKSPACE_PLAN_SCHEMA

    @classmethod
    def create(
        cls, config, stage: AlgorithmStage, study_algorithm_names: tuple[str, ...]
    ) -> "WorkspaceStudyPlan":
        names = tuple(str(name) for name in study_algorithm_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("A Workspace study subset must be ordered, unique, and non-empty")
        missing = [name for name in names if name not in stage.algorithm_names]
        if missing:
            raise ValueError(
                "Workspace study algorithms are outside the submitted stage: " + ", ".join(missing)
            )
        config_names = tuple(str(name) for name in config.algorithms)
        if config_names != stage.algorithm_names or _selected_parameters(
            config.to_dict(), stage.algorithm_names
        ) != stage.algorithm_parameters:
            raise ValueError(
                "The submitted algorithm identities or parameters changed; submit the algorithm stage again"
            )
        plan_id = f"workspace-plan-{uuid.uuid4().hex}"
        config_payload = frozen_config_payload(
            config, names, plan_kind=ExecutionPlanKind.WORKSPACE
        )
        policy_binding = _policy_binding_summary(config_payload, names)
        cells = _cell_payloads(plan_id, config_payload)
        formulation_payload = {
            key: deepcopy(config_payload.get(key))
            for key in (
                "objective",
                "variables",
                "robust_objective",
                "power_flow",
                "constraint_tolerances",
            )
        }
        explicit_matrix = tuple(
            {
                "cell_id": str(cell["cell_id"]),
                "case_name": str(cell["case_name"]),
                "formulation_sha256": canonical_sha256(formulation_payload),
                "scenario_sha256": canonical_sha256(config_payload.get("scenarios", {})),
                "study_algorithm_names": list(names),
            }
            for cell in cells
        )
        portfolio_fingerprint = canonical_sha256(config_payload.get("portfolio", {}))
        queue_task_count = len(cells) * int(config_payload.get("runs", 0)) * len(names)
        design = {
            "schema_version": WORKSPACE_PLAN_SCHEMA,
            "algorithm_stage_id": stage.stage_id,
            "algorithm_stage_sha256": stage.content_sha256,
            "study_algorithm_names": list(names),
            "config": config_payload,
            "policy_binding": policy_binding,
            "portfolio_fingerprint": portfolio_fingerprint,
            "explicit_matrix": list(explicit_matrix),
            "queue_task_count": queue_task_count,
            "cells": list(cells),
        }
        return cls(
            plan_id=plan_id,
            created_at=utc_now(),
            algorithm_stage_id=stage.stage_id,
            algorithm_stage_sha256=stage.content_sha256,
            study_algorithm_names=names,
            config_payload=config_payload,
            policy_binding=policy_binding,
            portfolio_fingerprint=portfolio_fingerprint,
            explicit_matrix=explicit_matrix,
            queue_task_count=queue_task_count,
            cells=cells,
            design_sha256=canonical_sha256(design),
        )

    def design_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_stage_id": self.algorithm_stage_id,
            "algorithm_stage_sha256": self.algorithm_stage_sha256,
            "study_algorithm_names": list(self.study_algorithm_names),
            "config": deepcopy(self.config_payload),
            "policy_binding": deepcopy(self.policy_binding),
            "portfolio_fingerprint": self.portfolio_fingerprint,
            "explicit_matrix": deepcopy(list(self.explicit_matrix)),
            "queue_task_count": self.queue_task_count,
            "cells": deepcopy(list(self.cells)),
        }


@dataclass(frozen=True, slots=True)
class IndividualExperimentPlan:
    plan_id: str
    created_at: str
    algorithm_stage_id: str
    algorithm_stage_sha256: str
    algorithm_names: tuple[str, ...]
    config_payload: dict
    policy_binding: dict
    design_sha256: str
    schema_version: str = INDIVIDUAL_PLAN_SCHEMA

    @classmethod
    def create(cls, config, stage: AlgorithmStage) -> "IndividualExperimentPlan":
        names = stage.algorithm_names
        if tuple(str(name) for name in config.algorithms) != names:
            raise ValueError(
                "The individual experiment must use the complete unchanged submitted algorithm stage"
            )
        config_payload = frozen_individual_config_payload(config, names)
        if _selected_parameters(config.to_dict(), names) != stage.algorithm_parameters:
            raise ValueError(
                "Algorithm parameters changed after submission; submit the algorithm stage again"
            )
        policy_binding = _policy_binding_summary(config_payload, names)
        design = {
            "schema_version": INDIVIDUAL_PLAN_SCHEMA,
            "algorithm_stage_id": stage.stage_id,
            "algorithm_stage_sha256": stage.content_sha256,
            "algorithm_names": list(names),
            "config": config_payload,
            "policy_binding": policy_binding,
        }
        return cls(
            plan_id=f"individual-plan-{uuid.uuid4().hex}",
            created_at=utc_now(),
            algorithm_stage_id=stage.stage_id,
            algorithm_stage_sha256=stage.content_sha256,
            algorithm_names=names,
            config_payload=config_payload,
            policy_binding=policy_binding,
            design_sha256=canonical_sha256(design),
        )

    def design_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "algorithm_stage_id": self.algorithm_stage_id,
            "algorithm_stage_sha256": self.algorithm_stage_sha256,
            "algorithm_names": list(self.algorithm_names),
            "config": deepcopy(self.config_payload),
            "policy_binding": deepcopy(self.policy_binding),
        }


def audit_receipt_payload(*, design_sha256: str, audit_payload: dict) -> dict:
    payload = {
        "schema_version": "calo-rpd-execution-audit-receipt-v1",
        "design_sha256": str(design_sha256),
        "audit": deepcopy(dict(audit_payload)),
    }
    payload["audit_sha256"] = canonical_sha256(payload)
    return payload


def scientific_job_sha256(
    *, plan_id: str, cell_id: str, algorithm: str, run_index: int, seed_payload: dict, config: dict
) -> str:
    return canonical_sha256(
        {
            "schema_version": "calo-rpd-scientific-job-v1",
            "plan_id": str(plan_id),
            "cell_id": str(cell_id),
            "algorithm": str(algorithm),
            "algorithm_parameters": dict(config.get("algorithm_parameters", {})).get(
                str(algorithm), {}
            ),
            "run_index": int(run_index),
            "seed": deepcopy(seed_payload),
            "budget": deepcopy(config.get("budget", {})),
            "case_name": str(config.get("case_name", "")),
            "objective": deepcopy(config.get("objective", {})),
            "variables": deepcopy(config.get("variables", {})),
            "scenarios": deepcopy(config.get("scenarios", {})),
        }
    )


def resume_contract_sha256(config_payload: dict) -> str:
    """Hash every frozen design field that may affect resumed scientific work."""

    keys = [
        "case_name",
        "study_case_plan",
        "algorithms",
        "algorithm_parameters",
        "runs",
        "master_seed",
        "population_size",
        "max_iterations",
        "budget",
        "objective",
        "variables",
        "robust_objective",
        "power_flow",
        "constraint_tolerances",
        "scenarios",
        "execution_purpose",
        "requested_compute_device",
        "scientific_backend",
        "require_backend_parity",
        "parity_objective_tolerance",
        "parity_violation_tolerance",
        "parity_voltage_tolerance",
        "parity_angle_tolerance_deg",
        "execution_plan_id",
        "execution_plan_design_sha256",
        "algorithm_stage_id",
        "workspace_plan_cell_id",
        "execution_plan_kind",
        "result_contract",
    ]
    if str(config_payload.get("execution_plan_kind", "")) != (
        ExecutionPlanKind.INDIVIDUAL_EXPERIMENT.value
    ):
        keys.extend(("portfolio", "portfolio_id"))
    return canonical_sha256({key: deepcopy(config_payload.get(key)) for key in keys})
