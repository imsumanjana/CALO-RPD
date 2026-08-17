"""One-way Portfolio goal and Workspace Study planning contracts.

Portfolio freezes only scientist-authored evidence intent and intrinsic constraints.  Study
consumes that immutable goal, offers deterministic recommendations, and freezes the scientist's
concrete selections.  This module deliberately contains no execution-controller, audit, optimizer,
or policy lifecycle behavior.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import hashlib
import json
import uuid

from .catalog import OUTPUT_REQUIREMENTS
from .models import ArticlePreset, EvidenceProfile, PortfolioConfig, PortfolioKind
from .planner import PortfolioPlanner


APPLIED_PORTFOLIO_GOAL_SCHEMA = "calo-rpd-applied-portfolio-goal-v1"
STUDY_RECOMMENDATION_SCHEMA = "calo-rpd-study-recommendation-v1"
APPLIED_STUDY_SETUP_SCHEMA = "calo-rpd-applied-study-setup-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _broad_portfolio_payload(portfolio: PortfolioConfig) -> dict[str, Any]:
    """Return Portfolio-owned fields only; exact execution choices are intentionally absent."""

    return {
        "portfolio_type": portfolio.kind.value,
        "evidence_profile": portfolio.evidence_profile.value,
        "article_or_output_preset": portfolio.article_preset.value,
        "requested_outputs": list(portfolio.requested_outputs),
        "storage_or_evidence_class": portfolio.storage_profile.value,
        "require_independent_validation": bool(portfolio.require_independent_validation),
        "name": str(portfolio.name),
    }


@dataclass(frozen=True, slots=True)
class AppliedPortfolioGoal:
    portfolio_goal_id: str
    created_at: str
    algorithm_stage_id: str
    algorithm_stage_sha256: str
    selected_algorithm_names: tuple[str, ...]
    selected_algorithm_parameter_sha256_by_name: dict[str, str]
    portfolio: dict[str, Any]
    intrinsic_requirements: dict[str, Any]
    intrinsic_warnings: tuple[str, ...]
    content_sha256: str
    schema_version: str = APPLIED_PORTFOLIO_GOAL_SCHEMA

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "portfolio_goal_id": self.portfolio_goal_id,
            "created_at": self.created_at,
            "algorithm_stage_id": self.algorithm_stage_id,
            "algorithm_stage_sha256": self.algorithm_stage_sha256,
            "selected_algorithm_names": list(self.selected_algorithm_names),
            "selected_algorithm_parameter_sha256_by_name": deepcopy(
                self.selected_algorithm_parameter_sha256_by_name
            ),
            **deepcopy(self.portfolio),
            "intrinsic_requirements": deepcopy(self.intrinsic_requirements),
            "intrinsic_warnings": list(self.intrinsic_warnings),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "AppliedPortfolioGoal":
        content = deepcopy(dict(record.get("content", record)))
        portfolio_keys = (
            "portfolio_type",
            "evidence_profile",
            "article_or_output_preset",
            "requested_outputs",
            "storage_or_evidence_class",
            "require_independent_validation",
            "name",
        )
        return cls(
            portfolio_goal_id=str(content["portfolio_goal_id"]),
            created_at=str(content["created_at"]),
            algorithm_stage_id=str(content["algorithm_stage_id"]),
            algorithm_stage_sha256=str(content["algorithm_stage_sha256"]),
            selected_algorithm_names=tuple(content["selected_algorithm_names"]),
            selected_algorithm_parameter_sha256_by_name=deepcopy(
                dict(content["selected_algorithm_parameter_sha256_by_name"])
            ),
            portfolio={key: deepcopy(content[key]) for key in portfolio_keys},
            intrinsic_requirements=deepcopy(dict(content["intrinsic_requirements"])),
            intrinsic_warnings=tuple(content.get("intrinsic_warnings", [])),
            content_sha256=str(record.get("content_sha256", content.get("content_sha256", ""))),
            schema_version=str(content.get("schema_version", APPLIED_PORTFOLIO_GOAL_SCHEMA)),
        )


class PortfolioGoalPlanner:
    """Derive broad evidence constraints without reading any concrete Study configuration."""

    @staticmethod
    def create(
        portfolio: PortfolioConfig,
        stage,
        selected_algorithm_names: tuple[str, ...] | list[str],
    ) -> AppliedPortfolioGoal:
        if stage is None:
            raise ValueError(
                "Submit algorithms first. Portfolio can select a comparison scope only from "
                "the immutable submitted stage. No goal or execution plan has started."
            )
        names = tuple(str(name) for name in selected_algorithm_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Select at least one submitted algorithm for the Portfolio comparison scope")
        outside = [name for name in names if name not in stage.algorithm_names]
        if outside:
            raise ValueError(
                "Portfolio comparison scope contains algorithms outside the submitted stage: "
                + ", ".join(outside)
            )

        broad = deepcopy(portfolio)
        preset_requirements = PortfolioPlanner.apply_article_preset(None, broad)
        if not broad.requested_outputs:
            raise ValueError("Select at least one Portfolio deliverable")
        unknown = [key for key in broad.requested_outputs if key not in OUTPUT_REQUIREMENTS]
        if unknown:
            raise ValueError("Unknown Portfolio deliverables: " + ", ".join(unknown))
        required_algorithms = set(preset_requirements.required_algorithms)
        required_algorithms.update(
            "CALO" for key in broad.requested_outputs if OUTPUT_REQUIREMENTS[key].requires_calo
        )
        missing_required = sorted(required_algorithms - set(names))
        if missing_required:
            raise ValueError(
                "Include the intrinsically required submitted algorithm(s): "
                + ", ".join(missing_required)
            )

        minimum_runs = max(OUTPUT_REQUIREMENTS[key].minimum_runs for key in broad.requested_outputs)
        minimum_algorithms = max(
            OUTPUT_REQUIREMENTS[key].minimum_algorithms for key in broad.requested_outputs
        )
        minimum_blocks = max(
            OUTPUT_REQUIREMENTS[key].minimum_blocks for key in broad.requested_outputs
        )
        if len(names) < minimum_algorithms:
            raise ValueError(
                f"The requested Portfolio evidence requires at least {minimum_algorithms} "
                f"algorithms; the comparison scope contains {len(names)}"
            )
        if broad.kind is PortfolioKind.SINGLE_RUN and minimum_runs > 1:
            raise ValueError(
                "A single-run Portfolio cannot request evidence that intrinsically requires "
                f"at least {minimum_runs} paired runs"
            )

        required_fields = {"final_metrics", "decoded_controls", "seed_provenance"}
        for key in broad.requested_outputs:
            required_fields.update(OUTPUT_REQUIREMENTS[key].required_fields)
            if OUTPUT_REQUIREMENTS[key].accelerator_records:
                required_fields.add("accelerator_telemetry")
        warnings: list[str] = []
        recommended_omitted = [
            name
            for name in preset_requirements.recommended_algorithms
            if name in stage.algorithm_names and name not in names
        ]
        if recommended_omitted:
            warnings.append(
                "Recommended submitted comparators are outside this Portfolio scope: "
                + ", ".join(recommended_omitted)
            )
        recommended_missing = [
            name for name in preset_requirements.recommended_algorithms if name not in stage.algorithm_names
        ]
        if recommended_missing:
            warnings.append(
                "Recommended comparators are not in the submitted stage: "
                + ", ".join(recommended_missing)
            )
        if broad.evidence_profile is EvidenceProfile.EXPLORATORY:
            warnings.append("Exploratory evidence does not establish publication sufficiency.")

        intrinsic = {
            "hard_minimum_runs": int(minimum_runs),
            "minimum_algorithms": int(minimum_algorithms),
            "minimum_benchmark_blocks": int(minimum_blocks),
            "required_algorithm_names": sorted(required_algorithms),
            "robust_scenario_required": any(
                OUTPUT_REQUIREMENTS[key].robust_only for key in broad.requested_outputs
            ),
            "required_storage_fields": sorted(required_fields),
            "independent_validation_required": bool(
                broad.require_independent_validation
                or any(OUTPUT_REQUIREMENTS[key].requires_validation for key in broad.requested_outputs)
            ),
        }
        goal_id = f"portfolio-goal-{uuid.uuid4().hex}"
        created_at = _utc_now()
        parameter_hashes = {
            name: str(stage.algorithm_parameter_sha256_by_name[name]) for name in names
        }
        content = {
            "schema_version": APPLIED_PORTFOLIO_GOAL_SCHEMA,
            "portfolio_goal_id": goal_id,
            "created_at": created_at,
            "algorithm_stage_id": stage.stage_id,
            "algorithm_stage_sha256": stage.content_sha256,
            "selected_algorithm_names": list(names),
            "selected_algorithm_parameter_sha256_by_name": parameter_hashes,
            **_broad_portfolio_payload(broad),
            "intrinsic_requirements": intrinsic,
            "intrinsic_warnings": warnings,
        }
        return AppliedPortfolioGoal(
            portfolio_goal_id=goal_id,
            created_at=created_at,
            algorithm_stage_id=stage.stage_id,
            algorithm_stage_sha256=stage.content_sha256,
            selected_algorithm_names=names,
            selected_algorithm_parameter_sha256_by_name=parameter_hashes,
            portfolio=_broad_portfolio_payload(broad),
            intrinsic_requirements=intrinsic,
            intrinsic_warnings=tuple(warnings),
            content_sha256=_content_sha256(content),
        )


@dataclass(frozen=True, slots=True)
class StudyRecommendation:
    recommendation_id: str
    portfolio_goal_id: str
    portfolio_goal_sha256: str
    algorithm_stage_id: str
    algorithm_stage_sha256: str
    hard_minimum_runs: int
    recommended_runs: int
    recommended_case_or_case_matrix: tuple[str, ...]
    minimum_benchmark_blocks: int
    recommended_formulation: dict[str, Any]
    recommended_scenario_mode: str
    recommended_budget: dict[str, Any]
    recommended_storage_fields: tuple[str, ...]
    recommended_validation_tasks: tuple[str, ...]
    default_reuse_and_resume_values: dict[str, Any]
    explanation_by_field: dict[str, str]
    recommendation_sha256: str
    schema_version: str = STUDY_RECOMMENDATION_SCHEMA

    def payload(self) -> dict[str, Any]:
        return {
            "recommendation_schema_version": self.schema_version,
            "recommendation_id": self.recommendation_id,
            "portfolio_goal_id": self.portfolio_goal_id,
            "portfolio_goal_sha256": self.portfolio_goal_sha256,
            "algorithm_stage_id": self.algorithm_stage_id,
            "algorithm_stage_sha256": self.algorithm_stage_sha256,
            "hard_minimum_runs": self.hard_minimum_runs,
            "recommended_runs": self.recommended_runs,
            "recommended_case_or_case_matrix": list(self.recommended_case_or_case_matrix),
            "minimum_benchmark_blocks": self.minimum_benchmark_blocks,
            "recommended_formulation": deepcopy(self.recommended_formulation),
            "recommended_scenario_mode": self.recommended_scenario_mode,
            "recommended_budget": deepcopy(self.recommended_budget),
            "recommended_storage_fields": list(self.recommended_storage_fields),
            "recommended_validation_tasks": list(self.recommended_validation_tasks),
            "default_reuse_and_resume_values": deepcopy(self.default_reuse_and_resume_values),
            "explanation_by_field": deepcopy(self.explanation_by_field),
        }


@dataclass(frozen=True, slots=True)
class AppliedStudySetup:
    study_setup_id: str
    created_at: str
    portfolio_goal_id: str
    portfolio_goal_sha256: str
    recommendation_id: str
    recommendation_sha256: str
    algorithm_stage_id: str
    algorithm_stage_sha256: str
    selected_values: dict[str, Any]
    recommendation_delta: dict[str, Any]
    satisfied_hard_constraints: dict[str, Any]
    deviation_warnings: tuple[str, ...]
    concrete_cells: tuple[dict[str, Any], ...]
    queue_task_count: int
    content_sha256: str
    schema_version: str = APPLIED_STUDY_SETUP_SCHEMA

    def content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_setup_id": self.study_setup_id,
            "created_at": self.created_at,
            "portfolio_goal_id": self.portfolio_goal_id,
            "portfolio_goal_sha256": self.portfolio_goal_sha256,
            "recommendation_id": self.recommendation_id,
            "recommendation_sha256": self.recommendation_sha256,
            "algorithm_stage_id": self.algorithm_stage_id,
            "algorithm_stage_sha256": self.algorithm_stage_sha256,
            "selected_values": deepcopy(self.selected_values),
            "recommendation_delta": deepcopy(self.recommendation_delta),
            "satisfied_hard_constraints": deepcopy(self.satisfied_hard_constraints),
            "deviation_warnings": list(self.deviation_warnings),
            "concrete_cells": deepcopy(list(self.concrete_cells)),
            "queue_task_count": self.queue_task_count,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "AppliedStudySetup":
        content = deepcopy(dict(record.get("content", record)))
        return cls(
            study_setup_id=str(content["study_setup_id"]),
            created_at=str(content["created_at"]),
            portfolio_goal_id=str(content["portfolio_goal_id"]),
            portfolio_goal_sha256=str(content["portfolio_goal_sha256"]),
            recommendation_id=str(content["recommendation_id"]),
            recommendation_sha256=str(content["recommendation_sha256"]),
            algorithm_stage_id=str(content["algorithm_stage_id"]),
            algorithm_stage_sha256=str(content["algorithm_stage_sha256"]),
            selected_values=deepcopy(dict(content["selected_values"])),
            recommendation_delta=deepcopy(dict(content["recommendation_delta"])),
            satisfied_hard_constraints=deepcopy(dict(content["satisfied_hard_constraints"])),
            deviation_warnings=tuple(content.get("deviation_warnings", [])),
            concrete_cells=tuple(deepcopy(content.get("concrete_cells", []))),
            queue_task_count=int(content.get("queue_task_count", 0)),
            content_sha256=str(record.get("content_sha256", content.get("content_sha256", ""))),
            schema_version=str(content.get("schema_version", APPLIED_STUDY_SETUP_SCHEMA)),
        )


class WorkspaceStudyPlanner:
    """Recommend and validate concrete Study choices against one immutable Portfolio goal."""

    @staticmethod
    def _validate_binding(goal: AppliedPortfolioGoal, stage) -> None:
        if stage is None:
            raise ValueError("The submitted algorithm stage is missing")
        if (
            goal.algorithm_stage_id != stage.stage_id
            or goal.algorithm_stage_sha256 != stage.content_sha256
        ):
            raise ValueError(
                f"Portfolio goal {goal.portfolio_goal_id!r} is bound to stage "
                f"{goal.algorithm_stage_id!r} ({goal.algorithm_stage_sha256[:16]}…), but the "
                f"current stage is {stage.stage_id!r} ({stage.content_sha256[:16]}…). "
                "Apply a new Portfolio goal for the current submitted stage."
            )

    @classmethod
    def recommend(cls, goal: AppliedPortfolioGoal, stage, config) -> StudyRecommendation:
        cls._validate_binding(goal, stage)
        minimum = int(goal.intrinsic_requirements["hard_minimum_runs"])
        profile = EvidenceProfile(str(goal.portfolio["evidence_profile"]))
        profile_runs = {
            EvidenceProfile.DIAGNOSTIC: 1,
            EvidenceProfile.EXPLORATORY: 10,
            EvidenceProfile.JOURNAL: 30,
            EvidenceProfile.TRANSACTIONS: 50,
            EvidenceProfile.CUSTOM: minimum,
        }[profile]
        recommended_runs = max(minimum, profile_runs)
        cases = tuple(
            dict.fromkeys(
                str(value)
                for value in (getattr(config, "study_case_plan", None) or [])
                if str(value).strip()
            )
        ) or (str(config.case_name),)
        min_blocks = int(goal.intrinsic_requirements["minimum_benchmark_blocks"])
        scenario_mode = (
            "robust"
            if bool(goal.intrinsic_requirements["robust_scenario_required"])
            else str(getattr(config.scenarios, "mode", "deterministic"))
        )
        validation_tasks = (
            ("independent_validation",)
            if bool(goal.intrinsic_requirements["independent_validation_required"])
            else ()
        )
        seed_payload = {
            "recommendation_schema_version": STUDY_RECOMMENDATION_SCHEMA,
            "portfolio_goal_id": goal.portfolio_goal_id,
            "portfolio_goal_sha256": goal.content_sha256,
            "algorithm_stage_id": stage.stage_id,
            "algorithm_stage_sha256": stage.content_sha256,
            "hard_minimum_runs": minimum,
            "recommended_runs": recommended_runs,
            "recommended_case_or_case_matrix": list(cases),
            "minimum_benchmark_blocks": min_blocks,
            "recommended_formulation": {
                "objective": deepcopy(config.to_dict().get("objective", {})),
                "power_flow": deepcopy(config.to_dict().get("power_flow", {})),
            },
            "recommended_scenario_mode": scenario_mode,
            "recommended_budget": {
                "population_size": int(config.population_size),
                "max_evaluations": int(config.budget.max_evaluations),
                "master_seed": int(config.master_seed),
            },
            "recommended_storage_fields": list(
                goal.intrinsic_requirements["required_storage_fields"]
            ),
            "recommended_validation_tasks": list(validation_tasks),
            "default_reuse_and_resume_values": {
                "reuse_compatible_results": True,
                "resume_enabled": True,
                "checkpoint_interval_evaluations": int(
                    getattr(config, "checkpoint_interval_evaluations", 500)
                ),
            },
            "explanation_by_field": {
                "runs": (
                    f"Hard minimum {minimum} from requested outputs; recommended "
                    f"{recommended_runs} from the {profile.value} evidence profile."
                ),
                "cases": f"At least {min_blocks} benchmark block(s) are required.",
                "scenarios": (
                    "Robust scenarios are required by a requested deliverable."
                    if scenario_mode == "robust"
                    else "The Portfolio goal does not require robust scenario evidence."
                ),
                "budget": "Current compatible formulation budget is proposed without changing it.",
            },
        }
        digest = _content_sha256(seed_payload)
        recommendation_id = f"study-recommendation-{digest[:24]}"
        payload = {**seed_payload, "recommendation_id": recommendation_id}
        recommendation_sha = _content_sha256(payload)
        return StudyRecommendation(
            recommendation_id=recommendation_id,
            portfolio_goal_id=goal.portfolio_goal_id,
            portfolio_goal_sha256=goal.content_sha256,
            algorithm_stage_id=stage.stage_id,
            algorithm_stage_sha256=stage.content_sha256,
            hard_minimum_runs=minimum,
            recommended_runs=recommended_runs,
            recommended_case_or_case_matrix=cases,
            minimum_benchmark_blocks=min_blocks,
            recommended_formulation=deepcopy(seed_payload["recommended_formulation"]),
            recommended_scenario_mode=scenario_mode,
            recommended_budget=deepcopy(seed_payload["recommended_budget"]),
            recommended_storage_fields=tuple(seed_payload["recommended_storage_fields"]),
            recommended_validation_tasks=validation_tasks,
            default_reuse_and_resume_values=deepcopy(
                seed_payload["default_reuse_and_resume_values"]
            ),
            explanation_by_field=deepcopy(seed_payload["explanation_by_field"]),
            recommendation_sha256=recommendation_sha,
        )

    @classmethod
    def apply_selection(
        cls,
        goal: AppliedPortfolioGoal,
        stage,
        recommendation: StudyRecommendation,
        selected_values: dict[str, Any],
    ) -> AppliedStudySetup:
        cls._validate_binding(goal, stage)
        if (
            recommendation.portfolio_goal_id != goal.portfolio_goal_id
            or recommendation.portfolio_goal_sha256 != goal.content_sha256
            or recommendation.algorithm_stage_id != stage.stage_id
            or recommendation.algorithm_stage_sha256 != stage.content_sha256
        ):
            raise ValueError("Refresh recommendations for the exact current Portfolio goal and stage")
        selected = deepcopy(dict(selected_values))
        runs = int(selected.get("runs", 0))
        if runs < recommendation.hard_minimum_runs:
            raise ValueError(
                f"Selected paired runs {runs} is below the Portfolio hard minimum "
                f"{recommendation.hard_minimum_runs} required by the requested evidence"
            )
        cases = tuple(str(value) for value in selected.get("study_case_plan", []) if str(value))
        if len(cases) < recommendation.minimum_benchmark_blocks:
            raise ValueError(
                f"Selected Study contains {len(cases)} benchmark block(s), below the Portfolio "
                f"hard minimum {recommendation.minimum_benchmark_blocks}"
            )
        scenario_mode = str(selected.get("scenario_mode", "deterministic"))
        if (
            bool(goal.intrinsic_requirements["robust_scenario_required"])
            and scenario_mode == "deterministic"
        ):
            raise ValueError("The applied Portfolio goal requires a robust scenario Study")
        run_delta = runs - recommendation.recommended_runs
        warnings: list[str] = []
        if run_delta < 0:
            warnings.append(
                f"Selected runs are {-run_delta} below the soft recommendation but satisfy the hard minimum."
            )
        elif run_delta > 0:
            warnings.append(
                f"Selected runs are {run_delta} above the recommendation; the Portfolio target remains satisfied."
            )
        delta = {
            "runs": run_delta,
            "cases_changed": list(cases) != list(recommendation.recommended_case_or_case_matrix),
            "scenario_mode_changed": scenario_mode != recommendation.recommended_scenario_mode,
            "budget_changed": {
                key: selected.get(key) != recommendation.recommended_budget.get(key)
                for key in ("population_size", "max_evaluations", "master_seed")
            },
            "reuse_and_resume_changed": {
                key: selected.get(key)
                != recommendation.default_reuse_and_resume_values.get(key)
                for key in (
                    "reuse_compatible_results",
                    "resume_enabled",
                    "checkpoint_interval_evaluations",
                )
            },
        }
        constraints = {
            "paired_runs": {
                "selected": runs,
                "minimum": recommendation.hard_minimum_runs,
                "satisfied": True,
            },
            "benchmark_blocks": {
                "selected": len(cases),
                "minimum": recommendation.minimum_benchmark_blocks,
                "satisfied": True,
            },
            "robust_scenario": {
                "required": bool(goal.intrinsic_requirements["robust_scenario_required"]),
                "selected_mode": scenario_mode,
                "satisfied": True,
            },
        }
        setup_id = f"study-setup-{uuid.uuid4().hex}"
        created_at = _utc_now()
        concrete_cells = tuple(
            {"ordinal": ordinal, "case_name": case_name}
            for ordinal, case_name in enumerate(cases)
        )
        queue_task_count = len(concrete_cells) * runs * len(goal.selected_algorithm_names)
        content = {
            "schema_version": APPLIED_STUDY_SETUP_SCHEMA,
            "study_setup_id": setup_id,
            "created_at": created_at,
            "portfolio_goal_id": goal.portfolio_goal_id,
            "portfolio_goal_sha256": goal.content_sha256,
            "recommendation_id": recommendation.recommendation_id,
            "recommendation_sha256": recommendation.recommendation_sha256,
            "algorithm_stage_id": stage.stage_id,
            "algorithm_stage_sha256": stage.content_sha256,
            "selected_values": selected,
            "recommendation_delta": delta,
            "satisfied_hard_constraints": constraints,
            "deviation_warnings": warnings,
            "concrete_cells": list(concrete_cells),
            "queue_task_count": queue_task_count,
        }
        return AppliedStudySetup(
            study_setup_id=setup_id,
            created_at=created_at,
            portfolio_goal_id=goal.portfolio_goal_id,
            portfolio_goal_sha256=goal.content_sha256,
            recommendation_id=recommendation.recommendation_id,
            recommendation_sha256=recommendation.recommendation_sha256,
            algorithm_stage_id=stage.stage_id,
            algorithm_stage_sha256=stage.content_sha256,
            selected_values=selected,
            recommendation_delta=delta,
            satisfied_hard_constraints=constraints,
            deviation_warnings=tuple(warnings),
            concrete_cells=concrete_cells,
            queue_task_count=queue_task_count,
            content_sha256=_content_sha256(content),
        )
