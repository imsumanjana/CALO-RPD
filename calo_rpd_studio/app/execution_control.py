"""Transactional ownership for Workspace and individual experiment execution."""

from __future__ import annotations

from copy import deepcopy
import json
import uuid

from calo_rpd_studio.experiments.execution_plans import (
    AlgorithmStage,
    ControllerKind,
    ExecutionLifecycle,
    ExecutionPlanKind,
    INDIVIDUAL_PLAN_SCHEMA,
    IndividualExperimentPlan,
    WorkspaceStudyPlan,
    audit_receipt_payload,
    resume_contract_sha256,
)


class ExecutionControlService:
    """Own plan identity and lifecycle without owning optimizer or policy behavior."""

    def __init__(self, database) -> None:
        self.database = database
        self.instance_id = f"app-instance-{uuid.uuid4().hex}"
        self.database.recover_execution_controller(owner_instance_id=self.instance_id)

    @staticmethod
    def _stage_from_row(row: dict | None) -> AlgorithmStage | None:
        if row is None:
            return None
        content = dict(row["content"])
        return AlgorithmStage(
            stage_id=str(row["id"]),
            created_at=str(row["created_at"]),
            algorithm_names=tuple(str(name) for name in content["algorithm_names"]),
            algorithm_parameters=deepcopy(dict(content["algorithm_parameters"])),
            algorithm_parameter_sha256_by_name=deepcopy(
                dict(content.get("algorithm_parameter_sha256_by_name", {}))
            ),
            policy_binding_summary=deepcopy(dict(content.get("policy_binding_summary", {}))),
            policy_binding_sha256=str(content.get("policy_binding_sha256", "")),
            source_provenance=deepcopy(dict(content.get("source_provenance", {}))),
            content_sha256=str(row["content_sha256"]),
            record_sha256=str(row["record_sha256"]),
            schema_version=str(row["schema_version"]),
        )

    def controller(self) -> dict:
        return self.database.get_execution_controller()

    def active_stage(self) -> AlgorithmStage | None:
        return self._stage_from_row(self.database.get_active_algorithm_stage())

    def submit_algorithm_stage(self, config) -> AlgorithmStage:
        stage = AlgorithmStage.create(config)
        self.database.replace_algorithm_stage(stage)
        return stage

    def discard_algorithm_stage(self) -> None:
        self.database.discard_algorithm_stage()

    def create_workspace_draft(
        self,
        config,
        study_algorithm_names: tuple[str, ...],
        *,
        portfolio_goal,
        recommendation,
        applied_study_setup,
    ) -> dict:
        stage = self.active_stage()
        if stage is None:
            raise RuntimeError("Submit at least one algorithm for experiment use first")
        plan = WorkspaceStudyPlan.create(
            config,
            stage,
            study_algorithm_names,
            portfolio_goal=portfolio_goal,
            recommendation=recommendation,
            applied_study_setup=applied_study_setup,
        )
        self.database.create_execution_plan(
            plan,
            plan_kind=ExecutionPlanKind.WORKSPACE.value,
            applied_study_setup=applied_study_setup,
            recommendation=recommendation,
        )
        return self.database.get_execution_plan(plan.plan_id) or {}

    def create_individual_draft(self, config) -> dict:
        stage = self.active_stage()
        if stage is None:
            raise RuntimeError("Submit at least one algorithm for experiment use first")
        plan = IndividualExperimentPlan.create(config, stage)
        self.database.create_execution_plan(
            plan, plan_kind=ExecutionPlanKind.INDIVIDUAL_EXPERIMENT.value
        )
        return self.database.get_execution_plan(plan.plan_id) or {}

    def active_plan(self, kind: ExecutionPlanKind | str) -> dict | None:
        value = kind.value if isinstance(kind, ExecutionPlanKind) else str(kind)
        return self.database.get_active_execution_plan(value)

    def record_audit(self, plan_id: str, audit_payload: dict) -> dict:
        plan = self.database.get_execution_plan(str(plan_id))
        if plan is None:
            raise KeyError(f"Unknown execution plan: {plan_id}")
        receipt = audit_receipt_payload(
            design_sha256=str(plan["design_sha256"]), audit_payload=dict(audit_payload)
        )
        return self.database.set_execution_plan_audited(str(plan_id), receipt)

    def discard_draft(self, plan_id: str) -> dict:
        return self.database.discard_unstarted_execution_plan(
            str(plan_id), message="Unstarted execution plan discarded by the scientist"
        )

    def stage(self, plan_id: str, kind: ExecutionPlanKind | str) -> dict:
        value = kind.value if isinstance(kind, ExecutionPlanKind) else str(kind)
        if value not in {item.value for item in ExecutionPlanKind}:
            raise ValueError(f"Unsupported execution-plan kind: {value}")
        controller = (
            ControllerKind.WORKSPACE.value
            if value == ExecutionPlanKind.WORKSPACE.value
            else ControllerKind.INDIVIDUAL_EXPERIMENT.value
        )
        return self.database.acquire_execution_controller(
            str(plan_id),
            controller_kind=controller,
            owner_instance_id=self.instance_id,
            resume=False,
        )

    def resume(self, plan_id: str, kind: ExecutionPlanKind | str) -> dict:
        value = kind.value if isinstance(kind, ExecutionPlanKind) else str(kind)
        if value not in {item.value for item in ExecutionPlanKind}:
            raise ValueError(f"Unsupported execution-plan kind: {value}")
        controller = (
            ControllerKind.WORKSPACE.value
            if value == ExecutionPlanKind.WORKSPACE.value
            else ControllerKind.INDIVIDUAL_EXPERIMENT.value
        )
        current = self.controller()
        plan = self.database.get_execution_plan(str(plan_id))
        if plan is None:
            raise KeyError(f"Unknown execution plan: {plan_id}")
        campaign_id = str(plan.get("campaign_id", "") or "")
        if campaign_id:
            self.verify_campaign_binding(str(plan_id), campaign_id)
        if str(current["controller"]) == controller and str(current["owner_plan_id"]) == str(
            plan_id
        ):
            stage = self.database.get_active_algorithm_stage()
            if (
                stage is None
                or str(stage["id"]) != str(plan["algorithm_stage_id"])
                or str(stage["content_sha256"])
                != str(plan["design"].get("algorithm_stage_sha256", ""))
            ):
                raise RuntimeError(
                    "The resumable plan no longer matches the active algorithm stage"
                )
            return self.transition(
                plan_id,
                expected=(
                    ExecutionLifecycle.PAUSED.value,
                    ExecutionLifecycle.INTERRUPTED_RESUMABLE.value,
                ),
                new_state=ExecutionLifecycle.RUNNING.value,
                message=(
                    "Workspace campaign resumed from its authenticated interrupted state"
                    if value == ExecutionPlanKind.WORKSPACE.value
                    else "Individual experiment resumed from its authenticated campaign state"
                ),
                campaign_id=str(plan.get("campaign_id", "") or ""),
            )
        return self.database.acquire_execution_controller(
            str(plan_id),
            controller_kind=controller,
            owner_instance_id=self.instance_id,
            resume=True,
        )

    def transition(
        self,
        plan_id: str,
        *,
        expected: tuple[str, ...],
        new_state: str,
        message: str,
        campaign_id: str = "",
        release_controller: bool = False,
    ) -> dict:
        controller = self.controller()
        if str(controller["owner_plan_id"]) != str(plan_id):
            raise RuntimeError(
                f"Plan {plan_id!r} does not own execution control; current owner is "
                f"{str(controller['owner_plan_id'])!r}"
            )
        if str(controller["owner_instance_id"]) != self.instance_id:
            raise RuntimeError(
                "This application instance does not hold the current execution-controller fencing token"
            )
        return self.database.transition_execution_plan(
            str(plan_id),
            controller_epoch=int(controller["epoch"]),
            expected_states=tuple(expected),
            new_state=str(new_state),
            message=str(message),
            campaign_id=str(campaign_id),
            release_controller=bool(release_controller),
        )

    def begin_run(self, plan_id: str, *, campaign_id: str = "") -> dict:
        plan = self.database.get_execution_plan(str(plan_id))
        if plan is None:
            raise KeyError(f"Unknown execution plan: {plan_id}")
        if str(plan["lifecycle_state"]) == ExecutionLifecycle.RUNNING.value:
            return plan
        return self.transition(
            plan_id,
            expected=(ExecutionLifecycle.STAGED.value,),
            new_state=ExecutionLifecycle.RUNNING.value,
            message="Numerical execution started from the immutable audited plan",
            campaign_id=campaign_id,
        )

    def request_pause(self, plan_id: str) -> dict:
        return self.transition(
            plan_id,
            expected=(ExecutionLifecycle.RUNNING.value,),
            new_state=ExecutionLifecycle.PAUSING.value,
            message="Safe pause latched; no new jobs may be admitted",
        )

    def commit_paused(self, plan_id: str, *, campaign_id: str = "") -> dict:
        plan = self.database.get_execution_plan(str(plan_id))
        if plan is None:
            raise KeyError(f"Unknown execution plan: {plan_id}")
        workspace = str(plan["plan_kind"]) == ExecutionPlanKind.WORKSPACE.value
        return self.transition(
            plan_id,
            expected=(ExecutionLifecycle.PAUSING.value, ExecutionLifecycle.RUNNING.value),
            new_state=ExecutionLifecycle.PAUSED.value,
            message=(
                "Workspace campaign paused durably; individual execution is now permitted"
                if workspace
                else "Individual experiment paused durably; Workspace resume remains blocked"
            ),
            campaign_id=campaign_id,
            release_controller=workspace,
        )

    def commit_terminal(
        self,
        plan_id: str,
        *,
        lifecycle: ExecutionLifecycle | str,
        message: str,
        campaign_id: str = "",
    ) -> dict:
        value = lifecycle.value if isinstance(lifecycle, ExecutionLifecycle) else str(lifecycle)
        if value not in {
            ExecutionLifecycle.COMPLETED.value,
            ExecutionLifecycle.COMPLETED_WITH_FAILURES.value,
            ExecutionLifecycle.CANCELLED.value,
            ExecutionLifecycle.FAILED_NON_RESUMABLE.value,
            ExecutionLifecycle.DISCARDED_UNSTARTED.value,
        }:
            raise ValueError(f"Lifecycle {value!r} is not terminal")
        plan = self.database.get_execution_plan(str(plan_id))
        if plan is None:
            raise KeyError(f"Unknown execution plan: {plan_id}")
        return self.transition(
            plan_id,
            expected=(
                ExecutionLifecycle.STAGED.value,
                ExecutionLifecycle.RUNNING.value,
                ExecutionLifecycle.PAUSING.value,
                ExecutionLifecycle.PAUSED.value,
                ExecutionLifecycle.INTERRUPTED_RESUMABLE.value,
            ),
            new_state=value,
            message=str(message),
            campaign_id=campaign_id,
            release_controller=True,
        )

    def cancel_retained(self, plan_id: str, *, message: str) -> dict:
        """Terminally close an idle/staged plan and every unfinished retained ledger row."""

        controller = self.controller()
        return self.database.cancel_retained_execution_plan(
            str(plan_id),
            controller_epoch=int(controller["epoch"]),
            owner_instance_id=self.instance_id,
            message=str(message),
        )

    def plan_configuration(self, plan_id: str, *, cell_id: str = ""):
        from calo_rpd_studio.experiments.experiment_config import ExperimentConfig

        plan = self.database.get_execution_plan(str(plan_id))
        if plan is None:
            raise KeyError(f"Unknown execution plan: {plan_id}")
        if cell_id:
            cells = {
                str(row["id"]): row for row in self.database.list_workspace_plan_cells(str(plan_id))
            }
            if str(cell_id) not in cells:
                raise KeyError(f"Unknown Workspace plan cell: {cell_id}")
            payload = cells[str(cell_id)]["config"]
        else:
            payload = dict(plan["design"]["config"])
        config = ExperimentConfig.from_dict(deepcopy(payload))
        config.execution_plan_id = str(plan["id"])
        config.execution_plan_design_sha256 = str(plan["design_sha256"])
        config.algorithm_stage_id = str(plan["algorithm_stage_id"])
        config.workspace_plan_cell_id = str(cell_id)
        current_plan_contract = (
            str(plan["plan_kind"]) != ExecutionPlanKind.INDIVIDUAL_EXPERIMENT.value
            or str(plan.get("schema_version", "")) == INDIVIDUAL_PLAN_SCHEMA
        )
        config.execution_plan_kind = str(plan["plan_kind"]) if current_plan_contract else ""
        return config

    def verify_campaign_binding(self, plan_id: str, campaign_id: str) -> None:
        """Reject resume unless the stored campaign matches the exact frozen plan/cell."""

        plan = self.database.get_execution_plan(str(plan_id))
        campaign = self.database.get_campaign(str(campaign_id))
        if plan is None or campaign is None:
            raise RuntimeError("The resumable plan or campaign record is missing")
        stored_payload = json.loads(str(campaign["config_json"]))
        if str(stored_payload.get("execution_plan_id", "")) != str(plan_id):
            raise RuntimeError("The retained campaign belongs to a different execution plan")
        cell_id = str(stored_payload.get("workspace_plan_cell_id", "") or "")
        expected_payload = self.plan_configuration(str(plan_id), cell_id=cell_id).to_dict()
        # Explicit migration for campaigns created before plan-kind/result-contract provenance was
        # serialized. Old evidence remains readable; current v2 individual plans must carry both.
        if "execution_plan_kind" not in stored_payload:
            stored_payload["execution_plan_kind"] = str(
                expected_payload.get("execution_plan_kind", "")
            )
        if "result_contract" not in stored_payload:
            stored_payload["result_contract"] = {}
        if resume_contract_sha256(stored_payload) != resume_contract_sha256(expected_payload):
            raise RuntimeError(
                "The retained campaign does not match the frozen plan subset, seed, budget, "
                "case, formulation, scenario, compute intent, or policy binding"
            )

    def current_owner_summary(self) -> tuple[str, str, str]:
        controller = self.controller()
        return (
            str(controller["controller"]),
            str(controller["owner_plan_id"]),
            str(controller["lifecycle_state"]),
        )
