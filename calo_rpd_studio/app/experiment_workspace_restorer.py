"""Central restoration of an existing experiment into the complete GUI workspace."""

from __future__ import annotations

import json
import logging
import numpy as np
from typing import Any, Mapping

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.app.workspaces import migrate_workspace_ui

_LOG = logging.getLogger(__name__)

class WorkspaceRestoreError(RuntimeError):
    """Structured restoration failure with an explicit stage and experiment identity."""

    def __init__(self, experiment_id: str, stage: str, message: str):
        self.experiment_id = str(experiment_id)
        self.stage = str(stage)
        super().__init__(f"Workspace restore failed at {self.stage}: {message}")

    def as_dict(self) -> dict[str, str]:
        return {
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "error": str(self),
        }



class ExperimentWorkspaceRestorer:
    """Restore the explicit latest scientific revision, never whichever campaign was updated last."""

    def __init__(self, state, workflow, pages) -> None:
        self.state = state
        self.workflow = workflow
        if isinstance(pages, Mapping):
            self.pages_by_key = dict(pages)
            self.pages = list(self.pages_by_key.values())
        else:
            # Compatibility for callers outside MainWindow: derive stable semantic keys only from
            # declared workspace_key attributes. Class-name strings are intentionally not used.
            self.pages = list(pages)
            self.pages_by_key = {
                str(getattr(page, "workspace_key")): page
                for page in self.pages
                if str(getattr(page, "workspace_key", "")).strip()
            }

    def _page(self, workspace_key: str) -> Any | None:
        return self.pages_by_key.get(str(workspace_key))

    def _select_revision(self, experiment_id: str) -> dict | None:
        revisions = self.state.database.list_experiment_revisions(str(experiment_id))
        if not revisions:
            return None
        # Scientific lineage order is the explicit monotonic revision_number. Prefer the latest
        # completed revision; only fall back to the latest declared revision if none completed.
        completed = [r for r in revisions if str(r.get("status", "")).lower() == "completed"]
        pool = completed or revisions
        return max(pool, key=lambda r: int(r.get("revision_number", 0)))

    def _config_for_revision(self, experiment_id: str, row: dict, revision: dict | None) -> tuple[ExperimentConfig, str]:
        target_id = str((revision or {}).get("id", "") or "")
        candidates: list[tuple[int, dict]] = []
        for campaign in self.state.database.list_campaigns(False):
            if str(campaign.get("experiment_id", "")) != str(experiment_id):
                continue
            try:
                payload = json.loads(str(campaign.get("config_json", "") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                _LOG.warning("Ignoring campaign with invalid config JSON during restore: %s", campaign.get("id"))
                continue
            revision_id = str(payload.get("experiment_revision_id", "") or "")
            if target_id and revision_id != target_id:
                continue
            candidates.append((int((revision or {}).get("revision_number", 0)), payload))
        if candidates:
            try:
                config = ExperimentConfig.from_dict(candidates[-1][1])
            except (TypeError, ValueError, KeyError) as exc:
                raise WorkspaceRestoreError(
                    str(experiment_id),
                    "configuration",
                    f"saved revision configuration is incompatible or corrupt: {exc}",
                ) from exc
            return config, target_id
        try:
            base_payload = json.loads(str(row.get("config_json", "{}") or "{}"))
            if not isinstance(base_payload, dict):
                raise TypeError("saved experiment config is not a JSON object")
            config = ExperimentConfig.from_dict(base_payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise WorkspaceRestoreError(
                str(experiment_id), "configuration", f"saved experiment configuration is corrupt: {exc}"
            ) from exc
        # Never silently claim a different revision. If the latest revision has no matching campaign
        # configuration, restore the immutable original definition and surface that fact.
        return config, str(config.experiment_revision_id or "")

    def restore(self, experiment_id: str) -> dict:
        row = self.state.database.get_experiment(experiment_id)
        if row is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        row = dict(row)
        revision = self._select_revision(str(experiment_id))
        config, restored_revision_id = self._config_for_revision(str(experiment_id), row, revision)
        config.resume_campaign_id = ""
        try:
            config.validate()
        except (TypeError, ValueError, KeyError) as exc:
            raise WorkspaceRestoreError(
                str(experiment_id), "configuration", f"saved configuration failed validation: {exc}"
            ) from exc
        self.state.config = config
        self.state.current_experiment_id = str(experiment_id)

        policy_binding_status = "not_applicable"
        policy_binding_error = ""
        binding_row = self.state.database.get_experiment_policy_binding(str(experiment_id))
        if binding_row is not None:
            binding = dict(binding_row.get("binding") or {})
            policy_binding_status = "recorded"
            checkpoint = str(binding.get("policy_checkpoint", binding_row.get("checkpoint_path", "")) or "")
            expected_sha = str(binding.get("policy_sha256", binding_row.get("sha256", "")) or "")
            if checkpoint and expected_sha:
                try:
                    inspected = self.state.policy_registry.inspect_checkpoint(checkpoint)
                    policy_binding_status = (
                        "verified" if inspected["sha256"].lower() == expected_sha.lower() else "checksum_mismatch"
                    )
                    if policy_binding_status == "checksum_mismatch":
                        policy_binding_error = (
                            f"Policy checksum mismatch for {checkpoint}: expected {expected_sha}, "
                            f"found {inspected['sha256']}"
                        )
                except Exception as exc:
                    policy_binding_status = "artifact_unavailable"
                    policy_binding_error = f"{type(exc).__name__}: {exc}; checkpoint={checkpoint}; expected_sha256={expected_sha}"
                    _LOG.error("Policy binding inspection failed during restore: %s", policy_binding_error)

        try:
            case = CaseLoader.load(config.case_name)
        except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
            raise WorkspaceRestoreError(
                str(experiment_id),
                "case_load",
                f"saved case {config.case_name!r} is missing, unreadable, or invalid: {exc}",
            ) from exc
        self.state.current_case = case
        # Restoration must use the exact persisted solver options. A failed PF blocks scientific
        # restoration instead of unlocking downstream pages with current_power_flow=None.
        try:
            power_flow = run_ac_power_flow(case, config.power_flow)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise WorkspaceRestoreError(
                str(experiment_id), "power_flow", f"saved power-flow state could not be reconstructed: {exc}"
            ) from exc
        if not bool(power_flow.converged):
            raise WorkspaceRestoreError(
                str(experiment_id),
                "power_flow",
                "base AC power flow did not converge with the saved PowerFlowOptions",
            )
        self.state.current_power_flow = power_flow
        self.state.update_config()
        self.state.case_changed.emit(case)

        for page in self.pages:
            loader = getattr(page, "load_from_config", None)
            if callable(loader):
                loader(config)
            refresher = getattr(page, "refresh", None)
            if callable(refresher) and page in {
                self._page("scenarios"), self._page("portfolio"), self._page("experiment")
            }:
                refresher()

        power_page = self._page("power_system")
        restore_case = getattr(power_page, "restore_case_state", None) if power_page is not None else None
        if callable(restore_case):
            restore_case(case, power_flow)

        live = self._page("live_optimization")
        if live is not None:
            live.load_experiment(str(experiment_id))
        for workspace_key in ("statistics", "results", "validation", "publication"):
            page = self._page(workspace_key)
            if page is None:
                continue
            fn = getattr(page, "refresh_experiments", None) or getattr(page, "refresh", None)
            if callable(fn):
                fn()
            selector = getattr(page, "select_experiment", None)
            if callable(selector):
                selector(str(experiment_id))

        workspace = self.state.database.get_workspace_state(str(experiment_id)) or {}
        migrated_ui, migration_report = migrate_workspace_ui(dict(workspace.get("ui") or {}))
        campaign_status = str(row.get("campaign_status", "completed"))
        completed = campaign_status in {"completed", "complete"}
        verified = len(self.state.database.list_runs(str(experiment_id), verified_only=True))
        # Legacy rows without workspace state are inferred conservatively only after config validation
        # and a successful exact-option PF restore. v6 never infers governing-intelligence readiness
        # from historical workflow state; WorkflowManager re-evaluates the currently active policy live.
        inferred_completed = {"power_system", "orpd", "algorithms", "portfolio", "scenarios"}
        self.workflow.restore(
            workspace.get("workflow"),
            infer_experiment=True,
            experiment_completed=completed,
            verified_results=verified,
            inferred_completed=inferred_completed,
        )
        return {
            "experiment_id": str(experiment_id),
            "campaign_status": campaign_status,
            "ui": migrated_ui,
            "workspace_migration": migration_report.as_dict(),
            "runs": len(self.state.database.list_runs(str(experiment_id))),
            "policy_binding_status": policy_binding_status,
            "policy_binding_error": policy_binding_error,
            "restored_revision_id": restored_revision_id,
            "revisions": self.state.database.list_experiment_revisions(str(experiment_id)),
        }
