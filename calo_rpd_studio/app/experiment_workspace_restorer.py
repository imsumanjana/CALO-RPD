"""Central restoration of an existing experiment into the complete GUI workspace."""

from __future__ import annotations

import json
import logging
from typing import Any

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow

_LOG = logging.getLogger(__name__)


class ExperimentWorkspaceRestorer:
    """Restore the explicit latest scientific revision, never whichever campaign was updated last."""

    def __init__(self, state, workflow, pages) -> None:
        self.state = state
        self.workflow = workflow
        self.pages = pages
        self._page_by_name = {page.__class__.__name__: page for page in pages}

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
            config = ExperimentConfig.from_dict(candidates[-1][1])
            return config, target_id
        base_payload = json.loads(str(row.get("config_json", "{}") or "{}"))
        config = ExperimentConfig.from_dict(base_payload)
        # Never silently claim a different revision. If the latest revision has no matching campaign
        # configuration, restore the immutable original definition and surface that fact.
        return config, str(config.experiment_revision_id or "")

    def _page(self, class_name: str) -> Any | None:
        return self._page_by_name.get(class_name)

    def restore(self, experiment_id: str) -> dict:
        row = self.state.database.get_experiment(experiment_id)
        if row is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        row = dict(row)
        revision = self._select_revision(str(experiment_id))
        config, restored_revision_id = self._config_for_revision(str(experiment_id), row, revision)
        config.resume_campaign_id = ""
        config.validate()
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

        case = CaseLoader.load(config.case_name)
        self.state.current_case = case
        # Restoration must use the exact persisted solver options. A failed PF blocks scientific
        # restoration instead of unlocking downstream pages with current_power_flow=None.
        power_flow = run_ac_power_flow(case, config.power_flow)
        if not bool(power_flow.converged):
            raise RuntimeError(
                f"Cannot restore experiment {experiment_id}: base AC power flow did not converge "
                f"with the saved PowerFlowOptions."
            )
        self.state.current_power_flow = power_flow
        self.state.update_config()
        self.state.case_changed.emit(case)

        for page in self.pages:
            loader = getattr(page, "load_from_config", None)
            if callable(loader):
                loader(config)
            refresher = getattr(page, "refresh", None)
            if callable(refresher) and page.__class__.__name__ in {
                "RobustScenariosPanel", "PortfolioManagerPanel", "ExperimentManagerPanel"
            }:
                refresher()

        power_page = self._page("PowerSystemPanel")
        restore_case = getattr(power_page, "restore_case_state", None) if power_page is not None else None
        if callable(restore_case):
            restore_case(case, power_flow)

        live = self._page("LiveOptimizationPanel")
        if live is not None:
            live.load_experiment(str(experiment_id))
        for class_name in (
            "StatisticalAnalysisPanel", "ResultsExplorerPanel", "ValidationAuditPanel", "PublicationExportPanel"
        ):
            page = self._page(class_name)
            if page is None:
                continue
            fn = getattr(page, "refresh_experiments", None) or getattr(page, "refresh", None)
            if callable(fn):
                fn()
            selector = getattr(page, "select_experiment", None)
            if callable(selector):
                selector(str(experiment_id))

        workspace = self.state.database.get_workspace_state(str(experiment_id)) or {}
        campaign_status = str(row.get("campaign_status", "completed"))
        completed = campaign_status in {"completed", "complete"}
        verified = len(self.state.database.list_runs(str(experiment_id), verified_only=True))
        # Legacy rows without workspace state are inferred conservatively only after config validation
        # and a successful exact-option PF restore. CALO is complete only with a verified binding.
        inferred_completed = {"power_system", "orpd", "algorithms", "portfolio", "scenarios"}
        if "CALO" in config.algorithms and policy_binding_status == "verified":
            inferred_completed.add("calo")
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
            "ui": dict(workspace.get("ui") or {}),
            "runs": len(self.state.database.list_runs(str(experiment_id))),
            "policy_binding_status": policy_binding_status,
            "policy_binding_error": policy_binding_error,
            "restored_revision_id": restored_revision_id,
            "revisions": self.state.database.list_experiment_revisions(str(experiment_id)),
        }
