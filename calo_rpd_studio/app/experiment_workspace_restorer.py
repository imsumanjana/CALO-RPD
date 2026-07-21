"""Central restoration of an existing experiment into the complete GUI workspace."""

from __future__ import annotations

import json

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow


class ExperimentWorkspaceRestorer:
    def __init__(self, state, workflow, pages) -> None:
        self.state = state
        self.workflow = workflow
        self.pages = pages

    def restore(self, experiment_id: str) -> dict:
        row = self.state.database.get_experiment(experiment_id)
        if row is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        # The experiment row preserves the original scientific definition. v5 execution revisions
        # are stored as campaigns, so restore the latest revision configuration for the workspace
        # without destroying the original definition/evidence horizon.
        campaigns = [
            item
            for item in self.state.database.list_campaigns(False)
            if str(item.get("experiment_id", "")) == str(experiment_id)
        ]
        latest_campaign = campaigns[0] if campaigns else None
        config_payload = (
            str(latest_campaign.get("config_json", ""))
            if latest_campaign
            else str(row["config_json"])
        )
        config = ExperimentConfig.from_dict(json.loads(config_payload))
        config.resume_campaign_id = ""
        self.state.config = config
        self.state.current_experiment_id = str(experiment_id)

        policy_binding_status = "not_applicable"
        binding_row = self.state.database.get_experiment_policy_binding(str(experiment_id))
        if binding_row is not None:
            binding = dict(binding_row.get("binding") or {})
            policy_binding_status = "recorded"
            checkpoint = str(
                binding.get("policy_checkpoint", binding_row.get("checkpoint_path", "")) or ""
            )
            expected_sha = str(binding.get("policy_sha256", binding_row.get("sha256", "")) or "")
            if checkpoint and expected_sha:
                try:
                    inspected = self.state.policy_registry.inspect_checkpoint(checkpoint)
                    policy_binding_status = (
                        "verified"
                        if inspected["sha256"].lower() == expected_sha.lower()
                        else "checksum_mismatch"
                    )
                except Exception:
                    # Inspection remains possible even when an old external checkpoint was moved;
                    # resuming new numerical work will still be blocked by strict binding checks.
                    policy_binding_status = "artifact_unavailable"

        case = CaseLoader.load(config.case_name)
        self.state.current_case = case
        try:
            self.state.current_power_flow = run_ac_power_flow(case)
        except Exception:
            self.state.current_power_flow = None
        self.state.update_config()
        self.state.case_changed.emit(case)

        # Rehydrate setup widgets from authoritative scientific config.
        for page in self.pages:
            loader = getattr(page, "load_from_config", None)
            if callable(loader):
                loader(config)
            refresher = getattr(page, "refresh", None)
            if callable(refresher) and page.__class__.__name__ in {
                "RobustScenariosPanel",
                "PortfolioManagerPanel",
                "ExperimentManagerPanel",
            }:
                refresher()

        power_page = self.pages[2]
        restore_case = getattr(power_page, "restore_case_state", None)
        if callable(restore_case):
            restore_case(case, self.state.current_power_flow)

        # Reconstruct historical graphs/results from stored numeric evidence.
        live = self.pages[8]
        live.load_experiment(str(experiment_id))
        for index in (9, 10, 11, 12):
            page = self.pages[index]
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
        self.workflow.restore(
            workspace.get("workflow"),
            infer_experiment=True,
            experiment_completed=completed,
            verified_results=verified,
        )
        return {
            "experiment_id": str(experiment_id),
            "campaign_status": campaign_status,
            "ui": dict(workspace.get("ui") or {}),
            "runs": len(self.state.database.list_runs(str(experiment_id))),
            "policy_binding_status": policy_binding_status,
            "revisions": self.state.database.list_experiment_revisions(str(experiment_id)),
        }
