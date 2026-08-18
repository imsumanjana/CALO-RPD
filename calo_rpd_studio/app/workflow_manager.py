"""Key-based, policy-first scientific workflow for CALO-RPD v6.2 beta."""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal

from calo_rpd_studio.algorithms.calo.policy_readiness import governing_policy_user_message

from .workspaces import workspace_index_for_key, workspace_key_for_index


@dataclass(frozen=True, slots=True)
class WorkflowDescriptor:
    key: str
    workspace_key: str
    title: str
    instruction: str

    @property
    def workspace_index(self) -> int:
        """Compatibility view for old callers; v6 logic is key-based."""
        return workspace_index_for_key(self.workspace_key)


SETUP_STEPS = (
    WorkflowDescriptor(
        "algorithms",
        "algorithms",
        "Select optimization algorithms",
        "Choose the comparison algorithms and submit their declared parameter configuration.",
    ),
    WorkflowDescriptor(
        "calo_intelligence",
        "calo_intelligence",
        "Choose the CALO policy mode",
        "Select a verified compatible TSH-CALO policy for policy-guided experiments, or continue using rule-based CALO where available.",
    ),
    WorkflowDescriptor(
        "power_system",
        "power_system",
        "Validate the power system",
        "Load a case, run the base AC power flow, then complete the independent PYPOWER cross-check.",
    ),
    WorkflowDescriptor(
        "orpd",
        "orpd",
        "Define the ORPD formulation",
        "Apply the common objective, decision variables, mixed-variable decoding, and constraint policy.",
    ),
    WorkflowDescriptor(
        "portfolio",
        "portfolio",
        "Plan the evidence portfolio",
        "Choose the evidence outputs and repeated-run protocol immediately after algorithm selection. Availability is recalculated against the final scenario configuration before execution.",
    ),
    WorkflowDescriptor(
        "scenarios",
        "scenarios",
        "Configure operating scenarios",
        "Apply deterministic or robust scenario settings and the declared robust aggregation rule.",
    ),
    WorkflowDescriptor(
        "experiment",
        "experiment",
        "Run or resume the experiment",
        "Revalidate the evidence plan against the final formulation/scenarios, audit fairness, reuse exact completed work, and execute only unfinished jobs.",
    ),
)


class WorkflowManager(QObject):
    """Controls workspace availability using stable keys instead of stack positions."""

    changed = pyqtSignal()
    navigation_requested = pyqtSignal(str)

    def __init__(self, state) -> None:
        super().__init__()
        self.state = state
        self.completed: set[str] = set()
        self.individual_completed: set[str] = set()
        self.experiment_started = False
        self.experiment_completed = False
        self.statistics_completed = False
        self.results_reviewed = False
        self.verified_results = 0
        self.governing_policy_sha = ""

    @staticmethod
    def _normalise_completed_key(key: str) -> str:
        return "calo_intelligence" if str(key) == "calo" else str(key)

    def reset(self) -> None:
        self.completed.clear()
        self.individual_completed.clear()
        self.experiment_started = False
        self.experiment_completed = False
        self.statistics_completed = False
        self.results_reviewed = False
        self.verified_results = 0
        self.governing_policy_sha = ""
        self.changed.emit()

    def _invalidate_after(self, key: str) -> None:
        key = self._normalise_completed_key(key)
        order = [step.key for step in SETUP_STEPS]
        if key not in order:
            return
        position = order.index(key)
        self.completed.difference_update(order[position + 1 :])
        self.experiment_started = False
        self.experiment_completed = False
        self.statistics_completed = False
        self.results_reviewed = False
        self.verified_results = 0

    def invalidate_from(self, key: str) -> None:
        key = self._normalise_completed_key(key)
        self.completed.discard(key)
        self._invalidate_after(key)
        self.changed.emit()

    def mark_completed(self, key: str) -> None:
        key = self._normalise_completed_key(key)
        self._invalidate_after(key)
        self.completed.add(key)
        self.changed.emit()

    def invalidate_individual_from(self, key: str) -> None:
        """Invalidate only the direct Individual setup sequence."""

        value = self._normalise_completed_key(key)
        order = ("power_system", "orpd", "scenarios")
        if value not in order:
            return
        position = order.index(value)
        self.individual_completed.difference_update(order[position:])
        self.changed.emit()

    def mark_individual_completed(self, key: str) -> None:
        """Complete one direct Individual setup step without satisfying Workspace gates."""

        value = self._normalise_completed_key(key)
        order = ("power_system", "orpd", "scenarios")
        if value not in order:
            raise KeyError(f"Unsupported Individual setup completion: {value}")
        position = order.index(value)
        self.individual_completed.difference_update(order[position + 1 :])
        self.individual_completed.add(value)
        self.changed.emit()

    def _individual_stage_algorithm_names(self) -> tuple[str, ...]:
        control = getattr(self.state, "execution_control", None)
        if control is None:
            return tuple(str(name) for name in getattr(self.state.config, "algorithms", []))
        stage = control.active_stage()
        return tuple(str(name) for name in stage.algorithm_names) if stage else ()

    def _individual_stage_requires_policy(self) -> bool:
        from calo_rpd_studio.algorithms.registry import POLICY_GATED_SPECS

        return any(
            name in POLICY_GATED_SPECS for name in self._individual_stage_algorithm_names()
        )

    def notify_governing_policy_changed(self) -> None:
        """Re-evaluate the governing policy and invalidate downstream bindings on an identity change."""
        status = self.governing_policy_status()
        current_sha = str(status.policy_sha256 or "") if status.ready else ""
        previous_sha = str(self.governing_policy_sha or "")
        if not status.ready:
            self.completed.discard("calo_intelligence")
            self._invalidate_after("calo_intelligence")
            if (
                not self._individual_stage_algorithm_names()
                or self._individual_stage_requires_policy()
            ):
                self.individual_completed.clear()
            self.governing_policy_sha = ""
        else:
            # A different active governing policy changes the scientific controller binding.  Preserve
            # immutable completed experiments in storage, but invalidate the unfinished in-memory setup.
            if previous_sha and current_sha and previous_sha.lower() != current_sha.lower():
                self._invalidate_after("calo_intelligence")
                if self._individual_stage_requires_policy():
                    self.individual_completed.clear()
            self.completed.add("calo_intelligence")
            self.governing_policy_sha = current_sha
        self.changed.emit()

    def governing_policy_status(self):
        return self.state.governing_policy_status()

    def governing_policy_ready(self) -> bool:
        return bool(self.governing_policy_status().ready)

    def mark_experiment_started(self) -> None:
        self.experiment_started = True
        self.experiment_completed = False
        self.statistics_completed = False
        self.results_reviewed = False
        self.verified_results = 0
        self.changed.emit()

    def mark_experiment_completed(self) -> None:
        self.experiment_started = True
        self.experiment_completed = True
        self.statistics_completed = False
        self.results_reviewed = False
        self.changed.emit()

    def mark_experiment_stopped(self) -> None:
        self.experiment_started = True
        self.experiment_completed = False
        self.statistics_completed = False
        self.results_reviewed = False
        # Preserve already verified database evidence. A safe stop must not make valid completed
        # verification work appear to have vanished; refresh/restore may increase this count later.
        self.verified_results = max(0, int(self.verified_results))
        self.changed.emit()

    def mark_statistics_completed(self) -> None:
        self.statistics_completed = True
        self.results_reviewed = False
        self.verified_results = 0
        self.changed.emit()

    def mark_results_reviewed(self) -> None:
        self.results_reviewed = True
        self.verified_results = 0
        self.changed.emit()

    def set_verified_results(self, count: int) -> None:
        self.verified_results = max(0, int(count))
        self.changed.emit()

    def snapshot(self) -> dict:
        completed = set(self.completed)
        if self.governing_policy_ready():
            completed.add("calo_intelligence")
        return {
            "schema_version": 3,
            "completed": sorted(completed),
            "individual_completed": sorted(self.individual_completed),
            "experiment_started": bool(self.experiment_started),
            "experiment_completed": bool(self.experiment_completed),
            "statistics_completed": bool(self.statistics_completed),
            "results_reviewed": bool(self.results_reviewed),
            "verified_results": int(self.verified_results),
            "governing_policy_sha": str(self.governing_policy_status().policy_sha256 or "")
            if self.governing_policy_ready()
            else "",
        }

    def restore(
        self,
        payload: dict | None,
        *,
        infer_experiment: bool = False,
        experiment_completed: bool = False,
        verified_results: int = 0,
        inferred_completed: set[str] | None = None,
    ) -> None:
        data = dict(payload or {})
        valid = {step.key for step in SETUP_STEPS}
        if data:
            restored = {
                self._normalise_completed_key(str(key)) for key in data.get("completed", [])
            }
            self.completed = {key for key in restored if key in valid}
            self.individual_completed = {
                str(key)
                for key in data.get("individual_completed", [])
                if str(key) in {"power_system", "orpd", "scenarios"}
            }
            self.experiment_started = bool(data.get("experiment_started", infer_experiment))
            self.experiment_completed = bool(data.get("experiment_completed", experiment_completed))
            self.statistics_completed = bool(data.get("statistics_completed", False))
            self.results_reviewed = bool(data.get("results_reviewed", False))
            self.verified_results = max(
                int(data.get("verified_results", verified_results)), int(verified_results)
            )
        else:
            restored = {
                self._normalise_completed_key(str(key)) for key in (inferred_completed or set())
            }
            self.completed = {key for key in restored if key in valid}
            self.individual_completed = set()
            self.experiment_started = bool(infer_experiment)
            self.experiment_completed = bool(experiment_completed)
            self.statistics_completed = False
            self.results_reviewed = False
            self.verified_results = max(0, int(verified_results))
        # Governing intelligence is never trusted solely from restored workflow JSON.  If the
        # workspace was saved under a different governing policy SHA, downstream setup must be
        # reconfirmed under the newly active scientific controller.
        saved_sha = str(data.get("governing_policy_sha", "") or "")
        status = self.governing_policy_status()
        if status.ready:
            current_sha = str(status.policy_sha256 or "")
            self.completed.add("calo_intelligence")
            if saved_sha and current_sha and saved_sha.lower() != current_sha.lower():
                self._invalidate_after("calo_intelligence")
                if self._individual_stage_requires_policy():
                    self.individual_completed.clear()
            self.governing_policy_sha = current_sha
        else:
            self.completed.discard("calo_intelligence")
            self._invalidate_after("calo_intelligence")
            if (
                not self._individual_stage_algorithm_names()
                or self._individual_stage_requires_policy()
            ):
                self.individual_completed.clear()
            self.governing_policy_sha = ""
        self.changed.emit()

    def _setup_complete(self, key: str) -> bool:
        key = self._normalise_completed_key(key)
        if key == "calo_intelligence":
            return self.governing_policy_ready()
        if key == "algorithms":
            return self.state.execution_control.active_stage() is not None
        if key == "portfolio":
            stage = self.state.execution_control.active_stage()
            goal = self.state.database.get_active_portfolio_goal()
            return bool(
                stage is not None
                and goal is not None
                and str(goal["algorithm_stage_id"]) == stage.stage_id
                and str(goal["algorithm_stage_sha256"]) == stage.content_sha256
            )
        return key in self.completed

    def workspace_state_key(self, key: str) -> tuple[str, str]:
        """Return visual state/reason for a stable workspace key."""
        key = str(key)
        descriptors = {step.key: step for step in SETUP_STEPS}
        if bool(getattr(self.state, "policy_training_active", False)) and key not in {
            "dashboard",
            "calo_intelligence",
        }:
            return (
                "locked",
                "Policy training is using the scientific workspace. Monitor it on the Dashboard or stop it safely before opening another workflow.",
            )
        if key in {"dashboard", "settings", "benchmark"}:
            return "available", "Always available."
        if key == "calo_intelligence":
            status = self.governing_policy_status()
            if status.ready:
                return (
                    "completed",
                    f"CALO governing intelligence READY · {status.policy_name} · {status.grade}.",
                )
            return "recommended", governing_policy_user_message(status)
        if key == "power_system":
            if not self._setup_complete("calo_intelligence"):
                return (
                    "locked",
                    "Select a verified, compatible TSH-CALO policy first.",
                )
            return (
                ("completed", "Power system validated.")
                if self._setup_complete("power_system")
                else ("recommended", descriptors["power_system"].instruction)
            )
        if key == "orpd":
            if not self._setup_complete("power_system"):
                return "locked", "Complete Power System validation first."
            return (
                ("completed", "ORPD formulation applied.")
                if self._setup_complete("orpd")
                else ("recommended", descriptors["orpd"].instruction)
            )
        if key == "algorithms":
            return (
                ("completed", "Algorithms staged for the experiment.")
                if self._setup_complete("algorithms")
                else (
                    "recommended",
                    "Algorithm selection is available as the first configuration step. "
                    "Submitting it stages the experiment configuration without starting work.",
                )
            )
        if key == "portfolio":
            if not self._setup_complete("algorithms"):
                return "locked", "Submit the algorithm selection first."
            stage = self.state.execution_control.active_stage()
            goal = self.state.database.get_active_portfolio_goal()
            goal_current = bool(
                goal is not None
                and stage is not None
                and str(goal["algorithm_stage_id"]) == stage.stage_id
                and str(goal["algorithm_stage_sha256"]) == stage.content_sha256
            )
            return (
                ("completed", "Evidence portfolio intent planned.")
                if goal_current
                else ("recommended", descriptors["portfolio"].instruction)
            )
        if key == "scenarios":
            if not self._setup_complete("portfolio"):
                return (
                    "locked",
                    "Apply a Portfolio goal first. Portfolio defines what evidence is wanted; "
                    "Study will then recommend how to produce it.",
                )
            return (
                ("completed", "Scenario configuration applied.")
                if self._setup_complete("scenarios")
                else ("recommended", descriptors["scenarios"].instruction)
            )
        if key == "experiment":
            if not self._setup_complete("algorithms"):
                return "locked", "Submit the algorithm selection first."
            stage = self.state.execution_control.active_stage()
            goal = self.state.database.get_active_portfolio_goal()
            if goal is None:
                latest = self.state.database.get_latest_portfolio_goal()
                if latest is not None and str(latest.get("status", "")) == "stale_stage":
                    return (
                        "locked",
                        f"Portfolio goal {str(latest['id'])!r} is stale for the current submitted "
                        "algorithm stage. Apply a new Portfolio goal before creating new Study work.",
                    )
                return (
                    "locked",
                    "Apply a Portfolio goal first. Portfolio defines what evidence is wanted; "
                    "Study will then recommend how to produce it.",
                )
            if (
                stage is None
                or str(goal["algorithm_stage_id"]) != stage.stage_id
                or str(goal["algorithm_stage_sha256"]) != stage.content_sha256
            ):
                return (
                    "locked",
                    "The applied Portfolio goal is stale for the current submitted algorithm "
                    "stage. Apply a new Portfolio goal before creating new Study work.",
                )
            return (
                ("completed", "The experiment execution is complete.")
                if self.experiment_completed
                else ("recommended", descriptors["experiment"].instruction)
            )
        if key == "live_optimization":
            return (
                ("available", "Live optimization telemetry is available.")
                if self.experiment_started
                else ("locked", "Start or resume an experiment first.")
            )
        if key == "statistics":
            if not self.experiment_completed:
                return "locked", "Complete the numerical experiment tasks first."
            return (
                ("completed", "Statistical analysis completed for the selected experiment.")
                if self.statistics_completed
                else (
                    "recommended",
                    "Compute only the statistics required by the experiment's frozen output contract.",
                )
            )
        if key == "results":
            return (
                ("available", "Stored numerical evidence is available for inspection.")
                if self.experiment_completed
                else ("locked", "Complete the experiment before exploring results.")
            )
        if key == "validation":
            return (
                ("available", "Independent validation and fairness audit are available.")
                if self.experiment_completed
                else ("locked", "Complete the experiment before independent validation.")
            )
        if key == "publication":
            return (
                ("available", "Publication export is available subject to strict evidence gates.")
                if self.experiment_completed
                else ("locked", "Complete the experiment before publication export.")
            )
        return "available", "Available."

    def workspace_state(self, workspace: str | int) -> tuple[str, str]:
        key = workspace_key_for_index(workspace) if isinstance(workspace, int) else str(workspace)
        return self.workspace_state_key(key)

    def individual_setup_state_key(self, key: str) -> tuple[str, str]:
        """Return direct Individual setup prerequisites without Workspace Portfolio coupling."""

        value = str(key)
        if value in {"power_system", "orpd"}:
            if value == "power_system":
                stage = self.state.execution_control.active_stage()
                if stage is None:
                    return "locked", "Submit at least one algorithm for experiment use first."
                requires_policy = self._individual_stage_requires_policy()
                if requires_policy and not self._setup_complete("calo_intelligence"):
                    return "locked", "Select a verified, compatible TSH-CALO policy first."
                return (
                    ("completed", "Individual power system validated.")
                    if value in self.individual_completed
                    else ("recommended", "Validate the individual experiment power system.")
                )
            if "power_system" not in self.individual_completed:
                return "locked", "Validate the individual experiment power system first."
            return (
                ("completed", "Individual ORPD formulation applied.")
                if value in self.individual_completed
                else ("recommended", "Apply the individual experiment ORPD formulation.")
            )
        if value != "scenarios":
            raise KeyError(f"Unsupported Individual setup key: {value}")
        if "orpd" not in self.individual_completed:
            return "locked", "Complete the individual ORPD formulation first."
        return (
            ("completed", "Individual scenario configuration applied.")
            if "scenarios" in self.individual_completed
            else (
                "recommended",
                "Choose and apply the scenario configuration for this individual experiment.",
            )
        )

    def is_workspace_enabled(self, workspace: str | int) -> bool:
        return self.workspace_state(workspace)[0] != "locked"

    def next_descriptor(self) -> WorkflowDescriptor | None:
        for descriptor in SETUP_STEPS:
            if descriptor.key == "experiment":
                if self.experiment_started and not self.experiment_completed:
                    return WorkflowDescriptor(
                        "live",
                        "live_optimization",
                        "Monitor or resume the active portfolio",
                        "Follow active jobs in Live Optimization or reopen the retained experiment after a safe pause/interruption.",
                    )
                if not self.experiment_completed:
                    return descriptor
                continue
            if not self._setup_complete(descriptor.key):
                return descriptor
        if (
            self.experiment_completed
            and not self.statistics_completed
            and self.state.config.portfolio.kind.value != "single_run"
        ):
            return WorkflowDescriptor(
                "statistics",
                "statistics",
                "Analyze requested repeated-run evidence",
                "Compute only the statistical outputs selected in Portfolio Manager.",
            )
        if self.experiment_completed and not self.results_reviewed:
            return WorkflowDescriptor(
                "results",
                "results",
                "Review portfolio results",
                "Inspect objective values, feasibility, controls, and stored evidence, then confirm the review.",
            )
        if (
            self.experiment_completed
            and self.verified_results <= 0
            and self.state.config.portfolio.require_independent_validation
        ):
            return WorkflowDescriptor(
                "validation",
                "validation",
                "Validate portfolio results",
                "Bulk-validate all required, not-yet-verified runs. Progress is resumable.",
            )
        if self.experiment_completed:
            return WorkflowDescriptor(
                "publication",
                "publication",
                "Generate the portfolio package",
                "Generate only the selected figures, tables, captions, and reproducibility records; incomplete exports are resumable.",
            )
        return None

    def progress(self) -> tuple[int, int]:
        required = [
            "algorithms",
            "calo_intelligence",
            "power_system",
            "orpd",
            "portfolio",
            "scenarios",
            "experiment",
        ]
        completed = sum(
            1
            for key in required
            if (self.experiment_completed if key == "experiment" else self._setup_complete(key))
        )
        return completed, len(required)
