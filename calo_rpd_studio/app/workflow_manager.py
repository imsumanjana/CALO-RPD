"""Guided, invalidation-aware scientific workflow for the GUI."""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass(frozen=True, slots=True)
class WorkflowDescriptor:
    key: str
    workspace_index: int
    title: str
    instruction: str


SETUP_STEPS = (
    WorkflowDescriptor(
        "power_system",
        1,
        "Validate the power system",
        "Load a case, run the base AC power flow, then complete the independent PYPOWER cross-check.",
    ),
    WorkflowDescriptor(
        "orpd",
        2,
        "Define the ORPD formulation",
        "Apply the common objective, decision variables, mixed-variable decoding, and constraint policy.",
    ),
    WorkflowDescriptor(
        "algorithms",
        3,
        "Select optimization algorithms",
        "Choose the comparison algorithms and apply their declared parameter configuration.",
    ),
    WorkflowDescriptor(
        "calo",
        4,
        "Confirm CALO intelligence",
        "Validate the CALO policy checkpoint and apply the frozen evaluation configuration.",
    ),
    WorkflowDescriptor(
        "scenarios",
        5,
        "Configure operating scenarios",
        "Apply deterministic or robust scenario settings and the robust aggregation rule.",
    ),
    WorkflowDescriptor(
        "experiment",
        6,
        "Run the experiment",
        "Audit fairness, then run the comparative experiment or the CALO ablation analysis.",
    ),
)


class WorkflowManager(QObject):
    """Controls workspace availability and invalidates downstream stages after changes."""

    changed = pyqtSignal()
    navigation_requested = pyqtSignal(int)

    def __init__(self, state) -> None:
        super().__init__()
        self.state = state
        self.completed: set[str] = set()
        self.experiment_started = False
        self.experiment_completed = False
        self.statistics_completed = False
        self.results_reviewed = False
        self.verified_results = 0

    def reset(self) -> None:
        self.completed.clear()
        self.experiment_started = False
        self.experiment_completed = False
        self.statistics_completed = False
        self.results_reviewed = False
        self.verified_results = 0
        self.changed.emit()

    def _invalidate_after(self, key: str) -> None:
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
        if key in self.completed:
            self.completed.discard(key)
        self._invalidate_after(key)
        self.changed.emit()

    def mark_completed(self, key: str) -> None:
        self._invalidate_after(key)
        self.completed.add(key)
        self.changed.emit()

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
        self.experiment_started = False
        self.experiment_completed = False
        self.statistics_completed = False
        self.results_reviewed = False
        self.verified_results = 0
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

    def calo_required(self) -> bool:
        return "CALO" in self.state.config.algorithms

    def _setup_complete(self, key: str) -> bool:
        if key == "calo" and not self.calo_required():
            return True
        return key in self.completed

    def workspace_state(self, index: int) -> tuple[str, str]:
        """Return visual state and explanatory tooltip for one workspace."""
        if index in (0, 12):
            return "available", "Always available."
        if index == 1:
            return (
                ("completed", "Power system validated.")
                if self._setup_complete("power_system")
                else ("recommended", SETUP_STEPS[0].instruction)
            )
        if index == 2:
            if not self._setup_complete("power_system"):
                return "locked", "Complete Power System validation first."
            return (
                ("completed", "ORPD formulation applied.")
                if self._setup_complete("orpd")
                else ("recommended", SETUP_STEPS[1].instruction)
            )
        if index == 3:
            if not self._setup_complete("orpd"):
                return "locked", "Apply the ORPD formulation first."
            return (
                ("completed", "Algorithm configuration applied.")
                if self._setup_complete("algorithms")
                else ("recommended", SETUP_STEPS[2].instruction)
            )
        if index == 4:
            if not self._setup_complete("algorithms"):
                return "locked", "Apply the algorithm selection first."
            if not self.calo_required():
                return "optional", "CALO is not selected; this workspace is optional."
            return (
                ("completed", "CALO policy configuration validated.")
                if self._setup_complete("calo")
                else ("recommended", SETUP_STEPS[3].instruction)
            )
        if index == 5:
            if not self._setup_complete("algorithms"):
                return "locked", "Apply the algorithm selection first."
            if self.calo_required() and not self._setup_complete("calo"):
                return "locked", "Validate and apply CALO Intelligence first."
            return (
                ("completed", "Scenario configuration applied.")
                if self._setup_complete("scenarios")
                else ("recommended", SETUP_STEPS[4].instruction)
            )
        if index == 6:
            if not self._setup_complete("scenarios"):
                return "locked", "Apply the robust scenario configuration first."
            return (
                ("completed", "At least one experiment has completed.")
                if self.experiment_completed
                else ("recommended", SETUP_STEPS[5].instruction)
            )
        if index == 7:
            if not self.experiment_started:
                return "locked", "Start an experiment first."
            return "available", "Live optimization telemetry is available."
        if index == 8:
            if not self.experiment_completed:
                return "locked", "Complete an experiment first."
            return (
                ("completed", "Statistical analysis completed for the selected experiment.")
                if self.statistics_completed
                else ("recommended", "Analyze the completed repeated-run experiment before reviewing individual results.")
            )
        if index == 9:
            if not self.statistics_completed:
                return "locked", "Complete Statistical Analysis first."
            return (
                ("completed", "Result review confirmed.")
                if self.results_reviewed
                else ("recommended", "Inspect the stored runs and confirm the result review before independent validation.")
            )
        if index == 10:
            if not self.results_reviewed:
                return "locked", "Complete the Results Explorer review first."
            return "available", "Independent validation is available for stored solutions."
        if index == 11:
            if self.verified_results <= 0:
                return "locked", "Independently verify at least one result first."
            return "available", f"{self.verified_results} verified result(s) are available for export."
        return "available", "Available."

    def is_workspace_enabled(self, index: int) -> bool:
        return self.workspace_state(index)[0] != "locked"

    def next_descriptor(self) -> WorkflowDescriptor | None:
        for descriptor in SETUP_STEPS:
            if descriptor.key == "calo" and not self.calo_required():
                continue
            if descriptor.key == "experiment":
                if self.experiment_started and not self.experiment_completed:
                    return WorkflowDescriptor(
                        "live",
                        7,
                        "Monitor the active optimization",
                        "The experiment is running. Follow objective, feasibility, evaluation count, and CALO telemetry in Live Optimization.",
                    )
                if not self.experiment_completed:
                    return descriptor
                continue
            if not self._setup_complete(descriptor.key):
                return descriptor
        if self.experiment_completed and not self.statistics_completed:
            return WorkflowDescriptor(
                "statistics",
                8,
                "Analyze repeated-run statistics",
                "Compute the statistical summary and convergence comparison before reviewing individual stored runs.",
            )
        if self.experiment_completed and not self.results_reviewed:
            return WorkflowDescriptor(
                "results",
                9,
                "Review experiment results",
                "Inspect objective values, feasibility, controls, and stored run metadata, then confirm the review.",
            )
        if self.experiment_completed and self.verified_results <= 0:
            return WorkflowDescriptor(
                "validation",
                10,
                "Validate experiment results",
                "Independently re-run and verify at least one stored solution before publication export.",
            )
        if self.verified_results > 0:
            return WorkflowDescriptor(
                "publication",
                11,
                "Export verified results",
                "Create the publication and reproducibility package from independently verified records.",
            )
        return None

    def progress(self) -> tuple[int, int]:
        required = ["power_system", "orpd", "algorithms"]
        if self.calo_required():
            required.append("calo")
        required.extend(["scenarios", "experiment"])
        completed = sum(
            1
            for key in required
            if (self.experiment_completed if key == "experiment" else self._setup_complete(key))
        )
        return completed, len(required)
