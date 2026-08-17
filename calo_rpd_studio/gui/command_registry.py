"""Authoritative ribbon metadata and generated Qt actions.

Scientific work remains in its existing services. Most commands navigate; the policy-training
command opens contextual inputs; the visible in-pane action owns the explicit readiness/start
state machine.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction, QKeySequence

from calo_rpd_studio.gui.icons.workspace_icons import workspace_icon


@dataclass(frozen=True, slots=True)
class CommandSpec:
    command_id: str
    category: str
    group: str
    label: str
    icon: str
    tooltip: str
    handler: str = "workspace"
    workspace: str = ""
    context: str = "generic"
    shortcut: str = ""
    primary: bool = False


RIBBON_CATEGORY_ORDER: tuple[str, ...] = (
    "Home",
    "Algorithms",
    "Workspace",
    "Experiment",
    "Compute",
    "Results",
    "Policies",
    "View",
    "Help",
)

INDIVIDUAL_EXPERIMENT_STEP_COMMAND_IDS: tuple[str, ...] = (
    "experiment.power",
    "experiment.formulation",
    "experiment.budget",
    "experiment.scenarios",
    "experiment.validation",
    "experiment.review",
)
INDIVIDUAL_EXPERIMENT_COMMAND_IDS: tuple[str, ...] = (
    "experiment.individual",
    *INDIVIDUAL_EXPERIMENT_STEP_COMMAND_IDS,
)


COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "home.overview",
        "Home",
        "Project",
        "Overview",
        "home",
        "Open the project and readiness overview.",
        workspace="dashboard",
        context="overview",
        primary=True,
    ),
    CommandSpec(
        "project.open",
        "Home",
        "Project",
        "Open",
        "open",
        "Open an experiment configuration.",
        "open",
        shortcut="Ctrl+O",
        primary=True,
    ),
    CommandSpec(
        "project.save",
        "Home",
        "Project",
        "Save",
        "save",
        "Save the current experiment configuration.",
        "save",
        shortcut="Ctrl+S",
    ),
    CommandSpec(
        "workspace.portfolio",
        "Workspace",
        "Study",
        "Portfolio",
        "portfolio",
        "Define and apply the broad evidence goal and comparison scope.",
        workspace="portfolio",
        context="portfolio",
    ),
    CommandSpec(
        "workspace.study",
        "Workspace",
        "Study",
        "Study",
        "study",
        "Review Portfolio recommendations, apply the concrete Study, then audit, stage, and run.",
        workspace="experiment",
        context="workspace_study",
        primary=True,
    ),
    CommandSpec(
        "workspace.validation",
        "Workspace",
        "Evidence",
        "Validate",
        "validation",
        "Open independent result validation and audit controls.",
        workspace="validation",
        context="validation",
    ),
    CommandSpec(
        "workspace.benchmark",
        "Workspace",
        "Evidence",
        "Bench",
        "benchmark",
        "Open benchmark campaign planning and retained evidence.",
        workspace="benchmark",
        context="benchmark",
    ),
    CommandSpec(
        "workspace.publication",
        "Workspace",
        "Evidence",
        "Export",
        "publication",
        "Open publication preview and export controls.",
        workspace="publication",
        context="publication",
    ),
    CommandSpec(
        "workspace.settings",
        "Workspace",
        "System",
        "Settings",
        "settings",
        "Open application appearance and density settings.",
        workspace="settings",
        context="settings",
    ),
    CommandSpec(
        "experiment.individual",
        "Experiment",
        "Execute",
        "Individual experiment",
        "experiment",
        "Configure, audit, stage, and explicitly run one experiment using the complete submitted algorithm stage.",
        workspace="experiment",
        context="individual_experiment",
        primary=True,
    ),
    CommandSpec(
        "experiment.power",
        "Experiment",
        "Setup",
        "Case",
        "network",
        "Open the Case step of the Individual Experiment setup.",
        workspace="experiment",
        context="individual_experiment.case",
    ),
    CommandSpec(
        "experiment.formulation",
        "Experiment",
        "Setup",
        "Formulation",
        "formulation",
        "Open the Formulation step of the Individual Experiment setup.",
        workspace="experiment",
        context="individual_experiment.formulation",
    ),
    CommandSpec(
        "experiment.budget",
        "Experiment",
        "Setup",
        "Budget + runs",
        "compute",
        "Open the Budget + runs step of the Individual Experiment setup.",
        workspace="experiment",
        context="individual_experiment.budget",
    ),
    CommandSpec(
        "experiment.scenarios",
        "Experiment",
        "Setup",
        "Scenarios",
        "scenario",
        "Open the Scenarios step of the Individual Experiment setup.",
        workspace="experiment",
        context="individual_experiment.scenarios",
    ),
    CommandSpec(
        "experiment.validation",
        "Experiment",
        "Setup",
        "Validate + outputs",
        "validation",
        "Open the Validate + outputs step of the Individual Experiment setup.",
        workspace="experiment",
        context="individual_experiment.validation",
    ),
    CommandSpec(
        "experiment.review",
        "Experiment",
        "Setup",
        "Review + launch",
        "run",
        "Open the Review + launch step of the Individual Experiment setup without starting it.",
        workspace="experiment",
        context="individual_experiment.review",
    ),
    CommandSpec(
        "algorithms.configure",
        "Algorithms",
        "Configuration",
        "Algorithms",
        "algorithm",
        "Select and submit the algorithms and comparator parameters staged for new experiments.",
        workspace="algorithms",
        context="algorithms",
        primary=True,
    ),
    CommandSpec(
        "algorithms.calo",
        "Algorithms",
        "CALO",
        "CALO intelligence",
        "policy",
        "Inspect rule-only CALO and governing-policy readiness.",
        workspace="calo_intelligence",
        context="calo",
    ),
    CommandSpec(
        "algorithms.flags",
        "Algorithms",
        "CALO",
        "CALO settings",
        "settings",
        "Configure and save the CALO and TSH-CALO settings used by new experiments.",
        workspace="algorithms",
        context="features",
    ),
    CommandSpec(
        "compute.settings",
        "Compute",
        "Execution",
        "Compute mode",
        "compute",
        "Select CUDA-preferred or CPU-only execution and a permitted device.",
        workspace="experiment",
        context="compute",
        primary=True,
    ),
    CommandSpec(
        "compute.device",
        "Compute",
        "Diagnostics",
        "Device status",
        "device",
        "Open live device, Safe-80 admission, and fallback diagnostics.",
        workspace="dashboard",
        context="device",
    ),
    CommandSpec(
        "compute.live",
        "Compute",
        "Monitoring",
        "Live optimization",
        "activity",
        "Open live optimization counters and plots.",
        workspace="live_optimization",
        context="live",
    ),
    CommandSpec(
        "compute.statistics",
        "Compute",
        "Analysis",
        "Statistics",
        "statistics",
        "Open statistical analysis for retained runs.",
        workspace="statistics",
        context="statistics",
    ),
    CommandSpec(
        "results.explorer",
        "Results",
        "Review",
        "Results explorer",
        "results",
        "Open retained results and run provenance.",
        workspace="results",
        context="results",
        primary=True,
    ),
    CommandSpec(
        "policies.training",
        "Policies",
        "Training",
        "Train policy",
        "training",
        "Open the policy-training inputs. Use the visible readiness and start action in the input pane; results are not selected for experiments automatically.",
        "training",
        context="training",
        primary=True,
    ),
    CommandSpec(
        "view.activity",
        "View",
        "Panes",
        "Activity pane",
        "activity",
        "Show or hide jobs, logs, warnings, device, and provenance.",
        "toggle_activity",
    ),
    CommandSpec(
        "view.theme",
        "View",
        "Appearance",
        "Light / dark",
        "theme",
        "Switch between the accessible light and dark themes.",
        "toggle_theme",
    ),
    CommandSpec(
        "view.reset",
        "View",
        "Layout",
        "Reset layout",
        "reset",
        "Restore the versioned safe default ribbon and dock layout.",
        "reset_layout",
    ),
    CommandSpec(
        "help.guide",
        "Help",
        "Guidance",
        "User guide",
        "help",
        "Open native and Docker operating guidance.",
        "guide",
        shortcut="F1",
        primary=True,
    ),
    CommandSpec(
        "help.about",
        "Help",
        "Application",
        "About",
        "info",
        "Show application identity and version.",
        "about",
    ),
)


class CommandRegistry(QObject):
    """Create and own every ribbon action from one immutable specification set."""

    command_triggered = pyqtSignal(str)
    command_selected = pyqtSignal(object)

    def __init__(self, owner: QObject) -> None:
        super().__init__(owner)
        self._specs = {item.command_id: item for item in COMMAND_SPECS}
        self._actions: dict[str, QAction] = {}
        self._disabled_reasons: dict[str, str] = {}
        for spec in COMMAND_SPECS:
            action = QAction(workspace_icon(spec.icon), spec.label, owner)
            action.setObjectName(f"Command.{spec.command_id}")
            action.setToolTip(spec.tooltip)
            action.setStatusTip(spec.tooltip)
            if spec.shortcut:
                action.setShortcut(QKeySequence(spec.shortcut))
            action.triggered.connect(
                lambda _checked=False, command_id=spec.command_id: self._dispatch(command_id)
            )
            self._actions[spec.command_id] = action

    @property
    def specs(self) -> tuple[CommandSpec, ...]:
        return COMMAND_SPECS

    def spec(self, command_id: str) -> CommandSpec:
        return self._specs[str(command_id)]

    def action(self, command_id: str) -> QAction:
        return self._actions[str(command_id)]

    def disabled_reason(self, command_id: str) -> str:
        return self._disabled_reasons.get(str(command_id), "")

    def set_available(self, command_id: str, available: bool, reason: str = "") -> None:
        spec = self.spec(command_id)
        action = self.action(command_id)
        action.setEnabled(bool(available))
        if available:
            self._disabled_reasons.pop(command_id, None)
            action.setToolTip(spec.tooltip)
            action.setStatusTip(spec.tooltip)
        else:
            explanation = str(reason or "This command is not available in the current state.")
            self._disabled_reasons[command_id] = explanation
            action.setToolTip(f"{spec.tooltip}\n\nUnavailable: {explanation}")
            action.setStatusTip(explanation)

    def select(self, command_id: str) -> None:
        self.command_selected.emit(self.spec(command_id))

    def _dispatch(self, command_id: str) -> None:
        self.select(command_id)
        self.command_triggered.emit(command_id)
