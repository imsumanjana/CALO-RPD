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


COMMAND_SPECS: tuple[CommandSpec, ...] = (
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
        "home.overview",
        "Home",
        "Navigate",
        "Overview",
        "home",
        "Open the project and readiness overview.",
        workspace="dashboard",
        context="overview",
        primary=True,
    ),
    CommandSpec(
        "home.resume",
        "Home",
        "Navigate",
        "Resume center",
        "resume",
        "Review resumable work without automatically resuming it.",
        workspace="resume_center",
        context="resume",
    ),
    CommandSpec(
        "home.find",
        "Home",
        "Navigate",
        "Workspaces",
        "search",
        "Show every workspace in the ribbon.",
        "find",
        shortcut="Ctrl+K",
    ),
    CommandSpec(
        "workspace.dashboard",
        "Workspace",
        "Home",
        "Overview",
        "home",
        "Open Overview.",
        workspace="dashboard",
    ),
    CommandSpec(
        "workspace.resume",
        "Workspace",
        "Home",
        "Resume",
        "resume",
        "Open Resume Center.",
        workspace="resume_center",
    ),
    CommandSpec(
        "workspace.calo",
        "Workspace",
        "Model",
        "CALO",
        "policy",
        "Open CALO Intelligence.",
        workspace="calo_intelligence",
    ),
    CommandSpec(
        "workspace.power",
        "Workspace",
        "Model",
        "Power",
        "network",
        "Open Power System.",
        workspace="power_system",
    ),
    CommandSpec(
        "workspace.orpd",
        "Workspace",
        "Model",
        "ORPD",
        "formulation",
        "Open ORPD Formulation.",
        workspace="orpd",
    ),
    CommandSpec(
        "workspace.algorithms",
        "Workspace",
        "Model",
        "Methods",
        "algorithm",
        "Open Algorithms.",
        workspace="algorithms",
    ),
    CommandSpec(
        "workspace.scenarios",
        "Workspace",
        "Model",
        "Scenarios",
        "scenario",
        "Open Robust Scenarios.",
        workspace="scenarios",
    ),
    CommandSpec(
        "workspace.portfolio",
        "Workspace",
        "Study",
        "Portfolio",
        "portfolio",
        "Open Portfolio.",
        workspace="portfolio",
    ),
    CommandSpec(
        "workspace.experiment",
        "Workspace",
        "Study",
        "Study",
        "study",
        "Open Experiment Manager.",
        workspace="experiment",
    ),
    CommandSpec(
        "workspace.live",
        "Workspace",
        "Study",
        "Live",
        "activity",
        "Open Live Optimization.",
        workspace="live_optimization",
    ),
    CommandSpec(
        "workspace.results",
        "Workspace",
        "Evidence",
        "Results",
        "results",
        "Open Results.",
        workspace="results",
    ),
    CommandSpec(
        "workspace.statistics",
        "Workspace",
        "Evidence",
        "Stats",
        "statistics",
        "Open Statistical Analysis.",
        workspace="statistics",
    ),
    CommandSpec(
        "workspace.validation",
        "Workspace",
        "Evidence",
        "Validate",
        "validation",
        "Open Validation.",
        workspace="validation",
    ),
    CommandSpec(
        "workspace.benchmark",
        "Workspace",
        "Evidence",
        "Bench",
        "benchmark",
        "Open Benchmark.",
        workspace="benchmark",
    ),
    CommandSpec(
        "workspace.publication",
        "Workspace",
        "Evidence",
        "Export",
        "publication",
        "Open Publication.",
        workspace="publication",
    ),
    CommandSpec(
        "workspace.settings",
        "Workspace",
        "System",
        "Settings",
        "settings",
        "Open Settings.",
        workspace="settings",
    ),
    CommandSpec(
        "experiment.setup",
        "Experiment",
        "Design",
        "Study setup",
        "study",
        "Edit compact study identity, case, repetitions, seed, and population inputs.",
        workspace="experiment",
        context="experiment",
        primary=True,
    ),
    CommandSpec(
        "experiment.power",
        "Experiment",
        "Model",
        "Power system",
        "network",
        "Load and inspect the power-system case.",
        workspace="power_system",
        context="case",
    ),
    CommandSpec(
        "experiment.formulation",
        "Experiment",
        "Model",
        "ORPD formulation",
        "formulation",
        "Configure objectives, variables, and constraints.",
        workspace="orpd",
        context="formulation",
    ),
    CommandSpec(
        "experiment.scenarios",
        "Experiment",
        "Model",
        "Robust scenarios",
        "scenario",
        "Configure deterministic or robust scenario inputs.",
        workspace="scenarios",
        context="scenarios",
    ),
    CommandSpec(
        "experiment.portfolio",
        "Experiment",
        "Design",
        "Portfolio",
        "portfolio",
        "Configure the evidence portfolio and study strength.",
        workspace="portfolio",
        context="portfolio",
    ),
    CommandSpec(
        "experiment.run",
        "Experiment",
        "Execute",
        "Run workspace",
        "run",
        "Open the explicit experiment launch controls. This command does not start a run.",
        workspace="experiment",
        context="run",
        primary=True,
    ),
    CommandSpec(
        "experiment.stop",
        "Experiment",
        "Execute",
        "Cancel task",
        "stop",
        "Request cancellation only when the active task declares itself cancellable.",
        "cancel",
    ),
    CommandSpec(
        "algorithms.configure",
        "Algorithms",
        "Configuration",
        "Algorithms",
        "algorithm",
        "Select approved algorithms and configure their parameters.",
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
        "Review the CALO optimization settings used by new experiments.",
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
        "compute.refresh",
        "Compute",
        "Diagnostics",
        "System readiness",
        "refresh",
        "Open the readiness controls. Refresh remains an explicit action in that workspace.",
        workspace="dashboard",
        context="device",
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
        "results.live",
        "Results",
        "Review",
        "Live results",
        "activity",
        "Open live optimization counters and plots.",
        workspace="live_optimization",
        context="live",
    ),
    CommandSpec(
        "results.statistics",
        "Results",
        "Evidence",
        "Statistics",
        "statistics",
        "Open statistical analysis for retained runs.",
        workspace="statistics",
        context="statistics",
    ),
    CommandSpec(
        "results.validation",
        "Results",
        "Evidence",
        "Validation",
        "validation",
        "Open independent result validation and audit controls.",
        workspace="validation",
        context="validation",
    ),
    CommandSpec(
        "results.benchmark",
        "Results",
        "Evidence",
        "Benchmark",
        "benchmark",
        "Open benchmark campaign planning and retained evidence.",
        workspace="benchmark",
        context="benchmark",
    ),
    CommandSpec(
        "results.publication",
        "Results",
        "Export",
        "Report & export",
        "publication",
        "Open publication preview and export controls.",
        workspace="publication",
        context="publication",
    ),
    CommandSpec(
        "policies.status",
        "Policies",
        "Policies",
        "Policy status",
        "policy",
        "Inspect policy inventory and truthful governing status.",
        workspace="calo_intelligence",
        context="policy",
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
        "policies.resume",
        "Policies",
        "Training",
        "Training sessions",
        "resume",
        "Open resumable work and training sessions.",
        workspace="resume_center",
        context="resume",
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
        "help.settings",
        "Help",
        "Application",
        "Settings",
        "settings",
        "Open application appearance and density settings.",
        workspace="settings",
        context="settings",
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
