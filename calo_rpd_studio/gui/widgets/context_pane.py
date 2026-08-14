"""Compact contextual input editors backed by the existing application state."""

from __future__ import annotations

import uuid
from copy import deepcopy
from html import escape
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.command_registry import CommandSpec
from calo_rpd_studio.gui.user_feedback import show_warning
from calo_rpd_studio.gui.widgets.global_status_bar import truthful_runtime_assignment
from calo_rpd_studio.power_system.case_identity import PROTECTED_HOLDOUT_BUS_COUNTS
from calo_rpd_studio.power_system.case_loader import CaseLoader


def _help(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("ContextHelp")
    label.setWordWrap(True)
    return label


_TRAINING_INPUT_HELP = {
    "library": (
        "Saved interrupted training runs found in the private application model directory and "
        "any locations you explicitly add. Selecting one keeps checkpoints in its own directory; "
        "it never copies them into the default directory."
    ),
    "plan": (
        "Optional settings template. Importing it fills the visible controls with compatible "
        "scientific training settings."
    ),
    "output": (
        "Directory for checkpoints, run records, and the trained policy. Choose a new directory; "
        "the result is not selected for experiments automatically."
    ),
    "resume": (
        "New training saves verified recovery points automatically after each safely completed "
        "training window. You may safely pause and resume any number of times before the finite "
        "campaign completes. Continue an existing interrupted directory only by selecting that "
        "saved run; its unchanged plan, state, recovery point, and saved-file integrity must "
        "pass compatibility checks. A completed model with authenticated final optimizer, RNG, "
        "receipt, and checkpoint state may explicitly add another identical finite segment."
    ),
    "campaign_id": (
        "Unique training-run identity used in saved plans and run records. Changing "
        "the text changes identity, not policy quality; use a new ID for a new design."
    ),
    "cases": (
        "Training systems presented to every independent member. More eligible cases broaden "
        "experience and increase work linearly; fewer cases reduce cost but narrow coverage. "
        "Protected case118 and case300 can never be selected."
    ),
    "members": (
        "Separately seeded policy networks. Increasing members improves seed-diversity and later "
        "ensemble uncertainty evidence but increases evaluations, time, and storage linearly; "
        "decreasing members reduces cost but weakens seed-robustness evidence."
    ),
    "master_seed": (
        "Deterministically generates member and episode seeds. A higher or lower number is not "
        "better; it selects a different reproducible trajectory and must not be result-tuned."
    ),
    "population": (
        "Candidates evaluated per generation. With a fixed evaluation budget, increasing it gives "
        "broader generations but fewer policy transitions; decreasing it gives more transitions "
        "but less within-generation diversity. Evaluations must be divisible by population."
    ),
    "evaluations": (
        "Exact candidate-evaluation budget for each member-case episode. Increasing it gives more "
        "experience and costs more power-flow work; decreasing it shortens training. It must be "
        "at least twice, and an exact multiple of, population. Pausing never expands or resets "
        "this finite budget."
    ),
    "device": (
        "Execution intent. CUDA preferred uses NVIDIA CUDA when safely available; CUDA only is "
        "strict when CPU fallback is off; CPU only computes on the CPU. Device choice changes "
        "the recorded execution method, not the declared scientific design."
    ),
    "cpu_fallback": (
        "When enabled, an unsafe or failed CUDA allocation may continue on CPU and record why. "
        "Disabling it makes an unavailable CUDA request stop instead; CPU execution can be much slower."
    ),
    "learning_rate": (
        "Adam optimizer step size. Increasing it makes weight updates faster and more aggressive "
        "but can destabilize learning; decreasing it is more conservative but may learn too "
        "slowly."
    ),
    "discount": (
        "Weight placed on later rewards. Increasing toward 1 emphasizes longer-term search "
        "effects but can raise variance; decreasing emphasizes immediate outcomes and can become "
        "myopic."
    ),
    "gae": (
        "Generalized Advantage Estimation horizon. Increasing it uses longer reward sequences "
        "with typically lower bias and higher variance; decreasing it relies more on short-step "
        "value estimates with higher bias and lower variance."
    ),
    "clip": (
        "Maximum PPO probability-ratio movement around the old policy. Increasing it permits "
        "larger, riskier policy changes; decreasing it stabilizes updates but can slow learning."
    ),
    "ppo_epochs": (
        "Optimizer passes over the same collected rollout. Increasing it reuses experience more "
        "and adds neural compute but can overfit stale data; decreasing it is more conservative "
        "and cheaper."
    ),
    "hidden_dim": (
        "Width of topology, aggregate, fusion, action, and value representations. Increasing it "
        "adds capacity, memory use, compute, and overfitting risk; decreasing it is faster but may "
        "underfit."
    ),
    "graph_steps": (
        "Topology message-passing rounds. Increasing it propagates information across more "
        "network hops but costs more and can oversmooth bus representations; decreasing it keeps "
        "reasoning local."
    ),
}

_TRAINING_INPUT_SUGGESTIONS = {
    "library": (
        "Suggested choice: New training for a fresh run, or one listed interrupted run for an "
        "exact resume. Added locations expand discovery only and have no low-to-high range."
    ),
    "plan": (
        "Suggested selection: leave blank for fresh defaults, or import one JSON settings "
        "template. A template does not make a policy available for experiments."
    ),
    "output": (
        "Suggested selection: one new empty directory, or one explicitly resumable directory "
        "from the same checked campaign."
    ),
    "resume": (
        "Suggested choice: automatic recovery stays on for new training. Exact resume is on only "
        "for a selected compatible interrupted run. The number of safe resume cycles is not "
        "limited, but every cycle continues the same finite evaluation plan. Completed-model "
        "extensions are also explicit finite segments and are never started automatically."
    ),
    "campaign_id": (
        "Suggested format: a short unique identifier with project, design, and run date; identity "
        "text has no quality-ranked low-to-high range."
    ),
    "cases": (
        "Suggested scope: all eligible bundled training cases, currently case30 and case57. "
        "Hard boundary: protected case118 and case300 remain excluded."
    ),
    "members": (
        "Suggested range: 3 to 5 independent members. Hard GUI range: 2 to 256; larger ensembles "
        "require proportional compute and independent evidence."
    ),
    "master_seed": (
        "Suggested range: any predeclared integer from 0 to 2,147,483,647. Seed magnitude does "
        "not indicate quality; never choose it from outcomes."
    ),
    "population": (
        "Suggested range: 20 to 64 candidates, starting at the current default of 20. Hard GUI "
        "range: 2 to 1,000,000 and the loaded resource envelope may impose a lower ceiling."
    ),
    "evaluations": (
        "Suggested range: 10,000 to 100,000 evaluations per member-case episode, beginning at "
        "10,000. Hard GUI range: 1 to 2,000,000,000; the value must be at least twice and exactly "
        "divisible by population."
    ),
    "device": (
        "Suggested choice: CUDA preferred for routine compatible training; NVIDIA CUDA only for "
        "strict GPU runs; CPU only when CPU execution is intentionally required."
    ),
    "cpu_fallback": (
        "Suggested choice: enabled with CUDA preferred for recoverability; disabled only when a "
        "strict CUDA-only training run is required."
    ),
    "learning_rate": (
        "Suggested range: 0.0001 to 0.001, starting at 0.0003. Hard GUI range: 0.0000001 to 1.0; "
        "values outside the suggested band need separate stability evidence."
    ),
    "discount": ("Suggested range: 0.95 to 0.995, starting at 0.99. Hard GUI range: 0.0 to 1.0."),
    "gae": ("Suggested range: 0.90 to 0.98, starting at 0.95. Hard GUI range: 0.0 to 1.0."),
    "clip": ("Suggested range: 0.10 to 0.25, starting at 0.20. Hard GUI range: 0.001 to 0.999."),
    "ppo_epochs": (
        "Suggested range: 3 to 10 update epochs, starting at 4. Hard GUI range: 1 to 1,000."
    ),
    "hidden_dim": (
        "Suggested range: 64 to 256 units, starting at 64. Hard GUI range: 8 to 8,192; larger "
        "models require additional memory, runtime, and overfitting evidence."
    ),
    "graph_steps": (
        "Suggested range: 2 to 4 message-passing steps, starting at 2. Hard GUI range: 1 to 128."
    ),
}


def _training_input_help(key: str) -> str:
    """Combine behavior guidance and a clearly qualified practical suggestion."""
    return (
        f"{_TRAINING_INPUT_HELP[key]}\n\n{_TRAINING_INPUT_SUGGESTIONS[key]}\n\n"
        "Suggested ranges and choices are conservative starting guidance, not validated optima, "
        "policy-quality evidence, or evidence that a policy is ready for experiments. Changing a checked input requires "
        "a fresh readiness check."
    )


class _TrainingInfoButton(QToolButton):
    """Small hover, click, keyboard-focus, and screen-reader help affordance."""

    def __init__(self, label: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self.help_text = str(text)
        self.setObjectName("TrainingInfoButton")
        self.setText("i")
        self.setAutoRaise(True)
        self.setFixedSize(20, 20)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        tooltip_html = escape(self.help_text)
        for prefix in (
            "Suggested range:",
            "Suggested selection:",
            "Suggested format:",
            "Suggested scope:",
            "Suggested choice:",
            "Hard GUI range:",
            "Hard boundary:",
        ):
            tooltip_html = tooltip_html.replace(prefix, f"<b>{prefix}</b>")
        tooltip_html = tooltip_html.replace("\n", "<br>")
        self.setToolTip(f"<div style='width: 340px'>{tooltip_html}</div>")
        self.setAccessibleName(f"Information about {label}")
        self.setAccessibleDescription(self.help_text)
        self.clicked.connect(self.show_help)

    def show_help(self) -> None:
        QToolTip.showText(self.mapToGlobal(self.rect().bottomLeft()), self.toolTip(), self)

    def focusInEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().focusInEvent(event)
        self.show_help()

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt API
        QToolTip.hideText()
        super().focusOutEvent(event)


class GenericContextEditor(QWidget):
    open_requested = pyqtSignal(str)

    def __init__(self, context_id: str, parent=None) -> None:
        super().__init__(parent)
        self.context_id = context_id
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = _help("No additional inputs.")
        self.summary.hide()
        layout.addWidget(self.summary)
        layout.addStretch(1)
        self.open_button = QPushButton("Open workspace")
        self.open_button.hide()
        self.open_button.clicked.connect(lambda: self.open_requested.emit(self.context_id))
        layout.addWidget(self.open_button)

    def configure(self, spec: CommandSpec) -> None:
        self.context_id = spec.workspace
        self.summary.setText("No additional inputs.")
        self.open_button.hide()


class ExperimentQuickEditor(QWidget):
    applied = pyqtSignal(str)
    open_requested = pyqtSignal(str)

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(
            _help(
                "Compact study inputs. Apply validates a complete copied configuration before replacing shared state."
            )
        )
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        self.name = QLineEdit()
        self.name.setMaximumWidth(280)
        self.case = QComboBox()
        self.case.setEditable(True)
        self.case.addItems(("case30", "case57", "case118", "case300"))
        self.case.setMaximumWidth(180)
        self.runs = QSpinBox()
        self.runs.setRange(1, 1_000_000)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.population = QSpinBox()
        self.population.setRange(2, 1_000_000)
        for widget in (self.runs, self.seed, self.population):
            widget.setMaximumWidth(150)
        form.addRow("Study name", self.name)
        form.addRow("Case", self.case)
        form.addRow("Runs", self.runs)
        form.addRow("Master seed", self.seed)
        form.addRow("Population", self.population)
        outer.addLayout(form)
        outer.addStretch(1)
        buttons = QHBoxLayout()
        open_button = QPushButton("Full setup")
        open_button.clicked.connect(lambda: self.open_requested.emit("experiment"))
        apply_button = QPushButton("Validate & apply")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(self.apply)
        buttons.addWidget(open_button)
        buttons.addWidget(apply_button)
        outer.addLayout(buttons)
        self.load()

    def load(self) -> None:
        config = self.state.config
        self.name.setText(str(config.name))
        self.case.setCurrentText(str(config.case_name))
        self.runs.setValue(int(config.runs))
        self.seed.setValue(int(config.master_seed))
        self.population.setValue(int(config.population_size))

    def apply(self) -> None:
        candidate = deepcopy(self.state.config)
        candidate.name = self.name.text().strip()
        candidate.case_name = self.case.currentText().strip()
        candidate.study_case_plan = [candidate.case_name]
        candidate.runs = self.runs.value()
        candidate.master_seed = self.seed.value()
        candidate.population_size = self.population.value()
        try:
            candidate.validate()
        except Exception as exc:
            show_warning(
                self,
                "Study inputs were not applied",
                "Review the study values and try again.",
                exc,
                source="study inputs",
            )
            return
        self.state.config = candidate
        self.state.update_config()
        self.applied.emit("Validated compact study inputs applied")


class ComputeQuickEditor(QWidget):
    applied = pyqtSignal(str)
    open_requested = pyqtSignal(str)

    def __init__(self, state, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(
            _help(
                "Configured intent is distinct from actual runtime assignment. Intel XPU is not executable."
            )
        )
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.mode = QComboBox()
        self.mode.addItem("CUDA-preferred", "cuda_preferred")
        self.mode.addItem("CPU-only", "cpu_only")
        self.device = QComboBox()
        self.device.setEditable(True)
        self.device.addItems(("auto", "cuda", "cuda:0", "cpu"))
        self.mode.setMaximumWidth(190)
        self.device.setMaximumWidth(190)
        form.addRow("Execution mode", self.mode)
        form.addRow("Requested device", self.device)
        outer.addLayout(form)
        self.truth = _help("")
        outer.addWidget(self.truth)
        outer.addStretch(1)
        buttons = QHBoxLayout()
        diagnostics = QPushButton("Device diagnostics")
        diagnostics.clicked.connect(lambda: self.open_requested.emit("dashboard"))
        apply_button = QPushButton("Validate & apply")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(self.apply)
        buttons.addWidget(diagnostics)
        buttons.addWidget(apply_button)
        outer.addLayout(buttons)
        self.load()

    def load(self) -> None:
        config = self.state.config
        mode_index = self.mode.findData(str(config.execution_backend))
        self.mode.setCurrentIndex(max(0, mode_index))
        self.device.setCurrentText(str(config.requested_compute_device))
        actual = truthful_runtime_assignment(config)
        self.truth.setText(
            f"Last recorded runtime assignment: {actual}. Safe-80 limits are ceilings, not usage."
        )

    def apply(self) -> None:
        candidate = deepcopy(self.state.config)
        candidate.execution_backend = str(self.mode.currentData())
        requested = self.device.currentText().strip().lower()
        if candidate.execution_backend == "cpu_only":
            requested = "cpu"
        candidate.requested_compute_device = requested
        # A configured-intent change invalidates any earlier runtime assignment. The resolver will
        # populate these fields on a copied run configuration immediately before execution.
        candidate.runtime_assigned_physical_device = ""
        candidate.runtime_assigned_logical_device = "cpu"
        candidate.runtime_compute_device = "cpu"
        candidate.runtime_fallback_policy = "unresolved"
        candidate.runtime_fallback_reason = ""
        candidate.runtime_device_resolution = {}
        candidate.runtime_resolution_process_id = 0
        try:
            candidate.validate()
        except Exception as exc:
            show_warning(
                self,
                "Compute inputs were not applied",
                "Select a compatible execution mode and device.",
                exc,
                source="compute inputs",
            )
            return
        self.state.config = candidate
        self.state.update_config()
        self.applied.emit(
            "Validated compute intent applied; actual assignment is recorded only at runtime"
        )


class TrainingPathEditor(QWidget):
    def __init__(self, model, training_controller, model_library, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.training_controller = training_controller
        self.model_library = model_library
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        campaign_group = QGroupBox("Campaign")
        campaign_form = QFormLayout(campaign_group)
        campaign_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        self.fields: dict[str, QLineEdit] = {}
        self.info_buttons: dict[str, _TrainingInfoButton] = {}
        self.path_rows: list[QWidget] = []
        self.library_picker = QComboBox()
        self.library_picker.setAccessibleName("Resumable policy training models")
        self.library_picker.currentIndexChanged.connect(self._library_selection_changed)
        library_host = QWidget()
        library_layout = QVBoxLayout(library_host)
        library_layout.setContentsMargins(0, 0, 0, 0)
        library_layout.setSpacing(6)
        library_layout.addWidget(self.library_picker)
        library_buttons = QHBoxLayout()
        self.add_library_location_button = QPushButton("Add to path")
        self.add_library_location_button.setToolTip(
            "Add another directory to future resumable-model scans."
        )
        self.add_library_location_button.setAccessibleName("Add a resumable training scan location")
        self.add_library_location_button.clicked.connect(self._add_library_location)
        self.refresh_library_button = QPushButton("Refresh")
        self.refresh_library_button.clicked.connect(self.refresh_model_library)
        library_buttons.addWidget(self.add_library_location_button)
        library_buttons.addWidget(self.refresh_library_button)
        library_buttons.addStretch(1)
        library_layout.addLayout(library_buttons)
        default_text = f"Default location: {self.model_library.default_directory}"
        if self.model_library.default_directory_error:
            default_text = "Default location unavailable · choose another training directory"
        self.default_library_path = QLabel(default_text)
        self.default_library_path.setObjectName("ContextHelp")
        self.default_library_path.setToolTip(self.model_library.default_directory_error)
        self.default_library_path.setWordWrap(True)
        library_layout.addWidget(self.default_library_path)
        campaign_form.addRow(self._info_label("library", "Saved training"), library_host)
        for key, label, placeholder in (
            ("plan", "Settings template", "Optional training settings (.json)"),
            ("output", "Training directory", "New or explicitly resumable training directory"),
        ):
            field = QLineEdit()
            field.setPlaceholderText(placeholder)
            field.setMaximumWidth(310)
            field.textChanged.connect(lambda value, name=key: self.model.set_value(name, value))
            if key == "plan":
                field.textChanged.connect(self._plan_path_changed)
            self.fields[key] = field
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.addWidget(field, 1)
            browse = QPushButton("Browse")
            browse.setMaximumWidth(72)
            browse.clicked.connect(lambda _checked=False, name=key: self._browse(name))
            row_layout.addWidget(browse)
            self.path_rows.append(row)
            campaign_form.addRow(self._info_label(key, label), row)
        self.resume = self.training_controller.resume
        self.resume.setText("Resume selected interrupted training")
        self.resume.setToolTip(
            "Continue only the selected interrupted training directory after its saved settings, "
            "state, recovery point, and saved-file integrity pass compatibility checks."
        )
        self.resume.toggled.connect(lambda _checked: self.refresh())
        self.automatic_recovery = QCheckBox("Automatic recovery for new training")
        self.automatic_recovery.setChecked(True)
        self.automatic_recovery.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.automatic_recovery.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )
        self.automatic_recovery.setAccessibleName("Automatic training recovery status")
        self.automatic_recovery.setAccessibleDescription(
            "Always on for new training. Verified recovery points are saved after each safely "
            "completed training window; this status is not a configurable input."
        )
        self.automatic_recovery.setToolTip(
            "New training saves verified recovery points automatically after each safely "
            "completed training window. Select an interrupted run from Saved training to "
            "continue it later."
        )
        self.completed_recovery = QLabel("Training complete · resume is not applicable")
        self.completed_recovery.setObjectName("ContextValue")
        self.unavailable_recovery = QLabel(
            "Resume unavailable - no verified safe checkpoint boundary"
        )
        self.unavailable_recovery.setObjectName("ContextValue")
        self.recovery_stack = QStackedWidget()
        self.recovery_stack.addWidget(self.automatic_recovery)
        self.recovery_stack.addWidget(self.resume)
        self.recovery_stack.addWidget(self.completed_recovery)
        self.recovery_stack.addWidget(self.unavailable_recovery)
        campaign_form.addRow(self._info_label("resume", "Recovery"), self.recovery_stack)
        self.load_plan_button = QPushButton("Import settings")
        self.load_plan_button.clicked.connect(self._load_plan)
        campaign_form.addRow("", self.load_plan_button)
        layout.addWidget(campaign_group)

        self.plan_group = QGroupBox("Training inputs")
        plan_form = QFormLayout(self.plan_group)
        plan_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        plan_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.campaign_id = QLineEdit()
        self.case_checks: dict[str, QCheckBox] = {}
        self._case_check_guard = False
        case_picker = QFrame()
        case_picker.setObjectName("TrainingCasePicker")
        self.case_picker_layout = QVBoxLayout(case_picker)
        self.case_picker_layout.setContentsMargins(0, 0, 0, 0)
        self.case_picker_layout.setSpacing(4)
        self.all_eligible_cases = QCheckBox("All eligible bundled cases")
        self.all_eligible_cases.setAccessibleName("Select all eligible policy training cases")
        self.all_eligible_cases.setToolTip(
            "Select every bundled case that is eligible for policy training."
        )
        self.case_picker_layout.addWidget(self.all_eligible_cases)
        for case_name in CaseLoader.available_cases():
            self._add_case_checkbox(
                case_name,
                protected=case_name in PROTECTED_HOLDOUT_BUS_COUNTS,
            )
        self.case_boundary = _help(
            "case118 and case300 are protected holdouts and cannot enter training."
        )
        self.case_boundary.setObjectName("TrainingCaseBoundary")
        self.case_picker_layout.addWidget(self.case_boundary)
        self.members = QSpinBox()
        self.members.setRange(2, 256)
        self.members.setValue(3)
        self.master_seed = QSpinBox()
        self.master_seed.setRange(0, 2_147_483_647)
        self.master_seed.setValue(2026)
        self.population = QSpinBox()
        self.population.setRange(2, 1_000_000)
        self.population.setValue(20)
        self.evaluations = QSpinBox()
        self.evaluations.setRange(1, 2_000_000_000)
        self.evaluations.setValue(10_000)
        self.device = QComboBox()
        self.device.addItem("CUDA preferred", "auto")
        self.device.addItem("NVIDIA CUDA only", "cuda")
        self.device.addItem("CPU only", "cpu")
        self.cpu_fallback = QCheckBox("Allow CPU fallback")
        self.cpu_fallback.setChecked(True)
        plan_form.addRow(self._info_label("campaign_id", "Campaign"), self.campaign_id)
        plan_form.addRow(self._info_label("cases", "Training cases"), case_picker)
        plan_form.addRow(self._info_label("members", "Independent members"), self.members)
        plan_form.addRow(self._info_label("master_seed", "Master seed"), self.master_seed)
        plan_form.addRow(self._info_label("population", "Population"), self.population)
        plan_form.addRow(
            self._info_label("evaluations", "Evaluations per episode"), self.evaluations
        )
        plan_form.addRow(self._info_label("device", "Compute"), self.device)
        plan_form.addRow(self._info_label("cpu_fallback", "Fallback"), self.cpu_fallback)

        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setDecimals(7)
        self.learning_rate.setRange(1e-7, 1.0)
        self.learning_rate.setValue(3e-4)
        self.discount = QDoubleSpinBox()
        self.discount.setDecimals(4)
        self.discount.setRange(0.0, 1.0)
        self.discount.setValue(0.99)
        self.gae = QDoubleSpinBox()
        self.gae.setDecimals(4)
        self.gae.setRange(0.0, 1.0)
        self.gae.setValue(0.95)
        self.clip = QDoubleSpinBox()
        self.clip.setDecimals(3)
        self.clip.setRange(0.001, 0.999)
        self.clip.setValue(0.20)
        self.ppo_epochs = QSpinBox()
        self.ppo_epochs.setRange(1, 1000)
        self.ppo_epochs.setValue(4)
        self.hidden_dim = QSpinBox()
        self.hidden_dim.setRange(8, 8192)
        self.hidden_dim.setValue(64)
        self.graph_steps = QSpinBox()
        self.graph_steps.setRange(1, 128)
        self.graph_steps.setValue(2)
        plan_form.addRow(self._info_label("learning_rate", "Learning rate"), self.learning_rate)
        plan_form.addRow(self._info_label("discount", "Discount factor"), self.discount)
        plan_form.addRow(self._info_label("gae", "GAE lambda"), self.gae)
        plan_form.addRow(self._info_label("clip", "PPO clip ratio"), self.clip)
        plan_form.addRow(self._info_label("ppo_epochs", "PPO update epochs"), self.ppo_epochs)
        plan_form.addRow(self._info_label("hidden_dim", "Hidden dimension"), self.hidden_dim)
        plan_form.addRow(self._info_label("graph_steps", "Graph steps"), self.graph_steps)
        layout.addWidget(self.plan_group)

        self._plan_controls = (
            self.campaign_id,
            self.all_eligible_cases,
            *self.case_checks.values(),
            self.members,
            self.master_seed,
            self.population,
            self.evaluations,
            self.device,
            self.cpu_fallback,
            self.learning_rate,
            self.discount,
            self.gae,
            self.clip,
            self.ppo_epochs,
            self.hidden_dim,
            self.graph_steps,
        )
        self._loading_plan = False
        self._new_plan_mode = True
        self._selected_saved_state = ""
        self._selected_saved_resumable = False
        self._selected_saved_extendable = False
        self._selected_extension_pending = False
        self.campaign_id.setText(f"tsh-calo-{uuid.uuid4().hex[:12]}")
        self._set_selected_cases(("case30", "case57"))
        self.refresh_model_library()
        self.campaign_id.textChanged.connect(
            lambda value: self._set_plan_value("campaign_id", value=str(value).strip())
        )
        self.all_eligible_cases.toggled.connect(self._all_eligible_cases_toggled)
        for checkbox in self.case_checks.values():
            checkbox.toggled.connect(self._case_selection_changed)
        self.members.valueChanged.connect(self._member_design_changed)
        self.master_seed.valueChanged.connect(self._member_design_changed)
        self.population.valueChanged.connect(self._population_changed)
        self.evaluations.valueChanged.connect(self._resource_design_changed)
        self.device.currentIndexChanged.connect(
            lambda _index: self._set_plan_value(
                "requested_device", value=str(self.device.currentData())
            )
        )
        self.cpu_fallback.toggled.connect(
            lambda value: self._set_plan_value("allow_cpu_fallback", value=bool(value))
        )
        for control, key, cast in (
            (self.learning_rate, "learning_rate", float),
            (self.discount, "discount_factor", float),
            (self.gae, "gae_lambda", float),
            (self.clip, "clip_ratio", float),
            (self.ppo_epochs, "ppo_epochs", int),
            (self.hidden_dim, "hidden_dim", int),
            (self.graph_steps, "graph_steps", int),
        ):
            control.valueChanged.connect(
                lambda value, name=key, converter=cast: self._set_plan_value(
                    "training", name, value=converter(value)
                )
            )
        for control in (
            self.campaign_id,
            self.all_eligible_cases,
            *self.case_checks.values(),
            self.members,
            self.master_seed,
            self.population,
            self.evaluations,
            self.device,
            self.cpu_fallback,
            self.learning_rate,
            self.discount,
            self.gae,
            self.clip,
            self.ppo_epochs,
            self.hidden_dim,
            self.graph_steps,
        ):
            if isinstance(control, QLineEdit):
                control.textChanged.connect(self._new_plan_input_changed)
            elif isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._new_plan_input_changed)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._new_plan_input_changed)
            else:
                control.valueChanged.connect(self._new_plan_input_changed)
        self.action_bar = QFrame()
        self.action_bar.setObjectName("TrainingActionBar")
        action_layout = QVBoxLayout(self.action_bar)
        action_layout.setContentsMargins(12, 10, 12, 12)
        action_layout.setSpacing(8)
        self.status = QLabel()
        self.status.setObjectName("ContextValue")
        self.status.setWordWrap(True)
        action_layout.addWidget(self.status)
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(0)
        self.training_progress.setFormat("0% committed")
        self.training_progress.setAccessibleName("Committed policy training progress")
        self.training_progress.hide()
        action_layout.addWidget(self.training_progress)
        self.training_progress_detail = QLabel()
        self.training_progress_detail.setObjectName("ContextHelp")
        self.training_progress_detail.setWordWrap(True)
        self.training_progress_detail.hide()
        action_layout.addWidget(self.training_progress_detail)
        self.training_action_button = QPushButton("Check readiness")
        self.training_action_button.setObjectName("PrimaryButton")
        self.training_action_button.setAccessibleName("Check policy training readiness")
        self.training_action_button.setMinimumHeight(38)
        self.training_action_button.clicked.connect(self._run_primary_action)
        action_layout.addWidget(self.training_action_button)
        self.pause_training_button = QPushButton("Pause after checkpoint")
        self.pause_training_button.setAccessibleName(
            "Pause policy training after the next checkpoint"
        )
        self.pause_training_button.setToolTip(
            "Finish the current bounded evaluation window, commit its verified checkpoint, "
            "and stop in a resumable state."
        )
        self.pause_training_button.clicked.connect(
            lambda: self.training_controller.state.task_status.cancel(
                "Pause requested · committing the current checkpoint window"
            )
        )
        self.pause_training_button.hide()
        action_layout.addWidget(self.pause_training_button)
        layout.addStretch(1)
        self.model.changed.connect(lambda _values: self.refresh())
        self.training_controller.activity_message.connect(self._training_activity)
        self.training_controller.training_completed.connect(self.refresh_model_library)
        self.training_controller.training_paused.connect(self.refresh_model_library)
        self.training_controller.training_progress.connect(self._training_progress_changed)
        self.model_library.changed.connect(self.refresh_model_library)
        # Initial selection updates the model and refreshes this editor. Keep it after the
        # status/action widgets exist because refresh() writes to both of them.
        self._select_new_training()

    def _training_activity(self, severity: str, message: str) -> None:
        self.refresh()
        if str(severity).upper() in {"WARNING", "ERROR", "CRITICAL"}:
            self.status.setText(str(message))

    def _training_progress_changed(self, event: dict) -> None:
        progress = max(0, min(100, int(event.get("progress_percent", 0))))
        cumulative = event.get("cumulative_candidate_evaluations")
        cumulative_text = (
            "" if cumulative is None else f" · {int(cumulative)} cumulative"
        )
        self.training_progress.setValue(progress)
        self.training_progress.setFormat(f"{progress}% committed")
        self.training_progress_detail.setText(
            f"Member {int(event.get('member_number', 0))}/"
            f"{int(event.get('member_count', 0))} · "
            f"{event.get('case_identity', 'case')} · "
            f"{int(event.get('episode_candidate_evaluations', 0))}/"
            f"{int(event.get('episode_evaluation_limit', 0))} evaluations · "
            f"checkpoint {str(event.get('checkpoint_sha256', ''))[:12]}{cumulative_text}"
        )
        self.training_progress.show()
        self.training_progress_detail.show()

    def _info_label(self, key: str, label: str) -> QWidget:
        host = QWidget()
        host.setObjectName("TrainingInputLabel")
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        caption = QLabel(label)
        caption.setObjectName("TrainingInputCaption")
        button = _TrainingInfoButton(label, _training_input_help(key), host)
        self.info_buttons[key] = button
        row.addWidget(caption)
        row.addWidget(button)
        row.addStretch(1)
        return host

    def _add_case_checkbox(self, case_name: str, *, protected: bool) -> QCheckBox:
        label = f"{case_name}  ·  Protected holdout" if protected else case_name
        checkbox = QCheckBox(label)
        checkbox.setAccessibleName(
            f"{case_name} protected holdout" if protected else f"Train policy on {case_name}"
        )
        checkbox.setProperty("caseIdentity", case_name)
        checkbox.setProperty("protectedHoldout", protected)
        if protected:
            checkbox.setChecked(False)
            checkbox.setEnabled(False)
            checkbox.setToolTip("Protected final holdout; policy training is prohibited.")
        self.case_checks[case_name] = checkbox
        if hasattr(self, "case_boundary"):
            self.case_picker_layout.insertWidget(
                max(0, self.case_picker_layout.count() - 1), checkbox
            )
        else:
            self.case_picker_layout.addWidget(checkbox)
        return checkbox

    def selected_training_cases(self) -> list[str]:
        return [
            case_name
            for case_name, checkbox in self.case_checks.items()
            if checkbox.isChecked() and checkbox.isEnabled()
        ]

    def _set_selected_cases(self, case_names) -> None:
        selected = tuple(str(item).strip() for item in case_names if str(item).strip())
        self._case_check_guard = True
        try:
            for case_name in selected:
                if case_name not in self.case_checks:
                    checkbox = self._add_case_checkbox(case_name, protected=False)
                    checkbox.toggled.connect(self._case_selection_changed)
                    self._plan_controls = (*getattr(self, "_plan_controls", ()), checkbox)
            for case_name, checkbox in self.case_checks.items():
                checkbox.setChecked(checkbox.isEnabled() and case_name in selected)
            eligible = [item for item in self.case_checks.values() if item.isEnabled()]
            self.all_eligible_cases.setChecked(
                bool(eligible) and all(item.isChecked() for item in eligible)
            )
        finally:
            self._case_check_guard = False

    def _all_eligible_cases_toggled(self, checked: bool) -> None:
        if self._case_check_guard:
            return
        self._case_check_guard = True
        try:
            for checkbox in self.case_checks.values():
                if checkbox.isEnabled():
                    checkbox.setChecked(bool(checked))
        finally:
            self._case_check_guard = False
        self._cases_changed()
        self._new_plan_input_changed()

    def _case_selection_changed(self, _checked: bool = False) -> None:
        if self._case_check_guard:
            return
        eligible = [item for item in self.case_checks.values() if item.isEnabled()]
        self._case_check_guard = True
        try:
            self.all_eligible_cases.setChecked(
                bool(eligible) and all(item.isChecked() for item in eligible)
            )
        finally:
            self._case_check_guard = False
        self._cases_changed()
        self._new_plan_input_changed()

    def _browse(self, key: str) -> None:
        current = self.fields[key].text().strip()
        if key == "output":
            if self.resume.isChecked():
                selected = QFileDialog.getExistingDirectory(
                    self, "Select interrupted training directory", current
                )
            else:
                selected_parent = QFileDialog.getExistingDirectory(
                    self, "Select location for new training output", current
                )
                selected = self._new_output_path(selected_parent) if selected_parent else ""
        else:
            caption = {"plan": "Select training settings template"}[key]
            selected, _filter = QFileDialog.getOpenFileName(
                self, caption, current, "JSON files (*.json);;All files (*)"
            )
        if selected:
            self.fields[key].setText(selected)

    def refresh_model_library(self, preferred_root: str | Path | None = None) -> None:
        selected = self.library_picker.currentData()
        current = selected.get("directory", "") if isinstance(selected, dict) else ""
        campaigns = self.model_library.saved_campaigns()
        self.library_picker.blockSignals(True)
        try:
            self.library_picker.clear()
            self.library_picker.addItem("New training", "")
            for campaign in campaigns:
                state_label = (
                    "Training complete"
                    if campaign["state"] == "completed"
                    else campaign["state"].title()
                )
                label = f"{campaign['campaign_id']}  ·  {state_label}"
                self.library_picker.addItem(label, campaign)
            selected_index = 0
            for index in range(1, self.library_picker.count()):
                record = self.library_picker.itemData(index)
                if isinstance(record, dict) and record.get("directory") == current:
                    selected_index = index
                    break
            if preferred_root and selected_index == 0:
                preferred = Path(preferred_root).expanduser().resolve()
                for index in range(1, self.library_picker.count()):
                    record = self.library_picker.itemData(index)
                    if not isinstance(record, dict):
                        continue
                    directory = Path(str(record.get("directory", ""))).expanduser().resolve()
                    if directory == preferred or preferred in directory.parents:
                        selected_index = index
                        break
            self.library_picker.setCurrentIndex(selected_index)
        finally:
            self.library_picker.blockSignals(False)
        if selected_index > 0 and (
            not isinstance(selected, dict)
            or selected.get("directory") != self.library_picker.currentData().get("directory", "")
        ):
            self._library_selection_changed(selected_index)

    def _add_library_location(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Add resumable-model location",
            str(self.model_library.default_directory.parent),
        )
        if not selected:
            return
        try:
            location = self.model_library.add_scan_location(selected)
        except (OSError, RuntimeError, ValueError) as exc:
            show_warning(
                self,
                "Location could not be added",
                "Choose a readable directory containing resumable training folders.",
                exc,
                source="training model library",
            )
            return
        self.refresh_model_library(location)

    def _library_selection_changed(self, _index: int = -1) -> None:
        record = self.library_picker.currentData()
        if isinstance(record, dict) and record.get("directory"):
            self._selected_saved_state = str(record.get("state", ""))
            self._selected_saved_resumable = bool(record.get("resumable", False))
            self._selected_saved_extendable = bool(record.get("extendable", False))
            self._selected_extension_pending = bool(record.get("extension_pending", False))
            self.training_controller.set_extension_mode(
                self._selected_saved_state == "completed"
                and self._selected_saved_extendable
            )
            self.training_controller.last_progress_event = dict(
                record.get("progress", {}) or {}
            )
            self.resume.setChecked(bool(record.get("resumable", False)))
            self.fields["plan"].setText(str(record["plan"]))
            self.fields["output"].setText(str(record["directory"]))
            self._load_plan(preserve_identity=True)
            if self.model.plan_payload is None:
                self.resume.setChecked(False)
                self.training_controller.set_extension_mode(False)
                show_warning(
                    self,
                    "Saved training could not be loaded",
                    "This saved run is incomplete or incompatible. Choose another saved run.",
                    ValueError(self.model.plan_error or "Saved training plan is invalid"),
                    source="training model library",
                )
            self.refresh()
            return
        self._select_new_training()

    def _select_new_training(self) -> None:
        self._selected_saved_state = ""
        self._selected_saved_resumable = False
        self._selected_saved_extendable = False
        self._selected_extension_pending = False
        self.training_controller.set_extension_mode(False)
        self.training_controller.last_progress_event = {}
        self.training_progress.hide()
        self.training_progress_detail.hide()
        self.resume.setChecked(False)
        self.fields["plan"].clear()
        self.model.clear_loaded_plan()
        self._new_plan_mode = True
        self.campaign_id.setText(f"tsh-calo-{uuid.uuid4().hex[:12]}")
        output = ""
        if not self.model_library.default_directory_error:
            output = self._new_output_path(str(self.model_library.default_directory))
        self.fields["output"].setText(output)
        self.refresh()

    def _new_output_path(self, selected_parent: str) -> str:
        campaign = self.campaign_id.text().strip() or f"tsh-calo-{uuid.uuid4().hex[:12]}"
        safe_name = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in campaign
        ).strip("-_")
        base = Path(selected_parent).expanduser() / (safe_name or "tsh-calo-training")
        candidate = base
        suffix = 2
        while candidate.exists():
            candidate = base.with_name(f"{base.name}-{suffix}")
            suffix += 1
        return str(candidate)

    def _load_plan(self, *, preserve_identity: bool | None = None) -> None:
        if preserve_identity is None:
            preserve_identity = self.resume.isChecked()
        self.model.load_plan(preserve_identity=preserve_identity)
        self._load_plan_controls()
        if self.model.plan_payload is not None:
            self._new_plan_mode = False

    def _plan_path_changed(self, value: str) -> None:
        if not str(value).strip():
            self._new_plan_mode = True

    def _set_plan_value(self, *path: str, value) -> None:
        if not self._loading_plan and not self._new_plan_mode:
            self.model.set_plan_value(*path, value=value)

    def _new_plan_input_changed(self, *_args) -> None:
        if self._loading_plan or not self._new_plan_mode:
            return
        self.model.clear_loaded_plan()

    def _cases_changed(self) -> None:
        cases = self.selected_training_cases()
        if not self._loading_plan and not self._new_plan_mode:
            self.model.set_member_design(
                development_cases=cases,
                member_count=self.members.value(),
                master_seed=self.master_seed.value(),
            )

    def _member_design_changed(self, *_args) -> None:
        if self._loading_plan or self._new_plan_mode:
            return
        self.model.set_member_design(
            development_cases=self.selected_training_cases(),
            member_count=self.members.value(),
            master_seed=self.master_seed.value(),
        )

    def _population_changed(self, value: int) -> None:
        self._resource_design_changed()

    def _resource_design_changed(self, *_args) -> None:
        if self._loading_plan or self._new_plan_mode:
            return
        self.model.set_resource_design(
            population_size=self.population.value(),
            max_evaluations=self.evaluations.value(),
        )

    def _load_plan_controls(self) -> None:
        payload = self.model.plan_payload
        if payload is None:
            return
        self._loading_plan = True
        try:
            training = dict(payload.get("training", {}))
            self.campaign_id.setText(str(payload.get("campaign_id", "")))
            self._set_selected_cases(payload.get("development_cases", ()))
            self.members.setValue(len(payload.get("members", ())))
            members = list(payload.get("members", ()))
            if members:
                self.master_seed.setValue(int(members[0].get("training_seed", 2026)))
            self.population.setValue(int(payload.get("population_size", 2)))
            self.evaluations.setValue(int(payload.get("max_evaluations", 1)))
            self.device.setCurrentIndex(
                max(0, self.device.findData(str(payload.get("requested_device", "auto"))))
            )
            self.cpu_fallback.setChecked(bool(payload.get("allow_cpu_fallback", True)))
            self.learning_rate.setValue(float(training.get("learning_rate", 3e-4)))
            self.discount.setValue(float(training.get("discount_factor", 0.99)))
            self.gae.setValue(float(training.get("gae_lambda", 0.95)))
            self.clip.setValue(float(training.get("clip_ratio", 0.20)))
            self.ppo_epochs.setValue(int(training.get("ppo_epochs", 4)))
            self.hidden_dim.setValue(int(training.get("hidden_dim", 64)))
            self.graph_steps.setValue(int(training.get("graph_steps", 2)))
        finally:
            self._loading_plan = False

    def refresh(self) -> None:
        controller = self.training_controller
        idle = controller.process is None
        training_active = not idle and controller._operation == "training"
        self.pause_training_button.setVisible(training_active)
        self.pause_training_button.setEnabled(
            training_active and not controller._pause_requested
        )
        if controller.last_progress_event:
            self._training_progress_changed(controller.last_progress_event)
        saved_locked = bool(self._selected_saved_state)
        if self._selected_saved_resumable:
            self.recovery_stack.setCurrentWidget(self.resume)
        elif self._selected_saved_state == "completed":
            self.recovery_stack.setCurrentWidget(self.completed_recovery)
        elif self._selected_saved_state:
            self.recovery_stack.setCurrentWidget(self.unavailable_recovery)
        else:
            self.recovery_stack.setCurrentWidget(self.automatic_recovery)
        self.library_picker.setEnabled(idle)
        self.add_library_location_button.setEnabled(idle)
        self.refresh_library_button.setEnabled(idle)
        for row in self.path_rows:
            row.setEnabled(idle and not saved_locked)
        self.resume.setEnabled(
            idle
            and self._selected_saved_resumable
            and bool(self.model.values.get("output"))
        )
        self.load_plan_button.setEnabled(
            idle
            and not saved_locked
            and not self.resume.isChecked()
            and bool(self.model.values.get("plan"))
        )
        for control in self._plan_controls:
            control.setEnabled(
                idle
                and not saved_locked
                and not self.resume.isChecked()
                and not bool(control.property("protectedHoldout"))
            )
        if controller.process is not None:
            self.status.setText(controller.status.text())
            action_label = "Pause pending" if controller._pause_requested else "Training active"
            self._set_primary_action(action_label, controller.status.text(), False)
            return
        if self._selected_saved_state == "completed":
            record = self.library_picker.currentData()
            candidate_error = (
                str(record.get("candidate_error", "")) if isinstance(record, dict) else ""
            )
            extension_error = (
                str(record.get("extension_error", "")) if isinstance(record, dict) else ""
            )
            if self._selected_saved_extendable:
                ready = controller._validated_fingerprint == self.model.fingerprint()
                self.status.setText(
                    (
                        "Ready to continue the paused finite extension segment"
                        if self._selected_extension_pending
                        else "Ready to extend with one finite identical-condition segment"
                    )
                    if ready
                    else (
                        "Paused extension is resumable - check extension readiness"
                        if self._selected_extension_pending
                        else "Completed model is extendable - check extension readiness"
                    )
                )
                self._set_primary_action(
                    (
                        "Continue extension"
                        if ready and self._selected_extension_pending
                        else ("Extend training" if ready else "Check extension readiness")
                    ),
                    (
                        "Continue every member from authenticated optimizer, RNG, receipt, and "
                        "checkpoint state under the unchanged finite plan."
                    ),
                    True,
                )
            elif candidate_error:
                self.status.setText(candidate_error)
                self._set_primary_action("Training complete", candidate_error, False)
            else:
                self.status.setText(
                    "Training complete · the saved candidate is available in the Policy library"
                    + (f". {extension_error}" if extension_error else "")
                )
                self._set_primary_action(
                    "Training complete",
                    "Use the Policy library's single Import action to add this saved candidate.",
                    False,
                )
            return
        if self._selected_saved_state and not self._selected_saved_resumable:
            self.status.setText(
                "Resume unavailable - the saved run has no verified safe checkpoint boundary"
            )
            self._set_primary_action("Resume unavailable", self.status.text(), False)
            return
        missing = self.model.missing(include_output=False)
        if missing:
            self.status.setText("Complete the required fields")
            self._set_primary_action("Check readiness", "Complete the required inputs.", False)
            return
        if self.model.plan_error:
            self.status.setText("Training settings could not be imported")
            self.status.setToolTip(self.model.plan_error)
            self._set_primary_action("Check readiness", "Load valid training settings.", False)
            return
        self.status.setToolTip("")
        ready = controller._validated_fingerprint == self.model.fingerprint()
        missing_output = self.model.missing(include_output=True)
        if ready and missing_output:
            self.status.setText("Select a new or resumable output directory")
            self._set_primary_action("Start training", "Select an output directory.", False)
            return
        output_path = Path(self.model.values["output"]).expanduser()
        if ready and self.resume.isChecked() and not output_path.is_dir():
            self.status.setText("Resume requires an existing training directory")
            self._set_primary_action(
                "Start training", "Select an existing compatible training directory.", False
            )
            return
        if ready and output_path.exists() and not self.resume.isChecked():
            self.status.setText(
                "Output already exists · choose a new directory or select the interrupted run "
                "from Saved training"
            )
            self._set_primary_action(
                "Start training",
                "Choose a new output directory or select the interrupted run from Saved training.",
                False,
            )
            return
        if ready:
            self.status.setText(
                "Ready to start" if saved_locked else "Ready to start · automatic recovery is on"
            )
            self._set_primary_action(
                "Start training", "Start the checked new-policy training run.", True
            )
            return
        self.status.setText(
            "Ready for validation"
            if saved_locked
            else "Ready for validation · automatic recovery is on"
        )
        self._set_primary_action(
            "Check readiness",
            "Validate the selected training inputs without starting training.",
            True,
        )

    def _set_primary_action(self, label: str, tooltip: str, enabled: bool) -> None:
        self.training_action_button.setText(label)
        self.training_action_button.setToolTip(tooltip)
        self.training_action_button.setStatusTip(tooltip)
        self.training_action_button.setEnabled(enabled)
        self.training_action_button.setAccessibleName(label)

    def _run_primary_action(self) -> None:
        if self.training_controller.process is not None:
            return
        if self._selected_saved_state == "completed" and not self._selected_saved_extendable:
            self.refresh()
            return
        missing = self.model.missing(include_output=False)
        if missing:
            self.fields[missing[0]].setFocus(Qt.FocusReason.ShortcutFocusReason)
            self.refresh()
            return
        if self.model.plan_payload is None and self.model.values.get("plan"):
            self._load_plan()
            if self.model.plan_payload is None:
                self.fields["plan"].setFocus(Qt.FocusReason.ShortcutFocusReason)
                self.refresh()
                return
        if self.model.plan_payload is None:
            self.model.create_plan(
                campaign_id=self.campaign_id.text().strip(),
                development_cases=self.selected_training_cases(),
                member_count=self.members.value(),
                master_seed=self.master_seed.value(),
                population_size=self.population.value(),
                max_evaluations=self.evaluations.value(),
                requested_device=str(self.device.currentData()),
                allow_cpu_fallback=self.cpu_fallback.isChecked(),
                training={
                    "hidden_dim": self.hidden_dim.value(),
                    "graph_steps": self.graph_steps.value(),
                    "learning_rate": self.learning_rate.value(),
                    "ppo_epochs": self.ppo_epochs.value(),
                    "clip_ratio": self.clip.value(),
                    "value_weight": 0.50,
                    "entropy_weight": 0.01,
                    "gradient_norm": 0.50,
                    "discount_factor": self.discount.value(),
                    "gae_lambda": self.gae.value(),
                },
            )
            self._load_plan_controls()
            if self.model.plan_payload is None:
                self.refresh()
                return
            self._new_plan_mode = False
        if self.training_controller._validated_fingerprint == self.model.fingerprint():
            missing_output = self.model.missing(include_output=True)
            if missing_output:
                self.fields[missing_output[0]].setFocus(Qt.FocusReason.ShortcutFocusReason)
                self.refresh()
                return
            self.training_controller.start_training()
        else:
            self.training_controller.check_readiness()
        self.refresh()

    def prepare_resume(self, record: dict) -> None:
        self.training_controller.prepare_resume(record)
        self.fields["plan"].setText(self.model.values["plan"])
        self.fields["output"].setText(self.model.values["output"])
        self._load_plan()
        self.refresh()


class ContextPane(QWidget):
    workspace_requested = pyqtSignal(str)
    status_message = pyqtSignal(str)

    def __init__(
        self,
        state,
        navigator: QWidget,
        training_model,
        training_controller,
        training_model_library,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("ContextPane")
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setAccessibleName("Context inputs")
        self.tabs.tabBar().hide()
        outer.addWidget(self.tabs)

        editor_scroll = QScrollArea()
        editor_scroll.setWidgetResizable(True)
        editor_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        editor_host = QFrame()
        editor_host.setObjectName("ContextEditor")
        editor_layout = QVBoxLayout(editor_host)
        editor_layout.setContentsMargins(14, 14, 14, 14)
        self.title = QLabel("Inputs")
        self.title.setObjectName("ContextTitle")
        self.description = QLabel("")
        self.description.setWordWrap(True)
        self.description.setObjectName("ContextDescription")
        self.description.hide()
        editor_layout.addWidget(self.title)
        editor_layout.addWidget(self.description)
        self.stack = QStackedWidget()
        editor_layout.addWidget(self.stack, 1)
        editor_scroll.setWidget(editor_host)
        self.tabs.addTab(editor_scroll, "Inputs")
        navigator.setParent(self)
        navigator.hide()

        self.generic = GenericContextEditor("overview")
        self.experiment = ExperimentQuickEditor(state)
        self.compute = ComputeQuickEditor(state)
        self.training = TrainingPathEditor(
            training_model, training_controller, training_model_library
        )
        for editor in (self.generic, self.experiment, self.compute, self.training):
            self.stack.addWidget(editor)
            if hasattr(editor, "open_requested"):
                editor.open_requested.connect(self.workspace_requested)
        self.experiment.applied.connect(self.status_message)
        self.compute.applied.connect(self.status_message)
        self.training.action_bar.setParent(self)
        outer.addWidget(self.training.action_bar)
        self.training.action_bar.hide()

    def show_command(self, spec: CommandSpec) -> None:
        self.title.setText(spec.label)
        self.title.setToolTip(spec.tooltip)
        if spec.context == "experiment":
            self.experiment.load()
            target = self.experiment
        elif spec.context == "compute":
            self.compute.load()
            target = self.compute
        elif spec.context == "training":
            target = self.training
        else:
            self.generic.configure(spec)
            target = self.generic
        self.stack.setCurrentWidget(target)
        self.training.action_bar.setVisible(target is self.training)
        self.tabs.setCurrentIndex(0)

    def focus_navigator(self) -> None:
        self.tabs.setCurrentIndex(0)
        self.title.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def prepare_training_resume(self, record: dict) -> None:
        self.tabs.setCurrentIndex(0)
        self.stack.setCurrentWidget(self.training)
        self.training.action_bar.show()
        self.training.prepare_resume(record)
