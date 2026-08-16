"""Experiment-bound algorithm selection and typed CALO/TSH-CALO settings."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.algorithms.registry import POLICY_GATED_SPECS, SPECS
from calo_rpd_studio.gui.user_feedback import show_error
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


CALO_COMPONENTS = (
    ("use_memory", "Persistent success memory"),
    ("use_dual_archives", "Dual feasible/boundary archives"),
    ("use_epsilon", "Adaptive epsilon constraint control"),
    ("use_mixed_variable", "Mixed-variable handling"),
    ("use_diversity_recovery", "Diversity recovery"),
    ("use_hpem", "Hierarchical prefix elite memory (HPEM)"),
    ("use_contextual_credit", "Contextual operator credit"),
    ("use_variable_intelligence", "Variable-group intelligence"),
    ("use_dual_lane", "Dual-lane learning"),
    ("use_cognitive_precision", "Counted cognitive precision"),
    ("use_exact_evaluation_cache", "Exact-evaluation cache"),
)

CALO_NUMERIC_DEFAULTS = {
    "epsilon_quantile": 0.75,
    "epsilon_control_fraction": 0.65,
    "epsilon_exponent": 2.0,
    "memory_capacity": 256,
    "memory_decay": 0.97,
    "credit_decay": 0.90,
    "credit_floor": 0.02,
    "group_credit_decay": 0.90,
    "memory_evidence_batches": 6,
    "max_learning_lane_fraction": 0.92,
    "feasible_archive_capacity": 32,
    "boundary_archive_capacity": 48,
    "stagnation_window": 12,
    "recovery_diversity_threshold": 0.06,
    "recovery_fraction": 0.18,
    "precision_start_radius": 0.04,
    "precision_min_radius": 0.0005,
    "precision_max_radius": 0.15,
    "evaluation_cache_capacity": 4096,
}

TSH_CALO_NUMERIC_DEFAULTS = {
    "epsilon_quantile": 0.75,
    "epsilon_control_fraction": 0.65,
    "epsilon_exponent": 2.0,
    "memory_capacity": 256,
    "memory_decay": 0.97,
    "credit_decay": 0.90,
    "credit_floor": 0.02,
    "group_credit_decay": 0.90,
    "max_learning_lane_fraction": 0.92,
    "feasible_archive_capacity": 32,
    "boundary_archive_capacity": 48,
    "stagnation_window": 12,
    "recovery_diversity_threshold": 0.06,
    "recovery_fraction": 0.18,
    "precision_start_radius": 0.04,
    "precision_min_radius": 0.0005,
    "precision_max_radius": 0.15,
    "bandit_window_size": 32,
    "bandit_exploration": 0.35,
}

POLICY_BINDING_FIELDS = {
    "policy_algorithm_id",
    "policy_id",
    "policy_name",
    "policy_checkpoint",
    "policy_sha256",
    "policy_architecture_version",
    "policy_state_schema_version",
    "policy_action_schema_version",
    "policy_training_environment_version",
    "policy_qualification_status",
    "policy_grade",
    "policy_active_at_binding",
    "policy_feature_flags",
    "policy_artifact_kind",
    "policy_ensemble_size",
    "policy_ensemble_members",
    "policy_training_provenance",
    "policy_qualification_id",
    "policy_qualification_receipt_sha256",
    "policy_qualification_receipt",
    "policy_ood_calibration_sha256",
    "ood_calibration",
    "policy_assessment_id",
    "policy_assessment_receipt_sha256",
    "policy_assessment_receipt",
    "policy_scientist_selection",
}


class AlgorithmsPanel(WorkspacePage):
    """Stage algorithm selection separately from saved CALO/TSH-CALO settings."""

    stage_completed = pyqtSignal()
    stage_discarded = pyqtSignal()
    saved = pyqtSignal(str)

    @staticmethod
    def _safe_defaults(name: str, spec) -> dict:
        defaults = dict(spec.default_parameters)
        if name == "CALO":
            defaults.update(
                use_ai=False,
                strict_policy_binding=False,
                allow_unqualified_policy=False,
            )
            for field in POLICY_BINDING_FIELDS:
                defaults.pop(field, None)
        return defaults

    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Algorithms",
            "Select and submit comparison optimizers for the experiment, then configure the "
            "CALO and TSH-CALO settings separately.",
            parent,
        )
        self.state = state
        self.specs = {**SPECS, **POLICY_GATED_SPECS}
        self._loading = False
        self.calo_component_controls: dict[str, QCheckBox] = {}
        self.calo_numeric_controls: dict[str, QSpinBox | QDoubleSpinBox] = {}
        self.tsh_numeric_controls: dict[str, QSpinBox | QDoubleSpinBox] = {}

        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("AlgorithmConfigurationStack")
        self.algorithm_page = self._build_algorithm_page()
        self.settings_page = self._build_settings_page()
        self.content_stack.addWidget(self.algorithm_page)
        self.content_stack.addWidget(self.settings_page)
        self.layout_root.addWidget(self.content_stack, 1)

        self.state.config_changed.connect(self.load_from_config)
        self.state.policy_state_changed.connect(lambda _status: self._apply_policy_gate())
        self.load_from_config(self.state.config)
        self._apply_policy_gate()
        self.show_context("algorithms")

    def _build_algorithm_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        card = SectionCard(
            "Primary optimizer registry",
            "Checkbox changes are a draft until submitted. Submitted algorithms and their "
            "parameters are staged in the experiment configuration; CALO and TSH-CALO "
            "parameters are edited separately in CALO settings.",
        )
        self.table = QTableWidget(len(self.specs), 4)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Algorithm", "Scientific description", "Parameters (JSON)"]
        )
        self.table.setAlternatingRowColors(True)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        for row, (name, spec) in enumerate(self.specs.items()):
            use = QTableWidgetItem()
            use.setFlags(use.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self.table.setItem(row, 0, use)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, name_item)
            description = QTableWidgetItem(spec.description)
            description.setFlags(description.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, description)
            parameters = QTableWidgetItem()
            if name in {"CALO", "TSH-CALO"}:
                parameters.setFlags(parameters.flags() & ~Qt.ItemFlag.ItemIsEditable)
                parameters.setText("Configure in CALO settings")
            self.table.setItem(row, 3, parameters)
        self._fit_algorithm_table_to_entries()
        card.layout_root.addWidget(self.table)

        self.algorithm_stage_status = QLabel()
        self.algorithm_stage_status.setObjectName("AlgorithmStageStatus")
        self.algorithm_stage_status.setWordWrap(True)
        self.algorithm_stage_status.setAccessibleName("Algorithms staged for the experiment")
        card.layout_root.addWidget(self.algorithm_stage_status)

        buttons = QHBoxLayout()
        self.submit_algorithms_button = QPushButton("Submit algorithms for experiment")
        self.submit_algorithms_button.setObjectName("PrimaryButton")
        self.reset_algorithms_button = QPushButton("Reset selection")
        self.submit_algorithms_button.clicked.connect(self.submit_algorithm_selection)
        self.reset_algorithms_button.clicked.connect(self.reset_algorithm_selection)
        buttons.addWidget(self.submit_algorithms_button)
        buttons.addWidget(self.reset_algorithms_button)
        buttons.addStretch(1)
        card.layout_root.addLayout(buttons)
        self.table.itemChanged.connect(self._algorithm_draft_changed)
        self.algorithm_registry_card = card
        layout.addWidget(card)
        layout.addStretch(1)
        return page

    def _fit_algorithm_table_to_entries(self) -> None:
        """Show exactly the registered rows; the main preview owns any page overflow."""

        header = self.table.horizontalHeader()
        header_height = max(header.height(), header.sizeHint().height())
        body_height = sum(
            self.table.rowHeight(row) for row in range(self.table.rowCount())
        )
        frame_height = self.table.frameWidth() * 2
        self.table.setFixedHeight(header_height + body_height + frame_height)
        self.table.updateGeometry()

    @staticmethod
    def _readonly_value(text: str = "") -> QLabel:
        label = QLabel(text)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _double_control(
        minimum: float,
        maximum: float,
        step: float,
        *,
        decimals: int = 4,
    ) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        control.setDecimals(decimals)
        return control

    @staticmethod
    def _integer_control(minimum: int, maximum: int, step: int = 1) -> QSpinBox:
        control = QSpinBox()
        control.setRange(minimum, maximum)
        control.setSingleStep(step)
        return control

    def _numeric_control(self, key: str) -> QSpinBox | QDoubleSpinBox:
        integer_ranges = {
            "memory_capacity": (1, 1_000_000),
            "memory_evidence_batches": (1, 1_000_000),
            "feasible_archive_capacity": (1, 1_000_000),
            "boundary_archive_capacity": (1, 1_000_000),
            "stagnation_window": (4, 1_000_000),
            "evaluation_cache_capacity": (0, 10_000_000),
            "bandit_window_size": (1, 1_000_000),
        }
        if key in integer_ranges:
            return self._integer_control(*integer_ranges[key])
        if key == "epsilon_exponent":
            return self._double_control(0.0001, 100.0, 0.1, decimals=4)
        if key == "precision_min_radius":
            return self._double_control(0.000001, 1.0, 0.0001, decimals=6)
        return self._double_control(0.0, 1.0, 0.01, decimals=4)

    @staticmethod
    def _add_form_card(
        layout: QVBoxLayout, title: str, description: str
    ) -> QFormLayout:
        card = SectionCard(title, description)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        card.layout_root.addLayout(form)
        layout.addWidget(card)
        return form

    def _add_numeric_rows(
        self,
        form: QFormLayout,
        labels: tuple[tuple[str, str], ...],
        destination: dict[str, QSpinBox | QDoubleSpinBox],
    ) -> None:
        for key, label in labels:
            control = self._numeric_control(key)
            destination[key] = control
            form.addRow(label, control)

    def _build_settings_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        profile = self._add_form_card(
            layout,
            "CALO profile",
            "Rule-based CALO settings apply to new experiments only and never start execution.",
        )
        self.calo_mode = self._readonly_value("Rule-based CALO · policy assistance off")
        self.calo_profile = QComboBox()
        self.calo_profile.addItem("Canonical defaults", "canonical")
        self.calo_profile.addItem("Custom research profile", "custom")
        self.calo_profile.currentIndexChanged.connect(self._apply_calo_profile_mode)
        self.calo_strict_benchmark = QCheckBox("Prevent historical priors and warm starts")
        self.calo_strict_benchmark.setChecked(True)
        self.calo_strict_benchmark.setEnabled(False)
        self.calo_fingerprint = self._readonly_value("Not saved")
        profile.addRow("Mode", self.calo_mode)
        profile.addRow("Profile", self.calo_profile)
        profile.addRow("Strict benchmark mode", self.calo_strict_benchmark)
        profile.addRow("Configuration fingerprint", self.calo_fingerprint)

        components = self._add_form_card(
            layout,
            "Canonical CALO components",
            "Canonical comparison keeps every approved component enabled. Select Custom research "
            "profile to declare a non-canonical component configuration.",
        )
        for key, label in CALO_COMPONENTS:
            control = QCheckBox()
            control.setAccessibleName(label)
            self.calo_component_controls[key] = control
            components.addRow(label, control)

        constraint = self._add_form_card(
            layout,
            "Constraint control",
            "These settings control CALO search behavior; ORPD feasibility tolerances remain in "
            "ORPD Formulation.",
        )
        self._add_numeric_rows(
            constraint,
            (
                ("epsilon_quantile", "Initial epsilon quantile"),
                ("epsilon_control_fraction", "Epsilon control fraction"),
                ("epsilon_exponent", "Epsilon decay exponent"),
            ),
            self.calo_numeric_controls,
        )

        memory = self._add_form_card(
            layout,
            "Memory, archives, and contextual credit",
            "Capacities and decays used by the rule-based CALO optimizer.",
        )
        self._add_numeric_rows(
            memory,
            (
                ("memory_capacity", "Success-memory capacity"),
                ("memory_decay", "Memory decay"),
                ("credit_decay", "Contextual-credit decay"),
                ("credit_floor", "Credit floor"),
                ("group_credit_decay", "Variable-group credit decay"),
                ("memory_evidence_batches", "Evidence batches before full influence"),
                ("max_learning_lane_fraction", "Maximum learning-lane fraction"),
                ("feasible_archive_capacity", "Feasible archive capacity"),
                ("boundary_archive_capacity", "Boundary archive capacity"),
            ),
            self.calo_numeric_controls,
        )

        recovery = self._add_form_card(
            layout,
            "Diversity recovery and cognitive precision",
            "Recovery and counted local-precision behavior remain inside the declared exact "
            "evaluation budget.",
        )
        self._add_numeric_rows(
            recovery,
            (
                ("stagnation_window", "Stagnation window"),
                ("recovery_diversity_threshold", "Recovery-diversity threshold"),
                ("recovery_fraction", "Recovery fraction"),
                ("precision_start_radius", "Precision starting radius"),
                ("precision_min_radius", "Precision minimum radius"),
                ("precision_max_radius", "Precision maximum radius"),
                ("evaluation_cache_capacity", "Exact-evaluation cache capacity"),
            ),
            self.calo_numeric_controls,
        )

        operations = self._add_form_card(
            layout,
            "Exact continuation and historical-learning restrictions",
            "Checkpointing is operational. Historical priors and cross-algorithm warm starts "
            "remain off for strict comparisons.",
        )
        self.calo_checkpoint_interval = self._integer_control(1, 2_000_000_000)
        self.calo_historical_priors = QCheckBox("Use historical parameter priors")
        self.calo_cross_warm_start = QCheckBox("Use cross-algorithm warm start")
        self.calo_historical_priors.setEnabled(False)
        self.calo_cross_warm_start.setEnabled(False)
        operations.addRow(
            "Checkpoint interval (counted evaluations)", self.calo_checkpoint_interval
        )
        operations.addRow("Historical parameter priors", self.calo_historical_priors)
        operations.addRow("Cross-algorithm warm start", self.calo_cross_warm_start)

        tsh = self._add_form_card(
            layout,
            "TSH-CALO settings",
            "Runtime choices are saved now. Policy identity, feature flags, qualification, and "
            "activation remain immutable CALO Intelligence authority.",
        )
        self.tsh_policy_status = self._readonly_value()
        self.tsh_policy_identity = self._readonly_value()
        self.tsh_policy_checksum = self._readonly_value()
        self.tsh_feature_flags = self._readonly_value()
        self.tsh_deterministic = QCheckBox("Use deterministic policy inference")
        self.tsh_inference_device = QComboBox()
        self.tsh_inference_device.addItems(("auto", "cuda", "cpu"))
        self.tsh_allow_cpu_fallback = QCheckBox(
            "Allow policy-inference CPU fallback"
        )
        self.tsh_baseline_fallback = QCheckBox("Permit rule-based baseline fallback")
        self.tsh_allow_cpu_fallback.setEnabled(False)
        self.tsh_baseline_fallback.setEnabled(False)
        self.tsh_checkpoint_interval = self._integer_control(1, 2_000_000_000)
        tsh.addRow("Governing-policy readiness", self.tsh_policy_status)
        tsh.addRow("Bound policy", self.tsh_policy_identity)
        tsh.addRow("Policy SHA-256", self.tsh_policy_checksum)
        tsh.addRow("Immutable feature flags", self.tsh_feature_flags)
        tsh.addRow("Inference mode", self.tsh_deterministic)
        tsh.addRow("Inference device", self.tsh_inference_device)
        tsh.addRow("CPU fallback", self.tsh_allow_cpu_fallback)
        tsh.addRow("Baseline fallback", self.tsh_baseline_fallback)
        tsh.addRow(
            "Checkpoint interval (counted evaluations)", self.tsh_checkpoint_interval
        )

        tsh_search = self._add_form_card(
            layout,
            "TSH-CALO search controls",
            "These parameters configure the optimizer around the immutable selected policy; "
            "they do not modify or retrain that policy.",
        )
        self._add_numeric_rows(
            tsh_search,
            (
                ("epsilon_quantile", "Initial epsilon quantile"),
                ("epsilon_control_fraction", "Epsilon control fraction"),
                ("epsilon_exponent", "Epsilon decay exponent"),
                ("memory_capacity", "Success-memory capacity"),
                ("memory_decay", "Memory decay"),
                ("credit_decay", "Contextual-credit decay"),
                ("credit_floor", "Credit floor"),
                ("group_credit_decay", "Variable-group credit decay"),
                ("max_learning_lane_fraction", "Maximum learning-lane fraction"),
                ("feasible_archive_capacity", "Feasible archive capacity"),
                ("boundary_archive_capacity", "Boundary archive capacity"),
                ("stagnation_window", "Stagnation window"),
                ("recovery_diversity_threshold", "Recovery-diversity threshold"),
                ("recovery_fraction", "Recovery fraction"),
                ("precision_start_radius", "Precision starting radius"),
                ("precision_min_radius", "Precision minimum radius"),
                ("precision_max_radius", "Precision maximum radius"),
                ("bandit_window_size", "Contextual-bandit window"),
                ("bandit_exploration", "Contextual-bandit exploration"),
            ),
            self.tsh_numeric_controls,
        )

        review = SectionCard(
            "Configuration review",
            "Save validates and writes CALO and TSH-CALO settings to the experiment "
            "configuration. It does not start scientific work.",
        )
        self.settings_status = QLabel("No settings changes have been saved in this view.")
        self.settings_status.setWordWrap(True)
        review.layout_root.addWidget(self.settings_status)
        buttons = QHBoxLayout()
        self.save_settings_button = QPushButton("Save CALO and TSH-CALO settings")
        self.save_settings_button.setObjectName("PrimaryButton")
        validate = QPushButton("Validate settings")
        restore = QPushButton("Restore canonical settings")
        self.save_settings_button.clicked.connect(self.save_calo_settings)
        validate.clicked.connect(self.validate_calo_settings)
        restore.clicked.connect(self.restore_calo_settings)
        buttons.addWidget(self.save_settings_button)
        buttons.addWidget(validate)
        buttons.addWidget(restore)
        buttons.addStretch(1)
        review.layout_root.addLayout(buttons)
        layout.addWidget(review)
        layout.addStretch(1)
        return page

    def show_context(self, context: str) -> None:
        settings = str(context) == "features"
        self.content_stack.setCurrentWidget(self.settings_page if settings else self.algorithm_page)

    def refresh(self) -> None:
        """Reload the saved in-memory experiment configuration without applying form edits."""
        self.load_from_config(self.state.config)
        controller = self.state.execution_control.controller()
        unlocked = str(controller["controller"]) == "none"
        resumable = False
        for kind in ("workspace", "individual_experiment"):
            plan = self.state.execution_control.active_plan(kind)
            if plan is not None and str(plan["lifecycle_state"]) in {
                "paused",
                "interrupted_resumable",
            }:
                resumable = True
                break
        stage_mutable = unlocked and not resumable
        self.submit_algorithms_button.setEnabled(stage_mutable)
        self.reset_algorithms_button.setEnabled(stage_mutable)
        reason = (
            ""
            if stage_mutable
            else "The submitted stage is locked by an active or resumable execution plan."
        )
        self.submit_algorithms_button.setToolTip(reason)
        self.reset_algorithms_button.setToolTip(reason)

    def _canonical_calo_values(self) -> dict:
        values = self._safe_defaults("CALO", SPECS["CALO"])
        values.update(CALO_NUMERIC_DEFAULTS)
        values.update(
            {
                "calo_profile": "canonical",
                "use_ai": False,
                "strict_policy_binding": False,
                "allow_unqualified_policy": False,
                "strict_benchmark_mode": True,
                "use_historical_parameter_priors": False,
                "use_cross_algorithm_warm_start": False,
            }
        )
        return values

    @staticmethod
    def _set_numeric(control: QSpinBox | QDoubleSpinBox, value: object) -> None:
        if isinstance(control, QSpinBox):
            control.setValue(int(value))
        else:
            control.setValue(float(value))

    @staticmethod
    def _numeric_value(control: QSpinBox | QDoubleSpinBox) -> int | float:
        return (
            int(control.value())
            if isinstance(control, QSpinBox)
            else float(control.value())
        )

    def _apply_calo_profile_mode(self, _index: int = -1) -> None:
        canonical = self.calo_profile.currentData() == "canonical"
        if canonical and not self._loading:
            defaults = self._canonical_calo_values()
            for key, control in self.calo_component_controls.items():
                control.setChecked(bool(defaults[key]))
            for key, control in self.calo_numeric_controls.items():
                self._set_numeric(control, defaults[key])
        for control in (
            *self.calo_component_controls.values(),
            *self.calo_numeric_controls.values(),
        ):
            control.setEnabled(not canonical)

    def _calo_is_canonical(self, parameters: dict) -> bool:
        defaults = self._canonical_calo_values()
        keys = set(self.calo_component_controls) | set(self.calo_numeric_controls)
        return all(
            parameters.get(key, defaults[key]) == defaults[key] for key in keys
        ) and not any(
            bool(parameters.get(key, False))
            for key in (
                "use_historical_parameter_priors",
                "use_cross_algorithm_warm_start",
            )
        )

    def load_from_config(self, config) -> None:
        self._loading = True
        try:
            for row, (name, spec) in enumerate(self.specs.items()):
                self.table.item(row, 0).setCheckState(
                    Qt.CheckState.Checked
                    if name in config.algorithms
                    else Qt.CheckState.Unchecked
                )
                if name not in {"CALO", "TSH-CALO"}:
                    parameters = {
                        **self._safe_defaults(name, spec),
                        **config.algorithm_parameters.get(name, {}),
                    }
                    self.table.item(row, 3).setText(json.dumps(parameters, sort_keys=True))
            self._fit_algorithm_table_to_entries()
            self._update_algorithm_stage_status(config.algorithms)

            calo = {
                **self._canonical_calo_values(),
                **config.algorithm_parameters.get("CALO", {}),
            }
            self.calo_profile.setCurrentIndex(0 if self._calo_is_canonical(calo) else 1)
            for key, control in self.calo_component_controls.items():
                control.setChecked(bool(calo[key]))
            for key, control in self.calo_numeric_controls.items():
                self._set_numeric(control, calo[key])
            self.calo_checkpoint_interval.setValue(
                max(1, int(calo.get("checkpoint_interval_evaluations", 500) or 500))
            )

            tsh = {
                **dict(POLICY_GATED_SPECS["TSH-CALO"].default_parameters),
                **TSH_CALO_NUMERIC_DEFAULTS,
                **config.algorithm_parameters.get("TSH-CALO", {}),
            }
            self.tsh_deterministic.setChecked(bool(tsh.get("deterministic_policy", False)))
            device = str(tsh.get("inference_device", "auto") or "auto").lower()
            self.tsh_inference_device.setCurrentText(
                device if device in {"auto", "cuda", "cpu"} else "auto"
            )
            self.tsh_allow_cpu_fallback.setChecked(False)
            self.tsh_baseline_fallback.setChecked(False)
            self.tsh_checkpoint_interval.setValue(
                max(1, int(tsh.get("checkpoint_interval_evaluations", 500) or 500))
            )
            for key, control in self.tsh_numeric_controls.items():
                self._set_numeric(control, tsh[key])
            self._update_policy_summary(tsh)
            self._update_fingerprint(calo, tsh)
        finally:
            self._loading = False
            self._apply_calo_profile_mode()
            self._apply_policy_gate()

    def _update_policy_summary(self, parameters: dict | None = None) -> None:
        values = dict(
            parameters or self.state.config.algorithm_parameters.get("TSH-CALO", {})
        )
        status = self.state.governing_policy_status()
        self.tsh_policy_status.setText(
            (
                f"Ready · {status.policy_name} · grade {status.grade}"
                if status.ready
                else str(status.reason or "No verified governing TSH-CALO policy is ready.")
            )
        )
        self.tsh_policy_identity.setText(
            str(values.get("policy_name") or values.get("policy_id") or "Not bound")
        )
        checksum = str(values.get("policy_sha256", "") or "")
        self.tsh_policy_checksum.setText(checksum if checksum else "Not bound")
        flags = dict(values.get("policy_feature_flags", {}) or {})
        self.tsh_feature_flags.setText(
            json.dumps(flags, sort_keys=True)
            if flags
            else "Provided by the immutable bound policy"
        )

    def _apply_policy_gate(self) -> None:
        status = self.state.governing_policy_status()
        ready = bool(status.ready and status.algorithm_id == "TSH-CALO")
        reason = (
            "TSH-CALO requires a verified compatible policy that has been selected for "
            "experiments."
        )
        for row, name in enumerate(self.specs):
            if name not in POLICY_GATED_SPECS:
                continue
            use = self.table.item(row, 0)
            if ready:
                use.setFlags(use.flags() | Qt.ItemFlag.ItemIsEnabled)
                use.setToolTip("")
            else:
                use.setCheckState(Qt.CheckState.Unchecked)
                use.setFlags(use.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                use.setToolTip(reason)
        self._update_policy_summary()

    @staticmethod
    def _validate_numeric_values(values: dict, *, label: str) -> None:
        for key, value in values.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{label} {key} must be finite")
        minimum = float(values["precision_min_radius"])
        start = float(values["precision_start_radius"])
        maximum = float(values["precision_max_radius"])
        if not 0.0 < minimum <= start <= maximum <= 1.0:
            raise ValueError(
                f"{label} precision radii must satisfy 0 < minimum <= start <= maximum <= 1"
            )
        if float(values["credit_floor"]) > float(values["credit_decay"]):
            raise ValueError(f"{label} credit floor cannot exceed contextual-credit decay")

    def _collect_calo_settings(self) -> dict:
        existing = dict(self.state.config.algorithm_parameters.get("CALO", {}))
        for key in POLICY_BINDING_FIELDS:
            existing.pop(key, None)
        components = {
            key: bool(control.isChecked())
            for key, control in self.calo_component_controls.items()
        }
        numeric = {
            key: self._numeric_value(control)
            for key, control in self.calo_numeric_controls.items()
        }
        self._validate_numeric_values(numeric, label="CALO")
        existing.update(components)
        existing.update(numeric)
        existing.update(
            {
                "calo_profile": str(self.calo_profile.currentData()),
                "use_ai": False,
                "strict_policy_binding": False,
                "allow_unqualified_policy": False,
                "strict_benchmark_mode": True,
                "use_historical_parameter_priors": False,
                "use_cross_algorithm_warm_start": False,
                "checkpoint_interval_evaluations": int(
                    self.calo_checkpoint_interval.value()
                ),
            }
        )
        return existing

    def _collect_tsh_settings(self) -> dict:
        existing = dict(self.state.config.algorithm_parameters.get("TSH-CALO", {}))
        numeric = {
            key: self._numeric_value(control)
            for key, control in self.tsh_numeric_controls.items()
        }
        self._validate_numeric_values(numeric, label="TSH-CALO")
        existing.update(numeric)
        existing.update(
            {
                "deterministic_policy": bool(self.tsh_deterministic.isChecked()),
                "inference_device": self.tsh_inference_device.currentText(),
                "allow_unqualified_policy": False,
                "allow_cpu_fallback": False,
                "baseline_fallback_permitted": False,
                "strict_policy_binding": True,
                "checkpoint_interval_evaluations": int(
                    self.tsh_checkpoint_interval.value()
                ),
            }
        )
        status = self.state.governing_policy_status()
        existing["use_ai"] = bool(status.ready and status.algorithm_id == "TSH-CALO")
        return existing

    def validate_calo_settings(self) -> bool:
        try:
            calo = self._collect_calo_settings()
            tsh = self._collect_tsh_settings()
            if calo.get("calo_profile") == "canonical" and not self._calo_is_canonical(
                calo
            ):
                raise ValueError("Canonical CALO profile differs from its registered defaults")
        except Exception as exc:
            show_error(
                self,
                "CALO settings are not valid",
                "Correct the CALO or TSH-CALO parameter relationship.",
                exc,
                source="CALO settings",
            )
            self.settings_status.setText(f"Validation failed · {exc}")
            return False
        self.settings_status.setText(
            "Settings are valid. Save them to bind these values to the current experiment "
            "configuration."
        )
        self._update_fingerprint(calo, tsh)
        return True

    def save_calo_settings(self) -> None:
        if not self.validate_calo_settings():
            return
        calo = self._collect_calo_settings()
        tsh = self._collect_tsh_settings()
        self.state.config.algorithm_parameters["CALO"] = calo
        self.state.config.algorithm_parameters["TSH-CALO"] = tsh
        self.state.update_config()
        self._update_fingerprint(calo, tsh)
        self.settings_status.setText(
            "Saved to the current experiment configuration · no experiment was started."
        )
        self.saved.emit("CALO and TSH-CALO settings saved for the current experiment")

    def _update_fingerprint(self, calo: dict, tsh: dict) -> None:
        payload = json.dumps(
            {"CALO": calo, "TSH-CALO": tsh},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        self.calo_fingerprint.setText(hashlib.sha256(payload).hexdigest())

    def restore_calo_settings(self) -> None:
        self._loading = True
        try:
            defaults = self._canonical_calo_values()
            self.calo_profile.setCurrentIndex(0)
            for key, control in self.calo_component_controls.items():
                control.setChecked(bool(defaults[key]))
            for key, control in self.calo_numeric_controls.items():
                self._set_numeric(control, defaults[key])
            self.calo_checkpoint_interval.setValue(500)
            tsh = {
                **dict(POLICY_GATED_SPECS["TSH-CALO"].default_parameters),
                **TSH_CALO_NUMERIC_DEFAULTS,
            }
            self.tsh_deterministic.setChecked(bool(tsh["deterministic_policy"]))
            self.tsh_inference_device.setCurrentText(str(tsh["inference_device"]))
            self.tsh_checkpoint_interval.setValue(500)
            for key, control in self.tsh_numeric_controls.items():
                self._set_numeric(control, tsh[key])
            self.settings_status.setText(
                "Canonical values restored in the form. Select Save to apply them."
            )
        finally:
            self._loading = False
            self._apply_calo_profile_mode()

    def _update_algorithm_stage_status(self, algorithms=None) -> None:
        stage = self.state.execution_control.active_stage()
        if stage is not None:
            self.algorithm_stage_status.setText(
                f"Submitted stage {stage.stage_id}: {', '.join(stage.algorithm_names)} · "
                f"content SHA-256 {stage.content_sha256[:16]}…"
            )
        else:
            self.algorithm_stage_status.setText(
                "No algorithms are staged for the experiment. Select one or more and submit."
            )

    def _algorithm_draft_changed(self, _item: QTableWidgetItem) -> None:
        if self._loading:
            return
        stage = self.state.execution_control.active_stage()
        if stage is not None:
            self.algorithm_stage_status.setText(
                "Draft selection changed. Currently staged: "
                f"{', '.join(stage.algorithm_names)}. Submit algorithms to replace the experiment staging."
            )
        else:
            self.algorithm_stage_status.setText(
                "Draft selection changed. No algorithms are currently staged; submit the fresh "
                "selection for the experiment."
            )

    def submit_algorithm_selection(self) -> None:
        selected: list[str] = []
        parameters = {
            name: dict(values)
            for name, values in self.state.config.algorithm_parameters.items()
        }
        try:
            for row in range(self.table.rowCount()):
                name = self.table.item(row, 1).text()
                if name not in {"CALO", "TSH-CALO"}:
                    parsed = json.loads(self.table.item(row, 3).text() or "{}")
                    if not isinstance(parsed, dict):
                        raise ValueError(f"{name} parameters must be a JSON object")
                    parameters[name] = parsed
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    selected.append(name)
            if not selected:
                raise ValueError("Select at least one primary optimizer.")
        except Exception as exc:
            show_error(
                self,
                "Algorithm selection was not submitted",
                "Review the selected algorithms and comparator parameter values.",
                exc,
                source="algorithm selection",
            )
            return
        parameters["CALO"] = {
            **self._canonical_calo_values(),
            **parameters.get("CALO", {}),
        }
        parameters["TSH-CALO"] = {
            **dict(POLICY_GATED_SPECS["TSH-CALO"].default_parameters),
            **TSH_CALO_NUMERIC_DEFAULTS,
            **parameters.get("TSH-CALO", {}),
        }
        candidate = deepcopy(self.state.config)
        candidate.algorithms = selected
        candidate.algorithm_parameters = parameters
        try:
            stage = self.state.execution_control.submit_algorithm_stage(candidate)
        except Exception as exc:
            show_error(
                self,
                "Algorithm selection was not submitted",
                "The current execution plan must release the submitted stage before it can change.",
                exc,
                source="algorithm staging",
            )
            return
        self.state.config = candidate
        self.state.update_config()
        self.state.notify_execution_state_changed()
        self.stage_completed.emit()
        self.saved.emit(
            f"Algorithms submitted for experiment use: {', '.join(selected)} · {stage.stage_id}"
        )

    def reset_algorithm_selection(self) -> None:
        try:
            self.state.execution_control.discard_algorithm_stage()
        except Exception as exc:
            show_error(
                self,
                "Algorithm staging was not reset",
                "Finish, cancel, or close the exact controlling plan before resetting algorithms.",
                exc,
                source="algorithm staging",
            )
            return
        existing = self.state.config.algorithm_parameters
        parameters = {
            "CALO": {
                **self._canonical_calo_values(),
                **dict(existing.get("CALO", {})),
            },
            "TSH-CALO": {
                **dict(POLICY_GATED_SPECS["TSH-CALO"].default_parameters),
                **TSH_CALO_NUMERIC_DEFAULTS,
                **dict(existing.get("TSH-CALO", {})),
            },
        }
        parameters.update(
            {
                name: self._safe_defaults(name, spec)
                for name, spec in self.specs.items()
                if name not in {"CALO", "TSH-CALO"}
            }
        )
        self.state.config.algorithms = []
        self.state.config.algorithm_parameters = parameters
        self.state.update_config()
        self.state.notify_execution_state_changed()
        self.stage_discarded.emit()
        self.saved.emit(
            "Algorithm staging discarded; select and submit a fresh experiment set"
        )

    # Compatibility names retained for callers from the previous AlgorithmsPanel surface.
    apply = submit_algorithm_selection
    restore_defaults = reset_algorithm_selection
