"""Portfolio-first evidence planning immediately after algorithm selection."""

from __future__ import annotations

import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.user_feedback import log_technical_error, show_error
from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage
from calo_rpd_studio.gui.widgets.workspace_tabs import WorkspaceTabs
from calo_rpd_studio.portfolio.catalog import categories
from calo_rpd_studio.portfolio.models import (
    ArticlePreset,
    EvidenceProfile,
    PortfolioConfig,
    PortfolioKind,
    StorageProfile,
)
from calo_rpd_studio.portfolio.study_planning import PortfolioGoalPlanner


class PortfolioManagerPanel(ScrollablePage):
    stage_completed = pyqtSignal()
    study_requested = pyqtSignal()

    def __init__(self, state, parent=None) -> None:
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        self._items: dict[str, QTreeWidgetItem] = {}
        self._algorithm_items: dict[str, QTreeWidgetItem] = {}
        self._loaded_algorithm_stage_id = ""

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(15)
        layout.addWidget(
            PageHeader(
                "Portfolio Manager",
                "Define the broad evidence and deliverable goal. Study will recommend the concrete cases, repetitions, scenarios, and execution setup after this goal is applied.",
            )
        )

        definition = QWidget()
        definition_layout = QHBoxLayout(definition)
        definition_layout.setContentsMargins(18, 18, 18, 18)
        definition_layout.setSpacing(16)
        study_scope = QGroupBox("Goal")
        study_form = QFormLayout(study_scope)
        output_scope = QGroupBox("Output and storage")
        output_form = QFormLayout(output_scope)
        self.kind = QComboBox()
        self.kind.addItem("Single-run diagnostic portfolio", PortfolioKind.SINGLE_RUN.value)
        self.kind.addItem(
            "Overall repeated experiment portfolio", PortfolioKind.OVERALL_EXPERIMENT.value
        )
        self.profile = QComboBox()
        self.profile.addItem("Diagnostic minimum", EvidenceProfile.DIAGNOSTIC.value)
        self.profile.addItem("Exploratory minimum", EvidenceProfile.EXPLORATORY.value)
        self.profile.addItem("Rigorous minimum", EvidenceProfile.JOURNAL.value)
        self.profile.addItem("Comprehensive minimum", EvidenceProfile.TRANSACTIONS.value)
        self.profile.addItem("Powered/custom run plan", EvidenceProfile.CUSTOM.value)
        self.preset = QComboBox()
        self.preset.addItem("No output preset", ArticlePreset.NONE.value)
        self.preset.addItem("TLBO/MTLBO comparison", ArticlePreset.TLBO_MTLBO.value)
        self.preset.addItem("CALO deterministic study", ArticlePreset.CALO_DETERMINISTIC.value)
        self.preset.addItem("CALO robust study", ArticlePreset.CALO_ROBUST.value)
        self.preset.addItem(
            "Experience and accelerator study", ArticlePreset.CALO_TRANSFER_ACCELERATOR.value
        )
        self.storage = QComboBox()
        self.storage.addItem("Minimal diagnostic", StorageProfile.MINIMAL.value)
        self.storage.addItem("Full single-run diagnostics", StorageProfile.FULL_SINGLE_RUN.value)
        self.storage.addItem(
            "Repeated-run statistical evidence", StorageProfile.REPEATED_STATISTICS.value
        )
        self.storage.addItem("Full robust scenario evidence", StorageProfile.ROBUST_FULL.value)
        self.goal_summary = QLineEdit()
        self.goal_summary.setPlaceholderText(
            "Plain-language ultimate target, for example: compare feasible solution quality"
        )
        study_form.addRow("Portfolio type", self.kind)
        study_form.addRow("Evidence strength", self.profile)
        study_form.addRow("Ultimate target", self.goal_summary)
        output_form.addRow("Output preset", self.preset)
        output_form.addRow("Storage profile", self.storage)
        definition_layout.addWidget(study_scope, 1)
        definition_layout.addWidget(output_scope, 1)

        algorithm_box = QWidget()
        algorithm_layout = QVBoxLayout(algorithm_box)
        algorithm_layout.setContentsMargins(18, 18, 18, 18)
        algorithm_layout.setSpacing(10)
        self.algorithm_stage_status = QLabel()
        self.algorithm_stage_status.setWordWrap(True)
        self.algorithm_stage_status.setObjectName("InfoText")
        algorithm_layout.addWidget(self.algorithm_stage_status)
        self.algorithm_filter = QTreeWidget()
        self.algorithm_filter.setObjectName("WorkspaceStudyAlgorithmFilter")
        self.algorithm_filter.setAccessibleName("Algorithms included in this Workspace study")
        self.algorithm_filter.setHeaderLabels(["Include", "Submitted algorithm", "Parameters"])
        self.algorithm_filter.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.algorithm_filter.header().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.algorithm_filter.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        algorithm_layout.addWidget(self.algorithm_filter, 1)
        algorithm_actions = QHBoxLayout()
        use_all_algorithms = QPushButton("Use all staged algorithms")
        clear_algorithms = QPushButton("Clear comparison scope")
        use_all_algorithms.clicked.connect(self._use_all_staged_algorithms)
        clear_algorithms.clicked.connect(self._clear_study_filter)
        algorithm_actions.addWidget(use_all_algorithms)
        algorithm_actions.addWidget(clear_algorithms)
        algorithm_actions.addStretch(1)
        algorithm_layout.addLayout(algorithm_actions)

        output_box = QWidget()
        output_layout = QVBoxLayout(output_box)
        output_layout.setContentsMargins(18, 18, 18, 18)
        explanation = QLabel(
            "Select only the outputs needed. Unavailable outputs are retained in the plan with an explicit reason rather than causing unnecessary evaluations."
        )
        explanation.setWordWrap(True)
        output_layout.addWidget(explanation)
        self.require_validation = QCheckBox(
            "Require independent validation for publication-facing deliverables"
        )
        self.require_validation.setChecked(True)
        output_layout.addWidget(self.require_validation)
        self.outputs = QTreeWidget()
        self.outputs.setObjectName("PortfolioRequestedOutputs")
        self.outputs.setAccessibleName("Requested figures, tables, and evidence")
        self.outputs.setHeaderLabels(["Generate", "Output", "Minimum evidence"])
        self.outputs.setAlternatingRowColors(True)
        output_header = self.outputs.header()
        output_header.setStretchLastSection(False)
        output_header.setMinimumSectionSize(72)
        output_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        output_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        output_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        generate_width = self.outputs.fontMetrics().horizontalAdvance("Generate") + 32
        output_header.resizeSection(0, max(132, generate_width))
        for category, requirements in categories().items():
            parent_item = QTreeWidgetItem(["", category, ""])
            parent_item.setFlags(parent_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.outputs.addTopLevelItem(parent_item)
            for req in requirements:
                minimum = f"{req.minimum_runs} run(s), {req.minimum_algorithms} algorithm(s)"
                if req.minimum_blocks > 1:
                    minimum += f", {req.minimum_blocks} blocks"
                child = QTreeWidgetItem(["", req.label, minimum])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                child.setCheckState(0, Qt.CheckState.Unchecked)
                child.setData(0, Qt.ItemDataRole.UserRole, req.key)
                parent_item.addChild(child)
                self._items[req.key] = child
            parent_item.setExpanded(True)
        output_layout.addWidget(self.outputs, 1)
        select_row = QHBoxLayout()
        select_defaults = QPushButton("Select recommended")
        clear = QPushButton("Clear all")
        select_defaults.clicked.connect(self._select_recommended)
        clear.clicked.connect(self._clear_outputs)
        select_row.addWidget(select_defaults)
        select_row.addWidget(clear)
        select_row.addStretch(1)
        output_layout.addLayout(select_row)

        plan_box = QWidget()
        plan_layout = QVBoxLayout(plan_box)
        plan_layout.setContentsMargins(18, 18, 18, 18)
        self.plan_summary = QLabel()
        self.plan_summary.setWordWrap(True)
        self.plan_detail = QLabel()
        self.plan_detail.setWordWrap(True)
        self.plan_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        plan_layout.addWidget(self.plan_summary)
        plan_layout.addWidget(self.plan_detail)
        buttons = QHBoxLayout()
        preview = QPushButton("Preview portfolio goal")
        apply_button = QPushButton("Apply portfolio goal")
        apply_button.setObjectName("PrimaryButton")
        self.open_study = QPushButton("Open Workspace Study")
        self.open_study.setEnabled(False)
        preview.clicked.connect(self.refresh_plan)
        apply_button.clicked.connect(self.apply)
        self.open_study.clicked.connect(self.study_requested.emit)
        buttons.addWidget(preview)
        buttons.addWidget(apply_button)
        buttons.addWidget(self.open_study)
        buttons.addStretch(1)
        plan_layout.addLayout(buttons)

        self.section_tabs = WorkspaceTabs("Portfolio planning sections")
        self.section_tabs.add_section(
            "Goal",
            definition,
            "Choose the broad ultimate evidence target, rigor, preset, and evidence class.",
        )
        self.section_tabs.add_section(
            "Comparison scope",
            algorithm_box,
            "Choose a non-empty comparison subset of the submitted stage without changing it.",
        )
        self.section_tabs.add_section(
            "Deliverables and evidence",
            output_box,
            "Select the figures, tables, and evidence required from the study.",
        )
        self.section_tabs.add_section(
            "Goal summary",
            plan_box,
            "Preview intrinsic constraints and apply only the immutable broad Portfolio goal.",
        )
        self.section_tabs.setMinimumHeight(520)
        layout.addWidget(self.section_tabs, 1)

        for widget in (self.kind, self.profile, self.preset, self.storage):
            widget.currentIndexChanged.connect(self._controls_changed)
        self.goal_summary.textChanged.connect(self._controls_changed)
        self.outputs.itemChanged.connect(lambda *_: self.refresh_plan())
        self.algorithm_filter.itemChanged.connect(lambda *_: self.refresh_plan())
        self.state.config_changed.connect(lambda _: self.refresh())
        self.state.execution_state_changed.connect(lambda _: self.refresh())
        self.refresh()

    def _refresh_algorithm_stage(self) -> None:
        stage = self.state.execution_control.active_stage()
        previous = {
            name
            for name, item in self._algorithm_items.items()
            if item.checkState(0) == Qt.CheckState.Checked
        }
        self.algorithm_filter.blockSignals(True)
        self.algorithm_filter.clear()
        self._algorithm_items.clear()
        try:
            if stage is None:
                self._loaded_algorithm_stage_id = ""
                self.algorithm_stage_status.setText(
                    "No submitted algorithm stage is available. Return to Algorithms, select at least one optimizer, and submit it."
                )
                return
            self.algorithm_stage_status.setText(
                f"Submitted stage {stage.stage_id} · {len(stage.algorithm_names)} algorithm(s) · "
                f"content SHA-256 {stage.content_sha256[:16]}…"
            )
            if self._loaded_algorithm_stage_id == stage.stage_id:
                selected = previous
            else:
                selected = set(stage.algorithm_names)
                workspace_plan = self.state.execution_control.active_plan("workspace")
                if (
                    workspace_plan is not None
                    and str(workspace_plan.get("algorithm_stage_id", "")) == stage.stage_id
                    and str(workspace_plan["design"].get("algorithm_stage_sha256", ""))
                    == stage.content_sha256
                ):
                    retained = tuple(
                        str(name)
                        for name in workspace_plan["design"].get("study_algorithm_names", [])
                    )
                    if retained and set(retained).issubset(stage.algorithm_names):
                        selected = set(retained)
                elif workspace_plan is None:
                    active_goal = self.state.database.get_active_portfolio_goal()
                    if (
                        active_goal is not None
                        and str(active_goal["algorithm_stage_id"]) == stage.stage_id
                        and str(active_goal["algorithm_stage_sha256"]) == stage.content_sha256
                    ):
                        retained = tuple(
                            str(name)
                            for name in active_goal["content"].get(
                                "selected_algorithm_names", []
                            )
                        )
                        if retained and set(retained).issubset(stage.algorithm_names):
                            selected = set(retained)
            for name in stage.algorithm_names:
                parameters = stage.algorithm_parameters.get(name, {})
                item = QTreeWidgetItem(
                    ["", name, json.dumps(parameters, sort_keys=True, separators=(",", ":"))]
                )
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    0,
                    Qt.CheckState.Checked if name in selected else Qt.CheckState.Unchecked,
                )
                self.algorithm_filter.addTopLevelItem(item)
                self._algorithm_items[name] = item
            self._loaded_algorithm_stage_id = stage.stage_id
        finally:
            self.algorithm_filter.blockSignals(False)

    def _selected_study_algorithms(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, item in self._algorithm_items.items()
            if item.checkState(0) == Qt.CheckState.Checked
        )

    def _stage_matches_current_config(self, stage) -> bool:
        payload = self.state.config.to_dict()
        parameters = dict(payload.get("algorithm_parameters", {}) or {})
        selected_parameters = {
            name: dict(parameters.get(name, {}) or {}) for name in stage.algorithm_names
        }
        return (
            tuple(str(name) for name in self.state.config.algorithms) == stage.algorithm_names
            and selected_parameters == stage.algorithm_parameters
        )

    def _use_all_staged_algorithms(self) -> None:
        self.algorithm_filter.blockSignals(True)
        try:
            for item in self._algorithm_items.values():
                item.setCheckState(0, Qt.CheckState.Checked)
        finally:
            self.algorithm_filter.blockSignals(False)
        self.refresh_plan()

    def _clear_study_filter(self) -> None:
        self.algorithm_filter.blockSignals(True)
        try:
            for item in self._algorithm_items.values():
                item.setCheckState(0, Qt.CheckState.Unchecked)
        finally:
            self.algorithm_filter.blockSignals(False)
        self.refresh_plan()

    def _selected_outputs(self) -> list[str]:
        return [
            key for key, item in self._items.items() if item.checkState(0) == Qt.CheckState.Checked
        ]

    def _set_outputs(self, keys: list[str]) -> None:
        self.outputs.blockSignals(True)
        try:
            selected = set(keys)
            for key, item in self._items.items():
                item.setCheckState(
                    0, Qt.CheckState.Checked if key in selected else Qt.CheckState.Unchecked
                )
        finally:
            self.outputs.blockSignals(False)

    def _select_recommended(self) -> None:
        kind = PortfolioKind(str(self.kind.currentData()))
        if kind is PortfolioKind.SINGLE_RUN:
            from calo_rpd_studio.portfolio.models import DEFAULT_SINGLE_RUN_OUTPUTS

            self._set_outputs(DEFAULT_SINGLE_RUN_OUTPUTS)
        else:
            from calo_rpd_studio.portfolio.models import DEFAULT_EXPERIMENT_OUTPUTS

            self._set_outputs(DEFAULT_EXPERIMENT_OUTPUTS)
        self.refresh_plan()

    def _clear_outputs(self) -> None:
        self._set_outputs([])
        self.refresh_plan()

    def _controls_changed(self, *_args) -> None:
        kind = PortfolioKind(str(self.kind.currentData()))
        if kind is PortfolioKind.SINGLE_RUN:
            self.profile.setCurrentIndex(self.profile.findData(EvidenceProfile.DIAGNOSTIC.value))
            self.profile.setEnabled(False)
            self.storage.setCurrentIndex(
                self.storage.findData(StorageProfile.FULL_SINGLE_RUN.value)
            )
        else:
            self.profile.setEnabled(True)
        self.refresh_plan()

    def _build_config(self) -> PortfolioConfig:
        prior = getattr(self.state.config, "portfolio", PortfolioConfig())
        return PortfolioConfig(
            kind=PortfolioKind(str(self.kind.currentData())),
            evidence_profile=EvidenceProfile(str(self.profile.currentData())),
            article_preset=ArticlePreset(str(self.preset.currentData())),
            requested_outputs=self._selected_outputs(),
            # Legacy execution-owned fields remain serializable for old records but are not
            # Portfolio controls and are excluded from AppliedPortfolioGoal content.
            custom_runs=int(prior.custom_runs),
            require_independent_validation=self.require_validation.isChecked(),
            reuse_compatible_results=bool(prior.reuse_compatible_results),
            enable_resume=bool(prior.enable_resume),
            checkpoint_interval_evaluations=int(prior.checkpoint_interval_evaluations),
            storage_profile=StorageProfile(str(self.storage.currentData())),
            name=(
                self.goal_summary.text().strip()
                or (
                    "Single-run diagnostic portfolio"
                    if str(self.kind.currentData()) == PortfolioKind.SINGLE_RUN.value
                    else "Overall experiment portfolio"
                )
            ),
        )

    def refresh_plan(self) -> None:
        stage = self.state.execution_control.active_stage()
        if stage is None:
            self.plan_summary.setText(
                "Submit algorithms first. Portfolio can select a comparison scope only from "
                "the immutable submitted stage. No goal or execution plan has started."
            )
            self.plan_detail.setText(
                "Open Algorithms, choose the permitted pool, and select Submit "
                "algorithms for experiment. No plan or execution has started."
            )
            return
        if not self._stage_matches_current_config(stage):
            retained_names = ", ".join(stage.algorithm_names)
            current_names = ", ".join(str(name) for name in self.state.config.algorithms) or "none"
            self.plan_summary.setText(
                "The submitted algorithm stage does not match the current experiment configuration."
            )
            self.plan_detail.setText(
                f"Retained submitted identities: {retained_names}. Current experiment "
                f"selection: {current_names}. Identities and/or parameters differ. Open "
                "Algorithms, review the current selection and parameters, and submit them "
                "again before applying a Portfolio goal. The retained stage was not changed, "
                "and no goal or execution plan has started."
            )
            return
        portfolio = self._build_config()
        if not portfolio.requested_outputs:
            self.plan_summary.setText("Select at least one output to preview the portfolio plan.")
            self.plan_detail.clear()
            return
        subset = self._selected_study_algorithms()
        if not subset:
            self.plan_summary.setText(
                "Select at least one submitted algorithm for this Workspace study."
            )
            self.plan_detail.setText(
                "The study filter changes only this Workspace draft. It does not "
                "modify the submitted algorithm stage or start execution."
            )
            return
        try:
            goal = PortfolioGoalPlanner.create(portfolio, stage, subset)
            constraints = goal.intrinsic_requirements
            warnings = "\n".join(f"• {item}" for item in goal.intrinsic_warnings) or "None"
            self.plan_summary.setText(
                "Portfolio goal is ready to apply. Study setup is not required here."
            )
            self.plan_detail.setText(
                f"Comparison scope: {', '.join(goal.selected_algorithm_names)}\n"
                f"Requested deliverables: {', '.join(goal.portfolio['requested_outputs'])}\n"
                f"Study hard minimum: {constraints['hard_minimum_runs']} paired run(s), "
                f"{constraints['minimum_benchmark_blocks']} benchmark block(s)\n"
                f"Required stored evidence: {', '.join(constraints['required_storage_fields'])}\n"
                f"Robust Study required: {'yes' if constraints['robust_scenario_required'] else 'no'}\n"
                f"Intrinsic warnings:\n{warnings}\n"
                "No Study plan, audit, controller, campaign, task, or numerical work has been created."
            )
        except ValueError as exc:
            self.plan_summary.setText("The portfolio preview needs another selection.")
            self.plan_detail.setText(str(exc))
        except Exception as exc:
            self.plan_summary.setText(
                "The portfolio plan is incomplete. Review Activity > Logs for details."
            )
            self.plan_detail.clear()
            log_technical_error("portfolio planning", exc)

    def refresh(self) -> None:
        self._refresh_algorithm_stage()
        stage = self.state.execution_control.active_stage()
        portfolio = getattr(self.state.config, "portfolio", PortfolioConfig())
        self.kind.setCurrentIndex(max(0, self.kind.findData(portfolio.kind.value)))
        self.profile.setCurrentIndex(
            max(0, self.profile.findData(portfolio.evidence_profile.value))
        )
        self.preset.setCurrentIndex(max(0, self.preset.findData(portfolio.article_preset.value)))
        self.storage.setCurrentIndex(max(0, self.storage.findData(portfolio.storage_profile.value)))
        self.goal_summary.setText(str(portfolio.name))
        self.require_validation.setChecked(bool(portfolio.require_independent_validation))
        self._set_outputs(list(portfolio.requested_outputs))
        self._controls_changed()
        controller = self.state.execution_control.controller()
        workspace_plan = self.state.execution_control.active_plan("workspace")
        retained_workspace = bool(
            workspace_plan is not None
            and str(workspace_plan["lifecycle_state"])
            in {"staged", "running", "pausing", "paused", "interrupted_resumable"}
        )
        locked = str(controller["controller"]) != "none" or retained_workspace
        self.section_tabs.setEnabled(not locked)
        active_goal = self.state.database.get_active_portfolio_goal()
        goal_current = bool(
            active_goal is not None
            and stage is not None
            and str(active_goal["algorithm_stage_id"]) == stage.stage_id
            and str(active_goal["algorithm_stage_sha256"]) == stage.content_sha256
        )
        self.open_study.setEnabled(goal_current and not locked)
        self.open_study.setVisible(goal_current)
        if locked:
            owner = (
                str(controller["owner_plan_id"])
                if str(controller["owner_plan_id"])
                else str(workspace_plan["id"])
            )
            self.algorithm_stage_status.setText(
                self.algorithm_stage_status.text() + f"\nLocked by retained execution plan {owner}."
            )

    def apply(self) -> None:
        try:
            portfolio = self._build_config()
            stage = self.state.execution_control.active_stage()
            if stage is None:
                raise ValueError(
                    "Submit algorithms first. Portfolio can select a comparison scope only from "
                    "the immutable submitted stage. No goal or execution plan has started."
                )
            if not self._stage_matches_current_config(stage):
                retained_names = ", ".join(stage.algorithm_names)
                current_names = (
                    ", ".join(str(name) for name in self.state.config.algorithms) or "none"
                )
                raise ValueError(
                    f"Submitted stage {stage.stage_id!r} ({stage.content_sha256[:16]}…) retains "
                    f"[{retained_names}], while the current Algorithms draft contains "
                    f"[{current_names}] and/or different parameters. Return to Algorithms, "
                    "review the exact identities and parameters, and submit the stage again. "
                    "No Portfolio goal or Study plan was changed."
                )
            subset = self._selected_study_algorithms()
            if not subset:
                raise ValueError("Select at least one submitted algorithm for this Workspace study")
            goal = PortfolioGoalPlanner.create(portfolio, stage, subset)
            self.state.database.replace_portfolio_goal(goal)
            # Shared configuration retains broad display preferences only. Exact Study choices,
            # including runs/reuse/resume/checkpoints, are intentionally untouched here.
            portfolio.kind = PortfolioKind(str(goal.portfolio["portfolio_type"]))
            portfolio.evidence_profile = EvidenceProfile(
                str(goal.portfolio["evidence_profile"])
            )
            portfolio.requested_outputs = list(goal.portfolio["requested_outputs"])
            portfolio.storage_profile = StorageProfile(
                str(goal.portfolio["storage_or_evidence_class"])
            )
            portfolio.require_independent_validation = bool(
                goal.portfolio["require_independent_validation"]
            )
            self.state.config.portfolio = portfolio
            self.state.config.portfolio_id = ""
            self.state.config.workspace_study_contract = {}
            self.state.update_config()
            self.state.notify_execution_state_changed()
            self.refresh_plan()
            self.stage_completed.emit()
            self.plan_detail.setText(
                f"Portfolio goal applied: {goal.portfolio_goal_id} · SHA-256 "
                f"{goal.content_sha256[:16]}…\nOpen Workspace Study to review recommendations "
                "and choose the concrete setup. No experiment has started."
            )
        except Exception as exc:
            show_error(
                self,
                "Portfolio goal was not applied",
                str(exc),
                exc,
                source="portfolio planning",
            )
