"""Portfolio-first evidence planning immediately after algorithm selection."""

from __future__ import annotations

from dataclasses import asdict

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.portfolio.catalog import OUTPUT_REQUIREMENTS, categories
from calo_rpd_studio.portfolio.fingerprint import stable_sha256
from calo_rpd_studio.portfolio.models import (
    ArticlePreset,
    EvidenceProfile,
    PortfolioConfig,
    PortfolioKind,
    StorageProfile,
)
from calo_rpd_studio.portfolio.planner import PortfolioPlanner
from calo_rpd_studio.gui.widgets.page_header import PageHeader
from calo_rpd_studio.gui.widgets.scrollable_page import ScrollablePage


class PortfolioManagerPanel(ScrollablePage):
    stage_completed = pyqtSignal()

    def __init__(self, state, parent=None) -> None:
        content = QWidget()
        super().__init__(content, parent)
        self.state = state
        self._items: dict[str, QTreeWidgetItem] = {}

        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(15)
        layout.addWidget(
            PageHeader(
                "Portfolio Manager",
                "Choose the evidence portfolio before execution. The planner derives the minimum paired runs, stored fields, validation, statistics, and export tasks required for the selected article outputs.",
            )
        )

        definition = QGroupBox("Portfolio definition")
        form = QFormLayout(definition)
        self.kind = QComboBox()
        self.kind.addItem("Single-run diagnostic portfolio", PortfolioKind.SINGLE_RUN.value)
        self.kind.addItem(
            "Overall repeated experiment portfolio", PortfolioKind.OVERALL_EXPERIMENT.value
        )
        self.profile = QComboBox()
        self.profile.addItem("Diagnostic — 1 run", EvidenceProfile.DIAGNOSTIC.value)
        self.profile.addItem("Exploratory — 10 runs", EvidenceProfile.EXPLORATORY.value)
        self.profile.addItem("Journal — 30 runs", EvidenceProfile.JOURNAL.value)
        self.profile.addItem("Transactions — 50 runs", EvidenceProfile.TRANSACTIONS.value)
        self.profile.addItem("Custom", EvidenceProfile.CUSTOM.value)
        self.custom_runs = QSpinBox()
        self.custom_runs.setRange(1, 1000)
        self.custom_runs.setValue(30)
        self.preset = QComboBox()
        self.preset.addItem("No article preset", ArticlePreset.NONE.value)
        self.preset.addItem("Article 1 — TLBO/MTLBO", ArticlePreset.TLBO_MTLBO.value)
        self.preset.addItem(
            "Article 2 — CALO deterministic", ArticlePreset.CALO_DETERMINISTIC.value
        )
        self.preset.addItem("Article 3 — CALO robust", ArticlePreset.CALO_ROBUST.value)
        self.preset.addItem(
            "Article 4 — experience/accelerator", ArticlePreset.CALO_TRANSFER_ACCELERATOR.value
        )
        self.storage = QComboBox()
        self.storage.addItem("Minimal diagnostic", StorageProfile.MINIMAL.value)
        self.storage.addItem(
            "Full single-run article diagnostics", StorageProfile.FULL_SINGLE_RUN.value
        )
        self.storage.addItem(
            "Repeated-run statistical evidence", StorageProfile.REPEATED_STATISTICS.value
        )
        self.storage.addItem("Full robust scenario evidence", StorageProfile.ROBUST_FULL.value)
        form.addRow("Portfolio type", self.kind)
        form.addRow("Evidence strength", self.profile)
        form.addRow("Custom repeated runs", self.custom_runs)
        form.addRow("Article preset", self.preset)
        form.addRow("Storage profile", self.storage)
        layout.addWidget(definition)

        output_box = QGroupBox("Requested figures, tables, and evidence")
        output_layout = QVBoxLayout(output_box)
        explanation = QLabel(
            "Select only the outputs needed. Unavailable outputs are retained in the plan with an explicit reason rather than causing unnecessary evaluations."
        )
        explanation.setWordWrap(True)
        output_layout.addWidget(explanation)
        self.outputs = QTreeWidget()
        self.outputs.setHeaderLabels(["Generate", "Output", "Minimum evidence"])
        self.outputs.setAlternatingRowColors(True)
        self.outputs.header().setStretchLastSection(False)
        self.outputs.header().resizeSection(0, 90)
        self.outputs.header().resizeSection(1, 420)
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
        layout.addWidget(output_box, 1)

        execution = QGroupBox("Reuse, validation, and resumability")
        exec_form = QFormLayout(execution)
        self.require_validation = QCheckBox(
            "Require independent validation for publication-facing outputs"
        )
        self.reuse = QCheckBox(
            "Reuse exact compatible completed results using scientific fingerprints"
        )
        self.resume = QCheckBox("Enable campaign and job resume")
        self.require_validation.setChecked(True)
        self.reuse.setChecked(True)
        self.resume.setChecked(True)
        exec_form.addRow("", self.require_validation)
        exec_form.addRow("", self.reuse)
        exec_form.addRow("", self.resume)
        layout.addWidget(execution)

        plan_box = QGroupBox("Derived minimal experiment plan")
        plan_layout = QVBoxLayout(plan_box)
        self.plan_summary = QLabel()
        self.plan_summary.setWordWrap(True)
        self.plan_detail = QLabel()
        self.plan_detail.setWordWrap(True)
        self.plan_detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        plan_layout.addWidget(self.plan_summary)
        plan_layout.addWidget(self.plan_detail)
        buttons = QHBoxLayout()
        preview = QPushButton("Preview required work")
        apply_button = QPushButton("Apply portfolio plan")
        apply_button.setObjectName("PrimaryButton")
        preview.clicked.connect(self.refresh_plan)
        apply_button.clicked.connect(self.apply)
        buttons.addWidget(preview)
        buttons.addWidget(apply_button)
        buttons.addStretch(1)
        plan_layout.addLayout(buttons)
        layout.addWidget(plan_box)
        layout.addStretch(1)

        for widget in (self.kind, self.profile, self.preset, self.storage):
            widget.currentIndexChanged.connect(self._controls_changed)
        self.custom_runs.valueChanged.connect(self._controls_changed)
        self.outputs.itemChanged.connect(lambda *_: self.refresh_plan())
        self.state.config_changed.connect(lambda _: self.refresh())
        self.refresh()

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
            self.custom_runs.setEnabled(False)
            self.storage.setCurrentIndex(
                self.storage.findData(StorageProfile.FULL_SINGLE_RUN.value)
            )
        else:
            self.profile.setEnabled(True)
            self.custom_runs.setEnabled(
                str(self.profile.currentData()) == EvidenceProfile.CUSTOM.value
            )
        self.refresh_plan()

    def _build_config(self) -> PortfolioConfig:
        return PortfolioConfig(
            kind=PortfolioKind(str(self.kind.currentData())),
            evidence_profile=EvidenceProfile(str(self.profile.currentData())),
            article_preset=ArticlePreset(str(self.preset.currentData())),
            requested_outputs=self._selected_outputs(),
            custom_runs=int(self.custom_runs.value()),
            require_independent_validation=self.require_validation.isChecked(),
            reuse_compatible_results=self.reuse.isChecked(),
            enable_resume=self.resume.isChecked(),
            storage_profile=StorageProfile(str(self.storage.currentData())),
            name=(
                "Single-run diagnostic portfolio"
                if str(self.kind.currentData()) == PortfolioKind.SINGLE_RUN.value
                else "Overall experiment portfolio"
            ),
        )

    def refresh_plan(self) -> None:
        try:
            portfolio = self._build_config()
            temp_config = self.state.config
            plan = PortfolioPlanner.plan(temp_config, portfolio, benchmark_blocks=1)
            disabled = (
                "\n".join(
                    f"• {OUTPUT_REQUIREMENTS[key].label if key in OUTPUT_REQUIREMENTS else key}: {reason}"
                    for key, reason in plan.disabled_outputs.items()
                )
                or "None"
            )
            fields = ", ".join(plan.required_fields)
            warnings = "\n".join(f"• {item}" for item in plan.warnings) or "None"
            self.plan_summary.setText(plan.summary())
            self.plan_detail.setText(
                f"Required stored evidence: {fields}\n"
                f"Independent validation: {'required' if plan.require_validation else 'not mandatory'}\n"
                f"Unavailable selections:\n{disabled}\n"
                f"Planner warnings:\n{warnings}"
            )
        except Exception as exc:
            self.plan_summary.setText(f"Portfolio plan is incomplete: {exc}")
            self.plan_detail.clear()

    def refresh(self) -> None:
        portfolio = getattr(self.state.config, "portfolio", PortfolioConfig())
        self.kind.setCurrentIndex(max(0, self.kind.findData(portfolio.kind.value)))
        self.profile.setCurrentIndex(
            max(0, self.profile.findData(portfolio.evidence_profile.value))
        )
        self.preset.setCurrentIndex(max(0, self.preset.findData(portfolio.article_preset.value)))
        self.storage.setCurrentIndex(max(0, self.storage.findData(portfolio.storage_profile.value)))
        self.custom_runs.setValue(int(portfolio.custom_runs))
        self.require_validation.setChecked(bool(portfolio.require_independent_validation))
        self.reuse.setChecked(bool(portfolio.reuse_compatible_results))
        self.resume.setChecked(bool(portfolio.enable_resume))
        self._set_outputs(list(portfolio.requested_outputs))
        self._controls_changed()

    def apply(self) -> None:
        try:
            portfolio = self._build_config()
            portfolio.validate()
            PortfolioPlanner.apply_article_preset(self.state.config, portfolio)
            plan = PortfolioPlanner.plan(self.state.config, portfolio, benchmark_blocks=1)
            if not [key for key in portfolio.requested_outputs if key not in plan.disabled_outputs]:
                raise ValueError(
                    "None of the selected outputs can be generated from the current formulation."
                )
            self.state.config.portfolio = portfolio
            self.state.config.runs = int(plan.required_runs)
            self.state.config.resume_enabled = bool(portfolio.enable_resume)
            self.state.config.reuse_compatible_results = bool(portfolio.reuse_compatible_results)
            self.state.config.checkpoint_interval_evaluations = int(
                portfolio.checkpoint_interval_evaluations
            )
            fingerprint = stable_sha256({"portfolio": portfolio.to_dict(), "plan": asdict(plan)})
            portfolio_id = self.state.database.create_portfolio(
                portfolio.name, portfolio.to_dict(), asdict(plan), fingerprint
            )
            self.state.config.portfolio_id = portfolio_id
            self.state.update_config()
            self.refresh_plan()
            self.stage_completed.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Portfolio planning error", str(exc))
