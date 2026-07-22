"""Application appearance, persistent preferences, and local result-history controls."""

from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton

from calo_rpd_studio.version import VERSION

from calo_rpd_studio.gui.dialogs.experiment_history_dialog import ExperimentHistoryDialog
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class ApplicationSettingsPanel(WorkspacePage):
    def __init__(self, state, settings, parent=None) -> None:
        super().__init__(
            "Application Settings",
            "Choose the interface appearance, manage local experiment history, and review persistent application information.",
            parent,
        )
        self.state = state
        self.settings = settings

        appearance = SectionCard("Appearance")
        form = QFormLayout()
        self.theme = QComboBox()
        self.theme.addItem("Light", "light")
        self.theme.addItem("Dark", "dark")
        index = self.theme.findData(state.theme)
        self.theme.setCurrentIndex(max(index, 0))
        apply_button = QPushButton("Apply appearance")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(self.apply)
        form.addRow("Interface appearance", self.theme)
        form.addRow("", apply_button)
        appearance.layout_root.addLayout(form)
        self.layout_root.addWidget(appearance)

        storage = SectionCard(
            "Experiment history",
            "Review or remove old experiment records and their referenced local convergence/population trace files.",
        )
        storage_form = QFormLayout()
        self.history_summary = QLabel()
        self.history_summary.setWordWrap(True)
        manage = QPushButton("Manage experiment history")
        manage.clicked.connect(self.manage_history)
        storage_form.addRow("Stored data", self.history_summary)
        storage_form.addRow("", manage)
        storage.layout_root.addLayout(storage_form)
        self.layout_root.addWidget(storage)

        information = SectionCard("Application")
        info = QFormLayout()
        info.addRow("Name", QLabel("CALO-RPD Studio"))
        info.addRow("Version", QLabel(VERSION))
        info.addRow("Result database", QLabel(state.database.path))
        information.layout_root.addLayout(info)
        self.layout_root.addWidget(information)
        self.layout_root.addStretch(1)

        state.runs_changed.connect(self.refresh_history_summary)
        self.refresh_history_summary()

    def apply(self) -> None:
        theme = str(self.theme.currentData())
        self.settings.set_value("appearance", theme)
        self.state.set_theme(theme)

    def refresh_history_summary(self) -> None:
        summary = self.state.database.history_storage_summary()
        size_mb = summary["trace_bytes"] / (1024 * 1024)
        self.history_summary.setText(
            f"{summary['experiments']} experiment(s), {summary['runs']} completed run(s), "
            f"{summary['validations']} validation record(s), {summary['trace_files']} trace file(s), "
            f"{size_mb:.2f} MB referenced trace storage"
        )

    def manage_history(self) -> None:
        ExperimentHistoryDialog(self.state, self).exec()
        self.refresh_history_summary()
