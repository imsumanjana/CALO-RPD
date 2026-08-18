"""Application appearance, persistent preferences, and local result-history controls."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QLabel, QLineEdit, QPushButton, QWidget

from calo_rpd_studio.version import PRODUCT_VERSION

from calo_rpd_studio.gui.dialogs.experiment_history_dialog import ExperimentHistoryDialog
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.gui.widgets.workspace_tabs import WorkspaceTabs


class ApplicationSettingsPanel(WorkspacePage):
    density_changed = pyqtSignal(str)

    def __init__(self, state, settings, parent=None) -> None:
        super().__init__(
            "Application Settings",
            "Choose the interface appearance, manage local experiment history, and review persistent application information.",
            parent,
        )
        self.state = state
        self.settings = settings

        appearance = QWidget()
        appearance_layout = QGridLayout(appearance)
        appearance_layout.setContentsMargins(18, 18, 18, 18)
        appearance_layout.setHorizontalSpacing(16)
        appearance_layout.setVerticalSpacing(10)
        self.theme = QComboBox()
        self.theme.addItem("Light", "light")
        self.theme.addItem("Dark", "dark")
        index = self.theme.findData(state.theme)
        self.theme.setCurrentIndex(max(index, 0))
        self.density = QComboBox()
        self.density.addItem("Comfortable", "comfortable")
        self.density.addItem("Compact", "compact")
        density_index = self.density.findData(
            str(self.settings.value("interface_density", "comfortable"))
        )
        self.density.setCurrentIndex(max(density_index, 0))
        apply_button = QPushButton("Apply appearance")
        apply_button.setObjectName("PrimaryButton")
        apply_button.clicked.connect(self.apply)
        theme_label = QLabel("Interface appearance")
        theme_label.setBuddy(self.theme)
        density_label = QLabel("Interface density")
        density_label.setBuddy(self.density)
        self.theme.setAccessibleName("Interface appearance")
        self.density.setAccessibleName("Interface density")
        appearance_layout.addWidget(theme_label, 0, 0)
        appearance_layout.addWidget(density_label, 0, 1)
        appearance_layout.addWidget(self.theme, 1, 0)
        appearance_layout.addWidget(self.density, 1, 1)
        appearance_layout.addWidget(apply_button, 2, 0, 1, 2)
        appearance_layout.setColumnStretch(0, 1)
        appearance_layout.setColumnStretch(1, 1)
        appearance_layout.setRowStretch(3, 1)

        storage = QWidget()
        storage_layout = QGridLayout(storage)
        storage_layout.setContentsMargins(18, 18, 18, 18)
        storage_layout.setHorizontalSpacing(16)
        storage_layout.setVerticalSpacing(10)
        storage_description = QLabel(
            "Review or remove old experiment records and their referenced local "
            "convergence/population trace files."
        )
        storage_description.setWordWrap(True)
        self.history_summary = QLabel()
        self.history_summary.setWordWrap(True)
        manage = QPushButton("Manage experiment history")
        manage.clicked.connect(self.manage_history)
        storage_layout.addWidget(storage_description, 0, 0, 1, 3)
        storage_layout.addWidget(QLabel("Stored data"), 1, 0)
        storage_layout.addWidget(self.history_summary, 1, 1)
        storage_layout.addWidget(manage, 1, 2)
        storage_layout.setColumnStretch(1, 1)
        storage_layout.setRowStretch(2, 1)

        research_assistant = QWidget()
        assistant_layout = QGridLayout(research_assistant)
        assistant_layout.setContentsMargins(18, 18, 18, 18)
        assistant_layout.setHorizontalSpacing(16)
        assistant_layout.setVerticalSpacing(10)
        assistant_note = QLabel(
            "Optional local explanations can use an Ollama model on this computer. The assistant "
            "reads retained parameter evidence only; it cannot change experiments, training, "
            "policies, parameters, or results. It is unavailable while scientific work is active."
        )
        assistant_note.setWordWrap(True)
        self.local_assistant_enabled = QCheckBox("Enable local parameter explanations")
        self.local_assistant_enabled.setChecked(
            str(self.settings.value("local_parameter_assistant_enabled", "false")).lower()
            in {"1", "true", "yes"}
        )
        self.local_assistant_endpoint = QLineEdit(
            str(self.settings.value("local_parameter_assistant_endpoint", "http://127.0.0.1:11434"))
        )
        self.local_assistant_model = QLineEdit(
            str(self.settings.value("local_parameter_assistant_model", "qwen3.5:9b"))
        )
        self.local_assistant_endpoint.setAccessibleName("Local Ollama address")
        self.local_assistant_model.setAccessibleName("Local explanation model")
        save_assistant = QPushButton("Save local assistant settings")
        save_assistant.clicked.connect(self.save_local_assistant_settings)
        self.local_assistant_status = QLabel("")
        self.local_assistant_status.setWordWrap(True)
        assistant_layout.addWidget(assistant_note, 0, 0, 1, 2)
        assistant_layout.addWidget(self.local_assistant_enabled, 1, 0, 1, 2)
        assistant_layout.addWidget(QLabel("Ollama address"), 2, 0)
        assistant_layout.addWidget(self.local_assistant_endpoint, 2, 1)
        assistant_layout.addWidget(QLabel("Model"), 3, 0)
        assistant_layout.addWidget(self.local_assistant_model, 3, 1)
        assistant_layout.addWidget(save_assistant, 4, 0, 1, 2)
        assistant_layout.addWidget(self.local_assistant_status, 5, 0, 1, 2)
        assistant_layout.setColumnStretch(1, 1)
        assistant_layout.setRowStretch(6, 1)

        information = QWidget()
        information_layout = QGridLayout(information)
        information_layout.setContentsMargins(18, 18, 18, 18)
        information_layout.setHorizontalSpacing(16)
        information_layout.setVerticalSpacing(10)
        information_layout.addWidget(QLabel("Name"), 0, 0)
        information_layout.addWidget(QLabel("CALO-RPD Studio"), 0, 1)
        information_layout.addWidget(QLabel("Version"), 0, 2)
        information_layout.addWidget(QLabel(PRODUCT_VERSION), 0, 3)
        information_layout.setColumnStretch(1, 1)
        information_layout.setColumnStretch(3, 1)

        database_label = QLabel("Result database")
        database_label.setObjectName("FieldLabel")
        self.database_path = QLineEdit(str(state.database.path))
        self.database_path.setObjectName("ResultDatabasePath")
        self.database_path.setReadOnly(True)
        self.database_path.setProperty("fullWidthInput", True)
        self.database_path.setAccessibleName("Result database path")
        self.database_path.setToolTip(str(state.database.path))
        self.database_path.setCursorPosition(0)
        database_label.setBuddy(self.database_path)
        information_layout.addWidget(database_label, 1, 0, 1, 4)
        information_layout.addWidget(self.database_path, 2, 0, 1, 4)
        information_layout.setRowStretch(3, 1)

        self.section_tabs = WorkspaceTabs("Application settings sections")
        self.section_tabs.add_section(
            "Appearance",
            appearance,
            "Choose the application theme and information density.",
        )
        self.section_tabs.add_section(
            "Experiment history",
            storage,
            "Review locally stored experiments, runs, validations, and trace storage.",
        )
        self.section_tabs.add_section(
            "Research assistant",
            research_assistant,
            "Configure optional local explanations of retained parameter evidence.",
        )
        self.section_tabs.add_section(
            "Application",
            information,
            "Review the active version and result database location.",
        )
        self.layout_root.addWidget(self.section_tabs, 1)

        state.runs_changed.connect(self.refresh_history_summary)
        self.refresh_history_summary()

    def apply(self) -> None:
        theme = str(self.theme.currentData())
        density = str(self.density.currentData())
        self.settings.set_value("appearance", theme)
        self.settings.set_value("interface_density", density)
        self.state.set_theme(theme)
        self.density_changed.emit(density)

    def save_local_assistant_settings(self) -> None:
        from calo_rpd_studio.assistant import LocalAssistantConfig

        config = LocalAssistantConfig(
            enabled=bool(self.local_assistant_enabled.isChecked()),
            endpoint=self.local_assistant_endpoint.text().strip(),
            model=self.local_assistant_model.text().strip(),
        )
        try:
            config.validate()
        except ValueError as exc:
            self.local_assistant_status.setText(f"Settings were not saved: {exc}")
            return
        self.settings.set_value("local_parameter_assistant_enabled", config.enabled)
        self.settings.set_value("local_parameter_assistant_endpoint", config.endpoint)
        self.settings.set_value("local_parameter_assistant_model", config.model)
        self.settings.sync()
        self.local_assistant_status.setText(
            "Local explanation settings saved. No scientific configuration was changed."
        )

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
