"""Application appearance and persistent preference controls."""
from __future__ import annotations

from PyQt6.QtWidgets import QComboBox, QFormLayout, QLabel, QPushButton

from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class ApplicationSettingsPanel(WorkspacePage):
    def __init__(self, state, settings, parent=None) -> None:
        super().__init__(
            "Application Settings",
            "Choose the interface appearance and review persistent application information.",
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

        information = SectionCard("Application")
        info = QFormLayout()
        info.addRow("Name", QLabel("CALO-RPD Studio"))
        info.addRow("Version", QLabel("1.0.6"))
        info.addRow("Result database", QLabel(state.database.path))
        information.layout_root.addLayout(info)
        self.layout_root.addWidget(information)
        self.layout_root.addStretch(1)

    def apply(self) -> None:
        theme = str(self.theme.currentData())
        self.settings.set_value("appearance", theme)
        self.state.set_theme(theme)
