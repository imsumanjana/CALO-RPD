"""Twenty-algorithm selection and parameter configuration."""
from __future__ import annotations

import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from calo_rpd_studio.algorithms.registry import SPECS
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class AlgorithmsPanel(WorkspacePage):
    stage_completed = pyqtSignal()

    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Algorithms",
            "Select the primary comparison methods and edit declared parameters. CALO alone uses the AI controller.",
            parent,
        )
        self.state = state

        card = SectionCard(
            "Primary optimizer registry",
            "Every selected baseline uses the same ORPD evaluator, variable decoder, constraints, and experiment protocol.",
        )
        self.table = QTableWidget(len(SPECS), 4)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Algorithm", "Scientific description", "Parameters (JSON)"]
        )
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        for row, (name, spec) in enumerate(SPECS.items()):
            use = QTableWidgetItem()
            use.setFlags(use.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(
                Qt.CheckState.Checked
                if name in state.config.algorithms
                else Qt.CheckState.Unchecked
            )
            self.table.setItem(row, 0, use)

            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 1, name_item)

            description = QTableWidgetItem(spec.description)
            description.setFlags(description.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(row, 2, description)

            parameters = {
                **spec.default_parameters,
                **state.config.algorithm_parameters.get(name, {}),
            }
            self.table.setItem(row, 3, QTableWidgetItem(json.dumps(parameters)))

        card.layout_root.addWidget(self.table, 1)
        buttons = QHBoxLayout()
        apply_button = QPushButton("Apply algorithm selection")
        apply_button.setObjectName("PrimaryButton")
        defaults = QPushButton("Restore canonical defaults")
        apply_button.clicked.connect(self.apply)
        defaults.clicked.connect(self.restore_defaults)
        buttons.addWidget(apply_button)
        buttons.addWidget(defaults)
        buttons.addStretch(1)
        card.layout_root.addLayout(buttons)
        self.layout_root.addWidget(card, 1)

    def load_from_config(self, config) -> None:
        for row, (name, spec) in enumerate(SPECS.items()):
            self.table.item(row, 0).setCheckState(
                Qt.CheckState.Checked if name in config.algorithms else Qt.CheckState.Unchecked
            )
            parameters = {**spec.default_parameters, **config.algorithm_parameters.get(name, {})}
            self.table.item(row, 3).setText(json.dumps(parameters))

    def apply(self) -> None:
        selected: list[str] = []
        parameters: dict[str, dict] = {}
        try:
            for row in range(self.table.rowCount()):
                name = self.table.item(row, 1).text()
                parsed = json.loads(self.table.item(row, 3).text() or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError(f"{name} parameters must be a JSON object")
                parameters[name] = parsed
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    selected.append(name)
            if not selected:
                raise ValueError("Select at least one primary optimizer.")
        except Exception as exc:
            QMessageBox.critical(self, "Algorithm configuration error", str(exc))
            return

        self.state.config.algorithms = selected
        self.state.config.algorithm_parameters = parameters
        self.state.update_config()
        self.stage_completed.emit()

    def restore_defaults(self) -> None:
        for row, (name, spec) in enumerate(SPECS.items()):
            self.table.item(row, 3).setText(json.dumps(spec.default_parameters))
