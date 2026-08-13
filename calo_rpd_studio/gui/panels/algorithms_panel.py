"""Registered-algorithm selection and parameter configuration."""

from __future__ import annotations

import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from calo_rpd_studio.algorithms.registry import POLICY_GATED_SPECS, SPECS
from calo_rpd_studio.gui.user_feedback import show_error
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class AlgorithmsPanel(WorkspacePage):
    stage_completed = pyqtSignal()

    @staticmethod
    def _safe_defaults(name: str, spec) -> dict:
        defaults = dict(spec.default_parameters)
        if name == "CALO":
            defaults.update(
                use_ai=False,
                strict_policy_binding=False,
                allow_unqualified_policy=False,
            )
            for field in ("policy_id", "policy_checkpoint", "policy_sha256"):
                defaults.pop(field, None)
        return defaults

    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Algorithms",
            "Select primary comparison methods or the separately gated TSH-CALO policy workflow.",
            parent,
        )
        self.state = state
        self.specs = {**SPECS, **POLICY_GATED_SPECS}

        card = SectionCard(
            "Primary optimizer registry",
            "Every selected baseline uses the same ORPD evaluator, variable decoder, constraints, and experiment protocol.",
        )
        self.table = QTableWidget(len(self.specs), 4)
        self.table.setHorizontalHeaderLabels(
            ["Use", "Algorithm", "Scientific description", "Parameters (JSON)"]
        )
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        for row, (name, spec) in enumerate(self.specs.items()):
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
                **self._safe_defaults(name, spec),
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
        self.state.config_changed.connect(self.load_from_config)
        self.state.policy_state_changed.connect(lambda _status: self._apply_policy_gate())
        self._apply_policy_gate()

    def _apply_policy_gate(self) -> None:
        status = self.state.governing_policy_status()
        ready = bool(status.ready and status.algorithm_id == "TSH-CALO")
        reason = (
            "TSH-CALO requires a verified compatible policy that has been selected for experiments."
        )
        for row, name in enumerate(self.specs):
            if name not in POLICY_GATED_SPECS:
                continue
            use = self.table.item(row, 0)
            parameters = self.table.item(row, 3)
            if ready:
                use.setFlags(use.flags() | Qt.ItemFlag.ItemIsEnabled)
                parameters.setFlags(parameters.flags() | Qt.ItemFlag.ItemIsEditable)
                use.setToolTip("")
                parameters.setToolTip("")
            else:
                use.setCheckState(Qt.CheckState.Unchecked)
                use.setFlags(use.flags() & ~Qt.ItemFlag.ItemIsEnabled)
                parameters.setFlags(parameters.flags() & ~Qt.ItemFlag.ItemIsEditable)
                use.setToolTip(reason)
                parameters.setToolTip(reason)

    def load_from_config(self, config) -> None:
        for row, (name, spec) in enumerate(self.specs.items()):
            self.table.item(row, 0).setCheckState(
                Qt.CheckState.Checked if name in config.algorithms else Qt.CheckState.Unchecked
            )
            parameters = {
                **self._safe_defaults(name, spec),
                **config.algorithm_parameters.get(name, {}),
            }
            self.table.item(row, 3).setText(json.dumps(parameters))
        self._apply_policy_gate()

    def apply(self) -> None:
        selected: list[str] = []
        parameters: dict[str, dict] = {}
        try:
            for row in range(self.table.rowCount()):
                name = self.table.item(row, 1).text()
                parsed = json.loads(self.table.item(row, 3).text() or "{}")
                if not isinstance(parsed, dict):
                    raise ValueError(f"{name} parameters must be a JSON object")
                if name == "CALO" and (
                    parsed.get("use_ai") is True
                    or any(
                        str(parsed.get(field, "")).strip()
                        for field in ("policy_id", "policy_checkpoint", "policy_sha256")
                    )
                ):
                    raise ValueError(
                        "Primary CALO is rule-only in v12; use the separately gated TSH-CALO row "
                        "for a future new qualified policy"
                    )
                parameters[name] = parsed
                if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                    selected.append(name)
            if not selected:
                raise ValueError("Select at least one primary optimizer.")
        except Exception as exc:
            show_error(
                self,
                "Algorithm settings were not applied",
                "Review the selected algorithms and parameter values.",
                exc,
                source="algorithm settings",
            )
            return

        self.state.config.algorithms = selected
        self.state.config.algorithm_parameters = parameters
        self.state.update_config()
        self.stage_completed.emit()

    def restore_defaults(self) -> None:
        for row, (name, spec) in enumerate(self.specs.items()):
            self.table.item(row, 3).setText(json.dumps(self._safe_defaults(name, spec)))
