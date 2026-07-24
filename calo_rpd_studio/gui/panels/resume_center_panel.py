"""Universal resume centre for interrupted experiments, training, validation, and exports."""

from __future__ import annotations

import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)

from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage


class ResumeCenterPanel(WorkspacePage):
    workspace_requested = pyqtSignal(int)
    experiment_restore_requested = pyqtSignal(str)
    policy_training_resumed = pyqtSignal(str)
    validation_resumed = pyqtSignal(str)
    portfolio_export_resumed = pyqtSignal(str)

    def __init__(self, state, experiment_manager, parent=None) -> None:
        super().__init__(
            "Resume Center",
            "Resume interrupted scientific work without repeating valid completed jobs. Campaign, policy-training, validation, and portfolio-export tasks are tracked independently.",
            parent,
        )
        self.state = state
        self.manager = experiment_manager
        self._rows: list[dict] = []

        card = SectionCard(
            "Unfinished work",
            "Running tasks are marked interrupted after an unclean shutdown. Safe pause stops new admissions and retains all completed runs and checkpoints.",
        )
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Type", "Task", "Progress", "Status", "Last activity", "Task ID"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.ResizeMode.ResizeToContents
        )
        self.table.setColumnHidden(5, True)
        card.layout_root.addWidget(self.table, 1)

        row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        resume_selected = QPushButton("Resume selected")
        resume_selected.setObjectName("PrimaryButton")
        resume_all = QPushButton("Resume all compatible")
        inspect = QPushButton("Inspect")
        archive = QPushButton("Archive")
        delete = QPushButton("Delete unfinished record")
        refresh.clicked.connect(self.refresh)
        resume_selected.clicked.connect(self.resume_selected)
        resume_all.clicked.connect(self.resume_all)
        inspect.clicked.connect(self.inspect_selected)
        archive.clicked.connect(self.archive_selected)
        delete.clicked.connect(self.delete_selected)
        for button in (refresh, resume_selected, resume_all, inspect, archive, delete):
            row.addWidget(button)
        row.addStretch(1)
        card.layout_root.addLayout(row)
        self.layout_root.addWidget(card, 1)

        self.manager.completed.connect(lambda _: self.refresh())
        self.manager.cancelled.connect(lambda _: self.refresh())
        self.manager.failed.connect(lambda _: self.refresh())
        self.refresh()

    def refresh(self) -> None:
        self._rows = [
            {
                "id": item.id,
                "task_type": item.task_type,
                "title": item.title,
                "progress": item.progress_text,
                "status": item.status,
                "updated_at": item.updated_at,
                "state": item.state,
                "resumable": item.resumable,
            }
            for item in self.state.resume_service.unfinished()
        ]
        self.table.setRowCount(len(self._rows))
        for row, item in enumerate(self._rows):
            values = [
                item["task_type"],
                item["title"],
                item["progress"],
                item["status"],
                item["updated_at"],
                item["id"],
            ]
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, col, cell)
        if self._rows:
            self.table.selectRow(0)

    def _selected(self) -> dict | None:
        row = self.table.currentRow()
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def _resume(self, item: dict) -> bool:
        if self.manager.running:
            QMessageBox.information(
                self, "Task busy", "Pause or finish the active scientific task first."
            )
            return False
        task_type = item["task_type"]
        if task_type == "experiment":
            campaign_id = str(item["state"].get("campaign_id", ""))
            if not campaign_id:
                QMessageBox.critical(
                    self, "Resume failed", "The experiment resume record has no campaign ID."
                )
                return False
            campaign = self.state.database.get_campaign(campaign_id)
            experiment_id = str(
                (campaign or {}).get("experiment_id", "") or item["state"].get("experiment_id", "")
            )
            if experiment_id:
                self.experiment_restore_requested.emit(experiment_id)
            return bool(self.manager.resume_campaign(campaign_id))
        if task_type == "policy_training":
            self.workspace_requested.emit(1)
            self.policy_training_resumed.emit(item["id"])
            return True
        if task_type == "validation":
            self.workspace_requested.emit(11)
            self.validation_resumed.emit(item["id"])
            return True
        if task_type == "portfolio_export":
            self.workspace_requested.emit(12)
            self.portfolio_export_resumed.emit(item["id"])
            return True
        return False

    def resume_selected(self) -> None:
        item = self._selected()
        if item and self._resume(item):
            self.refresh()

    def resume_all(self) -> None:
        resumed: list[str] = []
        deferred: list[str] = []
        unsupported: list[str] = []
        # Global scientific exclusivity permits only one executing/resumed task at a time. Select
        # the first compatible record regardless of task type and explicitly summarize the rest.
        for item in list(self._rows):
            task_type = str(item.get("task_type", ""))
            if task_type not in {"experiment", "policy_training", "validation", "portfolio_export"}:
                unsupported.append(f"{task_type}: {item.get('title', item.get('id', ''))}")
                continue
            if resumed:
                deferred.append(f"{task_type}: {item.get('title', item.get('id', ''))}")
                continue
            if self._resume(item):
                resumed.append(f"{task_type}: {item.get('title', item.get('id', ''))}")
            else:
                deferred.append(f"{task_type}: {item.get('title', item.get('id', ''))}")
        self.refresh()
        if deferred or unsupported:
            lines = []
            if resumed:
                lines.append("Resumed: " + resumed[0])
            if deferred:
                lines.append("Deferred/manual action required: " + "; ".join(deferred))
            if unsupported:
                lines.append("Unsupported resume record types: " + "; ".join(unsupported))
            QMessageBox.information(self, "Resume all compatible", "\n".join(lines))

    def inspect_selected(self) -> None:
        item = self._selected()
        if item:
            QMessageBox.information(self, "Resume record", json.dumps(item, indent=2, default=str))

    def archive_selected(self) -> None:
        item = self._selected()
        if item:
            self.state.resume_service.archive(item["id"])
            self.refresh()

    def delete_selected(self) -> None:
        item = self._selected()
        if not item:
            return
        answer = QMessageBox.question(
            self,
            "Delete unfinished record",
            "Delete this resume record and its checkpoint directory? Completed experiment results are not deleted.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.state.resume_service.delete(item["id"])
            self.refresh()
