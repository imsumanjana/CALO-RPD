"""Independent solution validation and provenance audit."""
from __future__ import annotations

import json

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.results.integrity_checker import check_run_record
from calo_rpd_studio.results.solution_validator import validate_stored_run


class ValidationAuditPanel(WorkspacePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Validation & Audit",
            "Reload saved decisions, independently rerun the physical model, recompute objectives and constraints, and record result integrity.",
            parent,
        )
        self.state = state

        controls = QHBoxLayout()
        self.experiment = QComboBox()
        self.run = QComboBox()
        refresh = QPushButton("Refresh")
        validate = QPushButton("Validate selected run")
        validate.setObjectName("PrimaryButton")
        refresh.clicked.connect(self.refresh_experiments)
        validate.clicked.connect(self.validate)
        self.experiment.currentIndexChanged.connect(self.refresh_runs)
        controls.addWidget(QLabel("Experiment"))
        controls.addWidget(self.experiment, 1)
        controls.addWidget(QLabel("Run"))
        controls.addWidget(self.run, 1)
        controls.addWidget(refresh)
        controls.addWidget(validate)
        self.layout_root.addLayout(controls)

        box = QGroupBox("Audit record")
        layout = QVBoxLayout(box)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)
        self.layout_root.addWidget(box, 1)

        state.runs_changed.connect(self.refresh_experiments)
        self.refresh_experiments()

    def refresh_experiments(self) -> None:
        current = self.experiment.currentData()
        self.experiment.blockSignals(True)
        self.experiment.clear()
        for experiment in self.state.database.list_experiments():
            self.experiment.addItem(
                f"{experiment['created_at']} — {experiment['name']}",
                experiment["id"],
            )
        index = self.experiment.findData(current)
        self.experiment.setCurrentIndex(max(index, 0))
        self.experiment.blockSignals(False)
        self.refresh_runs()

    def refresh_runs(self) -> None:
        self.run.clear()
        experiment_id = self.experiment.currentData()
        if not experiment_id:
            return
        for row in self.state.database.list_runs(experiment_id):
            self.run.addItem(
                f"Run {row['run_index'] + 1} — {row['algorithm']} — {row['validation_status']}",
                row["id"],
            )


    def select_run(self, experiment_id: str, run_id: str) -> None:
        """Select a reviewed run so validation opens on the intended record."""
        self.refresh_experiments()
        experiment_index = self.experiment.findData(experiment_id)
        if experiment_index >= 0:
            self.experiment.setCurrentIndex(experiment_index)
        self.refresh_runs()
        run_index = self.run.findData(run_id)
        if run_index >= 0:
            self.run.setCurrentIndex(run_index)

    def validate(self) -> None:
        run_id = self.run.currentData()
        if not run_id:
            return
        task = self.state.task_status
        if not task.begin("Validating stored solution", detail="Independent physical-model re-evaluation"):
            return
        QApplication.processEvents()
        try:
            record = self.state.database.get_run(run_id)
            basic = check_run_record(record)
            result = validate_stored_run(self.state.database, run_id)
            self.output.setPlainText(
                json.dumps(
                    {
                        "record_integrity": basic,
                        "independent_validation": result,
                    },
                    indent=2,
                    allow_nan=True,
                )
            )
            self.state.runs_changed.emit()
            if result.get("passed"):
                task.finish("Independent result validation passed")
            else:
                task.fail("Independent result validation failed")
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Validation failed", str(exc))
