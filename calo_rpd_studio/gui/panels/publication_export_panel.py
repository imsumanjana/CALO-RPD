"""Verified-only publication and reproducibility export workspace."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
)

from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.results.publication_export import PublicationExporter


class PublicationExportPanel(WorkspacePage):
    def __init__(self, state, parent=None) -> None:
        super().__init__(
            "Publication Export",
            "Export verified numerical results, LaTeX-compatible tables, metadata, and a reproducibility bundle. Unverified runs are excluded.",
            parent,
        )
        self.state = state

        card = SectionCard(
            "Export selection",
            "Only independently verified result records are eligible for publication export.",
        )
        experiment_row = QHBoxLayout()
        self.experiment = QComboBox()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        experiment_row.addWidget(QLabel("Experiment"))
        experiment_row.addWidget(self.experiment, 1)
        experiment_row.addWidget(refresh)
        card.layout_root.addLayout(experiment_row)

        output_row = QHBoxLayout()
        self.directory = QLineEdit("publication_export")
        choose = QPushButton("Choose directory")
        choose.clicked.connect(self.choose)
        output_row.addWidget(self.directory, 1)
        output_row.addWidget(choose)
        card.layout_root.addLayout(output_row)

        export = QPushButton("Export verified publication package")
        export.setObjectName("PrimaryButton")
        export.clicked.connect(self.export)
        card.layout_root.addWidget(export)
        self.layout_root.addWidget(card)

        self.status = QTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(180)
        self.layout_root.addWidget(self.status, 1)

        state.runs_changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        current = self.experiment.currentData()
        self.experiment.clear()
        for experiment in self.state.database.list_experiments():
            self.experiment.addItem(
                f"{experiment['created_at']} — {experiment['name']}",
                experiment["id"],
            )
        index = self.experiment.findData(current)
        self.experiment.setCurrentIndex(max(index, 0))

    def choose(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select publication export directory",
            self.directory.text() or ".",
        )
        if path:
            self.directory.setText(path)

    def export(self) -> None:
        experiment_id = self.experiment.currentData()
        if not experiment_id:
            return
        task = self.state.task_status
        if not task.begin("Exporting publication package", detail="Collecting independently verified records"):
            return
        QApplication.processEvents()
        try:
            path = PublicationExporter(self.state.database).export(
                experiment_id,
                self.directory.text().strip() or "publication_export",
            )
            count = len(
                self.state.database.list_runs(experiment_id, verified_only=True)
            )
            self.status.setPlainText(
                f"Export completed. Verified runs exported: {count}.\nDirectory: {path.resolve()}"
            )
            task.finish(f"Publication package exported with {count} verified run(s)")
        except Exception as exc:
            task.fail(str(exc))
            QMessageBox.critical(self, "Publication export failed", str(exc))
