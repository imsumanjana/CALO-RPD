"""Historical experiment classification and experience-repository controls."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.learning.experience_repository import (
    EXPERIMENT_ROLES,
    build_experience_repository,
    load_experience_repository,
)


class HistoricalExperienceWidget(QGroupBox):
    """Leakage-aware historical learning controls embedded in CALO Intelligence."""

    repository_changed = pyqtSignal(str)

    def __init__(self, state, experiment_manager=None, parent=None) -> None:
        super().__init__("Historical experience learning", parent)
        self.state = state
        self.experiment_manager = experiment_manager
        self._row_experiment_ids: dict[int, str] = {}

        root = QVBoxLayout(self)
        description = QLabel(
            "Classify old experiments before reuse. Only experiments explicitly marked TRAIN and "
            "learning-eligible can enter the experience repository. VALIDATION and TEST experiments "
            "are excluded from policy and algorithm learning to prevent benchmark leakage."
        )
        description.setWordWrap(True)
        root.addWidget(description)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Role",
                "Eligible",
                "Lock",
                "Experiment",
                "Created",
                "Runs",
                "Verified",
                "CALO transitions",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for column in (0, 1, 2, 4, 5, 6, 7):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setMinimumHeight(250)
        root.addWidget(self.table)

        table_actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh experiments")
        self.save_roles_button = QPushButton("Save classifications")
        self.refresh_button.clicked.connect(self.refresh)
        self.save_roles_button.clicked.connect(self.save_classifications)
        table_actions.addWidget(self.refresh_button)
        table_actions.addWidget(self.save_roles_button)
        table_actions.addStretch(1)
        root.addLayout(table_actions)

        repository_group = QGroupBox("Experience repository")
        repository_form = QFormLayout(repository_group)
        self.learning_mode = QComboBox()
        self.learning_mode.addItem("Cold Start — no historical learning", "cold_start")
        self.learning_mode.addItem("Historical Warm Start", "historical_warm_start")
        self.learning_mode.addItem(
            "Continual Learning — rebuild eligible repository automatically", "continual_learning"
        )
        self.learning_mode.setCurrentIndex(1)
        self.repository_path = QLineEdit(str(Path("historical_experience_v1.3.json").resolve()))
        choose = QPushButton("Choose…")
        choose.clicked.connect(self.choose_repository)
        repository_row = QWidget()
        repository_row_layout = QHBoxLayout(repository_row)
        repository_row_layout.setContentsMargins(0, 0, 0, 0)
        repository_row_layout.addWidget(self.repository_path, 1)
        repository_row_layout.addWidget(choose)
        self.verified_only = QCheckBox("Require independently verified runs")
        self.verified_only.setChecked(True)
        self.use_historical_trajectories = QCheckBox(
            "Use eligible CALO trajectories for offline policy pretraining"
        )
        self.use_historical_trajectories.setChecked(True)
        self.use_cross_algorithm_knowledge = QCheckBox(
            "Use eligible cross-algorithm solutions as a knowledge archive"
        )
        self.use_cross_algorithm_knowledge.setChecked(True)
        self.use_parameter_priors = QCheckBox("Use historical CALO parameter priors")
        self.use_parameter_priors.setChecked(True)
        self.allow_population_warm_start = QCheckBox(
            "Allow historical population warm-starting (practical mode; disable for cold-start benchmarks)"
        )
        self.allow_population_warm_start.setChecked(False)
        self.warm_start_percent = QSpinBox()
        self.warm_start_percent.setRange(0, 50)
        self.warm_start_percent.setValue(15)
        self.warm_start_percent.setSuffix(" %")
        self.pretraining_epochs = QSpinBox()
        self.pretraining_epochs.setRange(0, 1000)
        self.pretraining_epochs.setValue(4)

        repository_form.addRow("Learning mode", self.learning_mode)
        repository_form.addRow("Repository file", repository_row)
        repository_form.addRow("Validation filter", self.verified_only)
        repository_form.addRow("Policy learning", self.use_historical_trajectories)
        repository_form.addRow("Cross-algorithm knowledge", self.use_cross_algorithm_knowledge)
        repository_form.addRow("Algorithm parameter priors", self.use_parameter_priors)
        repository_form.addRow("Population warm start", self.allow_population_warm_start)
        repository_form.addRow("Warm-start population fraction", self.warm_start_percent)
        repository_form.addRow("Historical pretraining epochs", self.pretraining_epochs)
        root.addWidget(repository_group)

        actions = QHBoxLayout()
        self.build_button = QPushButton("Build Experience Repository")
        self.preview_button = QPushButton("Preview Eligible Data")
        self.apply_algorithm_button = QPushButton("Apply Historical Learning to CALO")
        self.apply_algorithm_button.setObjectName("PrimaryButton")
        self.build_button.clicked.connect(self.build_repository)
        self.preview_button.clicked.connect(self.preview_repository)
        self.apply_algorithm_button.clicked.connect(self.apply_algorithm_learning)
        actions.addWidget(self.build_button)
        actions.addWidget(self.preview_button)
        actions.addWidget(self.apply_algorithm_button)
        actions.addStretch(1)
        root.addLayout(actions)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(120)
        self.summary.setMaximumHeight(210)
        root.addWidget(self.summary)
        if self.experiment_manager is not None:
            self.experiment_manager.completed.connect(self._auto_rebuild_if_continual)
        self.learning_mode.currentIndexChanged.connect(self._sync_learning_mode)
        self._sync_learning_mode()
        self.refresh()

    def _sync_learning_mode(self, *_args) -> None:
        enabled = self.learning_mode.currentData() != "cold_start"
        for widget in (
            self.repository_path,
            self.verified_only,
            self.use_historical_trajectories,
            self.use_cross_algorithm_knowledge,
            self.use_parameter_priors,
            self.allow_population_warm_start,
            self.warm_start_percent,
            self.pretraining_epochs,
            self.build_button,
            self.preview_button,
            self.apply_algorithm_button,
        ):
            widget.setEnabled(enabled)

    def _auto_rebuild_if_continual(self, *_args) -> None:
        if self.learning_mode.currentData() != "continual_learning":
            return
        try:
            repository = build_experience_repository(
                self.state.database,
                self.repository_path.text(),
                verified_only=self.verified_only.isChecked(),
            )
        except Exception as exc:
            self.summary.setPlainText(f"Continual-learning repository rebuild failed: {exc}")
            return
        self.repository_path.setText(repository.path)
        self.summary.setPlainText(
            "Continual-learning repository refreshed from currently eligible TRAIN experiments.\n\n"
            + json.dumps(repository.summary, indent=2)
        )
        self.repository_changed.emit(repository.path)

    @staticmethod
    def _checkbox_host(check: QCheckBox) -> QWidget:
        host = QWidget()
        layout = QHBoxLayout(host)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(check)
        return host

    def _transition_count(self, experiment_id: str) -> int:
        total = 0
        for run in self.state.database.list_runs(experiment_id=experiment_id):
            if run.get("algorithm") != "CALO":
                continue
            try:
                result = json.loads(run.get("result_json") or "{}")
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            total += len((result.get("metadata") or {}).get("policy_trajectory") or [])
        return total

    def refresh(self) -> None:
        experiments = self.state.database.list_experiments()
        self.table.setRowCount(len(experiments))
        self._row_experiment_ids = {}
        for row_index, experiment in enumerate(experiments):
            experiment_id = experiment["id"]
            self._row_experiment_ids[row_index] = experiment_id
            role = str(experiment.get("data_role", "excluded"))
            locked = bool(experiment.get("learning_locked", 0))
            eligible = bool(experiment.get("learning_eligible", 0))

            role_combo = QComboBox()
            for item in EXPERIMENT_ROLES:
                role_combo.addItem(item.upper(), item)
            role_combo.setCurrentIndex(max(0, role_combo.findData(role)))
            eligible_check = QCheckBox()
            eligible_check.setChecked(eligible and role == "train")
            lock_check = QCheckBox()
            lock_check.setChecked(locked)
            role_combo.setEnabled(not locked)
            eligible_check.setEnabled(not locked and role == "train")

            def update_eligibility(_index, combo=role_combo, check=eligible_check):
                is_train = combo.currentData() == "train"
                check.setEnabled(is_train and combo.isEnabled())
                if not is_train:
                    check.setChecked(False)

            role_combo.currentIndexChanged.connect(update_eligibility)
            self.table.setCellWidget(row_index, 0, role_combo)
            self.table.setCellWidget(row_index, 1, self._checkbox_host(eligible_check))
            self.table.setCellWidget(row_index, 2, self._checkbox_host(lock_check))
            self.table.setItem(row_index, 3, QTableWidgetItem(str(experiment.get("name", ""))))
            self.table.setItem(
                row_index, 4, QTableWidgetItem(str(experiment.get("created_at", ""))[:19])
            )
            storage = self.state.database.experiment_storage_summary(experiment_id)
            self.table.setItem(row_index, 5, QTableWidgetItem(str(storage.get("runs", 0))))
            self.table.setItem(row_index, 6, QTableWidgetItem(str(storage.get("verified_runs", 0))))
            self.table.setItem(
                row_index, 7, QTableWidgetItem(str(self._transition_count(experiment_id)))
            )
        self.summary.setPlainText(
            "Existing experiments are excluded from learning by default. Mark development data as TRAIN, "
            "enable learning eligibility, save classifications, then build the repository."
        )

    def _cell_checkbox(self, row: int, column: int) -> QCheckBox:
        host = self.table.cellWidget(row, column)
        checks = host.findChildren(QCheckBox) if host is not None else []
        if not checks:
            raise RuntimeError("Historical-learning checkbox widget is missing")
        return checks[0]

    def save_classifications(self) -> None:
        try:
            for row in range(self.table.rowCount()):
                experiment_id = self._row_experiment_ids[row]
                role_combo = self.table.cellWidget(row, 0)
                role = str(role_combo.currentData())
                eligible = self._cell_checkbox(row, 1).isChecked()
                lock_requested = self._cell_checkbox(row, 2).isChecked()
                current = self.state.database.get_experiment(experiment_id) or {}
                currently_locked = bool(current.get("learning_locked", 0))
                if currently_locked and not lock_requested:
                    self.state.database.set_experiment_learning_role(
                        experiment_id,
                        str(current.get("data_role", "excluded")),
                        eligible=bool(current.get("learning_eligible", 0)),
                        locked=False,
                    )
                self.state.database.set_experiment_learning_role(
                    experiment_id,
                    role,
                    eligible=eligible,
                    locked=lock_requested,
                )
        except Exception as exc:
            QMessageBox.critical(self, "Historical experiment classification", str(exc))
            self.refresh()
            return
        self.refresh()
        self._auto_rebuild_if_continual()
        QMessageBox.information(
            self,
            "Historical experiment classification",
            "Experiment roles and learning eligibility were saved.",
        )

    def choose_repository(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Historical experience repository",
            self.repository_path.text(),
            "JSON (*.json)",
        )
        if path:
            self.repository_path.setText(path)

    def build_repository(self) -> None:
        try:
            repository = build_experience_repository(
                self.state.database,
                self.repository_path.text(),
                verified_only=self.verified_only.isChecked(),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Build Experience Repository", str(exc))
            return
        self.repository_path.setText(repository.path)
        self.summary.setPlainText(json.dumps(repository.summary, indent=2))
        self.repository_changed.emit(repository.path)

    def preview_repository(self) -> None:
        path = Path(self.repository_path.text().strip())
        if not path.exists():
            self.build_repository()
            path = Path(self.repository_path.text().strip())
        if not path.exists():
            return
        try:
            repository = load_experience_repository(path)
        except Exception as exc:
            QMessageBox.critical(self, "Historical Experience", str(exc))
            return
        preview = {
            "repository": repository.path,
            "summary": repository.summary,
            "selection_policy": repository.payload.get("selection_policy", {}),
            "note": (
                "Older CALO runs without v1.3 policy_trajectory records remain useful for cross-algorithm "
                "solution knowledge and parameter priors, but cannot be used for direct policy imitation."
            ),
        }
        self.summary.setPlainText(json.dumps(preview, indent=2))

    def apply_algorithm_learning(self) -> None:
        mode = str(self.learning_mode.currentData())
        if mode == "cold_start":
            parameters = dict(self.state.config.algorithm_parameters.get("CALO", {}))
            for key in (
                "historical_repository",
                "use_historical_parameter_priors",
                "use_cross_algorithm_warm_start",
                "historical_warm_start_fraction",
                "strict_benchmark_mode",
            ):
                parameters.pop(key, None)
            self.state.config.algorithm_parameters["CALO"] = parameters
            self.state.update_config()
            QMessageBox.information(
                self, "Historical Learning", "CALO is configured for a strict cold start."
            )
            return
        path = Path(self.repository_path.text().strip())
        if not path.exists():
            QMessageBox.warning(
                self,
                "Historical Learning",
                "Build or choose a valid historical experience repository first.",
            )
            return
        try:
            load_experience_repository(path)
        except Exception as exc:
            QMessageBox.critical(self, "Historical Learning", str(exc))
            return
        parameters = dict(self.state.config.algorithm_parameters.get("CALO", {}))
        parameters.update(
            {
                "historical_learning_mode": mode,
                "historical_repository": str(path.resolve()),
                "use_historical_parameter_priors": self.use_parameter_priors.isChecked(),
                "use_cross_algorithm_warm_start": (
                    self.use_cross_algorithm_knowledge.isChecked()
                    and self.allow_population_warm_start.isChecked()
                ),
                "historical_warm_start_fraction": self.warm_start_percent.value() / 100.0,
                # Historical transfer is explicit and scientifically separate from the strict
                # independent-run benchmark protocol.
                "strict_benchmark_mode": False,
            }
        )
        self.state.config.algorithm_parameters["CALO"] = parameters
        self.state.update_config()
        self.repository_changed.emit(str(path.resolve()))
        QMessageBox.information(
            self,
            "Historical Learning",
            "Historical CALO transfer-learning settings were applied with strict benchmark mode disabled. "
            "Use Cold Start or the locked TEST campaign for strict independent-run comparisons.",
        )

    def policy_training_options(self) -> dict:
        enabled = self.learning_mode.currentData() != "cold_start"
        return {
            "historical_repository": self.repository_path.text().strip() if enabled else "",
            "use_historical_trajectories": enabled and self.use_historical_trajectories.isChecked(),
            "historical_pretraining_epochs": self.pretraining_epochs.value() if enabled else 0,
            "learning_mode": str(self.learning_mode.currentData()),
        }
