"""Central Qt application state."""
from PyQt6.QtCore import QObject, pyqtSignal
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.results.database import ResultDatabase
from .task_status import TaskStatus


class AppState(QObject):
    config_changed = pyqtSignal(object)
    case_changed = pyqtSignal(object)
    runs_changed = pyqtSignal()
    theme_changed = pyqtSignal(str)

    def __init__(self, database_path="calo_rpd_results.sqlite"):
        super().__init__()
        self.config = ExperimentConfig()
        self.current_case = None
        self.current_power_flow = None
        self.current_experiment_id = ""
        self.database = ResultDatabase(database_path)
        self.theme = "light"
        self.task_status = TaskStatus()

    def update_config(self):
        self.config_changed.emit(self.config)

    def set_case(self, case):
        self.current_case = case
        self.current_power_flow = None
        self.case_changed.emit(case)

    def set_theme(self, theme):
        self.theme = theme
        self.theme_changed.emit(theme)
