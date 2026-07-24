"""Central Qt application state."""

from PyQt6.QtCore import QObject, pyqtSignal
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.resume.service import ResumeService
from calo_rpd_studio.algorithms.calo.policy_registry import PolicyRegistry
from calo_rpd_studio.algorithms.calo.policy_readiness import evaluate_governing_policy
from calo_rpd_studio.compute.topology import ComputeTopologyService, SafeResourceBudgetEngine
from calo_rpd_studio.compute.governor import AdaptiveComputeGovernor, GovernorConfig
from .task_status import TaskStatus


class AppState(QObject):
    config_changed = pyqtSignal(object)
    case_changed = pyqtSignal(object)
    runs_changed = pyqtSignal()
    theme_changed = pyqtSignal(str)
    compute_profile_changed = pyqtSignal(object)
    compute_governor_changed = pyqtSignal(object)
    policy_state_changed = pyqtSignal(object)
    policy_training_changed = pyqtSignal(bool, str)
    policy_training_plan_changed = pyqtSignal(object)

    def __init__(self, database_path="calo_rpd_results.sqlite"):
        super().__init__()
        self.config = ExperimentConfig()
        self.current_case = None
        self.current_power_flow = None
        self.current_experiment_id = ""
        self.database = ResultDatabase(database_path)
        self.resume_service = ResumeService(self.database)
        self.policy_registry = PolicyRegistry(self.database)
        self.resume_service.recover_after_restart()
        self.theme = "light"
        self.task_status = TaskStatus()
        self.compute_topology_service = ComputeTopologyService()
        self.compute_budget_engine = SafeResourceBudgetEngine(allocation_limit_fraction=0.80)
        self.compute_topology = None
        self.compute_protection_profile = None
        self.compute_governor = None
        self.compute_governor_decision = None
        self.policy_training_active = False
        self.policy_training_detail = ""
        self.policy_training_plan = {}

    def refresh_compute_profile(self):
        """Refresh the authoritative hardware map and Safe-80 protection profile.

        Hardware/resource identity is frozen for the lifetime of an active policy-training
        session.  A mid-training rescan could otherwise make the Dashboard show a resource
        topology different from the one that admitted the running branches.
        """
        if self.policy_training_active:
            raise RuntimeError(
                "Compute topology cannot be refreshed while policy training is active. "
                "Request Safe Stop first, then rescan the system."
            )
        topology = self.compute_topology_service.scan()
        profile = self.compute_budget_engine.calculate(topology)
        self.compute_topology = topology
        self.compute_protection_profile = profile
        self.compute_governor = AdaptiveComputeGovernor(
            profile,
            monitor=self.compute_topology_service.monitor,
            config=GovernorConfig(allocation_limit_fraction=float(profile.allocation_limit_fraction)),
        )
        self.compute_governor_decision = self.compute_governor.sample(active_branches=0)
        self.compute_profile_changed.emit(profile)
        self.compute_governor_changed.emit(self.compute_governor_decision)
        return topology, profile

    def sample_compute_governor(self, *, active_branches: int | None = None):
        """Refresh live protection telemetry without changing the authoritative hardware identity."""
        if self.compute_governor is None:
            if self.compute_protection_profile is None:
                return None
            self.compute_governor = AdaptiveComputeGovernor(
                self.compute_protection_profile,
                monitor=self.compute_topology_service.monitor,
                config=GovernorConfig(
                    allocation_limit_fraction=float(
                        self.compute_protection_profile.allocation_limit_fraction
                    )
                ),
            )
        if active_branches is None:
            active_branches = int(dict(self.policy_training_plan or {}).get("active_branches", 0) or 0)
        decision = self.compute_governor.sample(active_branches=max(0, int(active_branches)))
        self.compute_governor_decision = decision
        self.compute_governor_changed.emit(decision)
        return decision


    def begin_policy_training(self, detail: str = "Policy training active") -> None:
        self.policy_training_active = True
        self.policy_training_detail = str(detail or "Policy training active")
        self.policy_training_changed.emit(True, self.policy_training_detail)

    def end_policy_training(self, detail: str = "") -> None:
        self.policy_training_active = False
        self.policy_training_detail = str(detail or "")
        self.policy_training_changed.emit(False, self.policy_training_detail)

    def update_policy_training_plan(self, payload: dict | None) -> None:
        self.policy_training_plan = dict(payload or {})
        self.policy_training_plan_changed.emit(dict(self.policy_training_plan))

    def governing_policy_status(self):
        return evaluate_governing_policy(self.policy_registry)

    def notify_policy_state_changed(self):
        status = self.governing_policy_status()
        self.policy_state_changed.emit(status)
        return status

    def update_config(self):
        self.config_changed.emit(self.config)

    def set_case(self, case):
        self.current_case = case
        self.current_power_flow = None
        self.case_changed.emit(case)

    def set_theme(self, theme):
        self.theme = theme
        self.theme_changed.emit(theme)
