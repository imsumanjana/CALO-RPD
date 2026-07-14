"""Predefined deterministic and robust benchmark study matrix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.robustness.robust_objectives import RobustAggregation


@dataclass(frozen=True, slots=True)
class BenchmarkStudy:
    key: str
    label: str
    description: str
    configure: Callable[[ExperimentConfig], None]


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    cases: tuple[str, ...]
    studies: tuple[BenchmarkStudy, ...]

    def study(self, key: str) -> BenchmarkStudy:
        for item in self.studies:
            if item.key == key:
                return item
        raise KeyError(key)


def _deterministic(config: ExperimentConfig) -> None:
    config.scenarios.mode = "deterministic"
    config.robust_objective.aggregation = RobustAggregation.EXPECTED


def _mixed(config: ExperimentConfig) -> None:
    _deterministic(config)
    config.variables.generator_voltages = True
    config.variables.transformer_taps = True
    config.variables.shunt_compensation = True
    config.variables.discrete_transformer_taps = True
    config.variables.discrete_shunts = True


def _load_mean_risk(config: ExperimentConfig) -> None:
    config.scenarios.mode = "load_uncertainty"
    config.scenarios.count = max(20, int(config.scenarios.count))
    config.robust_objective.aggregation = RobustAggregation.MEAN_RISK


def _load_cvar(config: ExperimentConfig) -> None:
    config.scenarios.mode = "load_uncertainty"
    config.scenarios.count = max(20, int(config.scenarios.count))
    config.robust_objective.aggregation = RobustAggregation.CVAR
    config.robust_objective.cvar_alpha = 0.95


def _renewable(config: ExperimentConfig, aggregation: RobustAggregation) -> None:
    case = CaseLoader.load(config.case_name)
    load_buses = case.bus[case.bus[:, 2] > 0, 0].astype(int)
    bus = int(load_buses[0]) if len(load_buses) else int(case.bus[0, 0])
    total_load = float(case.bus[:, 2].clip(min=0).sum())
    config.scenarios.mode = "renewable_uncertainty"
    config.scenarios.count = max(20, int(config.scenarios.count))
    config.scenarios.renewable_bus = bus
    config.scenarios.renewable_rated_mw = max(1.0, 0.10 * total_load)
    config.robust_objective.aggregation = aggregation


def _renewable_mean_risk(config: ExperimentConfig) -> None:
    _renewable(config, RobustAggregation.MEAN_RISK)


def _renewable_cvar(config: ExperimentConfig) -> None:
    _renewable(config, RobustAggregation.CVAR)
    config.robust_objective.cvar_alpha = 0.95


def _branch_worst_case(config: ExperimentConfig) -> None:
    case = CaseLoader.load(config.case_name)
    count = min(5, len(case.branch))
    config.scenarios.mode = "branch_contingency"
    config.scenarios.branch_outages = list(range(count))
    config.robust_objective.aggregation = RobustAggregation.WORST_CASE


def _generator_worst_case(config: ExperimentConfig) -> None:
    case = CaseLoader.load(config.case_name)
    online = [int(i) for i, row in enumerate(case.gen) if row[7] > 0]
    # Preserve at least one online generator in small systems.
    config.scenarios.mode = "generator_contingency"
    config.scenarios.generator_outages = online[: min(3, max(0, len(online) - 1))]
    config.robust_objective.aggregation = RobustAggregation.WORST_CASE


def standard_benchmark_suite() -> BenchmarkSuite:
    return BenchmarkSuite(
        cases=("case30", "case57", "case118", "case300"),
        studies=(
            BenchmarkStudy("deterministic", "Deterministic ORPD", "Base deterministic ORPD.", _deterministic),
            BenchmarkStudy("mixed", "Mixed discrete-continuous ORPD", "Continuous generator voltages with discrete tap and shunt controls.", _mixed),
            BenchmarkStudy("load_mean_risk", "Load uncertainty · mean-risk", "Scenario-based load uncertainty aggregated by mean-risk.", _load_mean_risk),
            BenchmarkStudy("load_cvar", "Load uncertainty · CVaR", "Scenario-based load uncertainty aggregated by CVaR.", _load_cvar),
            BenchmarkStudy("renewable_mean_risk", "Renewable uncertainty · mean-risk", "Renewable injection uncertainty aggregated by mean-risk.", _renewable_mean_risk),
            BenchmarkStudy("renewable_cvar", "Renewable uncertainty · CVaR", "Renewable injection uncertainty aggregated by CVaR.", _renewable_cvar),
            BenchmarkStudy("branch_worst_case", "N-1 branch contingencies · worst case", "Selected branch outages aggregated by worst-case objective.", _branch_worst_case),
            BenchmarkStudy("generator_worst_case", "N-1 generator contingencies · worst case", "Selected generator outages aggregated by worst-case objective.", _generator_worst_case),
        ),
    )
