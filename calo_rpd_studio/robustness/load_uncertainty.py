"""Deterministic load perturbation scenario."""

from .scenario import Scenario


def load_scale_scenario(name, p_scale=1.0, q_scale=1.0, weight=1.0):
    def transform(case):
        case.bus[:, 2] *= p_scale
        case.bus[:, 3] *= q_scale
        return case

    return Scenario(name, weight, transform)
