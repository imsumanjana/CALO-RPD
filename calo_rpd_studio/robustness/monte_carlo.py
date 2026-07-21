"""Monte Carlo scenario utility."""

from .scenario_generator import ScenarioGeneratorConfig, generate_load_scenarios


def generate_monte_carlo(count, seed, p_std=0.05, q_std=0.05):
    return generate_load_scenarios(ScenarioGeneratorConfig(count, p_std, q_std), seed)
