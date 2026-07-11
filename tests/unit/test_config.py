from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.evaluation_budget import BudgetPolicy

def test_config_enum_roundtrip(tmp_path):
    c=ExperimentConfig();c.budget.policy=BudgetPolicy.EQUAL_WALL_CLOCK;c.budget.wall_clock_seconds=2.5
    path=c.save(tmp_path/'config.yaml');loaded=ExperimentConfig.load(path)
    assert loaded.budget.policy is BudgetPolicy.EQUAL_WALL_CLOCK
    assert loaded.budget.wall_clock_seconds==2.5
