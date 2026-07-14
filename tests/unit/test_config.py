from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.evaluation_budget import BudgetPolicy

def test_config_enum_roundtrip(tmp_path):
    c=ExperimentConfig();c.budget.policy=BudgetPolicy.EQUAL_WALL_CLOCK;c.budget.wall_clock_seconds=2.5
    path=c.save(tmp_path/'config.yaml');loaded=ExperimentConfig.load(path)
    assert loaded.budget.policy is BudgetPolicy.EQUAL_WALL_CLOCK
    assert loaded.budget.wall_clock_seconds==2.5


def test_config_roundtrip_preserves_hybrid_compute_policy(tmp_path):
    config = ExperimentConfig(
        execution_backend="adaptive_hybrid",
        gpu_utilization_target=72,
        cpu_utilization_target=48,
        gpu_memory_limit=88,
        gpu_parallel_jobs=2,
        xpu_utilization_target=68,
        xpu_memory_limit=82,
        xpu_parallel_jobs=3,
        system_memory_limit=79,
    )
    loaded = ExperimentConfig.load(config.save(tmp_path / "hybrid.yaml"))
    assert loaded.execution_backend == "adaptive_hybrid"
    assert loaded.gpu_utilization_target == 72
    assert loaded.cpu_utilization_target == 48
    assert loaded.gpu_memory_limit == 88
    assert loaded.gpu_parallel_jobs == 2
    assert loaded.xpu_utilization_target == 68
    assert loaded.xpu_memory_limit == 82
    assert loaded.xpu_parallel_jobs == 3
    assert loaded.system_memory_limit == 79


def test_config_roundtrip_preserves_weighted_device_shares(tmp_path):
    config = ExperimentConfig(
        execution_backend="weighted_split",
        cuda_task_share=50,
        xpu_task_share=30,
        cpu_task_share=20,
    )
    loaded = ExperimentConfig.load(config.save(tmp_path / "weighted.yaml"))
    assert loaded.execution_backend == "weighted_split"
    assert (loaded.cuda_task_share, loaded.xpu_task_share, loaded.cpu_task_share) == (50, 30, 20)


def test_weighted_device_shares_must_sum_to_100():
    config = ExperimentConfig(cuda_task_share=60, xpu_task_share=30, cpu_task_share=20)
    try:
        config.validate()
    except ValueError as exc:
        assert "sum to 100" in str(exc)
    else:
        raise AssertionError("invalid weighted device shares were accepted")
