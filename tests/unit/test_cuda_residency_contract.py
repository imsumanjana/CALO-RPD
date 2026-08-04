from __future__ import annotations

import json
from pathlib import Path

from calo_rpd_studio.accelerated.cuda_timing import (
    CUDA_NUMERICAL_TIME_SHARE_TARGET,
    POLICY_EPOCHS_PER_BOUNDARY,
    POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY,
    summarize_cuda_window_timing,
)
from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
    HeterogeneousTrainingConfig,
)
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def test_cuda_window_timing_qualifies_only_consistent_samples_at_target():
    accepted = summarize_cuda_window_timing(
        label="power-system",
        device="cuda:0",
        wall_seconds=10.0,
        cuda_event_seconds=9.6,
    )
    below_target = summarize_cuda_window_timing(
        label="policy",
        device="cuda:0",
        wall_seconds=10.0,
        cuda_event_seconds=9.49,
    )
    contradictory = summarize_cuda_window_timing(
        label="invalid",
        device="cuda:0",
        wall_seconds=1.0,
        cuda_event_seconds=1.1,
    )

    assert accepted.target_met is True
    assert accepted.cuda_time_share == 0.96
    assert below_target.target_met is False
    assert contradictory.measurement_consistent is False
    assert contradictory.target_met is False


def test_cuda_batching_defaults_and_generated_schema_are_synchronized():
    config = ExperimentConfig()
    training = HeterogeneousTrainingConfig()
    schema = json.loads(
        Path("calo_rpd_studio/data/schemas/experiment_config.schema.json").read_text(
            encoding="utf-8"
        )
    )

    assert CUDA_NUMERICAL_TIME_SHARE_TARGET == 0.95
    assert POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY == 100
    assert POLICY_EPOCHS_PER_BOUNDARY == 10
    assert config.tensor_batch_size == POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY
    assert config.calibration_batch_sizes == [100, 200, 400]
    assert training.training_tensor_batch_size == POWER_SYSTEM_EVALUATIONS_PER_BOUNDARY
    assert schema["properties"]["tensor_batch_size"]["default"] == 100
    assert schema["properties"]["calibration_batch_sizes"]["default"] == [100, 200, 400]
