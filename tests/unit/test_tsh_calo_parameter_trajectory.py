from calo_rpd_studio.algorithms.calo.tsh_calo_parameter_trajectory import (
    summarize_parameter_trajectory,
)


def test_parameter_trajectory_summary_uses_scaled_values_and_reward() -> None:
    rows = []
    for index in range(4):
        rows.append(
            {
                "group_parameter_names": [
                    "attraction",
                    "differential",
                    "exploration_sigma",
                    "memory_weight",
                    "diversity_weight",
                    "recovery_fraction",
                ],
                "group_parameter_values": [[0.1 + index] * 6] * 3,
                "reward": float(index),
                "recovery_triggered": index == 3,
            }
        )
    summary = summarize_parameter_trajectory(rows)
    attraction = next(row for row in summary["parameters"] if row["parameter"] == "attraction")
    assert attraction["observations"] == 4
    assert attraction["reward_correlation"] is not None
    assert attraction["mean_when_recovery_triggered"] == 3.1
    assert summary["automatic_parameter_change"] is False
