import numpy as np
import pytest

from calo_rpd_studio.algorithms.calo.ai_controller import PARAMETER_HIGH, PARAMETER_LOW, PARAMETER_NAMES
from calo_rpd_studio.algorithms.calo.tsh_calo_transition_kernel import effective_recovery_fraction


def test_policy_recovery_fraction_is_operational_and_bounded_by_scientist_ceiling():
    actions = np.zeros((3, len(PARAMETER_NAMES)), dtype=float)
    recovery_index = PARAMETER_NAMES.index("recovery_fraction")
    actions[:, recovery_index] = np.asarray([0.0, 0.5, 1.0])
    groups = np.asarray([0, 0, 1, 2], dtype=int)

    value = effective_recovery_fraction(actions, groups, maximum_fraction=0.40)
    scaled = PARAMETER_LOW[recovery_index] + actions[:, recovery_index] * (
        PARAMETER_HIGH[recovery_index] - PARAMETER_LOW[recovery_index]
    )
    expected = (2.0 * scaled[0] + scaled[1] + scaled[2]) / 4.0

    assert value == pytest.approx(min(expected, 0.40))


def test_policy_recovery_fraction_respects_lower_scientist_ceiling():
    actions = np.ones((3, len(PARAMETER_NAMES)), dtype=float)
    groups = np.asarray([0, 1, 2], dtype=int)

    assert effective_recovery_fraction(actions, groups, maximum_fraction=0.18) == pytest.approx(0.18)
