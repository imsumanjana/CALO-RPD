from __future__ import annotations
import numpy as np
import pytest
from calo_rpd_studio.power_system.case_model import PowerSystemCase


@pytest.fixture
def toy_case():
    # MATPOWER-compatible 3-bus case: slack, PV, PQ.
    bus = np.array(
        [
            [1, 3, 0, 0, 0, 0, 1, 1.04, 0, 230, 1, 1.10, 0.90],
            [2, 2, 20, 10, 0, 0, 1, 1.01, 0, 230, 1, 1.10, 0.90],
            [3, 1, 45, 15, 0, 0, 1, 1.00, 0, 230, 1, 1.10, 0.90],
        ],
        float,
    )
    gen = np.array(
        [
            [1, 40, 0, 100, -100, 1.04, 100, 1, 200, 0],
            [2, 30, 0, 100, -100, 1.01, 100, 1, 150, 0],
        ],
        float,
    )
    branch = np.array(
        [
            [1, 2, 0.02, 0.06, 0.03, 200, 200, 200, 0, 0, 1, -360, 360],
            [1, 3, 0.08, 0.24, 0.025, 200, 200, 200, 0, 0, 1, -360, 360],
            [2, 3, 0.06, 0.18, 0.02, 200, 200, 200, 1.0, 0, 1, -360, 360],
        ],
        float,
    )
    return PowerSystemCase("toy3", 100.0, bus, gen, branch)
