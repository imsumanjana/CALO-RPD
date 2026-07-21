import numpy as np
from calo_rpd_studio.power_system.ybus import build_ybus
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.power_system.voltage_stability import kessel_glavitsch_l_index


def test_ybus_and_power_flow(toy_case):
    y = build_ybus(toy_case)
    assert y.ybus.shape == (3, 3)
    pf = run_ac_power_flow(toy_case)
    assert pf.converged
    assert pf.max_mismatch < 1e-7
    assert np.all(np.isfinite(pf.vm_pu))
    assert pf.total_loss_mw > 0
    l = kessel_glavitsch_l_index(pf.case, pf.voltage)
    assert np.isfinite(l.maximum)
