from __future__ import annotations
import importlib.util
import pytest
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.power_system.independent_validator import validate_against_pypower
pytestmark=pytest.mark.skipif(importlib.util.find_spec('pypower') is None,reason='PYPOWER is not installed')
@pytest.mark.parametrize('name',['case30','case57','case118','case300'])
def test_internal_solver_crosschecks_pypower(name):
    case=CaseLoader.load(name);internal=run_ac_power_flow(case);assert internal.converged;cross=validate_against_pypower(case,internal);assert cross.available;assert cross.passed,cross.message
