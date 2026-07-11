import numpy as np
from calo_rpd_studio.robustness.cvar import weighted_cvar
from calo_rpd_studio.robustness.scenario_generator import ScenarioGeneratorConfig,generate_load_scenarios
from calo_rpd_studio.robustness.contingencies import n_minus_one_generator_scenarios
from calo_rpd_studio.power_system.case_model import BUS_TYPE,REF

def test_cvar_and_seeded_scenarios():
    assert weighted_cvar([1,2,10],[.4,.4,.2],.8)>=2
    a=generate_load_scenarios(ScenarioGeneratorConfig(3,.1,.1),123)
    b=generate_load_scenarios(ScenarioGeneratorConfig(3,.1,.1),123)
    assert [x.name for x in a]==[x.name for x in b]

def test_generator_outage_preserves_reference(toy_case):
    s=n_minus_one_generator_scenarios([0])[0].apply(toy_case)
    assert np.sum(s.bus[:,BUS_TYPE].astype(int)==REF)==1
