import numpy as np
from calo_rpd_studio.orpd.problem import ORPDProblem,Evaluation
from calo_rpd_studio.orpd.feasibility_rules import better

def test_decoder_and_problem(toy_case):
    problem=ORPDProblem(toy_case)
    assert problem.dimension>0
    z=np.full(problem.dimension,.5)
    case,physical=problem.decoder.decode(z)
    assert len(physical)==problem.dimension
    ev=problem.evaluate(z)
    assert np.isfinite(ev.value)
    state=problem.solution_state(z)
    assert state['case_checksum']==toy_case.checksum()
    assert state['scenarios'][0]['converged']

def test_feasibility_first_rule():
    feasible=Evaluation(10,True,0)
    infeasible=Evaluation(1,False,.1)
    assert better(feasible,infeasible)
    assert not better(infeasible,feasible)
