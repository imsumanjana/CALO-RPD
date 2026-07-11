import numpy as np
import pytest
from calo_rpd_studio.algorithms.registry import SPECS,create_optimizer
from calo_rpd_studio.algorithms.base_optimizer import OptimizerConfig
from calo_rpd_studio.orpd.problem import Evaluation

class SphereProblem:
    dimension=5
    def evaluate(self,x):
        x=np.asarray(x);value=float(np.sum((x-.25)**2));return Evaluation(value,True,0,{'sphere':value},{f'x{i}':float(v) for i,v in enumerate(x)})
    def solution_state(self,x):return {'normalized_decision_vector':np.asarray(x).tolist(),'scenarios':[{'converged':True}]}

@pytest.mark.parametrize('name',list(SPECS))
def test_all_primary_algorithms_smoke(name):
    problem=SphereProblem();params=dict(SPECS[name].default_parameters);params['use_ai']=False if name=='CALO' else params.get('use_ai',False)
    opt=create_optimizer(name,problem,OptimizerConfig(population_size=8,max_evaluations=40,max_iterations=40,parameters=params),seed=123)
    result=opt.run()
    assert result.evaluations<=40
    assert np.isfinite(result.best_objective)
    assert result.best_vector.shape==(5,)
