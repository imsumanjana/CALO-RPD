"""Numerically stable optimizer problem descriptors."""
def orpd_problem_features(problem):
    case=problem.case;return {'dimension':problem.dimension,'buses':case.n_bus,'generators':case.n_gen,'branches':case.n_branch,'scenario_count':len(problem.scenarios),'network_density':case.n_branch/max(case.n_bus*(case.n_bus-1)/2,1)}
