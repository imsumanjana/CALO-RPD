"""Fairness audit for comparative experiments."""
from dataclasses import dataclass,field
from .evaluation_budget import BudgetPolicy
@dataclass(slots=True)
class FairnessReport:fair:bool;errors:list[str]=field(default_factory=list);warnings:list[str]=field(default_factory=list)
def validate_fairness(config):
    errors=[];warnings=[]
    try:config.validate()
    except Exception as exc:errors.append(str(exc));return FairnessReport(False,errors,warnings)
    if config.budget.policy is BudgetPolicy.EQUAL_EVALUATIONS and config.budget.max_evaluations<config.population_size:errors.append('Equal-evaluation budget must be at least the population size.')
    if config.budget.policy is BudgetPolicy.ALGORITHM_NATIVE:warnings.append('Algorithm-native limits do not provide a universal equal-cost comparison.')
    if config.budget.policy is BudgetPolicy.EQUAL_WALL_CLOCK:warnings.append('Wall-clock comparisons depend on hardware, operating-system load, and implementation details; retain full provenance.')
    if len(set(config.algorithms))!=len(config.algorithms):errors.append('Each primary algorithm may appear only once in one comparison protocol.')
    return FairnessReport(not errors,errors,warnings)
