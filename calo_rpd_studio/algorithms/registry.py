"""Registry of the twenty primary optimizers."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .base_optimizer import OptimizerConfig
from .calo.optimizer import CALOOptimizer
from .tlbo import TLBOOptimizer
from .pso import PSOOptimizer
from .clpso import CLPSOOptimizer
from .mtla_de import MTLADEOptimizer
from .qode import QODEOptimizer
from .dragonfly import DragonflyOptimizer
from .simulated_annealing import SimulatedAnnealingOptimizer
from .salp_swarm import SalpSwarmOptimizer
from .ant_colony_continuous import AntColonyContinuousOptimizer
from .bat import BatOptimizer
from .crow_search import CrowSearchOptimizer
from .firefly import FireflyOptimizer
from .flower_pollination import FlowerPollinationOptimizer
from .grasshopper import GrasshopperOptimizer
from .grey_wolf import GreyWolfOptimizer
from .moth_flame import MothFlameOptimizer
from .multi_verse import MultiVerseOptimizer
from .whale import WhaleOptimizer
from .imperialist_competitive import ImperialistCompetitiveOptimizer
from .torch_suite import TorchCanonicalOptimizer

@dataclass(frozen=True,slots=True)
class AlgorithmSpec:
    name:str; cls:type; description:str; default_parameters:dict[str,Any]

SPECS={
'CALO':AlgorithmSpec('CALO',CALOOptimizer,'CALO Core v2 with adaptive epsilon-feasibility, dual archives, per-individual operator allocation, mixed-variable learning, success-distribution memory, online operator credit, diversity recovery, and hierarchical AI control.',{'use_ai':True,'use_memory':True,'use_dual_archives':True,'use_epsilon':True,'use_mixed_variable':True,'use_diversity_recovery':True,'use_local_intensification':True,'epsilon_quantile':0.75,'epsilon_control_fraction':0.65,'epsilon_exponent':2.0,'stagnation_window':12,'ai_credit_blend':0.65,'ai_policy_weight':0.35}),
'TLBO':AlgorithmSpec('TLBO',TLBOOptimizer,'Teaching-Learning-Based Optimization.',{}),
'PSO':AlgorithmSpec('PSO',PSOOptimizer,'Particle Swarm Optimization.',{'inertia':.7298,'c1':1.49618,'c2':1.49618}),
'CLPSO':AlgorithmSpec('CLPSO',CLPSOOptimizer,'Comprehensive Learning Particle Swarm Optimization.',{'refresh_gap':7,'c':1.49445}),
'MTLA-DE':AlgorithmSpec('MTLA-DE',MTLADEOptimizer,'Modified teaching-learning search with DE/rand/1/bin hybridization.',{'f':.5,'cr':.9}),
'QODE':AlgorithmSpec('QODE',QODEOptimizer,'Quasi-Oppositional Differential Evolution.',{'f':.5,'cr':.9}),
'DA':AlgorithmSpec('DA',DragonflyOptimizer,'Dragonfly Algorithm.',{}),
'SA':AlgorithmSpec('SA',SimulatedAnnealingOptimizer,'Continuous Simulated Annealing.',{'temperature':1.0,'cooling':.995,'step_scale':.1}),
'SSA':AlgorithmSpec('SSA',SalpSwarmOptimizer,'Salp Swarm Algorithm.',{}),
'ACO':AlgorithmSpec('ACO',AntColonyContinuousOptimizer,'ACOR continuous-domain Ant Colony Optimization.',{'q':.5,'xi':.85}),
'BA':AlgorithmSpec('BA',BatOptimizer,'Bat Algorithm.',{'loudness':.9,'pulse_rate':.5}),
'CSA':AlgorithmSpec('CSA',CrowSearchOptimizer,'Crow Search Algorithm.',{'awareness_probability':.1,'flight_length':2.0}),
'FA':AlgorithmSpec('FA',FireflyOptimizer,'Firefly Algorithm.',{'alpha':.2,'beta0':1.0,'gamma':1.0}),
'FPA':AlgorithmSpec('FPA',FlowerPollinationOptimizer,'Flower Pollination Algorithm.',{'switch_probability':.8}),
'GOA':AlgorithmSpec('GOA',GrasshopperOptimizer,'Grasshopper Optimization Algorithm.',{}),
'GWO':AlgorithmSpec('GWO',GreyWolfOptimizer,'Grey Wolf Optimizer.',{}),
'MFO':AlgorithmSpec('MFO',MothFlameOptimizer,'Moth-Flame Optimization.',{}),
'MVO':AlgorithmSpec('MVO',MultiVerseOptimizer,'Multi-Verse Optimizer.',{}),
'WOA':AlgorithmSpec('WOA',WhaleOptimizer,'Whale Optimization Algorithm.',{'spiral_b':1.0}),
'ICA':AlgorithmSpec('ICA',ImperialistCompetitiveOptimizer,'Imperialist Competitive Algorithm.',{'imperialists':5}),
}

def create_optimizer(name,problem,config:OptimizerConfig,seed=0,progress_callback=None,cancel_callback=None):
    if name not in SPECS:raise KeyError(f'Unknown optimizer: {name}')
    parameters=dict(config.parameters or {})
    device=str(parameters.get('execution_device','cpu')).lower()
    backend=str(parameters.get('optimizer_backend','legacy')).lower()
    if name!='CALO' and backend=='torch':
        return TorchCanonicalOptimizer(name,problem,config,seed,progress_callback,cancel_callback)
    return SPECS[name].cls(problem,config,seed,progress_callback,cancel_callback)

def primary_algorithm_names():return tuple(SPECS)
