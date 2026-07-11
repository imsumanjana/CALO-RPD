"""Standard optimizer result record."""
from __future__ import annotations
from dataclasses import dataclass,field
from typing import Any
import numpy as np
@dataclass(slots=True)
class OptimizerResult:
    algorithm:str;seed:int;parameters:dict[str,Any];best_vector:np.ndarray;decoded_controls:dict[str,float];best_objective:float;objective_components:dict[str,float];total_constraint_violation:float;feasible:bool;evaluations:int;iterations:int;convergence_history:list[float];runtime_seconds:float;final_population:np.ndarray|None;termination_reason:str;metadata:dict[str,Any]=field(default_factory=dict)
