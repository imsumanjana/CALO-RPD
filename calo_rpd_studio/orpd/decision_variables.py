"""Decision-variable descriptors."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
class VariableKind(str,Enum): CONTINUOUS='continuous'; DISCRETE='discrete'
@dataclass(slots=True)
class DecisionVariable:
    name:str; lower:float; upper:float; kind:VariableKind=VariableKind.CONTINUOUS; values:tuple[float,...]=()
