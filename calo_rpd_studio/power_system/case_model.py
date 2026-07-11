"""MATPOWER-compatible case model and column constants."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

# bus columns
BUS_I=0; BUS_TYPE=1; PD=2; QD=3; GS=4; BS=5; BUS_AREA=6; VM=7; VA=8; BASE_KV=9; ZONE=10; VMAX=11; VMIN=12
PQ=1; PV=2; REF=3; NONE=4
# generator columns
GEN_BUS=0; PG=1; QG=2; QMAX=3; QMIN=4; VG=5; MBASE=6; GEN_STATUS=7; PMAX=8; PMIN=9
# branch columns
F_BUS=0; T_BUS=1; BR_R=2; BR_X=3; BR_B=4; RATE_A=5; RATE_B=6; RATE_C=7; TAP=8; SHIFT=9; BR_STATUS=10; ANGMIN=11; ANGMAX=12

@dataclass(slots=True)
class PowerSystemCase:
    name: str
    base_mva: float
    bus: np.ndarray
    gen: np.ndarray
    branch: np.ndarray
    gencost: np.ndarray|None=None
    def __post_init__(self):
        self.bus=np.asarray(self.bus,dtype=float); self.gen=np.asarray(self.gen,dtype=float); self.branch=np.asarray(self.branch,dtype=float)
        if self.gencost is not None: self.gencost=np.asarray(self.gencost,dtype=float)
    @property
    def n_bus(self): return int(self.bus.shape[0])
    @property
    def n_gen(self): return int(self.gen.shape[0])
    @property
    def n_branch(self): return int(self.branch.shape[0])
    def clone(self):
        return PowerSystemCase(self.name,float(self.base_mva),self.bus.copy(),self.gen.copy(),self.branch.copy(),None if self.gencost is None else self.gencost.copy())
    def bus_index_map(self): return {int(v):i for i,v in enumerate(self.bus[:,BUS_I])}
    def checksum(self):
        h=hashlib.sha256(); h.update(np.asarray([self.base_mva],dtype=np.float64).tobytes())
        for arr in (self.bus,self.gen,self.branch): h.update(np.ascontiguousarray(arr,dtype=np.float64).tobytes())
        return h.hexdigest()
    def to_dict(self):
        return {'name':self.name,'baseMVA':self.base_mva,'bus':self.bus.tolist(),'gen':self.gen.tolist(),'branch':self.branch.tolist(),'gencost':None if self.gencost is None else self.gencost.tolist()}
    @classmethod
    def from_dict(cls,data,name=None):
        return cls(name or data.get('name','custom'),float(data['baseMVA']),np.asarray(data['bus']),np.asarray(data['gen']),np.asarray(data['branch']),None if data.get('gencost') is None else np.asarray(data['gencost']))
