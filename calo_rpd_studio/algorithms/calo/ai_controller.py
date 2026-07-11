"""CALO policy checkpoint loading and reproducible inference."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import torch
from .policy_network import CALOPolicyNetwork
@dataclass(slots=True)
class PolicyDecision: operator:int;probabilities:np.ndarray;parameters:dict[str,float];value_estimate:float
PARAMETER_NAMES=('exploitation','peer_learning','exploration_sigma','memory_weight','recovery_intensity','recovery_fraction')
class AIController:
    def __init__(self,checkpoint=None,seed=0,deterministic=False,input_dim=14):
        self.rng=np.random.default_rng(seed);self.deterministic=deterministic;self.network=CALOPolicyNetwork(input_dim=input_dim);self.metadata={};self.checkpoint_path='';self.checksum=''
        if checkpoint and Path(checkpoint).exists():self.load(checkpoint)
        else:
            torch.manual_seed(2026)
            for p in self.network.parameters():torch.nn.init.xavier_uniform_(p) if p.ndim>1 else torch.nn.init.zeros_(p)
        self.network.eval()
    def load(self,path):
        path=Path(path);payload=torch.load(path,map_location='cpu',weights_only=False);state=payload.get('model_state_dict',payload);self.network.load_state_dict(state);self.metadata=dict(payload.get('metadata',{}));self.checkpoint_path=str(path);self.checksum=hashlib.sha256(path.read_bytes()).hexdigest();self.network.eval()
    def decide(self,state_vector):
        x=torch.tensor(np.asarray(state_vector,float),dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():logits,raw,value=self.network(x);probs=torch.softmax(logits,dim=-1)[0].cpu().numpy();raw=raw[0].cpu().numpy()
        operator=int(np.argmax(probs)) if self.deterministic else int(self.rng.choice(len(probs),p=probs/probs.sum()))
        # Bounded engineering ranges.
        lo=np.asarray([.15,.10,.005,.0,.10,.05]);hi=np.asarray([1.50,1.20,.30,1.00,1.50,.50]);vals=lo+raw*(hi-lo)
        return PolicyDecision(operator,probs,{k:float(v) for k,v in zip(PARAMETER_NAMES,vals)},float(value.item()))
