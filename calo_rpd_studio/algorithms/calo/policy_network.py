"""Compact actor-critic network for CALO operator and parameter control."""
from __future__ import annotations
import torch
from torch import nn
class CALOPolicyNetwork(nn.Module):
    def __init__(self,input_dim=14,hidden_dim=64,n_operators=6,n_parameters=6):
        super().__init__();self.input_dim=input_dim;self.hidden_dim=hidden_dim;self.n_operators=n_operators;self.n_parameters=n_parameters
        self.backbone=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,hidden_dim),nn.Tanh())
        self.operator_head=nn.Linear(hidden_dim,n_operators);self.parameter_head=nn.Linear(hidden_dim,n_parameters);self.value_head=nn.Linear(hidden_dim,1)
    def forward(self,x):
        h=self.backbone(x);return self.operator_head(h),torch.sigmoid(self.parameter_head(h)),self.value_head(h).squeeze(-1)
