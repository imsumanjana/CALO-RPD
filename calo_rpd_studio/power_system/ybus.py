"""Sparse bus and branch admittance matrix construction."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from .case_model import *

@dataclass(slots=True)
class AdmittanceMatrices:
    ybus: csr_matrix; y_from: csr_matrix; y_to: csr_matrix

def build_ybus(case:PowerSystemCase)->AdmittanceMatrices:
    n=case.n_bus; nl=case.n_branch; idx=case.bus_index_map()
    rows=[]; cols=[]; vals=[]
    yf=np.zeros((nl,n),dtype=complex); yt=np.zeros((nl,n),dtype=complex)
    for k,br in enumerate(case.branch):
        if br[BR_STATUS]<=0: continue
        f=idx[int(br[F_BUS])]; t=idx[int(br[T_BUS])]
        z=complex(br[BR_R],br[BR_X]); y=0j if abs(z)==0 else 1/z; b=1j*br[BR_B]/2
        tap=br[TAP] if br[TAP]!=0 else 1.0; shift=np.deg2rad(br[SHIFT]); a=tap*np.exp(1j*shift)
        yff=(y+b)/(a*np.conj(a)); yft=-y/np.conj(a); ytf=-y/a; ytt=y+b
        yf[k,f]=yff; yf[k,t]=yft; yt[k,f]=ytf; yt[k,t]=ytt
        for r,c,v in ((f,f,yff),(f,t,yft),(t,f,ytf),(t,t,ytt)):
            rows.append(r);cols.append(c);vals.append(v)
    ybus=csr_matrix((vals,(rows,cols)),shape=(n,n),dtype=complex)
    sh=(case.bus[:,GS]+1j*case.bus[:,BS])/case.base_mva
    ybus=ybus+csr_matrix((sh,(np.arange(n),np.arange(n))),shape=(n,n))
    return AdmittanceMatrices(ybus,csr_matrix(yf),csr_matrix(yt))
