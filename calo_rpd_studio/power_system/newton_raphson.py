"""Sparse Newton-Raphson AC power-flow kernel."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(slots=True)
class NewtonResult:
    converged:bool; voltage:np.ndarray; iterations:int; max_mismatch:float; mismatch_history:list[float]

def solve_newton_raphson(ybus,sbus,v0,ref,pv,pq,tolerance=1e-8,max_iterations=30):
    v=np.asarray(v0,dtype=complex).copy(); pvpq=np.r_[pv,pq].astype(int); pq=np.asarray(pq,dtype=int)
    history=[]
    for it in range(max_iterations+1):
        i=ybus@v; calc=v*np.conj(i); mis=sbus-calc
        f=np.r_[mis[pvpq].real,mis[pq].imag]; norm=float(np.max(np.abs(f))) if f.size else 0.0; history.append(norm)
        if norm<tolerance: return NewtonResult(True,v,it,norm,history)
        if it==max_iterations: break
        vm=np.abs(v); va=np.angle(v); g=ybus.real.toarray(); b=ybus.imag.toarray(); p=calc.real; q=calc.imag
        nbus=len(v); H=np.zeros((nbus,nbus)); N=np.zeros((nbus,nbus)); M=np.zeros((nbus,nbus)); L=np.zeros((nbus,nbus))
        for a in range(nbus):
            for c in range(nbus):
                if a==c:
                    H[a,a]=-q[a]-b[a,a]*vm[a]**2
                    N[a,a]=p[a]/max(vm[a],1e-15)+g[a,a]*vm[a]
                    M[a,a]=p[a]-g[a,a]*vm[a]**2
                    L[a,a]=q[a]/max(vm[a],1e-15)-b[a,a]*vm[a]
                else:
                    d=va[a]-va[c]; H[a,c]=vm[a]*vm[c]*(g[a,c]*np.sin(d)-b[a,c]*np.cos(d))
                    N[a,c]=vm[a]*(g[a,c]*np.cos(d)+b[a,c]*np.sin(d))
                    M[a,c]=-vm[a]*vm[c]*(g[a,c]*np.cos(d)+b[a,c]*np.sin(d))
                    L[a,c]=vm[a]*(g[a,c]*np.sin(d)-b[a,c]*np.cos(d))
        j=np.block([[H[np.ix_(pvpq,pvpq)],N[np.ix_(pvpq,pq)]],[M[np.ix_(pq,pvpq)],L[np.ix_(pq,pq)]]])
        try: dx=np.linalg.solve(j,f)
        except np.linalg.LinAlgError: return NewtonResult(False,v,it,norm,history)
        va[pvpq]+=dx[:len(pvpq)]; vm[pq]+=dx[len(pvpq):]
        if np.any(vm<=0) or not np.all(np.isfinite(dx)): return NewtonResult(False,v,it,norm,history)
        v=vm*np.exp(1j*va)
    return NewtonResult(False,v,max_iterations,history[-1],history)
