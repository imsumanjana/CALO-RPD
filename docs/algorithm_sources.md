# Algorithm Formulations and Sources

The primary benchmark registry is enumerated at runtime. All implementations use the same bounded
normalized search space and the same ORPD evaluator. The source code identifies the software
formulation actually executed; parameter values are stored with every run.

| ID | Method | Implementation basis |
|---|---|---|
| CALO | Cognitive Adaptive Learning Optimizer | CALO methodology in this repository |
| TLBO | Teaching-Learning-Based Optimization | Rao, Savsani, and Vakharia, *Computer-Aided Design*, 2011 |
| PSO | Particle Swarm Optimization | Kennedy and Eberhart, IEEE ICNN, 1995 |
| CLPSO | Comprehensive Learning PSO | Liang et al., *IEEE Transactions on Evolutionary Computation*, 2006 |
| CMA-ES | Active Covariance Matrix Adaptation Evolution Strategy | Hansen, Akimoto, and Baudis, maintained [official pycma implementation](https://github.com/CMA-ES/pycma), [Zenodo DOI 10.5281/zenodo.2559635](https://doi.org/10.5281/zenodo.2559635) |
| MTLA-DE | Modified teaching-learning + DE/rand/1/bin | Explicit hybrid formulation in `algorithms/mtla_de.py`; report this exact software formulation |
| L-SHADE | Success-history adaptive DE with linear population-size reduction | Tanabe and Fukunaga, IEEE CEC 2014, [DOI 10.1109/CEC.2014.6900380](https://doi.org/10.1109/CEC.2014.6900380); corrected L-SHADE 1.0.1 mechanics from the [authors' source release](https://ryojitanabe.github.io/publication) |
| QODE | Quasi-Oppositional Differential Evolution | Quasi-oppositional initialization with DE/rand/1/bin |
| DA | Dragonfly Algorithm | Mirjalili, *Neural Computing and Applications*, 2016 |
| SA | Simulated Annealing | Metropolis acceptance with bounded continuous Gaussian neighborhood |
| SSA | Salp Swarm Algorithm | Mirjalili et al., *Advances in Engineering Software*, 2017 |
| ACO | ACOR | Socha and Dorigo, continuous-domain ant colony optimization, 2008 |
| BA | Bat Algorithm | Yang, 2010 |
| CSA | Crow Search Algorithm | Askarzadeh, *Computers & Structures*, 2016 |
| FA | Firefly Algorithm | Yang, 2009/2010 formulation |
| FPA | Flower Pollination Algorithm | Yang, 2012 |
| GOA | Grasshopper Optimization Algorithm | Saremi et al., *Advances in Engineering Software*, 2017 |
| GWO | Grey Wolf Optimizer | Mirjalili et al., *Advances in Engineering Software*, 2014 |
| MFO | Moth-Flame Optimization | Mirjalili, *Knowledge-Based Systems*, 2015 |
| MVO | Multi-Verse Optimizer | Mirjalili et al., *Neural Computing and Applications*, 2016 |
| WOA | Whale Optimization Algorithm | Mirjalili and Lewis, *Advances in Engineering Software*, 2016 |
| ICA | Imperialist Competitive Algorithm | Atashpaz-Gargari and Lucas, CEC 2007 |

## Reproducibility note

For literature variants that have multiple published parameterizations, the executable source file and
saved run parameters define the precise formulation used by CALO-RPD Studio. The software does not
claim binary identity with third-party executables. Any manuscript comparison should cite the original
method and disclose the exact configuration exported with the run.

L-SHADE uses the corrected reference defaults for historical memory (`H=5`), p-best rate (`0.11`),
and archive rate (`1.4`). Common-budget studies deliberately supply the campaign population size rather
than the reference executable's `18D` initial-size convention so function-evaluation fairness remains
explicit. For constrained ORPD tasks, selection is the shared Deb feasibility-first rule. Successful
parameter weights use objective decrease between feasible solutions, violation decrease between
infeasible solutions, and a positive transition weight when feasibility is first attained. This
constraint adapter is a disclosed study wrapper; it is not claimed to be part of unconstrained L-SHADE.

CMA-ES delegates ask/tell covariance adaptation to pinned `cma` 4.4.4 with active covariance enabled,
smooth bound transformation on the normalized unit box, and an explicit per-run random generator. It
is comparison based, so the constrained wrapper passes dense ranks from the common Deb ordering rather
than inventing a scale-dependent penalty coefficient. Exact ties receive the same rank. Mixed tap/shunt
variables are latent continuous coordinates decoded by the common mixed-variable decoder; results must
therefore be described as this declared encoding, not as a native integer CMA-ES formulation. The small
covariance controller stays in CPU RAM while the common population evaluator may execute on CUDA. That
residency difference is stored with every run and CUDA-native optimizer-kernel speed is not claimed.

## Mathematical reference adapters

`orpd/mathematical_reference.py` exposes two separately classified reference paths. SciPy SLSQP is
used only for a deterministic local solve of the continuous tap/shunt relaxation. The report pins
the installed SciPy version, finite-difference scheme, start-vector hash, settings, termination,
iterations, backend calls, common-evaluator calls and independent PYPOWER checks. Because AC ORPD is
nonconvex, this result is explicitly not a certified lower bound or global optimum. Its normalized
point is separately decoded and re-evaluated on the original mixed-variable lattice; it is called an
incumbent only when that original evaluation is feasible.

The exhaustive adapter accepts only formulations whose every declared control is discrete and whose
complete Cartesian lattice fits an explicit candidate ceiling. It rejects continuous controls and
oversized lattices. Exactness, when a feasible winner exists, is limited to the complete declared
finite lattice under the common deterministic evaluator; it is never extended to a continuous or
differently controlled ORPD task. Both paths are outside the stochastic algorithm registry and do
not receive an artificial equal-FE budget.

## v3 accelerator implementation rule

The v3 PyTorch suite translates the declared canonical population operators to FP64 tensor kernels.
The common physical evaluator, mixed-variable decoder, feasibility-first rule, boundary handling, and
evaluation budget are shared across all methods. Accelerator conversion is not treated as an algorithmic
enhancement: a CUDA result should agree with the trusted CPU formulation within the declared numerical
tolerances. Any deliberate change to a baseline equation must be registered under a distinct method name
and evaluated as a separate algorithm rather than silently replacing the canonical baseline.
