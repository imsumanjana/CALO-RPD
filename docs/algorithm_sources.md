# Algorithm Formulations and Sources

The primary benchmark registry contains twenty algorithms. All implementations use the same bounded
normalized search space and the same ORPD evaluator. The source code identifies the software
formulation actually executed; parameter values are stored with every run.

| ID | Method | Implementation basis |
|---|---|---|
| CALO | Cognitive Adaptive Learning Optimizer | CALO methodology in this repository |
| TLBO | Teaching-Learning-Based Optimization | Rao, Savsani, and Vakharia, *Computer-Aided Design*, 2011 |
| PSO | Particle Swarm Optimization | Kennedy and Eberhart, IEEE ICNN, 1995 |
| CLPSO | Comprehensive Learning PSO | Liang et al., *IEEE Transactions on Evolutionary Computation*, 2006 |
| MTLA-DE | Modified teaching-learning + DE/rand/1/bin | Explicit hybrid formulation in `algorithms/mtla_de.py`; report this exact software formulation |
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

## v3 accelerator implementation rule

The v3 PyTorch suite translates the declared canonical population operators to FP64 tensor kernels.
The common physical evaluator, mixed-variable decoder, feasibility-first rule, boundary handling, and
evaluation budget are shared across all methods. Accelerator conversion is not treated as an algorithmic
enhancement: a CUDA result should agree with the trusted CPU formulation within the declared numerical
tolerances. Any deliberate change to a baseline equation must be registered under a distinct method name
and evaluated as a separate algorithm rather than silently replacing the canonical baseline.
