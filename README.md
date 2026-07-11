# CALO-RPD Studio

**CALO-RPD Studio 1.0.3** is a scientific desktop platform for deterministic and
robust optimal reactive power dispatch (ORPD), reproducible comparison of twenty
optimizers, and research on the **Cognitive Adaptive Learning Optimizer (CALO)**.

The software uses one common physical evaluator for all algorithms, normalized
mixed-variable encoding, AC Newton-Raphson power flow, explicit constraint audits,
seeded experiment records, statistical tests, independent result validation, and
publication export. CALO alone contains the AI policy controller; the remaining
nineteen primary algorithms are conventional comparison baselines.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Launch

```bash
python main.py
# or
calo-rpd-studio
```

## Command-line workflows

```bash
calo-rpd-benchmark --case case30 --algorithms CALO,TLBO,PSO --runs 5 --budget 5000
calo-rpd-train --epochs 20 --seed 2026
calo-rpd-validate --case case118
calo-rpd-export --database calo_rpd_results.sqlite --experiment <EXPERIMENT_ID>
```

## Primary algorithms

CALO, TLBO, PSO, CLPSO, MTLA-DE, QODE, Dragonfly, Simulated Annealing,
Salp Swarm, continuous-domain Ant Colony Optimization, Bat, Crow Search,
Firefly, Flower Pollination, Grasshopper, Grey Wolf, Moth-Flame, Multi-Verse,
Whale Optimization, and Imperialist Competitive Algorithm.

## Guided scientific workflow

The GUI enforces the research sequence instead of exposing every workspace at once:

1. **Power System** — load a case, run the base AC power flow, and pass the independent PYPOWER cross-check.
2. **ORPD Formulation** — apply objectives, control variables, discrete device behavior, and constraints.
3. **Algorithms** — select the comparison methods and apply declared parameters.
4. **CALO Intelligence** — when CALO is selected, validate and apply the frozen policy checkpoint.
5. **Robust Scenarios** — apply deterministic or robust scenario configuration.
6. **Experiment Manager** — pass the fairness audit; only then are experiment run buttons enabled.
7. **Live Optimization** — becomes available when an experiment starts.
8. **Statistical Analysis** — becomes available after the experiment completes.
9. **Results Explorer** — becomes available after statistical analysis; confirm result review before validation.
10. **Validation & Audit** — independently re-evaluate stored decisions.
11. **Publication Export** — unlocks only after at least one result from the current experiment is verified.

Dashboard and Application Settings remain available throughout. Changing an upstream scientific stage invalidates dependent downstream workflow state.

The bottom application status bar reports **Ready**, **Busy**, **Completed**, or **Failed**, together with the active operation, progress, elapsed time, and safe cancellation for supported long-running tasks.

## Reproducibility

Each run records its seed tuple, algorithm parameters, case checksum, objective,
scenario configuration, evaluation budget, convergence history, final physical
state, software environment, and validation state. The default CALO policy ships
with metadata and a SHA-256 checksum. See `docs/reproducibility.md`.

## Documentation

- `docs/architecture.md`
- `docs/mathematical_formulation.md`
- `docs/calo_methodology.md`
- `docs/algorithm_sources.md`
- `docs/reproducibility.md`
- `docs/validation.md`
- `docs/user_guide.md`
- `RELEASE_VALIDATION.md`

## License

MIT License. Scientific results remain the responsibility of the experimenter;
all comparative claims should be based on the complete predefined protocol and
verified raw results.
