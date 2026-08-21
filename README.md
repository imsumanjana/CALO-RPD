# CALO-RPD Studio v12.0.0-dev.1

CALO-RPD Studio is a Python scientific desktop application for robust optimal reactive power dispatch (ORPD), with deterministic experiment orchestration, metaheuristic optimization, PyQt6 workflows, SQLite-backed persistence, CUDA-preferred/CPU-only execution, and a separately governed TSH-CALO policy lifecycle.

> **Development status only.** This branch is not a release candidate or final release. Historical v6.x release evidence does not qualify the active v12 source. Policy training, qualification, activation, protected-case evaluation, publication, and release remain separate gated workflows.

## Start here

For a repository checkout on Windows, create the environment once and run the setup-aware bootstrap:

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe .\bootstrap.py --setup
```

Routine native launch is separate from setup:

```powershell
& .\Launch-CALO-RPD.ps1
```

For an installed environment, `calo-rpd-native` launches the application directly. `calo-rpd-studio` remains the setup-aware packaged entry point.

See [`docs/NATIVE_WINDOWS_GUIDE.md`](docs/NATIVE_WINDOWS_GUIDE.md) for setup, CUDA/CPU selection, logs, data locations, shutdown, Docker operation, and troubleshooting.

## Scientific and execution boundaries

- TSH-CALO A-E are the production architecture surface; F remains experimental, evidence-gated, and disabled by default.
- CUDA-preferred and CPU-only are executable modes. Intel XPU metadata may remain readable historically but is not an executable v12 mode.
- Safe-80 resource values are admission ceilings based on currently available memory, not claims about current utilization.
- Deterministic scientific baselines, exact function-evaluation accounting, seeds, immutable experiment plans, and provenance identities must be preserved.
- Protected case118/case300 workflows remain isolated from ordinary development/training use.

## Policy lifecycle

Policy-free CALO remains a normal algorithm path. TSH-CALO policy work is deliberately separate:

`training candidate -> formal qualification -> explicit activation -> immutable experiment binding`

Training completion never implies qualification or activation. Existing checked-in model records are historical evidence and are not ordinary development artifacts.

## Repository layout

- `calo_bootstrap/` and `bootstrap.py` — setup/repair and setup-aware launch.
- `calo_rpd_studio/app/` and `calo_rpd_studio/gui/` — application state and scientist-facing desktop workflow.
- `calo_rpd_studio/power_system/`, `orpd/`, and `robustness/` — cases, AC power flow, ORPD formulation, constraints, objectives, and scenarios.
- `calo_rpd_studio/algorithms/` — optimization algorithms; `algorithms/calo/` owns TSH-CALO lifecycle implementation.
- `calo_rpd_studio/compute/` and `accelerated/` — resource admission and CUDA/CPU execution.
- `calo_rpd_studio/experiments/`, `benchmarking/`, `results/`, `resume/`, and `continuation/` — experiment planning, execution, persistence, and recovery.
- `containers/`, `Dockerfile`, and `compose.yaml` — optional reproducible container runtime.
- `tests/` — unit, integration, regression, scientific, and GUI coverage.
- `.ai/` and `scripts/ai-index*` — repository intelligence used by coding agents; indexing is navigation evidence, not correctness proof.

## Documentation

Use [`docs/DOCUMENTATION_STATUS.md`](docs/DOCUMENTATION_STATUS.md) to distinguish current guidance from immutable historical records. Important current references include:

- [`docs/architecture.md`](docs/architecture.md)
- [`docs/user_guide.md`](docs/user_guide.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/CONTAINER_RUNBOOK.md`](docs/CONTAINER_RUNBOOK.md)
- [`docs/NATIVE_WINDOWS_GUIDE.md`](docs/NATIVE_WINDOWS_GUIDE.md)

Release history is consolidated in [`CHANGELOG.md`](CHANGELOG.md). Historical audit, implementation, qualification, and release records are retained for provenance but do not override the current v12 status or current source behavior.

## Development

The project is packaged with setuptools through `pyproject.toml`. The active Linux/Python 3.11 validation and container dependency locks are:

- `requirements-lock-cpu-py311-linux.txt`
- `requirements-lock-cuda128-py311-linux.txt`
- `requirements-lock-ci-py311-linux.txt`

Repository intelligence should be consulted before broad source exploration:

```text
python scripts/ai-index check
python scripts/ai-index context "<task/question>" --no-semantic
```

After code changes, update repository intelligence and run only the validation appropriate to the change and current project authorization.
