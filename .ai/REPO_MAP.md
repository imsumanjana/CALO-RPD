# CALO-RPD repository map

CALO-RPD Studio is a Python 3.11+ scientific desktop application for robust optimal reactive power dispatch. The active v12 development line combines a PyQt6 desktop shell, ORPD/power-flow domain models, metaheuristic optimizers, TSH-CALO policy lifecycle tooling, deterministic experiment orchestration, SQLite-backed persistence, and CUDA-preferred/CPU-only compute scheduling.

## Read-first map
- `calo_bootstrap/`, `bootstrap.py`, launch scripts — setup/repair and native application launch.
- `calo_rpd_studio/app/` + `gui/` — application state, workflows, task orchestration, scientist-facing PyQt UI.
- `calo_rpd_studio/power_system/` + `orpd/` + `robustness/` — case loading, AC power flow, ORPD formulation, objectives/constraints and robust scenarios.
- `calo_rpd_studio/algorithms/` — optimization algorithms; `algorithms/calo/` owns TSH-CALO policy/training/qualification artifacts and lifecycle rules.
- `calo_rpd_studio/compute/` + `accelerated/` — resource admission, device binding, CUDA/CPU execution, persistent workers and accelerator kernels.
- `calo_rpd_studio/experiments/` + `benchmarking/` — experiment plans/configuration, budgets, seeds, runners, fairness and benchmark orchestration.
- `calo_rpd_studio/results/`, `resume/`, `continuation/` — SQLite results, integrity/publication surfaces, restart/resume/extension state.
- `calo_rpd_studio/scripts/`, `.github/workflows/`, `containers/` — validation/release tooling, CI and packaging/container harnesses.
- `tests/` — pytest unit/integration/regression/scientific/GUI coverage. Use `.ai/test-map.json` before choosing tests.

## Main entry points
- Native app: `calo_bootstrap.launcher:main` / `calo_rpd_studio.app.application:main`.
- Project console scripts are declared in `pyproject.toml`; policy training, qualification, release and validation commands are intentionally separate.
- CI: `.github/workflows/ci.yml`; AI-index freshness will be verified by `.github/workflows/ai-index.yml` after bootstrap.
- AI tooling: `./scripts/ai-index init|update|status|check|validate|context "question"` (PowerShell: `scripts/ai-index.ps1`).

## Primary execution flow
`bootstrap/launcher -> application -> MainWindow/AppState -> scientific workflow -> ExperimentManager/runner -> optimizer + ORPD evaluator -> compute scheduler/device -> ResultDatabase -> GUI/results/publication`

Policy lifecycle is deliberately independent: `training candidate -> formal qualification -> explicit activation -> immutable experiment binding`.

## Build / development
- Package/build: setuptools via `pyproject.toml`.
- Runtime dependencies: NumPy/SciPy/Pandas/Matplotlib/PyQt6/PYPOWER/PyYAML/psutil/NVML/cma/PyTorch.
- Tests: pytest + pytest-qt; quality tools: ruff, black, mypy, coverage.
- Container paths: `Dockerfile`, `compose.yaml`, `containers/`.

Historical release reports, frozen/publication evidence, trained policies and vendored PGLIB data are intentionally excluded from the AI symbol index. Consult them only for a task that specifically concerns historical/release evidence.
