# optimization

**Purpose:** Common optimizer contracts, implementations, registry and portfolio execution surfaces.

**Important state:** Optimizer population/archive/RNG state plus experiment-supplied configuration.

**Major flow:** experiment plan -> registry -> optimizer -> ORPD objective/constraints -> result.

**Constraints/invariants:** Deterministic seeded behavior, common feasibility semantics and exact evaluation budgets.

**Common failure points:** Budget drift, ranking/feasibility inconsistency, RNG nondeterminism and bound errors.

## Primary files
- `calo_rpd_studio/portfolio/exporter.py`
- `calo_rpd_studio/algorithms/torch_suite.py`
- `calo_rpd_studio/portfolio/study_planning.py`
- `calo_rpd_studio/algorithms/base_optimizer.py`
- `calo_rpd_studio/portfolio/models.py`
- `calo_rpd_studio/algorithms/lshade.py`
- `calo_rpd_studio/portfolio/fingerprint.py`
- `calo_rpd_studio/algorithms/registry.py`
- `calo_rpd_studio/portfolio/planner.py`
- `calo_rpd_studio/algorithms/cma_es.py`
- `calo_rpd_studio/algorithms/grasshopper.py`
- `calo_rpd_studio/portfolio/catalog.py`

## Important public/entry symbols
- `AntColonyContinuousOptimizer` — `calo_rpd_studio/algorithms/ant_colony_continuous.py:8-48`
- `EvaluationBatchInvariantError` — `calo_rpd_studio/algorithms/base_optimizer.py:15-25`
- `OptimizerConfig` — `calo_rpd_studio/algorithms/base_optimizer.py:66-70`
- `BaseOptimizer` — `calo_rpd_studio/algorithms/base_optimizer.py:73-296`
- `BatOptimizer` — `calo_rpd_studio/algorithms/bat.py:9-35`
- `CLPSOOptimizer` — `calo_rpd_studio/algorithms/clpso.py:9-46`
- `CMAESOptimizer` — `calo_rpd_studio/algorithms/cma_es.py:27-102`
- `CrowSearchOptimizer` — `calo_rpd_studio/algorithms/crow_search.py:9-38`
- `DragonflyOptimizer` — `calo_rpd_studio/algorithms/dragonfly.py:8-34`
- `FireflyOptimizer` — `calo_rpd_studio/algorithms/firefly.py:9-43`
- `FlowerPollinationOptimizer` — `calo_rpd_studio/algorithms/flower_pollination.py:10-35`
- `GrasshopperOptimizer` — `calo_rpd_studio/algorithms/grasshopper.py:8-36`
- `GreyWolfOptimizer` — `calo_rpd_studio/algorithms/grey_wolf.py:8-28`
- `ImperialistCompetitiveOptimizer` — `calo_rpd_studio/algorithms/imperialist_competitive.py:9-40`
- `LegacyMTLBOOptimizer` — `calo_rpd_studio/algorithms/legacy_mtlbo.py:9-27`
- `LSHADEOptimizer` — `calo_rpd_studio/algorithms/lshade.py:59-257`
- `MothFlameOptimizer` — `calo_rpd_studio/algorithms/moth_flame.py:8-47`
- `MTLADEOptimizer` — `calo_rpd_studio/algorithms/mtla_de.py:9-43`

## Dependencies
- `calo-policy`, `compute`, `core`, `power-system`

## Dependents
- `calo-policy`, `core`, `desktop`, `experiments`, `tests`

## Associated tests
- `tests/conftest.py`
- `tests/gui/test_guided_workflow.py`
- `tests/integration/test_workspace_execution_control.py`
- `tests/regression/test_seed_reproducibility.py`
- `tests/unit/test_algorithms.py`
- `tests/unit/test_calo_v4.py`
- `tests/unit/test_cma_es.py`
- `tests/unit/test_convergence_semantics.py`
- `tests/unit/test_lshade.py`
- `tests/unit/test_phase6_command_and_native_contracts.py`
- `tests/unit/test_tsh_calo_optimizer.py`
- `tests/unit/test_v120_phase2_contracts.py`
- `tests/unit/test_v2_benchmarking.py`
- `tests/unit/test_v32_portfolio_resume.py`
- `tests/unit/test_v343_export_completion.py`

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
