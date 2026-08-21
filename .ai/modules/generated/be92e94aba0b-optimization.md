# optimization

<!-- AUTO-GENERATED MODULE INTELLIGENCE -->
**Purpose:** Common optimizer contracts, implementations, registry and portfolio execution surfaces.

**Important state:** Optimizer population/archive/RNG state plus experiment-supplied configuration.

**Major flow:** experiment plan -> registry -> optimizer -> ORPD objective/constraints -> result.

**Constraints/invariants:** Deterministic seeded behavior, common feasibility semantics and exact evaluation budgets.

**Common failure points:** Budget drift, ranking/feasibility inconsistency, RNG nondeterminism and bound errors.

## Start here
- No public/entry surface identified automatically; use curated module guidance.

## Preferred edit targets
- None explicitly identified; consult `.ai/architectural-semantics.json`.

## Deep/internal implementation
- None identified.

## State owners
- None explicitly identified.

## Cross-module dependencies
- `calo-policy`
- `compute`
- `core`
- `power-system`

## Dependents
- `calo-policy`
- `core`
- `desktop`
- `experiments`
- `tests`

## Related tests
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
- `tests/unit/test_v34_release_blockers.py`
- `tests/unit/test_v3_accelerated_backend.py`
- `tests/unit/test_v541_release_integrity.py`
- `tests/unit/test_v560_release_integrity.py`
- `tests/unit/test_v570_release_integrity.py`
- `tests/unit/test_v57_audit_closure_integration.py`
- `tests/unit/test_v580_release_integrity.py`
- `tests/unit/test_v590_scientific_closure.py`
- `tests/unit/test_v5_continuation.py`
- `tests/unit/test_v600_release_integrity.py`
- `tests/unit/test_v660_remaining_audit.py`
- `tests/unit/test_workspace_execution_plans.py`

## Files
- `calo_rpd_studio/algorithms/AGENTS.md`
- `calo_rpd_studio/algorithms/__init__.py`
- `calo_rpd_studio/algorithms/_helpers.py`
- `calo_rpd_studio/algorithms/ant_colony_continuous.py`
- `calo_rpd_studio/algorithms/base_optimizer.py`
- `calo_rpd_studio/algorithms/bat.py`
- `calo_rpd_studio/algorithms/clpso.py`
- `calo_rpd_studio/algorithms/cma_es.py`
- `calo_rpd_studio/algorithms/crow_search.py`
- `calo_rpd_studio/algorithms/dragonfly.py`
- `calo_rpd_studio/algorithms/firefly.py`
- `calo_rpd_studio/algorithms/flower_pollination.py`
- `calo_rpd_studio/algorithms/grasshopper.py`
- `calo_rpd_studio/algorithms/grey_wolf.py`
- `calo_rpd_studio/algorithms/imperialist_competitive.py`
- `calo_rpd_studio/algorithms/legacy_mtlbo.py`
- `calo_rpd_studio/algorithms/lshade.py`
- `calo_rpd_studio/algorithms/moth_flame.py`
- `calo_rpd_studio/algorithms/mtla_de.py`
- `calo_rpd_studio/algorithms/multi_verse.py`
- `calo_rpd_studio/algorithms/pso.py`
- `calo_rpd_studio/algorithms/qode.py`
- `calo_rpd_studio/algorithms/registry.py`
- `calo_rpd_studio/algorithms/result.py`
- `calo_rpd_studio/algorithms/salp_swarm.py`
- `calo_rpd_studio/algorithms/simulated_annealing.py`
- `calo_rpd_studio/algorithms/tlbo.py`
- `calo_rpd_studio/algorithms/torch_suite.py`
- `calo_rpd_studio/algorithms/whale.py`
- `calo_rpd_studio/portfolio/AGENTS.md`
- `calo_rpd_studio/portfolio/__init__.py`
- `calo_rpd_studio/portfolio/catalog.py`
- `calo_rpd_studio/portfolio/exporter.py`
- `calo_rpd_studio/portfolio/fingerprint.py`
- `calo_rpd_studio/portfolio/models.py`
- `calo_rpd_studio/portfolio/planner.py`
- `calo_rpd_studio/portfolio/study_planning.py`

> Generated routing is evidence, not auditing. Curated `.ai/modules/*.md` guidance remains authoritative when more specific.
