# bootstrap

<!-- AUTO-GENERATED MODULE INTELLIGENCE -->
**Purpose:** Native setup, dependency repair and application launch without coupling ordinary launch to installation.

**Important state:** Environment/prerequisite and launch state; not scientific study state.

**Major flow:** bootstrap/launcher -> prerequisite checks -> calo_rpd_studio.app.application.

**Constraints/invariants:** Ordinary launch must not install packages or perform policy lifecycle work.

**Common failure points:** Missing/incompatible prerequisites, platform launch paths and setup/launch boundary regressions.

## Start here
- `calo_bootstrap/launcher.py`

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
- `desktop`
- `experiments`
- `persistence`
- `power-system`

## Dependents
- `calo-policy`
- `tests`

## Related tests
- `tests/unit/test_mathematical_reference.py`
- `tests/unit/test_prerequisites_and_resources.py`
- `tests/unit/test_v120_phase2_contracts.py`
- `tests/unit/test_v650_must_resolve.py`
- `tests/unit/test_v680_policy_independence.py`

## Files
- `Launch-CALO-RPD.ps1`
- `bootstrap.py`
- `calo_bootstrap/AGENTS.md`
- `calo_bootstrap/__init__.py`
- `calo_bootstrap/launcher.py`
- `calo_bootstrap/prerequisites.py`
- `calo_bootstrap/wizard.py`
- `calo_rpd_studio/algorithms/calo/run_checkpoint.py`
- `calo_rpd_studio/scripts/run_benchmark.py`
- `calo_rpd_studio/scripts/run_final_benchmark.py`
- `calo_rpd_studio/scripts/run_mathematical_reference.py`

> Generated routing is evidence, not auditing. Curated `.ai/modules/*.md` guidance remains authoritative when more specific.
