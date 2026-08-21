# Current requirement traceability

**Updated:** 2026-08-21  
**Scope:** active v12 development checkpoint only

Superseded row-by-row history is available from Git checkpoint `ba597eb`. The table below retains only current requirements and the next proof boundary.

| Requirement | Current implementation authority | Current evidence status | Remaining proof |
|---|---|---|---|
| Development identity is explicit and non-release | `version.py`, `pyproject.toml`, `ACTIVE_DEVELOPMENT_STATUS.json`, `verify_active_version.py` | Implemented; cleanup changes pending validation | Fresh current-source active-version and complete Phase 6 validation |
| CUDA-preferred/CPU-only execution and Safe-80 admission | `compute/`, `accelerated/`, execution contracts | Implemented; prior source-bound engineering evidence only | Final-candidate physical and container repetition |
| Intel XPU is non-executable | compute-mode schemas and validators | Implemented | Retain in current-source regression and packaging checks |
| Exact FE accounting and deterministic plans | experiment, optimizer, checkpoint, and result contracts | Implemented; current-source validation pending | Complete validator plus separately authorized scientific evidence |
| Individual experiments remain independent of Workspace Portfolio/Study | `execution_plans.py`, `ExperimentManagerPanel`, execution controller, persistence schemas | Implemented | Fresh owner Phase 6 unit/GUI/integration validation |
| Workspace uses Portfolio goal to Study setup to immutable cells | portfolio/study planners, Workspace plan, controller, database | Implemented | Fresh owner Phase 6 validation; no automatic scientific claim |
| Fairness audit precedes Stage and Run | audit receipt, stage identity, controller ownership | Implemented fail closed | Current-source GUI/integration replay |
| Policy lifecycle is explicit | policy registry, training, qualification, activation, immutable binding | Implemented boundaries; no current quality claim | Candidate-bound formal qualification and separate activation |
| Resume/extension is compatibility-bound | parameter layout, training schema, state, RNG, optimizer, accounting contracts | Implemented | Fresh synthetic/current-source engineering validation, then candidate-specific evidence |
| Protected cases remain isolated | qualification/training guards and explicit authorization boundaries | Implemented fail closed | Separately authorized protected-case gate only |
| Repository intelligence protects architectural routing | `.ai/`, `scripts/ai-index`, protected `AGENTS.md` blocks | Current before cleanup | `python scripts/ai-index update`, guard check, then index check after cleanup |
| Release claims require direct final-candidate evidence | release scripts, container contracts, this gate ledger | Not complete; release not authorized | Close engineering, human, policy/scientific, physical/container, publication, and explicit authorization gates |

## Claim boundary

The retained `phase6-20260817-235629` bundle proves only its exact older source and automated engineering scope. It does not validate later source, this cleanup, a policy, a scientific conclusion, protected cases, a final container candidate, human acceptance, or release readiness.
