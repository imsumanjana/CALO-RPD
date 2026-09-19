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

## Engineering closure maintenance ? 2026-09-19

### CLOSURE-20260919-01: Active source verifier compares obsolete progress strings

Implementation: `calo_rpd_studio/scripts/verify_active_version.py`, `ACTIVE_DEVELOPMENT_STATUS.json`.

Regression source: `tests/unit/test_active_development_contract.py`.

Evidence: implementation written; execution and acceptance pending. No scientific or release authority granted.

### CLOSURE-20260919-02: AI index byte metadata differs across Git-equivalent checkout line endings

Implementation: `scripts/ai-index`.

Regression source: `tests/tooling/test_ai_repo_intelligence_v2.py`.

Evidence: implementation written; execution and acceptance pending. No scientific or release authority granted.

### CLOSURE-20260919-03: Legacy policy-independence test assumes a retired combined GUI

Implementation: `tests/unit/test_v680_policy_independence.py`.

Regression source: `tests/unit/test_v680_policy_independence.py`.

Evidence: implementation written; execution and acceptance pending. No scientific or release authority granted.

### CLOSURE-20260919-04: Linux Qt CI jobs omit runtime shared libraries

Implementation: `.github/workflows/ci.yml`.

Regression source: `tests/unit/test_ci_qt_runtime_contract.py`.

Evidence: implementation written; execution and acceptance pending. No scientific or release authority granted.

### CLOSURE-20260919-05: Native distribution contract requires deleted agent handoff documents

Implementation: `calo_rpd_studio/scripts/verify_phase6_distribution.py`.

Regression source: `tests/unit/test_phase6_distribution_current_source.py`.

Evidence: implementation written; execution and acceptance pending. No scientific or release authority granted.

## Desktop testing continuation - 2026-09-19

The owner explicitly authorized Desktop Commander testing and repairs, including GUI wiring.
This section supersedes the earlier coding-only execution boundary for this maintenance pass;
it does not authorize Git publication, a release, real saved-policy operations, or protected cases.

Corrections implemented during the authorized testing continuation:
- Reconciled checkpoint serialization with exact extension parameter-schema validation; optional
  legacy guard omission remains byte/identity compatible, and nonempty guard identity is retained.
- Corrected cumulative PPO update-boundary verification across repeated finite extension segments.
  Per-episode authenticated receipt counts are summed; thresholds and policy promotion gates remain.
- Made progress follow the durable executor-owned counter, not a stale derived mirror; completed
  status now rejects incomplete training work instead of displaying a fabricated 100 percent.
- Gave missing/corrupt model files priority in policy-library diagnostics while retaining separate
  compatibility checks, including after a successful file-checksum cache hit.
- Preserved secondary incident/status write errors on qualification infrastructure exceptions,
  without masking the original failure or granting a scientific receipt.
- Updated packaged-GUI evidence to apply the actual application theme, retain its hash, and restore
  shared test application state. It still forbids importing from the source checkout.
- Replaced obsolete source-string/layout fixtures with current staged-plan, explicit readiness,
  independent-training, GUI tab, history, conservative influence, and device-probe contracts.

Evidence retained from intermediate source revisions (not final-release qualification):
- closure-20260919-121332-874-3277e72b: 25/25 stages passed with stable source and harness;
  subsequent repairs supersede that bundle for exact current-source claims.
- targeted-20260919-121422-383109: 64 lifecycle/accounting tests passed.
- targeted-20260919-122143-098683: all 106 GUI tests passed; the combined run had a separate
  unit-fixture failure, subsequently corrected. It is not labelled an entirely passing bundle.
- targeted-20260919-122419-813557: six workflow-restore and themed-package tests passed.
- native-windows-20260919-123229-328919: native Windows renderer passed with stable source;
  human usability/accessibility/scientist acceptance is not inferred.
- targeted-20260919-123149-833159: 142/143 repair tests passed; the remaining new enum-reference
  typo was corrected and its rerun passed in targeted-20260919-123452-116360.

Fresh broad regression, coverage, and final closure results are to be retained under ignored
validation/logs directories, with source hashes and complete skip/failure inventories. The broader
suite has historical-release and unavailable-platform skips; those are not successful executions.
Positive case118/case300 scientific checks remain excluded from this continuation. A proposed
explicit opt-in collection guard was blocked by the tool and was not applied; default CI scientific
scope therefore still needs a separately reviewed correction before unrestricted execution.

The branch remains uncommitted and unpublished. Release flags remain false. Hosted Linux/Windows
CI, Docker/physical candidate evidence, full scientific qualification, and release authorization
are not established by these local engineering tests. Do not carry an older PASS across changes.

## Lifecycle/platform/window-wide wiring continuation

| Requirement | Production authority / executable evidence | Current disposition |
|---|---|---|
| Immutable admitted configuration | ExperimentManager/Worker snapshots; test_execution_snapshot_contract | Implemented; final source rerun required |
| Exact campaign binding before computation | execution_control.plan_configuration; Worker.run/_prepare_campaign; real tampering/resume tests | Implemented; iterative actual numerical runs passed |
| Complete Individual lifecycle | test_real_experiment_lifecycle | Actual CPU, process pool, persistence, validation, export, reuse and recovery exercised |
| Complete Workspace cells | real WorkspaceCampaignCoordinator and case30/case57 lifecycle test | Iterative pass; final rerun required |
| Window-wide GUI signals | shared navigation blocker, idle close, idempotent connections, durable failure display; test_gui_wiring_acceptance | Iterative offscreen and native pass |
| Physical CUDA | test_physical_cuda_acceptance plus actual GPU experiment and hierarchical parity | Iterative actual CUDA pass, not physical soak or container qualification |
| Platform parity | experiment-acceptance.yml, isolated_test_entry.py, Linux filesystem script | Matrix harness implemented; each unexecuted lane remains pending |
| Every omission accounted | OMISSION_DISPOSITIONS, omission_plugin, test_inventory | 67 historical nodes/310 assertions retained; live omissions never accepted automatically |
