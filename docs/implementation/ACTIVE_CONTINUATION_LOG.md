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

---

Historical maintenance handoff follows; its unexecuted status is superseded above.

# Current maintenance checkpoint ? 2026-09-19

**Branch:** `agent/engineering-closure-20260919`
**Baseline HEAD:** `00620683c989429e6cfdcd2dc53e0bb5837531a8`
**Product:** `12.0.0.dev1`; development only
**Coding:** corrections for five identified engineering blockers are implemented locally.
**Validation:** NOT EXECUTED for these changes; current-source acceptance remains pending.

See `ENGINEERING_CLOSURE_20260919.md` for the exact scope and manual command. No real
policies, experiments, historical evidence, scientific semantics, or release flags were
modified. No commit, push, merge, publication, or release was performed.

The ignored `validation/Validate-CALO-Closure.ps1` creates unique evidence directories,
records source and harness hashes before/after, executes selected contract suites and native
packaging checks, and retains failures without inferring scientific or human acceptance.
Full Linux CI, physical CUDA/container qualification, and scientific release gates remain open.

The older checkpoint below is retained as historical provenance. Its branch, next-action
instructions, and passing bundle do not describe or validate this maintenance worktree.

---

# Active continuation checkpoint

**Updated:** 2026-08-21
**Checkout:** `C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio`
**Branch:** `agent/ai-repository-intelligence`
**Pre-cleanup Git checkpoint:** `ba597eb`
**Product:** `12.0.0.dev1` / `12.0.0-dev.1`
**Stage:** development only

This file contains only the current continuation boundary. The superseded append-only history is recoverable from Git checkpoint `ba597eb` and earlier commits.

## Current source state

- Current source implements the Phase 6 scientist-facing shell, explicit policy lifecycle, guarded TSH-CALO training/qualification surfaces, Individual-versus-Workspace execution ownership, Portfolio-to-Study flow, algorithm staging, safe pause/resume, and exact evaluation-accounting contracts.
- Later production-source changes exist after the newest retained full Phase 6 validation source `4560b2fba6ecc5c3271da7dfd680a0985ca501f3`.
- Therefore current-source Phase 6 engineering validation is **pending**. Do not describe the current tree as validated from the retained older bundle.
- Repository cleanup changes documentation, status routing, active-version verification, and removes obsolete historical integrity tests. These changes also require the next current-source validation run.

## Last retained engineering checkpoint

- Bundle: `validation/logs/phase6-20260817-235629`
- Result at its exact source: PASS, 17/17 commands.
- Source commit: `4560b2fba6ecc5c3271da7dfd680a0985ca501f3`.
- Scope: Python/dependency identity, diff/ignore contracts, active version, compile, Ruff, format, unit/GUI/integration checks, offscreen rendering, package build, distribution verification, and source stability.
- Excluded: policy training/evaluation/qualification/activation/deletion, protected cases, Docker/CUDA campaigns, publication, release, and human acceptance.

## Protected state retained during cleanup

- Current SQLite database and schema-migration backups.
- Runtime-owned SQLite `-shm`/`-wal` sidecars; ignored but not deleted.
- Checked-in trained-model records and local TSH-CALO training/qualification artifacts, because policy deletion requires exact reference and lifecycle review.
- The last runtime-required historical freeze `calo_v690_freeze.json` and its referenced historical training snapshot.
- Repository intelligence under `.ai/` and all protected `AGENTS.md` policy blocks.

## Next required action

After recreating `.venv`, the project owner may explicitly run:

```powershell
& .\validation\Validate-Phase6.ps1 -PythonExecutable ".\.venv\Scripts\python.exe"
```

Return the complete new `validation/logs/phase6-*` directory for read-only review. The run must not be interpreted as scientific qualification, policy authorization, protected-case evidence, release approval, or human acceptance.

### Windows snapshot storage follow-up

The broad local audit exposed a reproducible Windows Git loose-object path failure when reopening
content-addressed source snapshots. Snapshot Git commands now request long-object-path support
locally, and only newly created snapshot repositories retain the corresponding local option.
The live repository and global Git configuration are not modified. Overlong repository-root paths
are rejected before writing a snapshot, with an explicit request to choose a shorter location.
The snapshot suite passed all eight tests in targeted-20260919-125025-919911, including repeatable
content identity, nested storage, original-source preservation, and pre-write path rejection.
The complete final local results are recorded in the Git-ignored
`validation/session-reconnect-20260919/FINAL_TESTING_REPORT.json`; no result is inferred merely
from that path or this source note. Current-source release acceptance still requires the open gates.

## Experiment lifecycle and wiring continuation (2026-09-19)

Read `LIFECYCLE_WIRING_ACCEPTANCE.md` and `OMISSION_DISPOSITIONS.json` for current scope,
production corrections, mapped tests and omission rules. Desktop execution is explicitly
authorized in the current owner conversation. Real CPU/Workspace/GPU experiments and native
wiring checks have iterative passing evidence; final current-source regression remains distinct.
Do not reuse earlier closure counts as current-source acceptance or silently suppress omissions.
No source commit/push/merge/release has been performed by this continuation.


## Atomic GUI settings and reproducible validator continuation

The resumed desktop review reproduced rejected-edit mutation in Experiment configuration,
ORPD formulation and robust scenario settings. All three now validate independent drafts before
publishing shared configuration. Workspace Apply constructs the complete draft locally, persists
it first, then emits exactly one shared-config notification. Rejected persistence emits none.
The original red regressions and corrected runs are retained under `validation/logs`.
A 29-test GUI/Workspace rerun passed after these corrections, before subsequent evidence-tool edits.

The constructed window's 162 QPushButtons were inspected dynamically: no missing callbacks and
no duplicate clicked subscriptions were observed. Existing runtime tests cover repeated binding,
ribbon admission, mode routing, safe close and real manager execution. This inventory does not
prove every data-dependent future control or human usability. An attempted additional full-window
end-to-end test-file creation was blocked by the tool and was not added; do not claim that test ran.

The local closure runner mistakenly hashed generated console logs as executable validator inputs.
Its stability identity now covers actual top-level validator scripts; repository source remains
independently fully hashed. Regression tests prove log growth is irrelevant while script edits,
additions and removals invalidate identity. Lane admission also rejects missing/unstable harness
identity and missing omission inventories. Twenty evidence-tool regressions passed.

Final source-bound CPU, native Windows, physical CUDA, Linux/WSL and package-closure evidence must
be consolidated by `validation/summarize_lifecycle_acceptance.py` into
`validation/FINAL_ACCEPTANCE.json`. That external record must match current source snapshots;
this document itself never confers a PASS. Historical, delegated and separately authorized
omissions remain distinct, with no automatic release, publication or scientific qualification.
