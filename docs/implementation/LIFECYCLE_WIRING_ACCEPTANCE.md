# Experiment lifecycle, hardware and GUI-wiring acceptance

Updated: 2026-09-19. Active identity: 12.0.0.dev1. Development only.

## Scope and authority

Continuation of the existing engineering-closure maintenance goal on
`agent/engineering-closure-20260919`, based on `00620683`.
The owner authorized desktop testing and fixes. No push, merge, publication, real policy
training/qualification/activation/deletion, or protected-case execution is authorized by this record.
All numerical acceptance examples use disposable databases and bounded case30/case57 inputs.
No numbered development phase is opened or closed here.

## Corrected production boundaries

- The manager and worker own independent deep copies of admitted configuration. The public
  active-configuration accessor cannot mutate a running experiment.
- Frozen plan deserialization retains the exact algorithm parameter map instead of adding
  unselected defaults. Campaign persistence retains admitted scientific intent separately from
  runtime device annotations, which remain in numerical provenance.
- The numerical worker verifies and durably binds its campaign to its frozen plan before any
  optimizer job starts. A failed GUI callback can no longer silently bypass this requirement.
- Safe application closing retries on manager idle, after the worker actually terminates.
- Workflow connection setup is idempotent. Sidebar/ribbon admission and actual navigation share
  prerequisite, owner and training-lock checks.
- Completed-with-failures and unverifiable campaign outcomes produce failed/stopped presentation,
  not an unconditional green completion or completed scientist workflow.
- Windows snapshot creation uses short same-volume scratch space and checks full final Git object
  paths before writing; unsupported destinations produce a precise shorter-path request.
- The cross-episode aggregation regression now synchronizes real queue admission before measuring
  the real broker, without changing its production timing or relaxing its exact 80-candidate assertion.

## Executable acceptance sources

`tests/integration/test_real_experiment_lifecycle.py` exercises the actual audit, immutable staging,
Qt worker, one-worker and process-pool execution, bounded numerical evaluation, persistence,
independent result validation, export rejection/acceptance, exact verified reuse, safe pause and
new-AppState restart/resume, terminal cancellation, storage failure, tampered admission, multi-case
Workspace orchestration, and explicit physical CUDA execution.

`tests/gui/test_gui_wiring_acceptance.py` exercises repeated connection setup/navigation, exact
callback counts, both setup modes, unique action IDs/shortcuts, shared availability gates,
policy-free comparator admission, truthful failure presentation, and idle-bound deferred close.
The native lane shows a real Windows window and retains a screenshot. Automated tests do not
constitute human usability/accessibility acceptance.

`tests/unit/test_physical_cuda_acceptance.py` requires physical CUDA in its explicit lane, compares
case30/case57 FP64 evaluations with CPU, and records visible devices and bounded allocation evidence.
A CPU lane never establishes CUDA qualification; the formerly skipped hierarchical CPU/CUDA
numerical agreement test is included in the explicit CUDA lane.

`tests/unit/test_execution_snapshot_contract.py` verifies independent nested snapshots and duplicate
start rejection. The omission-plugin and test-inventory contract tests validate missing execution,
xfail/xpass/failure handling, collection skips, exact historical identities, successor requirements,
and named plus parameterized protected-case exclusions.

## Every omission remains visible

`docs/implementation/OMISSION_DISPOSITIONS.json` catalogs 67 historical test cases and all 310
assertions extracted from them. It records exact node IDs, normalized UTF-8/LF source hashes,
reasons and current successor tests. Catalog presence is not a successor PASS.

`scripts/qa/omission_plugin.py` records selected nodes, all per-phase outcomes, exact skip messages,
deselections, collection errors/skips, and selected-but-unexecuted tests. It writes a partial inventory
before execution and updates it so process interruption cannot erase the missing-work inventory.
`calo_rpd_studio/validation/test_inventory.py` never treats a current skip, missing successor,
missing platform, hidden GPU, failed outcome or incomplete run as acceptance evidence.

The six named/parameterized case118/case300 scientific checks remain separately authorized.
Ordinary unit rejection tests mentioning those names are not automatically deselected.

## Platform harnesses

- `.github/workflows/experiment-acceptance.yml`: Windows Python 3.11/3.13 and Linux Python
  3.11/3.12/3.13. Actions are pinned; the primary Linux lane remains hash-locked. Compatibility
  environments are recorded. This definition has not itself been executed by GitHub.
- `scripts/qa/isolated_test_entry.py`: disposable preferences and user directories for CI execution.
- `scripts/qa/gui_wiring_audit.py`: source-hashed static inventory of connections and potential
  duplicates/loop captures. Static inspection is not dynamic wiring acceptance.
- `scripts/qa/validate_platform_filesystem.py`: actual Linux file/directory symlink rejection,
  empty/sorted/hash-correct manifests and atomic replacement. No optional filesystem skips.
- Ignored `validation/run_lifecycle_checks.py`: separate CPU, CUDA and native lanes with source
  and harness hashes, XML/JSON outcomes, optional subprocess/multiprocessing coverage and hardware
  artifact retention. It never marks the product release-ready.

## Evidence status

Iterative source-stable runs established 36 passing lifecycle/wiring/control/snapshot tests,
10 passing Workspace/omission/safety tests, 11 physical-CUDA tests including a complete experiment,
8 native-window wiring tests, six actual Linux filesystem checks, and independently verified
export plus exact reuse. These results belong to their specific retained source snapshots.
Later instrumentation, test and documentation edits require fresh unchanged-source revalidation.
The authoritative current results are timestamped under ignored `validation/logs`, not inferred
from this document. Earlier failed/stopped attempts are retained, never overwritten as passes.

Full current-source regression, installed distribution checks, the remaining compatibility matrix,
container runtime evidence and physical soak are separate gates. Protected-case and real candidate
scientific qualification, human acceptance and explicit publication approval remain separate.

## Source-inspection and measurement corrections

The real Linux lane exposed repeated Git-status timeouts on a Windows-mounted checkout.
Source identity now returns explicitly dirty/non-durable as soon as a tracked difference is
proved, still inspects all nonignored untracked files before any clean claim, rechecks HEAD
before returning clean, and reports bounded inspection timeouts without accepting a build
declaration fallback. The original Windows and Linux failures remain retained.

Test-only instrumentation additionally traces Qt-created worker threads and Python process
pools. Coverage percentages describe the selected run only; coverage timings are not performance
evidence. The omission collector retains partial inventories and retries transient Windows
report-sharing denials boundedly, while persistent write failures still fail closed.


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
