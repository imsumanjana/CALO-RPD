# CALO-RPD v12.0 onward release update and fix plan

**Plan status:** Phase 1 and Phase 2 validation are accepted. Phase 3 GUI coding is implemented at
`12.0.0.dev1`, and corrected Windows-local run `phase3-20260807-052047` is accepted. The subsequent
remaining-gate run exposed responsive Results/Settings clipping and formatting defects that are now
corrected. Noninteractive corrected Windows and Linux xcb evidence remain pending. This is not a
release declaration,
scientific qualification, policy approval, or authority to open protected cases.

**Prepared:** 2026-08-06 (Asia/Calcutta)

**Target release line:** CALO-RPD v12.0 onward. Phase 1 establishes the v12.0 development identity;
Phase 4 may produce v12.0 release candidates; only Phase 5 may produce the final v12.0.0 release.
Until the Phase 1 identity change is implemented and verified, use `post-v6.9 development` rather
than claiming that the active tree is either the immutable v6.9.0 release or an already qualified
v12 release.

### v12 release-version policy

The v12 line begins in Phase 1 and uses these identities consistently across source, Python package
metadata, documentation, evidence, distributions, containers, and Git tags:

| Lifecycle stage | Python/PEP 440 version | Human-facing label | Permitted phase |
|---|---|---|---|
| Development | `12.0.0.devN` | `CALO-RPD v12.0.0-dev.N` | Phase 1 onward |
| Release candidate | `12.0.0rcN` | `CALO-RPD v12.0.0-rc.N` | Phase 4 after its prerequisites pass |
| Final release | `12.0.0` | `CALO-RPD v12.0.0` | Phase 5 only |
| Compatible fixes after release | `12.0.Z` | `CALO-RPD v12.0.Z` | A later approved maintenance plan |
| New compatible feature line | `12.Y.0` | `CALO-RPD v12.Y.0` | A later approved feature plan |
| Future breaking release | `13.0.0` or later | Matching major release label | A separate approved major plan |

Rules for the v12 transition:

- v12 is a new release and evidence identity, not a rename of v6.9.
- No v6.9 freeze, qualification, equivalence, hardware, GUI, container, performance, or scientific
  claim automatically qualifies v12.
- Historical v6.9 records remain immutable and explicitly historical.
- `N`, `Y`, and `Z` are monotonically increasing integers; a published version is never reused or
  overwritten for different source or artifacts.
- Every development/RC artifact records the full source commit and dirty/clean state in addition to
  the version.
- The application version does not silently redefine the TSH-CALO algorithm, state, action, policy,
  training, qualification, or evidence ABI. Those identities remain separately versioned and must
  be declared explicitly.
- A semantic, analysis, policy, container, or release-artifact change after an RC is frozen requires
  a new RC number and repetition of every affected gate.
- The final `12.0.0` version and `v12.0.0` tag are created only after Phase 5 passes and explicit
  release authorization is recorded.

## 1. Purpose

This plan converts the revised post-v6.9 source audit, the active continuation records, the current
implementation gates, and the requested GUI modernization into five ordered development phases.
It is the execution plan for correcting the active tree and producing a new release candidate; it
does not replace:

- [`IMPLEMENTATION_GATES.md`](IMPLEMENTATION_GATES.md), which remains the gate authority;
- [`REQUIREMENT_TRACEABILITY.md`](REQUIREMENT_TRACEABILITY.md), which records requirement evidence;
- [`ACTIVE_CONTINUATION_LOG.md`](ACTIVE_CONTINUATION_LOG.md), which records append-only execution
  history;
- [`RELEASE_READY_CONTINUATION_HANDOFF.md`](RELEASE_READY_CONTINUATION_HANDOFF.md), which defines the
  current handoff and claim boundary; or
- [`SCIENTIFIC_VALIDATION_PROTOCOL.md`](SCIENTIFIC_VALIDATION_PROTOCOL.md), which controls final
  scientific evidence.

The five phases must be completed in order. Work may be prepared in parallel only when it cannot
change an earlier phase's contract or invalidate its evidence. A later phase cannot be declared
complete while an earlier exit gate remains open.

### Mandatory phase execution protocol

The following operating sequence applies to Phase 3 onward and to any correction or maintenance
phase added to this plan:

1. Before announcing or starting phase development, create a phase-specific goal through the goal
   service. The goal must name the phase and its concrete coding deliverables. Do not reuse an
   unrelated or already completed goal as authority for the new phase.
2. Perform coding development only: implement production source, test source, schemas, migration
   code, documentation, and the phase validation harness. Writing test code is required where the
   change needs it; executing that code is reserved for the user.
3. Do not execute any test or validation task that can be performed manually by the user. This
   includes phase validators, pytest/tox/coverage, compilation, generated-schema checks, Ruff,
   mypy, package/build smoke checks, GUI/browser checks, Docker validation, benchmarks, campaigns,
   policy training/evaluation, qualification, and protected-case workflows. An exception requires
   a later explicit user instruction naming the command or check to execute.
4. At the end of the coding pass, create or update one detailed PowerShell validation script under
   the Git-ignored `validation/` directory. The validator must run all checks required for that
   phase, retain command outputs, record relevant source and validator SHA-256 identities, and write
   a newly timestamped detailed log directory. Neither the validator nor its logs may enter Git,
   manifests, packages, containers, or release artifacts.
5. Provide the exact manual command and expected log-directory pattern. The user executes the
   validator and returns the complete log directory. The agent then reviews only the returned
   evidence, applies focused coding corrections, updates the validator if needed, and requests a
   fresh manual rerun without running it.
6. Keep tool calls, file reads, and output focused on the current phase and returned failures. Avoid
   redundant test execution, repeated repository scans, or large evidence dumps so token usage stays
   low. No phase exit gate closes until the user's manual evidence has been reviewed and accepted.

## 2. Current audited baseline

The planning baseline observed on 2026-08-06 is:

- branch `main`, commit `6c3a4647bb7a70bf57616e714f8e747f8b26c0ab`;
- the source still declares version `6.9.0`, although it is a post-v6.9 development tree;
- the historical v6.9 freeze checks 150 files and currently reports 4 missing and 97 changed;
- the root manifest checks 484 paths and currently reports 5 missing and 168 changed;
- the active local suite passes `650 passed, 63 skipped` when the deliberately historical v6.9
  release-integrity file is excluded;
- the v6.9 release-integrity file correctly fails for the stale freeze and root manifest;
- compilation and generated experiment-schema checks pass;
- the retained v3 TSH-CALO screening is valid negative evidence, not qualification evidence;
- no fresh counted candidate has completed candidate-bound equivalence, accepted A-E ablation, or
  qualification;
- G10 protected scientific evidence and G11 final release production remain incomplete; and
- generated policy candidate files are present outside source control and must remain excluded from
  every distribution and container artifact.

These observations are development evidence only. Every claim must be recomputed against the exact
candidate that eventually enters Phase 5.

## 3. Non-negotiable boundaries

The following rules apply to every phase:

1. Preserve deterministic baseline behavior and exact function-evaluation accounting.
2. TSH-CALO A-E are the approved production-candidate scope. Change F remains independently
   feature-flagged, experimental, evidence-gated, and disabled by default.
3. A change to optimizer equations, selection pressure, qualification thresholds, effect
   definitions, reward, state/action spaces, or promotion semantics is a scientific or promotion
   semantics change. It requires explicit approval and a versioned evidence boundary before
   implementation.
4. Policy training remains independent. An experiment cannot train, modify, qualify, register, or
   activate a policy.
5. Experiments may consume only separately qualified, explicitly activated, immutable,
   checksum-valid, compatible policies. Otherwise they must use the deterministic safe baseline
   fallback.
6. Never auto-train, auto-qualify, auto-register, auto-activate, or silently reinterpret a policy.
7. Intel XPU is historical/view-only and must never be executable. Supported execution modes are
   CUDA-preferred and CPU-only, with formal CUDA-only behavior represented explicitly where needed.
8. Admission ceilings use at most 80% of currently free VRAM or currently available RAM.
9. Protected case118/case300 identities remain closed until the design, analysis, source, policy,
   containers, seeds, and statistical protocol are frozen.
10. Do not regenerate final freezes, root manifests, SBOMs, image digests, release metadata, or
    public release claims until Phase 5.
11. Do not fabricate or extrapolate hardware, container, performance, energy, thermal, GUI, or
    scientific evidence.
12. Preserve user files and unrelated changes. Do not push, merge, publish, tag, or release without
    explicit authorization.

## 4. Evidence vocabulary

Every plan item and traceability row must use one of these evidence classes:

- **Implemented:** source and focused tests exist.
- **Locally verified:** the exact local source passed the required deterministic tests.
- **Physically verified:** the exact source/artifact passed on the named physical device or target
  runtime with retained provenance.
- **Scientifically qualified:** the frozen campaign passed its preregistered criteria and retained
  independently validated raw evidence.
- **Release verified:** final distributions, images, metadata, manifests, CI, and clean-machine
  reproduction all bind to one immutable candidate.

A harness is not physical or scientific evidence. Development evidence from an earlier commit is
not final-candidate evidence.

---

## Phase 1 — Development identity, statistical correctness, and governance repair

### Goal

Establish the truthful v12.0 development identity and correct every analysis or qualification rule
that can change a scientific decision before any fresh policy candidate or formal campaign is
executed.

### Required work

#### 1.1 Establish the v12.0 development identity

- [x] Start the new release line at Python package version `12.0.0.dev1` and human-facing label
  `CALO-RPD v12.0.0-dev.1`.
- [x] Increment the development serial monotonically for subsequently retained development
  identities; never reuse one serial for different immutable evidence.
- [x] Update runtime/package version sources, README development wording, and active development
  metadata consistently to the same v12.0 development identity.
- [x] Add a version-consistency verifier covering `pyproject.toml`, runtime version constants,
  package metadata, README, active status, CLI `--version`, GUI About, evidence headers, and
  container labels.
- [x] Reserve `12.0.0rcN` for candidates that reach the Phase 4 release-candidate boundary; normal
  Phase 1-3 development must not identify itself as an RC.
- [x] Reserve final `12.0.0` for Phase 5 after every final release acceptance criterion passes.
- [x] Preserve all v6.9 freezes, reports, and manifests as immutable historical artifacts.
- [x] Add an active-development status record that explicitly says the final freeze and release
  qualification are open.
- [x] Reconcile G0 ownership: record that `Docker_Build.txt` is now tracked, inventory current
  generated policy artifacts, and preserve the append-only correction history.
- [x] Bind every new evidence record to a full Git commit and clean/dirty source identity.

#### 1.2 Correct paired statistical integrity

- [x] Implement one shared matched-pairs rank-biserial function from signed rank sums.
- [x] Declare orientation once: positive must consistently mean candidate improvement.
- [x] Define zero and tie handling consistently with the selected Wilcoxon zero method.
- [x] Reject unequal paired arrays in formal analysis instead of truncating them.
- [x] Require exact preregistered pair IDs; reject duplicates, missing IDs, extra IDs, and ambiguous
  reordering.
- [x] Allow incomplete exploratory analysis only under an explicit `incomplete_pairs` status that
  cannot qualify or promote a policy.
- [x] Centralize objective-improvement calculations and version their definition.
- [x] Replace the unit-floor hybrid relative metric with an approved objective-family scale or a
  separately approved symmetric definition.
- [x] Record the exact statistical test, alternative, zero method, library/version, and fallback
  reason.
- [x] Narrow broad exception handlers in formal statistical paths. A preregistered formal test that
  cannot execute should fail qualification rather than silently change test families.

#### 1.3 Correct convergence and selection metrics

- [x] Replace run-dependent pre-feasibility convergence AUC with either two separate metrics
  (time-to-first-feasible and post-feasibility AUC) or one independently preregistered common scale.
- [x] Version the metric and invalidate incompatible historical comparisons.
- [x] Decide whether diversity-distance normalization by dimension is scientifically required.
- [x] Treat any diversity-pressure change as Class B: obtain explicit approval, version the
  algorithm identity, and add ablation/falsification evidence before enabling it.

#### 1.4 Re-evaluate affected evidence

- [x] Recompute stored qualification statistics under the corrected version where raw pairs are
  complete and immutable.
- [x] Mark evidence that lacks exact pair IDs or required raw observations as legacy/unverifiable.
- [x] Preserve the historical v3 screening record unchanged; add a versioned correction record
  rather than rewriting its files.
- [x] Confirm that the corrected case57 matched-pairs rank-biserial value does not reverse the
  retained negative screening decision.
- [x] Revoke or require requalification for any promotion decision that depended on the old effect,
  improvement, or AUC definitions.

#### 1.5 Correct ledgers and CI governance

- [x] Reopen the traceability row that currently calls paired statistics locally verified.
- [x] Add a mandatory active-development identity/status test to CI.
- [x] Keep historical release tests scoped to their immutable releases.
- [x] Prohibit active metadata from saying `VERIFIED` unless the same candidate's gate output
  generated that field.
- [x] Make freeze diagnostics report missing, changed, invalid, and extra categories together in
  both human-readable and machine-readable output.
- [x] Classify all broad exception handlers as boundary, recovery, cleanup, or defect masking;
  narrow scientific/statistical handlers first.

### Mandatory tests

- [ ] Rank-biserial counterexample: differences `[-1, -2, +3]` produce `0.0`, not sign imbalance.
- [ ] Zero/tie cases match the declared signed-rank convention.
- [ ] Unequal arrays fail before statistical execution.
- [ ] Duplicate, missing, extra, and unkeyed/ambiguous reordering fail formal analysis; explicitly
  keyed records may arrive in a different input order and must align by identity rather than row.
- [ ] Both qualification engines produce the same versioned effect and improvement definitions.
- [ ] Formal statistical fallback records its method or fails closed.
- [ ] Convergence metrics use one common, independently declared scale.
- [ ] Historical corrected records are additive and do not mutate frozen files.
- [ ] Development identity, package identity, source identity, and active status agree.
- [ ] A v12 version-consistency test fails on any mixed v6.9/v12 active identity and permits v6.9
  only inside explicitly historical records.
- [ ] RC and final version labels are rejected before their permitted phase gates.

### Phase 1 exit gate

Phase 1 is complete only when the corrected statistical contract, v12.0 development identity, CI
identity gate, traceability rows, and retained correction records agree; active source/package/GUI/
CLI/container labels use one `12.0.0.devN` identity; v6.9 is historical-only; affected historical
promotion decisions are requalified or explicitly invalidated; and no new candidate has been
trained under an analysis definition known to be wrong.

**Phase 1 accepted handoff (2026-08-06):** All required production/source changes and mandatory test
implementations are present. The user-executed `phase1-20260806-230256` evidence passed 16/16
commands with complete source/validator hashes and states that no policy training, policy
evaluation, campaign, benchmark, or protected-case workflow executed. Phase 1 is accepted.
Diversity-pressure normalization was explicitly left unchanged; any future
change remains Class B and requires separate approval and evidence. The historical case57 effect
cannot be recomputed from complete immutable raw pair values in tracked source, so the additive
correction record marks that effect legacy/unverifiable; its negative decision remains unchanged
because the retained interval crossed zero and Holm-adjusted `p=0.052734375` failed the mandatory
threshold.

---

## Phase 2 — Runtime binding, fallback, exact accounting, and provenance hardening

### Goal

Make every entry point execute on a concretely resolved device, apply one explicit fallback policy,
preserve exact computation accounting, and retain enough state to explain every failure.

### Required work

#### 2.1 Define execution modes truthfully

- [x] Define **formal CUDA-only** mode: CUDA is required, CPU fallback is forbidden, and capacity
  exhaustion fails the run with retained partial state.
- [x] Define **exploratory CUDA-preferred** mode: a full-request CPU restart may be allowed only when
  explicitly configured and visibly recorded.
- [x] Keep **CPU-only** mode concrete and force `runtime_compute_device="cpu"`.
- [x] Apply the same fallback contract to `evaluate_population`,
  `evaluate_population_with_context`, tensor APIs, training, GUI, CLI, and final campaigns.
- [x] Make the over-dense-case fallback respect the declared mode rather than bypassing it.
- [x] Exclude any fallback run from CUDA-only timing, energy, parity, utilization, or equivalence
  claims.
- [x] Align comments, schemas, GUI terms, reports, and active metadata with the implemented modes.

#### 2.2 Resolve devices before every run

- [x] Introduce one mandatory pre-run device-resolution service for GUI, ordinary CLI, parallel
  CLI, benchmark CLI, and final-campaign tasks.
- [x] Resolve CUDA-preferred to a concrete `cuda:N` or record an explicit CPU degradation reason.
- [x] Reject unresolved formal-campaign combinations.
- [x] Add CLI compute-mode/device options without exposing allocator internals to ordinary users.
- [x] Persist requested mode, assigned physical device, runtime device, fallback policy, and actual
  computation device separately.

#### 2.3 Harden memory and device ownership

- [x] Use stable physical GPU UUID as the lease identity, with normalized PCI bus ID as a controlled
  fallback; retain the logical CUDA index separately.
- [x] Include host/container scope so independent GPUs do not collide and one physical GPU cannot be
  double-booked under reordered visibility.
- [x] Route parallel CUDA jobs through a device-aware queue; ordinary lease contention must wait or
  cancel cleanly rather than become an optimizer failure.
- [x] Preserve the current Safe-80 admission ceiling based on free/available memory.
- [x] Choose and document one truthful 0.80 contract. If experiment-level configuration remains
  fixed at 0.80, do not advertise a configurable 0.10-0.95 experiment setting.
- [x] Separate request statistics from governor-lifetime statistics.
- [x] Document or safely isolate the process-lifetime conservative allocator fraction.

#### 2.4 Enforce exact evaluation and failure state

- [x] Require one evaluation for every submitted batch candidate before registering any FE.
- [x] Reject short, long, empty-mismatched, or identity-reordered batch results with a typed invariant
  error.
- [x] Preserve evaluation count, last incumbent, feasibility/violation state, last numerical state,
  runtime device/fallback state, and checkpoint reference in every partial failure envelope.
- [x] Distinguish failure before the first FE, failure after partial work, cancellation, capacity
  exhaustion, lease contention, and invariant failure.
- [x] Persist failure envelopes atomically and preserve exact-resume boundaries.

#### 2.5 Harden topology and status compatibility

- [x] Choose one authority for FP64 capability and retain its provenance.
- [x] Make CPU-only PyTorch, unavailable drivers, stale snapshots, and synthetic snapshots fail
  closed rather than crash discovery.
- [x] Keep historical XPU records readable only through explicitly legacy/view-only schemas.
- [x] Remove active XPU capability/equivalence language from current status records.
- [x] Generate current status from executed gates rather than trusted static strings.

### Mandatory tests

Phase 2 evidence `phase2-20260807-003828` is accepted: 15/15 commands passed with 23/23 dedicated
contracts, 44/44 affected regressions, exact source/validator hashing, and no prohibited workflow.

- [x] Inject `CudaCapacityExhausted`: formal mode fails; exploratory mode performs one explicit full
  CPU restart; provenance and claim eligibility differ.
- [x] Cases above the dense Torch limit obey the same fallback flag across every public API.
- [x] GUI, benchmark CLI, parallel CLI, and final campaign resolve identical device semantics.
- [x] Reordered/filtered `CUDA_VISIBLE_DEVICES` maps one physical UUID to one lease.
- [x] Parallel CUDA jobs queue per physical device; lease contention is not a scientific failure.
- [x] Short, long, empty, and reordered batch outputs fail before FE registration.
- [x] A run failing after `N` evaluations stores exactly `N` and its last numerical state.
- [x] CPU-only PyTorch and stale CUDA snapshots do not crash topology scanning.
- [x] Request and lifetime VRAM telemetry remain unambiguous across consecutive requests.
- [x] Historical XPU records remain viewable while all executable XPU plans fail.

### Phase 2 exit gate

Phase 2 is complete only when every entry point binds the same concrete runtime contract, formal
CUDA-only runs cannot fall back, exploratory fallback is explicit, exact FE/cardinality invariants
are central, physical device leases and queues are correct, partial failures are retained, and
current status records describe CUDA/CPU-only reality.

**Accepted gate evidence:** `phase2-20260807-003828` passed 15/15 commands, including generated
schema, Ruff diagnostics/format, 23/23 Phase 2 contracts, and 44/44 affected regressions. All 20
retained evidence hashes and all 35 current source hashes matched; validator identity matched and no
prohibited workflow ran. Phase 2 is accepted without creating a release-readiness claim.

---

## Phase 3 — Modern, organized scientist GUI

### Goal

Replace the dense, flat, form-heavy interface with a modern scientific workspace that emphasizes
the next decision, progressive disclosure, compact structured inputs, readable evidence, and
accessible navigation without changing scientific semantics.

### 3.1 Information architecture

Preserve stable internal workspace keys and historical migration behavior, but present them in five
collapsible groups:

1. **Home** — Overview, Resume Center.
2. **Model** — CALO Intelligence, Power System, ORPD Formulation, Algorithms, Robust Scenarios.
3. **Study** — Portfolio, Experiment Manager, Live Optimization.
4. **Evidence** — Results, Statistical Analysis, Validation, Benchmark, Publication.
5. **System** — Application Settings.

Required shell changes:

- [x] Replace the flat sixteen-item list with grouped, collapsible navigation.
- [x] Add a compact/expanded navigation rail with persisted width and group state.
- [x] Add a consistent SVG icon system; icons supplement rather than replace text.
- [x] Add workspace search or a command palette.
- [x] Show progress/status badges without exposing backend or allocator internals.
- [x] Hide irrelevant locked child pages until their parent stage becomes actionable; preserve an
  accessible explanation for every blocked action.
- [x] Keep stable workspace keys authoritative so existing saved sessions restore correctly.

### 3.2 Dashboard redesign

The Dashboard must answer these questions without opening a long form:

1. Is the system ready?
2. Which case and protocol are active?
3. Which qualified policy governs the study?
4. What is the next legal scientific action?
5. What recently completed, failed, or needs review?

Required Dashboard content:

- [x] One prominent **Next required action** card.
- [x] Compact readiness cards for data, policy, compute, validation, and storage.
- [x] Active study/case/protocol summary.
- [x] Recent experiments, resumable work, failures, and evidence status.
- [x] Compact training/run activity with a detailed drawer.
- [x] No full protocol-configuration form on the initial Dashboard.
- [x] Move study construction into a dedicated step-by-step Study Setup workflow.

### 3.3 Compact input and content rules

The GUI must not use unnecessarily long input areas. Apply these rules to every workspace:

- [x] Use structured controls instead of free-form text whenever the data has a known type:
  combo boxes for enumerations, spin boxes for numbers, date/time controls, file pickers, chips for
  small lists, searchable selectors for cases/algorithms, and editable tables for repeated rows.
- [x] Cap ordinary form content width; do not stretch a short value field across the full window.
- [x] Target ordinary control widths of roughly 240-480 logical pixels and a readable form column of
  roughly 720-900 logical pixels, adapting at smaller screens.
- [x] Use short inline fields for seeds, identifiers, thresholds, paths, and scalar parameters.
- [x] Use multi-line editors only for genuinely long notes, declarations, scripts, or advanced
  structured text. Default them to a compact 3-6 lines with an explicit expand dialog.
- [x] Convert long read-only reports into a summary, status chips, tables, and a collapsible
  **View details** drawer. Logs may use a resizable dedicated viewer.
- [x] Break complex configuration into a seven-step workflow: case, formulation, algorithms,
  budget/runs, scenarios, validation/outputs, review/launch.
- [x] Put advanced or engineering-only fields behind an explicit Advanced/Diagnostics disclosure.
- [x] Keep primary and destructive actions visually distinct and close to the content they affect.
- [x] Show inline validation and concise corrective guidance; do not reserve large empty panels for
  future messages.
- [x] Avoid nested scrolling. Each workspace should normally have one page scroll surface; dialogs
  and dedicated log/table viewers may scroll independently.

### 3.4 Visual system

- [x] Retain the calm slate/blue identity but define reusable semantic tokens for background,
  surface, border, text, accent, ready, attention, blocked, failed, historical, and focus states.
- [x] Use a coherent typography scale with a readable base size, clear heading hierarchy, and
  tabular numerals for scientific values.
- [x] Use an 8-pixel spacing system and consistent card/form density.
- [x] Provide compact and comfortable density modes if dense scientific tables require both.
- [x] Require 40-44 logical-pixel minimum primary interaction targets.
- [x] Support light and dark themes with equivalent semantic contrast.
- [x] Make layouts adaptive at 1280x720, 1440x900, 1920x1080, and high-DPI scaling.
- [x] Keep plots, tables, units, uncertainty, missing values, warnings, and provenance readable at a
  glance.

### 3.5 Accessibility and interaction

- [x] Add explicit accessible names/descriptions for non-trivial controls.
- [x] Associate labels and editors; define logical keyboard focus order.
- [x] Make the complete ordinary workflow operable by keyboard.
- [x] Provide visible focus styling for buttons, navigation, tabs, tables, plots, and form controls.
- [x] Do not use color alone to communicate state.
- [x] Add source support for text scaling/high DPI and screen-reader-visible status changes.
- [x] Retain atomic protocol application, workflow locks, policy independence, and safe fallback
  behavior under the redesigned presentation.

### 3.6 GUI verification

Corrective user run `phase3-20260807-052047` passed 18/18 commands and supersedes the failed first
run for current Windows-local evidence. All four light/dark/high-DPI render cells were readable and
reported zero missing glyphs, replacement characters, clipping, compact-input, and long-editor
failures. All durable and current-source hashes matched on review, and no prohibited workflow ran.
The prior run remains immutable failed history. A tracked all-workspace keyboard/accessibility
collector and ignored noninteractive Windows and Linux xcb validators encode the remaining proof
without executing scientific actions or collecting reviewer answers.

- [ ] Add light/dark reference renders on Linux xcb; Windows reference renders are accepted.
- [x] Add a font-family and glyph-availability gate; a screenshot containing replacement/tofu glyphs
  must fail even if its dimensions and byte size look valid.
- [x] Verify 1280x720, 1440x900, 1920x1080, and at least one 200% scale Windows render.
- [ ] Complete all-workspace clipping, overflow, compact-field, expanded-editor, and scroll review;
  Dashboard matrix checks are accepted.
- [ ] Complete retained automated interaction evidence for grouped navigation, search, collapse persistence,
  historical workspace restoration, and
  blocked-page explanations.
- [ ] Test keyboard-only Study Setup and presentation navigation without launching scientific work.
- [ ] Run the prohibited ordinary-view terminology audit across every visible workspace and dialog.
- [ ] Pass the noninteractive information-hierarchy and ordinary-terminology audits across all
  sixteen workspaces; do not collect manual reviewer answers.

### Phase 3 exit gate

Phase 3 is complete only when all sixteen workspaces are reachable through the new grouped shell,
ordinary inputs are compact and structured, legitimately long content uses focused expandable
viewers, Dashboard and Study Setup responsibilities are separated, accessibility and multi-platform
render tests pass, historical workspaces restore safely, and no scientific/runtime contract changes
through the presentation layer.

**Current gate state:** Windows-local baseline accepted; corrected automated rerun pending. Phase 2
evidence `phase2-20260807-003828` and Phase 3 Windows evidence `phase3-20260807-052047` are accepted.
Runs `phase3-20260807-045558` and `phase3-remaining-windows-20260807-092741` remain immutable failed
history. Phase 3 is not yet corrected-Windows-validated, Linux-xcb-rendered, or release-ready.
Automated evidence does not infer a human screen-reader or scientist study.

---

## Phase 4 — Fresh candidate, component evidence, and protected scientific qualification

### Goal

After Phases 1-3 stabilize the contracts, produce one fresh A-E/F-off candidate and execute the
frozen development, qualification, and protected scientific evidence in the legally permitted
order.

### Required work

#### 4.1 Pre-candidate physical training gate

- [ ] Bind the exact clean source to the target NVIDIA device and environment.
- [ ] Demonstrate dedicated VRAM allocation, batched counted contexts, zero CPU-CUDA inner-loop
  transfers, zero hidden context solves, and greater-than-95% eligible CUDA event-time share under
  the already declared gate.
- [ ] If the physical gate fails, port the measured eligible remainder without changing scientific
  semantics, then repeat it.
- [ ] Do not infer whole-application utilization, throughput, benefit, energy, or thermal claims from
  scoped event-time evidence.

#### 4.2 Freeze and train one fresh candidate

- [ ] Freeze one new non-tuning training plan under the corrected source and analysis versions.
- [ ] Enable approved A-E; keep F disabled and excluded from formal A-E evidence.
- [ ] Bind exact source, code identity, cases, curriculum, seeds, hyperparameters, FE/scenario
  budgets, runtime mode, dependency environment, and container identity.
- [ ] Keep development and protected identities separate; do not use protected cases for training,
  tuning, reward design, checkpoint selection, or calibration.
- [ ] Train independently; retain interruptions, failures, exact resume, receipts, and raw artifacts.
- [ ] Emit an immutable unqualified candidate only. Do not register or activate it.

#### 4.3 Candidate-bound development gates

- [ ] Run physical CPU/CUDA candidate equivalence with fallback forbidden.
- [ ] Run the frozen A-E component matrix with paired equal-FE cells and independent validation.
- [ ] Retain failed components and falsification outcomes as well as accepted evidence.
- [ ] Remove or keep disabled any component that does not support its claimed incremental value.
- [ ] Run the corrected development screening without changing thresholds after results are seen.
- [ ] Proceed to formal eligibility only if every frozen prerequisite passes.
- [ ] Never let screening issue a receipt or any campaign register/activate a policy.

#### 4.4 Complete the scientific campaign

- [ ] Populate only independently reviewed, checksum-bound ORPD profiles for external cases.
- [ ] Execute disclosed mathematical-reference multistart/exhaustive comparisons and report negative
  or infeasible results honestly.
- [ ] Freeze training, validation, and protected identities cryptographically.
- [ ] Freeze source, policy, analysis, containers, seeds, budgets, scenarios, and statistical plan
  before opening protected cases.
- [ ] Execute the runtime-enumerated comparator campaign under paired equal-FE conditions.
- [ ] Report feasibility probability and violation distributions before objective comparisons.
- [ ] Report paired effects, intervals, multiplicity-controlled tests, anytime outcomes, failures,
  solver calls, wall time, RAM/VRAM, transfers, thermal, power, and scoped energy separately.
- [ ] Do not pool CPU fallback and CUDA-only results or heterogeneous timing strata.
- [ ] Independently validate retained final solutions and preserve raw arrays/failures.
- [ ] Limit every claim to the tested cases, formulation, objectives, scenarios, budgets, devices, and
  execution modes.

#### 4.5 Candidate decision

- [ ] If any prerequisite or frozen criterion fails, retain the candidate as negative/unqualified
  evidence and return to a new versioned plan; do not weaken thresholds or reinterpret results.
- [ ] If all qualification criteria pass, issue a checksum-bound qualification receipt through the
  independent lifecycle authority.
- [ ] Registration and activation remain separate, explicit, audited human actions.

### Mandatory evidence

- [ ] Exact candidate/source/container/dependency identities.
- [ ] Physical device-equivalence report.
- [ ] A-E component evidence and falsification rows.
- [ ] Corrected statistical-analysis version and exact pair manifest.
- [ ] Independent validation records for every retained final solution.
- [ ] Complete failures, cancellations, fallback exclusions, and resource provenance.
- [ ] Protected-campaign opening record and immutable raw evidence.
- [ ] Qualification or negative decision with prohibited-claim boundary.

### Phase 4 exit gate

Phase 4 is complete only when the exact candidate has passed all required development and protected
gates under the corrected contracts, or has been retained as a final negative result with no release
claim. Phase 5 can proceed only with a scientifically qualified candidate or an explicitly approved
release scope that excludes policy-benefit claims and any unqualified policy artifact.

---

## Phase 5 — Immutable candidate reproduction, packaging, and final release

### Goal

Produce one reproducible, clean, internally consistent release in which source, distributions,
containers, evidence, metadata, manifests, documentation, and public claims bind to the same
immutable commit.

### Required work

#### 5.1 Freeze the candidate

- [ ] Start from a clean clone of the exact approved candidate commit.
- [ ] Confirm no generated policies, checkpoints, lineages, result databases, logs, screenshots,
  credentials, user data, caches, or publication exports are tracked or packaged.
- [ ] Run the complete requirement audit and close every traceability row with direct evidence.
- [ ] Promote the final accepted RC identity to `12.0.0` only after all Phase 5 prerequisites pass,
  and confirm v12.0.0 is consistent in code, package metadata, documentation, status, and artifacts.

#### 5.2 Build distributions

- [ ] Create a previously absent staging directory.
- [ ] Build exactly one wheel and one sdist from the clean candidate.
- [ ] Install and test each distribution outside the checkout with checkout `PYTHONPATH` removed.
- [ ] Verify all required CLI and GUI entry points from the installed wheel.
- [ ] Generate separate immutable manifests for the source tree, wheel, and sdist.

#### 5.3 Build and qualify containers

- [ ] Build immutable CPU and CUDA images from the same exact source commit and locked dependencies.
- [ ] Retain image digests, maximum provenance, embedded/external SBOMs, scanner identity/database
  timestamp, complete vulnerability reports, and filesystem/application manifests.
- [ ] Run CPU without NVIDIA access and CUDA with exactly the selected GPU.
- [ ] Verify non-root UID/GID, read-only root, dropped capabilities, no-new-privileges, bounded temp,
  persistent data, health, restart, cancellation, and shared physical-device leases.
- [ ] Repeat final-candidate parity, memory pressure, staging, controlled fallback, recovery,
  cancellation, soak, thermal, power, and scoped energy gates without extrapolation.
- [ ] Complete the supported WSL2/WSLg and target-laptop qualification record.

#### 5.4 Run final CI and clean-machine reproduction

- [ ] Run unit, invariant, parity, migration, GUI, accessibility, integration, regression,
  scientific, container, distribution, and release-integrity suites for the exact candidate.
- [ ] Run the compatibility Python/OS matrix.
- [ ] Run trusted self-hosted physical CUDA jobs where required.
- [ ] Retain installed-wheel GUI renders and interactive browser/client proof.
- [ ] Reproduce wheel, sdist, CPU image, and CUDA image behavior on clean target systems.
- [ ] Require all final artifact uploads and reports to include the exact candidate identity.

#### 5.5 Generate final records and release

- [ ] Generate active release metadata from executed gate outputs; do not hand-author `VERIFIED`
  fields.
- [ ] Generate the new release freeze only after the final source and artifacts are immutable.
- [ ] Generate the root/source manifest and distinct wheel/sdist/image manifests.
- [ ] Update scientific-equivalence, hardware-qualification, GUI, container, validation, and
  vulnerability records from final evidence.
- [ ] Update README, user guide, changelog, implementation report, audit closure, citation, and
  claim limitations.
- [ ] Verify every digest and cross-reference against the exact commit and artifact.
- [ ] Obtain explicit authorization before tag, push, publication, or release.

### Final release acceptance criteria

The release is ready only when all statements below are simultaneously true:

- [ ] Phases 1-5 and G0-G11 are closed with direct evidence.
- [ ] The active release identity is exactly v12.0.0; v6.9 appears only in explicitly historical
  records and migration/compatibility fixtures.
- [ ] Statistical definitions, pairing, improvement orientation, AUC, and fallback method provenance
  are correct and versioned.
- [ ] Exact FE/batch accounting and partial-failure provenance pass.
- [ ] GUI, CLI, and final campaigns bind identical execution semantics.
- [ ] Formal CUDA evidence contains no CPU fallback.
- [ ] Every policy used by an experiment is qualified, active, immutable, checksum-valid, and
  compatible, or deterministic baseline fallback is explicit.
- [ ] F is disabled by default and excluded from A-E production qualification.
- [ ] Protected evidence is frozen, independently validated, and leakage-free.
- [ ] The modern GUI passes accessibility, terminology, responsive-layout, font/glyph, and packaged
  interaction gates.
- [ ] Wheel, sdist, CPU image, and CUDA image contain no generated policy or user data.
- [ ] Source, artifact manifests, image digests, SBOMs, metadata, documentation, and CI all bind to
  one immutable candidate.
- [ ] Public claims are no broader than the exact retained evidence.

### Phase 5 exit gate

Phase 5 is complete only after every final acceptance criterion passes and explicit release
authorization is recorded. After Phase 1 establishes the new identity and until that moment,
describe the tree as a v12.0 development or release candidate with pending qualification, not as
v12.0.0 final or release-ready.

---

## 5. Five-phase summary

| Phase | Primary outcome | Blocks |
|---|---|---|
| 1. Identity and scientific correctness | Truthful `12.0.0.devN` identity and corrected, versioned qualification analysis | Fresh candidate training and formal evidence |
| 2. Runtime and provenance hardening | Concrete device binding, explicit fallback, exact FE/cardinality, physical leases, partial failures | Trustworthy hardware and campaign execution |
| 3. Modern scientist GUI | Grouped navigation, modern Dashboard, compact inputs, progressive disclosure, accessibility | Final GUI/package qualification |
| 4. Candidate and scientific evidence | Fresh A-E/F-off candidate, equivalence, ablations, protected campaign, independent decision | Release production |
| 5. Immutable release | Clean distributions/images, final CI, manifests, SBOMs, metadata, documentation, authorization | Public release |

## 6. Required record updates after each phase

After each phase, update all four authoritative records together:

1. append exact commands, observed facts, artifacts, failures, decisions, and next action to
   `ACTIVE_CONTINUATION_LOG.md`;
2. update only the directly evidenced gate status in `IMPLEMENTATION_GATES.md`;
3. update requirement evidence and remaining proof in `REQUIREMENT_TRACEABILITY.md`; and
4. update the current boundary and exact resume action in `RELEASE_READY_CONTINUATION_HANDOFF.md`.

Never mark a phase complete from source existence, a passing unit test, or a harness alone. Record
failed attempts and rejected evidence; do not silently delete, overwrite, or reinterpret them.

## 7. Immediate next action

Run both ignored noninteractive remaining-gate lanes documented in
`validation/PHASE3_REMAINING_VALIDATION.md`: the Windows automated keyboard/accessibility collector
and the Linux xcb light/dark all-workspace validator. Return both complete timestamped directories
for source/hash and evidence review. Do not begin Phase 4, start policy training/evaluation, open
protected cases, or regenerate release artifacts while Phase 3 remains open.
