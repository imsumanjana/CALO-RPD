# CALO-RPD v12.0 onward release update and fix plan

**Plan status:** Phase 1 and Phase 2 validation are accepted. Phase 3 GUI coding is implemented at
`12.0.0.dev1`; Windows-local run `phase3-20260807-052047` and pre-tabbed automation `112621` are
accepted for their exact source. Tabbed-source run `120240` failed formatting and exposed a
Portfolio table-width gap during screenshot review; both are corrected in current source.
Corrected Windows run `121530` is now accepted at 10/10; Linux xcb evidence remains pending. This is
not a release declaration,
scientific qualification, policy approval, or authority to open protected cases.

**Policy-development boundary:** All existing policies are development-only, unqualified, inactive,
non-final, and excluded from release. Phase 4 performs coding and development validation only. It
does not train, evaluate, qualify, register, activate, or delete a policy. Old-policy deletion and
completely new A-E/F-off policy training occur only after the Phase 4 development freeze through
separately authorized user-controlled actions.

**Prepared:** 2026-08-06 (Asia/Calcutta)

**Target release line:** CALO-RPD v12.0 onward. Phase 1 establishes the v12.0 development identity;
Phase 4 produces a validated development freeze only; Phase 5 may produce v12.0 release candidates
and, after every final gate passes, the final v12.0.0 release.
Until the Phase 1 identity change is implemented and verified, use `post-v6.9 development` rather
than claiming that the active tree is either the immutable v6.9.0 release or an already qualified
v12 release.

### v12 release-version policy

The v12 line begins in Phase 1 and uses these identities consistently across source, Python package
metadata, documentation, evidence, distributions, containers, and Git tags:

| Lifecycle stage | Python/PEP 440 version | Human-facing label | Permitted phase |
|---|---|---|---|
| Development | `12.0.0.devN` | `CALO-RPD v12.0.0-dev.N` | Phase 1 onward |
| Release candidate | `12.0.0rcN` | `CALO-RPD v12.0.0-rc.N` | Phase 5 after the post-development policy-scope decision |
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
- [x] Reserve `12.0.0rcN` for Phase 5 release candidates created only after the Phase 4 development
  freeze and post-development policy-scope decision; Phase 1-4 development is not an RC.
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
- [x] Reclaim wide workspace surfaces with balanced side-by-side groups, and replace long vertical
  section-card stacks with accessible tabs in ORPD Formulation, Robust Scenarios, Portfolio Manager,
  Application Settings, and Benchmark & Evidence.
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
run for its Windows-local source. The later noninteractive run
`phase3-remaining-windows-20260807-112621` passed 10/10, including all sixteen workspaces in light
and dark/200%, 13 Phase 3 contracts, and both focused regressions. Its 47 durable and 36 source hashes
matched. Both runs remain valid for their exact manifests, and no prohibited workflow ran.

The subsequent tabbed-layout refinement changes current GUI source, so it is not covered by those
accepted manifests. Its first run, `phase3-remaining-windows-20260807-120240`, passed 9/10 commands
and all tab interactions/renders but failed formatting; screenshot review also exposed underused
width and shortened text in Portfolio Requested outputs. The Portfolio tree and affected formatting
are corrected, and `phase3-remaining-windows-20260807-121530` passed all 10 Windows commands. The
tracked v3 all-workspace collector visits every shared section tab, exercises
keyboard selection, retains screenshots, and now fails on tree unused width, horizontal overflow,
or header/cell clipping. Ignored noninteractive Windows and Linux xcb validators encode the current
proof without executing scientific actions or collecting reviewer answers.

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

**Current gate state:** Windows-local baseline and pre-tabbed automated evidence are accepted. Phase 2
evidence `phase2-20260807-003828`, Phase 3 Windows baseline `phase3-20260807-052047`, and pre-tabbed
automation `phase3-remaining-windows-20260807-112621` are accepted for their exact source manifests.
Runs `phase3-20260807-045558`, `phase3-remaining-windows-20260807-092741`,
`phase3-remaining-windows-20260807-100054`, `phase3-remaining-windows-20260807-111826`, and
`phase3-remaining-windows-20260807-120240` remain failed history. Corrected tabbed Windows source is
accepted by `phase3-remaining-windows-20260807-121530`; it is not yet Linux-xcb-rendered and is not
release-ready. Automated evidence does not
infer a human screen-reader or scientist study.

---

## Phase 4 — Production development completion, empty-policy hardening, and development freeze

### Goal

After Phases 1-3 stabilize the contracts, complete and harden the production implementation without
training, evaluating, qualifying, registering, activating, or deleting any policy. Treat every old
policy as development-only, unqualified, inactive, non-final, and excluded from release. End Phase 4
with a source-bound development freeze that is ready for a separately controlled empty-policy
cleanup and new-policy training process.

### Required work

#### 4.1 Complete the approved scientific and runtime implementation

- [ ] Finish the approved A-E implementation against one canonical runtime/training transition
  authority; keep F independently feature-gated, experimental, and disabled by default.
- [ ] Preserve deterministic behavior, exact FE/scenario accounting, mixed-variable validity,
  fail-closed nonconvergence, immutable resume identity, and protected-case isolation.
- [ ] Complete whole-population counted-context batching, selected-device retention, and outer-boundary
  materialization without hidden power-flow/context reruns or candidate-level CPU-CUDA loops.
- [ ] Complete explicit CUDA-preferred and CPU-only execution behavior. Formal execution requires an
  identified NVIDIA device and forbids fallback; exploratory fallback is a separately identified
  full-request CPU restart.
- [ ] Keep admission at no more than 80% of currently free VRAM or currently available RAM. Intel XPU
  remains non-executable.
- [ ] Implement every remaining unit, invariant, parity, failure, resume, fallback, leakage, and
  regression contract required to prove the final development semantics without a trained policy.

#### 4.2 Make the empty-policy state a supported first-class workflow

- [ ] Start the GUI and CLI safely when no policy artifacts, registrations, or active-policy records
  exist.
- [ ] Keep policy-dependent formal experiments visibly locked until one future policy is independently
  qualified, registered, explicitly activated, immutable, checksum-valid, and ABI-compatible.
- [ ] Keep non-policy configuration, result inspection, diagnostics, import/training preparation, and
  lifecycle administration usable without silently fabricating or selecting a policy.
- [ ] Reject missing, corrupt, incompatible, unqualified, inactive, or stale policy references with
  an explicit explanation and deterministic safe fallback where the workflow legally permits it.
- [ ] Preserve strict separation between training, qualification, registration, activation, and
  experiment snapshot binding; none may trigger another automatically.
- [ ] Ensure deleting or removing an old artifact cannot leave a dangling active registration or
  cause an old policy to be regenerated, downloaded, selected, or reinterpreted.

#### 4.3 Complete production, container, package, and CI hardening

- [ ] Complete deterministic clean-install, migration, rollback, configuration, result, interruption,
  cancellation, recovery, and corrupted-artifact behavior.
- [ ] Complete wheel/sdist staging rules, installed-distribution entry points, package-data boundaries,
  and checkout-independent execution without generating final release manifests early.
- [ ] Complete CPU and NVIDIA CUDA container definitions with locked dependencies, non-root runtime,
  read-only root, dropped capabilities, bounded temporary storage, persistent data, health, restart,
  cancellation, and shared physical-device leases.
- [ ] Complete trusted CI definitions for supported Python/OS lanes, packaged GUI rendering,
  container/distribution checks, and self-hosted physical CUDA evidence where required.
- [ ] Prepare noninteractive, source-bound manual validators for physical NVIDIA batching/VRAM,
  CPU/CUDA evaluator parity, containers, packages, clean-machine behavior, and empty-policy startup.
- [ ] Do not run policy training/evaluation, protected campaigns, scientific qualification, release
  freezes, or public-claim generation as part of Phase 4 development or its validator.

#### 4.4 Prepare controlled old-policy removal without deleting anything

- [ ] Implement a read-only inventory of old policy files, members, checkpoints, lineages, receipts,
  registrations, active references, database rows, package inclusions, and exact SHA-256 identities.
- [ ] Implement a dry-run removal plan that resolves every target, refuses paths outside the
  designated policy store, and reports dependent database/application references.
- [ ] Implement transactional deactivation/reference cleanup and a separate explicitly authorized
  deletion action with an immutable deletion receipt. The deletion action is not executed in Phase 4.
- [ ] Prove through source/package/container manifests that old policies and generated training data
  cannot enter wheels, sdists, images, release archives, or default runtime state.
- [ ] Use synthetic temporary fixtures for compatibility and removal tests; do not depend on an old
  policy as the final scientific baseline, training initializer, or release artifact.
- [ ] Preserve historical negative-evidence records without preserving an executable or activatable
  policy artifact in the final release scope.

#### 4.5 Close development and create the source-bound development freeze

- [ ] Close every Phase 1-4 development, runtime, GUI, empty-policy, physical-device, container,
  package, migration, provenance, and traceability row with direct evidence or an explicit blocker.
- [ ] Bind the exact source commit, dirty/clean state, schemas, dependencies, environment, supported
  devices, container declarations, and validator identities.
- [ ] Demonstrate that the development freeze contains no qualified/active/final policy and makes no
  policy-benefit, scientific-superiority, protected-case, or final-release claim.
- [ ] Freeze the application development interfaces required for later new-policy training:
  A-E/F-off configuration, policy ABI, training environment, evaluator, decoder/repair behavior,
  accounting, curriculum schema, receipt schema, and qualification authority boundaries.
- [ ] Do not promote to `12.0.0rcN`, regenerate final release manifests, or call the tree release-ready
  during Phase 4.

### Mandatory evidence

- [ ] Exact source, schema, dependency, environment, package, container, and validator identities.
- [ ] Development-only CPU/CUDA evaluator parity and physical batching/VRAM evidence, with claim scope.
- [ ] Empty-policy GUI, CLI, database, migration, restart, and locked-workflow evidence.
- [ ] Unit, invariant, fallback, leakage, resume, failure, package, container, and clean-machine evidence.
- [ ] Old-policy inventory and dry-run removal report with no deletion performed.
- [ ] Proof that distributions, images, manifests, and default runtime state exclude old/generated policies.
- [ ] Explicit records that policy training/evaluation, qualification, protected cases, registration,
  activation, and release publication were not executed.
- [ ] Development-freeze decision with remaining external proof and prohibited-claim boundaries.

### Phase 4 exit gate

Phase 4 is complete when the application implementation is source-bound, development-complete, and
validated across its required engineering boundaries; empty-policy operation is safe; old policies
are inventoried and excluded from release artifacts; controlled deletion is prepared but not
executed; and no policy training/evaluation or qualification has occurred. Phase 4 produces a
development freeze, not a trained policy, release candidate, or release.

### Post-Phase 4 controlled policy transition — outside coding development

After Phase 4 closes, and only under separate explicit user actions:

1. review the exact old-policy inventory and dry-run removal plan;
2. explicitly authorize deletion of the resolved old-policy targets and retain the deletion receipt;
3. verify that the frozen application starts and behaves safely with an empty policy store;
4. freeze a new A-E/F-off training and qualification plan against the Phase 4 development freeze;
5. train a completely new policy without using old policy weights, activation state, qualification,
   or evidence as the final candidate;
6. independently evaluate and qualify that exact new policy without automatic registration or
   activation; and
7. choose the Phase 5 scope: include only the newly qualified checksum-bound policy, or approve a
   policy-free release with no policy-benefit claim.

Phase 5 cannot start until that post-development policy-scope decision is recorded. Training,
evaluation, qualification, protected-case execution, deletion, registration, and activation are not
performed by the Phase 4 coding agent or Phase 4 validator.

---

## Phase 5 — Clean release-scope reproduction, packaging, and final release

### Goal

Produce one reproducible, clean, internally consistent release in which source, optional newly
qualified policy, distributions, containers, evidence, metadata, manifests, documentation, and
public claims bind to the recorded immutable identities and approved policy scope.

### Required work

#### 5.1 Freeze the release source and policy scope

- [ ] Start from a clean clone of the exact approved Phase 4 development-freeze commit.
- [ ] Bind the recorded post-development decision to exactly one scope: a newly trained and qualified
  policy with its own checksum, or a policy-free release with no policy-benefit claim.
- [ ] Confirm no generated policies, checkpoints, lineages, result databases, logs, screenshots,
  credentials, user data, caches, publication exports, or old policy artifacts are tracked or
  packaged. An explicitly included new qualified policy must use a separate immutable manifest.
- [ ] Run the complete requirement audit and close every traceability row with direct evidence.
- [ ] Promote the final accepted RC identity to `12.0.0` only after all Phase 5 prerequisites pass,
  and confirm v12.0.0 is consistent in code, package metadata, documentation, status, and artifacts.

#### 5.2 Build distributions

- [ ] Create a previously absent staging directory.
- [ ] Build exactly one wheel and one sdist from the clean release source.
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
- [ ] Repeat final-source parity, memory pressure, staging, controlled fallback, recovery,
  cancellation, soak, thermal, power, and scoped energy gates without extrapolation. Run
  policy-specific gates only when the approved scope includes the newly qualified policy.
- [ ] Complete the supported WSL2/WSLg and target-laptop qualification record.

#### 5.4 Run final CI and clean-machine reproduction

- [ ] Run unit, invariant, parity, migration, GUI, accessibility, integration, regression,
  scientific, container, distribution, and release-integrity suites applicable to the exact approved
  source and policy scope.
- [ ] Run the compatibility Python/OS matrix.
- [ ] Run trusted self-hosted physical CUDA jobs where required.
- [ ] Retain installed-wheel GUI renders and interactive browser/client proof.
- [ ] Reproduce wheel, sdist, CPU image, and CUDA image behavior on clean target systems.
- [ ] Require all final artifact uploads and reports to include the exact source identity and, when
  applicable, the separate new-policy identity.

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
- [ ] GUI and CLI bind identical execution semantics; when policy/scientific campaigns are included
  in the approved scope, those campaigns bind the same semantics.
- [ ] Formal CUDA evidence contains no CPU fallback.
- [ ] No old policy artifact is present. Every included policy is newly trained after the development
  freeze, qualified, immutable, checksum-valid, compatible, and separately manifested; every use by
  an experiment additionally requires explicit activation. Otherwise the approved release is
  policy-free and makes no policy-benefit claim.
- [ ] F is disabled by default and excluded from any new A-E production qualification.
- [ ] Protected evidence is frozen, independently validated, and leakage-free when the approved
  release scope includes scientific policy claims; otherwise exclusion is explicitly proven.
- [ ] The modern GUI passes accessibility, terminology, responsive-layout, font/glyph, and packaged
  interaction gates.
- [ ] Wheel, sdist, CPU image, and CUDA image contain no generated policy or user data.
- [ ] Source, artifact manifests, image digests, SBOMs, metadata, documentation, and CI all bind to
  one immutable source identity plus, only when included, one separate qualified policy checksum.
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
| 1. Identity and scientific correctness | Truthful `12.0.0.devN` identity and corrected, versioned qualification analysis | Accurate later development and evidence |
| 2. Runtime and provenance hardening | Concrete device binding, explicit fallback, exact FE/cardinality, physical leases, partial failures | Trustworthy hardware and campaign execution |
| 3. Modern scientist GUI | Grouped navigation, modern Dashboard, compact inputs, progressive disclosure, accessibility | Final GUI/package qualification |
| 4. Development completion and freeze | Final A-E/F-off-capable code, empty-policy safety, production hardening, old-policy removal preparation, validated development freeze | Post-development deletion and new-policy decision |
| 5. Immutable release | Clean policy-free or newly-qualified-policy scope, distributions/images, final CI, manifests, SBOMs, metadata, documentation, authorization | Public release |

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
