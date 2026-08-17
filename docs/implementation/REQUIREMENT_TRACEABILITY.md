# Modernization requirement traceability

This is the live completion audit for the remediation objective defined in
[`../COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md`](../COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md).
It intentionally distinguishes implementation, local evidence, physical/external evidence, and
approval-gated scientific work. A passing unit suite does not close a hardware, container, or
scientific-evidence requirement.

Status vocabulary:

- **Locally verified** — implementation exists and relevant local tests pass.
- **Implemented / external proof pending** — code/harness exists, but the required external runtime
  has not produced an attestation.
- **Partial** — material requirements remain.
- **Approval-gated** — changing scientific CALO behavior is prohibited until the recorded decision.

## Runtime, memory, and compatibility

| Requirement | Status | Authoritative evidence | Remaining proof |
|---|---|---|---|
| Calculate the 80% allowance from memory free/available at admission, not installed capacity | Physically verified on the observed RTX 4060 host and source-bound CUDA container | `compute/memory_budget.py`, `accelerated/vram_residency.py`, focused tests; clean `d6a950c` host evidence plus exact image `cuda-1f02a94` bounded resource-recovery record SHA `5507116b0851d1ce000bebce237bcf1cf12020b8a0cbf2e27c7bc459cb693e77` | Repeat for the immutable final candidate and trusted CI runner |
| Prevent independent processes from overcommitting one physical GPU | Physically verified across host processes and independent source-bound containers | `compute/device_lease.py`, device-residency tests; clean `d6a950c` host evidence and exact-image shared-volume holder/contender/post-release acquisition evidence SHA `c13946dc5799054e64a90eef4bc8f0ce797646ceb6b445a0de6c37c2d8e9cdcf` | Repeat for the immutable final candidate and trusted CI runner |
| Keep CUDA-resident, microbatch retry, staged-host, governed CPU fallback, and fail-closed states distinct | Physically exercised on host and in the source-bound CUDA image with controlled failure boundaries | `accelerated/device_resident_orpd.py`, `accelerated/vram_residency.py`, v6.9 VRAM tests; clean `d6a950c` host evidence and exact-image resource-recovery record retained host staging, controlled backoff, clean CPU restart and CUDA recovery | Controlled OOM is not natural hardware exhaustion; repeat for the immutable final candidate |
| Retain a bounded physical accelerator soak with protection and observed-only telemetry | Physically verified on the observed RTX 4060 host and exact source-bound CUDA image | Host evidence at clean `67bd18e`; exact image `cuda-1f02a94` completed `3600.000632077` seconds with 3,600 GREEN samples, zero safe/protection stops, verified 3,602-event chain, maximum UTC gap `1.105776` seconds, and result SHA `aab8b13c7e01a27260e7ca0934ac472e0e845103c0c07356d9468f031bb391b5` | Repeat for final candidate/CI; CPU temperature, GPU power limit, and whole-system energy remain unavailable and must not be inferred |
| Remove utilization targets, device-memory percentages, task shares, and work-stealing from experiment execution | Locally verified | Current `ExperimentConfig`, current JSON schema, automatic CUDA-first scheduler, GUI configuration tests | None for experiment execution; policy-training routing is separately approval-gated |
| Remove executable Intel XPU support without falsifying historical records | Locally verified | XPU-free bootstrap/scheduler/runtime, view-only legacy backend migration, historical compatibility tests | Clean-machine compatibility run in CI |
| Preserve old populated databases through rollback-safe migration | Implemented / fresh proof pending | SQLite schema v3, online pre-migration backup, integrity check, SHA-256 receipt, transactional DDL, future-version rejection, legacy combined Portfolio/Study preservation, additive Portfolio goal/Study setup tables, and execution-controller migration contracts; `tests/integration/test_database_workflow.py`; `tests/integration/test_workspace_execution_control.py` | Fresh owner Phase 6 schema-v29 validator and CI matrix execution |

## Containers and reproducibility

| Requirement | Status | Authoritative evidence | Remaining proof |
|---|---|---|---|
| Reproducible CPU and NVIDIA CUDA images with immutable dependencies | Physically built and source-bound on target workstation | Exact clean `1f02a94` produced CPU OCI `sha256:463be6ab…` and CUDA OCI `sha256:218ec1cc…` with maximum BuildKit provenance/SBOM, retained metadata, immutable build declarations and distinct runtime-loaded tags; build declaration remains operator metadata, not a signature | Repeat for the eventual final candidate/clean machine and retain CI artifacts |
| Non-root, read-only, capability-dropped runtime with persistent data volume | Core and corrected GUI runtime physically verified | Exact images passed UID 10001, read-only root, capability/no-new-privilege invocation, writable data, schema round-trip and shared `/data/device-leases`; independent CUDA containers proved exclusion then post-release acquisition. Clean corrected image `cpu-31a4713` loaded the Qt xcb plugin, required a live app PID for health, rendered the Dashboard, preserved marker SHA `8ddd6fb1d67b6840d1b9a9887f2c0a522ad1de4696760d872a2461eedf7ea6c3` across restart, restored on PID 40 without a restart loop, and stopped in the bounded cancellation path with exit 143/no OOM | Browser-control tooling failed before interaction and is not claimed; repeat browser interaction and the full gate on final candidate/CI |
| CPU image must work without NVIDIA hardware; CUDA image must see exactly the selected GPU | Physically verified on target workstation for core/soak paths | CPU image reported CUDA unavailable; CUDA image reported one NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch `2.10.0+cu128`, CUDA `12.8`; container case30/case57 parity/resource recovery and independently audited 3,600-sample exact-image soak passed | Repeat on trusted CI runner and eventual final candidate after GUI correction |
| SBOM and vulnerability evidence | Physically retained for exact CPU/CUDA images | BuildKit embedded SBOM/provenance plus pinned local Trivy 0.70.0 at immutable scanner digest; CPU/CUDA CycloneDX, complete JSON and zero-fixable-HIGH/CRITICAL gates retained. Both report 700 total, 23 critical and 128 high by upstream/vendor severity, zero with an available fix under the gate | Retain database identity/timestamps and repeat for final digests/CI; do not imply remaining unfixable advisories are harmless |
| Lenovo LOQ WSL2/WSLg/GPU qualification | Pending external environment | `docs/CONTAINER_RUNBOOK.md` qualification procedure | Execute and retain target-laptop report |

## Scientist workflow and experiment protocol

| Requirement | Status | Authoritative evidence | Remaining proof |
|---|---|---|---|
| Normal GUI contains no venue promise or engineering/development language | Locally, Linux-container and extracted-wheel rendered | Rendered-widget contract across five normal workspaces; corrected Linux image pre/post-restart render; clean `383e5bc` wheel imported outside the checkout and rendered 16-workspace Dashboard shell at 1440x900 with zero forbidden visible terms, evidence SHA `12e7ca4e5fb921b4c58d9d7434c87fb58fe8c77b3e3dbd6eca4717b674052181` and PNG SHA `adc340f602011436ded5f321a55e5cb3855a8a0e1e50fe613032c1089789ca1f` | Execute the configured packaged lane in GitHub Actions and complete browser interaction |
| Evidence-strength/custom protocol recommends outputs and power-aware repeated runs | Locally verified | `experiments/study_strength.py`, Dashboard protocol UI, study-strength tests | Scientist acceptance review |
| Apply protocol once and propagate atomically without partial mutation | Locally verified | Deep-copy validation, before/after diff, shared state replacement, rollback tests | None |
| Experiment protocol must not alter independent policy-training configuration | Locally verified | Separate configuration/lifecycle paths and policy-independence tests | None |
| Every power-system experiment binds a qualified, active, immutable policy snapshot | Locally verified | State/workflow/experiment-manager binding gates and policy-system tests | End-to-end packaged run |
| No-AI CALO is restricted to qualification/ablation rather than normal experiments | Locally verified | Hidden normal control, execution guards, GUI and policy tests | None |

## CALO scientific architecture and evidence

| Requirement | Status | Authoritative evidence | Remaining proof |
|---|---|---|---|
| Explain final runtime/training architecture before changing CALO semantics | Approved / confirmation complete | Exact 2026-08-03 decision “Approve TSH-CALO A–E, with F experimental and evidence-gated”; `CALO_ARCHITECTURE_CHANGE_PROPOSAL.md`; nine-part pre-implementation confirmation | Preserve the recorded boundary; deviations require new approval |
| Canonical runtime/training transition authority | Locally verified; physical evaluator parity retained | `algorithms/calo/transition_kernel.py`; shared runtime/training authority invariant; native one-step parity; 22 frozen seeded optimizer snapshot/exact-budget cases; 45-test CALO/continuation focus; complete active tree 461 passed, 63 skipped; clean `63f56ad` physical case30/case57 evaluator parity with zero semantic mismatches | Retain parity in every later B–F stage; TSH policy/inference and pressure/soak qualification remain separate |
| Topology context, hierarchical actions, and uncertainty/bandit shield | B–D mechanics and evidence producer locally verified; real ensemble unqualified; frozen screen negative | B/C evidence plus `tsh_calo_shield.py`; deterministic mechanics tests; `ae7b304` frozen paired A–E removal matrix and strict evidence verifier; completed historical five-member IEEE 30/57 v2 ensemble; valid v3 development screen with 40 exact-FE paired records and immutable negative evidence | Candidate v2/design cannot proceed; a fresh counted-v4 candidate must execute and pass the preregistered A–E matrix and target CPU/CUDA gate; no component or benefit claim exists yet |
| Physics-informed repair | Counted runtime/training and paired-evidence implementation locally verified; disabled by default | Runtime v1.1/training v5; final counted Newton state/Jacobian; batch device-resident voltage/type/generation/branch retention; one outer-boundary context materialization; analytic relaxed-control plus active constraint derivatives; exact FE/scenario accounting; zero hidden solve/evaluator reruns; dynamic fail-closed masks; `ae7b304` incremental E comparison | Prove greater-than-95% eligible CUDA share physically, then execute the frozen incremental-value/cost comparison on a fresh E-enabled candidate; no benefit claim yet |
| Evidence-driven population schedule | Experimental mechanics locally verified; disabled by default; promotion evidence absent | `tsh_calo_population_schedule.py`; separate `enabled` plus `experimental_mode` gates; preregistered design hash; feasibility/archive/diversity/budget/spacing conditions; deterministic feasibility-first contraction; no hidden FE; exact resume; nine focused tests | Must earn inclusion through paired-seed anytime/feasibility evidence without unacceptable cost, instability, overfitting or regression; otherwise keep disabled or remove |
| Versioned TSH-CALO candidate lifecycle | Locally verified through unqualified evaluation; negative screening evidence retained | `tsh_calo_policy_artifact.py`, `tsh_calo_qualification.py`, `tsh_calo_qualification_campaign.py`, algorithm-aware registry; exact ABI/provenance; protected-holdout rejection; non-serializable qualification-only evaluation capability; OS-released single-writer lease; failed-integrity resume rejection; valid v3 screen completed 40/40 records and correctly emitted no receipt | Candidate v2 remains unqualified/inactive and is barred from formal qualification under this design; the raced v1 screen remains barred from scientific use; a future candidate must satisfy a new frozen screen and direct A–E evidence before formal qualification |
| Independent TSH-CALO policy training | Versioned batched counted-E mechanics complete; historical five-member ensemble remains negative/unqualified | `tsh_calo_training*.py`, `accelerated/device_resident_orpd.py`, `accelerated/torch_orpd.py`; v5/campaign-v2/session-v2/receipt-v2 ABI; whole-population tensor PF/context retention; one outer-boundary host materialization; zero candidate-level CPU-CUDA loop; exact source/design/seed/curriculum and FE/scenario/update/reward accounting; signed evaluator-device/batching/rerun provenance; CUDA-first Safe-80; E proposal-only; F rejected; unqualified-only output. Historical v2/v3-ABI ensemble remains immutable under SHA `3adf5017dc51f33d76214aeb505da598984b9da0e4263e1ee8fe59a667180ceb` | First pass the physical greater-than-95% CUDA timing/VRAM gate. Then freeze a new non-tuning v5/campaign-v2 A–E/F-off plan, train one fresh candidate, and execute equivalence, A–E evidence and qualification in order |
| Immutable TSH-CALO ensemble inference and fallback | Integrated into production, qualification, component-ablation and device-equivalence boundaries; fresh-candidate physical equivalence pending | Ensemble assembly/load; activated-qualified production binding guard; non-serializable development capabilities; exact SHA/ABI/feature/member/calibration checks; CUDA-first current-free-VRAM 80% admission; governed CPU fallback; ensemble disagreement/shield; pre-evaluation block or explicit frozen-CALO relaunch; deterministic traces and exact resume; `e77431e` compares exact actions and bounded numerics while requiring dedicated VRAM | Execute `calo-rpd-tsh-device-equivalence` on the future counted-v4 candidate; candidate v2 failed screening and cannot be reused or activated |
| Reuse already-counted power-flow state for topology/physics context | CPU and batched tensor mechanics locally verified | `ORPDProblem.evaluate_with_context`; `AcceleratedORPDProblem.evaluate_population_with_context`; device-resident voltage/diagnostics/types/generation/flows; one packed final transfer; reconstructed ephemeral contexts; optimizer/training batch routing; exact FE/scenario accounting; no serialized solver objects or hidden PF reruns; bounded evaluation/voltage/branch/Jacobian/sensitivity parity | Physical NVIDIA share/VRAM evidence and incremental-value/cost evidence remain required |
| Correct paired statistics, effect estimates, CIs, multiplicity control, power and anytime metrics | Implemented; user validation pending | `statistics/paired.py`; both qualification engines; exact-pair/effect/fallback/convergence tests in `tests/unit/test_v120_phase1_contracts.py`; additive historical correction record | Run the local-only, Git-ignored `validation/Validate-Phase1.ps1`; return its complete hashed log directory with exact source identity; later execute only separately authorized frozen campaigns |
| Modern strong stochastic baselines | Locally verified at implementation level | Source-traceable L-SHADE 1.0.1 and pinned pycma 4.4.4 CMA-ES with deterministic snapshots | External benchmark execution |
| Deterministic/mathematical reference solutions and broader licensed case corpus | Implementation partial; development adapters physically exercised | Protocol and pinned official PGLib-OPF v23.07 validation assets; exact license/source provenance and restricted parser. Clean `07f9476` adds SciPy SLSQP local continuous-relaxation plus exhaustive all-discrete adapters, separate original-lattice validation, exact call/termination/derivative disclosure, no-bound/no-gap fail-closed semantics, protected-case refusal, independent PYPOWER checks and immutable evidence CLI. Real case30 development reports SHA `8be27e3bb467a78d524930422bafa372729c3527782e06803c949f04449763dc` and `abff7f42274c5f9ad347a2d1af67bf8a585c478de7655438cd503aac13ae4ee5` retain negative/infeasible outcomes without overclaim | Execute frozen multistart/reference comparisons; add separately certified bounds where mathematically valid; populate independently human-reviewed checksum-bound ORPD profiles before using imported AC cases as ORPD formulations |
| PGLib/stress/OOD and cryptographically protected holdouts | Partial | Frozen case-role protocol and protected identities; three non-protected PGLib case14 groups now checksum-load from source and built wheel; protected import and ORPD conversion each fail closed without explicit test-only authorization | Complete reviewed ORPD profiles and execute the still-unopened protected final tests only after the full design freeze |
| Publish complete code, policy, formulation, image, seeds, raw failures, validation and claim scope | Harness partial | Artifact verifier, package exclusions, manifests, policy/config hashes, CI uploads | Final qualified policy, opened results, image attestations, release freeze |

## v12 Phase 2 runtime traceability - 2026-08-06

| Requirement | Implementation evidence | Validation state | Remaining proof |
|---|---|---|---|
| Truthful formal/exploratory/CPU execution semantics | `compute/execution_contract.py`; config schema; GUI and CLI bindings | Accepted in `phase2-20260807-003828` | Later physical CUDA evidence where separately required |
| Mandatory concrete pre-run device resolution | `compute/device_binding.py`; experiment runner; parallel runner; GUI worker; benchmark CLIs | Accepted in `phase2-20260807-003828` | Later physical CUDA evidence |
| Stable physical ownership and queueing | UUID/PCI identity; scoped `ExclusiveDeviceLease`; frozen resolved scheduler route | Accepted in `phase2-20260807-003828` | Later multi-process physical-device exercise |
| Safe-80 and unambiguous VRAM telemetry | Availability-based admission; fixed 0.80 schemas; request/lifetime telemetry blocks | Accepted in `phase2-20260807-003828` | Later physical pressure evidence |
| Exact FE/cardinality/identity accounting | Central `validate_batch_evaluations`; strict registration; normalized candidate identity | Accepted in `phase2-20260807-003828` | Retain under later regressions |
| Partial-failure provenance and persistence | `calo-partial-run-failure-v2`; atomic existing failure transaction | Accepted in `phase2-20260807-003828` | Later injected integration evidence |
| Current topology/status truth and XPU boundary | FP64 smoke authority; active status verifier; XPU execution rejection | Accepted in `phase2-20260807-003828` | Retain under later regressions |

At this implementation checkpoint, Phase 1 evidence `phase1-20260806-230256` was accepted and Phase
2 was validation-pending. The later final Phase 2 evidence below supersedes that pending state. No
such development evidence qualifies any policy, protected case, hardware-performance claim,
release candidate, or release.

First Phase 2 evidence `phase2-20260807-001858` retained 13/15 passing commands, including 23/23
dedicated Phase 2 contracts, complete 33/33 source-manifest coverage, 20/20 retained artifact hash
matches, and no prohibited workflow execution. Generated-schema property order and one stale
pre-Safe80 error-message expectation failed. Both source-only corrections are applied; a new full
manual validator run is required before any row above may be accepted as validated.

Second evidence `phase2-20260807-003024` retained 14/15 passing commands, 23/23 Phase 2 contracts,
44/44 affected regressions, schema consistency, Ruff diagnostics, and 20/20 artifact hash matches.
Its 34 source-manifest paths exactly covered the 34 captured changed paths. Ruff format alone found
mixed line endings in the corrected VRAM regression file. That formatting is corrected; a fresh
manual run must bind the updated status and governance documents before Phase 2 can be accepted.

Final evidence `phase2-20260807-003828` closed that requirement: 15/15 commands, 23/23 Phase 2
contracts, 44/44 regressions, 20/20 retained artifact hashes, 35/35 current source hashes, exact
validator identity, and no prohibited workflows. Phase 2 is accepted.

## v12 Phase 3 GUI traceability - 2026-08-07

| Requirement | Implementation evidence | Validation state | Remaining proof |
|---|---|---|---|
| Stable grouped information architecture | `app/workspaces.py`; `gui/navigation/sidebar.py`; inline SVG registry | Corrected Windows contracts and all-workspace automation accepted; Linux boundary owner-accepted manually without a retained automated bundle | Phase 4 source-bound rerun; no automated Linux claim |
| Dashboard decision hierarchy | next-action/readiness/recent-work/activity disclosure in `dashboard_panel.py` | Windows light/dark/high-DPI and all-workspace evidence accepted; Linux boundary owner-accepted manually | Phase 4 source-bound rerun; human acceptance remains separate |
| Six-step mode-correct experiment setup | `StudySetupWorkflow`; inline `PowerSystemPanel`, `ORPDFormulationPanel`, and `RobustScenariosPanel` instances over shared `AppState`; `AppliedPortfolioGoal`; `StudyRecommendation`; separate Workspace/Individual prerequisite ledgers | Case, Formulation, Budget/runs, Scenarios, Validate/outputs, and Review/launch stay in one workspace. Workspace Study shows the exact applied goal, hard minimum, deterministic recommendation, editable scientist selection, and delta; Apply Study alone creates the draft. Individual Experiment shows the complete submitted stage, editable direct values, and no Portfolio prerequisite. Source implemented; schema-v29 owner validation pending. | Human usability/scientist acceptance remains separate |
| Compact structured inputs | application-wide density policy; integer chips; 240-480px limits; responsive Results grid; dedicated copyable database-path field | Corrected Windows clipping/input-width automation accepted; Linux boundary owner-accepted manually | Phase 4 source-bound rerun |
| Horizontal use of wide workspaces | shared `WorkspaceTabs`; side-by-side ORPD, scenario, portfolio, settings, and benchmark groups; stretching Portfolio output tree | Corrected `121530` Windows light/dark-200% tabs and tree-width gate accepted; Linux boundary owner-accepted manually | Phase 4 source-bound rerun |
| Accessible semantic visual system | named tokens, light/dark QSS, focus borders, text badges, accessible names/buddies | Windows high-DPI contrast/name/buddy/keyboard automation accepted; Linux boundary owner-accepted manually | Human accessibility acceptance remains separate |
| Render and glyph evidence | isolated render CLI; deterministic existing-OS-font registration; font provenance; separate glyph/clipping/input counts | Four corrected Windows cells accepted; Linux boundary owner-accepted manually without an automated render bundle | No reproducible automated Linux claim |
| Durable local evidence identity | ignored validator; v2 source manifest with commit/dirty/status hashes; v3 workspace evidence and v4 Windows summary | Corrected `121530` Windows hashes accepted; Linux closure is an owner decision without a retained automated directory | Phase 4 source/hash-bound validation directory |

Phase 3 Windows-local baseline evidence is accepted. `phase3-20260807-045558` and
`phase3-remaining-windows-20260807-092741` remain immutable failed history;
`phase3-20260807-052047` is the accepted baseline rerun; and
`phase3-remaining-windows-20260807-112621` accepted the corrected pre-tabbed source at 10/10. The
new tabbed-layout source postdates that manifest and therefore required fresh noninteractive
Windows and resolution of the Linux xcb boundary. Windows is accepted in `121530`; the owner later
accepted the manually validated Linux boundary without claiming an automated directory. Manual reviewer answers are excluded by user instruction,
and automated evidence does not claim a human screen-reader or scientist study. No GUI result
qualifies policy, protected-case, performance, scientific-superiority, release-candidate, or release
claims.

The first tabbed-source run, `phase3-remaining-windows-20260807-120240`, remains failed evidence:
Ruff formatting failed on five files even though 14 contracts, five layout regressions, and both
all-tab cells passed. Its screenshots also revealed underused Portfolio Requested outputs width and
shortened evidence text. Source now fixes the resize policy and the v3 workspace collector makes
tree unused width, overflow, and header/cell fit an automated gate. Corrected rerun `121530` passes.

Corrected Windows run `phase3-remaining-windows-20260807-121530` supersedes `120240` for current
Windows source and is accepted at 10/10. Fifteen contracts, six layout regressions, both v3
workspace cells, all source/evidence hashes, and the Portfolio width records pass. Windows correction
work is complete. Phase 3 was subsequently closed by the owner's manual Linux/xcb acceptance; this
does not create reproducible automated Linux evidence.

On 2026-08-12 the project owner explicitly accepted the manually validated Linux xcb boundary and
directed Phase 4 to proceed. Phase 3 is therefore closed by owner decision. No automated Linux log
directory was retained, so the Linux rendering row must continue to say `owner-accepted/manual` and
must not be cited as reproducible automated, human accessibility, scientist-acceptance, policy,
scientific-performance, RC, or release evidence.

## Current verification checkpoint

- Active identity is now `12.0.0.dev1` / `12.0.0-dev.1`, stage `development`. Phase 1 source and
  test implementations exist, but no tests were run during the user-directed coding-only task.
  Therefore prior pass counts below are historical checkpoints and do not verify the current v12
  working tree. The exact pending checks are encoded in the local-only, Git-ignored
  `validation/Validate-Phase1.ps1` harness.

- Latest active development suite, excluding the deliberately stale v6.9 release-integrity file:
  **638 passed, 63 skipped**, 68% source coverage, exact CPU/CUDA/CI locks, schema generation,
  compileall, Ruff lint/format and pinned 15-module mypy. The preceding complete-tree checkpoint
  after Change F reported **502 passed, 63 skipped, 2 failed**;
  both failures are the expected stale release freeze/root manifest and are not regenerated during
  G9 development.
- Repository Ruff lint and format: **pass across 422 files** at the last complete source gate; the
  subsequently added A–E and device-equivalence files pass focused Ruff/type gates.
- Generated experiment schema: **current**.
- Focused automatic scheduling/configuration/GUI set: **54 passed**.
- Focused database/history/learning/resume/continuation set: **29 passed**.
- Historical release-freeze tests remain separate and are regenerated only at final G11.
- Physical source-bound case30/case57 evaluator parity, bounded resource recovery and scoped CUDA
  hot-path timing are retained. Fresh-candidate device equivalence, final Docker/CI repetition,
  WSL2, final protected tests and external scientific campaign artifacts are **not yet evidence**.
- Valid v3 development screening evidence SHA-256:
  `039f2bfe31e39196e126da3961c65e4a248133ed09b009a93f64c933b2292778`. It contains 40/40 exact-
  FE paired records and zero failures, but the frozen decision is grade `U`, score `0`, and
  `passed=false`. Case30 had no feasible observations in either arm; case57's apparent median
  improvement was not significant after Holm correction (`p=0.052734375`) and its 95% interval
  crossed zero. No receipt, registration, activation, qualification, or policy-benefit claim exists.

## Historical candidate route - superseded on 2026-08-07

The following route was the recorded next step before the development-first scheduling decision. It
is preserved as immutable decision history and is no longer the current next action.

The exact decision was recorded on 2026-08-03:

> Approve TSH-CALO A–E, with F experimental and evidence-gated.

Changes A–E and disabled Change-F mechanics have passed local correctness gates. The failed v1
campaign remains immutable. V2 completed a five-member unqualified ensemble with exact real CUDA
training provenance, but its valid v3 development screen failed the frozen criteria. Case30 was
infeasible in both arms, and case57 did not pass Holm-controlled objective inference. The candidate
therefore remains inactive and cannot enter ordinary experiments or formal qualification. Do not
weaken the thresholds, run post-hoc formal trials, open protected cases, or create a receipt. The
repository now owns the missing A–E paired-evidence producer, candidate-bound CPU/CUDA equivalence
validator and batched device-resident counted-context path, but none is physical candidate evidence.
The next legal G9 action is to prove on the target NVIDIA device that eligible counted training
exceeds 95% CUDA work with dedicated VRAM and no inner transfer loop. Only after that passes may one
freeze a new non-tuning v5/campaign-v2 A–E/F-off plan and train one fresh candidate. On that exact
candidate, execute physical device equivalence, the frozen A–E matrix and screening/formal eligibility
without adapting thresholds or opening protected cases. Change F remains excluded and disabled.

## Historical development-first policy boundary - 2026-08-07 (superseded 2026-08-14)

The following bullets record the former development-first decision. They remain historical evidence
but no longer define compatibility or formal-qualification admission; the 2026-08-14 stage-neutral
contract rows below supersede that policy-lifecycle restriction:

- Phase 4 completes and validates production source with no policy training/evaluation or candidate
  qualification. It must support a safe empty policy store and exclude old/generated policies from
  packages, containers, manifests, and release state.
- Old policies are development-only, unqualified, inactive, non-final, and barred from reuse as the
  final candidate, training initialization, qualification evidence, or release artifact.
- Phase 4 prepares a checksum-bound inventory and dry-run removal mechanism but performs no deletion.
- After the Phase 4 development freeze, explicit user authorization is required to delete old
  policies. Empty-policy behavior is then verified before a new A-E/F-off plan is frozen.
- Training, candidate equivalence, A-E component evidence, screening, qualification, and protected
  scientific campaigns apply only to a completely new post-development policy and remain outside the
  Phase 4 coding goal and validator.
- Phase 5 may package only that newly qualified policy, under a separate checksum identity, or an
  explicitly approved policy-free scope with no policy-benefit claim.
- `docs/DOCUMENTATION_STATUS.md` is the current routing index. Versioned reports, dated audits,
  prior release validation, and copied build/baseline Markdown remain historical records and cannot
  override this boundary or qualify current v12 source.

## Historical Phase 4 implementation trace — 2026-08-12

- Empty-policy GUI/database/config: `app/state_manager.py` clears both CALO and TSH-CALO immutable
  binding fields, calibration, receipt, and provenance data; Algorithms exposes TSH-CALO as a
  visibly gated separate row; CALO Intelligence no longer auto-discovers checkout policies and
  retains rule-only CALO availability.
- Old-policy exclusion: `PolicyRecord.post_development_eligible`, activation/binding guards, and
  post-freeze provenance fields bar every historical artifact from v12 use or initialization.
  Future exported TSH candidates identify the exact 40-character development-freeze commit, exact
  retained freeze payload SHA-256, and an empty initialization-policy checksum. The campaign CLI
  accepts only a clean, empty-policy, post-transition freeze report matching both identities. The
  production TSH inference loader repeats the provenance
  gate after checksum-valid artifact loading, preventing serialized-config bypass while keeping old
  artifacts inspectable as development history.
- Device/accounting boundary: TSH-CALO can enter the normal experiment runner only through its
  immutable qualified binding; its inference device is overwritten with the scheduler-resolved
  device and internal/baseline fallback is false. Existing whole-population counted-context and
  outer-boundary materialization source remains the canonical A-E path; F stays off.
- Controlled retirement: `policy_retirement.py`, `manage_policy_retirement.py`, and exact database
  lifecycle snapshot/cleanup implement file/database inventory, external-artifact blockers,
  cryptographic dry-run binding, disabled authorization templates, accepted-Phase-4-freeze payload
  binding, source/path confinement, transactional cleanup, and immutable external receipts. The GUI
  exports only inventory/plan.
- Freeze/package/CI: `create_development_freeze_candidate.py` embeds a sorted SHA-256 manifest of
  every Git-tracked and non-ignored untracked source file, hashes raw Git status, and verifies the
  exact Phase 4 interface/dependency/container/exclusion contract plus policy-empty authority
  boundary. Distribution verification requires the new modules; package and Docker ignores exclude
  policy and validation content; container smoke verifies its actual filesystem manifest;
  physical-CUDA CI runs policy-free evaluator evidence. The ignored validator independently hashes
  the complete tracked/untracked source set and fails if Git status changes during the run.
- Acceptance authority: `accept_development_freeze.py` verifies a complete passing hash manifest,
  all 30 Phase 4 command identities, source/freeze/validator agreement, and an explicit decision ID
  before writing a non-overwriting receipt outside the immutable run. The receipt binds production
  source content while excluding only recorded development-policy artifacts. Future retirement,
  training provenance, registry readiness, and experiment configuration require its SHA-256. The
  Phase 4 validator never calls this command and no receipt exists before returned-log acceptance.
- Test source: the new Phase 4 retirement, development-freeze, and empty-policy integration suites
  cover synthetic path confinement, explicit authorization, stale-binding removal, gated TSH-CALO,
  disabled F/internal fallback, production artifact provenance eligibility, and source identity.
  Existing affected lifecycle fixtures now use explicit post-development provenance. Historical
  GUI training/recovery/continuation callbacks are fail-closed and cannot be re-enabled by an old
  worker completion path.
- Proof remains pending. Codex did not execute any manual-capable command. Run
  `& .\validation\Validate-Phase4.ps1` and return the complete new `validation\logs\phase4-*`
  directory. Until accepted, no package/container/CUDA/GUI pass, clean development freeze, Phase 4
  completion, RC, policy, scientific, or release claim is made.

## v12 Phase 5 release-preparation traceability - 2026-08-12

| Requirement | Implementation evidence | Current proof | Remaining gate |
|---|---|---|---|
| Exact policy scope | `scripts/release_policy_scope.py`; disabled template; exact policy/qualification/Phase 4/freeze bindings | Test and validator source only | Combined validation, transition evidence, explicit decision |
| Exactly one wheel and sdist with distinct manifests | `verify_distribution_stage.py`; `generate_distribution_manifests.py` | Source implemented | Combined manual build/install/hash evidence |
| CPU/CUDA immutable evidence aggregation | `create_release_preparation.py`; container smoke filesystem manifest; Buildx/Trivy commands in Phase 5 validator | Source implemented | Actual image IDs, metadata, SBOMs, scans and GPU runtime evidence |
| Clean installed artifacts | isolated wheel/sdist environments; cleared checkout `PYTHONPATH`; packaged GUI render | Harness implemented | Returned clean-install and render artifacts |
| Final CI contract | `verify_release_ci_contract.py`; updated pinned-action workflow | Source implemented | Actual hosted/self-hosted CI execution remains separate |
| Final metadata/freeze authority | `finalize_release_records.py`; disabled explicit authorization | Fail-closed development source only | Clean reviewed `12.0.0`, approved scope, accepted evidence, explicit authority |
| Combined phase identity | `Validate-Phase4-And-Phase5.ps1` | Ignored noninteractive wrapper implemented | User run and read-only acceptance review |

No Phase 5 implementation row is release evidence. Active identity remains `12.0.0.dev1`; release
scope, deletion/empty-store transition, combined validation, RC, final records, tag, push,
publication, and release remain unexecuted.

The first owner combined attempt, `phase4-20260812-165006`, is interrupted partial diagnostics and
does not satisfy any proof row. Its format/type/engineering/GUI diagnostics and native-command
wrapper defects were corrected in source. The ignored validators now stream output and fail at the
first nonzero native exit. No correction was validated by Codex; every proof row above still
requires the fresh owner-executed combined run.

Owner run `phase4-20260812-182252` passed environment/version/compile/schema/Ruff and failed first at
Ruff formatting, proving the corrected fail-fast route while providing no Phase 4 acceptance. The
eight reported files were mechanically formatted; Phase 5 did not start, and fresh combined proof
remains required.

Owner run `phase4-20260812-182752` advanced through formatting, typed safety, and 112 engineering
tests; GUI passed 36/37 and failed only on one visible development-stage phrase. Normal-interface
policy/campaign wording was corrected without changing policy semantics. The run remains failed
evidence and Phase 5 did not start.

Owner run `phase4-20260812-184454` passed 14 result IDs through wheel/sdist construction and failed
only when the distribution gate confused the legitimate `calo_rpd_studio/validation/` application
namespace with root Git-ignored validation evidence. Distribution/container path classification and
regression source are corrected; Phase 5 did not start and the correction still needs a fresh run.

Owner run `phase4-20260812-185135` passed 17 result IDs through the corrected archive gate and clean
wheel installation. Its clean smoke failed on a harness-only path predicate: the valid clean
environment is intentionally beneath Git-ignored repository validation storage. The validator now
requires imports from that clean environment and excludes the actual checkout source-package path.
This preserves checkout-independence and entry-point proof without misclassifying installed code.
Phase 5 did not start; fresh combined evidence remains required.

Owner run `phase4-20260812-190643` passed 24 result IDs through CPU/CUDA container construction and
smoke plus physical NVIDIA discovery. The first physical parity command rejected the dirty source
before computation because it requested durable evidence. The corrected tools separate underlying
engineering success from durable qualification: an explicit Phase 4-only option may retain
full-commit dirty-source development evidence, but it records non-durable/development-only status and
cannot set qualification true. Clean source remains mandatory for formal durable evidence, Phase 5
release preparation, and any later qualification claim. Phase 5 did not start.

A forward Phase 5 harness audit also narrowed wheel, sdist, and packaged-GUI import exclusions from
the whole repository to the checkout source package, while positively requiring wheel/sdist imports
inside their isolated environments. This correction is source only and still needs the fresh
user-executed combined run.

The combined harness now retains an early executable-resolution preflight for Python, Docker,
NVIDIA-SMI, and Trivy. This is routing evidence only, but it prevents known missing scanner tooling
from being discovered after lengthy Phase 4 and image-build work.

Owner evidence `phase4-20260812-195901` passes the complete 32-result Phase 4 development contract
for its retained dirty source identity. Subsequent Phase 5 typing corrections changed source, so it
is a successful checkpoint rather than proof of the current tree. Phase 5 evidence
`phase5-20260812-201822` passes five gates
and fails typed trust boundaries before tests, distributions, Phase 5 images, scans, or release
preparation. The two typing defects are corrected in source: PyYAML remains runtime-validated across
a narrowly annotated import, and all final-record JSON inputs must decode to mappings. No combined
pass, clean final source, approved scope, RC, or final release exists yet.

Combined attempt `phase4-phase5-20260812-202511` passed executable preflight and environment capture
but failed Phase 4 `02-version`. The failure was confined to an obsolete exact status-string
contract in `verify_active_version.py`; every other version check passed. The verifier now binds the
current revalidation-pending status and explicitly validates seventh/eighth attempt history. Phase 5
did not start, and the correction remains unvalidated until the next owner combined run.

Combined attempt `phase4-phase5-20260812-202852` passed 32/32 Phase 4 commands and reached Phase 5.
The Phase 5 child `phase5-20260812-204823` passed 23 commands through isolated sdist smoke before
`22-cpu-build` failed because Docker Desktop's classic image store does not support the required
local-image provenance/SBOM attestations. Supply-chain requirements were not relaxed. The ignored
combined wrapper now resolves Docker driver status and requires
`io.containerd.snapshotter.v1` during preflight, binding the environment result into the retained
summary before any phase starts. Fresh owner validation remains pending after enabling Docker
Desktop's containerd image store and restarting it.

Combined attempt `phase4-phase5-20260813-000340` supplies the previously pending combined
development evidence: Phase 4 passed 32/32 and Phase 5 passed 41/41 with matching recorded source
identity and no policy, protected-case, publication, RC, or final-release operation. The separate
release-policy-scope and final-authorization rows remain open.

Phase 6 traceability is now routed through `PHASE_6_NEW_CHAT_PROMPT.md` and the active goal. Required
implementation rows are: registry-generated ribbon/navigation; compact contextual editors; central
result/preview documents; truthful activity/log/progress/device/policy status; disabled-primary and
constrained-layout corrections; an enabled navigation entry that opens but never starts the
independent new-policy plan/check/start workflow; and native Windows setup/start separation without
Docker. The passing ignored Phase 6 bundle supplies automated development evidence; automation
cannot infer human usability, screen-reader, or scientist acceptance.

Phase 6 implementation rows are now source-complete and route through the exact continuation prompt:

| Phase 6 requirement | Implemented source | Current evidence / remaining gate |
|---|---|---|
| One ribbon command authority | `command_registry.py`; registry-generated `ribbon_bar.py`; `main_window.py` adapters | `phase6-20260813-032036`: unit, GUI, and render checks passed |
| Compact shared-state inputs | `context_pane.py`; copied full-config validation before state replacement | Phase 6 GUI contracts passed 6/6 |
| Central documents and activity | `document_workspace.py`; `activity_center.py`; `global_status_bar.py` | Truthful determinate/indeterminate and configured/actual checks and renders passed |
| Independent new-policy entry | `independent_training_panel.py`; built-in `tsh_calo_schema.py` architecture; hidden disabled legacy action; exact readiness fingerprint | Historical navigation evidence passed; built-in-architecture follow-up awaits current-source validation; no lifecycle transition executed |
| Responsive accessible shell | light/dark disabled-primary selectors; versioned docks/ribbon layout; F6 region cycle | Automated light/dark/constrained evidence passed; human acceptance remains uninferred |
| Native and packaged operation | `Launch-CALO-RPD.ps1`; `calo-rpd-native`; native guide; Phase 6 distribution verifier | Fresh wheel/sdist and Phase 6 package-membership checks passed; Docker unchanged |
| Proportional retained evidence | unit/GUI/integration/render sources; ignored `Validate-Phase6.ps1` and instructions | `phase6-20260813-032036` passed 19/19 with a hashed retained bundle |

The post-validation training-interface follow-up removes scientist-selected engineering receipts
and the erroneous separate-foundation status. The existing application-owned TSH-CALO schema
supplies the approved A–E architecture automatically, with optional E and experimental F disabled
by default. Fresh plans
bind to the current source identity; imported plans supply scientific settings only. Current-source
validation is pending; the earlier 19/19 bundle remains historical.

All scientific equations/accounting, A-E approval, F-off default, CUDA-preferred/CPU-only modes,
Safe-80 ceilings, zero-policy fallback, lifecycle authority, persistence, and release gates remain
unchanged. Phase 6 source completion is not validation, policy authorization, RC, publication,
release readiness, or final release.

The first two owner validator attempts, `phase6-20260813-031026` and
`phase6-20260813-031046`, supply only pre-command harness evidence. Both failed during Windows
PowerShell environment-architecture capture before Phase 6 command `01`; both preserve identical
nonignored source state and record no policy, Docker, CUDA-campaign, protected-case, publication,
release, or inferred-human-acceptance work. The architecture lookup is corrected in the ignored
validator, but current-source Phase 6 validation remains missing until a fresh complete rerun.

The fresh consolidated run `phase6-20260813-032036` subsequently passed 19/19 checks and preserved
identical before/after nonignored source-status SHA-256
`beabf2d918c4717e61e3b9c12ba449fdf2e59a38ccd549da54e2fb74cbabe9bf`. It closes the Phase 6
automated GUI/native/packaging development gate only. Its retained summary records every policy,
protected-case, Docker, CUDA-campaign, publication, and release execution field false; separate
qualification and release gates remain open.

Post-validation visual traceability is retained in `phase6-panel-sweep-20260813-041700`: all 16
workspace panels and four shell states were rendered and visually inspected. The authoritative
command registry now includes the complete Workspace ribbon palette; the left pane has one hidden-
tab input surface and no navigator; and `policies.training` performs the readiness/start state
transition through the same action while retaining exact-input fingerprinting, explicit start
confirmation, and unqualified/inactive output. Targeted GUI contracts passed 7/7 without executing
training or any policy/scientific lifecycle operation.

The focused training-input follow-up keeps that navigation contract but corrects its content:
after a valid frozen plan is loaded, the input-only dock presents campaign/cases, member count,
population, evaluation horizon, compute/fallback, PPO, and model controls. Any edit is included in
the exact readiness fingerprint and a non-overwriting hash-addressed launch plan. Focused validation
is pending; no legacy trainer or automatic lifecycle transition was restored.

The left Inputs dock and expanded ribbon are now invariant shell regions: no close/toggle/compact
command remains, and layout version 3 rejects older hidden states. `DocumentWorkspace` provides a
scrollable 920x650 minimum scientific canvas with a taller CALO-RPD-branded tab header so preview
blocks retain useful dimensions under constrained windows. Focused render validation is pending.

The policy-training case input now derives from the bundled case catalog instead of free text:
`case30` and `case57` are automatically selected by `All eligible bundled cases`, and the protected
`case118`/`case300` holdouts are visible, disabled, and excluded. Loaded eligible custom development
cases remain plan-bound selections. `application_icon()`, native-tab-base suppression, explicit
`QMainWindow::separator` styling, and simplified dock/Activity boundaries address the observed
blank Windows icon and warm native separator lines. Focused owner validation remains pending.

The current two user paths, compatible-resume choice, and all applicable campaign/compute/PPO controls have an accessible
`i` affordance with hover, click, keyboard-focus,
accessible-name, and accessible-description coverage. Directional explanations distinguish
evidence volume, PPO aggressiveness, architecture capacity, compute fallback, and identity-only
changes. Internal authority paths are not user controls. Focused validation is pending.

The current policy/GUI consolidation adds these pending current-source rows:

| Follow-up requirement | Implemented source | Current evidence / remaining gate |
|---|---|---|
| Independent policy-training resume only | `resume_center_panel.py`; `main_window.py`; `context_pane.py`; `independent_training_panel.py` | Source routes records only to independent prefill; focused tests written; owner validator not run |
| No dormant legacy training tree | simplified `calo_intelligence_panel.py`; stale close callback removed from `main_window.py` | Static source contract written; current-source execution pending |
| Single policy import | `calo_intelligence_panel.py` | One visible import action and focused count contract; execution pending |
| Clean resume inspection | `resume_center_panel.py` | Raw state JSON removed; focused summary/details contract written; execution pending |
| Contained ribbon category paint | button-based `ribbon_bar.py`; light/dark category-button selectors | No native category tab painter remains; visual and automated validation pending |
| Standard scientist-facing errors | `gui/user_feedback.py` plus scientific panel adapters | Exceptions log to Activity and short modal/status copy is implemented; current-source validator pending |
| Product-facing scientific language | `version.py`; `policy_readiness.py`; ribbon/status/navigation; Dashboard; CALO Intelligence; training, benchmark, algorithm, experiment, and power-system panels | Ordinary UI hides proposal, phase, build-stage, development/candidate, feature-flag, ABI, checksum, and source-authority terms; exact internal records and gates remain intact; focused owner validation pending |
| Product/internal version separation | `version.py`; `ACTIVE_DEVELOPMENT_STATUS.json`; `verify_active_version.py`; Phase 6 command/native and v12 status contracts | Owner run `phase6-20260813-183722` failed command `05` only after six passing prechecks because `product_version` was absent; record/verifier/test correction implemented; fresh owner rerun pending |
| Current Phase 6 source hygiene | `experiment_manager.py`; `main_window.py`; algorithms, portfolio, statistics, and context GUI imports; active-status/verifier attempt history | Owner run `phase6-20260813-184612` passed eight checks through compilation and failed `07-ruff` on seven unused names/imports; mechanical corrections implemented; command `08` onward and full rerun pending |
| Current Phase 6 formatting | Exact 27-file set from retained `phase6-20260813-185633/commands/08-format.txt`; active-status/verifier history | Owner run passed nine checks through Ruff and failed `08-format`; deterministic Ruff formatting applied to exactly all 27 reported files; commands `09` onward and complete owner rerun pending |
| Current Phase 6 unit contracts | `test_phase6_command_and_native_contracts.py`; `test_v120_phase3_gui_contracts.py`; active-status/verifier history | Owner run `phase6-20260813-190343` passed ten checks through format and command `09-unit` reported 57 passed/2 failed; both failures were stale literals now aligned to the intentional single-document header suppression and product-facing `Method verification` tab; fresh complete owner rerun pending |
| Interrupt-safe bounded Phase 6 GUI validation | isolated `SessionRecoveryJournal` and suppressed external startup probes in `test_phase6_ribbon_workspace.py`; two-minute per-test watchdog; verbose command `10`; exact expected-command sequence in ignored `Validate-Phase6.ps1` | Owner run `phase6-20260813-191340` passed 11 checks including all 59 unit tests, then command `10` was interrupted after about 54 minutes and was falsely summarized PASS; that PASS is rejected, the hang/summary paths are corrected without changing production behavior, and commands `10` onward require a fresh owner run |
| Focused GUI teardown isolation | pytest-qt `before_close_func` in `test_phase6_ribbon_workspace.py`; Activity handler detach; test-only direct close acceptance | Owner run `phase6-20260813-202657` passed commands `01`-`09`, all 60 unit tests, and 18/21 command-`10` GUI tests before code-124 teardown timeout after test 18 passed; focused cleanup no longer enters production finalization, production close safeguards remain unchanged, and commands `10` onward require a fresh owner run |
| Product-facing memory and empty-portfolio feedback | `global_status_bar.py`; aligned GUI assertion; guarded `PortfolioManagerPanel.refresh_plan`; focused static contract | Owner run `phase6-20260813-205516` passed commands `01`-`09` and completed all 21 command-`10` tests with 20 passed/1 stale-wording failure; safety-limit wording is now asserted, expected empty selection is a clean prompt rather than ERROR noise, and commands `10` onward require a fresh owner run |
| Offscreen Base architecture help parity | `validate_phase6_gui_contracts.py`; `test_phase6_command_and_native_contracts.py`; active-status/verifier history | Owner run `phase6-20260813-212634` passed commands `01`-`12` and first failed `13-gui-render` because the renderer omitted `architecture` from its 18-key help expectation; aligned with exact mismatch diagnostics, commands `13` onward require a fresh owner run |
| Visible policy-training execution action | `context_pane.py`; `main_window.py`; `command_registry.py`; light/dark themes; focused GUI/static/offscreen contracts | Owner run `phase6-20260813-215626` passed all automated Phase 6 checks, but manual inspection found that readiness/start labels only replaced the ribbon command. The ribbon is now navigation-only and a persistent footer below the scrolling inputs visibly presents gated `Check readiness` then `Start training`; fresh owner validation pending |
| Visible exact-resume choice and coherent output selection | `context_pane.py`; `independent_training_panel.py`; training campaign exact-resume backend; focused GUI/static/offscreen contracts | Manual start review found that the existing-output dialog referenced a hidden resume option and fresh Browse selected an already-existing directory. The canonical resume checkbox is now visible/off by default; new output selection proposes a non-existing campaign child, while resume selects an existing interrupted directory and retains exact plan/status/checkpoint validation; fresh owner validation pending |
| Per-user resumable-model library | `TrainingModelLibrary` in `independent_training_panel.py`; `main_window.py`; `context_pane.py`; focused unit/GUI/offscreen contracts | Fresh TSH-CALO output defaults below OS-managed per-user app data; default and explicitly added directories are scanned for interrupted/running plan/status pairs; selection preserves the original directory and identity and turns on visible exact resume; no copying, auto-start, qualification, selection, or activation; fresh owner validation pending |
| Concise dirty-source readiness feedback | `IndependentTrainingPanel`; `ActivityCenter`; focused GUI/static contracts | The clean-source requirement remains fail-closed. Full process output is DEBUG-only in Activity Logs; status/Warnings explain uncommitted application changes and confirm no training started; fresh owner validation pending |
| TSH-CALO-only policy-training surface | `TrainingLaunchModel`; `IndependentTrainingPanel`; `TrainingPathEditor`; focused unit/GUI/offscreen contracts | Removed the inapplicable CALO/TSH-CALO training selector and mutable architecture input; every campaign/PPO/resume input now applies to TSH-CALO, while rule-based CALO remains unchanged in ordinary algorithm/experiment selection; fresh owner validation pending |
| Safe training-editor startup | `TrainingPathEditor`; Phase 6 static/offscreen contracts | Initial new-training selection occurs only after status and primary-action widgets exist, preventing the manual-launch pre-window `AttributeError`; fresh owner launch and validation pending |
| Safe-80 readiness/training parity | `preflight_tsh_calo_training_resources`; `train_tsh_calo --check`; `IndependentTSHCALOTrainer`; focused unit/GUI/offscreen contracts | Readiness uses the trainer's exact policy-shape estimate and CUDA/CPU admission and releases its temporary lease, so an over-limit configuration cannot become start-enabled; fresh owner validation pending |
| Transition-bound rollout memory | `TrainingLaunchModel._rollout_capacity`; `set_resource_design`; independent session accounting | Fresh plans retain at most the exact post-initial-population transition count, synchronized with evaluation/population inputs, instead of treating raw candidate evaluations as PPO states; evaluation budget and Safe-80 ceilings remain unchanged; fresh owner validation pending |
| Offscreen resource-contract repository identity | `validate_phase6_gui_contracts.REPOSITORY_ROOT`; owner run `phase6-20260814-003900` | Commands through compile passed; Ruff first found two uses of one undefined lowercase root, now replaced by the source-derived repository root; fresh complete owner rerun pending |
| Isolated model-library GUI contracts | `_window` fixture temporary `SettingsManager`; `phase6-20260814-004621` | Repeated Qt windows no longer inherit scan locations from an earlier validation directory; production registered-path persistence is unchanged; fresh full rerun pending |
| Product-facing resume integrity language | `TrainingPathEditor` resume information control; internal exact-resume backend | Visible help says saved-file integrity instead of checksum while exact internal checkpoint identity/hash verification remains unchanged; 22/24 GUI tests passed before correction; fresh full rerun pending |
| Completed current Phase 6 automated gate | `validation/logs/phase6-20260814-004927`; command `01`-`17` retained evidence | PASS: 73 unit, 24 focused GUI, 21 GUI-regression, and 9 integration tests; offscreen light/dark/constrained renders; build and distribution verification; nonignored source stable. No policy/scientific, protected-case, Docker, CUDA-campaign, publication, or release workflow executed; human acceptance is not inferred. |
| Completed-training and saved-policy discovery | `TrainingModelLibrary`; `TrainingPathEditor`; `CALOIntelligencePanel`; `MainWindow`; synchronized `SettingsManager`; focused unit/GUI contracts | Completed campaigns and manifest-bound ensemble candidates appear from the private default or explicitly added bounded scan roots. Newly added/output roots refresh immediately; a completed row uses the single explicit Import action and cannot resume/start. Discovery performs no automatic registration, qualification, selection, activation, or experiment binding. Focused Ruff/format, 18 command/native, and 25 GUI contracts pass; complete follow-up validator pending. |
| Globally visible checkbox indicators | `ModernSpinBoxStyle.drawPrimitive`; light/dark indicator sizing; Phase 6 source and offscreen render contracts | Every `QCheckBox` receives a palette-aware rounded boundary and surface plus vector checked/partial marks; hover, focus, pressed, and disabled states are explicit while radio buttons are unchanged. Light/dark state renders require measurable perimeter contrast. Source and ignored validator updates are complete; execution and complete follow-up validation are pending. |
| Truthful new-training recovery presentation | `TrainingPathEditor` Recovery stack; existing `IndependentTSHCALOTrainingCampaign._advance_session`; focused source/GUI/offscreen contracts | Fresh training visibly shows automatic recovery on, matching existing verified recovery-point writes after safely committed windows. The interactive exact-resume choice appears only for selected running/interrupted campaigns, completed campaigns cannot resume, and fresh starts never receive `--resume`. No training/checkpoint/accounting or lifecycle semantics changed; contracts and ignored validator are updated but unexecuted. |
| Current completed-output, checkbox, and recovery follow-up validation | `validation/logs/phase6-20260814-131637`; `ModernSpinBoxStyle`; `TrainingPathEditor`; composited `validate_phase6_gui_contracts` evidence | PASS: complete `01`-`17` sequence; 74 unit, 25 focused GUI, 21 GUI-regression, and 9 integration tests; light/dark unchecked, checked, partial, focused, and disabled border renders; fresh wheel/sdist; distribution verification; hashes; stable nonignored source. No training, policy/scientific lifecycle, protected-case, Docker, CUDA-campaign, publication, or release workflow executed; no human acceptance inferred. |
| Latest retained full Phase 6 pass, superseded for corrective follow-up | `validation/logs/phase6-20260814-132200`; validation summary; source-status inventory; source manifest | PASS: complete `01`-`17` sequence with the same 74 unit, 25 focused GUI, 21 regression, and 9 integration counts plus render/build/distribution/source-stability evidence. Historical only: its recorded source omits the later registry/database/extension implementation and predates the current selection-scroll, first-row deletion, and architecture/training-parameter compatibility contracts. Fresh owner validation remains required. |
| Detailed finite-plan training progress | `IndependentTSHCALOTrainingCampaign._record_event`; `_checkpoint_progress`; `training_events.jsonl`; `train_tsh_calo.emit_training_event`; `IndependentTrainingPanel`; `GlobalStatusBarWidget`; `ActivityCenter` | Structured committed-checkpoint events include exact finite total/committed candidate evaluations, member/case/session/transition identity, and checkpoint hash; GUI retains bounded raw output, shows one percentage scale in global status, and keeps detail in Jobs/Logs. Tests/validator written but unexecuted; fresh owner evidence pending. |
| Cooperative checkpoint-safe pause | `request_tsh_calo_training_pause`; `_honor_pause_after_checkpoint`; `TaskStatus.paused`; GUI pause actions and pause-receipt verification | Request is idempotent and bound to campaign/plan; acknowledgment occurs only after durable checkpoint commit with no uncommitted window. Exit code `75` is accepted as paused only with a matching status receipt. Forced interruption remains fail-closed. Synthetic contracts written; owner validation pending. |
| Unlimited resume cycles with finite evaluation budget | Immutable execution-plan hash and exact-resume checks; repeated-pause deterministic-parity unit contract; status/GUI help text | No resume-count counter or ceiling exists. Each resume continues the same seeds, checkpoint, and exact finite candidate-evaluation total; pause cannot reset or extend the plan. Infinite-budget training is intentionally absent. Contracts are unexecuted pending the complete owner validator. |
| Authenticated completed-model continuation state | Base completion manifest; per-member final `.resume` checkpoints; trainer/session serialization | Each member binds model and optimizer state, NumPy/Torch RNG, PPO updates, all receipts, device/memory provenance, session/environment and collector state, and exact accounting by path/SHA-256/receipt count. Legacy manifests without this complete binding are non-extendable. Synthetic tests written; owner validation pending. |
| Explicit repeatable finite extension segments | `tsh_calo_training_extension.py`; `train_tsh_calo --extend`; architecture/parameter compatibility contract; policy parameter-layout signature; completed `TrainingModelLibrary` record; `TrainingPathEditor` extension readiness/start action | Each child authenticates the unchanged base plan and parent manifest, continues every member under identical case/seed/configuration conditions, has an exact finite additional FE budget, writes separate unqualified artifacts/checkpoints, and may parent another child without a count ceiling. A different software source commit is retained as execution provenance and does not itself block extension. Campaign/source/freeze identity plus reserved writer metadata are excluded from training-parameter schema authority in both GUI readiness and CLI extension parsing; the campaign schema remains binding to training semantics. Changed architecture/resume schemas, parameter tensor names/shapes/dtypes, exact training values, or added/removed training-parameter fields fail closed. A completed legacy-authority campaign reuses its authenticated embedded hashes without requiring historical files to be reselected. No automatic extension or parent mutation. Validation pending. |
| Extension scientific and lifecycle boundary | Extension manifests, GUI confirmation/help, active status, deterministic/safe-pause contracts | Additional optimization may improve, plateau, overfit, or degrade a model; no strength, superiority, qualification, registration, activation, selection, or experiment binding is inferred. Independent qualification remains separate. |
| Nonduplicated training progress | `TrainingPathEditor`; `GlobalStatusBarWidget`; `ActivityCenter`; constrained offscreen/source contracts | One percentage scale remains in the bottom bar, checkpoint detail remains in Activity, and the left footer retains only concise state plus safe pause. Implemented; owner validation pending. |
| Left-pane content containment | Responsive size policies for training library/path rows, context editor host, and stack; constrained 1120x720 offscreen geometry check | Horizontal overflow is prohibited and path controls shrink within the permanent input dock. Implemented; owner render validation pending. |
| All completed campaigns in Policy library | `TrainingModelLibrary.completed_campaigns`; merged completed/registered rows in `CALOIntelligencePanel` | Valid and attention-state completed campaigns remain visible; import is explicit and a completed campaign remains one row after registry import. No lifecycle authority is inferred. Validation pending. |
| Guarded physical completed/standalone-model deletion | `TrainingModelLibrary.validate_completed_campaign_deletion`; `delete_completed_campaign`; `PolicyRegistry.remove_unqualified_candidate`; atomic database removal; `CALOIntelligencePanel._delete_completed_training`; `_delete_standalone_policy_file`; synthetic unit/GUI contracts | Exact confirmed deletion accepts an unregistered campaign or one inactive unqualified unreferenced SHA-valid registration, including the first standalone library row. Active, qualified, evidence/reference-bearing, sibling-registered, scan-root, symlink, incomplete, and undiscoverable targets are rejected. Validator performs no deletion. |
| Qualified policy activation and Power System handoff | Existing `PolicyRegistry.activate` and immutable binding checks; `CALOIntelligencePanel`; `MainWindow._governing_policy_event` | Activation remains qualified/compatible/integrity/receipt gated. Apply binds the ready policy and navigates to unlocked Power System without starting scientific work. Validation pending. |
| Entry-sized scrollbar-free Policy library | `CALOIntelligencePanel._resize_policy_table_to_entries`; table scrollbar and size policies; refresh/resize hooks; GUI/offscreen contracts | Table height is header plus every visible row, with no nested horizontal or vertical scrollbar. The main preview owns overflow for the full page. Implemented; owner validation pending. |
| Full-width Governing policy form | Expanding policy groups, `QFormLayout.AllNonFixedFieldsGrow`, expanding policy/status fields | Policy library and Governing policy occupy the available content width; governing controls no longer stop at their size hint. Implemented; owner render validation pending. |
| Input-generated new-training plan | `TrainingPathEditor._run_primary_action`; `TrainingLaunchModel.create_plan`; visible campaign/case/member/resource/compute/PPO inputs | New training has no external plan path. Readiness constructs the full internal plan from visible values plus application-owned safe schema/resource/provenance/resume defaults without starting work. Synthetic GUI/offscreen contracts written; owner validation pending. |
| Minimal managed training campaign UI | `context_pane.py`; `IndependentTrainingPanel._configuration_changed` | Add to path, the managed default path, Settings template, and Import settings are absent. Refresh and Training directory remain; saved plans load internally only for authenticated resume/extension. Owner validation pending. |
| Vanished saved-campaign reset | `TrainingPathEditor.refresh_model_library`; `_select_new_training` | When Refresh loses the selected campaign, index-zero fallback clears its stale plan/error and creates a new campaign/output identity instead of retrying a missing JSON file. Synthetic regression written; owner validation pending. |
| Launch-independent fresh-plan provenance and truthful errors | `TrainingLaunchModel._current_source_commit`; `TrainingPathEditor.refresh` | A checkout source identity is resolved from the imported package root, not an arbitrary native-launch working directory. Fresh generation failures and retained saved-plan load failures have distinct UI states. Synthetic unit/GUI/offscreen contracts written; owner validation pending. |
| Explicit model removal | One usable `Delete model files` action; identity/reference/qualification guards; retirement CLI retained outside the ordinary library UI | Unregistered completed campaigns and narrowly eligible inactive, unqualified, unreferenced completed or standalone candidates expose confirmed exact deletion. Active, qualified, referenced, or otherwise governed records remain blocked; the separate inventory/dry-run CLI remains available when explicitly authorized. No automatic or validator deletion. |
| Reachable Governing policy bottom without selection jump | `ScrollablePage` height synchronization; CALO bottom clearance; main-preview outer scroll; constrained contracts | Dynamic Policy Library height propagates through the stack, so manual outer scrolling reaches the complete Governing policy group above Activity. Model selection never repositions the page; nested policy-table scrolling remains prohibited. Implemented; owner validation pending. |
| One-action in-library policy qualification workflow | `prepare_automatic_source_snapshot`; `qualification_candidate_contract`; `tsh_calo_automatic_qualification.py`; `CALOIntelligencePanel.qualify_selected_policy`; snapshot-rooted `qualify_tsh_calo`; `PolicyRegistry.inspect_qualification_evidence`; atomic `ResultDatabase.admit_verified_policy_qualification` | One explicit `Qualify policy` action creates a deterministic clean internal snapshot without modifying the live worktree, freezes the exact architecture/training-parameter/model contract, and starts or exactly resumes only the fixed 120-cell paired quality campaign. Product version and development-stage labels are provenance, not gates. Durable completed cells drive real bottom-bar percentage and support unlimited exact resumes without changing the finite seeds, plan, or FE budget. Verified passing evidence is admitted automatically but never activates or binds. Exact contract/checkpoint/design/seed/evidence/receipt/calibration hashes, zero-failure completeness, protected-case closure, and canonical gate recomputation fail closed. Retained legacy component-bound evidence remains readable; owner validation pending. |
| Comparable-policy selection guidance | `qualification_comparison_protocol`; conservative evidence summaries; `pareto_dominates`; Policy comparison dialog | Direct comparison is restricted to matching cases, paired seeds, FE/population budgets, analysis/calibration definitions, thresholds, anytime design, and the stage-neutral candidate architecture contract. A strongest label requires one policy to dominate every comparable policy across conservative feasibility, improvement, win-rate, effect, anytime, and Holm measures. Otherwise the UI requires scientist judgment. Source version, age, and training duration are not quality evidence. Implemented; owner validation pending. |
| Foreground-run interaction and observability | `MainWindow._apply_task_interaction_lock`; `TaskStatus`; `GlobalStatusBarWidget`; `ActivityCenter`; `CALOIntelligencePanel._update_qualification_progress` | Every foreground run disables the ribbon, Inputs, and document workspace while leaving Activity and the global status bar enabled. Qualification reports committed retained cells as exact percentage and cell count; detailed output remains in Activity. Synthetic GUI/static contracts written; owner validation pending. |
| Cumulative exact model training evaluations | `count_tsh_calo_candidate_training_evaluations`; authenticated per-member episode receipts; candidate SHA; `PolicyRegistry.training_evaluation_count`; Policy Library column | Reports completed candidate evaluations used to produce the exact model. Each completed extension child includes base plus all of its completed finite segments; the parent remains unchanged. Qualification and experiment evaluations are excluded, and unverifiable legacy accounting shows `Not available`. Synthetic contracts written; owner validation pending. |
| Stage-neutral formal qualification admission | `PolicyRegistry.inspect_qualification_candidate`; `CALOIntelligencePanel._qualification_candidate_blocker`; exact candidate SHA; native TSH-CALO artifact inspection | Development phase, software revision, and missing historical acceptance receipts do not decide quality eligibility. The exact immutable artifact must prove the current ABI, ensemble structure, member architecture/provenance, authenticated training receipts, feature flags, and protected-case isolation before formal evidence can be admitted. Synthetic contracts written; owner validation pending. |
| Independent resume/extension compatibility | `tsh_calo_training_compatibility_contract`; parameter-layout and training-schema hashes; extension checkpoints | Qualification admission does not weaken exact continuation. Resume/extension require compatible architecture, parameter names/shapes/dtypes, persisted training fields and state, and evaluation accounting. Software revision is provenance only. Synthetic contracts written; owner validation pending. |
| Qualification micro-step observability | `TSH_CALO_QUALIFICATION_EVENT_SCHEMA`; `qualification_events.jsonl`; campaign progress callback; `CALOIntelligencePanel._apply_qualification_event`; `ActivityCenter`; `GlobalStatusBarWidget` | Every 500 formal-cell evaluations emits case/run/side, exact live FE, feasible objective, violation, first-feasible FE, throughput, and ETA. The bottom bar shows live overall/current-cell progress while naming durable cells separately; in-cell data is never mislabeled as committed evidence. Synthetic contracts written; owner validation pending. |
| Qualification checkpoint-safe pause and exact partial-cell resume | `request_tsh_calo_qualification_pause`; per-cell authenticated exact-run envelopes; qualification status/control receipts; exit code 75 verification | Pause latches only after a complete population transition, acknowledges only a checksum-bound checkpoint or cell record, and retains no lifecycle authority. Resume restores exact optimizer/RNG/history/accounting state under the same frozen plan. Pause count is unlimited; finite seeds, cells, and FE budget are immutable. Synthetic contracts written; owner validation pending. |
| Transactional qualification cell evidence | `TSHCALOQualificationCampaign._commit_terminal_cell`; canonical `qcell-*` identity; `qualification_cell_index.json`; infrastructure-incident status | Exactly one checksum-bound success or scientific-failure terminal artifact is permitted per planned case/run/side/seed/budget identity. Telemetry, status, callback, index, or evidence-write faults are infrastructure failures and cannot rewrite scientific disposition. Duplicate identities and success/failure collisions fail closed. Fault-injection tests and ignored validator written; owner execution pending. |
| Qualification completion and admission authority | `qualification_completion.json`; `policy_qualification_admission._verify_transactional_completion`; current evidence schema v3 with explicit legacy v2 reader | Admission requires the final plan/seed/index/event/status/evidence/receipt binding, 120 unique successful cells, exact FE, zero scientific failures, zero infrastructure incidents, protected-case closure, and canonical gate recomputation. A stray receipt/evidence file has no authority. Owner validation pending. |
| Corrected-source fresh qualification after retained incident | `inspect_tsh_calo_qualification_resume_state`; `frozen_qualification_restart_design_sha256`; `CALOIntelligencePanel._retained_qualification_resume` | The contradictory `e266bd7598befa54` campaign is detected read-only, remains byte-for-byte retained, and cannot resume or admit evidence. The next workflow creates new run/source provenance only after every operative frozen design field matches: exact candidate, cases, paired seeds, population, FE budget, analysis/OOD/thresholds, and lifecycle boundaries. No new qualification may launch before owner validator review. |
| Non-decisional feasibility assessment | `tsh_calo_feasibility_assessment.py`; transactional campaign finalizer; `inspect_feasibility_assessment`; `admit_verified_policy_assessment` | Current-source evidence reports exact full-feasibility percentage plus separate reach, efficiency, independent-validation, objective-coverage, and case ratings. No automated suitability decision or grade is emitted. Candidate/plan/cell/receipt/OOD identities remain fail-closed. Synthetic contracts written; owner validation pending. |
| Scientist selection separated from assessment and activation | `ResultDatabase.record_scientist_policy_selection`; `PolicyRegistry.select_assessed_policy`; `CALOIntelligencePanel.select_policy_for_use`; strict experiment binding | Assessment produces inactive `assessed`; exact explicit selection produces inactive `scientist_selected`; activation remains separate and re-verifies retained evidence. Legacy qualified policies remain readable. No automatic registration, selection, activation, or experiment binding occurs. Validation pending. |
| Connected feasibility and training-influence UI | Feasibility and Training-parameter influence groups below Governing policy; selected candidate SHA and rating schema | The selected model shows individual percentage ratings and tooltips. The influence block shows immutable training inputs and, with at least three comparable assessed campaigns and two distinct values, the strongest per-rating standardized univariate association. Otherwise it reports insufficient evidence. No causal or tuning claim and no training/architecture mutation. Validation pending. |
| Completed-assessment Influence handoff and reachability | `CALOIntelligencePanel.refresh_policy_library`; `_select_policy_id`; `_parsed_training_plan_result`; `_resize_evidence_table_to_entries`; `_reveal_influence_analysis`; focused GUI/offscreen contracts | Rebuilding or refreshing the library recomputes evidence for the same selected row; admission reselects the exact policy and reveals Influence. Immutable selected-plan values remain visible with one assessed campaign while associations truthfully remain unestimated. Plan/report errors are explicit, and the outer page scroll reaches every evidence row. Implemented; fresh focused and Phase 6 owner validation pending. |
| Single-row Policy Library actions | `CALOIntelligencePanel` action layout and eligibility visibility; Phase 6 static/GUI contracts; `feasibility-influence-20260815-215306` | Compare, archive, and Show archived controls are absent. The visible library and scientist-facing influence cohort request non-archived records only. Activation is hidden until scientist selection or legacy qualification makes it eligible, so no disabled `Scientist selection required` placeholder is shown. Import, assess, select, eligible activation, guarded delete, and refresh share one row. Historical registry compatibility remains internal for governance/deletion protection; no record is deleted automatically. Focused lint/format/unit/GUI/source-stability validation passed; broader scientific and release gates remain separate. |
| Windows PowerShell native-stderr validator handling | `Validate-Qualification-Evidence-Transactions.ps1`; returned `feasibility-influence-20260815-203150` | First run stopped at `02-diff-check` because PowerShell 5.1 promoted Git's informational line-ending stderr to a terminating error. Recorder now retains stderr and uses the native exit code as command authority. Ruff/tests did not run in the returned bundle; fresh complete validation pending. |
| Feasibility/influence Ruff follow-up | `test_tsh_calo_policy_lifecycle.py`; returned `feasibility-influence-20260815-203426` | Diff-check completed with exit 0; Ruff found one missing checksum-helper import reported at two call sites in a synthetic legacy-policy fixture. Import corrected; production feasibility/influence source was not implicated. Later commands and full validation remain pending. |
| Transactional feasibility/influence focused validation | `validation/logs/feasibility-influence-20260815-204719`; `validation-summary.json`; `validation-log-sha256.txt`; source before/after inventories | PASS: Ruff and format, 60 focused unit/fault-injection contracts, 36 focused GUI contracts, all six exit codes zero, and byte-identical nonignored source status. Deterministic synthetic fixtures only; no real campaign, training/parameter mutation, policy selection/activation/binding, protected case, Docker, scientific claim, or release evidence. A final replay after ledger closure binds the documented source state. |
| Explicit fresh/resume feasibility intent | `inspect_verified_paused_automatic_qualification_workspace`; `discard_incomplete_automatic_qualification_workspace`; `CALOIntelligencePanel.resume_selected_assessment`; Start fresh assessment and conditional Resume assessment buttons; focused unit/static/GUI contracts; `feasibility-influence-20260815-215306` | Resume is exposed only for exactly one authenticated candidate/plan/source/checkpoint-bound safe pause. Fresh assessment allocates a new identity and, after confirmation, deletes only canonical incomplete/resumable candidate workspaces; it never merges their cells. Completed evidence, source snapshots, integrity failures, and immutable infrastructure incidents remain protected. Focused replay passed 61 unit/fault-injection and 37 GUI contracts plus Ruff/format/source stability; final source-bound authority is the newest complete PASS produced after ledger closure. |
| Live GUI state without relaunch | `MainWindow._refresh_workspace_from_source`; `TrainingModelLibrary.refresh`; `TrainingPathEditor`; `CALOIntelligencePanel`; `ResultsExplorerPanel`; focused unit/GUI/offscreen contracts | Results Refresh reloads experiments, the shared training/policy refresh invalidates cached file observations and rescans, and workspace activation reloads applicable read-only state. Implemented; fresh owner Phase 6 validation pending. |
| Retired Resume Center and minimal Home | schema-4 `workspaces.py` migration; `command_registry.py`; `MainWindow`; `UnfinishedWorkDialog`; Phase 6 contracts | Resume Center is not constructed or navigable; historical identities map to Overview without shifting Settings/Benchmark. Policy resume/extension remains in Train policy with readiness and explicit start. Home is exactly Overview/Open/Save. No retained resumable data was deleted. Implemented; fresh owner validation pending. |
| Algorithms-first ribbon entry | `RIBBON_CATEGORY_ORDER`; `RibbonBar`; Algorithms-first `SETUP_STEPS`; `WorkflowManager.workspace_state_key`; `test_phase6_ribbon_workspace.py`; `test_guided_workflow.py`; offscreen GUI validator | Home is followed immediately by Algorithms. Algorithms, CALO Intelligence, and CALO Settings are enabled at first launch; the guided setup starts with Algorithms and entering it changes only the visible workspace. A valid submitted stage immediately enables Portfolio, Workspace Study, and Individual experiment navigation, while Reset Selection locks them again. Navigation does not satisfy or bypass page-level Apply, Audit, Stage, controller, or Run prerequisites. The internal TSH-CALO experiment option remains policy-gated. Implemented; fresh owner Phase 6 validation pending. |
| Experiment-bound Algorithms and CALO/TSH-CALO settings | `AlgorithmsPanel.content_stack`; `submit_algorithm_selection`; `reset_algorithm_selection`; `save_calo_settings`; `stage_completed`; `stage_discarded`; `ExperimentConfig.algorithms`; `ExperimentConfig.algorithm_parameters`; config-roundtrip, Phase 6 GUI, and offscreen contracts | Algorithms and CALO settings are separate selection/settings surfaces. Algorithm edits are drafts; Submit replaces the staged identities/parameters and completes the step, while Reset clears staging/selections, restores comparator defaults, invalidates downstream setup, and preserves CALO/TSH-CALO settings. Settings Save cannot stage algorithms. None starts work. TSH-CALO policy identity/checksum/feature flags/lifecycle remain read-only CALO Intelligence authority; selection stays policy-gated and unsafe fallbacks remain false. Implemented; fresh owner Phase 6 validation pending. |
| Algorithms registry content-fitted layout | `AlgorithmsPanel._fit_algorithm_table_to_entries`; `algorithm_registry_card`; main-preview scroll ownership; Phase 6 GUI/offscreen contracts | The registry table height equals its header, registered rows, and frame; the card does not stretch into unused page height. Nested vertical scrolling is disabled and the main preview owns overflow, preserving access to every row and staging action. Implemented; fresh owner Phase 6 validation pending. |
| Focused duplicate-free Workspace ownership | `RIBBON_CATEGORY_ORDER`; `COMMAND_SPECS`; `RibbonBar`; Phase 6 command, GUI, and offscreen contracts | Workspace contains only Portfolio, Study, Validate, Bench, Export, and Settings, and those labels appear nowhere else. Overview/CALO/Power/ORPD/Methods/Scenarios/Live/Results/Stats have purpose-owned routes outside Workspace. Every visible label is unique; stable workspaces and prerequisite gates remain unchanged. Implemented; fresh owner Phase 6 validation pending. |
| Immutable Workspace and individual execution ownership | `experiments/execution_plans.py`; `portfolio/study_planning.py`; `app/execution_control.py`; `app/workspace_campaign.py`; schema-v3 `results/database.py`; Portfolio/Algorithms/Study panels; focused unit/integration/GUI contracts | Explicit Algorithms Submit persists the only active stage. Current Workspace v3 binds one AppliedPortfolioGoal, deterministic recommendation, applied selection/delta, non-empty subset, explicit cells, and queue count; legacy Workspace v1/v2 designs remain readable. Individual v2 binds the complete staged algorithms independently. Canonical hashes, Audit -> Stage -> Run, singleton fencing, safe pause/handoff, resume, and duplicate admission remain fail closed. Source implemented; schema-v29 owner validation pending. |
| Truthful Workspace draft/audit status | `ExperimentManagerPanel._invalidate_fairness`; signal-blocked shared-state hydration; `refresh_execution_state`; `test_workspace_execution_ui.py`; validator schemas v26-v28 | Completing setup does not bypass fairness. A never-audited draft reports that its fairness pass is required rather than claiming an earlier configuration changed. Programmatic refresh preserves truthful state; a real post-audit edit disables Stage immediately and requires re-audit. Audit receipt, immutable-plan, controller, and Run gates are unchanged. The correction passed in schema v27; schema v28 replays it against the current source. |
| Independent direct experiment versus automated Workspace study | `WorkflowManager.individual_completed`; `ExperimentManagerPanel`; `frozen_individual_config_payload`; `result_contracts.py`; `ExperimentWorker`; Individual v2 and Workspace v3 schemas; focused unit/GUI/integration/restore contracts | Experiment > Individual experiment owns its setup ledger, repetitions, exact verified reuse, direct audit/result contract, and immutable full-stage plan without consuming Portfolio or Workspace Study records. Workspace alone consumes the current Portfolio goal, Study recommendation/selection, outputs, case matrix, and automated cells through the same ExperimentManager. Controller exclusivity, Audit -> Stage -> Run, scientific semantics, and policy gates remain intact. Implemented in source; complete owner Phase 6 schema-v29 validation pending. |
| One-way Portfolio goal to Workspace Study | `PortfolioGoalPlanner`; `WorkspaceStudyPlanner`; `AppliedPortfolioGoal`; `AppliedStudySetup`; Workspace plan v3; schema-v3 `portfolio_goals` and `applied_study_setups`; Portfolio and Experiment Manager panels | Portfolio Apply persists only broad immutable intent and never changes exact runs or creates a plan. Workspace Study requires the exact current goal, exposes deterministic hard minima/recommendations, records scientist overrides, and creates/replaces only an unstarted draft on explicit Apply Study. New goals or stages invalidate unstarted downstream state; retained lifecycle plans remain immutable. Legacy combined rows are not relabeled. | Source implemented; schema-v29 owner validation pending; no scientific, usability, publication, or release claim |
