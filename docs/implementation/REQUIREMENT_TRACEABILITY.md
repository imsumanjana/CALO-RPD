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
| Preserve old populated databases through rollback-safe migration | Locally verified | SQLite schema v1, online pre-migration backup, integrity check, SHA-256 receipt, transactional DDL, future-version rejection; `tests/integration/test_database_workflow.py` | CI matrix execution |

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
| Stable grouped information architecture | `app/workspaces.py`; `gui/navigation/sidebar.py`; inline SVG registry | Corrected Windows contracts and all-workspace automation accepted | Linux xcb interaction evidence |
| Dashboard decision hierarchy | next-action/readiness/recent-work/activity disclosure in `dashboard_panel.py` | Windows light/dark/high-DPI and all-workspace evidence accepted | Linux xcb evidence |
| Seven-step Study Setup | `StudySetupWorkflow`; existing authoritative panels routed without state duplication | Windows programmatic keyboard/workflow interaction accepted | Linux xcb interaction evidence |
| Compact structured inputs | application-wide density policy; integer chips; 240-480px limits; responsive Results grid; dedicated copyable database-path field | Corrected Windows clipping/input-width automation accepted | Linux xcb all-workspace automation |
| Horizontal use of wide workspaces | shared `WorkspaceTabs`; side-by-side ORPD, scenario, portfolio, settings, and benchmark groups; stretching Portfolio output tree | Corrected `121530` Windows light/dark-200% tabs and tree-width gate accepted; visual review confirms no right-side waste or evidence truncation | Linux xcb per-tab screenshots and tree-width evidence |
| Accessible semantic visual system | named tokens, light/dark QSS, focus borders, text badges, accessible names/buddies | Windows high-DPI contrast/name/buddy/keyboard automation accepted | Linux xcb automated evidence |
| Render and glyph evidence | isolated render CLI; deterministic existing-OS-font registration; font provenance; separate glyph/clipping/input counts | Four corrected Windows cells accepted with zero glyph/replacement/clipping/input/editor failures | Linux light/dark xcb matrix |
| Durable local evidence identity | ignored validator; v2 source manifest with commit/dirty/status hashes; v3 workspace evidence and v4 Windows summary | Corrected `121530`: validator/status identities matched at review; 17/17 source entries and 79/79 evidence hashes matched; acceptance ledger postdates manifest | Source-bound Linux directory |

Phase 3 Windows-local baseline evidence is accepted. `phase3-20260807-045558` and
`phase3-remaining-windows-20260807-092741` remain immutable failed history;
`phase3-20260807-052047` is the accepted baseline rerun; and
`phase3-remaining-windows-20260807-112621` accepted the corrected pre-tabbed source at 10/10. The
new tabbed-layout source postdates that manifest and therefore required fresh noninteractive
Windows and Linux xcb evidence. Windows is now accepted in `121530`; Linux remains. Manual reviewer answers are excluded by user instruction,
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
work is complete; the remaining Phase 3 proof is the separate Linux xcb directory only.

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

## Current development-first policy boundary - 2026-08-07

The preceding candidate-related rows retain the scientific evidence requirements and immutable
negative history, but their execution timing is superseded by the user's development-first decision:

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
