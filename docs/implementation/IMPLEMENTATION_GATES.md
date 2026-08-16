# CALO-RPD modernization implementation gates

This file controls implementation of the remediation plan in
`docs/COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md`.
The ordered five-phase post-v6.9 execution plan is
[`POST_V6_9_RELEASE_UPDATE_AND_FIX_PLAN.md`](POST_V6_9_RELEASE_UPDATE_AND_FIX_PLAN.md).
The live requirement-to-evidence audit is
[`REQUIREMENT_TRACEABILITY.md`](REQUIREMENT_TRACEABILITY.md); it distinguishes local tests from
physical/external and scientific-campaign proof.

## Baseline

- Starting commit: `307402df5c7a44a6bb852770347b1b1ef995548d`.
- Starting branch: `main`; implementation branch: `codex/calo-complete-modernization`.
- The starting checkout already contained uncommitted v6.9 work. Those changes are treated as
  pre-existing work and must not be discarded or silently attributed to this modernization.
- The v6.9 freeze and `MANIFEST.sha256` describe a historical candidate snapshot. They must fail
  after a frozen file changes; they are regenerated only for a new verified release artifact.
- The requested audit Markdown is an analysis artifact, not proof that any implementation gate has
  passed.

## User-directed phase execution protocol

- Create a concrete phase-specific goal through the goal service before stating or starting each
  numbered development phase. No phase source work begins before the goal exists.
- Development turns are coding-only. The agent may write production code, test code, schemas,
  documentation, and validation automation, but must not execute manual-capable tests or validation
  unless the user later authorizes a specific named command.
- At the end of each phase coding pass, prepare or update a detailed PowerShell validator inside the
  Git-ignored `validation/` tree. The script must capture a new timestamped log directory, command
  outcomes, relevant source hashes, and its own identity. The agent does not run it.
- The user runs the validator and returns its complete log directory. The agent reviews that evidence
  read-only, makes only evidence-backed corrections, and prepares the next manual rerun when needed.
- Validators and their logs never enter Git, manifests, packages, container contexts, or release
  artifacts. Keep commands and evidence reads narrowly scoped to reduce token use.

## Change classes

### Class A — behavior-preserving engineering

May proceed without algorithm approval:

- tests and release-harness scoping;
- schema/code contract alignment where runtime behavior is already strict;
- container/build/CI infrastructure;
- XPU compatibility readers and removal of executable XPU paths;
- GUI terminology and presentation changes;
- configuration decomposition that preserves scientific fingerprints and values;
- memory admission, telemetry, resource leases, staging and explicitly governed backend fallback;
- policy lifecycle orchestration and immutable experiment binding;
- diagnostics and documentation that describe existing behavior truthfully.

Every Class A change still requires focused tests and compatibility evidence.

### Class B — scientific behavior or promotion semantics

Requires explicit user approval before implementation:

- operator allocation or operator equations;
- policy state/action space, architecture, reward or training transition;
- contextual-credit authority;
- HPEM, personal/contextual memory or variable-group behavior;
- environmental selection, adaptive epsilon, lane, precision or recovery behavior;
- qualification/effect-size/non-inferiority/AUC changes that can alter policy promotion;
- any change that makes an old CALO run scientifically non-equivalent under the same algorithm ID.

Diagnostic tests may expose a Class B defect before approval, but production behavior remains
unchanged until the proposal is approved.

## Gates

| Gate | Scope | Required evidence |
|---|---|---|
| G0 | Baseline and contracts | dirty-tree inventory, matching implementation branch, change classification |
| G1 | Foundational correctness | configuration contract tests, correctly scoped historical/current release tests, clean relevant lint/tests |
| G2 | Memory and fallback | shared GPU leases, 80% of free-at-admission tests, host available-RAM tests, OOM state-machine tests, parity/provenance |
| G3 | XPU removal | versioned migration fixtures, zero executable XPU paths, historical records viewable, CUDA/CPU regressions |
| G4 | Containers | hashed CPU/CUDA locks, non-root images, volumes, health checks, SBOM, CPU and physical-CUDA qualification |
| G5 | Experiment protocol | lossless migrations, one immutable global protocol, transactional application, fingerprint stability |
| G6 | Scientist GUI | no normal-view engineering/venue language, evidence wizard validation, headless interaction tests |
| G7 | Policy lifecycle | independent training state, central resource leases, every experiment bound to governing policy SHA |
| G8 | Algorithm proposal | alternatives, rationale, risks, versioning, ablations, falsification tests; explicit user approval |
| G9 | Approved algorithm | new algorithm version, parity boundaries, ablations, development/validation qualification |
| G10 | Scientific evidence | expanded cases/baselines/holdouts, power-aware runs, raw artifacts, independent validation |
| G11 | Release | complete requirement audit, staged artifact manifests, reproducible images, documentation and handoff |

## Current implementation status — 2026-08-04

This is an implementation ledger, not a release declaration. “Implemented” means the source and
focused tests exist; hardware-dependent gates remain open until they are executed on the target
machine/container. The historical v6.9 freeze and root manifest are intentionally stale during this
development branch and must not be regenerated before the final staged release audit.

| Gate | Status | Current evidence / remaining boundary |
|---|---|---|
| G0 | Complete; v12 correction recorded | Baseline commit and dirty-tree ownership recorded. At the user's direction, continuation work is committed directly on local `main`. `Docker_Build.txt` is now tracked. Generated `calo_policy_candidate` branch/artifact directories and `.codex-pytest-temp/` remain untracked user/runtime state and are excluded from source/release scope. |
| G1 | v12 Phase 1 coding implemented; user validation pending | Active package/runtime/CLI/GUI/README/container identity is `12.0.0.dev1` / `12.0.0-dev.1`; active status/index, the fail-closed version verifier, CI identity step and v12 contract tests exist. Historical v6.9 records remain immutable and its integrity test is release-scoped. No validation command was executed during this coding-only task; run the local-only, Git-ignored `validation/Validate-Phase1.ps1` harness and return its complete log directory. |
| G2 | Implemented; host and source-bound container physical evidence retained | Shared device lease, 80%-of-free VRAM/RAM admission, CUDA-resident/staged-host/CPU-fallback states and focused regression tests exist. Clean host commits `63f56ad`, `d6a950c` and `67bd18e` retain bounded FP64 case30/case57 parity, real VRAM pressure/recovery, host-staged CUDA execution, controlled OOM backoff, controlled clean CPU restart plus CUDA recovery, cross-process lease contention/release and a verified 3,600-second physical soak. Exact source-bound CUDA image `calo-rpd-studio:cuda-1f02a94` repeated case30/case57 parity, bounded resource recovery, cross-container lease exclusion/release and a continuous 3,600-sample GREEN soak with a verified 3,602-event chain. Controlled faults are explicitly not natural-hardware-OOM evidence; energy is GPU-board-only, not whole-system energy. Final-candidate and trusted-CI repetition remain G4/G11 evidence. |
| G3 | Implemented | New schemas expose only `cuda_preferred` and `cpu_only`; old CUDA modes migrate to `cuda_preferred`; historical XPU modes remain readable but validation rejects them as view-only. No executable XPU source/import remains. |
| G4 | Source-bound core and corrected GUI runtime retained; browser-client/final-candidate/CI repetition pending | Hardened CPU/CUDA profiles retain local-only noVNC/VNC, UID/GID 10001, dropped capabilities, no-new-privileges, read-only root, bounded `/tmp`, `/data`, shared-volume leases, one GPU and host-RAM ceiling. Exact clean commit `1f02a94` produced attested CPU/CUDA images, maximum BuildKit provenance/SBOM, metadata, filesystem manifests, CycloneDX and complete Trivy JSON; both privacy audits and zero-fixable-HIGH/CRITICAL gates passed. CPU/CUDA smoke, cross-container lease, case30/case57 parity, resource recovery and the accepted continuous 3,600-sample exact-image CUDA soak passed; the suspended attempt remains rejected. Clean commit `31a4713` corrected the rejected GUI dependency/health defect. Runtime image `sha256:f241c14c69d7896833e5805090d495f4ea14299de585cfb238ea13527b0deb5b` loaded `libqxcb.so` at build time, started `QApplication` on xcb, required a live Qt PID for health, rendered the 1600x1000 Dashboard, retained an exact volume-marker checksum across restart, returned on a new application PID with zero restart-loop count and stopped within the 20-second bound with exit 143/no OOM. The three captured renders share SHA-256 `28108327353d3a491f8d92daf3f081d3e8bfb8b8a0d53bd9540d1a2484025187`. In-app browser automation could not initialize because its desktop browser-control kernel failed before execution, so browser interaction is not claimed. Repeat browser interaction when that tool is available, then repeat the complete gate for the immutable final candidate and trusted CI. |
| G5 | Implemented | Study-strength protocols are validated on a deep copy, display a scientist-readable before/after diff, then atomically replace shared configuration and propagate through state signals. Run counts use a persisted paired-effect/power/Holm planning approximation, preserve governing-policy binding, and cannot be reduced by a fixed legacy evidence profile. Final run snapshots remain immutable in the experiment database. |
| G6 | Implemented; local installed-wheel Linux evidence retained, CI/browser execution pending | Normal experiment UI exposes two compute choices and no device percentages/batches/schema controls; policy UI hides No-AI/unqualified and routing internals; Dashboard readiness exposes available memory, admission status and recoverable queue progress instead of utilization/worker engineering. Rendered-widget contracts cover the five scientist workspaces. The Windows/offscreen suite and corrected Linux xcb image render pass. Clean commit `383e5bc` adds a wheel-contained packaged-GUI validator and changes the artifact job to install the wheel, change to `/tmp`, clear checkout `PYTHONPATH`, reject imports below `$GITHUB_WORKSPACE`, render and retain evidence before generating the staged manifest. The local Linux analogue imported only from `/tmp/wheel`, rendered all 16 workspaces' Dashboard shell at 1440x900 with no forbidden visible terms, and closed a valid clean session; screenshot SHA is `adc340f602011436ded5f321a55e5cb3855a8a0e1e50fe613032c1089789ca1f`. The latest complete active source tree passes `638 passed, 63 skipped` with 68% coverage, exact lock/schema/compile/Ruff gates and pinned 15-module mypy. Actual GitHub Actions execution and interactive browser proof remain pending. |
| G7 | Implemented | Policy training remains independently configured; qualified active-policy binding is synchronized into every new experiment while stored experiment snapshots remain immutable. |
| G8 | Complete | On 2026-08-03 the scientific lead stated exactly: “Approve TSH-CALO A–E, with F experimental and evidence-gated.” The current CALO is frozen as the baseline. TSH-CALO will use new algorithm/state/action/policy ABI versions, cannot auto-activate, and cannot inherit old superiority evidence. The required nine-part runtime/training confirmation was presented before upgraded implementation. |
| G9 | In progress — immediate CUDA batch-context development complete; physical share/candidate evidence pending | A–F mechanics remain green and F remains experimental/off by default. Runtime `tsh-calo-v1.1.0-counted-physics-candidate` now binds training environment v5/campaign v2/session v2/receipt v2. Counted ORPD populations retain voltage, diagnostics, bus types, generation and branch flows on the selected tensor device, then materialize evaluations/topology contexts once at the outer boundary with zero hidden power-flow reruns; optimizer and training no longer use a scalar CPU evaluation loop. Fresh CUDA receipts require a CUDA selected device, batched context API, zero CPU-CUDA inner-loop transfers and zero hidden context solves. Commits `ae7b304` and `e77431e` retain the frozen A–E evidence and immutable-candidate device-equivalence gates. Historical candidate v2 and its negative screen remain immutable, unqualified/inactive and ABI-incompatible; no threshold is weakened. Physical candidate-bound greater-than-95% CUDA timing/VRAM evidence, a fresh A–E/F-off candidate, accepted component evidence and qualification remain absent; protected cases stay closed. |
| G10 | Partially complete; mathematical-reference and A–E campaign implementations retained | Statistical corrections, honest claim boundaries, power-aware planning and a frozen-design preregistration protocol exist in `SCIENTIFIC_VALIDATION_PROTOCOL.md`. The runtime-enumerated 22-method campaign defaults to 98 initiated paired runs for 21 CALO-versus-comparator tests at effect 0.50, 95% power, Holm family control and 10% failures; pilot/simulation designs require an evidence SHA. case30/57 are validation replays, case118/300 are protected tests. Source-traceable L-SHADE 1.0.1, pinned pycma 4.4.4, official PGLib-OPF v23.07 validation assets and disclosed mathematical-reference adapters retain provenance, independent checks and protected-case refusal. The latest complete active suite is `638 passed, 63 skipped`; the new A–E campaign harness is not scientific evidence by itself. Human-reviewed external ORPD profiles, full multistart/reference execution, fresh-candidate accepted ablations, protected campaign execution and final qualification remain pending. |
| G11 | Harness implemented; final candidate blocked on G9/G10 and CI | A fresh dedicated distribution stage must be absent before each build, preventing obsolete artifacts from entering the wheel/sdist manifest. Generated policy checkpoints, lineages and training metadata are excluded. The verifier now requires the mathematical-reference and packaged-GUI commands in both distributions. Clean `383e5bc` produced a verified 349-member wheel and 401-member sdist; the extracted wheel passed Linux GUI, PGLib and reference-CLI checks and a six-file manifest. The CPU smoke image generates its filesystem manifest from `/opt/calo`; CI is configured to upload staged distributions/rendering and CycloneDX SBOM. This is development packaging evidence, not the final candidate. Final CI execution, exact final image attestations/digests, clean-machine reproduction, requirement closure and release freeze still follow G9/G10. |

### Historical Phase 4 policy-development boundary — 2026-08-07 (superseded 2026-08-14)

These bullets preserve the former execution order. They no longer make development phase, policy
age, or software revision a compatibility or qualification-admission criterion; the 2026-08-14
stage-neutral qualification compatibility gate below supersedes that lifecycle restriction.

- Phase 4 is coding and development validation only: finish A-E/F-off-capable production source,
  empty-policy behavior, runtime/CUDA/container/package/CI hardening, old-policy inventory and safe
  deletion tooling, and a source-bound development freeze.
- Every old policy is development-only, unqualified, inactive, non-final, and excluded from release.
  It is not modified, trained, evaluated, qualified, registered, activated, or deleted in Phase 4.
- Physical NVIDIA batching/VRAM and CPU/CUDA evaluator evidence may be collected as development
  infrastructure evidence without creating or evaluating a trained candidate.
- After Phase 4, deletion requires a separately reviewed exact inventory and explicit authorization.
  Only after empty-policy verification may the user freeze a new A-E/F-off plan and train a completely
  new policy against the development freeze.
- G9/G10 candidate, ablation, screening, qualification, and protected-campaign execution is deferred
  to that separately authorized post-development process. Phase 5 receives either one newly qualified
  checksum-bound policy or an explicitly approved policy-free scope.
- Phase 4 could not start while the Phase 3 Linux xcb boundary remained unresolved. The project owner
  subsequently accepted that manually validated boundary without claiming a retained automated Linux
  evidence directory, and the required development goal was created before Phase 4 source work.

### Phase 4 coding checkpoint — 2026-08-12

- The project owner closed the Linux xcb prerequisite by explicit manual acceptance, and the Phase 4
  goal was created before source edits. Baseline commit is
  `f800119cd3a14e2965c91040d0a8392013532089`.
- Production source now keeps primary rule-only CALO separate from the policy-gated TSH-CALO A-E/F-off
  registry path. A new experiment cannot select TSH-CALO without an immutable qualified active
  binding. Stale CALO/TSH policy, calibration, receipt, and provenance fields are cleared in the
  supported empty-policy state, and policy inference is pinned to the scheduler-resolved device with
  internal/baseline fallback disabled.
- Existing/pre-freeze policies cannot activate, bind, qualify through the historical GUI action,
  initialize a continuation, auto-discover from checkout data, or delete through the registry/GUI.
  The new retirement service inventories exact policy-store files and lifecycle database rows,
  creates a path-confined checksum-bound dry-run, and provides a later destructive transaction only
  behind matching clean source, exact inventory/plan, explicit authorization, and an external
  immutable receipt. No deletion was executed.
- Distribution verification requires the retirement/freeze tooling and continues to exclude all
  trained-model data except the package initializer. Physical-CUDA CI is policy-free during Phase 4;
  policy-hot-path evaluation was removed from that engineering lane.
- The development-freeze candidate command binds source identity, validator hash, interfaces,
  schemas, dependency locks, container declarations, exclusions, supported devices, 80% admission,
  zero release-scope policies, and prohibited-claim boundaries. It does not generate a release
  manifest or qualification receipt.
- Proportional test source and the ignored noninteractive `validation/Validate-Phase4.ps1` harness
  are implemented. Codex used the standalone Ruff formatter only as a mechanical source rewrite;
  it executed no Ruff check, test, validation, compilation, schema, lint, type, package, GUI,
  Docker, CUDA, policy, protected-case, or release command. G9-G11 and Phase 4 remain
  open pending the returned timestamped manual evidence directory and source-bound freeze review.

### v12 Phase 1 correction — 2026-08-06

- G10's earlier “locally verified” statistical wording is reopened for v12. Shared exact-keyed
  pairs, positive-is-better symmetric improvement, signed-rank-mass rank-biserial effect, declared
  Wilcoxon provenance/no test-family fallback, Holm handling, and separate time-to-feasible plus
  post-feasible AUC are implemented in both qualification engines under new schemas.
- The historical v3 screen is unchanged. Its old effect is marked legacy/unverifiable in an
  additive correction record because complete immutable raw pair values are not tracked. Its
  negative decision remains fixed by the crossing-zero interval and Holm-adjusted
  `p=0.052734375`; requalification is mandatory for any future promotion.
- No test, campaign, policy training/evaluation, benchmark or protected-case workflow was executed
  during this coding-only task. Phase 1 coding is complete, while G1/G10 local validation remains
  open pending the retained outputs produced by the Git-ignored
  `validation/Validate-Phase1.ps1` harness.

Latest verification evidence:

- Latest exact source gate: commit `05cd1b8` passed all three hash-complete CPU/CUDA/CI lock
  verifiers, generated schema, compileall, pinned 15-module mypy and repository Ruff lint; Ruff
  format was normalized across 422 Python files. The complete active tree then passed `638 tests,
  63 skipped` in `148.81s` with 68% checkout-source coverage. This excludes only the deliberately
  stale v6.9 release-integrity freeze, which remains a final G11 action.
- Current-source physical accelerator follow-up: case30/case57 each passed a 27-candidate FP64
  CPU/CUDA evaluator battery with zero scientific mismatches at clean `5b50095`; SHA-256 values are
  `424b8de56ce3cdb9b52c4bee4a38583fcf40a72c2a86ad5cb5144465eee81881` and
  `8708f691929e3931b838fc5c80b313895cac012435ebb83b438684245b1d67ff`. Clean `74268e3` then passed
  bounded resource recovery under SHA
  `73bb1f1bf6905f221b7993a2ec5d1bde50ccb6a074e22337a759234f5c64f13a`; its OOM/restart paths are
  controlled fault evidence, not natural hardware exhaustion.
- Scoped CUDA hot paths at clean `c1cf911` retained ten 100-evaluation ORPD windows per development
  case and three ten-epoch policy-update windows per case with dedicated VRAM, no declared inner
  CPU loop/fallback and measured CUDA-event shares above 99.98%. This is bounded accelerator-eligible
  event-time evidence, not whole-application utilization, requested extreme throughput, policy
  equivalence or scientific benefit.
- Commits `ae7b304` and `e77431e` implement the frozen A–E evidence producer and fresh-candidate
  device-equivalence gate. Focused development gates pass 33 and 11 tests respectively. No fresh
  candidate has executed them, so no A–E acceptance or policy CPU/CUDA equivalence is claimed.
- Current G9 development removes the counted-training scalar CPU ORPD loop: one tensor population
  retains final solver/context state on-device and uses one packed outer-boundary materialization.
  Seventy-six focused tests pass in 24.52 seconds, including no-rerun context and evaluation/voltage/
  branch/Jacobian/sensitivity parity. This is not yet physical greater-than-95% CUDA evidence; do
  not start fresh training until the target NVIDIA timing/VRAM gate passes.

- Clean-commit physical accelerator parity at `63f56adb4cf36e15210088eed92ff5325f76b02d`:
  development case30 and case57 both passed on the observed NVIDIA GeForce RTX 4060 Laptop GPU with
  27 deterministic candidates per case, zero feasibility/convergence/bus-type/scenario-count
  mismatches, and CUDA peak allocation/reservation within the recorded 80%-of-currently-free-VRAM
  allowance. Evidence SHA-256 values are
  `20f1f0da3b837e54d071359ac375b419fe0f750192bd290efdaa5edebba15b53` and
  `45ad11f4fff6bd045f4e8c9575bc19b818929f8a17768b2f4259dbfff07fd8e0`. This is a
  source/device/candidate-battery-scoped evaluator-parity claim, not pressure, soak, TSH policy,
  container, or performance qualification.
- Clean-commit physical resource recovery at `d6a950c519b6e3d586f546c60d97302ea3cd56a0`:
  a bounded 256 MiB allocation reduced observed free VRAM and the derived Safe-80 allowance, then
  recovered within the frozen 64 MiB tolerance; actual host staging computed on CUDA with no CPU
  inner-loop participation; controlled OOM backoff reduced microbatches `5 → 2`; a controlled typed
  capacity exhaustion performed a clean full-request CPU reference restart and subsequent CUDA
  recovery; and a second process was refused while the CUDA lease was owned and admitted after
  release. Evidence SHA-256 is
  `391180a00e721ce028bcd09141e260e0fcf7ccd5c3af23e124fbf9d513d6d89f`. The OOM
  and restart boundaries were explicit fault injection, not natural hardware-OOM evidence.
- Clean-commit physical CUDA soak at `67bd18ea704a4614e282b7bda3e2d29a28273d99`:
  3,600 GREEN samples over `3600.000156299968` seconds, no safe stop or protection stop,
  independently verified 3,602-event provenance chain, 46–60 °C observed temperature, 12.18–26.0 W
  observed GPU board power, and `24.33879127740349` Wh scoped GPU-board-energy integration. Result
  SHA-256 is `49b805c3019dadc2c97cafcff230b84c29c15ffd18f2bf5e54d5364edfa30800` and
  provenance SHA-256 is `4a44f4f37821f64d4affb861acb047177bb1f6af9e8671c481bf0007a96d75f4`.
  CPU temperature and GPU power-limit telemetry were unavailable; this is not whole-system energy,
  container, performance or policy-benefit evidence.
- continuation baseline at `7ec5b840193a4fe347c42e2d9ea1796fcac929e6`, before upgraded CALO
  implementation: **453 passed, 63 skipped**; Ruff lint/format pass across 363 checked Python files;
  generated experiment schema current;
- Change-A canonical transition refactor: **461 passed, 63 skipped** complete active-tree regression;
  **45 passed** focused CALO/parity/continuation; **22 passed** frozen optimizer seeded
  snapshot/exact-budget gate; **8 passed** dedicated canonical-kernel invariants. No B–F behavior
  was enabled by this evidence.
- Change-B topology-state/encoder contract: **5 passed** focused invariants and **466 passed, 63
  skipped** complete active-tree regression. This is implementation correctness only, not evidence
  that graph context improves CALO.
- Change-C hierarchical policy/action contract: **8 passed** focused invariants and **474 passed,
  63 skipped** complete active-tree regression. The focused set includes a bounded CPU/CUDA forward
  comparison on the available device; it is not physical CUDA qualification or optimization
  performance evidence.
- Change-D uncertainty/bandit/safety/fallback contract: **6 passed** focused invariants, **28
  passed** cumulative A–D focus and **480 passed, 63 skipped** complete active-tree regression.
  These tests establish mechanics and deterministic replay, not calibrated uncertainty quality or
  optimization benefit.
- Change-E optional physics proposal/accounting contract: **10 passed** focused invariants, **38
  passed** cumulative A–E focus and **490 passed, 63 skipped** complete active-tree regression. The
  operator remains masked by default and has no incremental-benefit evidence.
- Change-F experimental population-schedule mechanics: **9 passed** focused invariants and **47
  passed** cumulative A–F mechanics. The active development tree excluding the deliberately stale
  v6.9 release-integrity file is **499 passed, 63 skipped**. The complete tree is **502 passed, 63
  skipped, 2 failed**, where both failures are the expected stale freeze/root-manifest gates. The
  schedule remains disabled by default and has no promotion, benefit or acceptable-cost evidence.
- TSH-CALO immutable candidate lifecycle: **7 passed** dedicated artifact/registry tests and **33
  passed** with existing policy compatibility, binding, independence, topology and hierarchical-
  action regressions. The active development tree is **506 passed, 63 skipped**; Ruff lint/format
  pass across 378 Python files and the generated experiment schema is current. This proves lifecycle
  mechanics only; no candidate has been trained, qualified or activated.
- Independent TSH-CALO PPO core: **7 passed** dedicated design-hash/leakage/masked-update/exact-
  resume/export/separation tests and **22 passed** with lifecycle and hierarchical-action regressions.
  The active development tree is **513 passed, 63 skipped**; Ruff lint/format pass across 380 Python
  files and the generated schema is current. Rollout state production and target-CUDA admission/
  execution remain to be integrated and physically qualified.
- Immutable ensemble and shielded inference core: single members cannot activate; ensemble assembly
  preserves member hashes/provenance. **6 passed** dedicated admission/identity/replay/shield/fallback/
  separation tests and **35 passed** across inference, lifecycle, uncertainty shield, trainer and
  hierarchical actions. The active tree is **520 passed, 63 skipped**; Ruff lint/format pass across
  382 Python files and the schema is current. This is not end-to-end optimizer execution or physical
  CUDA qualification.
- Counted ORPD solver context: **3 passed** dedicated no-extra-call/equivalence/fail-closed tests and
  **37 passed** with topology, repair, transition and frozen-CALO runtime guards. The active tree is
  **523 passed, 63 skipped**; Ruff lint/format pass across 383 Python files and the schema is current.
  The context is ephemeral and not publication evidence by itself.
- TSH-CALO runtime context and versioned candidate-transition mechanics: **9 passed** dedicated
  measured-context/group-action/physics-failure/ABI/precision-channel invariants and **68 passed**
  with counted physics, canonical-kernel, seeded optimizer snapshots and deployed/native parity.
  The active tree excluding only the deliberately stale v6.9 release-integrity file is **532 passed,
  63 skipped**; repository Ruff lint/format passes across 387 Python files and the generated schema
  is current. This does not yet constitute an end-to-end optimizer, CUDA qualification or benefit
  evidence.
- TSH-CALO policy-gated optimizer mechanics: **8 passed** dedicated execution/preflight/fallback/
  accounting/registry/F-gate/independence/exact-resume cases and **66 passed** across optimizer,
  registry, frozen campaign and benchmark snapshots. The active tree excluding only the stale v6.9
  release-integrity file is **540 passed, 63 skipped**; repository Ruff lint/format passes across
  389 Python files and the generated schema is current. These are synthetic CPU mechanics, not a
  trained or scientifically qualified policy, target-CUDA evidence, or performance evidence.
- Immutable TSH-CALO qualification/calibration receipt: **5 added cases** cover exact receipt
  round-trip, protected-case rejection, generic-row activation refusal, mutation refusal and runtime
  revalidation; **27 passed** across qualification, lifecycle, inference and optimizer. The active
  tree is **545 passed, 63 skipped**; Ruff lint/format passes across 391 Python files and the schema
  is current. A receipt authenticates declared inputs only and does not prove that qualification
  evidence exists, is sufficient, or passed its preregistered criteria.
- Independent canonical-reward rollout collection: the training design now freezes discount/GAE
  factors; sampled actions must be committed with a versioned canonical `TransitionResult`; terminal-
  aware GAE/returns are deterministic; pending actions cannot be checkpointed; and collector restore
  requires the unchanged scientific design hash. **9 passed** dedicated training cases, **22 passed**
  with canonical/runtime transition guards and **547 passed, 63 skipped** on the active tree. This is
  a rollout data boundary, not a counted ORPD environment, completed training run or benefit result.
- Counted independent training environment: exact development-case/checksum/formulation/design
  binding; protected loaded-case rejection; counted full-population `evaluate_with_context` batches;
  selected counted topology context; canonical candidate generation/transition reward; exact FE and
  per-scenario call totals; opt-in counted E context/proposal and fail-closed dynamic mask; pre-solve F rejection; pending-observation/RNG/component
  resume; static absence of experiment, registry, activation and production-inference authority; and
  poison-on-solver-failure provenance. **8 passed** dedicated cases, **29 passed** with adjacent
  training/transition/context guards and **555 passed, 63 skipped** on the active tree. Ruff
  lint/format passes across **395 Python files** and the generated schema is current. No fresh
  training, qualification or benefit evidence exists.
- Independent trainer Safe-80 admission: current training ABI
  `tsh-calo-training-v5-batched-device-context-safe80` (historical v3/v4 artifacts remain immutable and non-native to v5); hash-bound rollout/population/topology/scenario envelope;
  versioned working-set estimate; per-state/batch envelope enforcement; CUDA-first 80%-of-current-
  free-VRAM admission; explicitly governed 80%-of-current-available-RAM CPU path; cross-process and
  local single-owner CUDA leases; allocator ceiling; exact-device resume; honest computation/memory
  semantics; immutable candidate provenance validation; and v1/v2 incompatibility classification.
  **7 passed** dedicated resource cases and **49 passed** across resource, trainer, environment,
  lifecycle, inference and optimizer. The active tree is **563 passed, 63 skipped**; Ruff lint/format
  passes across **397 Python files** and the generated schema is current. CUDA results are mocked
  mechanics, not target hardware proof.
- Counted fresh-member training session: joins only the independent trainer and development
  environment; commits canonical transition rewards; updates at admitted rollout/terminal boundaries;
  binds unique run/session/design/case/checksum/formulation/seed identities and exact FE, scenario,
  transition, update and reward-sequence accounting into a terminal receipt; requires a receipt for
  explicit unqualified export; jointly resumes trainer/environment/collector/rewards/metrics from a
  trusted authenticated checkpoint; and forbids checkpoint, receipt and continuation after solver
  failure. **5 passed** dedicated cases and **39 passed** across session, trainer, lifecycle,
  inference and optimizer. The active tree is **568 passed, 63 skipped**; Ruff lint/format passes
  across **400 Python files** and the generated schema is current. No real candidate, qualification,
  ablation, target-CUDA or benefit evidence exists.
- Frozen-plan independent training campaign: exact source/design/execution/seed/curriculum hashes;
  distinct member seeds, training-run IDs and candidate artifacts; globally unique session IDs;
  protected-case and F rejection; explicit clean-source start/resume command; per-transition trusted
  resume plus status path/SHA verification; terminal scientific/integrity failures; narrowly
  resumable accounting-complete infrastructure `OSError`; and explicit unqualified-only outputs.
  **7 passed** dedicated campaign cases and **62 passed**
  across campaign, session, trainer, environment, resources, lifecycle, inference and optimizer. The
  active tree is **576 passed, 63 skipped**; Ruff lint/format passes across **403 Python files** and
  the generated schema is current. These validation cases use toy fixtures; the separate real v1
  attempt below is incomplete and is not a scientific result.
- First real IEEE 30/57 attempt: frozen v1 plan/source/design/seed hashes retained; three members
  completed 20,000 FE/scenario calls and 70 updates each on admitted CUDA; member 4 stopped at
  episode 10 transition 20 when Windows denied an atomic status replacement. Its checkpoint is
  authenticated, session failure is false and accounting is complete at 840/840 calls. Campaign v1
  remains failed under its original semantics; partial candidates are unqualified and are not benefit
  evidence.
- Fresh IEEE 30/57 v2 campaign: five independently seeded members each completed 10 episodes,
  20,000 candidate evaluations/scenario calls, 490 transitions and 70 PPO updates on admitted
  `cuda:0` with no fallback. Aggregate accounting is 100,000 evaluations/calls, 2,450 transitions
  and 350 updates. The ensemble SHA-256 is
  `3adf5017dc51f33d76214aeb505da598984b9da0e4263e1ee8fe59a667180ceb`; its manifest SHA-256 is
  `ded60598652d552a70f03c811969092ab243437f2d5adaf8d7f75f665bc80f33`. It remains unqualified,
  inactive and unusable by ordinary experiments. This is real CUDA training provenance, not
  qualification, CPU/CUDA equivalence, OOD calibration quality, ablation or benefit evidence.
- Independent TSH-CALO qualification campaign: frozen plan/source/policy/seed/calibration hashes;
  counted development-only OOD fitting; non-serializable candidate-evaluation authority; paired
  exact-FE frozen-CALO comparison; retained failures; independent PYPOWER checks; feasibility-first
  endpoints, paired practical effects, deterministic bootstrap intervals, Holm correction and
  frozen anytime checkpoints; exact-resume cell retention; screening/formal separation; A–E direct
  evidence prerequisite; F exclusion; and no registry/activation authority. The first real screening
  attempt is retained failed-integrity after three timeout-surviving Python processes raced the same
  directory; four ambiguous cells are explicitly barred from scientific use. An OS-released
  single-writer evidence-directory lease and failed-resume guard now close that defect. **6 passed**
  dedicated campaign cases and **21 passed** across campaign, inference and optimizer. Valid v3
  screening then completed 40/40 unique records with zero failures, exact 2,000-FE/scenario-call
  accounting, paired seeds, admitted CPU policy inference without fallback and independent validation
  of all retained solutions. Its evidence SHA-256 is
  `039f2bfe31e39196e126da3961c65e4a248133ed09b009a93f64c933b2292778`. Case30 had 0/10 feasible
  runs in each arm. Case57 had 10/10 in each arm, median paired improvement `0.011492392668353543`,
  95% bootstrap CI `[-0.0019638316095621712, 0.01484160649928978]`, rank-biserial `0.4`, win rate
  `0.7`, and Holm-adjusted `p=0.052734375`. The frozen decision is grade `U`, `passed=false`, with no
  qualification or benefit claim. No receipt, registration or activation exists; protected cases
  remain unopened.

- automatic CUDA-first scheduling/config/GUI regressions: **54 passed**; versioned database migration,
  history, learning, resume and continuation regressions: **29 passed**;
- focused execution/schema/VRAM/GUI/policy suite: **39 passed**, followed by **27 passed** after
  current-schema serialization was tightened;
- full unit suite checkpoint: **358 passed, 62 skipped, 4 failed**; two failures were obsolete GUI
  contract assertions and have since been corrected (**12 passed** on rerun); the two remaining
  failures are the deliberately stale v6.9 freeze and package manifest gates;
- current complete development-tree suite, including offscreen GUI, integration, regression and
  scientific tests and excluding only that historical v6.9 release-integrity file:
  **453 passed, 63 skipped**, with the latest measured CI-style coverage gate passing at **66%**
  (threshold 60%);
- complete offscreen GUI/scientist contract: **33 passed** with a validated 1440x900 PNG artifact;
- repository-wide Ruff lint and format: **pass** across 359 Python files; the initial formatter pass
  mechanically normalized 115 files without intentional behavior changes;
- pinned mypy 1.20.2 bounded safety target with untyped-body checking: **pass, 9 source files**; artifact/container/lock/L-SHADE/
  study-planning focused regression: **24 passed, 1 platform skip**;
- latest power-planning/transactional-study/scientist-GUI checks: **18 passed**; schema and
  fixed-memory contract checks: **24 passed**;
- scientist-facing Dashboard readiness/queue contract plus study checks: **19 passed**;
- power-aware campaign/design-hash/case-role and rendered-interface checks: **48 passed**;
- focused L-SHADE mechanics, CPU/tensor execution, campaign integration and deterministic release
  regression checks: **39 passed**, followed by **36 passed** after source-exact rounding and repair
  telemetry corrections;
- no executable XPU match remains outside the explicit historical view-only compatibility reader;
- physical host CUDA parity/resource/recovery/thermal/power/GPU-board-energy evidence: **retained**;
  Docker image, container repetition, whole-system energy and WSL2/GUI evidence remain open.

## v12 Phase 2 runtime-contract gate - 2026-08-06

| Gate item | Source state | Evidence state |
|---|---|---|
| Pre-run resolution across GUI, direct, parallel, benchmark, and final campaign paths | Implemented | Accepted in `phase2-20260807-003828` |
| Formal CUDA-required/no-fallback and exploratory explicit full-request CPU restart | Implemented | Accepted in `phase2-20260807-003828` |
| Requested/physical/logical/runtime/actual device provenance and CUDA-claim exclusion | Implemented | Accepted in `phase2-20260807-003828` |
| UUID-first physical lease, normalized PCI fallback, host scope, and queued contention | Implemented | Accepted in `phase2-20260807-003828` |
| Safe-80 admission and request-versus-lifetime VRAM telemetry | Implemented | Accepted in `phase2-20260807-003828` |
| Exact batch cardinality/identity before FE registration | Implemented | Accepted in `phase2-20260807-003828` |
| Versioned partial-failure envelope with exact FE and checkpoint boundary | Implemented | Accepted in `phase2-20260807-003828` |
| FP64 authority, CPU-only topology, active CUDA/CPU status, XPU view-only boundary | Implemented | Accepted in `phase2-20260807-003828` |

The source suite and ignored `validation/Validate-Phase2.ps1` harness exist, but Codex did not run
them. This table records implementation presence only. Phase 2 does not pass until the user's hashed
manual validation run is reviewed; no policy or scientific-evidence command is part of that harness.

The first retained manual run, `phase2-20260807-001858`, passed 13/15 commands. Its 23 dedicated
Phase 2 contracts passed; the two failures were generated-schema property order and one stale
pre-Safe80 error-message assertion in the affected regression set. Both source corrections are
applied. The gate remains open pending a complete fresh manual rerun and evidence review.

The corrected run, `phase2-20260807-003024`, passed 14/15 commands: generated schema, Ruff
diagnostics, 23/23 Phase 2 contracts, and 44/44 affected regressions all passed. Ruff format alone
reported mixed line endings in `tests/unit/test_v690_vram_residency.py`. The file is normalized to
its established CRLF style without executing Ruff or tests. A new complete source-bound manual run
is still required; the gate remains open.

Final Phase 2 evidence `phase2-20260807-003828` passed 15/15 commands. All 20 retained evidence
hashes, all 35 source hashes, and validator identity matched at review; 23/23 Phase 2 contracts and
44/44 affected regressions passed with no prohibited workflow. The Phase 2 runtime-contract gate is
accepted.

## v12 Phase 3 scientist-GUI gate - 2026-08-07

| Gate item | Source state | Evidence state |
|---|---|---|
| Five grouped/collapsible navigation sections with stable keyed restoration | Implemented; first-run corrections applied | Corrected Windows contracts and interactions accepted; Linux xcb pending |
| Persisted compact/expanded rail, SVG icons, search, state badges, blocked explanations | Implemented; first-run corrections applied | Windows keyboard/search/persistence automation accepted; Linux xcb pending |
| Next-action Dashboard with five readiness categories and recent/resumable/failure evidence | Implemented; formatting-independent contract applied | Windows render and all-workspace automation accepted; Linux xcb pending |
| Seven-step Study Setup separated from the Dashboard | Implemented | Windows programmatic keyboard interaction accepted; Linux xcb pending |
| Bounded 240-480px controls, structured integer chips, expandable text, horizontal form groups, and tabbed multi-section workspaces | Results/Settings corrections, five-workspace tabs, and Portfolio output-tree width allocation implemented | Corrected Windows v3 all-tab/tree-width evidence accepted; Linux xcb pending |
| Progressive disclosure for activity, continuation, queue, and advanced details | Implemented | Windows programmatic keyboard interaction accepted; Linux xcb pending |
| Named semantic light/dark tokens, 8px spacing, 40/44px density modes, focus visibility | Implemented; unsupported QSS removed | Windows light/dark/high-DPI accepted; token-contrast and Linux xcb evidence pending |
| Accessibility names, form buddies, keyboard search/step navigation, non-color state text | Implemented | Corrected Windows semantic/keyboard automation accepted; Linux xcb and human acceptance remain separate |
| Render/glyph/clipping evidence generator and ignored Phase 3 validator | Deterministic system-font registration, per-section-tab captures, tree-column width/clipping audit, and durable v3 evidence implemented | Corrected Windows `121530` accepted; Linux xcb lane pending |

**Owner closure on 2026-08-12:** the project owner explicitly accepted the manually validated Linux
xcb boundary and directed Phase 4 to proceed. No automated Linux evidence directory was retained, so
the Linux row is closed by owner scope/gate acceptance, not by reproducible automated evidence. The
decision does not infer human accessibility/scientist acceptance or qualify any policy, scientific,
performance, RC, or release claim. Phase 3 is closed; Phase 4 began only after its goal was created.

Phase 3 run `phase3-20260807-045558` passed 11/18 commands and remains failed evidence. Its 60/62
tests included two brittle source-text failures; all four render cells were unreadable because the
offscreen Qt runtime had no discoverable font. Source corrections now register an existing OS font
without redistributing it, remove unsupported QSS, make tests formatting-independent, improve
failure counts, add commit/dirty hashes to manifests, and exclude ephemeral pytest temp artifacts.
Only the Ruff formatter was used as a mechanical source rewrite; Codex did not execute tests,
checks, compilation, renders, or validation at that correction checkpoint.
The corrective Windows rerun `phase3-20260807-052047` is accepted: 18/18 commands passed, with 62
Phase 3/GUI tests, 35 Phase 2 presentation regressions, Ruff/compile/version gates, and all four
light/dark/high-DPI render cells passing. The run retained source commit
`00b8ee07a6d59c0d805d0c043c91ae5ea73d45d0`, dirty state, status hash, 34/34 durable hashes and
32/32 current source hashes. It recorded no policy, evaluation, qualification, benchmark, campaign,
or protected-case workflow. The prior failed run remains immutable history.

Run `phase3-remaining-windows-20260807-092741` then failed 3/8 gates: three files required Ruff
formatting, Results Explorer clipped `Open experiment workspace`, and Application Settings clipped
the result-database label/path in both light and dark/200% cells. The retained record was intact:
45/45 durable hashes, 32/32 current source hashes, exact validator identity, and no prohibited
workflow. The layouts, evidence detail, and formatting are corrected in source.

At the user's direction, the replacement Windows validator is fully noninteractive and cannot use
reviewer answers. Run `phase3-remaining-windows-20260807-112621` passed 10/10 commands, including
13 Phase 3 contracts, both focused Results/Settings regressions, and light plus dark/200% evidence
across all sixteen workspaces. Its 47/47 durable hashes and 36/36 current source hashes matched, and
it executed no prohibited workflow. This accepts the corrected pre-tabbed Windows source while
preserving the earlier failed runs as history.

The subsequent user-directed refinement replaces vertical section stacks in ORPD Formulation,
Robust Scenarios, Portfolio Manager, Application Settings, and Benchmark & Evidence with a shared
accessible `WorkspaceTabs` surface. Related compact controls are arranged in balanced side-by-side
groups, while genuinely long paths retain dedicated width. The v2 all-workspace collector now visits
every new section tab by keyboard and retains a screenshot for each tab. These changes postdate the
accepted `112621` source manifest, so fresh noninteractive Windows and Linux light/dark xcb evidence
were mandatory. Corrected Windows run `121530` is now accepted; Phase 3 and Phase 4 remain blocked
pending the Linux directory.

Run `phase3-remaining-windows-20260807-120240` is intact failed evidence for the first tabbed source:
9/10 commands passed, including 14 contracts, five layout regressions, and both all-tab render cells;
79/79 evidence hashes and 17/17 source records matched. Ruff formatting rejected five files. Visual
review additionally found unused width and shortened evidence text in Portfolio Requested outputs,
which the v2 collector could not detect. Current source corrects the tree resize policy, adds a
rendered width regression, advances workspace evidence to v3 with fail-closed tree width/text checks,
and advances the ignored Windows summary to v4. Codex did not execute validation or tests; corrected
Windows evidence and Linux xcb evidence remain mandatory.

Corrected run `phase3-remaining-windows-20260807-121530` is accepted for its exact source: 10/10
commands, 15 Phase 3 contracts, six layout regressions, and both light/dark-200% v3 workspace cells
passed. Validator and Git-status identities matched the returned state at acceptance review; 17/17
source entries and 79/79 evidence hashes matched. The acceptance ledger now postdates that manifest.
Both Portfolio tree checks consume the complete 1054px viewport with zero
unused/overflow width and no clipped column. This closes the Windows correction goal. Phase 3 and
Phase 4 remain blocked only on the separate Linux xcb directory; Windows does not infer Linux or
human acceptance.

## Documentation and instruction alignment - 2026-08-07

All 60 live `AGENTS.md` files have been audited under the root policy. Scoped instructions that can
affect policy lifecycle, GUI, tests, containers, reports, data, training, CI, or implementation
documentation now state the Phase 4 development-only boundary directly; copied `AGENTS.md` files in
retained build/baseline artifacts were not modified. `docs/DOCUMENTATION_STATUS.md` identifies live
instructions and classifies versioned/datestamped records as historical evidence. This documentation
alignment does not start or pass Phase 4 and was not accompanied by any validation execution.

## Phase 4 development-completion gate - 2026-08-12

**Status: coding complete; manual validator evidence pending.** The required Phase 4 goal was
created before implementation. Production source now supports safe empty-policy/rule-only CALO,
strict immutable-policy-gated TSH-CALO, F disabled, no internal inference fallback, read-only exact
old-policy inventory/dry-run retirement preparation, and policy/validation exclusion from source
distributions and built-image manifests. No real policy was trained, evaluated, qualified,
registered, activated, or deleted.

The development-freeze candidate binds the complete Git-tracked and non-ignored untracked source
set, raw Git-status identity, declared interfaces, dependency locks, container/exclusion contracts,
ignored validator/instructions, and exact policy-lifecycle inventory. Future new-policy training is
barred unless a retained clean post-transition report has zero policy files/references/rows, matches
the exact freeze commit and payload SHA-256, and records empty initialization. This is development
infrastructure, not policy evidence or authorization to run policy workflows.

The validator cannot accept itself. After read-only review of a fully passing returned directory,
the separate `calo-rpd-accept-development-freeze` command may create a non-overwriting explicit
acceptance receipt outside the immutable run. It verifies all retained hashes and all 32 exact
result IDs produced by the 30 numbered stages, then binds the production-source content contract.
Later old-policy authorization and any
new training plan must identify that receipt SHA-256. No acceptance receipt exists yet.

The project owner separately accepted the Phase 3 Linux/xcb boundary manually. The remaining Phase
4 gate is a fresh owner-executed `validation/Validate-Phase4.ps1` run and read-only review of its
complete `phase4-*` directory. Until accepted, Phase 4, its active goal, the development freeze,
package/container/CUDA/GUI validation, RC, and release gates remain open.

## Phase 5 release-preparation development gate - 2026-08-12

**Status: coding complete; combined Phase 4/5 evidence pending.** The project owner explicitly
requested Phase 5 development before manual validation. This permits source/test/harness
implementation only and does not close any evidence, RC, final-release, policy-scope, or publication
gate.

Implemented source now provides exact policy-free/new-policy scope validation; source-bound release
preparation over distinct wheel/sdist, CPU/CUDA image, Buildx, SBOM, scanner, vulnerability,
filesystem, clean-install, packaged-GUI, and CI records; and a disabled, independently authorized
final metadata/source-manifest generator. The active identity remains `12.0.0.dev1`; scope is
pending; old policies remain untouched; and every RC/final/publication flag is false.

The ignored combined wrapper runs full Phase 4 first, refuses Phase 5 after a Phase 4 failure,
requires both child summaries to bind the same Git identity, and retains separate hash-complete
Phase 4, Phase 5, and combined directories. No validator was executed by Codex. G0-G11, the combined
validation decision, exact policy scope, any old-policy transition, `12.0.0` promotion, final-record
authorization, tag, push, publication, and release remain open.

### First combined-run correction checkpoint - 2026-08-12

The owner-executed `phase4-20260812-165006` run was interrupted during its buffered CUDA image
stage and is not accepted evidence; Phase 5 did not start. Earlier retained Phase 4 logs exposed
format, type, engineering-contract, and GUI-contract failures plus a validator defect that treated
native stderr as a terminating PowerShell exception. The affected source/tests are corrected. The
ignored Phase 4 and Phase 5 validators now stream live output, use native exit codes, and fail fast;
the combined wrapper streams child output and selects exact phase directories. Compose Phase 4
smoke no longer passes unsupported `run --read-only` because both services already declare
`read_only: true`. Codex executed no manual-capable validation command. All Phase 4/5 evidence gates
remain open pending a fresh combined owner run.

The next owner run `phase4-20260812-182252` confirmed live output and first-failure stopping. Its
environment, version, compile, schema, and Ruff checks passed; `06-format` rejected eight files and
Phase 5 did not start. Those exact files were mechanically formatted without running the follow-up
check. This run remains failed evidence, and all Phase 4/5 proof gates still require a fresh
combined run.

The combined wrapper now retains a failed child's complete summary/hash identity instead of writing
`phase4: null` after a normal child-gate failure. This improves failed-evidence routing only; it does
not permit Phase 5 to start or weaken the complete-pass requirement.

Owner run `phase4-20260812-182752` then passed the first eight commands, including 15-file mypy and
112 engineering tests. Its GUI suite passed 36/37 and failed only because a visible disabled-button
tooltip still said “post-development qualification authority.” Scientist-facing policy/campaign
wording is now consistently post-freeze/historical/accepted-source language. Phase 5 did not start;
the affected GUI source was mechanically formatted, but the correction is unvalidated until a
fresh combined owner run.

Owner run `phase4-20260812-184454` passed 14 result IDs through all GUI checks, lifecycle
preparation boundaries, and wheel/sdist construction. `14-distribution` alone failed because the
verifier treated application code under `calo_rpd_studio/validation/` as root local validation
evidence. Archive and container gates now distinguish and require the application validation
package while still rejecting root `validation/` and `validation_logs/`. Phase 5 did not start;
fresh combined proof remains required.

Owner run `phase4-20260812-185135` then passed 17 result IDs through distribution verification and
clean wheel installation. `17-clean-smoke` failed because its repository-wide path exclusion also
covered the intentionally repository-local, Git-ignored clean virtual environment. The validator
now proves both sides of the intended boundary: the import must be beneath the new clean environment
and must not be beneath the checkout source package. Phase 5 did not start; this harness correction
requires a fresh owner-executed combined run.

Owner run `phase4-20260812-190643` passed 24 result IDs through locked CPU/CUDA container smoke and
physical NVIDIA discovery, then stopped at `23-cuda-parity-30` before numerical work because the
dirty development source could not enter the durable qualification tier. Phase 4 acceptance is
explicitly allowed to bind a stable dirty development tree by its complete source manifest, so the
physical engineering tools now provide an explicit development-only tier. That tier cannot set
durable or qualification status; default clean-source behavior remains fail-closed. Phase 5 did not
start and a fresh owner-executed combined run remains required.

The related ignored Phase 5 clean-install gates were corrected proactively: wheel and sdist imports
must be inside their exact isolated environments and outside the checkout source package, while the
packaged-GUI check forbids only that source package. This prevents Git-ignored validation storage
inside the repository from being mistaken for an editable checkout import.

The ignored combined wrapper also performs an executable-resolution preflight for Python, Docker,
NVIDIA-SMI, and Trivy before Phase 4 starts. Missing release tooling therefore produces an immediate
retained combined failure instead of consuming a full Phase 4/container run first.

Owner run `phase4-20260812-195901` passed all 32 Phase 4 development-validation result IDs. This is
retained successful automated evidence for the exact recorded dirty source state, not the current
post-correction tree and not a clean final-source, policy, RC, release-ready, or final-release claim;
explicit Phase 4 acceptance-receipt generation remains separate. Phase 5 run
`phase5-20260812-201822` passed version, compile, schema,
Ruff, and format gates, then failed `06-types` on an untyped PyYAML import and an `Any` JSON return.
Those typed boundaries are corrected with a narrow import annotation plus a runtime-validated JSON
object loader. Fresh Phase 5 and combined proof remain required.

Combined attempt `phase4-phase5-20260812-202511` then passed preflight and the Phase 4 environment
record but failed `02-version` because the active-version verifier still required obsolete
pre-validation Phase 4 status text. All other version-report checks passed. The verifier now binds
the current revalidation-pending core status and separately verifies the exact seventh/eighth
attempt history. Phase 5 did not start; fresh current-source proof remains required.

Combined attempt `phase4-phase5-20260812-202852` passed all 32 Phase 4 commands. Its Phase 5 child
`phase5-20260812-204823` passed 23 commands through isolated sdist smoke and failed first at
`22-cpu-build`: the Docker classic image store cannot retain the required local provenance and SBOM
attestations. The attestation controls remain mandatory. The ignored combined preflight now requires
Docker driver status `io.containerd.snapshotter.v1` before beginning Phase 4, so this prerequisite
cannot again be discovered after the expensive preliminary gates. Enable Docker Desktop's
containerd image store, restart Docker Desktop, and perform a fresh owner combined run.

## Phase 6 GUI modernization gate

Owner combined attempt `phase4-phase5-20260813-000340` passed: Phase 4 is 32/32 and Phase 5 is
41/41 with zero failed commands for their exact retained source identity. This closes the combined
development/release-preparation validation prerequisite for starting Phase 6; it does not select a
release-policy scope or authorize a release.

Phase 6 started from clean commit `2d7130fb63c13d35d0419dd63b1d68e2050dcf72` after its goal was
created. Its gate is GUI/native-launch development only: one registry-generated ribbon, compact
context editors, central workspace tabs, truthful jobs/logs/progress/status, responsive themes, an
enabled entry to the independently gated new-policy training center, and first-class Windows launch
without Docker. The obsolete embedded legacy trainer remains unavailable, navigation cannot start
training, and no policy or scientific workflow is authorized during development.

Phase 6 production, test, documentation, packaging-check, and ignored-validator source is now
development-complete. The independent training adapter rejects a successful readiness result if any
bound path changed during the check, keeps output unqualified/inactive, and prevents silent window
close while its process is active. Native checkout and installed-wheel launch paths are distinct
from setup. No development-agent validation, Docker, policy, protected-case, publication, or release
operation was executed.

Owner attempts `phase6-20260813-031026` and `phase6-20260813-031046` are retained **failed harness
evidence**: both stopped before command `01` while Windows PowerShell evaluated the environment
architecture expression. Both prove exact nonignored source-status stability and record all
prohibited-operation and inferred-human-acceptance fields false, but neither validates Phase 6. The
ignored validator uses a Windows-compatible process-environment architecture value plus bitness
fallback after correction.

Authorized consolidated run `phase6-20260813-032036` passed all 19/19 checks. It binds commit
`2d7130fb63c13d35d0419dd63b1d68e2050dcf72` plus retained dirty source-status SHA-256
`beabf2d918c4717e61e3b9c12ba449fdf2e59a38ccd549da54e2fb74cbabe9bf`, proves identical
before/after nonignored source state, validates GUI contracts and regressions, renders light/dark/
constrained layouts, builds a fresh wheel and sdist, and verifies native/GUI package membership.
The Phase 6 development-validation gate is **closed / 19 of 19 passed**. Every policy, protected-
case, Docker, CUDA-campaign, publication, and release-execution field is false. Human usability,
screen-reader, and scientist acceptance remain uninferred, and all release-policy, qualification,
release-candidate, publication, release-readiness, and final-release gates remain separate and open.

A post-gate professional visual-refinement pass subsequently removed inactive ribbon-page
bleed-through, added a complete 16-workspace ribbon palette, made the left dock input-only, and
unified new-policy readiness/start behind the same ribbon action without weakening confirmation or
lifecycle gates. Targeted GUI contracts passed 7/7 and the retained
`phase6-panel-sweep-20260813-041700` captures cover every workspace plus light, dark, and constrained
shell states. This presentation evidence performed no policy, scientific, protected-case, Docker,
publication, or release workflow and does not alter the already separate release gates.

A focused follow-up restores complete active-plan training inputs to the left Inputs pane. The
retired legacy trainer stays hidden: the visible fields edit a loaded frozen TSH-CALO plan, produce
a separate hash-addressed plan, invalidate stale readiness, and feed the same explicitly confirmed
ribbon action. This source follow-up awaits the owner-run Phase 6 validator and does not reopen or
weaken any policy-lifecycle or release gate.

That follow-up also makes Inputs and the expanded ribbon permanent by removing their hide/compact
commands and migrating saved layout state. The central scientific preview now scrolls around a
roomy minimum canvas and uses a taller branded tab header. These are presentation contracts only;
focused owner validation remains pending.

The training-case control is now a catalog-backed checklist. One action selects all legally
eligible bundled cases (`case30` and `case57`); protected `case118` and `case300` are shown but
disabled and remain rejected by the scientific plan boundary. The Windows title bar now receives a
CALO-RPD application icon, while native tab bases and main-window/dock separators are explicitly
themed to remove the gray/yellow-looking artifacts observed in the dark shell. This focused source
change remains unvalidated until the owner runs the narrow validator; it authorizes no training or
protected-case work.

The current 18-field scientific training surface gives every visible field a compact accessible
information button. Its hover, click, and keyboard-focus tooltip explains purpose and directional
effects while avoiding permanent prose
in the input pane. This is a presentation/help contract only; focused owner validation remains
pending and no scientific setting or lifecycle authority changed.

The ribbon now explicitly synchronizes page visibility on category changes and same-category
reselections. This removes the stale inactive command-group edge that could paint as a short
colored fragment between the CALO-RPD heading and Home. The regression and ignored validator
sources are updated; this focused presentation correction remains owner-validation pending.

Ribbon group names no longer use a native `QGroupBox` title forced onto the lower border. Each
group instead owns a centered, fixed footer label within its layout, preventing Windows theme
painting from overflowing or clipping captions such as `Project` and `Navigate`. Both themes and
the focused GUI contract are updated; owner visual and automated validation remain pending.

Numeric inputs now use package-safe antialiased vector chevrons supplied by a Qt proxy style rather
than low-contrast native Windows spin glyphs. Light and dark themes provide a wider separated
stepper column with clear normal, hover, pressed, and disabled states while reserving text padding
so values do not collide with the controls. Focused source and validator contracts are updated;
owner visual and automated validation remain pending.

The independent training editor no longer displays a separate foundation or asks scientists for
development-freeze and Phase 4 acceptance paths. Training uses the existing built-in TSH-CALO
architecture; approved A–E remain available, optional E is off, and experimental F remains disabled
by default. Optional plans import only
scientific settings and are rebound to the current application source. The CLI keeps its clean
source readiness boundary, and candidate qualification and activation remain separate fail-closed
lifecycle operations. Focused source, test, documentation, and ignored-validator updates await
owner-run validation; no policy or release operation was executed.

All 18 current training-input information tooltips add practical low-to-high starting ranges for
numeric values and explicit recommended choices or constraints for nonnumeric values. The text
separates suggestions from hard GUI/scientific boundaries, preserves the protected-case exclusion,
and explicitly disclaims policy-quality or qualification evidence. Changing any checked input still
invalidates readiness. Focused owner validation remains pending.

Long-form workspace pages embedded inside the scientific preview now use the main preview as their
single vertical scroll owner. Their content expands to full preferred height, their nested page bar
is disabled, and wheel input can propagate to the outer canvas. Standalone page behavior and
purpose-specific table, editor, plot, activity, and log scrolling remain intact. The focused owner
visual and automated checks remain pending.

Ribbon category pages no longer receive direct visibility changes that compete with Qt's internal
stack. Qt alone selects the visible page; all noncurrent pages are disabled, mouse-transparent, and
removed from focus as a defense against hidden control activation, with opaque theme backgrounds
providing paint containment. The focused contract specifically blocks the inactive `Train policy`
button while Compute is selected. Owner visual and automated validation remain pending.

The ribbon product heading is now a dedicated 42-pixel identity row above the category tabs. It
contains only the CALO-RPD product name, user-facing product version, and application-state badge,
with explicit light/dark surfaces and accessibility metadata. The added height separates product
identity from category navigation and makes the entire heading content visible. Focused owner
visual and automated validation remain pending.

The navigation tabs and pages are additionally parented by a dedicated opaque frame below that
identity row. The identity surface stays above the navigation sibling, preventing platform-style
tab or page paint from crossing the heading boundary and ensuring that no ribbon command occupies
the product strip. Focused owner visual and automated validation remain pending.

After the misplaced fragment persisted, the native composite ribbon `QTabWidget` was removed.
The remaining native category `QTabBar` is now also superseded: category selection uses exclusive
styled buttons, and command pages use a separately owned `QStackedWidget` below them. No native
selected-tab primitive remains that could paint into the product heading. Registered actions,
category navigation, keyboard focus, permanent expansion, and inactive-page interaction guards are
preserved. Focused owner visual and automated validation remain pending.

Numeric stepper chevrons are now drawn after the complete stylesheet-driven spin-box control,
directly inside the style-computed upper and lower button rectangles. This avoids reliance on an
optional native arrow-primitive request while retaining the current dimensions and palette-aware
normal, interactive, and disabled colors. Focused owner visual and automated validation remain
pending.

Ordinary scientific panels now use a content-first central surface: the single-workspace document
tab, duplicate CALO-RPD badge, global workflow banner, and global Continue control are absent and
reserve no height. Workflow state remains enforced, with the Dashboard's existing next-action card
as the sole continuation surface. The document tab row is contextual and appears only while a
genuine secondary document is open. Focused owner visual and automated validation remain pending.

Ribbon category selection now uses a transparent label with a two-pixel lower accent rather than
a rounded filled block. This prevents the top edge of the selected first category from resembling
a misplaced command beneath the product heading while preserving category selection, hover,
keyboard, and command-page behavior. Focused owner visual and automated validation remain pending.

The policy center and recovery/error paths are now consolidated. CALO Intelligence no longer
instantiates the hidden legacy training/qualification tree or retains its callbacks; it exposes one
independent training entry and one policy import. Policy-training Resume Center records can only
prefill the independent training state machine, with readiness and start still explicit. Resume
inspection is a clean summary with narrowly scoped optional details, and shared user-feedback
helpers keep technical exceptions in Activity Logs across scientific panels. Focused tests and the
ignored validator were updated but not executed, so current-source validation remains pending.

Ordinary GUI language is now governed by a product boundary: screens expose scientific choices,
capabilities, verification, compatibility, and experiment selection, while proposal/phase/build,
candidate, freeze, feature-flag, ABI, checksum, and source-authority terminology remains confined
to provenance, logs, schemas, and engineering records. The exact development identity and all
policy gates remain unchanged internally. Product-language render and source contracts are written;
owner execution remains pending.

Owner validation `phase6-20260813-183722` reached the version-identity contract after six passing
prechecks and stopped at command `05`. The sole reported mismatch was the absent `product_version`
field in the active development-status record; the technical build identity remained correct. The
status record, verifier history contract, and focused unit contract are synchronized. This is a
source correction only; a fresh complete owner validation is still pending.

The next owner run, `phase6-20260813-184612`, verified the identity correction and compilation,
then failed `07-ruff` after eight passing checks on three unused exception bindings and four unused
Qt imports. Those seven mechanical findings are corrected with used exception bindings preserved.
No later command ran, so format, test, render, build, and distribution status for the current source
remain pending until another complete owner validation.

Owner run `phase6-20260813-185633` subsequently passed through `07-ruff` and stopped at
`08-format` after nine passing checks. Ruff named 27 files; deterministic formatting was applied
to exactly that set. The formatter operation is not validation evidence, and commands `09` onward
remain pending for a fresh complete owner run.

Owner run `phase6-20260813-190343` then passed ten checks through `08-format`; command `09-unit`
reported 57 passing tests and two stale source-literal assertions. The permanent scientific canvas
already suppresses its redundant tab header when it is the sole document, and the benchmark tab is
already product-facing `Method verification`. The tests now assert those approved behaviors instead
of requiring the removed `DocumentBrand` chrome or development-facing `Freeze gate` label. No
runtime behavior changed, and command `09` onward requires a fresh complete owner run.

Owner run `phase6-20260813-191340` passed all 59 command-`09` tests, then command `10-phase6-gui`
remained incomplete until the owner interrupted it after approximately 54 minutes. The resulting
summary incorrectly said PASS because Ctrl+C bypassed command result creation and the validator did
not require later commands. That PASS is rejected. GUI tests now isolate the session-recovery
journal so a real user's unfinished-session dialog cannot enter an offscreen test, print each test
name, and terminate the dedicated test process with retained thread stacks if any test exceeds two
minutes. The validator now fails closed unless the exact `01`-through-`17` sequence completes. A
focused fixture also suppresses scheduled real-session recovery and hardware discovery so these
layout/product contracts do not depend on external startup state. Production behavior is unchanged.
A fresh owner run is required; commands `10` onward remain unverified.

Owner run `phase6-20260813-202657` confirmed the bounded/fail-closed validator, passed commands
`01` through `09` including 60/60 unit tests, and passed the first 18 of 21 command-`10` GUI tests.
The watchdog then returned code 124 during teardown of the eighteenth test after its assertions had
passed. That test uniquely shows the top-level window, and pytest-qt was invoking the full
production close path during cleanup. The focused fixture now detaches its temporary log handler
and directly accepts only pytest-owned teardown; production recovery, policy, persistence, safety,
and close behavior is unchanged. Commands `10` onward require a fresh owner run.

Owner run `phase6-20260813-205516` then passed commands `01` through `09`, including 60/60 unit
tests, and completed command `10` with 20/21 GUI tests passing in 25.77 seconds. Its only failure
was a stale `ceiling` tooltip assertion; the product now correctly presents the same admission rule
as an `available-memory safety limit`. The assertion is aligned. Expected empty portfolio selection
during panel setup is now a concise input prompt rather than ERROR-level technical logging, while
unexpected planner exceptions and invalid apply actions remain fail-closed. Commands `10` onward
require a fresh owner run.

Owner run `phase6-20260813-212634` passed commands `01` through `12`, including 61/61 unit tests,
21/21 focused GUI tests, 21/21 GUI regressions, and 9/9 empty-policy tests. The first failure was
`13-gui-render`: its expected information-control set omitted the Base architecture control even
though the live widget and command-`10` contract correctly include all 18 controls. The renderer is
aligned and now reports exact missing/unexpected keys for future drift. Commands `13` onward require
a fresh owner run.

Owner run `phase6-20260813-215626` subsequently passed the complete Phase 6 sequence with 113 tests,
offscreen renders, packaging, distribution checks, and stable source identity. Manual inspection
then exposed that readiness/start stages were only reachable through repeated ribbon clicks and had
no visible input-pane control. The ribbon is now navigation-only, while a persistent input-pane
footer visibly owns `Check readiness` and gated `Start training`; explicit start confirmation and
no-auto-selection remain unchanged. This post-pass source correction requires one fresh owner run.

Manual start review then reached the existing-output guard and revealed that its compatible-resume
choice was still hidden. The canonical controller checkbox is now visible, off by default, and
bound directly into Campaign inputs. Existing paths cannot start as new runs; resume requires an
existing directory and still passes the exact stored-plan/status/checkpoint integrity gates. Fresh
Browse proposes a new child path while resume Browse selects the interrupted directory. This
current-source follow-up requires one fresh owner run.

The current follow-up adds a per-user resumable-model library. New TSH-CALO runs propose unique
children of the OS-managed CALO-RPD application-data `training-models` directory; explicitly added
directories join future scans without copying or relocating their campaigns. The picker admits
only stored `running` or `interrupted` plan/status pairs, preserves stored source identity, selects
their original output directory, and visibly enables the canonical exact-resume choice. CALO
remains built in and does not train. Dirty-source readiness remains fail-closed, but its traceback
is confined to Activity Logs while the visible status explains the corrective action. Focused
contracts are written; current-source owner validation is pending.

The policy-training pane is now unambiguously TSH-CALO-only. The inapplicable CALO/TSH-CALO
architecture selector and its help control were removed, and the training launch model no longer
accepts architecture as mutable input. Rule-based CALO remains available through ordinary algorithm
and experiment selection and is not changed by this GUI correction. All TSH-CALO readiness, exact
resume, qualification, activation, protected-case, and no-auto-selection boundaries remain intact.
Focused contracts are updated; current-source owner validation is pending.

The TSH-CALO-only training-editor startup order is corrected after a manual native launch exposed
that initial selection refreshed the editor before its status/action widgets existed. Initial
selection now follows construction of those controls, and focused static/offscreen contracts cover
the ordering and first visible readiness state. No execution was performed by the development
agent; current-source owner launch and validation remain pending.

Independent-training readiness now applies the same current Safe-80 network estimate and
CUDA/CPU device admission as actual trainer construction, releasing a readiness-only lease before
returning. Therefore an over-limit configuration cannot pass readiness and then fail immediately
after being reported as running. Fresh GUI plans now bind rollout memory to actual retained policy
transitions (`evaluations / population - 1`) rather than raw candidate evaluations, without changing
the requested evaluation budget or 80% ceiling. Focused contracts are written; owner execution and
validation remain pending.

Owner run `phase6-20260814-003900` passed the environment, ignored-path, active-status, and compile
checks before command `07-ruff` first failed on an undefined lowercase `root` used by the new
offscreen resource-contract reads. The retained command log confirms the visually adjacent unit
paths were passed separately. The contract now uses an explicit repository root derived from its
source file; no development-agent execution followed, and a fresh complete owner rerun is pending.

Authorized run `phase6-20260814-004621` passed through 73 command-`09` unit tests and completed
command `10` with 22/24 GUI tests passing. The two focused failures were isolated GUI-test settings
retaining an earlier scan location in the same Qt process and one visible resume tooltip using the
internal term `checksum`. The fixture now clears only temporary test settings per window, and the
tooltip uses saved-file-integrity language while internal verification remains exact. A complete
rerun is pending; no prohibited workflow executed.

Authorized run `phase6-20260814-004927` then passed the complete command `01` through `17`
sequence: 73 unit tests, 24 focused GUI tests, 21 affected GUI regression tests, 9 integration
tests, all three offscreen render modes, build, distribution verification, and nonignored-source
stability. This closes the Phase 6 automated GUI/native/packaging validation gate for the tested
source. Human usability, screen-reader, and scientist acceptance remain separate, and no policy,
protected-case, Docker, CUDA-campaign, publication, or release workflow was executed.

The subsequent saved-policy discovery correction includes completed campaigns in bounded scans,
persists explicit scan locations synchronously, refreshes both saved-training and Policy-library
views after a successful output, and represents an unregistered completed candidate without
silently changing its lifecycle. The single existing Import action remains the explicit
registration boundary; qualification, selection, activation, experiment binding, and completed-
identity resume remain blocked. Focused Ruff/format, 18 command/native, and 25 GUI contracts pass;
a fresh complete Phase 6 validator run remains pending for this follow-up source.

Screenshot inspection then exposed a theme-wide checkbox visibility defect: native state marks were
present, but the indicator boundary blended into the surface. The global Qt proxy style now draws a
palette-aware rounded boundary and surface for every checkbox, with vector checked/partial marks and
distinct hover, focus, pressed, and disabled treatment in both themes. Radio-button rendering and
all scientific, training, and lifecycle semantics are unchanged. Focused source and light/dark
render-contract updates are implemented but not executed; the complete current-source Phase 6
follow-up validator remains pending.

Subsequent user inspection exposed a semantic presentation defect in the Recovery row. The disabled
exact-resume checkbox shown for `New training` was being interpreted as disabled recoverability,
even though the campaign runner already writes a verified recovery point after every safely
committed training window. Fresh training now displays a checked, fixed automatic-recovery status;
the interactive exact-resume control appears only for a selected `running` or `interrupted` saved
campaign, and completed campaigns show resume as inapplicable. The `--resume` boundary, checkpoint
writes, deterministic accounting, and integrity rejection paths are unchanged. Focused contracts
are updated but unexecuted; complete current-source Phase 6 follow-up validation remains pending.

## Invariants

1. No old result, experiment, policy, or fingerprint is silently rewritten.
2. A changed scientific method receives a new algorithm/schema version and new evidence.
3. Training never auto-activates a policy.
4. A new experiment snapshots its governing policy and scientific protocol immutably.
5. Available-memory percentages are admission ceilings, never forced utilization targets.
6. CPU/GPU timing comparisons never mix fallback modes without explicit stratification.
7. Generated checkout files are not treated as packaged release contents.
8. A release freeze is generated from staged artifacts only after all gates pass.

### 2026-08-14 - Current Phase 6 GUI/native/packaging follow-up gate closed

Authorized retained run `validation/logs/phase6-20260814-131637` passed the exact command `01`-`17`
sequence against stable nonignored source: 74 unit, 25 focused GUI, 21 GUI-regression, and 9
empty-policy/training-navigation integration tests; composited light/dark checkbox-state renders;
fresh wheel/sdist build; and both distribution verifiers. The global checkbox-border and fresh-
training automatic-recovery presentation follow-ups are therefore automated-validation complete.
No scientific/policy, protected-case, Docker, CUDA-campaign, publication, or release gate was
executed or closed, and automated evidence does not establish human acceptance.

### 2026-08-14 - Finite-plan checkpoint-safe pause follow-up pending validation

The training runner now records durable structured progress at each committed recovery checkpoint
and accepts a campaign/plan-bound cooperative pause request only at that safe boundary. A confirmed
pause retains the committed percentage and becomes explicitly resumable; force-stop behavior inside
an uncommitted CUDA window remains non-resumable. Exact resume has no cycle-count limit, but every
cycle continues the immutable finite candidate-evaluation budget, so no infinite training semantics
were introduced. Synthetic parity and repeated-pause tests, GUI progress/pause contracts, and the
ignored validator revision are implemented but unexecuted. The completed
`phase6-20260814-132200` evidence is therefore historical for this follow-up; the gate remains open
until a fresh complete owner-run Phase 6 bundle passes. No policy or scientific workflow was run.

### 2026-08-14 - Completed-model extension gate open pending fresh validation

Completed campaigns produced by current source now bind every member's full final trainer/session
checkpoint into the completion manifest. An explicit child extension must authenticate that manifest,
the unchanged plan/design/device conditions, checkpoint hashes, receipt counts, and exact finite
segment budget. It preserves its parent, writes a separately identified child manifest/candidates,
and advances cumulative FE accounting without a segment-count ceiling. Paused children use the same
safe checkpoint protocol. Legacy completed artifacts without authenticated optimizer/RNG checkpoint
state are deliberately non-extendable. Extension does not establish improvement or any policy
lifecycle authority. Source and synthetic contracts are implemented but unexecuted; the gate remains
open until a fresh complete owner Phase 6 evidence bundle passes.

### 2026-08-14 - Responsive training and governed policy-library gate open pending validation

The left training footer now delegates live percentage to the single persistent bottom status bar
and detailed checkpoints to Activity, while retaining the safe-pause action. Responsive size-policy
and constrained-render contracts cover the fixed-width input pane. The Policy library lists all
completed campaigns, merges imported registry state, and exposes explicit import, qualified-only
activation, and exact unregistered-campaign file deletion. Deletion is irreversible but runtime-
confirmed and fail-closed for scan roots, symlinks, incomplete state, and registered/active policy
paths. Applying a ready governing policy binds it to experiment settings and exposes Power System
as the next explicit page without starting a case, power flow, evaluation, or experiment. These are
implemented source and synthetic contracts only. The prior `phase6-20260814-132200` bundle predates
them; this gate remains open until a fresh complete owner validator bundle passes.

### 2026-08-14 - Full-width Policy library and Governing policy layout gate open

The Policy library table now has no internal horizontal or vertical scrollbar and its height tracks
the header plus all currently visible entries. Removing its vertical stretch allows the library and
governing groups to keep their preferred content height while both expand across the page. The
governing form uses expanding fields, and the main preview retains sole page-level scroll ownership
when many entries make the whole page taller than the viewport. Constrained GUI, row-height,
scrollbar-policy, full-width geometry, static, and offscreen contracts are implemented but
unexecuted. The current gate remains open until a complete owner Phase 6 validator bundle passes.

### 2026-08-14 - CALO Intelligence reachability and removal-control gate open

The narrow input pane now gives its full field width to the default saved-training path and reserves
the path's calculated wrapped height. Policy Library retains an always-visible `Delete model files`
control for exact unregistered completed campaigns and adds a separate `Review policy removal`
control for registered entries; the former is disabled when retirement authorization is required.
Scrollable pages synchronize their current content height into the central workspace stack, giving
the main preview sufficient range to reach the entire Governing policy block above Activity even as
the entry-sized table grows. Focused responsive-path, action-state, dynamic-height, and bottom-
reachability contracts are implemented but unexecuted. The gate remains open pending a fresh full
owner Phase 6 validator bundle.

### 2026-08-14 - Imported completed-candidate removal and active reveal gate open

The previously disabled registered-row deletion control now uses a narrow, exact removal contract.
Only an inactive, unqualified, SHA-256-valid imported completed candidate with no qualification,
experiment, lineage, or sibling-registration reference may proceed. One confirmation atomically
suppresses/removes that exact registration before deleting the already verified completed campaign
directory; partial filesystem failure leaves the campaign discoverable as an unregistered row for
safe retry. Other registered policy state remains routed to reviewed retirement. Selection and page
display also request outer-preview visibility for the complete Governing policy group with explicit
bottom clearance above Activity. Source and synthetic contracts are implemented but unexecuted; the
gate remains open until a fresh complete owner Phase 6 validator bundle passes.

### 2026-08-14 - Scroll-preserving model deletion and cross-version extension gate open

This corrective source pass supersedes the automatic reveal and two-button removal presentation
above. Selecting a Policy Library row no longer calls `ensureWidgetVisible` or changes the outer
main-preview scroll value. Dynamic page-height synchronization remains, so manual page scrolling
can reach the complete Governing policy block. The redundant `Review policy removal` GUI action is
removed; one confirmed `Delete model files` action now accepts an exact unregistered completed
campaign or an inactive, unqualified, unreferenced, checksum-valid completed/standalone candidate,
including the first displayed row. All active, qualified, evidence-bound, lineage-bound, ambiguous,
symlink, scan-root, incomplete, or integrity-invalid targets fail closed. The retirement inventory
and dry-run CLI remains separate and requires its existing explicit authority.

Extension admission also no longer requires the currently checked-out source commit to equal the
campaign's origin commit. Both origin and executing commits are retained as provenance. Admission
instead authenticates exact plan values, the frozen algorithm/policy/state/action/training-
environment architecture identities, resume checkpoint identity, and the complete training,
trainer, session, and environment parameter field sets. Campaign/source/freeze identity and
the reserved `writer_metadata` namespace remain provenance rather than parameter-schema authority;
the campaign schema itself remains binding and may change only with training semantics.
A software revision with the same contract remains extendable; architecture changes or
added/removed training parameters fail closed. Source and synthetic
contracts are implemented but unexecuted. This gate remains open until a fresh complete owner-run
Phase 6 validator bundle passes; no policy/scientific workflow or deletion was executed here.

The completion audit further binds parameter names, tensor shapes, and dtypes from the retained
policy state and ignores unrelated compatibility-writer metadata. Both the GUI readiness path and
the `--extend` command use the metadata-tolerant authenticated plan parser. A completed campaign's embedded
legacy freeze and acceptance hashes remain authenticated lineage; `--extend` does not require the
user to reselect historical authority files. Fresh training and unfinished base-campaign resume
retain the stricter external-authority requirement. These additions remain source/test contracts
only until the same complete owner-run validator passes.

### 2026-08-14 - One-action Policy Library qualification-admission gate open

CALO Intelligence now exposes one user-authorized `Qualify policy` transaction. It checks the exact
immutable candidate, inventories every current non-ignored source file, and creates a separate clean
deterministic internal Git snapshot without modifying or committing the development worktree. The
snapshot commit and full file/SHA-256 manifest become the exact qualification source identity. The
transaction then creates and freezes the candidate-bound A-E component-ablation plan, starts or
exactly resumes its retained finite cells, rejects the
candidate if any approved component fails; then creates and freezes the formal paired qualification
plan, starts or exactly resumes it, re-verifies completed evidence, and atomically admits only a pass.
The fixed protocol uses case30/case57, 30 paired runs per case, population 20, and 10,000 exact
evaluations per optimizer cell: 480 A-E cells followed by 120 formal cells. Interrupted execution has
no resume-count ceiling, but no resume may alter the finite plans, seeds, completed cells, or FE budget.

The former separate `Check formal plan`, `Run / resume qualification`, and `Admit passed evidence`
controls are removed. Automatic advancement is confined to the explicitly requested qualification
transaction and its checksum-bound stages. It never activates or binds a policy. A verified admitted
pass only enables the separate `Activate for experiments` action; governing-policy Apply remains the
separate immutable experiment handoff.

Admission recomputes the canonical qualification decision and validates exact policy, plan, source,
seed, evidence, A-E, OOD calibration, receipt, paired-cell completeness, and protected-case closure
before one atomic registry transition. Comparison has no promotion authority. It labels a unique
leader only when that policy Pareto-dominates all policies with the same case/seed/budget/analysis/
threshold design; incomparable designs and metric trade-offs remain explicit scientist decisions.
Software version, training age, and training duration are not selection metrics. Source and
synthetic test contracts are implemented but unexecuted; the gate remains open until a fresh full
owner Phase 6 validator bundle passes. No real A-E campaign, qualification, admission, activation,
policy binding, or experiment workflow was run here.

### 2026-08-14 - Policy Library cumulative training-evaluation accounting gate open

Each native TSH-CALO library row now derives a model-specific cumulative candidate-evaluation count
from authenticated training episode receipts bound to the exact candidate SHA. A completed extension
child includes the base plus each completed finite extension segment; the parent remains immutable.
Qualification and experiment evaluation counts are outside this field. Unverifiable or legacy
accounting is reported as unavailable rather than inferred. Source and synthetic contracts are
implemented but unexecuted, so this gate remains open pending a fresh complete owner Phase 6 bundle.

### 2026-08-14 - Stage-neutral qualification compatibility gate open

Development phase, policy age, and originating software revision no longer decide whether an
immutable candidate may enter formal qualification. Admission requires the current frozen TSH-CALO
runtime ABI, exact candidate checksum, validated epistemic-ensemble structure, authenticated member
training receipts, and protected-case isolation. `Qualify policy` exposes ABI/file/ensemble/source
blockers before evaluation and then owns only the frozen qualification transaction. Automatic
passed-evidence admission is part of that explicit transaction; activation, qualification receipt/
calibration binding, and experiment binding remain separate fail-closed gates.

Resume and extension are not relaxed by this qualification change. They remain bound to the retained
training compatibility contract and exact persisted continuation state. A source revision alone does
not break compatibility; architecture, parameter layout/schema, persisted-state schema, or evaluation-
accounting changes do. Source and synthetic contracts are implemented but unexecuted; fresh complete
owner Phase 6 validation is pending.

### 2026-08-14 - Architecture-bound qualification and observable-run gate open

New automatic qualification plans no longer depend on the historical component-development gate.
The frozen candidate contract binds immutable architecture, state/action/training schemas, ensemble
membership, feature configuration, authenticated training-design provenance, and exact checkpoint
identity. The only optimizer stage is the predeclared 120-cell paired model-quality campaign; its
feasibility, exact-FE, objective, effect-size, significance, anytime, independent-validation, OOD,
and protected-case gates remain unchanged. Historical component-bound evidence is readable only for
backward verification.

Qualification publishes committed/failed retained cell count as a real percentage in the global
bottom bar. All task-changing workspace surfaces are disabled while any foreground task is busy,
while Activity and the global status bar remain enabled for detailed inspection and safe task
control. Source and synthetic contracts are implemented but unexecuted. This gate remains open until
a fresh complete owner Phase 6 validator bundle passes.

### Qualification micro-progress and safe-pause follow-up (2026-08-14)

The fixed 120-cell quality design and all scientific thresholds remain unchanged. Each optimizer now
uses its existing exact-run state envelope at a 500-evaluation formal-cell interval. Structured live
events are durably appended and sent to Activity; the bottom bar reports both live FE progress and
the independent durable-cell count. A cooperative pause is acknowledged only at a completed
population transition after the checkpoint or cell record exists and its SHA-256 is recorded.
Exact resume restores the same population, adaptive state, histories, FE counter, and RNG streams;
it neither restarts the cell nor changes the finite budget. Source and synthetic contracts are
implemented but unexecuted, so this gate remains open for the complete owner Phase 6 validator.

### Input-generated new-training plan follow-up (2026-08-15)

The ordinary new-training UI no longer exposes external scan-location or settings-template
management. Visible campaign, eligible-case, ensemble/member-seed, finite budget, compute/fallback,
and PPO/model inputs are combined with application-owned safe resource, schema, provenance, and
resume defaults to construct a fresh internal plan at readiness. Retained campaigns continue to
load their own immutable plan internally for exact resume or compatible finite extension. A removed
selected campaign now resets to New training instead of leaving an invisible stale plan path.
Synthetic source/GUI/offscreen contracts are implemented but unexecuted; G6/Phase 6 validation
remains open pending a fresh complete owner validator bundle.

The first owner GUI inspection of that follow-up exposed two readiness defects. The shared
`plan_error` presentation called a fresh-plan construction failure a saved-plan load failure, and
fresh-plan source identity was resolved from the launch working directory instead of the imported
checkout. The UI now distinguishes those error domains, while the source identity lookup is anchored
to the package root with the existing immutable build-declaration fallback. Contracts are written
but unexecuted; G6 remains open pending the fresh complete owner validator bundle.

### Transactional qualification-evidence gate open (2026-08-15)

The corrected runner uses canonical plan/policy/case/run/side/seed/budget cell identities, mutually
exclusive atomic success/failure commits, a checksum-bound unique-cell index, and a separate
infrastructure-incident state. Event, callback, status, index, or finalization faults cannot be
graded as scientific cell failures. Current-schema evidence is admission-ineligible until a final
completion authority binds the exact plan, seeds, terminal index, event log, status, evidence, and
receipt with zero infrastructure incidents. Contradictory retained terminal artifacts fail closed.

The observed `e266bd7598befa54` campaign is preserved without repair as infrastructure-aborted and
cannot resume. A new corrected-source run must retain every operative frozen design field while
receiving new run/source provenance identities. Implementation and synthetic fault-injection
contracts exist but were not executed. This gate remains open until the Git-ignored transactional
qualification validator passes and its complete fresh log directory is reviewed; no qualification,
policy admission, activation, protected-case, release, or scientific claim follows from source work.

### Feasibility-assessment and influence-analysis gate open (2026-08-15)

The current campaign finalizer now emits checksum-bound feasibility ratings and explicitly records
`automated_suitability_decision: null`. It does not issue a quality pass, rejection, grade, policy
selection, activation, or experiment binding. Overall feasibility is the exact candidate-cell full-
feasibility percentage; first-feasible reach/efficiency, independent validation, objective coverage,
and case-specific percentages remain separately inspectable. The complete frozen scientific design
and transactional evidence requirements are unchanged.

Verified admission creates only an inactive assessed dossier. Scientist selection and activation
are separate exact-identity lifecycle transitions, and experiment binding still requires both the
recorded selection and explicit activation plus re-verification of candidate, evidence, receipt, OOD,
and compatibility bindings. Training-parameter influence uses only immutable authenticated plans and
matching rating/protocol schemas, reports per-rating univariate observational associations with
minimum cohort/variation requirements, and performs no tuning or training mutation. Protected cases
remain unavailable to influence analysis.

Source and synthetic contracts are implemented but unexecuted. This gate remains open pending a
fresh owner run of the ignored validator and review of its entire timestamped log directory. No new
feasibility assessment, policy selection, activation, or release claim is authorized by source work.

The Policy Library follow-up removes the feasibility-comparison and archive buttons, hides the
activation control until scientist selection or retained legacy qualification makes activation
eligible, and places all remaining controls in one row. This is presentation-only; the assessment,
selection, activation, deletion protection, and frozen scientific gates remain unchanged and open
pending the same owner validator.

Owner attempt `feasibility-influence-20260815-203150` is infrastructure-incomplete: PowerShell 5.1
terminated on Git's LF-to-CRLF stderr warning during `02-diff-check`, before Ruff or tests. The
ignored validator now logs native stderr without treating it as failure when the native exit code is
zero. This gate remains open pending a fresh complete run; the incomplete bundle is not a test pass
or failure of production behavior.

Owner attempt `feasibility-influence-20260815-203426` advanced through diff-check with exit code 0
and stopped at Ruff on a missing checksum-helper import in synthetic lifecycle test source. That
import is corrected; no production/scientific semantics changed. Later commands did not execute, so
the gate remains open for a fresh complete run.

Complete run `feasibility-influence-20260815-204719` passed Ruff, format, 60 focused unit and
fault-injection contracts, 36 GUI contracts, and nonignored source stability. All six commands exited
zero. Retained environment flags prove the validator used only deterministic synthetic fixtures and
did not run a real feasibility/qualification campaign, train or mutate parameters, use a real policy,
open protected cases, select/activate/bind a policy, run Docker, or establish release/scientific
evidence. The transactional feasibility/influence implementation gate is satisfied subject to one
source-stable replay after this required ledger closure; broader Phase 6, scientific-assessment, and
release gates remain separate and are not closed by this focused evidence.

The subsequent owner-directed removal of **Show archived** changes the visible Policy Library and
therefore supersedes the `feasibility-influence-20260815-205030` source identity for this follow-up.
The current library and scientist-facing influence cohort request non-archived records only;
historical archive storage remains internal and no data migration or deletion occurs. Updated
static/GUI contracts and validator coverage are implemented but unexecuted. A fresh complete focused
validator PASS is required before treating this follow-up as validated; scientific, activation,
broader Phase 6, and release gates remain separate.

The next follow-up separates fresh and resume intent for multi-cell feasibility execution. Resume is
visible only for one exact authenticated safe pause; ambiguous, unpaused, completed, or incident
state is not resumable through the UI. Fresh start allocates a new run identity and, after explicit
confirmation, permanently removes only canonical candidate-bound incomplete/resumable workspaces.
Completed evidence, shared source snapshots, integrity failures, and immutable infrastructure
incidents remain protected. Frozen scientific and training/architecture semantics are unchanged.
Production/test/validator source is implemented but unexecuted, so a new complete focused validator
PASS is required before this exact behavior is relied upon.

Authorized attempts `feasibility-influence-20260815-215112` and `-215150` reached only the format
gate: diff and Ruff passed, and Ruff format identified the GUI panel as the sole mechanically
unformatted file.
After that formatting correction, complete replay `feasibility-influence-20260815-215306` passed
Ruff/format, 61 focused unit and fault-injection contracts, 37 focused GUI contracts, all command
exit codes, and nonignored-source stability. Retained flags prove that no real assessment,
qualification, training/parameter mutation, real-policy use, protected-case access, selection,
activation, binding, Docker, or release workflow occurred. The focused engineering gate for the
fresh/resume and non-archived UI behavior is satisfied, subject to the source-stable complete replay
produced after this ledger closure. Scientific assessment, activation, broader Phase 6, and release
gates remain separate and open.

### 2026-08-16 - Live-refresh and navigation simplification gate open

The current GUI source makes refresh a fresh read boundary: Results reloads the experiment selector,
Policy/Saved training invalidates cached file-integrity observations before rescanning, and each
workspace activation reloads its applicable read-only source state. Resume Center is retired from
construction and command metadata; authenticated policy resume and finite extension remain in the
explicit Train policy workflow. Home contains only Overview, Open, and Save. Schema-4 workspace
migration maps historical Resume Center identities to Overview and preserves the later historical
index meanings. No resumable data or lifecycle authority is removed. Source and synthetic contracts
are implemented but unexecuted, so this engineering gate remains open pending a fresh complete
owner-run Phase 6 validator. Scientific, lifecycle, qualification, protected-case, release, and
human-acceptance gates remain separate.
