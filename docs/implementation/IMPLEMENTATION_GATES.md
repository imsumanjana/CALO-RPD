# CALO-RPD modernization implementation gates

This file controls implementation of the remediation plan in
`docs/COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md`.
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
| G0 | Complete | Baseline commit and dirty-tree ownership recorded. At the user's direction, continuation work is committed directly on local `main`; the untracked user-owned `Docker_Build.txt` remains untouched. |
| G1 | Implemented, final release check pending | Current configuration contains no utilization, memory-percentage, lane-share, device-job-count or work-stealing knobs. Historical fields are accepted only at the strict loader migration boundary and discarded. SQLite now uses schema version 1, creates and integrity-checks an online backup before populated v0 migration, records its SHA-256 receipt, migrates transactionally, preserves representative legacy rows, reopens idempotently and rejects future schemas without mutation. Historical release artifacts remain immutable. |
| G2 | Implemented; bounded host physical evidence retained, container repetition pending | Shared device lease, 80%-of-free VRAM/RAM admission, CUDA-resident/staged-host/CPU-fallback states and focused regression tests exist. Clean commit `63f56ad` passed bounded physical FP64 CPU/CUDA evaluator parity on the RTX 4060 for development case30 and case57. Clean commit `d6a950c` passed bounded real VRAM pressure/recovery, actual host-staged CUDA execution, controlled OOM backoff, controlled clean CPU restart plus CUDA recovery, and cross-process lease contention/release. Clean commit `67bd18e` completed a 3,600-second physical CUDA soak with 3,600 GREEN samples, no protection stop, verified hash-chained provenance, observed temperature/power telemetry, and scoped GPU-board-energy integration. Controlled faults are explicitly not natural-hardware-OOM evidence; energy is not whole-system energy. Repeat source-bound parity/resource/lease/soak validation in the container/WSL2 target. |
| G3 | Implemented | New schemas expose only `cuda_preferred` and `cpu_only`; old CUDA modes migrate to `cuda_preferred`; historical XPU modes remain readable but validation rejects them as view-only. No executable XPU source/import remains. |
| G4 | Source-bound CPU/CUDA images and core runtime/security gates retained; GUI/soak completion pending | Hardened CPU/CUDA profiles retain local-only noVNC/VNC, UID/GID 10001, dropped capabilities, no-new-privileges, read-only root, bounded `/tmp`, `/data`, shared-volume leases, one GPU, host-RAM ceiling, health and runbook contracts. Exact clean commit `1f02a94` produced separate attested OCI and runtime-loaded CPU/CUDA images with immutable source declarations, BuildKit maximum provenance/SBOM, metadata, filesystem manifests, CycloneDX and complete Trivy JSON. Both 336-file privacy audits passed; both pinned local Trivy gates retained 700 advisories and passed with zero fixable HIGH/CRITICAL findings. CPU-only and physical one-RTX-4060 CUDA smoke passed; independent-container lease exclusion/release passed; container case30/case57 parity and resource recovery passed. The exact CUDA image one-hour soak is running. noVNC GUI interaction/restart and final soak audit remain before this workstation image gate closes; CI execution on the exact eventual candidate remains separate. |
| G5 | Implemented | Study-strength protocols are validated on a deep copy, display a scientist-readable before/after diff, then atomically replace shared configuration and propagate through state signals. Run counts use a persisted paired-effect/power/Holm planning approximation, preserve governing-policy binding, and cannot be reduced by a fixed legacy evidence profile. Final run snapshots remain immutable in the experiment database. |
| G6 | Implemented; Linux packaged-lane execution pending | Normal experiment UI exposes two compute choices and no device percentages/batches/schema controls; policy UI hides No-AI/unqualified and routing internals; Dashboard readiness exposes available memory, admission status and recoverable queue progress instead of utilization/worker engineering. A rendered-widget contract checks Dashboard, Experiment Manager, Portfolio Manager, CALO Intelligence and Benchmark & Evidence for venue/development/backend/schema/XPU/Safe-80/utilization language. The complete Windows/offscreen GUI suite passes locally (33 tests) and produced a validated 1440x900 dashboard PNG. CI persists the corresponding Linux rendering plus accessibility evidence; its first packaged execution remains pending. |
| G7 | Implemented | Policy training remains independently configured; qualified active-policy binding is synchronized into every new experiment while stored experiment snapshots remain immutable. |
| G8 | Complete | On 2026-08-03 the scientific lead stated exactly: “Approve TSH-CALO A–E, with F experimental and evidence-gated.” The current CALO is frozen as the baseline. TSH-CALO will use new algorithm/state/action/policy ABI versions, cannot auto-activate, and cannot inherit old superiority evidence. The required nine-part runtime/training confirmation was presented before upgraded implementation. |
| G9 | In progress — valid negative screen retained; candidate rejected for formal qualification under the frozen design | A–F mechanics remain green and F remains experimental/off by default. `TSHCALOOptimizer` still requires a separately qualified, explicitly activated immutable ensemble in ordinary execution. Real campaign v2 completed five distinct IEEE 30/57 CUDA members and an immutable unqualified ensemble with exact aggregate 100,000 FE/scenario calls. Valid screening v3 completed all 40 paired cells at exactly 2,000 FE with zero failures and independent validation of every retained solution. Case30 produced no feasible run in either arm; case57 produced a `1.149%` median paired improvement whose 95% interval crossed zero and whose Holm-adjusted `p=0.052734375` missed the frozen threshold. The evidence decision is grade `U`, score `0`, `passed=false`, and `no qualification or policy-benefit claim`; no receipt, registration, or activation occurred. Therefore formal qualification and protected-case opening are not permitted for this candidate/design. Physical target CPU/CUDA parity/pressure, counted Jacobian availability for E, A–E ablation benefit and any qualification remain absent. |
| G10 | Partially complete | Statistical corrections, honest claim boundaries, power-aware planning and a frozen-design preregistration protocol exist in `SCIENTIFIC_VALIDATION_PROTOCOL.md`. The runtime-enumerated 22-method campaign defaults to 98 initiated paired runs for 21 CALO-versus-comparator tests at effect 0.50, 95% power, Holm family control and 10% failures; pilot/simulation designs require an evidence SHA. Immutable campaign design is hashed while runtime status remains updateable. case30/57 are validation replays, case118/300 are protected tests, and a confirmatory plan cannot omit all protected tests or relabel one. Source-traceable L-SHADE 1.0.1 supplies corrected success-history DE mechanics on CPU/tensor paths. The official pinned pycma 4.4.4 engine supplies active CMA-ES with a disclosed feasibility-first dense-rank adapter, latent mixed-variable encoding, CPU control residency and CUDA-capable common evaluation. Both have deterministic snapshots and focused formulation tests. External campaign execution, mathematical-solver comparisons, approved-architecture ablations and physical qualification remain. |
| G11 | Harness partially implemented | A fresh dedicated `artifacts/python-dist` stage is required to be absent before each build, preventing obsolete distributions from entering the new wheel/sdist manifest. Generated policy checkpoints, lineages and training metadata are explicitly excluded from packages. The CPU smoke container generates its filesystem manifest from the built `/opt/calo` tree. CI uploads those staged records with the GUI rendering and CycloneDX SBOM. Final release freeze, actual image digests/attestations, clean-machine reproduction and requirement-by-requirement closure still follow G9/G10. |

Latest verification evidence:

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
  per-scenario call totals; E runtime mask; pre-solve F rejection; pending-observation/RNG/component
  resume; static absence of experiment, registry, activation and production-inference authority; and
  poison-on-solver-failure provenance. **8 passed** dedicated cases, **29 passed** with adjacent
  training/transition/context guards and **555 passed, 63 skipped** on the active tree. Ruff
  lint/format passes across **395 Python files** and the generated schema is current. No fresh
  training, qualification or benefit evidence exists.
- Independent trainer Safe-80 admission: training ABI
  `tsh-calo-training-v3-counted-safe80-receipts`; hash-bound rollout/population/topology/scenario envelope;
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

## Invariants

1. No old result, experiment, policy, or fingerprint is silently rewritten.
2. A changed scientific method receives a new algorithm/schema version and new evidence.
3. Training never auto-activates a policy.
4. A new experiment snapshots its governing policy and scientific protocol immutably.
5. Available-memory percentages are admission ceilings, never forced utilization targets.
6. CPU/GPU timing comparisons never mix fallback modes without explicit stratification.
7. Generated checkout files are not treated as packaged release contents.
8. A release freeze is generated from staged artifacts only after all gates pass.
