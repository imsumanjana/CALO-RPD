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
| Physics-informed repair | Counted runtime/training and paired-evidence implementation locally verified; disabled by default | Runtime v1.1/training v4; final counted Newton Jacobian; analytic relaxed-control plus active voltage/angle/generator/thermal constraint derivatives; Q-limit-switched control masking; bounded lattice/continuous trust budget; exact FE/scenario accounting; no hidden solve/evaluator call; dynamic fail-closed masks; `ae7b304` incremental E comparison against the frozen A–D path | Execute the frozen incremental-value/cost comparison on a fresh E-enabled candidate and target CPU/CUDA path; no benefit claim yet |
| Evidence-driven population schedule | Experimental mechanics locally verified; disabled by default; promotion evidence absent | `tsh_calo_population_schedule.py`; separate `enabled` plus `experimental_mode` gates; preregistered design hash; feasibility/archive/diversity/budget/spacing conditions; deterministic feasibility-first contraction; no hidden FE; exact resume; nine focused tests | Must earn inclusion through paired-seed anytime/feasibility evidence without unacceptable cost, instability, overfitting or regression; otherwise keep disabled or remove |
| Versioned TSH-CALO candidate lifecycle | Locally verified through unqualified evaluation; negative screening evidence retained | `tsh_calo_policy_artifact.py`, `tsh_calo_qualification.py`, `tsh_calo_qualification_campaign.py`, algorithm-aware registry; exact ABI/provenance; protected-holdout rejection; non-serializable qualification-only evaluation capability; OS-released single-writer lease; failed-integrity resume rejection; valid v3 screen completed 40/40 records and correctly emitted no receipt | Candidate v2 remains unqualified/inactive and is barred from formal qualification under this design; the raced v1 screen remains barred from scientific use; a future candidate must satisfy a new frozen screen and direct A–E evidence before formal qualification |
| Independent TSH-CALO policy training | Versioned counted-E mechanics complete; historical five-member ensemble remains negative/unqualified | `tsh_calo_training*.py`, `scripts/train_tsh_calo.py`; v4 counted-physics/Safe-80/receipt ABI; exact source/scientific/execution/seed/curriculum hashes; protected-identity guards; canonical rewards and exact FE/scenario/update/reward accounting; authenticated resume; CUDA-first Safe-80; opt-in E/no-feasibility authority; F rejected; unqualified-only output; no lifecycle authority. Historical v2/v3-ABI ensemble completed 100,000 FE/scenario calls and remains immutable under SHA `3adf5017dc51f33d76214aeb505da598984b9da0e4263e1ee8fe59a667180ceb` | Freeze a new non-tuning counted-v4 plan and train one fresh A–E candidate; then execute device equivalence, A–E evidence and qualification in order. Old candidate remains inactive |
| Immutable TSH-CALO ensemble inference and fallback | Integrated into production, qualification, component-ablation and device-equivalence boundaries; fresh-candidate physical equivalence pending | Ensemble assembly/load; activated-qualified production binding guard; non-serializable development capabilities; exact SHA/ABI/feature/member/calibration checks; CUDA-first current-free-VRAM 80% admission; governed CPU fallback; ensemble disagreement/shield; pre-evaluation block or explicit frozen-CALO relaunch; deterministic traces and exact resume; `e77431e` compares exact actions and bounded numerics while requiring dedicated VRAM | Execute `calo-rpd-tsh-device-equivalence` on the future counted-v4 candidate; candidate v2 failed screening and cannot be reused or activated |
| Reuse already-counted power-flow state for topology/physics context | Locally verified through topology and opt-in Change E paths | `ORPDProblem.evaluate_with_context`; default topology-only path retains no derivatives; explicit E path retains ephemeral final Jacobian/control/constraint context; base/weighted selection; optimizer/training construction; exact FE/scenario accounting; no serialized solver objects or hidden PF calls; case30 analytic sensitivity matched finite differences below `1.6e-8` max absolute error | Target CPU/CUDA execution and incremental-value/cost evidence remain required |
| Correct paired statistics, effect estimates, CIs, multiplicity control, power and anytime metrics | Locally verified at harness level | `statistics/`, campaign design, `SCIENTIFIC_VALIDATION_PROTOCOL.md`, statistical tests | Execute final frozen campaign |
| Modern strong stochastic baselines | Locally verified at implementation level | Source-traceable L-SHADE 1.0.1 and pinned pycma 4.4.4 CMA-ES with deterministic snapshots | External benchmark execution |
| Deterministic/mathematical reference solutions and broader licensed case corpus | Implementation partial; development adapters physically exercised | Protocol and pinned official PGLib-OPF v23.07 validation assets; exact license/source provenance and restricted parser. Clean `07f9476` adds SciPy SLSQP local continuous-relaxation plus exhaustive all-discrete adapters, separate original-lattice validation, exact call/termination/derivative disclosure, no-bound/no-gap fail-closed semantics, protected-case refusal, independent PYPOWER checks and immutable evidence CLI. Real case30 development reports SHA `8be27e3bb467a78d524930422bafa372729c3527782e06803c949f04449763dc` and `abff7f42274c5f9ad347a2d1af67bf8a585c478de7655438cd503aac13ae4ee5` retain negative/infeasible outcomes without overclaim | Execute frozen multistart/reference comparisons; add separately certified bounds where mathematically valid; populate independently human-reviewed checksum-bound ORPD profiles before using imported AC cases as ORPD formulations |
| PGLib/stress/OOD and cryptographically protected holdouts | Partial | Frozen case-role protocol and protected identities; three non-protected PGLib case14 groups now checksum-load from source and built wheel; protected import and ORPD conversion each fail closed without explicit test-only authorization | Complete reviewed ORPD profiles and execute the still-unopened protected final tests only after the full design freeze |
| Publish complete code, policy, formulation, image, seeds, raw failures, validation and claim scope | Harness partial | Artifact verifier, package exclusions, manifests, policy/config hashes, CI uploads | Final qualified policy, opened results, image attestations, release freeze |

## Current verification checkpoint

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

## Next legal implementation step

The exact decision was recorded on 2026-08-03:

> Approve TSH-CALO A–E, with F experimental and evidence-gated.

Changes A–E and disabled Change-F mechanics have passed local correctness gates. The failed v1
campaign remains immutable. V2 completed a five-member unqualified ensemble with exact real CUDA
training provenance, but its valid v3 development screen failed the frozen criteria. Case30 was
infeasible in both arms, and case57 did not pass Holm-controlled objective inference. The candidate
therefore remains inactive and cannot enter ordinary experiments or formal qualification. Do not
weaken the thresholds, run post-hoc formal trials, open protected cases, or create a receipt. The
repository now owns the missing A–E paired-evidence producer and candidate-bound CPU/CUDA
equivalence validator, but neither harness is evidence. The next legal G9 action is to freeze a new
non-tuning counted-v4 development-only plan and train one fresh A–E candidate identity. On that exact
candidate, execute physical device equivalence, the frozen A–E matrix and screening/formal eligibility
without adapting thresholds or opening protected cases. Change F remains excluded and disabled.
