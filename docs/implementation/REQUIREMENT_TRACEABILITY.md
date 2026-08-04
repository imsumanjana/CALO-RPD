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
| Calculate the 80% allowance from memory free/available at admission, not installed capacity | Physically verified on the observed RTX 4060 for the bounded probe | `compute/memory_budget.py`, `accelerated/vram_residency.py`, focused tests; clean `d6a950c` evidence recorded 7,441,743,872 free bytes before bounded pressure, 7,160,725,504 during pressure, and corresponding Safe-80 allowances of 5,953,395,097 and 5,728,580,403 bytes | Repeat in container/WSL2 target qualification |
| Prevent independent processes from overcommitting one physical GPU | Physically verified for the clean Windows-host probe | `compute/device_lease.py`, device-residency tests; clean `d6a950c` physical evidence refused a contender while owned and admitted it after release | Repeat in container/WSL2 multi-process qualification |
| Keep CUDA-resident, microbatch retry, staged-host, governed CPU fallback, and fail-closed states distinct | Physically exercised with controlled failure boundaries | `accelerated/device_resident_orpd.py`, `accelerated/vram_residency.py`, v6.9 VRAM tests; clean `d6a950c` evidence retained actual host-staged CUDA execution, controlled `5 → 2` OOM backoff, clean CPU full-request restart, and CUDA recovery | Controlled OOM is not natural hardware exhaustion; repeat transfer/recovery in container |
| Retain a bounded physical accelerator soak with protection and observed-only telemetry | Physically verified on the observed RTX 4060 host | Clean `67bd18e` evidence retained 3,600 GREEN samples over `3600.000156299968` seconds, no protection stop, independently verified 3,602-event provenance, 46–60 °C and 12.18–26.0 W observations, and `24.33879127740349` Wh GPU-board-energy integration; result SHA `49b805c3019dadc2c97cafcff230b84c29c15ffd18f2bf5e54d5364edfa30800` | Repeat in source-bound CUDA container/WSL2; CPU temperature, GPU power limit, and whole-system energy remain unavailable and must not be inferred |
| Remove utilization targets, device-memory percentages, task shares, and work-stealing from experiment execution | Locally verified | Current `ExperimentConfig`, current JSON schema, automatic CUDA-first scheduler, GUI configuration tests | None for experiment execution; policy-training routing is separately approval-gated |
| Remove executable Intel XPU support without falsifying historical records | Locally verified | XPU-free bootstrap/scheduler/runtime, view-only legacy backend migration, historical compatibility tests | Clean-machine compatibility run in CI |
| Preserve old populated databases through rollback-safe migration | Locally verified | SQLite schema v1, online pre-migration backup, integrity check, SHA-256 receipt, transactional DDL, future-version rejection; `tests/integration/test_database_workflow.py` | CI matrix execution |

## Containers and reproducibility

| Requirement | Status | Authoritative evidence | Remaining proof |
|---|---|---|---|
| Reproducible CPU and NVIDIA CUDA images with immutable dependencies | Physically built and source-bound on target workstation | Exact clean `1f02a94` produced CPU OCI `sha256:463be6ab…` and CUDA OCI `sha256:218ec1cc…` with maximum BuildKit provenance/SBOM, retained metadata, immutable build declarations and distinct runtime-loaded tags; build declaration remains operator metadata, not a signature | Repeat for the eventual final candidate/clean machine and retain CI artifacts |
| Non-root, read-only, capability-dropped runtime with persistent data volume | Physically verified for CPU/CUDA smoke and lease | Exact images passed UID 10001, read-only root, capability/no-new-privilege invocation, writable data, schema round-trip and shared `/data/device-leases`; independent CUDA containers proved exclusion then post-release acquisition | Complete noVNC GUI/restart/cancellation qualification and repeat in CI |
| CPU image must work without NVIDIA hardware; CUDA image must see exactly the selected GPU | Physically verified on target workstation | CPU image reported CUDA unavailable; CUDA image reported one NVIDIA GeForce RTX 4060 Laptop GPU, PyTorch `2.10.0+cu128`, CUDA `12.8`; container case30/case57 parity and resource recovery passed | Complete exact-image one-hour soak and repeat on trusted CI runner/final candidate |
| SBOM and vulnerability evidence | Physically retained for exact CPU/CUDA images | BuildKit embedded SBOM/provenance plus pinned local Trivy 0.70.0 at immutable scanner digest; CPU/CUDA CycloneDX, complete JSON and zero-fixable-HIGH/CRITICAL gates retained. Both report 700 total, 23 critical and 128 high by upstream/vendor severity, zero with an available fix under the gate | Retain database identity/timestamps and repeat for final digests/CI; do not imply remaining unfixable advisories are harmless |
| Lenovo LOQ WSL2/WSLg/GPU qualification | Pending external environment | `docs/CONTAINER_RUNBOOK.md` qualification procedure | Execute and retain target-laptop report |

## Scientist workflow and experiment protocol

| Requirement | Status | Authoritative evidence | Remaining proof |
|---|---|---|---|
| Normal GUI contains no venue promise or engineering/development language | Locally verified | Rendered-widget contract across five normal workspaces; 33-test offscreen GUI suite and screenshot | First packaged Linux rendering |
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
| Topology context, hierarchical actions, and uncertainty/bandit shield | B–D locally verified; real ensemble unqualified; frozen screen negative | B/C evidence plus `tsh_calo_shield.py`; deterministic mechanics tests; completed five-member IEEE 30/57 v2 ensemble; valid v3 development screen with 40 exact-FE paired records and immutable negative evidence | Candidate v2/design cannot proceed to formal qualification; a future candidate needs a new preregistered development design, accepted paired A–E evidence, target CPU/CUDA proof, and all frozen criteria satisfied; no benefit claim |
| Physics-informed repair | Counted runtime/training implementation locally verified; disabled by default | Runtime v1.1/training v4; final counted Newton Jacobian; analytic relaxed-control plus active voltage/angle/generator/thermal constraint derivatives; Q-limit-switched control masking; bounded lattice/continuous trust budget; exact FE/scenario accounting; no hidden solve/evaluator call; dynamic fail-closed masks; 18 direct tests and 136-pass TSH/counted-ORPD family | Run preregistered paired incremental-value/cost ablation on a fresh candidate and target CPU/CUDA path; no benefit claim yet |
| Evidence-driven population schedule | Experimental mechanics locally verified; disabled by default; promotion evidence absent | `tsh_calo_population_schedule.py`; separate `enabled` plus `experimental_mode` gates; preregistered design hash; feasibility/archive/diversity/budget/spacing conditions; deterministic feasibility-first contraction; no hidden FE; exact resume; nine focused tests | Must earn inclusion through paired-seed anytime/feasibility evidence without unacceptable cost, instability, overfitting or regression; otherwise keep disabled or remove |
| Versioned TSH-CALO candidate lifecycle | Locally verified through unqualified evaluation; negative screening evidence retained | `tsh_calo_policy_artifact.py`, `tsh_calo_qualification.py`, `tsh_calo_qualification_campaign.py`, algorithm-aware registry; exact ABI/provenance; protected-holdout rejection; non-serializable qualification-only evaluation capability; OS-released single-writer lease; failed-integrity resume rejection; valid v3 screen completed 40/40 records and correctly emitted no receipt | Candidate v2 remains unqualified/inactive and is barred from formal qualification under this design; the raced v1 screen remains barred from scientific use; a future candidate must satisfy a new frozen screen and direct A–E evidence before formal qualification |
| Independent TSH-CALO policy training | Versioned counted-E mechanics complete; historical five-member ensemble remains negative/unqualified | `tsh_calo_training*.py`, `scripts/train_tsh_calo.py`; v4 counted-physics/Safe-80/receipt ABI; exact source/scientific/execution/seed/curriculum hashes; protected-identity guards; canonical rewards and exact FE/scenario/update/reward accounting; authenticated resume; CUDA-first Safe-80; opt-in E/no-feasibility authority; F rejected; unqualified-only output; no lifecycle authority. Historical v2/v3-ABI ensemble completed 100,000 FE/scenario calls and remains immutable under SHA `3adf5017dc51f33d76214aeb505da598984b9da0e4263e1ee8fe59a667180ceb` | A fresh v4 candidate requires preregistered non-tuning design, physical CPU/CUDA equivalence and accepted ablations before qualification; old candidate remains inactive |
| Immutable TSH-CALO ensemble inference and fallback | Integrated into production and explicit qualification boundaries; physical parity pending | Ensemble assembly/load; activated-qualified production binding guard; qualification-only non-serializable capability; exact SHA/ABI/feature/member/calibration checks; CUDA-first current-free-VRAM 80% admission; governed CPU fallback; ensemble disagreement/shield; pre-evaluation block or explicit frozen-CALO relaunch; deterministic traces and exact resume; v3 screen used admitted CPU inference without fallback | Retain target CPU/CUDA parity and memory-pressure traces for a future qualified candidate; candidate v2 failed screening and cannot be activated |
| Reuse already-counted power-flow state for topology/physics context | Locally verified through topology and opt-in Change E paths | `ORPDProblem.evaluate_with_context`; default topology-only path retains no derivatives; explicit E path retains ephemeral final Jacobian/control/constraint context; base/weighted selection; optimizer/training construction; exact FE/scenario accounting; no serialized solver objects or hidden PF calls; case30 analytic sensitivity matched finite differences below `1.6e-8` max absolute error | Target CPU/CUDA execution and incremental-value/cost evidence remain required |
| Correct paired statistics, effect estimates, CIs, multiplicity control, power and anytime metrics | Locally verified at harness level | `statistics/`, campaign design, `SCIENTIFIC_VALIDATION_PROTOCOL.md`, statistical tests | Execute final frozen campaign |
| Modern strong stochastic baselines | Locally verified at implementation level | Source-traceable L-SHADE 1.0.1 and pinned pycma 4.4.4 CMA-ES with deterministic snapshots | External benchmark execution |
| Deterministic/mathematical reference solutions and broader licensed case corpus | Partial | Protocol plus pinned official PGLib-OPF v23.07 case14 typical/API/SAD validation assets; retained CC-BY-4.0 license/attribution; code-rooted manifest SHA-256 and asset SHA-256; non-executing restricted parser; exact source/physical provenance | Add and verify disclosed deterministic/nonlinear solver adapters; populate independently human-reviewed checksum-bound ORPD profiles before using imported AC cases as ORPD formulations |
| PGLib/stress/OOD and cryptographically protected holdouts | Partial | Frozen case-role protocol and protected identities; three non-protected PGLib case14 groups now checksum-load from source and built wheel; protected import and ORPD conversion each fail closed without explicit test-only authorization | Complete reviewed ORPD profiles and execute the still-unopened protected final tests only after the full design freeze |
| Publish complete code, policy, formulation, image, seeds, raw failures, validation and claim scope | Harness partial | Artifact verifier, package exclusions, manifests, policy/config hashes, CI uploads | Final qualified policy, opened results, image attestations, release freeze |

## Current verification checkpoint

- Active development suite after qualification-campaign implementation, excluding the deliberately
  stale v6.9 release-integrity file: **582 passed, 63 skipped**. The preceding complete-tree checkpoint
  after Change F reported **502 passed, 63 skipped, 2 failed**;
  both failures are the expected stale release freeze/root manifest and are not regenerated during
  G9 development.
- Repository Ruff lint and format: **pass across 406 files**.
- Generated experiment schema: **current**.
- Focused automatic scheduling/configuration/GUI set: **54 passed**.
- Focused database/history/learning/resume/continuation set: **29 passed**.
- Historical release-freeze tests remain separate and are regenerated only at final G11.
- Physical CUDA, Docker runtime, WSL2, thermals, energy, final protected tests, and external campaign
  artifacts are **not yet evidence**.
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
next legal G9 scientific action, if this candidate line is continued, is to preregister and train a
new development-only candidate/variant identity and earn direct A–E paired-ablation evidence before
another formal-eligibility decision. Independently, the next open gate evidence remains physical
CPU/CUDA parity and memory-pressure qualification. Change F remains excluded and disabled.
