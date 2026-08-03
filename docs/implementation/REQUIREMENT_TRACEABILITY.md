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
| Calculate the 80% allowance from memory free/available at admission, not installed capacity | Locally verified | `compute/memory_budget.py`, `accelerated/vram_residency.py`, `tests/unit/test_v690_vram_residency.py`, `tests/unit/test_v600_alpha_architecture.py` | RTX 4060 memory-pressure trace |
| Prevent independent processes from overcommitting one physical GPU | Implemented / external proof pending | `compute/device_lease.py`, device-residency tests and CUDA runner soak job | Multi-process physical-CUDA contention attestation |
| Keep CUDA-resident, microbatch retry, staged-host, governed CPU fallback, and fail-closed states distinct | Locally verified | `accelerated/device_resident_orpd.py`, `accelerated/vram_residency.py`, container runbook, v6.9 VRAM tests | Physical OOM/retry/transfer trace |
| Remove utilization targets, device-memory percentages, task shares, and work-stealing from experiment execution | Locally verified | Current `ExperimentConfig`, current JSON schema, automatic CUDA-first scheduler, GUI configuration tests | None for experiment execution; policy-training routing is separately approval-gated |
| Remove executable Intel XPU support without falsifying historical records | Locally verified | XPU-free bootstrap/scheduler/runtime, view-only legacy backend migration, historical compatibility tests | Clean-machine compatibility run in CI |
| Preserve old populated databases through rollback-safe migration | Locally verified | SQLite schema v1, online pre-migration backup, integrity check, SHA-256 receipt, transactional DDL, future-version rejection; `tests/integration/test_database_workflow.py` | CI matrix execution |

## Containers and reproducibility

| Requirement | Status | Authoritative evidence | Remaining proof |
|---|---|---|---|
| Reproducible CPU and NVIDIA CUDA images with immutable dependencies | Implemented / external proof pending | Digest-pinned `Dockerfile`, dated Debian snapshot, separate hash-complete CPU/CUDA locks, static container tests | Actual BuildKit image digests and provenance |
| Non-root, read-only, capability-dropped runtime with persistent data volume | Implemented / external proof pending | `compose.yaml`, `containers/`, `scripts/container_smoke.py`, container contract tests | Docker runtime smoke |
| CPU image must work without NVIDIA hardware; CUDA image must see exactly the selected GPU | Implemented / external proof pending | Compose profiles, compute-mode smoke contract, manually gated physical-CUDA CI lane | CPU container run and trusted RTX 4060 runner result |
| SBOM and vulnerability evidence | Implemented / external proof pending | CycloneDX/BuildKit/Trivy workflow lanes with immutable action SHAs | Uploaded SBOM and scanner report for final digests |
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
| Canonical runtime/training transition authority | Locally verified | `algorithms/calo/transition_kernel.py`; shared runtime/training authority invariant; native one-step parity; 22 frozen seeded optimizer snapshot/exact-budget cases; 45-test CALO/continuation focus; complete active tree 461 passed, 63 skipped | Retain parity in every later B–F stage and execute external CPU/CUDA qualification |
| Topology context, hierarchical actions, and uncertainty/bandit shield | B–D locally verified | B/C evidence plus `tsh_calo_shield.py`; six ensemble/OOD/bandit/resume/mixture/budget/lattice/fallback tests; cumulative A–D focus 28 passed; complete active tree 480 passed and 63 skipped | Fit calibration on development assets only, train fresh candidates, run paired ablations and protected tests; no benefit claim yet |
| Physics-informed repair | Locally verified; disabled by default | `tsh_calo_physics_repair.py`; ten mask/convergence/conditioning/trust/lattice/no-hidden-solver/exact-FE/failure tests; cumulative A–E focus 38 passed; complete tree 490 passed and 63 skipped | Supply retained counted-evaluation Jacobian context in the eventual runtime, then run separate incremental-value/cost ablation; no benefit claim yet |
| Evidence-driven population schedule | Experimental mechanics locally verified; disabled by default; promotion evidence absent | `tsh_calo_population_schedule.py`; separate `enabled` plus `experimental_mode` gates; preregistered design hash; feasibility/archive/diversity/budget/spacing conditions; deterministic feasibility-first contraction; no hidden FE; exact resume; nine focused tests | Must earn inclusion through paired-seed anytime/feasibility evidence without unacceptable cost, instability, overfitting or regression; otherwise keep disabled or remove |
| Versioned TSH-CALO candidate lifecycle | Locally verified at artifact/registry level | `tsh_calo_policy_artifact.py`, algorithm-aware `policy_schema.py` and `policy_registry.py`; seven dedicated tests; immutable SHA-256, exact ABI and independent-training provenance; protected-holdout rejection; candidate-only registration; qualified-only explicit TSH activation/binding | Implement independent TSH training and formal TSH qualification, create fresh candidates, and prove runtime consumption/fallback; no candidate or qualification evidence exists yet |
| Independent TSH-CALO policy training | PPO core locally verified; rollout integration and physical execution pending | `tsh_calo_training.py`; independent config/design hash; protected-holdout exclusion; masked hierarchical PPO; exact checksum-bound model/optimizer/RNG resume; design-drift rejection; unqualified-only export; static absence of experiment/registry/activation authority; seven dedicated tests | Connect counted development-only rollout state/reward production through the canonical transition authority, apply Safe-80 device admission, execute fresh training, and retain target CPU/CUDA provenance; no trained candidate exists yet |
| Immutable TSH-CALO ensemble inference and fallback | Core locally verified; optimizer integration and physical execution pending | Ensemble candidate assembly/load, activated-qualified binding guard, `tsh_calo_inference.py`; exact SHA/ABI/feature/member/calibration checks; CUDA-first current-free-VRAM 80% admission; governed CPU fallback; ensemble disagreement and safety shield; explicit block or frozen-CALO relabel; six dedicated tests | Connect to the counted TSH runtime transition without changing frozen CALO, capture target CPU/CUDA parity and memory-pressure traces, and qualify a real ensemble; no optimization result exists yet |
| Reuse already-counted power-flow state for topology/physics context | Locally verified through runtime candidate generation | `ORPDProblem.evaluate_with_context`; ephemeral scenario solve records; base/weighted converged selection; measured scenario descriptors; topology-state construction; optional physics proposal consumes supplied counted context only; no solver objects in result metadata or hidden PF calls | Integrate the versioned kernel into end-to-end optimizer/rollout orchestration and retain source-selection provenance; Jacobian retention for optional E remains separate |
| Correct paired statistics, effect estimates, CIs, multiplicity control, power and anytime metrics | Locally verified at harness level | `statistics/`, campaign design, `SCIENTIFIC_VALIDATION_PROTOCOL.md`, statistical tests | Execute final frozen campaign |
| Modern strong stochastic baselines | Locally verified at implementation level | Source-traceable L-SHADE 1.0.1 and pinned pycma 4.4.4 CMA-ES with deterministic snapshots | External benchmark execution |
| Deterministic/mathematical reference solutions and broader licensed case corpus | Partial | Protocol specifies reference and licensed-import requirements | Add/verify solver adapters and checksummed datasets without redistributing restricted assets |
| PGLib/stress/OOD and cryptographically protected holdouts | Partial | Frozen case-role protocol and protected case identities | Populate licensed/imported assets and execute unopened final tests |
| Publish complete code, policy, formulation, image, seeds, raw failures, validation and claim scope | Harness partial | Artifact verifier, package exclusions, manifests, policy/config hashes, CI uploads | Final qualified policy, opened results, image attestations, release freeze |

## Current verification checkpoint

- Active development suite after runtime context and candidate-transition integration, excluding the
  deliberately stale v6.9 release-integrity file: **532 passed, 63 skipped**. The preceding complete-tree checkpoint
  after Change F reported **502 passed, 63 skipped, 2 failed**;
  both failures are the expected stale release freeze/root manifest and are not regenerated during
  G9 development.
- Repository Ruff lint and format: **pass**.
- Generated experiment schema: **current**.
- Focused automatic scheduling/configuration/GUI set: **54 passed**.
- Focused database/history/learning/resume/continuation set: **29 passed**.
- Historical release-freeze tests remain separate and are regenerated only at final G11.
- Physical CUDA, Docker runtime, WSL2, thermals, energy, final protected tests, and external campaign
  artifacts are **not yet evidence**.

## Next legal implementation step

The exact decision was recorded on 2026-08-03:

> Approve TSH-CALO A–E, with F experimental and evidence-gated.

Changes A–E, the disabled Change-F mechanics, immutable ensemble lifecycle, independent PPO core,
shielded inference core and versioned runtime candidate-transition mechanics have passed their local
correctness gates. Change F has not earned promotion and remains off. The next legal step is end-to-end
development-only optimizer/rollout orchestration using the retained context and canonical completion
transition, followed by formal qualification. Fresh candidates must be
trained without protected-test leakage. Paired-seed component ablations must retain the disabled
baseline and may not weaken preregistered acceptance criteria.
