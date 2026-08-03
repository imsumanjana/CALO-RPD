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
| Explain final runtime/training architecture before changing CALO semantics | Complete proposal; decision pending | `CALO_ARCHITECTURE_CHANGE_PROPOSAL.md` and the final A–F architecture confirmation | Exact approval of A–E and evidence-gated F |
| Canonical runtime/training transition authority | Approval-gated | Proposed Change A | Implement only after approval; seeded zero-difference gate |
| Topology context, hierarchical actions, and uncertainty/bandit shield | Approval-gated | Proposed Changes B–D | Versioned implementation, training, ablations, protected tests |
| Physics-informed repair | Approval-gated | Proposed Change E | Separate flag and incremental-value ablation |
| Evidence-driven population schedule | Approval-gated and experimental | Proposed Change F | Must earn inclusion through anytime/feasibility evidence |
| Correct paired statistics, effect estimates, CIs, multiplicity control, power and anytime metrics | Locally verified at harness level | `statistics/`, campaign design, `SCIENTIFIC_VALIDATION_PROTOCOL.md`, statistical tests | Execute final frozen campaign |
| Modern strong stochastic baselines | Locally verified at implementation level | Source-traceable L-SHADE 1.0.1 and pinned pycma 4.4.4 CMA-ES with deterministic snapshots | External benchmark execution |
| Deterministic/mathematical reference solutions and broader licensed case corpus | Partial | Protocol specifies reference and licensed-import requirements | Add/verify solver adapters and checksummed datasets without redistributing restricted assets |
| PGLib/stress/OOD and cryptographically protected holdouts | Partial | Frozen case-role protocol and protected case identities | Populate licensed/imported assets and execute unopened final tests |
| Publish complete code, policy, formulation, image, seeds, raw failures, validation and claim scope | Harness partial | Artifact verifier, package exclusions, manifests, policy/config hashes, CI uploads | Final qualified policy, opened results, image attestations, release freeze |

## Current verification checkpoint

- Active development suite: **453 passed, 63 skipped**.
- Repository Ruff lint and format: **pass**.
- Generated experiment schema: **current**.
- Focused automatic scheduling/configuration/GUI set: **54 passed**.
- Focused database/history/learning/resume/continuation set: **29 passed**.
- Historical release-freeze tests remain separate and are regenerated only at final G11.
- Physical CUDA, Docker runtime, WSL2, thermals, energy, final protected tests, and external campaign
  artifacts are **not yet evidence**.

## Next legal implementation step

Scientific implementation begins only after the exact decision:

> Approve TSH-CALO A–E, with F experimental and evidence-gated.

Until then, only architecture-neutral engineering, documentation correction, diagnostic tests, and
external qualification preparation may proceed.
