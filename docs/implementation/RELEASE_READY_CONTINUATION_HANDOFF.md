# CALO-RPD Studio release-ready continuation handoff

**Prepared:** 2026-08-03
**Repository:** `https://github.com/imsumanjana/CALO-RPD.git`
**Continuation branch:** `codex/calo-complete-modernization`
**Starting baseline recorded by this modernization:** `307402df5c7a44a6bb852770347b1b1ef995548d`
**Authoritative objective:** implement the remediation in
[`../COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md`](../COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md)
without overstating unexecuted hardware or scientific evidence.

**Continuation update (2026-08-03):** work resumed from `main` at
`7ec5b840193a4fe347c42e2d9ea1796fcac929e6`. The scientific lead supplied the exact decision
“Approve TSH-CALO A–E, with F experimental and evidence-gated.” The frozen current CALO remains
the baseline; TSH-CALO requires new algorithm and policy ABI versions, and no candidate may
auto-activate or inherit prior superiority evidence.

This document is the continuation contract for another Codex account. Begin by checking out the
branch above and reading this file, then inspect the current worktree and latest commit rather than
reconstructing the project from conversation history.

## 1. Immediate continuation command

```text
Continue the CALO-RPD modernization from
docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md on branch
codex/calo-complete-modernization. Preserve the scientific approval boundary and close the
remaining gates in order. Do not regenerate the release freeze until G9 and G10 are complete.
```

After cloning or fetching:

```bash
git fetch origin
git switch codex/calo-complete-modernization
git status --short
git log -1 --oneline --decorate
```

The commit containing this file is the authoritative continuation point. Do not assume the old
starting baseline is the current implementation state.

## 2. Mandatory scientific approval boundary

The user approved strengthening CALO in principle, but the exact architecture scope has not yet
been confirmed. No code change may alter CALO search, training, policy, reward, promotion, state,
action, operator, archive/memory, selection, epsilon, precision/recovery, or transition semantics
until the user explicitly confirms:

> **Approve TSH-CALO A–E, with F experimental and evidence-gated.**

The complete decision record is
[`CALO_ARCHITECTURE_CHANGE_PROPOSAL.md`](CALO_ARCHITECTURE_CHANGE_PROPOSAL.md). It proposes:

- **A:** one canonical runtime/training transition kernel;
- **B:** topology-aware bus/branch/control graph context alongside the existing aggregate state;
- **C:** hierarchical group-conditioned actions for generator voltages, taps, and shunts;
- **D:** uncertainty-shielded PPO/rule/contextual-bandit mixture;
- **E:** optional, fully counted Jacobian-informed feasibility-repair proposals; and
- **F:** an experimental population schedule that is disabled by default and must earn inclusion
  through a separate ablation.

If the exact sentence is supplied, record it in
[`IMPLEMENTATION_GATES.md`](IMPLEMENTATION_GATES.md), mark G8 complete, and implement only the
approved scope. A different or partial approval must be translated into an explicit A–F decision
before scientific implementation.

## 3. Completed foundations that must be preserved

### Memory and execution

- The shared admission contract uses 80% of VRAM/RAM free or available at the admission boundary,
  not 80% of installed capacity.
- CUDA is preferred; active compatible tensors remain device-resident within the frozen allowance.
- CUDA OOM handling reduces the active microbatch and retries before any governed CPU decision.
- Staged host memory, clean-restart CPU fallback, and fail-closed outcomes are distinct and recorded.
- One selected CUDA device is leased; experiment scheduling no longer uses utilization targets,
  memory percentages, task shares, GPU-job counts, or work stealing.
- CPU and GPU still perform computation. VRAM/RAM are storage resources, not compute engines.

Relevant sources:

- `calo_rpd_studio/compute/memory_budget.py`
- `calo_rpd_studio/compute/device_lease.py`
- `calo_rpd_studio/accelerated/vram_residency.py`
- `calo_rpd_studio/accelerated/device_resident_orpd.py`
- `calo_rpd_studio/compute/resource_scheduler.py`

### XPU removal and compatibility

- Executable Intel XPU bootstrap, scheduler, worker, sidecar, and repair paths are removed.
- Current execution modes are only `cuda_preferred` and `cpu_only`.
- Historical XPU configurations remain readable as view-only records and cannot execute silently.
- Historical utilization/share fields are accepted only at the strict configuration migration
  boundary and are discarded.

### Persistence

- SQLite application schema is version 1 through `PRAGMA user_version`.
- A populated version-0 database receives an online, integrity-checked backup before DDL.
- Backup path and SHA-256 are written to `schema_migrations`.
- Migration is transactional, representative legacy rows are preserved, reopen is idempotent, and
  future schema versions fail closed without mutation.

### Containers and CI

- CPU and CUDA 12.8 images use separate exact SHA-256 requirement locks.
- The Python base is digest-pinned and Debian APT uses a dated snapshot.
- Compose is local-only, non-root UID/GID 10001, read-only, capability-dropped, and selects one
  explicit NVIDIA device.
- Release-critical CI lanes are hash-locked and all external actions use 40-character commit SHAs.
- Cross-version compatibility jobs intentionally resolve bounded platform-specific graphs, run
  `pip check`, and upload their exact `pip freeze --all` environments.
- CPU image, CUDA build-only, SBOM, Trivy, GUI rendering, staged distribution, and opt-in trusted
  physical-CUDA lanes exist.

### Scientist workflow

- Normal experiment UI exposes only “Accelerated when available” and “CPU only”.
- Normal GUI contracts reject venue promises and engineering/development language.
- Evidence-strength protocols are power-aware, present required outputs and validity rules, show a
  before/after diff, and apply atomically across the shared experiment configuration.
- Protocol application does not mutate independent policy-training configuration.
- Every new power-system experiment requires and snapshots a compatible, qualified, active,
  checksum-valid governing policy.
- Policy training never auto-activates a policy; No-AI CALO remains restricted to expert
  qualification/ablation use.

### Scientific harness already available

- Current campaign registry contains 22 methods and 21 CALO-versus-comparator tests.
- Default confirmatory planning initiates 98 paired runs for standardized effect 0.50, 95% power,
  Holm family control, and 10% failure allowance.
- L-SHADE 1.0.1 and pycma 4.4.4 CMA-ES are source-traceable strong comparators with deterministic
  snapshots and common evaluation contracts.
- Case roles, protected holdouts, design hashes, paired statistics, effect estimates, confidence
  intervals, multiplicity control, raw failures, and independent validation are represented in the
  harness.

## 4. Current verified checkpoint

The most recent complete active-tree run, excluding only the intentionally stale historical v6.9
release-integrity file, produced:

```text
453 passed, 63 skipped
```

Other verified evidence:

- complete offscreen GUI/scientist suite: 33 passed and a validated 1440×900 dashboard PNG;
- latest measured CI-style coverage: 66%, threshold 60%;
- repository Ruff lint and format: pass across 359 Python files;
- generated experiment schema: current;
- bounded mypy safety target: pass on the recorded nine-module target;
- automatic scheduler/configuration/GUI focus: 54 passed;
- database/history/learning/resume/continuation focus: 29 passed;
- latest CI/container contract focus after environment-evidence changes: 5 passed.

Reproduce the active suite on Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q --ignore=tests\unit\test_v690_release_integrity.py
.\.venv\Scripts\ruff.exe check calo_bootstrap calo_rpd_studio tests
.\.venv\Scripts\ruff.exe format --check calo_bootstrap calo_rpd_studio tests
.\.venv\Scripts\python.exe -m calo_rpd_studio.scripts.generate_experiment_schema --check
```

Reproduce the release-lock checks in a Linux/Python 3.11 environment:

```bash
python -m pip install --require-hashes --requirement requirements-lock-ci-py311-linux.txt
python -m calo_rpd_studio.scripts.verify_requirements_lock requirements-lock-cpu-py311-linux.txt \
  --expect-index https://download.pytorch.org/whl/cpu \
  --expect-pin torch==2.10.0+cpu --expect-pin cma==4.4.4
python -m calo_rpd_studio.scripts.verify_requirements_lock requirements-lock-cuda128-py311-linux.txt \
  --expect-index https://download.pytorch.org/whl/cu128 \
  --expect-pin torch==2.10.0+cu128 --expect-pin cma==4.4.4
```

## 5. Remaining gates in required order

### G8 — exact architecture approval recorded

**Status:** complete on 2026-08-03. Approved A–E for careful production-candidate implementation;
approved F only as an independent experiment, disabled by default and promotable only through its
preregistered ablation/falsification gate.

Exit criteria:

1. User explicitly approves/rejects each A–F component.
2. Current CALO is confirmed as a frozen baseline.
3. The new algorithm and policy ABI receive new versions.
4. The new policy cannot auto-activate and cannot inherit old superiority evidence.

Do not treat “make it stronger” or general approval in principle as the exact A–F decision.

### G9 — implement the approved TSH-CALO architecture

**Continuation progress:** Change A is locally complete. Runtime CALO and independent PPO rollouts
share `calo_rpd_studio/algorithms/calo/transition_kernel.py`. Direct evidence is 461 passed and 63
skipped on the complete active tree, 45 focused CALO/parity/continuation tests, 22 frozen seeded
optimizer snapshot/exact-budget cases, and eight dedicated kernel invariants. This is parity/refactor
evidence only; B–F, a fresh policy, qualification, and ablation evidence remain absent.

Change B is also locally complete: the new TSH-CALO candidate ABI and strict aggregate-plus-topology
state/encoder have five focused graph-versioning, validation, permutation and topology-change tests;
the complete tree passes 466 tests with 63 skips. The builder requires an already-computed converged
power-flow result and cannot create a hidden solver call. This does not establish performance benefit.
C–F, fresh training, qualification and every ablation remain pending.

Change C is locally complete behind the new ABI: global regime, physical control-group and learner-
context action heads, strict masks, bounded controls and independently replayable learner sampling
have eight focused tests. The complete tree passes 474 tests with 63 skips. A bounded policy-forward
CPU/CUDA check passed on the available device, but this is not target-hardware qualification. D–F,
training, qualification, activation and scientific evidence remain pending.

Change D is locally complete: ensemble disagreement, frozen-reference OOD attenuation, exact
sliding-window bandit resume, the declared four-way mixture, FE/lattice/operator shielding,
deterministic intervention traces and explicit block-or-relabelled-baseline fallback have six
focused tests. The cumulative A–D focus is 28 passed and the full tree is 480 passed with 63 skips.
No uncertainty calibration, policy quality or optimization benefit has been established.

Change E is locally complete and disabled by default. The new seventh operator consumes only a
supplied counted-evaluation linearization, masks unsafe contexts, is trust-bounded/lattice-valid,
performs no hidden power-flow call, declares no feasibility and requires an explicit trusted FE.
Ten focused tests pass; cumulative A–E focus is 38 passed and the full tree is 490 passed with 63
skips. No physics-repair benefit or acceptable-cost evidence exists yet.

Change F mechanics are locally complete but remain experimental and disabled by default. Enabling
requires both the independent feature flag and explicit experimental mode. The design hash binds
the preregistered feasibility, archive-coverage, diversity, remaining-budget, spacing and bounded-
contraction conditions. Contraction is deterministic, feasibility-first, adds no FE and resumes
exactly. Nine focused tests pass; cumulative A–F mechanics are 47 passed. Excluding the deliberately
stale v6.9 release-integrity file, the active tree is 499 passed with 63 skips. The complete tree is
502 passed, 63 skipped and the same two expected stale freeze/manifest failures. This is not evidence
of performance benefit or grounds for promotion.

The immutable TSH-CALO candidate boundary is also locally complete. Candidate export records the
exact algorithm/policy/state/action/training ABIs, feature flags, independent-training design and
seed hashes, source commit and development-case identities, and always labels the portable artifact
unqualified. Protected holdouts are rejected before writing. Registration, qualification,
activation and experiment binding remain distinct actions; TSH-CALO cannot use the older research-
only unqualified activation/binding path. Integrity-checked loading and binding preserve SHA-256 and
scientific provenance under the separate `TSH-CALO` key. Seven dedicated lifecycle tests, a 33-test
policy compatibility/independence focus and the 506-pass active tree are green. No real candidate has
been trained, qualified or activated.

The independent TSH-CALO PPO core is locally complete. It uses a separate training configuration and
scientific design hash, rejects protected holdouts, accepts already-built topology rollout states,
applies the declared masks to global/group/context actions, resumes model/optimizer/local RNG state
only with the expected SHA-256 and unchanged scientific design, and can export only an unqualified
candidate after at least one update. It has no experiment-manager, experiment-runner, registry,
activation or binding authority. Seven dedicated tests, the 22-test trainer/lifecycle/action focus
and the 513-pass active tree pass. This does not represent a real training run; development rollout
production, Safe-80 target-device admission and physical CPU/CUDA evidence remain open.

The immutable ensemble and shielded inference core is locally complete. Independent single-member
candidates are assembled without losing member SHA/provenance and cannot activate individually.
Runtime loading requires a qualified, explicitly activated binding and verifies the ensemble SHA,
exact ABIs, feature flags, member identities and frozen OOD-calibration hash. Admission selects CUDA
first when a conservative working-set estimate fits within 80% of currently free VRAM; CPU is used
only when CUDA is unavailable, that admitted estimate cannot fit, or a real CUDA allocation fails,
and it is separately checked against 80% of currently available RAM. The NVIDIA GPU or CPU performs
the computation; memory is only admitted storage. Ensemble disagreement, OOD attenuation, the
declared shield and explicit block/relabelled-baseline fallback are traced. Six dedicated tests, a
35-test inference/lifecycle/shield/trainer/action focus and the 520-pass active tree pass. The core is
not yet connected to actual TSH optimizer transitions and is not physical CUDA evidence.

The ORPD evaluator now exposes already-counted scenario solver context through a separate additive
API. It returns the same `Evaluation` values as ordinary evaluation, retains each exact power-flow
object only ephemerally, never inserts it into JSON result metadata, prefers the counted base solve
(otherwise the highest-weight converged solve), and fails closed when none converged. It performs no
second power-flow call for state construction. Three dedicated tests, a 37-test CALO/topology/repair
focus and the 523-pass active tree pass. The TSH optimizer and training rollout still need to consume
this boundary.

The versioned runtime context and candidate-transition mechanics now consume that boundary without
rerunning power flow. Scenario descriptors are measured from the already-counted scenario cases;
unknown roles are explicitly neutral and do not make an OOD claim. Per-learner shielded group and
operator actions plus bounded group parameters drive candidate generation. Optional physics repair
requires a supplied counted context and fails closed if it becomes unavailable; every proposal still
requires trusted evaluation. Forced recovery and precision remain explicit, with TSH precision
successes isolated in memory channel 7 because operator 6 belongs to physics repair. Nine dedicated
runtime invariants and a 68-test runtime/frozen-snapshot/parity focus pass. The active tree excluding
only the deliberately stale v6.9 release-integrity file passes 532 tests with 63 skips; repository
Ruff lint/format and the generated schema pass. End-to-end optimizer/rollout orchestration, real
candidate training, target CUDA evidence, qualification and every benefit claim remain pending.

Required order after approval:

1. Extract the canonical transition kernel with **zero behavior change**.
2. Prove seeded baseline parity before continuing.
3. Introduce new versioned graph-state and hierarchical-action schemas.
4. Add topology encoder and group-conditioned action heads behind the new algorithm ID.
5. Add ensemble uncertainty, out-of-distribution attenuation, bandit residual, action shield, and
   complete deterministic intervention traces.
6. Add Change E behind an independent feature/ablation flag.
7. Add Change F behind an independent experimental flag, disabled by default.
8. Train fresh candidates without protected-test leakage.
9. Qualify candidates; never auto-activate.

Mandatory implementation tests:

- canonical-refactor output must match the frozen baseline exactly for seeded CPU fixtures;
- no FE overshoot, hidden solver calls, or uncounted repair evaluation;
- action masks and all shield/bandit/ensemble mixtures replay deterministically;
- exact checkpoint/resume across every new state component;
- graph permutation and topology-change invariants;
- CPU/CUDA numerical parity and mixed-variable lattice validity;
- protected identities cannot enter training, tuning, reward design, or checkpoint selection;
- old policy artifacts remain immutable and are rejected by the new ABI rather than reinterpreted.

Mandatory ablation set:

1. frozen current CALO;
2. canonical-refactor CALO;
3. graph context only;
4. hierarchical actions only;
5. graph plus hierarchy;
6. uncertainty shield;
7. contextual bandit residual;
8. physics repair;
9. population schedule; and
10. full approved TSH-CALO.

Remove any component whose incremental evidence does not support its claimed benefit.

### G10 — complete scientific evidence

The harness is not the evidence. Execute and retain a frozen campaign after G9:

1. Add a safe, checksummed import path and reviewed ORPD control profiles for independently sourced
   PGLib-OPF typical/API/SAD cases. Do not silently equate AC-OPF case loading with an ORPD
   formulation.
2. Add disclosed deterministic/nonlinear mathematical reference solvers where mixed-variable
   semantics permit; separate continuous relaxation bounds from feasible ORPD solutions.
3. Freeze training, validation, and protected test identities cryptographically.
4. Freeze source, policy, hyperparameters, statistical plan, containers, and analysis scripts before
   opening protected tests.
5. Execute paired equal-FE trials at multiple budgets, recording anytime target attainment.
6. Report feasibility probability and violation distributions before objective comparisons.
7. Report paired effects, uncertainty intervals, Holm-controlled tests, failures, wall time, peak
   RAM/VRAM, transfers, thermals, energy, and solver calls separately.
8. Do not pool fallback modes or heterogeneous timing strata.
9. Independently validate final physical solutions and retain raw arrays and failure records.
10. Limit claims to the tested ORPD distribution, cases, objectives, scenarios, and budgets.

The protected test assets must remain unopened until the design is frozen. Any semantic source,
policy, hyperparameter, or analysis change after opening creates a new candidate and requires a new
protected test opening.

### G4/G6 physical and packaged qualification — execute, do not simulate

This workstation previously exposed a Docker client but no running daemon and no Compose/Buildx
plugin, so container attestations have not been generated. On the Lenovo LOQ target:

1. Install/enable Docker Desktop WSL2, WSLg, NVIDIA Windows driver support, Compose and Buildx.
2. Run the CPU profile without NVIDIA access.
3. Run the CUDA profile with exactly one selected RTX 4060 device.
4. Verify non-root UID/GID, read-only root, `/data` persistence, health check, schema version, config
   round-trip, GUI reachability, and restart behavior.
5. Exercise VRAM pressure, microbatch backoff, staged-host transfer, clean CPU restart, cancel,
   checkpoint, and recovery paths.
6. Execute bounded FP64 parity, thermal, power, energy, and multi-process device-lease tests.
7. Record driver, WSL, Docker, BuildKit, image digest, SBOM, scanner database timestamp and complete
   reports.

Suggested commands after Docker is available:

```powershell
docker version
docker compose version
docker buildx version
docker compose --profile cpu build cpu
docker compose --profile cpu up --abort-on-container-exit cpu
$env:CALO_GPU_DEVICE='0'
docker compose --profile cuda build cuda
docker compose --profile cuda up --abort-on-container-exit cuda
```

Also dispatch `.github/workflows/ci.yml` with `run_physical_cuda=true` only on a trusted self-hosted
runner labelled `calo-cuda`.

### G11 — produce the actual release candidate

Only after G9, G10, CI, Docker and target-hardware gates are green:

1. Start from a clean clone of the exact candidate commit.
2. Run the complete requirement audit in
   [`REQUIREMENT_TRACEABILITY.md`](REQUIREMENT_TRACEABILITY.md); every row must have direct evidence.
3. Build one wheel and one sdist into a new, previously absent staging directory.
4. Run `verify_distribution_stage`; generated policies/checkpoints must not enter either artifact.
5. Generate staged artifact and image-filesystem SHA-256 manifests.
6. Build immutable CPU/CUDA images and retain their digests, provenance, SBOMs and vulnerability
   reports.
7. Install the wheel and run CPU and CUDA smoke/parity tests on clean machines.
8. Update release metadata, validation, scientific-equivalence and hardware-qualification records
   from real evidence only.
9. Regenerate the release freeze and root manifest from staged artifacts, never from a dirty
   checkout.
10. Tag/release only after all generated records verify against the exact commit and artifacts.

The current `calo_v690_freeze.json` and root `MANIFEST.sha256` are historical snapshots and are
expected to be stale on this development branch. Do not “fix” them early.

## 6. Known incomplete evidence — do not overclaim

- No physical RTX 4060 CUDA pressure, thermal, energy, or multi-process lease attestation yet.
- No successful Docker CPU/CUDA runtime build on this workstation yet.
- No WSL2/WSLg target-laptop report yet.
- No GitHub Actions run artifacts from the new workflow yet.
- No approved TSH-CALO implementation or fresh policy yet.
- No approved-architecture ablation campaign yet.
- No imported/reviewed PGLib typical/API/SAD ORPD corpus yet.
- No deterministic mathematical-solver comparison package yet.
- No opened protected final-test campaign yet.
- No final image digests, SBOMs, vulnerability reports, release freeze, or clean-machine
  reproduction yet.

## 7. Repository hygiene and user-data boundaries

Do not commit or package:

- `*.sqlite` result databases;
- `.coverage*`, `htmlcov/`, `.pytest_cache/`, `.mypy_cache/`, or `.ruff_cache/`;
- `.venv/`, local environments, caches, or downloaded wheels;
- `*.pt`, `*.pt.sha256`, policy lineage directories, generated training metadata, or recovery files;
- `results_data/`, `artifacts/`, screenshots generated outside the declared evidence stage, or
  publication exports;
- credentials, tokens, machine-local signing/HMAC keys, or user experiment data.

Historical frozen manifests and reports are source records; generated runtime policy/result data are
not source records. Preserve pre-existing user files unless deletion is explicitly authorized.

## 8. Authoritative documents

- Full audit and remediation objective:
  [`../COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md`](../COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md)
- Implementation and approval gates: [`IMPLEMENTATION_GATES.md`](IMPLEMENTATION_GATES.md)
- Requirement-to-evidence ledger: [`REQUIREMENT_TRACEABILITY.md`](REQUIREMENT_TRACEABILITY.md)
- Architecture proposal: [`CALO_ARCHITECTURE_CHANGE_PROPOSAL.md`](CALO_ARCHITECTURE_CHANGE_PROPOSAL.md)
- Frozen scientific protocol: [`SCIENTIFIC_VALIDATION_PROTOCOL.md`](SCIENTIFIC_VALIDATION_PROTOCOL.md)
- Container operation and qualification: [`../CONTAINER_RUNBOOK.md`](../CONTAINER_RUNBOOK.md)

## 9. Release-ready definition

The repository is release-ready only when all of the following are simultaneously true:

- the exact scientific architecture was approved and implemented under a new version;
- baseline parity and every new invariant pass;
- every claimed component earns its place in paired ablations;
- the protected, power-aware scientific campaign is complete and independently validated;
- CPU and selected-GPU containers pass on clean target systems;
- physical CUDA, memory-pressure, fallback, thermal, energy and recovery evidence is retained;
- all CI lanes pass for the exact candidate commit;
- distribution and container artifacts contain no generated policy/user data;
- every traceability row is closed with direct evidence;
- manifests, freeze, metadata, SBOMs and image digests all bind to the same immutable candidate; and
- public claims are restricted to the exact tested ORPD problem distribution and budgets.

Until then, describe the branch as a modernization candidate with implemented harnesses and pending
scientific/hardware qualification—not as a final release.
