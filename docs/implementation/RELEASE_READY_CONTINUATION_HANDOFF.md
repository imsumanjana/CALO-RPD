# CALO-RPD Studio release-ready continuation handoff

**Prepared:** 2026-08-04
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

### Required workflow for every remaining phase

Before announcing or starting development for a numbered phase, create a phase-specific goal using
the goal service. Then perform coding only. Source tests and validation automation may be written,
but the agent must not execute tests, validators, compilation/schema/lint/type checks, builds,
GUI/Docker smoke checks, benchmarks, campaigns, policy workflows, qualification, or protected cases
when the user can run them manually. Only a later explicit instruction naming a particular command
overrides this execution boundary.

End each phase coding pass by preparing a detailed PowerShell validator under the Git-ignored
`validation/` directory. The validator must produce a newly timestamped log tree with command
results and source/validator hashes. Neither the validator nor its logs may be added to Git or any
release artifact. Give the user its exact command, wait for the complete returned log directory,
then review the evidence read-only and make focused corrections. Keep repository scans and output
minimal to reduce token consumption. A phase is not validated merely because its code or validator
exists.

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
582 passed, 63 skipped
```

Other verified evidence:

- complete offscreen GUI/scientist suite: 33 passed and a validated 1440×900 dashboard PNG;
- latest measured CI-style coverage: 66%, threshold 60%;
- repository Ruff lint and format: pass across 406 Python files;
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
35-test inference/lifecycle/shield/trainer/action focus and the 520-pass active tree pass. Later
checkpoints connect this core to the TSH optimizer; physical CUDA evidence remains absent.

The ORPD evaluator now exposes already-counted scenario solver context through a separate additive
API. It returns the same `Evaluation` values as ordinary evaluation, retains each exact power-flow
object only ephemerally, never inserts it into JSON result metadata, prefers the counted base solve
(otherwise the highest-weight converged solve), and fails closed when none converged. It performs no
second power-flow call for state construction. Three dedicated tests, a 37-test CALO/topology/repair
focus and the 523-pass active tree pass. Later optimizer and training-environment checkpoints now
consume this boundary; this earlier count remains its introduction checkpoint.

The versioned runtime context and candidate-transition mechanics now consume that boundary without
rerunning power flow. Scenario descriptors are measured from the already-counted scenario cases;
unknown roles are explicitly neutral and do not make an OOD claim. Per-learner shielded group and
operator actions plus bounded group parameters drive candidate generation. Optional physics repair
requires a supplied counted context and fails closed if it becomes unavailable; every proposal still
requires trusted evaluation. Forced recovery and precision remain explicit, with TSH precision
successes isolated in memory channel 7 because operator 6 belongs to physics repair. Nine dedicated
runtime invariants and a 68-test runtime/frozen-snapshot/parity focus pass. The active tree excluding
only the deliberately stale v6.9 release-integrity file passes 532 tests with 63 skips; repository
Ruff lint/format and the generated schema pass. Later checkpoints add the end-to-end optimizer and
counted training environment; real candidate training, target CUDA evidence, qualification and every
benefit claim remain pending.

The policy-gated `TSHCALOOptimizer` now joins these cores without entering the frozen default
comparison campaign. It loads and validates the immutable ensemble before any power-system
evaluation, uses counted contexts for each selected learner, applies independently shielded actions,
calls the canonical completion transition, keeps candidate FE and per-scenario PF counts explicit,
and checkpoints population, contexts, archives, memories, bandit, optimizer RNG and policy RNG for
exact stochastic resume. Policy/device/OOD/mixture/action/intervention/fallback provenance is retained.
A rejected policy blocks at preflight or requires an explicit relaunch under the `CALO-v5.9`
identity; it is never silently executed or reported as TSH-CALO. Experimental F is rejected by this
fixed-population production path before evaluation. Runtime v1.1 now requests final counted Newton
linearization only for an immutable E-enabled policy, exposes the operator only when every learner has
a finite, conditioned, nonzero control/constraint context, and records proposal cost without adding an
uncounted solve or feasibility authority. Default and topology-only paths retain no derivatives.
Historical v1.0 candidate evidence is not reinterpreted. Fresh v1.1 training, formal qualification,
physical CUDA/CPU parity, component ablations and protected tests remain pending.

TSH activation and binding now require an immutable qualification receipt rather than trusting a
generic database status. The receipt binds the exact policy SHA, qualification run/protocol/source,
seed manifest, referenced evidence artifact, development-case identities and frozen OOD calibration.
Protected holdout identities are rejected before receipt creation. Activation freezes the receipt in
policy metadata; experiment binding carries it; inference verifies the policy, receipt and calibration
hashes again. Missing, cross-policy or mutated receipts fail closed. Five added receipt-path cases and
a 27-test qualification/lifecycle/inference/optimizer focus pass; the active tree is 545 passed with
63 skips. This establishes integrity and separation mechanics only. No evidence artifact was produced
or evaluated, no policy was scientifically qualified, and no acceptance claim is made.

Independent PPO rollout collection now accepts only rewards from the canonical versioned transition
result. Discount and GAE factors are part of the training design hash; terminal-aware advantages and
returns are deterministic; an unevaluated pending action cannot be checkpointed; and restored
collector state must match the exact scientific design. Nine dedicated training cases, a 22-test
training/transition focus and the 547-pass active tree pass. The collector still consumes already-
executed transitions.

The counted development-only ORPD environment adapter is now locally complete. It binds the declared
development-case identity, loaded case checksum, formulation fingerprint, training design and fixed
environment design; rejects a loaded protected case by content identity; evaluates only full batches
through `ORPDProblem.evaluate_with_context`; reports candidate FE and per-scenario solver calls; builds
topology state from the selected already-counted context; and executes raw single-member hierarchical
actions through `generate_tsh_offspring` and the canonical `complete_tsh_transition`. The v5 CUDA
path submits a whole counted population to the device-resident tensor evaluator, retains final
voltage/diagnostic/type/generation/branch state in VRAM, and reconstructs contexts only after one
packed outer-boundary materialization with no hidden CPU power-flow rerun. It has no
experiment, registry, qualification, activation, GUI or production-inference authority. Training ABI
v4 admits E only from the same explicit counted linearization boundary, dynamically masks it otherwise,
and keeps every proposal's trusted evaluation inside the ordinary FE batch; F is rejected before any
solve. Pending observations, derivative contexts, counters and all environment state/RNG/components
resume under unchanged design/problem hashes. This is mechanics evidence only: fresh ensemble-member
training, qualification, ablations and any benefit claim remain pending.

Independent trainer Safe-80 admission is now locally complete under current training-environment ABI
`tsh-calo-training-v5-batched-device-context-safe80`. Historical v3/v4 artifacts remain immutable and are
not silently migrated. Every new training design declares and hashes maximum rollout,
population, node, directed-edge, control and scenario counts. A deterministic versioned estimator
accounts for parameters/buffers, gradients and Adam moments, retained rollout state, autograd
activations, fragmentation safety and an explicit runtime floor. The trainer checks every state and
batch against that envelope, admits CUDA first only when the estimate fits within 80% of VRAM free
at admission, otherwise uses CPU only when CUDA is unavailable or explicit fallback is permitted and
the estimate fits within 80% of currently available RAM. CUDA training holds cross-process and local
single-owner leases and applies the admitted allocator ceiling; lease contention blocks instead of
spilling to CPU. Exact resume requires the same admitted computation device. Candidate provenance
records the estimate, live admission, selected compute device, fallback reason and the truthful fact
that the NVIDIA GPU or CPU computes while memory stores data. Episode receipt v2 additionally binds
the counted ORPD device, batch-context use, target host boundary, inner transfer count and hidden
context-solve count. Mutated Safe-80 or computation records
are rejected, and earlier v1/v2 training-environment artifacts are non-native rather than silently
migrated. Seven dedicated resource tests and a 49-test resource/training/environment/lifecycle/
inference/optimizer focus pass. The active tree excluding only the deliberately stale v6.9 release-
integrity file is 563 passed with 63 skips; Ruff lint/format passes across 397 Python files and the
generated schema is current. This is local CPU and mocked-CUDA mechanics evidence; no physical CUDA
training, fresh member, qualification, ablation or benefit result exists.

Fresh-member session orchestration is now locally complete as a development-only boundary. A session
joins exactly one trainer and one counted environment under a separately hashed session identity,
commits only canonical transition rewards, updates PPO at the admitted rollout boundary or terminal,
and issues an integrity-checked receipt only after terminal accounting completes. The receipt binds
the training/session/environment designs, run and unique session IDs, loaded development-case
identity/checksum, formulation fingerprint, seed, exact candidate and scenario-call totals,
transition and update counts, and the canonical reward-sequence hash. Candidate export now requires
at least one such receipt but remains an explicit, separate action and always produces an unqualified
artifact. Trainer, environment, collector, reward history and update metrics share one trusted,
authenticated exact-resume checkpoint; solver failure poisons the session and forbids checkpointing,
receipt issuance and continuation. The orchestration module has no experiment, registry,
qualification, activation or inference authority. Five dedicated session/receipt cases and a
39-test adjacent focus pass. The active tree excluding only the deliberately stale v6.9
release-integrity file is 568 passed with 63 skips; Ruff lint/format passes across 400 Python files
and the generated schema is current. The exercised toy episode is test-fixture mechanics only: no
real member, training campaign, qualification evidence, component benefit or scientific claim was
produced.

The explicit fresh-member campaign boundary is now locally complete. A versioned JSON plan freezes
the exact source commit, development cases, independent member IDs/seeds, globally unique sessions,
matched case curriculum, PPO/environment controls, FE budget, resource envelope, A–F flags and
requested Safe-80 execution route. Separate scientific-design, execution-plan and seed-manifest
hashes prevent device routing from changing the equations while retaining the actual route. Start
requires a new output directory; resume is a separate command and verifies the unchanged plan,
recorded checkpoint path/SHA and authenticated session envelope. Per-transition checkpoints retain
exact progress; interruption is resumable, while a counted solver failure is recorded and may not
retry under the same campaign identity. Member candidates must have distinct training-run IDs and
source artifacts before unqualified ensemble assembly. The `calo-rpd-train-tsh` command requires the
plan's exact checked-out Git commit, a clean tracked plus non-ignored-untracked tree, and the matching retained clean
empty-policy development-freeze report and payload SHA-256; it has no experiment, registration,
qualification, activation or inference authority. Seven dedicated campaign cases, a 62-test adjacent
training/lifecycle/inference/optimizer focus and the 576-pass active tree pass; Ruff lint/format
passes across 403 Python files and the schema is current. Those validation cases use toy fixtures;
they do not establish qualification, protected-test, ablation or benefit evidence.

The first frozen IEEE 30/57 execution (`tsh-calo-ieee30-57-v1-20260803`) was retained as a failed
campaign rather than rewritten. Its plan was bound to source `d639630`, scientific design
`5f4a29361e8d8d93428fa71bdd054176c08f9d4eb99957e2e9e42f61f9d601b7`, execution plan
`490b0b0158a663a228071774c48a639f5cefaf7d70f6ac9d0d9cb6d8825fdc62` and seed manifest
`aa92dd769bc34cbc536b85f841379775773c3ae9639895357a9ebdbcc2baeb49`. Three members completed
10 episodes, 20,000 FE/scenario calls and 70 PPO updates each on admitted `cuda:0`; the fourth
reached episode 10 transition 20. Windows then denied one atomic status-file replacement. The
retained checkpoint independently authenticated and reports an unfailed session with complete
840/840 accounting. The v1 runner nevertheless classified every exception as terminal, so its
failed status is not bypassed. Recovery semantics now distinguish only an `OSError` with an
unfailed, accounting-complete active session as an explicit resumable infrastructure interruption;
solver, trainer, integrity and design failures remain terminal. This correction passes a synthetic
status-lock/resume test. The partial candidates are not an ensemble and have not been qualified or
used as evidence of benefit.

The fresh v2 campaign (`tsh-calo-ieee30-57-v2-20260804`) then completed under corrected
interruption semantics from source `78e6f800b675670365bebf58f876e3da4fef117d`. Its plan-file,
scientific-design, execution-plan and seed-manifest SHA-256 values are respectively
`2d63f2d48e9152e9ae51e9f0812ef63a9a0aac90d08d1115d2da19285e08554c`,
`b6aaf7fe785fc85bae31a9e52280a7e80d2161828e33d8c7a8a84e3d3d63b7f3`,
`e6b08c79c856a2e427e59bc5d187633f4db00f675e7cdd75424fd2e35dc931c2` and
`65d1113ac8adae9cd9990bac1683078d613faf60800f3c26ba46dd55701bc287`. Five independently seeded
members each completed 10 alternating IEEE 30/57 episodes, 20,000 candidate evaluations, 20,000
scenario power-flow calls, 490 canonical transitions and 70 PPO updates. Aggregate accounting is
100,000 evaluations/calls, 2,450 transitions and 350 updates. Every member used admitted `cuda:0`
NVIDIA computation with a 67,108,864-byte estimated working set below 80% of currently free VRAM;
no fallback occurred. The immutable ensemble SHA-256 is
`3adf5017dc51f33d76214aeb505da598984b9da0e4263e1ee8fe59a667180ceb`, and the campaign manifest
SHA-256 is `ded60598652d552a70f03c811969092ab243437f2d5adaf8d7f75f665bc80f33`. This remains a
`candidate_unqualified` artifact and real CUDA training provenance only. It is not CPU/CUDA parity,
calibrated uncertainty, policy benefit, component-ablation, qualification or activation evidence.

The independent qualification boundary is now implemented. A frozen plan and explicit
`calo-rpd-qualify-tsh` command fit OOD calibration from counted development-only topology states,
then run retained paired-seed, equal-FE candidate-versus-frozen-CALO cells. Every retained solution
is checked through the independent PYPOWER validator; evidence includes feasibility-first results,
paired practical effects, deterministic bootstrap intervals, Holm correction, frozen anytime
checkpoints, failures, device admission and exact policy/plan/seed/calibration hashes. The candidate
is exercised only through a non-serializable qualification capability; ordinary production
inference still requires a qualified, active immutable binding. Screening can never issue a receipt.
Formal receipt issuance additionally requires at least 30 pairs per case and checksum-verified,
accepted A–E component evidence. Change F is rejected from the production qualification plan.
Neither the qualifier nor its CLI can register or activate a policy. The first real screening attempt
under this harness is retained as failed-integrity evidence: shell timeouts left two Python children
running, a third explicit resume was started, and the three processes raced one output directory.
All six verified wrapper/child process IDs were stopped; four ambiguous cell files and their hashes
are retained under a `scientific_use_permitted=false` failure record. They are not screening results.
The qualifier now holds a non-blocking OS-released single-writer lease for the entire output directory
and refuses any failed-integrity resume. Six dedicated campaign cases and a 21-test focused
campaign/inference/optimizer pass.

A new, single-writer v3 screening identity
(`tsh-calo-ieee30-57-v2-screening-v3-20260804`) then completed validly from source
`33f29370ade972ee00ae07c22bf3d204a2dbaedd`. Its scientific-design, execution-plan and seed-manifest
identities are `9d1583703ee5ae211b00269dab6054676354d2d524ebf0085e92e17e63ce92a9`,
`0a3ba70c35b1601b850af49da2772aa59c127e2ca8682fdfe5bd627b5657c46a` and
`4cfe6933eea9abe26a4273db66a676458ff63450a53394c24c2aff730b8b22f1`. All 40 expected records are
unique and complete, with zero failure records, exact 2,000 candidate evaluations and scenario
power-flow calls per cell, paired seeds, independent validation of every retained solution, admitted
CPU policy inference, and no fallback. OOD calibration used 16 development-only states and 640
counted evaluations/calls; its logical identity is
`f90d93045cd31b918301ab801da701dc70b77ead151384c4f4080100628b485c`.

The v3 result is valid negative evidence. IEEE 30 produced no feasible run in either arm (0/10
baseline and 0/10 candidate), so objective inference is unavailable. IEEE 57 produced 10/10 feasible
runs in each arm and a median paired relative objective improvement of `0.011492392668353543`, but its
95% bootstrap interval `[-0.0019638316095621712, 0.01484160649928978]` crosses zero and the one-sided
Wilcoxon/Holm-adjusted value `0.052734375` misses the frozen `0.05` threshold. The immutable evidence
SHA-256 is `039f2bfe31e39196e126da3961c65e4a248133ed09b009a93f64c933b2292778`; its decision is grade `U`,
score `0.0`, `passed=false`, and claim scope `no qualification or policy-benefit claim`. Screening
correctly emitted no receipt and performed no registration or activation. Candidate v2 remains
unqualified and inactive, is not eligible for formal qualification under this frozen design, and
protected cases remain unopened. Do not weaken the criteria or reinterpret this screen as benefit.

Commits `ae7b304` and `e77431e` close the remaining repository-owned G9 evidence-development gaps;
they do not close the evidence gate. `calo-rpd-ablate-tsh` now executes a frozen development-only
matrix over frozen CALO, canonical refactor, graph-only, hierarchy-only, graph+hierarchy,
+uncertainty, +bandit and full approved A–E. It retains every paired equal-FE cell, independent
validation, falsification rows, anytime outcomes and globally Holm-controlled incremental decisions,
then emits checksum-bound A–E files with no lifecycle authority. Formal qualification now rejects a
bare accepted Boolean and requires exact source/policy/case/design/seed/analysis/authority bindings.
Production feature validation is unchanged and experimental F cannot enter these A–E files.

`calo-rpd-tsh-device-equivalence` now provides the separate physical candidate gate. On the same
immutable ensemble and development state it forbids fallback, requires exact CPU/CUDA deterministic
actions and masks, bounded numerical agreement for probabilities/parameters/uncertainty/value/OOD,
and verifies that the CUDA copy increased dedicated-VRAM allocation with every model parameter on
CUDA. Its focused tests pass, but no fresh counted-v4 candidate has executed it. The historical v2/
v3-ABI candidate remains ineligible.

Immediate CUDA-path development is also complete: production TSH evaluation and independent
training now use `AcceleratedORPDProblem.evaluate_population_with_context` for one bounded
population, not one CPU evaluator call per candidate. Seventy-six focused tests pass in 24.52
seconds, including evaluation, voltage, bus-type, branch-flow, Jacobian and sensitivity parity plus
a forbidden-reference-rerun guard. This is structural development evidence only. No greater-than-
95% end-to-end CUDA claim is accepted until target-device timing and dedicated-VRAM evidence binds
the current source and exact plan; fresh training remains paused until then.

Current legal execution order, without tuning:

1. Prove on the target NVIDIA device that the current counted-training path exceeds 95% eligible
   CUDA work, uses dedicated VRAM and performs no CPU-CUDA inner loop; port any measured remainder
   that prevents the gate from passing.
2. Commit/freeze one new v5/campaign-v2 development plan with A–E enabled and F disabled.
3. Train its fresh independent ensemble under the exact plan; do not adapt from intermediate results.
4. Run candidate-bound physical CPU/CUDA equivalence on that exact artifact.
5. Run the frozen A–E development matrix and retain failed as well as accepted component evidence.
6. Run the already-frozen screening/formal eligibility path only if its prerequisites pass.
7. Keep case118/case300 closed until the complete design/source/policy/container freeze.

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
9. population schedule as a separately experimental F study, never as formal A–E evidence; and
10. full approved A–E TSH-CALO.

Remove any component whose incremental evidence does not support its claimed benefit.

### G10 — complete scientific evidence

The harness is not the evidence. Execute and retain a frozen campaign after G9:

1. The safe checksummed PGLib-OPF typical/API/SAD import path and strict profile boundary are
   implemented. Populate only independently human-reviewed checksum-bound ORPD control profiles;
   never equate AC-OPF case loading with an ORPD formulation.
2. Disclosed SciPy SLSQP continuous-relaxation and exhaustive all-discrete reference adapters are
   implemented. Execute the frozen multistart/reference campaign and add certified bounds only where
   a separate mathematically valid certification exists.
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

Docker Desktop, Compose and Buildx are available on this Lenovo LOQ workstation. Exact clean
source-bound CPU/CUDA development images, attestations, SBOM/vulnerability reports, cross-container
lease evidence, parity/resource probes and a continuous physical CUDA soak are retained. Clean
commit `31a4713` also has a corrected Linux CPU GUI runtime proof with live-Qt health, xcb rendering,
restart/persistence and bounded cancellation. These are development qualifications, not the final
release images. Repeat the procedure for the immutable final candidate and trusted CI:

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

- Bounded physical FP64 CPU/CUDA evaluator parity is now retained for development case30/case57 on
  clean commit `63f56ad`, but it is not TSH-policy, pressure, soak, container, or performance proof.
- Bounded physical RTX 4060 VRAM pressure/recovery, host staging, controlled fallback/recovery, and
  multi-process lease evidence is retained at clean commit `d6a950c`.
- A one-hour physical RTX 4060 host soak is retained at clean commit `67bd18e`: 3,600 GREEN samples,
  no protection stop, verified 3,602-event hash chain, 46–60 °C, 12.18–26.0 W, and `24.33879127740349`
  Wh of scoped GPU-board-energy integration. CPU temperature, GPU power limit, whole-system energy,
  container execution, performance and policy benefit remain outside this evidence.
- Exact clean commit `1f02a94` has retained source-bound CPU/CUDA OCI attestations, loaded runtime
  smoke, zero-forbidden-file manifests, CycloneDX/full local Trivy reports with zero fixable
  HIGH/CRITICAL findings, one-device RTX 4060 visibility, cross-container lease exclusion/release,
  containerized case30/case57 parity plus resource recovery, and an independently audited continuous
  one-hour exact-image CUDA soak. The first GUI attempt was rejected because Qt xcb lacked
  `libxcb-shape.so.0` while the noVNC-only health probe briefly reported healthy in a restart loop.
  Clean correction commit `31a4713` produced image
  `sha256:f241c14c69d7896833e5805090d495f4ea14299de585cfb238ea13527b0deb5b`;
  build-time xcb closure, direct QApplication, live-app/all-child health, 1600x1000 Dashboard
  rendering, exact volume persistence across restart, new application process and bounded exit
  143/no-OOM cancellation passed. Render SHA-256 is
  `28108327353d3a491f8d92daf3f081d3e8bfb8b8a0d53bd9540d1a2484025187`.
  Browser interaction is not claimed because the desktop browser-control kernel failed before
  execution. Browser retry, CI rerun on the eventual final candidate and clean-machine reproduction
  remain open.
- No WSL2/WSLg target-laptop report yet.
- Clean `383e5bc` adds and locally exercises the installed-wheel Linux GUI boundary. Fresh wheel
  SHA is `27b94fecbf7ecdba85837c9c790d3d0d99a25f4bc07c62e2da73dc32f4e93479`, sdist SHA is
  `274bcbc078ad41da638264f6bce68268352104c566ce1dd4cf7d8379ddf69d20`, and the six-file staged
  manifest SHA is `8c47c03b42c63a09d747e922803a1b0ca812399dea320a1f19df45377deb1e4a`.
  The Linux wheel-only Dashboard render passed with PNG SHA
  `adc340f602011436ded5f321a55e5cb3855a8a0e1e50fe613032c1089789ca1f`. No GitHub Actions run
  artifacts from the updated workflow exist yet.
- A fresh five-member TSH-CALO ensemble exists, but it is unqualified, inactive and not benefit
  evidence.
- No approved-architecture ablation campaign yet.
- Official PGLib-OPF v23.07 case14 typical/API/SAD AC-OPF validation assets are now imported through
  exact code-rooted manifest/source hashes with retained CC-BY-4.0 attribution and package smoke
  evidence. They are deliberately not represented as an ORPD corpus: no human-reviewed external
  voltage/tap/shunt profile has been populated, and the runtime refuses to infer one.
- The mathematical-reference package and clean-source evidence CLI are implemented at `07f9476`.
  Bounded case30 development probes retained an SLSQP iteration-limit/infeasible projection and a
  six-point exhaustive no-feasible-point screen with independent PYPOWER agreement. These validate
  the interface and claim discipline only; no frozen multistart comparison, certified lower bound,
  feasible case30 reference optimum or protected-case result exists yet.
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

## 10. v12 Phase 1 coding handoff — 2026-08-06

- The active development identity is `12.0.0.dev1` with human label `12.0.0-dev.1`. It is not an RC
  or final release. Active status explicitly leaves final freeze, release qualification and
  protected-case evidence open.
- Historical v6.9 metadata, citation, status records and freeze remain unchanged and are indexed as
  historical-only. They do not qualify v12.
- Both qualification paths now require exact preregistered keys, share one positive-is-better
  relative-improvement definition and one signed-rank-mass effect, record the declared SciPy
  Wilcoxon method/version, and fail without switching test families. Old campaign plans/evidence do
  not silently resume under the v12 analysis ABI.
- Convergence evidence now separates time to first feasibility from post-feasibility incumbent AUC.
  Diversity-pressure behavior was not changed and remains approval-gated Class B scope.
- The retained v3 candidate remains immutable, unqualified and inactive. Its corrected effect is
  legacy/unverifiable from tracked immutable raw pair values, while the negative decision remains
  fixed because its interval crossed zero and Holm-adjusted `p=0.052734375` failed the threshold.
- Phase 1 tests were written but not executed. No Python test command, policy training, policy
  evaluation, campaign, benchmark or protected-case workflow ran. The exact safe validation and
  evidence-capture commands are in the local-only, Git-ignored
  `validation/Validate-Phase1.ps1` harness.
- The next action is user-run Phase 1 validation and return of the retained logs. Do not begin Phase
  2, policy training/evaluation, release-candidate versioning, or release production from this
  handoff without a new instruction and the appropriate gate authority.

## 11. v12 Phase 2 coding handoff - 2026-08-06

- Phase 1 manual evidence `phase1-20260806-230256` is accepted: 16/16 commands passed with complete
  source and validator hashing and no policy/scientific campaign execution.
- Phase 2 introduces one runtime-resolution record separating requested mode/device, physical
  UUID-or-PCI identity, logical CUDA index, runtime device, fallback policy/reason, actual compute
  device, lease scopes, Safe-80 admission, and CUDA-only claim eligibility.
- Formal work is `cuda_preferred` plus `execution_purpose=formal`: identified NVIDIA CUDA is
  required and CPU fallback is forbidden. Exploratory work may make only an explicit, fully
  provenance-labelled full-request CPU restart. CPU-only is concrete `cpu`. Final campaigns are
  formal and cannot opt down to CPU.
- CUDA physical-device leases are UUID-first, PCI-controlled fallback, host-scoped, queued, and
  separate from logical indices. Scheduler jobs remain bound to the device frozen before campaign
  rows are created.
- Batch cardinality and candidate identity are checked before FE registration. Partial failures
  retain exact count, incumbent, constraints, convergence/numerical state, runtime/fallback state,
  and checkpoint reference under a versioned schema.
- Dense/capacity fallback paths now report actual CPU computation and cannot contribute to
  CUDA-only timing, energy, parity, utilization, or equivalence claims. Active status keeps Intel
  XPU nonexecutable; historical records remain readable only as legacy evidence.
- Codex did not run tests, training, policy evaluation, qualification, benchmark, campaign, or
  protected cases. The next action is strictly manual validation:

  ```powershell
  Set-Location 'C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio'
  powershell -NoProfile -ExecutionPolicy Bypass -File .\validation\Validate-Phase2.ps1
  ```

- Return the complete new `validation/logs/phase2-YYYYMMDD-HHMMSS/` directory. Phase 2 and the
  transition to Phase 3 remain blocked on review of that evidence and correction of any failures.

### Phase 2 first-run correction checkpoint - 2026-08-07

- Reviewed `validation/logs/phase2-20260807-001858`: 13/15 commands passed, all 23 dedicated Phase 2
  contracts passed, and the affected runtime regressions reported 43 passed/1 failed.
- Retained evidence integrity passed independently: 20/20 artifact hashes and 33/33 declared source
  hashes matched, the validator identity matched, and no policy training/evaluation, qualification,
  benchmark, campaign, or protected-case workflow was recorded.
- Corrected the two failures without changing runtime semantics: reordered three generated-schema
  properties to match the authoritative generator and updated the historical VRAM regression to
  expect the Phase 2 Safe80 rejection message (`no greater than 0.80`).
- Updated active development status to bind the failed run and state that corrections are applied
  with a rerun pending. Codex did not execute tests or the validator while making these corrections.
- The next legal action remains the same manual command above. Return the complete newly timestamped
  Phase 2 log directory; do not begin Phase 3 until its evidence is reviewed and the gate closes.

### Phase 2 second-run formatting correction - 2026-08-07

- Reviewed `phase2-20260807-003024`: 14/15 commands passed. The corrected schema passed, all 23
  Phase 2 contracts passed, and all 44 affected regressions passed.
- All 20 retained artifact hashes matched; the 34-path manifest exactly covered the 34 captured
  changed paths; validator size/hash matched; and no policy, qualification, benchmark, campaign,
  or protected-case workflow executed.
- The sole failure was Ruff format detecting mixed line endings in
  `tests/unit/test_v690_vram_residency.py`. Normalized that file to its established CRLF style as a
  mechanical source correction. No Ruff command, test, validator, or other validation was run.
- Active status now binds this failed run as formatting-corrected/rerun-pending. Return a complete
  fresh Phase 2 manual-validation directory before closing the gate or starting Phase 3.

## 12. Phase 2 accepted; v12 Phase 3 GUI coding handoff - 2026-08-07

- Accepted `phase2-20260807-003828`: 15/15 commands passed, including 23/23 Phase 2 contracts,
  44/44 affected regressions, schema, Ruff diagnostics/format, compilation, and active identity.
  All 20 retained evidence hashes, 35/35 current source hashes, and validator identity matched; no
  policy, qualification, benchmark, campaign, or protected-case workflow executed.
- Created and confirmed the required Phase 3 goal before beginning GUI source development.
- Preserved all sixteen stable workspace keys and historical migration indexes while presenting
  them as Home, Model, Study, Evidence, and System groups in the required scientist-facing order.
- Added persisted compact/expanded navigation, persisted group collapse state, inline SVG icons,
  Ctrl+K workspace search, visible text status badges, hidden locked children, and an accessible
  blocked-workspace explanation.
- Rebuilt the Dashboard hierarchy around one Next required action, Data/Policy/Compute/Validation/
  Storage readiness, active context, recent experiments, resumable work, failures, evidence status,
  and a compact activity summary with a details drawer. The historical atomic protocol controls
  remain available for migration compatibility but are no longer displayed on the Dashboard.
- Organized Experiment Manager into seven steps: case, formulation, algorithms, budget/runs,
  scenarios, validation/outputs, and review/launch. It routes to authoritative shared panels rather
  than copying scientific state. Continuation and queue details use progressive disclosure.
- Added application-wide bounded input widths, 40/44px compact/comfortable targets, form-label
  buddies, accessible names, compact long-text editors with explicit expansion, integer chips for
  structured outage lists, and named light/dark semantic tokens on an 8px spacing system.
- Added a Phase 3 render-evidence CLI covering theme, size, scale, glyph availability, replacement
  characters, clipping, compact inputs, screenshot hashes, and isolated in-memory settings.
- Added Phase 3 source/unit contracts and updated affected Dashboard and Phase 2 status regressions.
  Codex did not execute any test, validator, render, formatter, linter, compile, or schema command.
- Prepared the Git-ignored `validation/Validate-Phase3.ps1` harness. It captures complete source and
  validator hashes, tests the relevant GUI/runtime boundaries, and renders 1280x720, 1440x900,
  1920x1080, light/dark, and a 200% scale cell. The agent did not run it.
- Phase 3 source coding is complete but validation remains open. Run the manual command below and
  return the complete new `validation/logs/phase3-YYYYMMDD-HHMMSS/` directory. Linux rendering and
  scientist acceptance remain separately required before the Phase 3 exit gate closes.

  ```powershell
  Set-Location 'C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio'
  powershell -NoProfile -ExecutionPolicy Bypass -File .\validation\Validate-Phase3.ps1
  ```

Until then, describe the branch as a modernization candidate with implemented harnesses and pending
scientific/hardware qualification—not as a final release.

## 13. Phase 3 first-run correction handoff - 2026-08-07

- Preserve `phase3-20260807-045558` as failed evidence: 11/18 commands passed. The run had 60/62
  tests pass, while all four Windows render cells failed with visibly unreadable tofu text because
  the PyQt offscreen platform could not discover a font.
- Corrections are implemented: twelve files mechanically formatted; brittle density and Dashboard
  assertions replaced; unsupported Qt CSS removed; deterministic existing-system-font registration
  and font provenance added; packaged/render glyph handling strengthened; failure counts clarified.
- The ignored validator now writes a v2 source manifest with commit, dirty state, Git-status hash,
  validator identity, and summary-bound source-manifest hash. Inaccessible `pytest-temp` contents
  are excluded from the durable evidence hash ledger.
- No application test, Ruff check, compile, render, validator, policy workflow, benchmark,
  qualification, campaign, or protected case was executed by Codex for these corrections. Ruff
  format was used only as the required mechanical source rewrite.
- Rerun manually and return the complete new directory:

  ```powershell
  Set-Location 'C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio'
  powershell -NoProfile -ExecutionPolicy Bypass -File .\validation\Validate-Phase3.ps1
  ```

- A Windows pass will not close Linux rendering, keyboard-only workflow review, or scientist
  acceptance. Phase 4 and release claims remain blocked while the Phase 3 gate is open.

## 14. Phase 3 Windows evidence accepted; remaining external gates encoded - 2026-08-07

- Accepted the user-run `phase3-20260807-052047` evidence read-only. It passed 18/18 commands:
  active identity, Git integrity, compilation, Ruff diagnostics/format, 11 Phase 3 contracts,
  16 GUI regressions, 35 Phase 2 presentation regressions, and four Windows light/dark/high-DPI
  render cells. The combined test count was 62 Phase 3/GUI passes plus 35 Phase 2 regressions.
- Verified the retained boundary reported by that run: source commit
  `00b8ee07a6d59c0d805d0c043c91ae5ea73d45d0`, dirty state, 31 Git-status lines with a bound status
  hash, 34/34 durable hashes, 32/32 current source hashes, and no ephemeral pytest artifacts in the
  durable hash set. No policy training/evaluation, qualification, benchmark, campaign, or protected
  case was executed.
- Reconciled `ACTIVE_DEVELOPMENT_STATUS.json` and the active-version/source contracts to identify
  `phase3-20260807-052047` as accepted Windows-local evidence while preserving
  `phase3-20260807-045558` as immutable failed history. The Phase 3 overall gate remains open.
- Added `validate_phase3_workspace_accessibility.py`. It directly audits all sixteen presentation
  surfaces in light/dark-capable Qt processes and retains screenshots, glyph/replacement/clipping,
  compact-input, long-editor, nested-scroll, accessible-name, form-buddy, ordinary-terminology,
  token-contrast, Ctrl+K, navigation persistence, Study Setup, and disclosure evidence. Direct stack
  access is explicitly presentation-only and does not claim workflow completion or scientific
  reachability.
- Extended the existing render collector with an explicit `--qt-platform xcb` lane so Linux desktop
  evidence can run under Xvfb instead of being silently forced to the offscreen plugin.
- Prepared three local-only ignored files: `validation/Validate-Phase3-Remaining.ps1` for current
  Windows source checks and mandatory human keyboard/screen-reader/scientist attestations,
  `validation/Validate-Phase3-Remaining-Linux.sh` for Linux xcb light/dark and all-workspace evidence,
  and `validation/PHASE3_REMAINING_VALIDATION.md` for exact commands and claim boundaries.
- Codex did not execute a test, compile, Ruff command, render, validator, GUI interaction, policy or
  scientific workflow, benchmark, qualification, campaign, or protected case. The user must run
  and return both new evidence directories. Automated evidence does not infer human acceptance.
- Phase 4 remains blocked until both directories pass review. At this historical checkpoint its
  design still contained fresh-candidate/protected work; section 19 supersedes that forward plan
  with coding-only development completion and a separate post-development policy transition.

## 15. Phase 3 remaining-gate failure corrected; validation made noninteractive - 2026-08-07

- Reviewed `phase3-remaining-windows-20260807-092741` read-only. It passed 5/8 automated commands;
  active identity, compilation, Ruff diagnostics and 12 Phase 3 contracts passed. Ruff format found
  three files, and both light and dark/200% all-workspace cells failed exactly two workspaces.
- Verified the failed evidence boundary: 45/45 durable hashes, 32/32 current source hashes, exact
  validator/current Git-status identity, commit
  `00b8ee07a6d59c0d805d0c043c91ae5ea73d45d0`, and no prohibited workflow. The separate supplemental
  Phase 2 rerun passed 14/15 and independently reported only the same three formatting targets; its
  20/20 durable and 56/56 source hashes matched and it does not invalidate accepted Phase 2.
- Retained screenshots proved real clipping. Results Explorer placed four labels, four selectors and
  three actions in one horizontal row, reducing `Open experiment workspace` from a 193px preferred
  width to 113px. Replaced it with a four-column labeled responsive grid and separate action row.
- Application Settings allowed the long result-database path to compress its form label to 43px and
  truncate the path. Replaced that row with a dedicated labeled, copyable, read-only, full-width path
  field; ordinary controls remain compact.
- Clipping evidence now records text, accessible name, tooltip, actual width/height and preferred
  width/height so future failures are diagnosable directly from JSON.
- Removed every interactive reviewer prompt from the ignored Windows validator. Its v2 lane now
  uses only source identity, formatting, focused behavior contracts, programmatic keyboard actions,
  accessibility semantics, contrast, all-workspace light/dark/high-DPI audits and retained hashes.
  It records that reviewer input is not collected and human screen-reader/scientist acceptance is
  not inferred.
- Added the noninteractive-only validator rule to repository instructions. The prior 10/10 manual
  answers, including `screen_reader=NA`, do not constitute evidence and are not carried forward.
- Applied Ruff format only as a mechanical source rewrite. The first pass reformatted two files and
  exposed an invalid hyphen in a new test function name; after correcting that identifier, the next
  pass reformatted the remaining file. A final two-file GUI-test pass changed nothing. No Ruff
  diagnostic or format-check gate ran.
- Codex did not execute tests, validators, compilation, lint checks, renders, GUI interactions,
  policy/scientific workflows, benchmarks, qualification, campaigns, or protected cases. The next
  action is the user-executed corrected Windows rerun followed by the separate Linux xcb lane.

## 16. Phase 3 Windows automation accepted; tabbed workspace refinement implemented - 2026-08-07

- Accepted `phase3-remaining-windows-20260807-112621` read-only for its exact pre-refinement source:
  all 10 commands passed, the 13 Phase 3 contracts and two focused GUI regressions passed, and all
  sixteen light plus dark/200% workspace audits passed. The 47/47 durable hashes, 36/36 source
  hashes, corrected validator identity, and current Git-status hash matched. No policy training or
  evaluation, scientific evaluation, benchmark, qualification campaign, or protected case ran.
- Implemented the user's subsequent presentation refinement without changing scientific state or
  behavior. ORPD Formulation, Robust Scenarios, Portfolio Manager, Application Settings, and
  Benchmark & Evidence now use a shared accessible `WorkspaceTabs` surface instead of long vertical
  section stacks. Related compact controls are arranged in side-by-side groups to use wide desktop
  space, while genuinely long database/manifest paths retain dedicated width.
- Added rendered contracts for the model-configuration tabs and Application Settings tab labels,
  plus source contracts covering all five workspaces and the shared tab surface.
- Upgraded the non-scientific all-workspace evidence schema to v2. It visits every shared section tab,
  checks keyboard switching and accessible tab/page metadata, and retains a source-bound screenshot
  for each tab in both Windows and Linux validation lanes.
- Updated the ignored Windows validator to include the new tab regression module. The existing exact
  manual command remains authoritative; no reviewer response is requested and no human acceptance is
  inferred.
- The accepted `112621` directory predates these edits and remains valid history rather than proof of
  current source. Codex did not run a validator, pytest, Ruff, compilation, render, GUI interaction,
  policy/scientific workflow, benchmark, qualification, campaign, or protected case. Run the updated
  ignored Windows validator and return its complete new directory, then return the Linux xcb lane.

## 17. Phase 3 tabbed-layout evidence failed; Portfolio width and evidence gate corrected - 2026-08-07

- Reviewed `phase3-remaining-windows-20260807-120240` read-only. Its validator and current status
  identities matched; 79/79 durable evidence hashes and 17/17 source entries matched. The run passed
  9/10 commands, including 14 Phase 3 contracts, five responsive-layout regressions, and both
  light/dark all-tab cells. No prohibited workflow ran.
- The formal failure was Ruff formatting on five files. Separate screenshot review found a genuine
  acceptance gap: Portfolio Manager's Requested outputs tree used only the left portion of its tab
  while truncating Minimum evidence text, and the v2 evidence schema did not inspect tree columns.
- Portfolio Manager now assigns a fixed selection column, stretching Output column, and
  content-sized Minimum evidence column, with accessible/stable tree identity. A rendered regression
  covers resize modes, full viewport use, and header fit.
- Workspace evidence schema v3 now records and fails on tree unused width, horizontal overflow, and
  retained header/cell clipping. The ignored Windows validator emits summary schema v4 with explicit
  tree-layout totals and failures. A mechanical formatter was applied to affected files, but no
  test, validation, format-check, compilation, render, policy/scientific workflow, benchmark,
  qualification, campaign, or protected case was executed by Codex.
- Run `powershell -ExecutionPolicy Bypass -File .\validation\Validate-Phase3-Remaining.ps1` and
  return the complete new timestamped Windows directory. Phase 3 remains open until that evidence
  passes review and the separate Linux xcb lane is also supplied.

## 18. Corrected Phase 3 Windows evidence accepted; Linux xcb remains - 2026-08-07

- Accepted `phase3-remaining-windows-20260807-121530` for its exact source-bound state. The v4
  summary passed 10/10 commands: source/whitespace/version/compile/Ruff gates, 15 Phase 3 contracts,
  six layout regressions, and light plus dark/200% v3 all-workspace evidence.
- The retained validator SHA-256 and current validator match. Before the post-acceptance ledger
  update, the retained and current Git-status hashes matched; 17/17 returned source entries and
  79/79 evidence hashes matched with no missing or mismatched file. The status/contract/ledger edits
  now intentionally postdate the manifest. The run requested no reviewer input and recorded no
  prohibited workflow.
- Each workspace cell passed five tab sets, sixteen tab screenshots, keyboard and contrast gates,
  with no failed workspace. The Portfolio tree-width record shows a 1054px viewport and header,
  zero unused/overflow width, and no clipped section. Visual screenshot review confirms the table
  fills the tab and Minimum evidence is readable in light and dark/200%.
- The Windows correction goal is closed. Phase 3 overall remains `open_linux_xcb_evidence_pending`:
  there is no returned `phase3-remaining-linux-*` directory, and the Windows report explicitly does
  not infer Linux rendering or human acceptance.
- Codex did not rerun tests or validation after recording acceptance. The next user-executed Linux
  xcb lane must bind the post-evidence ledger identities with current source.

## 19. Phase 4 redesigned as policy-free development completion - 2026-08-07

- The user directed that no old policy is final and that all old policies will be deleted after
  development before a completely new policy is trained. Old policies are therefore development-only,
  unqualified, inactive, non-final, and excluded from release; they are not reused as weights,
  qualification evidence, activation state, or a final baseline.
- The Phase 4 design now ends at a validated development freeze. It covers final A-E/F-off-capable
  code, canonical semantics/accounting, empty-policy GUI/CLI/database behavior, CUDA/runtime,
  containers, packages, CI, clean-machine behavior, old-policy inventory, and safe dry-run removal
  tooling. It executes no policy training/evaluation, qualification, protected campaign, deletion,
  registration, activation, or release operation.
- Post-Phase 4 actions are separately authorized and user-controlled: approve the exact deletion
  inventory, delete old policies with a receipt, verify empty-policy behavior, freeze a new training
  plan against the completed source, train and qualify a completely new policy, and select either a
  newly-qualified-policy or policy-free Phase 5 release scope.
- Phase 5 now owns both RC and final identities and must contain no old policy. Any included new policy
  has a separate immutable checksum/manifest; otherwise the release is explicitly policy-free and
  makes no policy-benefit claim.
- This decision supersedes the forward scheduling in older handoff sections that placed fresh
  candidate training inside the next phase. Those sections remain historical evidence of the earlier
  plan. Phase 3 Linux xcb evidence remains the only prerequisite before a Phase 4 goal may be created.

## 20. Live instruction and Markdown alignment - 2026-08-07

- All 60 live `AGENTS.md` files were read and checked against the new Phase 4 boundary. Relevant
  scoped files were updated; the 60 copies retained inside baseline and wheel-smoke artifacts were
  deliberately preserved as historical evidence.
- The 252-file Markdown inventory, including the new status index, was reviewed by role.
  Current-facing guidance was synchronized;
  dated audits, versioned reports, patch notes, release validation, and artifact copies were not
  rewritten as current proof.
- `docs/DOCUMENTATION_STATUS.md` is now the routing index. It points continuations to the active log,
  gates, v12 five-phase plan, handoff, and traceability ledger before older documents.
- Current README, user, architecture, reproducibility, container, portfolio, methodology, throughput,
  scientific-protocol, and proposal text now state that Phase 4 is development-only, empty-policy is
  supported, XPU is non-executable, old policies cannot be reused, and policy deletion/new training
  are separately authorized post-freeze actions.
- No test, validator, lint, formatting, compilation, schema, GUI, browser, Docker, package, policy,
  scientific, protected-case, or release command was executed. Phase 4 has not started and no Phase
  4 goal exists.

## 21. Phase 3 owner closure and Phase 4 start - 2026-08-12

- The project owner accepted the manually validated Linux xcb boundary and instructed continuation.
  No source/hash-bound automated Linux directory was retained. This closes the Phase 3 project gate
  by explicit owner decision while preserving the evidence limitation; it does not convert manual
  acceptance into automated proof or infer human accessibility/scientist acceptance.
- The retained Windows `phase3-remaining-windows-20260807-121530` directory remains the automated
  source-bound Phase 3 evidence. The Linux boundary is recorded as owner-accepted/manual only.
- The required Phase 4 goal was created before source development. Phase 4 baseline is clean commit
  `f800119cd3a14e2965c91040d0a8392013532089`.
- Resume by auditing the Phase 4 plan against current source, implementing uncovered development
  gaps, writing proportional tests without executing them, and ending with the ignored
  noninteractive Phase 4 validator for the user to run.

## 22. Phase 4 coding handoff; validation not yet executed - 2026-08-12

- Phase 4 production source now supports policy-free startup and rule-only CALO while exposing
  TSH-CALO as a separate immutable-policy-gated experiment path. Stale bindings are removed, old
  policies cannot activate/bind/qualify/resume/fork/delete through normal surfaces, and checkout
  policy files are not auto-discovered.
- New post-development provenance requires the future independent ensemble to identify the exact
  development-freeze source commit and retained freeze payload SHA-256, and prove an empty
  initialization-policy checksum. The training command requires the exact clean, empty-policy,
  post-transition freeze report; a dirty Phase 4 candidate cannot authorize training. Registry
  readiness accepts only that new TSH-CALO ABI plus independent qualification/activation. Scheduler-
  resolved device binding prevents internal CUDA-to-CPU policy migration; F remains separately off.
- Old-policy retirement is prepared without execution: exact file/database/external-reference
  inventory, cryptographic dry-run, disabled template, path/source/document authorization checks,
  transactional database cleanup, staged file removal, and an immutable external receipt. The GUI
  exports only the inventory/plan. No old policy or negative evidence was modified or deleted.
- Packaging/CI/development-freeze tooling now excludes generated policies and validation data,
  requires the new lifecycle source in distributions, keeps the physical Phase 4 CUDA lane
  policy-free, and binds a development-only report to the complete tracked/non-ignored source
  manifest, raw Git-status identity, exact declared interfaces, validator/instructions, and policy
  inventory. Its parser rejects incomplete self-hashed substitutes. No final manifest, SBOM claim,
  RC, policy benefit, protected-case, or release assertion was generated.
- The separate `calo-rpd-accept-development-freeze` command now enforces the missing decision
  boundary. It can consume only a fully passing, hash-complete Phase 4 run and writes a distinct
  receipt outside that run under an explicit decision ID. The validator never invokes it. After the
  returned run is reviewed, use the receipt SHA-256 in the separately authorized retirement record
  and every future new-policy plan/candidate. Do not create the receipt before log acceptance.
- The ignored manual harness is `validation/Validate-Phase4.ps1`; instructions are in
  `validation/PHASE4_VALIDATION.md`. Codex has not executed it or any constituent validation. The
  exact next action is `& .\validation\Validate-Phase4.ps1` from the repository root, followed by
  return of the entire newly created `validation\logs\phase4-YYYYMMDD-HHMMSS` directory.
- Keep the active Phase 4 goal open. Review returned logs read-only, preserve failed evidence, make
  source-backed corrections if necessary, and request a fresh run. Close the goal only after every
  required Phase 4 engineering gate and the source-bound development-freeze decision are accepted.
  Do not delete policies or begin Phase 5 automatically after closure.

## 23. Phase 4 final source audit complete; manual gate remains - 2026-08-12

- The final source-only audit corrected governing-policy readiness argument alignment and replaced
  its positional active-record construction with one keyword-labelled helper. A pre-freeze record
  now deterministically reports `development_only` instead of risking an exception.
- GUI policy-removal planning now uses its own `ResultDatabase(..., read_only=True)` handle. The
  accepted production-source contract excludes later-retired content only inside the designated
  trained-model store; a `.pt`-named file elsewhere remains bound to the source digest.
- The active status now truthfully records Phase 4 as started and Phase 3 as closed through the
  owner's manual Linux/xcb acceptance, without claiming a retained automated Linux directory.
  Current verifier/test source and release-plan routing use the same state.
- The ignored harness still has SHA-bound source, validator, instructions, policy inventory,
  distributions, CPU/CUDA images, GUI, physical CUDA parity/batching/recovery, and final source
  stability coverage. Its 30 numbered stages emit 32 exact result IDs; the separate acceptance
  parser requires that complete set and every retained file hash.
- No manual-capable check or validator was executed by Codex. The only next gate action is the
  user-executed command `& .\validation\Validate-Phase4.ps1`, followed by return of the entire new
  `validation\logs\phase4-*` directory. Phase 5, policy deletion, new-policy work, and acceptance-
  receipt creation remain outside this handoff.

## 24. Phase 5 development complete under combined-validation sequencing - 2026-08-12

- The owner directed Phase 5 development to finish before validation. Phase 4 development was
  closed, a Phase 5 goal was created, and the active tree advanced to Phase 5 development while
  retaining `12.0.0.dev1`, pending scope, and all release flags false.
- Release-policy scope is now explicit and fail-closed. Policy-free includes no policy and permits
  no policy-benefit claim. Newly-qualified-policy requires one exact new TSH-CALO A-E/F-off artifact,
  immutable qualification receipt, Phase 4 acceptance, clean post-transition freeze, empty old-
  policy initialization, and confined checksum-valid evidence. Neither route auto-activates.
- Release preparation now cross-binds exactly one wheel/sdist stage, separate archive/member
  manifests, installed-wheel GUI and CLI, independent wheel/sdist clean installs, CPU/CUDA image
  IDs, Buildx metadata, SBOMs, scanner/database identity, full vulnerability reports, real image
  filesystem manifests, and CI coverage. The preparation record cannot claim RC/final/release-ready.
- Final-record generation is a separate explicit authority and cannot operate at the current
  development identity. Even when later authorized, it only writes metadata/source manifests; it
  performs no version edit, Git action, publication, or release.
- Manual handoff is `validation/Validate-Phase4-And-Phase5.ps1`, documented by
  `validation/PHASE4_PHASE5_VALIDATION.md`. Return the new Phase 4, Phase 5, and combined directories.
  Do not begin deletion/policy workflows, choose scope by inference, promote `12.0.0`, or publish.

## 25. First combined-run corrections; fresh run required - 2026-08-12

- Preserve `phase4-20260812-165006` as interrupted partial diagnostics, not accepted evidence.
  Phase 5 did not start and the matching combined directory is incomplete.
- Known source diagnostics from that run were corrected without executing checks. The Phase 4 and
  Phase 5 command wrappers now stream live output, retain it per command, use native exit codes, and
  stop immediately at the first failure. The combined wrapper streams child output and cannot
  confuse a `phase4-phase5-*` directory with a Phase 4 child directory.
- The next action remains one fresh owner-executed combined command. If it fails, return the newly
  displayed Phase 4/Phase 5/combined directory paths; the first failure will now be visible without
  waiting for later container stages.
- Run `phase4-20260812-182252` subsequently demonstrated that behavior: five preliminary gates
  passed and `06-format` stopped Phase 4 before Phase 5 or any container work. Ruff mechanically
  formatted the eight named files; no follow-up check was executed by Codex. Preserve the failed
  run and execute one fresh combined run.
- Future combined failure summaries retain the failed child's directory, hashes, source identity,
  exit code, and command counts rather than leaving the child field null. This is evidence-routing
  metadata, not acceptance.
- Run `phase4-20260812-182752` passed environment/version/compile/schema/Ruff/format/mypy and all 112
  engineering tests. GUI was 36/37; the sole failure was visible “post-development” wording in the
  disabled qualification tooltip. It and related normal-interface phrases now use post-freeze,
  historical, or accepted-source wording. Preserve the failed run and rerun the combined validator.
- Run `phase4-20260812-184454` subsequently passed 14 result IDs through package construction.
  Distribution verification falsely rejected `calo_rpd_studio/validation/__init__.py`. The archive
  and container gates now allow/require that application package but still reject Git-ignored root
  validation evidence. Preserve this failed run; Phase 5 did not start.
- Run `phase4-20260812-185135` passed 17 result IDs, including the corrected distribution gate and
  clean wheel installation. `17-clean-smoke` then rejected a valid installed module solely because
  the clean virtual environment is intentionally stored below Git-ignored `validation/logs` inside
  the repository. The ignored validator now requires the import to be within the clean environment
  while excluding only the checkout's `calo_rpd_studio` source directory. Entry-point checks remain
  mandatory. Preserve the failed run; Phase 5 did not start and a fresh combined run is required.
- Run `phase4-20260812-190643` passed 24 result IDs through both locked container smokes and physical
  NVIDIA discovery. `23-cuda-parity-30` stopped before parity because durable accelerator evidence
  correctly rejected the dirty Phase 4 development tree. The physical parity, CUDA hot-path, and
  recovery CLIs now have an explicit development-only dirty-source option used only by the ignored
  Phase 4 validator. It requires a full commit and can report engineering success, but records
  non-durable status and cannot set qualification true. Default formal behavior still requires clean
  source. Preserve the failed run; Phase 5 did not start and fresh combined validation is required.
- The same overly broad repository-root predicate was removed proactively from the ignored Phase 5
  wheel/sdist/package-GUI isolation checks. They now require the relevant isolated environment and
  reject the checkout source package specifically, avoiding another false failure caused by storing
  local evidence beneath Git-ignored `validation/logs`.
- The ignored combined wrapper now preflights Python, Docker, NVIDIA-SMI, and Trivy and records the
  resolved executables before launching Phase 4. The owner must install missing Trivy manually; the
  validator will now report that prerequisite immediately.
- Run `phase4-20260812-195901` passed all 32 Phase 4 result IDs and is the current complete automated
  Phase 4 checkpoint for its exact recorded source state. The Phase 5 typing correction subsequently
  changed source, so the bundle does not validate the current tree and cannot create the separately
  gated Phase 4 acceptance receipt or clean final-source evidence without a fresh combined run.
- Phase 5 child `phase5-20260812-201822` passed five preliminary gates and stopped at `06-types`.
  The untyped PyYAML import is now narrowly annotated while runtime mapping validation remains; the
  final-record loader now requires JSON objects for the Phase 5 summary and all later authorization,
  scope, freeze, preparation, and combined inputs. Preserve the failed combined directory
  `phase4-phase5-20260812-195901`; rerun combined validation after correction.
- Combined retry `phase4-phase5-20260812-202511` stopped at Phase 4 `02-version`. Only the active
  status runtime contract failed because the verifier expected obsolete pre-validation ledger text;
  all identity, non-release, policy, GUI, container-label, and qualification-version checks passed.
  The verifier is aligned to the current ledger and now checks seventh/eighth attempt history
  separately. Preserve the failed directories and run the full combined validator again.
- Combined retry `phase4-phase5-20260812-202852` passed Phase 4 completely. Phase 5 child
  `phase5-20260812-204823` passed 23 commands and failed first at `22-cpu-build`, where Docker's
  classic image store rejected the required provenance/SBOM attestations. Do not remove or weaken
  those flags. Enable Docker Desktop's containerd image store and restart it. The ignored combined
  wrapper now verifies `io.containerd.snapshotter.v1` before starting Phase 4 and retains the check
  in its preflight log; then run the complete combined validator again.
