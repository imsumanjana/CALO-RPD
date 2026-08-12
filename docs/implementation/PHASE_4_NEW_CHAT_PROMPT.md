# New-chat prompt: Phase 4 development completion and freeze

> **Superseded execution prompt as of 2026-08-12.** Phase 4 development is complete and Phase 5
> release-preparation development has also been implemented under the owner's combined-validation
> sequence. Do not restart Phase 4 from this prompt. Use `ACTIVE_CONTINUATION_LOG.md` and
> `validation/PHASE4_PHASE5_VALIDATION.md` for the current manual handoff.

> Use this entire file as the first request in a new Codex chat. It is intentionally self-contained,
> but the new chat must verify every status statement against the live repository before acting.

## Role and objective

Work in this repository:

`C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio`

Complete **Phase 4: Production development completion, empty-policy hardening, and development
freeze** for the CALO-RPD Studio v12 line. This is a coding-development phase. It must finish and
harden the production implementation, create or update the required test source and validation
automation, update the authoritative documentation, and end with a source-bound development-freeze
candidate for manual validation.

Do not treat this prompt as proof that Phase 3 is complete or that Phase 4 is authorized to start.
The live worktree, applicable `AGENTS.md` files, active ledgers, and returned source-bound evidence
take precedence.

## Mandatory first actions and start gate

1. Confirm the exact repository path and inspect the current working tree without discarding,
   overwriting, staging, committing, or attributing unrelated/user-owned changes.
2. Read the root `AGENTS.md` completely, then read every deeper `AGENTS.md` applicable to each path
   before editing that path.
3. Read these current records in this order:
   - `docs/DOCUMENTATION_STATUS.md`;
   - `docs/implementation/ACTIVE_CONTINUATION_LOG.md`;
   - `docs/implementation/IMPLEMENTATION_GATES.md`;
   - Phase 4 and the post-Phase 4 transition in
     `docs/implementation/POST_V6_9_RELEASE_UPDATE_AND_FIX_PLAN.md`;
   - `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md`;
   - `docs/implementation/REQUIREMENT_TRACEABILITY.md`;
   - `docs/implementation/CALO_ARCHITECTURE_CHANGE_PROPOSAL.md`;
   - `docs/implementation/SCIENTIFIC_VALIDATION_PROTOCOL.md`.
4. Verify that Phase 3 is formally closed in all authoritative ledgers. On 2026-08-12 the project
   owner explicitly accepted the manually validated Linux xcb boundary and directed Phase 4 to
   proceed; no automated Linux evidence directory was retained. Preserve that limitation and do
   not reinterpret the owner decision as automated proof or human accessibility/scientist evidence.
5. If Phase 3 is still open, do not create a Phase 4 goal, do not state that Phase 4 has started, and
   do not edit Phase 4 source. Report the exact blocker and the precise user-run validation command or
   missing log directory, then stop.
6. If and only if the Phase 3 prerequisite is directly verified as closed, create a new goal with the
   goal service before stating or starting development. Use this objective:

   `Complete CALO-RPD v12 Phase 4 production development, empty-policy hardening, engineering and release-infrastructure completion, controlled old-policy removal preparation without deletion, authoritative record updates, and a source-bound development-freeze validator for manual execution.`

7. If another goal is unfinished, resolve it according to the goal-service rules before creating the
   Phase 4 goal. Do not silently replace or abandon an active goal.

## Controlling development and scientific boundaries

- Perform coding development only. You may implement production code, test source, schemas,
  migrations, CI definitions, packaging/container definitions, documentation, and validation
  harnesses.
- Do not run tests, validators, pytest, tox, coverage, Ruff, format checks, lint, mypy, compile checks,
  schema checks, build/package smoke tests, GUI/browser checks, Docker commands, benchmarks, hardware
  qualification, campaigns, or any other validation that the user can run manually.
- Do not perform policy training, policy evaluation, qualification, protected-case execution,
  registration, activation, candidate creation, policy promotion, or policy deletion.
- Treat every existing policy, checkpoint, member, lineage, receipt, registration, and activation
  record as development-only, unqualified, inactive, non-final, and excluded from release and from
  initialization of the future policy.
- Do not use an old policy as a final baseline, candidate, initializer, qualification input, or
  release artifact. Do not reinterpret historical negative evidence as success.
- Preserve the exact approved scientific boundary: TSH-CALO A-E are approved; F is experimental,
  independently feature-flagged, evidence-gated, and disabled by default.
- Preserve deterministic behavior, exact function-evaluation and scenario accounting,
  mixed-variable validity, fail-closed nonconvergence, protected-case isolation, immutable identity,
  and exact resume semantics.
- Intel XPU must remain non-executable. Current modes are CUDA-preferred and CPU-only.
- Admission ceilings are at most 80% of currently free VRAM or currently available RAM. Do not
  fabricate GPU, performance, thermal, energy, container, scientific, or release evidence.
- Do not generate final release freezes, final manifests, SBOM claims, RC/final identities, public
  claims, tags, pushes, publications, or releases during Phase 4.
- Do not push, merge, publish, stage, or commit unless the user explicitly authorizes the exact action.
- Do not use subagents unless the user explicitly requests delegation.

## Required Phase 4 development

### 1. Complete the approved scientific and runtime implementation

- Finish the approved A-E implementation against one canonical runtime/training-transition
  authority while retaining F as separately gated and off by default.
- Close remaining production gaps in whole-population counted-context batching, selected-device
  retention, outer-boundary materialization, and exact accounting. No hidden power-flow/context
  reruns or candidate-level CPU-CUDA transfer loops are permitted.
- Complete CUDA-preferred and CPU-only behavior. Formal CUDA execution must bind an identified NVIDIA
  device and fail closed without fallback. Any exploratory fallback must be an explicitly identified
  full-request CPU restart, never a hidden mid-request migration.
- Implement proportional unit, invariant, parity, failure, resume, fallback, leakage, protected-case,
  and regression test source without executing it.
- Before any semantic CALO refactor, preserve or extend the frozen canonical-parity contract. Do not
  change scientific semantics beyond the approved A-E/F boundary without explicit user approval.

### 2. Make empty-policy operation a first-class supported state

- Make GUI, CLI, database, migration, restart, recovery, and clean-install behavior safe when no
  policy artifacts, registrations, or active-policy records exist.
- Keep non-policy configuration, baseline workflows, result inspection, diagnostics, lifecycle
  administration, and future training preparation usable through an explicit safe fallback.
- Keep policy-dependent formal experiments visibly locked until a future policy is independently
  qualified, registered, explicitly activated, immutable, checksum-valid, and ABI-compatible.
- Reject missing, corrupt, incompatible, unqualified, inactive, or stale references with clear,
  deterministic explanations. Never silently fabricate, download, select, regenerate, or reinterpret
  a policy.
- Preserve strict separation among training, qualification, registration, activation, and immutable
  experiment binding. None may trigger another automatically.

### 3. Complete production, package, container, and CI hardening

- Complete deterministic clean-install, migration, rollback, configuration, result, interruption,
  cancellation, recovery, and corrupt-artifact behavior.
- Complete wheel/sdist staging, installed entry points, package-data exclusions, and
  checkout-independent execution without generating final release manifests early.
- Complete CPU and NVIDIA CUDA container definitions with locked dependencies, non-root runtime,
  read-only root, dropped capabilities, no-new-privileges, bounded temporary storage, persistent data,
  health, restart/cancellation, and shared physical-device leases.
- Ensure old/generated policies, checkpoints, user data, results, logs, credentials, caches, and
  validation material cannot enter distributions, images, manifests, or default runtime state.
- Complete trusted CI definitions for supported Python/OS lanes, packaged GUI rendering,
  distribution/container checks, and the separately run physical NVIDIA evidence lanes.
- Prepare noninteractive, source-bound manual validation automation for physical batching/VRAM,
  CPU/CUDA evaluator parity, containers, distributions, clean-machine behavior, and empty-policy
  startup. Implement it, but do not run it.

### 4. Prepare controlled old-policy removal without deleting anything

- Implement a read-only inventory of old policy files, members, checkpoints, lineages, receipts,
  registry/activation references, database rows, package inclusion, dependencies, and exact SHA-256
  identities.
- Implement a path-confined dry-run removal plan that refuses any target outside the designated
  policy store and reports all dependent references.
- Implement transactional deactivation/reference cleanup and a separate deletion action that
  requires explicit authorization and emits an immutable deletion receipt.
- Do not execute the deactivation or deletion action in Phase 4.
- Use temporary synthetic fixtures for removal and compatibility tests. Preserve historical negative
  evidence without treating its executable policy artifact as release content.

### 5. Prepare the source-bound development freeze

- Close every Phase 1-4 development, runtime, GUI, empty-policy, physical-device, container, package,
  migration, provenance, and traceability row with returned direct evidence or an explicit blocker.
- Bind the exact source commit, dirty/clean state, schemas, dependencies, environment, supported
  devices, container declarations, validation-script identity, and relevant source hashes.
- Freeze the development interfaces needed by the later new-policy process: A-E/F-off configuration,
  policy ABI, training environment, evaluator, decoder/repair semantics, accounting, curriculum,
  receipt schema, and qualification-authority boundaries.
- Record explicitly that the development freeze contains no qualified/active/final policy and makes
  no policy-benefit, superiority, protected-case, RC, final-release, or release-readiness claim.

## Working method

- Inspect before editing and preserve unrelated work. Use `rg`/`rg --files` for searches and
  `apply_patch` for file edits.
- Keep tool calls and output narrowly scoped. Do not repeatedly dump the same ledger or source.
- Implement the complete coherent change rather than leaving only a plan when safe in-scope coding
  remains.
- Add test source proportional to every production change, but leave execution to the user.
- Never alter or erase failed evidence. New evidence must use a new timestamped directory.
- Update all four authoritative records together after material work:
  1. `ACTIVE_CONTINUATION_LOG.md` with exact changes, commands not run, blockers, and next action;
  2. `IMPLEMENTATION_GATES.md` with only directly evidenced status;
  3. `REQUIREMENT_TRACEABILITY.md` with implemented coverage and remaining proof;
  4. `RELEASE_READY_CONTINUATION_HANDOFF.md` with the exact continuation boundary.
- Keep `docs/DOCUMENTATION_STATUS.md`, README, architecture, reproducibility, user guide, and runbook
  synchronized when behavior or routing changes.

## Mandatory manual validator handoff

At the end of the Phase 4 coding pass, create or update a detailed, noninteractive PowerShell
validator under the Git-ignored `validation/` directory. It must:

- create a new timestamped log directory on every run;
- record repository path, source commit, dirty/clean status, status hash, environment, dependencies,
  exact command lines, start/end timestamps, exit codes, stdout/stderr, and summaries;
- record SHA-256 hashes for the validator, relevant source files, schemas, manifests, and produced
  evidence;
- cover all manually runnable Phase 4 engineering gates, including empty-policy GUI/CLI/database,
  CPU/CUDA parity and physical batching, packages, containers, clean-machine behavior, migration,
  recovery, exclusion, and development-freeze identity;
- fail closed on missing commands, missing artifacts, hash mismatches, prohibited policy content,
  incomplete evidence, or a test failure;
- never prompt for reviewer names, roles, PASS/FAIL answers, screen-reader declarations, scientist
  acceptance, or any `Read-Host` input;
- state explicitly that automation does not infer human screen-reader/usability/scientist acceptance;
- perform no policy training/evaluation, qualification, activation, registration, deletion,
  protected-case campaign, or release publication;
- remain absent from Git, manifests, distributions, container contexts, and release artifacts.

Do not run the validator. Give the user one exact PowerShell command to run it and ask for the entire
new timestamped log directory. Also save a short ignored Markdown instruction file beside the
validator describing prerequisites, the exact command, expected duration/space, outputs, and how to
return the logs.

When the user returns logs, review them read-only. Accept only source/hash-bound evidence. Make
evidence-backed coding corrections, update the same ignored validator when necessary, and request a
fresh noninteractive rerun. Never take manual PASS/FAIL answers from a reviewer.

The validator must never self-accept. After a complete passing directory is reviewed and the owner
explicitly accepts the Phase 4 gate, create the separate non-overwriting development-freeze
acceptance receipt outside the immutable run. Bind it to the complete hash manifest and production-
source content contract, and require its SHA-256 for later old-policy authorization and every new-
policy plan/candidate. Do not create this receipt before returned-log acceptance.

## Completion and goal handling

Do not mark Phase 4 complete merely because code or a validator exists. Completion requires returned
direct evidence satisfying the Phase 4 exit gate, safe empty-policy operation, old-policy inventory
and exclusion, deletion preparation without deletion, and a source-bound development-freeze
decision. Phase 4 does not produce a trained policy, release candidate, or final release.

When—and only when—the Phase 4 objective is genuinely achieved with no required work remaining, mark
the Phase 4 goal complete through the goal service and report its final goal usage. Otherwise keep the
goal active and state the exact manual evidence or external condition still required.

After Phase 4 closes, do not automatically delete policies or begin Phase 5. The separately
authorized post-Phase 4 transition must review the inventory, authorize deletion, retain its receipt,
verify the empty store, decide whether to train and qualify a completely new A-E/F-off policy, and
record either a newly-qualified-policy or policy-free Phase 5 scope.
