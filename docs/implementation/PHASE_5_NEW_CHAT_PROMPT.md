# New-chat prompt: Phase 5 clean release reproduction and final release

> **Superseded execution prompt as of 2026-08-12.** Phase 5 release-preparation development was
> completed in the active chat after the project owner explicitly chose combined Phase 4/5 manual
> validation. Do not paste this file to restart or duplicate Phase 5. Continue from
> `ACTIVE_CONTINUATION_LOG.md` and run the ignored combined validator documented in
> `validation/PHASE4_PHASE5_VALIDATION.md`. The prerequisite list below remains useful as the later
> final-release gate; it is not evidence that those decisions or validations have passed.

> Use this entire file as the first request in a new Codex chat. It is intentionally self-contained,
> but the new chat must verify all prerequisites from the live repository and retained evidence.

## Role and objective

Work in this repository:

`C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio`

Complete **Phase 5: Clean release-scope reproduction, packaging, and final release** for CALO-RPD
Studio v12. Phase 5 must reproduce one clean, internally consistent release whose source, approved
policy scope, distributions, images, evidence, metadata, manifests, documentation, and public claims
bind to the recorded immutable identities.

This is a coding and release-engineering development phase under a manual-validation workflow. You
may implement or correct production/release code, tests, schemas, workflows, packaging, container
definitions, documentation, and validation automation. Do not execute manual-capable tests, builds,
containers, campaigns, or validators yourself. The user will run the validator and return the full
logs for read-only review.

## Non-negotiable Phase 5 start gate

Before creating a goal or editing Phase 5 source:

1. Confirm the repository path and inspect the current working tree without discarding, overwriting,
   staging, committing, or attributing unrelated/user-owned changes.
2. Read the root `AGENTS.md` completely and every deeper `AGENTS.md` that applies to a path before
   editing it.
3. Read these current records in order:
   - `docs/DOCUMENTATION_STATUS.md`;
   - `docs/implementation/ACTIVE_CONTINUATION_LOG.md`;
   - `docs/implementation/IMPLEMENTATION_GATES.md`;
   - Phase 4, the post-Phase 4 transition, and Phase 5 in
     `docs/implementation/POST_V6_9_RELEASE_UPDATE_AND_FIX_PLAN.md`;
   - `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md`;
   - `docs/implementation/REQUIREMENT_TRACEABILITY.md`;
   - `docs/implementation/SCIENTIFIC_VALIDATION_PROTOCOL.md`;
   - the final Phase 4 validator instructions and returned accepted log directories.
4. Verify all of the following from direct, source/hash-bound evidence:
   - Phase 3 is closed;
   - Phase 4 is closed and its goal is complete;
   - the exact clean Phase 4 development-freeze commit and retained freeze payload SHA-256 are
     recorded;
   - the separate explicit Phase 4 acceptance receipt exists, matches the accepted log/source
     contract, and its SHA-256 is recorded;
   - the old-policy inventory and dry-run removal plan were reviewed;
   - the user separately authorized the exact old-policy deletion targets;
   - deletion completed with an immutable receipt and no out-of-scope target;
   - the policy store is verified empty and the frozen application behaves safely in that state;
   - the post-development policy decision is recorded as exactly one of:
     1. `policy-free`, with no policy-benefit claim; or
     2. `newly-qualified-policy`, naming one completely new A-E/F-off policy with exact immutable
        checksum, ABI/provenance, qualification evidence, and separate manifest;
   - no old policy weights, activation state, qualification, or evidence were reused for that policy.
5. If any prerequisite is absent, ambiguous, failed, or only asserted in prose, do not create the
   Phase 5 goal and do not edit Phase 5 source. Report the exact missing evidence and stop. Do not
   perform the missing deletion, policy training, evaluation, qualification, protected campaign,
   registration, activation, or policy-scope decision inside this Phase 5 task.
6. If and only if every start condition is verified, create a new Phase 5 goal through the goal
   service before stating or starting development. Use this objective:

   `Complete CALO-RPD v12 Phase 5 clean release-scope reproduction, distribution and CPU/CUDA image hardening, final CI and clean-machine validation automation, immutable manifest/SBOM/metadata/documentation generation, and evidence-gated v12.0.0 release preparation for the already approved policy scope, without publishing or releasing without explicit authorization.`

7. If another goal is unfinished, resolve it under the goal-service rules before creating Phase 5.

## Approved-scope rules

- Bind Phase 5 to exactly the recorded `policy-free` or `newly-qualified-policy` decision. Do not
  switch scope silently or infer a policy choice from files found in the worktree.
- In a policy-free scope, exclude every policy/checkpoint/lineage/receipt from release artifacts and
  make no policy-benefit, qualification, or policy-assisted-performance claim.
- In a newly-qualified-policy scope, include only the one exact post-development policy named in the
  approved decision. It must be immutable, checksum-valid, ABI-compatible, independently qualified,
  A-E/F-off, separately manifested, and never treated as automatically active.
- No old policy artifact may be tracked, packaged, copied into an image, regenerated, downloaded,
  selected, or included as compatibility payload.
- Historical negative evidence remains preserved as records but cannot qualify the release policy.
- F remains independently feature-flagged, experimental, disabled by default, and excluded from A-E
  production qualification.

## Execution and authorization boundaries

- This agent performs coding and release-engineering development only. Implement test and validation
  sources but do not execute manual-capable tests or validation.
- Do not run pytest, tox, coverage, Ruff, formatting, lint, mypy, compile/schema checks, package
  builds, installed-wheel checks, GUI/browser checks, Docker/Compose/Buildx, vulnerability scanners,
  SBOM tools, CUDA jobs, benchmarks, campaigns, protected cases, clean-machine runs, or release
  commands. Put the exact commands in the ignored Phase 5 validator for the user.
- Do not train, retrain, fine-tune, evaluate, qualify, register, activate, promote, delete, or modify
  any policy during Phase 5.
- Do not fabricate or infer CI, hardware, GUI, container, distribution, security, scientific,
  performance, thermal, energy, or clean-machine evidence.
- Intel XPU is non-executable. Release modes are CUDA-preferred and CPU-only, with admission bounded
  by at most 80% of currently free VRAM or currently available RAM.
- Preserve deterministic semantics, exact FE/scenario accounting, protected-case isolation,
  fail-closed behavior, and the approved A-E/F boundary.
- Do not regenerate final `VERIFIED` records from source existence or hand-authored assertions.
- Do not tag, push, publish, upload, create a public release, or mark v12.0.0 final without explicit
  user authorization after every acceptance gate has direct evidence.
- Do not stage or commit unless the user explicitly authorizes the exact action.
- Do not use subagents unless the user explicitly requests delegation.

## Required Phase 5 development

### 1. Freeze source identity and approved policy scope

- Start from or reproduce a clean clone of the exact accepted Phase 4 development-freeze commit and
  verify the retained freeze report by its payload SHA-256 plus the separate Phase 4 acceptance
  receipt by its receipt SHA-256 and production-source content contract.
- Fail closed if the source is dirty, the commit is unavailable, the Phase 4 freeze identity differs,
  or the approved policy-scope record is missing or inconsistent.
- Confirm exclusions for generated policies, old policies, checkpoints, lineages, databases, results,
  logs, screenshots, credentials, caches, user data, publication exports, validation directories, and
  local environment files.
- For a newly-qualified-policy scope, bind the policy through a separate immutable manifest; never
  merge policy identity into or substitute it for source identity.
- Complete the requirement audit in code/documentation and expose every unresolved traceability row
  to the manual validator.
- Keep version identity truthful. An RC may be prepared only when its prerequisites are evidenced.
  Promote the accepted RC to exactly `12.0.0` only after all Phase 5 acceptance criteria pass.

### 2. Complete distribution reproduction

- Implement a fail-closed, previously absent staging-directory workflow.
- Prepare exactly one wheel and one sdist from the clean immutable release source.
- Prepare installed-distribution checks outside the checkout with checkout `PYTHONPATH` removed.
- Cover every required CLI and GUI entry point from the installed wheel.
- Produce distinct immutable source-tree, wheel, and sdist manifests and cross-checks.
- Ensure neither distribution includes policy/user/generated data outside the exact approved scope.

### 3. Complete CPU and CUDA image reproduction

- Prepare immutable CPU and NVIDIA CUDA images from the same exact source commit and locked
  dependencies.
- Implement retention and verification for image digests, maximum provenance, embedded/external
  SBOMs, scanner identity and database timestamp, full vulnerability reports, and application/filesystem
  manifests.
- Require CPU execution without NVIDIA access and CUDA execution with exactly the selected GPU.
- Cover non-root UID/GID, read-only root, dropped capabilities, no-new-privileges, bounded temp,
  persistent data, health, restart, cancellation, and shared physical-device leases.
- Prepare final-source parity, memory pressure, staging, controlled fallback, recovery, cancellation,
  soak, thermal, power, and scoped energy validation without extrapolation.
- Include policy-specific image/runtime gates only for the exact approved newly-qualified-policy
  scope; otherwise prove their exclusion.
- Prepare WSL2/WSLg and target-laptop qualification automation and evidence retention.

### 4. Complete final CI and clean-machine reproduction automation

- Implement or finalize unit, invariant, parity, migration, GUI, accessibility, integration,
  regression, scientific-scope, container, distribution, and release-integrity lanes applicable to
  the exact source and policy scope.
- Cover the supported Python/OS compatibility matrix and trusted self-hosted physical CUDA jobs.
- Retain installed-wheel GUI renders and reproducible browser/client interaction artifacts.
- Prepare clean target-system reproduction for wheel, sdist, CPU image, and CUDA image.
- Bind every artifact upload and report to the exact source identity and, only when applicable, the
  separate new-policy identity.

### 5. Prepare generated final records and release documentation

- Generate release metadata only from retained executed-gate outputs; do not hand-author success.
- Generate the final source freeze only after the exact source and artifacts are immutable and
  accepted by returned evidence.
- Generate distinct root/source, wheel, sdist, CPU-image, CUDA-image, and optional policy manifests.
- Prepare scientific-equivalence, hardware, GUI, container, validation, vulnerability, and
  clean-machine records from the corresponding retained evidence.
- Synchronize version metadata, README, user guide, changelog, implementation report, audit closure,
  citation, installation/container instructions, known limitations, and claim scope.
- Verify every cross-reference and digest against the exact commit/artifact through the manual
  validator.
- Stop before tag, push, publication, or release and request explicit user authorization.

## Working method and record maintenance

- Inspect before editing, preserve unrelated changes, use `rg` for search, and use `apply_patch` for
  file edits.
- Keep scans and outputs narrowly scoped to reduce token use. Do not repeatedly reread or dump the
  same evidence.
- Add proportional production and test source, but leave execution to the user.
- Never delete or overwrite a failed validation directory. Every rerun receives a new timestamped
  directory and remains immutable evidence.
- Update all four authoritative records together after material work:
  1. `ACTIVE_CONTINUATION_LOG.md`;
  2. `IMPLEMENTATION_GATES.md`;
  3. `REQUIREMENT_TRACEABILITY.md`;
  4. `RELEASE_READY_CONTINUATION_HANDOFF.md`.
- Update `docs/DOCUMENTATION_STATUS.md` and current-facing release documentation whenever routing,
  behavior, version, or claim scope changes.
- Record direct evidence separately from implemented harnesses, user-returned local validation,
  physical/external proof, scientific qualification, RC acceptance, and final authorization.

## Mandatory manual Phase 5 validator

At the end of each Phase 5 coding/correction pass, create or update one detailed, noninteractive
PowerShell validator under the Git-ignored `validation/` directory. Do not run it. It must:

- create a new timestamped log directory for every run and refuse to overwrite prior evidence;
- record exact repository/source/policy-scope identities, commit and dirty state, status hash,
  environment/dependency identities, commands, timestamps, exit codes, stdout/stderr, and summaries;
- record SHA-256 for itself, relevant source, schemas, dependency locks, distributions, images,
  manifests, SBOMs, reports, GUI renders, and evidence files;
- fail closed unless the Phase 4 freeze and post-Phase 4 policy-scope prerequisites match exactly;
- exercise the complete applicable Phase 5 unit/CI, package, installed-distribution, GUI,
  accessibility, container, clean-machine, physical-CUDA, integrity, exclusion, manifest, metadata,
  and release-readiness gates;
- skip policy-specific gates only when the approved scope is explicitly policy-free, and record that
  exclusion and the absence of policy-benefit claims;
- never train, evaluate, qualify, register, activate, delete, or modify a policy;
- never use `Read-Host` or request reviewer names, roles, PASS/FAIL answers, attestations,
  screen-reader declarations, scientist acceptance, or other manual answers;
- record that automation cannot infer human screen-reader/usability/scientist acceptance;
- remain absent from Git, manifests, packages, images, SBOM inputs, and release artifacts.

Save an ignored Markdown instruction file next to the validator containing prerequisites, the exact
PowerShell command, expected duration/disk requirements, any separately required trusted-runner or
clean-machine invocation, output layout, and instructions for returning the complete logs.

Give the user the exact commands but do not execute them. When logs are returned, review them
read-only and verify source/status/validator/artifact hashes before accepting any result. Make only
evidence-backed corrections and require a new timestamped rerun. Never accept a reviewer-provided
manual PASS/FAIL answer as substitute evidence.

## Phase 5 acceptance and release authorization

Do not mark Phase 5 complete until direct evidence proves all applicable Phase 5 and G0-G11 gates,
including:

- exact and consistent v12 identity;
- exact FE/batch/partial-failure provenance;
- policy-free exclusion or one separately manifested newly qualified policy;
- no old policy artifact anywhere in release scope;
- F disabled and excluded from A-E production qualification;
- no CPU fallback in formal CUDA evidence;
- correct GUI/CLI semantic binding and packaged accessibility/responsive behavior;
- clean wheel, sdist, CPU image, and CUDA image reproduction;
- complete immutable manifests, digests, SBOMs, scanner reports, metadata, documentation, and CI;
- protected evidence and leakage boundaries appropriate to the approved scope;
- public claims no broader than retained evidence.

Even after the technical gates pass, do not tag, push, publish, upload, or release until the user
explicitly authorizes those exact actions. If authorization has not been given, report the tree as an
accepted release candidate or release-preparation state—not as a published final release.

When the complete Phase 5 development objective is genuinely achieved and no required development
or evidence work remains, mark the goal complete through the goal service and report final goal
usage. A missing external proof, manual validation directory, scope decision, or release authorization
must remain explicit; never close the goal merely because the budget is low or source exists.
