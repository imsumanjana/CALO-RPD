# CALO-RPD agent instructions

## Scope
These instructions apply to the entire repository unless a deeper `AGENTS.md` narrows them.

## Scientific and release boundaries
- Preserve deterministic baseline behavior and exact function-evaluation accounting.
- TSH-CALO A–E are approved for production-candidate implementation. F is experimental, independently feature-flagged, and disabled by default.
- Do not change scientific semantics outside the approved A–F proposal without explicit user approval.
- Policy training is independent. Experiments may only consume separately qualified, explicitly activated, immutable, checksum-valid policies.
- Never auto-train, auto-modify, auto-qualify, or auto-activate a policy from an experiment workflow.
- Phase 4 is development completion only: do not train, evaluate, qualify, register, activate, or
  delete any policy. Complete production code, empty-policy safety, engineering hardening,
  old-policy inventory/dry-run removal tooling, and a source-bound development freeze.
- Treat every existing policy as development-only, unqualified, inactive, non-final, barred from
  final-policy initialization or evidence, and excluded from release. Deletion requires a reviewed
  exact inventory and separate explicit authorization after the Phase 4 development freeze.
- Any policy used after development must be completely new, trained against the frozen source, and
  independently qualified. Phase 5 may instead use an explicitly approved policy-free scope.
- Preserve explicit safe fallback for unavailable, incompatible, or rejected policies.
- Intel XPU must not be executable. Current modes are CUDA-preferred and CPU-only.
- Admission ceilings use at most 80% of currently free VRAM or currently available RAM. CUDA computes on NVIDIA GPUs; CPU fallback computes on CPUs.
- Do not fabricate hardware, container, performance, energy, thermal, or scientific evidence.
- Do not make release-ready or superiority claims before their documented gates have direct evidence.

## Workflow
- Follow `docs/implementation/IMPLEMENTATION_GATES.md` in order and keep the handoff and traceability ledger current.
- Before stating or starting any numbered development phase, create a new phase-specific goal with
  the goal service. Do not begin source implementation until that goal exists. If another goal is
  unfinished, resolve it according to the goal-service rules before starting the new phase.
- Phase work is coding-only unless the user explicitly authorizes named execution commands in a
  later message. Implement required production code, test source, schemas, documentation, and
  validation harnesses, but do not execute tests or validation that the user can run manually.
- Do not run phase validators, pytest/tox/coverage, compile checks, schema checks, lint/format checks,
  type checks, package/build smoke tests, GUI/browser smoke tests, Docker validation, benchmarks,
  campaigns, policy training/evaluation, qualification, or protected-case workflows as part of
  phase development. Give the user the exact manual command instead.
- End every phase-development pass by creating or updating a detailed PowerShell validator under
  the Git-ignored `validation/` directory. It must exercise the checks required for that phase,
  create a newly timestamped detailed log directory, record command results and relevant source/
  validator hashes, and remain absent from Git source, manifests, distributions, and release
  artifacts. Do not run the validator; ask the user to run it and return the complete log directory.
- Phase validators must be noninteractive. Do not request or gate on reviewer names, roles,
  PASS/FAIL answers, free-form attestations, screen-reader declarations, or other `Read-Host` input.
  Use reproducible automated checks and retained artifacts only. Record explicitly when human
  screen-reader, usability, or scientist acceptance is not inferred by automation.
- Review user-returned validation logs read-only, make evidence-backed coding corrections, update the
  same ignored validator when necessary, and request a fresh user-executed noninteractive rerun.
  Minimize tool calls and output to reduce token usage; avoid redundant scans or repeated evidence
  dumps.
- Before semantic CALO changes, prove canonical-refactor parity against the frozen baseline.
- Add unit, invariant, parity, ablation, falsification, leakage, fallback, and regression tests proportional to each change.
- Keep protected cases out of training, tuning, reward design, and checkpoint selection.
- During Phase 4, use empty-policy behavior and synthetic temporary policy fixtures for development
  tests; do not execute policy workflows or depend on an old policy as a release baseline.
- Do not regenerate release freezes, manifests, SBOMs, image digests, or public release claims before their gates close.
- Preserve user files and unrelated changes. Do not push, merge, publish, or release without explicit approval.
