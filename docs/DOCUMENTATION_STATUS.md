# Current documentation map

**Checkpoint date:** 2026-08-21  
**Product identity:** CALO-RPD Studio `12.0.0.dev1` / `12.0.0-dev.1`  
**Status:** development only; not a release candidate or final release

This repository intentionally keeps current operational and technical documentation only. Superseded audit reports, implementation reports, closure records, phase prompts, release manifests, and intermediate plans were removed from the active tree during the 2026-08-21 cleanup. Their exact historical content remains recoverable from Git checkpoint `ba597eb` and earlier history.

## Authority order

1. Applicable `AGENTS.md` instructions.
2. Current source and tests.
3. `ACTIVE_DEVELOPMENT_STATUS.json`.
4. `docs/implementation/ACTIVE_CONTINUATION_LOG.md`.
5. `docs/implementation/IMPLEMENTATION_GATES.md`.
6. `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md`.
7. `docs/implementation/REQUIREMENT_TRACEABILITY.md`.
8. Current technical and user documentation.

A document or old validation result never proves current scientific quality, hardware parity, policy qualification, usability acceptance, release-candidate status, or release readiness.

## Current documents

- `README.md` — setup, repository layout, and safety boundaries.
- `docs/architecture.md` — current architecture and execution flow.
- `docs/user_guide.md` — scientist-facing workflows.
- `docs/NATIVE_WINDOWS_GUIDE.md` — native setup and launch.
- `docs/CONTAINER_RUNBOOK.md` — container usage and boundaries.
- `docs/reproducibility.md` — source, plan, seed, and evidence identity.
- `docs/validation.md` — validation layers and claim limits.
- `docs/calo_methodology.md` and `docs/mathematical_formulation.md` — scientific method references.
- `docs/algorithm_sources.md` and `docs/throughput_engine.md` — implementation references.
- `docs/portfolio_resume.md` — current portfolio/resume behavior.
- `docs/implementation/CALO_ARCHITECTURE_CHANGE_PROPOSAL.md` — approved TSH-CALO A-E/F boundary.
- `docs/implementation/SCIENTIFIC_VALIDATION_PROTOCOL.md` — candidate-bound scientific protocol.

## Current checkpoint boundary

- The active Git checkpoint before this cleanup is `ba597eb` on `agent/ai-repository-intelligence`.
- The newest retained complete engineering bundle is `validation/logs/phase6-20260817-235629`, source-bound to `4560b2fba6ecc5c3271da7dfd680a0985ca501f3`.
- Later source changes mean that bundle is a prior engineering checkpoint, not current-source validation.
- Fresh current-source Phase 6 owner validation remains required.
- Policy training, formal qualification, activation, protected-case execution, publication, and release remain separately authorized workflows.

## Historical recovery

Use Git history when an old report is genuinely needed, for example:

```powershell
git show ba597eb:<historical-path>
```

Do not restore an old report into the active tree merely to infer current status.
