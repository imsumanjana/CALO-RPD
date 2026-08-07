# Documentation status and precedence

**Current as of 2026-08-07.** This index prevents historical release, audit, training, and validation
records from being mistaken for active v12 instructions. It changes document routing only; it does
not claim that a development, scientific, hardware, container, release-candidate, or release gate
has passed.

## Precedence for current work

When statements differ, use this order:

1. the repository `AGENTS.md` files that apply to the edited path;
2. `docs/implementation/ACTIVE_CONTINUATION_LOG.md` for the newest verified state;
3. `docs/implementation/IMPLEMENTATION_GATES.md` for ordered gate status;
4. `docs/implementation/POST_V6_9_RELEASE_UPDATE_AND_FIX_PLAN.md` for the five-phase v12 plan;
5. `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md` for detailed continuation evidence;
6. `docs/implementation/REQUIREMENT_TRACEABILITY.md` for claim-to-evidence boundaries;
7. current-facing architecture, runbook, reproducibility, and user documentation.

The live worktree and returned source-bound validation evidence remain authoritative over prose. A
Markdown statement alone is never scientific, hardware, container, performance, qualification, or
release proof.

## Current v12 development boundary

- v12 starts at `12.0.0-dev.1`; Phase 4 has not started until its phase-specific goal exists and the
  Phase 3 Linux xcb prerequisite is accepted.
- Phase 4 completes production code, empty-policy safety, integration, CUDA/CPU runtime hardening,
  containers, packages, CI, old-policy inventory/dry-run removal tooling, documentation, and a
  source-bound development freeze.
- Phase 4 performs no policy training, evaluation, qualification, protected-case campaign,
  registration, activation, deletion, RC creation, final manifest freeze, or release.
- Every existing policy is development-only, unqualified, inactive, non-final, excluded from release
  packages, and barred from candidate selection or initialization.
- Executable modes are CUDA-preferred and CPU-only. Intel XPU remains readable as historical metadata
  only and is never executable.
- After Phase 4, an exact old-policy inventory must be reviewed before any separately authorized
  deletion. Empty-store verification precedes either a completely new A-E/F-off policy workflow or
  an explicitly approved policy-free Phase 5 route.
- TSH-CALO A-E remain approved. F remains experimental, independently feature-flagged, evidence-gated,
  and disabled by default.

## Current-facing documents

- `README.md`: active v12 status followed by clearly labeled historical release material.
- `docs/user_guide.md`: current GUI and workflow behavior.
- `docs/architecture.md`: package/data-flow structure, subject to the Phase 4 scheduling note.
- `docs/reproducibility.md`: current empty-policy boundary and future reproducibility contract.
- `docs/CONTAINER_RUNBOOK.md`: current CPU/CUDA container-development contract.
- `docs/algorithm_sources.md`, `docs/mathematical_formulation.md`, and `docs/validation.md`: current
  technical references where unchanged by a newer implementation ledger.
- `docs/implementation/CALO_ARCHITECTURE_CHANGE_PROPOSAL.md`: approved A-E/F architecture with the
  current scheduling addendum.
- `docs/implementation/SCIENTIFIC_VALIDATION_PROTOCOL.md`: frozen future scientific protocol; not a
  Phase 4 execution plan.
- `docs/implementation/PHASE_4_NEW_CHAT_PROMPT.md` and
  `docs/implementation/PHASE_5_NEW_CHAT_PROMPT.md`: self-contained operational handoff prompts. They
  remain subordinate to the live worktree, applicable `AGENTS.md` files, and authoritative ledgers.

## Historical records

The following families are retained and must not be rewritten to resemble current evidence:

- versioned root and `docs/` implementation reports;
- `CHANGELOG.md`, versioned patch notes, architecture-boundary, and API-compatibility records;
- v4-v5.8 policy/training/continuation reports and closure documents;
- dated audits, including the v3.3 deep audit and the 2026-08-03 containerization/scientific audit;
- `RELEASE_VALIDATION.md` and release manifests for v6.9;
- prior release sections in `README.md`;
- retained validation logs, screenshots, generated distributions, and baseline/build artifact copies.

Historical documents may contain XPU execution, old policy training/resume, prior version numbers,
old test counts, or obsolete next-step language. Those statements describe their own recorded scope
only. They do not override current v12 instructions, authorize a workflow, or qualify current source.

## Maintenance rule

After a material milestone, update the active continuation log, the affected gate and traceability
documents, and current-facing user/architecture documentation. Preserve historical evidence. If an
old operational document remains useful but could be mistaken for current instructions, add a brief
historical-status banner and point back to this index.
