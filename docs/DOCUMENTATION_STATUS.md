# Documentation status and precedence

**Current as of 2026-08-21.** This index separates active v12 guidance from immutable historical release, audit, training, and validation evidence. Documentation does not itself prove a scientific, hardware, qualification, release-candidate, or release gate.

## Precedence for current work

When statements differ, use this order:

1. applicable repository `AGENTS.md` files;
2. the live worktree and source-bound returned validation evidence;
3. `docs/implementation/ACTIVE_CONTINUATION_LOG.md` for the newest verified development state;
4. `docs/implementation/IMPLEMENTATION_GATES.md` for ordered gate status;
5. `docs/implementation/POST_V6_9_RELEASE_UPDATE_AND_FIX_PLAN.md` for the v12 plan;
6. `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md` for retained continuation evidence;
7. `docs/implementation/REQUIREMENT_TRACEABILITY.md` for claim-to-evidence boundaries;
8. current-facing architecture, runbook, reproducibility, and user documentation.

A Markdown statement alone is never scientific, hardware, container, performance, qualification, or release proof.

## Current v12 boundary

- The active identity is `12.0.0-dev.1` / `12.0.0.dev1` and remains development-only.
- The tree is not a release candidate or final release; final freeze, qualification, publication, tagging, and release remain separately authorized gates.
- Historical v6.x evidence does not qualify the active v12 source.
- Executable compute modes are CUDA-preferred and CPU-only. Intel XPU may appear in historical metadata but is not executable in v12.
- TSH-CALO A-E remain the production architecture surface. F remains experimental, independently feature-gated, evidence-gated, and disabled by default.
- Policy-free CALO remains valid. TSH-CALO training, qualification, activation, and experiment binding are separate explicit lifecycle stages.
- Existing checked-in model records are historical evidence. Their age or software revision alone does not grant or revoke current qualification.
- Protected case118/case300 work remains isolated from ordinary development and policy training.
- Deterministic scientific semantics, exact function-evaluation accounting, immutable plan identities, seeds, resume/extension compatibility, and provenance boundaries remain protected.

## Current-facing documents

- `README.md` — concise active v12 overview, entry points, boundaries, and documentation links.
- `CHANGELOG.md` — consolidated release history.
- `docs/NATIVE_WINDOWS_GUIDE.md` — current native setup/launch, CUDA/CPU, data, shutdown, Docker, and troubleshooting guidance.
- `docs/user_guide.md` — current GUI/workflow behavior.
- `docs/architecture.md` — current package/data-flow architecture.
- `docs/reproducibility.md` — reproducibility and evidence boundaries.
- `docs/CONTAINER_RUNBOOK.md` — current CPU/CUDA container contract.
- `docs/algorithm_sources.md`, `docs/mathematical_formulation.md`, and `docs/validation.md` — current technical references where not superseded by newer authoritative source or ledgers.
- `docs/implementation/CALO_ARCHITECTURE_CHANGE_PROPOSAL.md` — approved A-E/F architecture record with later addenda.
- `docs/implementation/SCIENTIFIC_VALIDATION_PROTOCOL.md` — frozen candidate-bound scientific protocol, not a development acceptance mechanism.

The phase 4/5/6 new-chat and exact-continuation prompt files under `docs/implementation/` are retained only as historical handoff evidence. They are not product documentation, are not packaged into distributions, and are subordinate to current source, applicable `AGENTS.md`, and current ledgers.

## Current implementation plans still awaiting owner validation

The following root plans remain active implementation records and should not be cleaned as obsolete until their corresponding current source has completed the required owner validation and their durable requirements have been folded into normal documentation:

- `WORKSPACE_AND_INDIVIDUAL_EXPERIMENT_EXECUTION_PLAN.txt`
- `PORTFOLIO_TO_STUDY_ONE_WAY_WORKFLOW_PLAN.txt`
- `INDIVIDUAL_EXPERIMENT_STEP_BY_STEP_PANEL_PLAN.txt`

Their existence does not authorize experiments, policy workflows, protected-case work, publication, or release.

## Historical records

Retain versioned release/audit/implementation reports, findings-closure CSVs, frozen manifests, qualification status records, release validation records, dated audit evidence, and checked-in historical model records as immutable provenance for their named scope. Historical documents may contain old version numbers, obsolete workflow language, prior test counts, or XPU-era execution descriptions. Those statements do not override current v12 guidance.

`STATUS_RECORD_INDEX.json` distinguishes the active development status from historical status records. Historical v6.9 integrity tests and manifests intentionally describe their own release checkout and are not current-v12 qualification gates.

## Generated and local evidence

Generated publication exports, local validation transport/remediation scripts, policy-retirement inventories/plans, caches, build outputs, SQLite databases, and runtime model/checkpoint outputs are not durable source authority and should remain outside tracked core source unless a specific evidence policy explicitly says otherwise.

## Maintenance rule

After a material milestone, update the active continuation/gate/traceability records and any affected current-facing documentation. Preserve historical evidence rather than rewriting it to look current. Once a temporary plan or handoff is superseded and no longer part of current routing, move it out of active guidance or retain it only as clearly marked historical evidence.
