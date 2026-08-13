# Documentation status and precedence

**Current as of 2026-08-12.** This index prevents historical release, audit, training, and validation
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

- v12 starts at `12.0.0-dev.1`. Phase 3 was closed on 2026-08-12 by explicit project-owner
  acceptance of the manually validated Linux xcb boundary; no automated Linux evidence directory
  was retained. The required Phase 4 goal was then created before development started.
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
- At the project owner's direction, Phase 4 development closed without a separate validator run and
  Phase 5 release-preparation coding followed. The ignored combined validator runs full Phase 4
  first and starts Phase 5 only if Phase 4 passes. This sequencing decision does not accept either
  phase from source existence.
- Phase 5 development is complete at `12.0.0.dev1`, and the owner-run combined Phase 4/5 validator
  passed for its recorded source identity. An explicit policy-free or newly-qualified-policy scope,
  any old-policy transition, final-record authorization, `12.0.0` promotion, tag, push,
  publication, and release remain pending.
- Phase 6 GUI/native-launch development is complete. Authorized consolidated run
  `phase6-20260813-032036` passed all 19/19 automated unit, GUI, regression, empty-policy, render,
  build, distribution-safety, and native/GUI package-membership checks for its recorded source
  identity. It ran no Docker, policy/scientific, protected-case, publication, or release workflow;
  automated evidence does not infer human screen-reader, usability, or scientist acceptance.
- Phase 6's post-validation visual refinement is also source-complete: the ribbon owns all
  workspace navigation, the left dock is input-only, policy readiness/start uses one ribbon action,
  and all 16 workspace panels plus light/dark/constrained shell states were rendered and visually
  reviewed in `phase6-panel-sweep-20260813-041700`.
- Phase 4 coding now implements separate TSH-CALO experiment selection, empty-policy stale-binding
  cleanup, post-development-only activation/binding provenance, no internal inference fallback,
  read-only old-policy inventory/dry-run export, authorization-bound later retirement, distribution
  exclusions, policy-free physical-CUDA CI, and a complete tracked/non-ignored-source-bound
  development-freeze candidate report with exact interface/policy-empty validation. Proportional
  test source, separate post-review acceptance-receipt tooling, and the ignored
  `validation/Validate-Phase4.ps1`
  harness exist but have not been executed by Codex. Phase 4 remains open pending returned manual
  validation evidence and a development-freeze decision.

## Current-facing documents

- `README.md`: active v12 status followed by clearly labeled historical release material.
- `docs/NATIVE_WINDOWS_GUIDE.md`: current one-time setup, routine native launch, installed-wheel,
  CUDA/CPU, activity/data/shutdown, Docker, and troubleshooting guidance.
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
- `docs/implementation/PHASE_4_NEW_CHAT_PROMPT.md`,
  `docs/implementation/PHASE_5_NEW_CHAT_PROMPT.md`, and
  `docs/implementation/PHASE_6_NEW_CHAT_PROMPT.md`: self-contained operational handoff prompts. They
  remain subordinate to the live worktree, applicable `AGENTS.md` files, and authoritative ledgers.
- `docs/implementation/PHASE_6_EXACT_CONTINUATION_PROMPT.md`: consumed dirty-worktree Phase 6
  continuation with a completion banner; it prevents restarting the original audit and routes the
  owner to the retained passing validator bundle and current ledgers.

The current Phase 6 GUI follow-up restores complete frozen-plan training inputs to the single
input-only dock without re-enabling the historical embedded trainer. Its focused validator has not
yet been owner-executed; the earlier Phase 6 bundle remains historical evidence for its exact source.
Inputs and the expanded ribbon are now permanent shell regions; the central preview is scrollable
with a roomy minimum canvas and a taller branded tab header.
Training cases use a catalog checklist: all legally eligible bundled development cases are selected
automatically, while `case118` and `case300` are displayed as disabled protected holdouts. The
native title-bar icon and separator styling are also corrected in source; the focused validator has
not yet been owner-executed for these additions.
The current applicable training controls have compact accessible hover/focus information buttons with
purpose and directional-effect guidance. Development and acceptance paths are not user inputs,
and the policy-training pane is dedicated to trainable **TSH-CALO**. Policy-free **CALO** remains
an ordinary algorithm/experiment choice and is not shown among inapplicable training inputs;
TSH-CALO uses the existing independent readiness/start workflow. Optional plans are scientific-settings
templates only. This source addition awaits the same focused owner validator.

The latest Phase 6 policy/GUI consolidation removes the dormant embedded training implementation,
routes policy-training resume records only to the independent prefill/check/explicit-start state
machine, keeps one policy import, replaces raw Resume Center JSON with a concise summary, and uses
Activity Logs for technical exception details. Ribbon categories now use exclusive styled buttons
instead of a native category tab painter. This source is unvalidated until the owner runs the
current ignored Phase 6 validator; the earlier 19/19 bundle applies only to its recorded source.

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
