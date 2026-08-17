# Documentation status and precedence

**Current as of 2026-08-17.** This index prevents historical release, audit, training, and validation
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

## Current v12 engineering and policy boundary

- v12 starts at `12.0.0-dev.1`. Phase 3 was closed on 2026-08-12 by explicit project-owner
  acceptance of the manually validated Linux xcb boundary; no automated Linux evidence directory
  was retained. The required Phase 4 goal was then created before development started.
- Phase 4 completes production code, empty-policy safety, integration, CUDA/CPU runtime hardening,
  containers, packages, CI, old-policy inventory/dry-run removal tooling, documentation, and a
  source-bound development freeze.
- Phase 4 performs no policy training, evaluation, qualification, protected-case campaign,
  registration, activation, deletion, RC creation, final manifest freeze, or release.
- Existing policies receive no qualification or release authority from prior results. Their age,
  development phase, and software revision are provenance only; an exact immutable candidate may
  enter fresh formal qualification when it passes the current ABI/ensemble/receipt contract.
- Executable modes are CUDA-preferred and CPU-only. Intel XPU remains readable as historical metadata
  only and is never executable.
- Old-policy deletion remains a separately reviewed retirement workflow and is not a prerequisite
  for inspecting or qualifying another intact compatible policy. A policy-free route remains valid.
- TSH-CALO A-E remain approved. F remains experimental, independently feature-flagged, evidence-gated,
  and disabled by default.
- At the project owner's direction, Phase 4 development closed without a separate validator run and
  Phase 5 release-preparation coding followed. The ignored combined validator runs full Phase 4
  first and starts Phase 5 only if Phase 4 passes. This sequencing decision does not accept either
  phase from source existence.
- Phase 5 development is complete at `12.0.0.dev1`, and the owner-run combined Phase 4/5 validator
  passed for its recorded source identity. An explicit policy-free or qualified-compatible-policy scope,
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
- Historical Phase 4 coding implemented separate TSH-CALO experiment selection, empty-policy stale-
  binding cleanup, no internal inference fallback,
  read-only old-policy inventory/dry-run export, authorization-bound later retirement, distribution
  exclusions, policy-free physical-CUDA CI, and a complete tracked/non-ignored-source-bound
  development-freeze candidate report with exact interface/policy-empty validation. Proportional
  test source, separate post-review acceptance-receipt tooling, and the ignored
  `validation/Validate-Phase4.ps1` harness exist. Its development acceptance identity is engineering
  evidence, not policy compatibility or scientific quality authority.

## Current-facing documents

- `README.md`: active v12 status followed by clearly labeled historical release material.
- `docs/NATIVE_WINDOWS_GUIDE.md`: current one-time setup, routine native launch, installed-wheel,
  CUDA/CPU, activity/data/shutdown, Docker, and troubleshooting guidance.
- `docs/user_guide.md`: current GUI and workflow behavior.
- `docs/architecture.md`: package/data-flow structure, subject to the current stage-neutral policy
  compatibility and qualification boundary.
- `docs/reproducibility.md`: current empty-policy boundary and future reproducibility contract.
- `docs/CONTAINER_RUNBOOK.md`: current CPU/CUDA container-development contract.
- `docs/algorithm_sources.md`, `docs/mathematical_formulation.md`, and `docs/validation.md`: current
  technical references where unchanged by a newer implementation ledger.
- `docs/implementation/CALO_ARCHITECTURE_CHANGE_PROPOSAL.md`: approved A-E/F architecture with the
  current scheduling addendum.
- `docs/implementation/SCIENTIFIC_VALIDATION_PROTOCOL.md`: frozen candidate-bound scientific
  protocol; not a development-phase acceptance mechanism.
- `docs/implementation/PHASE_4_NEW_CHAT_PROMPT.md`,
  `docs/implementation/PHASE_5_NEW_CHAT_PROMPT.md`, and
  `docs/implementation/PHASE_6_NEW_CHAT_PROMPT.md`: self-contained operational handoff prompts. They
  remain subordinate to the live worktree, applicable `AGENTS.md` files, and authoritative ledgers.
- `docs/implementation/PHASE_6_EXACT_CONTINUATION_PROMPT.md`: consumed dirty-worktree Phase 6
  continuation with a completion banner; it prevents restarting the original audit and routes the
  owner to the retained passing validator bundle and current ledgers.
- `WORKSPACE_AND_INDIVIDUAL_EXPERIMENT_EXECUTION_PLAN.txt`: current source-implemented,
  owner-validation-pending contract for explicit algorithm stages, immutable Workspace subsets,
  full-stage individual plans, exclusive execution ownership, safe pause/resume, and terminal
  cancellation. It does not modify or authorize CALO/TSH-CALO architecture, policy lifecycle,
  training, qualification, scientific execution, or release activity.

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
TSH-CALO uses the existing independent readiness/start workflow. The historical optional-template
surface is now superseded: **New training** generates its full plan from visible inputs plus
application-owned safe defaults, while saved campaigns load their retained plan internally for exact
resume/extension. The ordinary UI no longer exposes Add to path, the managed default path, Settings
template, or Import settings. This source addition awaits the same focused owner validator.

The latest Phase 6 policy/GUI consolidation removes the dormant embedded training implementation,
keeps policy-training resume and finite extension inside the independent Train policy
check/explicit-start state machine, keeps one policy import, and uses Activity Logs for technical
exception details. Resume Center is retired from the constructed workspace and command routes; Home
contains only Overview, Open, and Save. Workspace entry now reloads applicable live read-only state,
Results Refresh reloads its experiment selector, and the shared Policy/Saved training refresh invalidates its
file-integrity cache. Ribbon categories use exclusive styled buttons instead of a native category tab
painter. This source is unvalidated until the owner runs the current ignored Phase 6 validator; prior
bundles apply only to their recorded source.

Policy Library qualification is now one explicit `Qualify policy` transaction. It automatically
preflights the candidate, freezes the current non-ignored worktree into a separate clean deterministic
internal Git snapshot without modifying the development checkout, and binds the snapshot commit plus
file/SHA-256 manifest as source identity. It freezes the candidate architecture, parameter/schema,
ensemble, training-design, and exact-model contract, then starts or exactly resumes the fixed 120-cell
paired quality plan and admits only reverified passing evidence. Software/development labels are not
qualification gates. Structured micro-events report every 500 evaluations to the bottom-bar detail,
Activity, and durable `qualification_events.jsonl`, while committed cells remain explicitly separate
from live uncommitted work. The bottom bar exposes cooperative **Pause safely**; a pause is accepted
only after an exact authenticated optimizer checkpoint or cell record is committed and checksum-
verified. The same frozen finite plan can restore the partial cell any number of times without
changing seeds or evaluation accounting. Activity remains enabled while the rest of the workspace is
locked during a run. Separate plan-browser, run/resume, and
admission buttons are obsolete. Qualification never activates or binds: a pass only enables the
separate activation action, and governing-policy Apply remains the separate experiment handoff.
These source/test contracts have not been executed; the current ignored Phase 6 validator remains the
owner command.

The current experiment workflow treats **Experiment > Individual experiment** as a direct,
scientist-owned path: it has separate setup completion, editable repetitions and verified exact
reuse, a direct result contract, and an immutable full-stage v2 plan with no applied Workspace
Portfolio or Study prerequisite. The automated Workspace path is one-way: **Portfolio** persists
only a broad immutable `AppliedPortfolioGoal`; **Workspace > Study** consumes that exact goal,
offers deterministic hard minima/recommendations, retains scientist overrides and their delta, and
creates the Workspace v3 draft only through explicit Apply Study. Audit, Stage, singleton-controller,
and Run remain separate. SQLite schema v3 adds goal/setup records while legacy combined Portfolio
and Workspace v1/v2 records remain historical/readable. The last schema-v27 bundle predates both
current corrections; schema-v29 owner validation is pending.

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
