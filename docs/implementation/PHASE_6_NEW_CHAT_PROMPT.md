# New-chat prompt: Phase 6 ribbon workspace and GUI modernization

> Use this entire file as the first request in a fresh Codex chat. The prompt authorizes Phase 6
> coding only after the live start gate below is satisfied. It does not itself prove that the
> current Phase 4/Phase 5 combined validator has finished or passed, create the Phase 6 goal, start
> Phase 6, or authorize tests, policy work, publication, or release.

## Repository and objective

Work in this repository:

`C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio`

Complete **Phase 6: Ribbon workspace and GUI shell modernization** for CALO-RPD Studio v12.

Replace the current panel-dominant interface with a modern, organized, Microsoft-product-inspired
PyQt6 application shell. Do not copy Microsoft branding, artwork, proprietary icons, or product
identity. The intended interaction model is:

- a tabbed ribbon toolbar at the top with grouped icon commands;
- a short, contextual input pane at the left that changes when a command is selected;
- a large central tabbed workspace for results, plots, tables, previews, comparisons, and reports;
- a persistent activity area for dynamic logs, warnings, job state, and real progress;
- a compact status bar for compute mode, selected device, memory state, policy state, application
  version, and current task;
- responsive, accessible light and dark presentation with no unnecessary long input fields, stacked
  blocks, or unused right-side space.

This is a GUI architecture and coding-development phase. Preserve existing scientific, policy,
execution, persistence, packaging, and result semantics. Reuse existing panels and models through
adapters wherever possible instead of rewriting working internal logic.

Phase 6 must also make native Windows execution a first-class supported path alongside Docker and
replace the misleading disabled legacy-training surface with an enabled entry to the existing
independent new-policy training workflow. Opening the training center may be enabled; navigation
must never start training. Plan/check/start remain explicit, prerequisite-gated user actions, and
the obsolete embedded legacy trainer must not be reactivated.

## Mandatory live start gate

Before creating the goal or editing source:

1. Confirm the exact repository path and inspect the working tree without discarding, overwriting,
   staging, committing, or attributing unrelated/user-owned changes.
2. Read the root `AGENTS.md` completely. Read only the deeper `AGENTS.md` files that govern files
   actually in the planned Phase 6 edit set.
3. Read the following current records in order, using narrowly scoped reads and searches:
   - `ACTIVE_DEVELOPMENT_STATUS.json`;
   - `docs/implementation/ACTIVE_CONTINUATION_LOG.md`;
   - `docs/implementation/IMPLEMENTATION_GATES.md`;
   - `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md`;
   - `docs/implementation/REQUIREMENT_TRACEABILITY.md`;
   - the latest applicable ignored combined-validation instructions and returned summaries under
     `validation/`, when available.
4. Verify that no Phase 4/Phase 5 combined validator or Docker build is currently running. Do not
   edit source while an evidence-producing validator is active.
5. Verify and report the latest evidence-backed Phase 4/Phase 5 state. Do not call Phase 5 or the
   final release complete merely because images or source files exist. Any unresolved Phase 5 or
   release gate must remain explicit, but a release-publication decision must not be silently pulled
   into this GUI-only phase.
6. If an active validator is still running, or if a current development goal is unfinished, do not
   start Phase 6. Report the exact state and stop or resolve the goal strictly under the goal-service
   rules.
7. If the start gate is satisfied, create a new Phase 6 goal through the goal service **before
   stating or starting Phase 6 development**. Use this objective:

   `Complete CALO-RPD Studio v12 Phase 6 ribbon-workspace GUI modernization with one reusable command registry, a tabbed icon ribbon, contextual compact left input editors, a central results and preview workspace, dynamic job/log/progress presentation, responsive accessible themes, compatibility-preserving panel adapters, proportional test source and documentation, and one Git-ignored noninteractive manual validator, without changing scientific or policy semantics or executing manual-capable validation.`

8. Do not set a goal token budget unless the user explicitly requests an exact goal-service token
   budget in the new chat.

## Non-negotiable scientific, policy, and release boundaries

- Preserve deterministic scientific behavior and exact function-evaluation/scenario accounting.
- Preserve the approved boundary: TSH-CALO A-E are production-candidate changes; F remains
  experimental, independently feature-flagged, evidence-gated, and disabled by default.
- Do not change algorithm equations, optimizer behavior, convergence semantics, protected-case
  handling, qualification logic, policy ABI, evidence meaning, or fallback rules for visual
  convenience.
- Do not train, retrain, tune, evaluate, qualify, register, activate, promote, modify, or delete any
  policy. Existing policies remain development-only, unqualified, inactive, non-final, and excluded
  from release unless a later separately evidenced decision says otherwise.
- Never make an experiment workflow auto-train, auto-select, auto-qualify, or auto-activate a policy.
- F must be visibly identified as experimental wherever exposed and remain off by default.
- Intel XPU remains non-executable. Supported execution modes remain CUDA-preferred and CPU-only.
- CUDA must mean NVIDIA GPU execution. CPU fallback must remain explicit and provenance-recorded.
  Admission remains bounded by at most 80% of currently free VRAM or currently available RAM.
- Do not fabricate scientific, policy, GPU, performance, progress, timing, thermal, energy,
  accessibility, container, release, or human-acceptance evidence.
- Do not tag, push, publish, release, delete user data, stage, or commit unless the user separately
  authorizes the exact action.
- Do not use subagents unless the user explicitly requests delegation.

## Coding-only execution boundary

- Implement production GUI code, adapters, state models, event plumbing, test source, documentation,
  and the ignored validation harness.
- Do not run pytest, tox, coverage, Ruff, formatting, lint, mypy, compile/schema checks, GUI smoke
  checks, screenshot/render checks, accessibility automation, package builds, installed-distribution
  checks, Docker/Compose/Buildx, CUDA jobs, benchmarks, campaigns, or phase validators. The user will
  run the final Phase 6 validator manually.
- Do not launch the GUI merely to inspect it. Build reproducible GUI inspection into the final
  validator for the user.
- Source-formatting tools may be described in the validator, but the development agent must not run
  them when the user can run them manually.
- Use `apply_patch` for edits. Preserve unrelated and user-owned work.

## Required target layout

Implement a cohesive `QMainWindow`-based shell with these primary regions:

```text
Quick access / project identity / current device and run state
Tabbed ribbon: Home | Experiment | Algorithms | Compute | Results | Policies | View | Help
Ribbon command groups with icons, concise labels, tooltips, shortcuts, and accessible names
Contextual left input dock | Central document-style result/preview workspace
Bottom activity dock: Jobs | Logs | Warnings | Device | Provenance
Status bar: task | progress | compute mode | selected device | memory | policy state | version
```

The implementation may refine exact labels after mapping existing capabilities, but it must preserve
the architecture and interaction model.

### Ribbon requirements

- Use one central command/action registry as the authoritative definition of command ID, category,
  group, label, icon, tooltip, shortcut, accessibility name, enabled state, handler, and associated
  contextual editor/workspace target.
- Do not wire every toolbar button independently to panel-specific callbacks.
- Generate ribbon categories and grouped command buttons from registry metadata.
- Support large primary and compact secondary `QToolButton` presentation without duplicating
  actions.
- Use application-owned, redistribution-safe SVG assets or appropriate Qt-standard icons. Do not
  depend on proprietary Microsoft assets or require network icon downloads at runtime.
- Provide visible disabled reasons for gated commands rather than silently doing nothing.
- Implement a compact/collapsed ribbon mode and persist the user's choice.
- Ribbon navigation must work by keyboard and expose stable accessible names.

Suggested categories:

- **Home:** new/open/save/duplicate project or experiment, validate, run, stop, recent work;
- **Experiment:** case, budget, seed, comparison study, benchmark campaign, reproducibility and
  provenance preview;
- **Algorithms:** CALO, TSH-CALO, approved baselines, parameters, advanced configuration, feature
  flags and documentation;
- **Compute:** CUDA-preferred, CPU-only, GPU selection, admission and resource/device diagnostics;
- **Results:** explorer, tables, plots, comparisons, report preview, export and provenance;
- **Policies:** status/inventory/qualification/checksum/history, an enabled **New policy training**
  command that opens the independent training center without starting work, and separately gated
  lifecycle actions without implicit policy work;
- **View:** pane visibility, compact ribbon, theme, UI scale, full-screen result and reset layout;
- **Help:** user guide, container guide, shortcuts, diagnostics, version and about.

### Contextual left input pane

- Implement one reusable left `QDockWidget` containing a `QStackedWidget` or equivalent contextual
  editor host.
- Clicking a ribbon command must focus or open the correct compact editor page.
- Keep normal inputs short and proportionate. Do not stretch line edits, spin boxes, combo boxes, or
  path selectors across unused width.
- Use reusable form rows, validation messages, field help, section headers, and collapsible advanced
  sections.
- Preserve values when users switch commands or workspace tabs.
- Reuse the application's existing configuration/state authority. Do not create a second competing
  state model merely for the new GUI.
- Allow the pane to resize, collapse, float, restore, and return to a safe default width.
- Put primary Validate/Run or Apply actions consistently at the bottom when relevant.

### Central workspace

- Implement a document-style central tab workspace for overview, live results, tables, plot preview,
  network preview, comparison, report preview, policy details, and run provenance.
- Provide stable workspace/document IDs so commands focus an existing singleton tab where
  appropriate instead of opening duplicates.
- Allow suitable tabs to close, reorder, pin, restore, and expand.
- Preserve results when the contextual editor changes.
- Adapt existing result and visualization widgets before replacing their internal implementations.
- Replace layouts that leave large unused horizontal areas with sensible side-by-side content,
  nested tabs, splitters, or responsive grids.
- Do not turn every existing panel into a top-level tab if a smaller reusable view or adapter is
  sufficient.

### Jobs, dynamic logs, and progress

- Implement a reusable application activity model rather than letting individual panels invent
  independent progress and log behavior.
- Show queued/running/completed/failed/cancelled state, current stage, elapsed time, relevant counters,
  execution mode, selected device, result location, and cancellation state.
- Use determinate progress only when the runtime exposes a truthful total such as cases, candidates,
  generations, scenarios, evaluations, or known workflow stages.
- Use indeterminate progress when no defensible total exists. Never fabricate percentages or time
  remaining.
- Preserve exact function-evaluation and scenario accounting; GUI progress is presentation, not a
  replacement accounting authority.
- Provide searchable/filterable logs with job and severity filters, pause autoscroll, copy, retained
  log location, and clear-display behavior that does not delete durable evidence.
- Keep warnings and failures clearly distinguishable by icon/text as well as colour.
- Ensure GUI updates are thread-safe and do not block the Qt event loop.
- Connect existing workers/signals through adapters. Do not rewrite algorithms to emit UI-specific
  behavior when a generic activity adapter can translate existing lifecycle events.

### Status bar and resource presentation

- Show concise current task, progress, compute mode, selected device, live resource state, policy
  status, and version.
- Make the distinction between CUDA-preferred, actual CUDA execution, CPU-only, and explicit fallback
  visible and truthful.
- Resource indicators must use existing authoritative telemetry/admission services where available.
  They must not imply that configured limits are actual consumption.
- Policy status must never imply qualified/active/final from file presence alone.

### Independent new-policy training center

- Replace the blue-but-disabled `Legacy training unavailable` primary action and the clipped
  multi-button legacy row. Do not merely re-enable its obsolete handler.
- Provide an enabled ribbon/command-registry entry that opens a compact contextual editor plus a
  central training-plan/activity workspace for an immutable compatible policy candidate.
- Adapt the existing independent `calo-rpd-train-tsh` plan/check/start workflow. Keep plan, readiness
  check and explicit start as separate user actions with visible prerequisites and generated command
  or job identity.
- Navigation and editor selection must never start, resume, retrain, evaluate, qualify, register,
  activate, or delete a policy.
- Start must remain fail-closed until the independent workflow's own source, configuration, device,
  empty/approved store, output, and lifecycle prerequisites pass. Do not weaken those checks merely
  to show an enabled module.
- Show existing sessions and recovery only when compatible records exist. Do not select an old
  policy or treat it as an initializer.
- Fix light and dark theme selector precedence so every disabled primary command is visibly disabled.
- Replace long one-row action groups with responsive contextual actions, overflow, or stacked/grid
  presentation that remains usable at constrained widths.

### Native Windows execution without Docker

- Preserve Docker CPU/CUDA operation, but do not require Docker to launch the ordinary desktop GUI.
- Provide one authoritative native Windows launch path from an installed wheel/virtual environment
  and one repository-development bootstrap path. Reuse the same application entry point and state
  authority; do not create a divergent native application.
- Implement a concise PowerShell launcher or equivalent packaged command that discovers/uses the
  project virtual environment safely, reports missing prerequisites clearly, and launches without
  policy side effects. Do not silently install or upgrade large dependencies on every ordinary
  launch.
- Keep dependency setup separate from routine startup. Document first setup, normal launch, CUDA
  verification, CPU-only selection, logs, shutdown, data locations, and troubleshooting.
- Native CUDA must use the installed CUDA-capable PyTorch and the selected NVIDIA device. Native
  CPU-only mode must remain available. Intel XPU remains non-executable.
- Add package/entry-point and clean-install regression source so wheel-based native startup does not
  import from the checkout accidentally.

### Design system, responsiveness, and accessibility

- Create reusable design tokens for colour, spacing, typography, radii, icon sizes, control heights,
  borders, focus states, and semantic states.
- Support coherent light and dark themes without panel-specific stylesheet fragments becoming the
  primary styling mechanism.
- Use high-DPI-safe icons and layouts.
- Define compact, normal, and constrained-width behavior. Do not rely on one workstation resolution.
- Ensure minimum sizes do not force unnecessary horizontal whitespace or clip essential actions.
- Preserve keyboard operation and visible focus. Include `F6`-style region cycling, workspace tab
  shortcuts, ribbon category navigation, and safe command shortcuts where appropriate.
- Every icon-only command must have a tooltip and accessible name. Colour cannot be the only state
  indicator.
- Automation must not claim human screen-reader, usability, or scientist acceptance.

### Layout persistence and safe recovery

- Persist ribbon compact state, theme, dock visibility/placement, splitter sizes, selected workspace
  tab, and appropriate recent workspace state through the existing application settings authority.
- Version the layout state so incompatible older layouts fail safely to a known default.
- Provide **Reset layout** and safe recovery from corrupt or incomplete layout state.
- Do not persist transient secrets, policy payloads, untrusted logs, or unsafe object serialization.

## Compatibility-preserving migration strategy

1. Audit only relevant GUI shell, panel, worker/event, application-state, settings, and result-view
   files. Produce a concise command-to-panel/state/event mapping before editing.
2. Introduce the reusable shell, command registry, contextual editor host, workspace host, activity
   model, and design system as small coherent modules.
3. Adapt existing panels incrementally. Extract reusable editor/view portions only where necessary;
   otherwise host existing widgets behind compatibility adapters.
4. Preserve stable configuration keys, signals, database behavior, entry points, project loading,
   result semantics, worker lifecycle, cancellation, and saved state.
5. Avoid simultaneous redesign of scientific/runtime internals. When an interface adapter can bridge
   an existing API, use it.
6. Remove obsolete GUI routing only after all consumers are migrated in source and covered by
   proportional regression tests.
7. Keep the application usable during migration through one authoritative shell; do not retain two
   divergent production navigation systems indefinitely.

## Token-efficient working method: keep usage near the lower end

The primary motive is accurate development with low token usage. Follow all of these rules:

- Start in this fresh chat with this precise Phase 6 prompt.
- Inspect only relevant GUI and state-management files, plus the narrow worker/result/settings
  interfaces needed for integration.
- Use one central command registry instead of wiring every button independently.
- Build reusable form, ribbon, workspace, activity, and status components.
- Adapt existing panels instead of rewriting their internal logic.
- Avoid repeated broad repository audits, repeated full-tree listings, and repeated rereads of the
  same ledgers or source.
- Do not execute any manual-capable validation yourself.
- Write the complete Phase 6 validator once, at the end of the development pass, after the production
  architecture and test source stabilize.
- Let the user's local validator produce one consolidated timestamped result.
- When the user returns logs, make only evidence-backed corrections. Do not repeatedly redesign
  speculative details or treat subjective preference as an automated failure.
- Keep tool output narrow. Avoid dumping large logs, complete diffs, or entire large files when a
  targeted search/range is sufficient.
- Preserve current scientific and policy code unless a narrowly scoped interface adapter is strictly
  required.
- Prefer one cohesive implementation pass and one consolidated manual validation over many tiny
  validation cycles.
- Keep commentary concise and report decisions, changed architecture, blockers, and exact manual
  next actions without repeating prior evidence.
- Do not spend tokens estimating work repeatedly after development starts. Implement against this
  defined scope and expose genuine blockers.

## Proportional test source to implement but not execute

Add or update test source for at least:

- command registry uniqueness, category/group ordering, command-to-editor/workspace mappings, enabled
  state and disabled reason;
- ribbon construction, compact mode, keyboard focus, accessible names and shortcut uniqueness;
- contextual editor switching, value preservation, validation routing and bounded field sizing;
- workspace singleton/focus rules, tab close/reorder/restore and result preservation;
- activity-model state transitions, truthful determinate/indeterminate progress, cancellation and
  thread-safe signal delivery;
- log filtering/search/autoscroll state without durable-log deletion;
- CPU/CUDA/fallback/policy status presentation against authoritative runtime state;
- light/dark theme application, design-token use, high-DPI behavior and focus visibility;
- layout persistence, schema/version migration, corrupt-state recovery and reset layout;
- representative constrained, normal and wide window layouts without clipped controls, overlapping
  widgets, excessive input growth, or avoidable unused space;
- compatibility of stable project/configuration keys, GUI entry points, existing panel actions and
  result views;
- packaged GUI/resource inclusion and absence of validation/log content from distributions.

Do not assert pixel-perfect equality across platforms when semantic layout assertions are more
stable. Use deterministic screenshot artifacts for manual inspection without treating screenshots
alone as accessibility or usability proof.

## Documentation and authoritative records

- Document the new application shell, command categories, workspace behavior, shortcuts, layout
  persistence, themes, job/log/progress behavior, CPU/CUDA status interpretation, and known limits.
- Update relevant user-facing guidance and architecture documentation.
- After material work, update these authoritative records together without claiming unexecuted proof:
  1. `docs/implementation/ACTIVE_CONTINUATION_LOG.md`;
  2. `docs/implementation/IMPLEMENTATION_GATES.md`;
  3. `docs/implementation/REQUIREMENT_TRACEABILITY.md`;
  4. `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md`.
- Update `ACTIVE_DEVELOPMENT_STATUS.json` truthfully for Phase 6 coding/manual-validation state and
  keep its verifier/test contracts synchronized without converting source existence into acceptance.
- Keep implementation, automated local evidence, human usability/accessibility review, scientific
  evidence, and release authorization as distinct claim layers.

## One mandatory Phase 6 manual validator at the end

At the end of the complete Phase 6 coding pass, create or update one detailed, noninteractive
PowerShell validator under the Git-ignored repository-root `validation/` directory. Do not run it.
Do not create repeated partial validators during implementation unless the final architecture makes
the existing draft technically invalid.

The validator must:

- verify that it and all produced logs remain Git-ignored and excluded from source distributions,
  wheels, containers, manifests, SBOM inputs, and release artifacts;
- create a new timestamped log directory and never overwrite previous evidence;
- record repository path, full source commit, tracked dirty/clean state, source-status hash, validator
  SHA-256, relevant source/resource hashes, environment identity, commands, start/end timestamps,
  durations, exit codes, stdout/stderr, and a machine-readable plus Markdown summary;
- fail fast at the first failed command while retaining the partial summary and complete hashes;
- run the applicable version, compile/schema, Ruff/format, mypy, targeted and regression test,
  packaged-resource, wheel/sdist, clean-install, GUI startup/shutdown, and layout suites;
- exercise representative Windows window sizes and DPI/scaling configurations supported by the
  automation, plus Docker/noVNC rendering when it is an applicable retained requirement;
- retain deterministic screenshots for ribbon categories, compact ribbon, contextual editors,
  central workspace tabs, activity/log/progress states, constrained layouts, and light/dark themes;
- verify widget geometry, minimum/size-policy behavior, clipping/overlap, bounded input width, focus
  traversal, accessible names, shortcuts, theme tokens, layout persistence and corrupt-state reset;
- use synthetic jobs/events and safe temporary data for progress/log/state tests; never perform policy
  training/evaluation or protected scientific campaigns;
- record explicitly that automation does not infer human screen-reader, general usability, visual
  preference, or scientist acceptance;
- never use `Read-Host` or ask for reviewer names, roles, PASS/FAIL answers, free-form attestations,
  screen-reader declarations, or manual answers;
- perform no policy training, evaluation, qualification, registration, activation, modification, or
  deletion and no release, publication, push, or tag action.

Create or update a Git-ignored Markdown instruction file beside the validator containing:

- prerequisites;
- exact PowerShell command;
- expected duration and disk requirements;
- expected output directories and important artifacts;
- troubleshooting for GUI/DPI/Docker prerequisites;
- instructions to return the complete timestamped log directory, not only terminal excerpts.

Give the user the exact command but do not execute it. When the user returns the consolidated log,
review it read-only, locate the first evidence-backed failure, make the smallest correct source or
harness correction, update the same ignored validator if needed, and request one fresh timestamped
rerun. Do not accept a manual reviewer answer as substitute validation evidence.

## Phase 6 completion criteria

Do not mark the Phase 6 goal complete merely because the new shell exists. Completion requires:

- one authoritative command registry generates the ribbon and command actions;
- the required ribbon, contextual left input pane, central result/preview workspace, activity area,
  and status bar are integrated in the production GUI;
- important existing panels and actions are accessible through the new shell without divergent
  duplicate routing;
- unnecessary long inputs, stacked-block sprawl, and avoidable unused right-side space are resolved
  across representative layouts;
- real activity/log/progress state is presented truthfully without altering FE/scenario accounting;
- CPU/CUDA/fallback and policy state are presented accurately;
- layout/theme/accessibility behavior and safe persistence are implemented;
- relevant documentation, test source, version/status contracts, and authoritative ledgers are
  current;
- the complete Git-ignored noninteractive validator and instructions are ready for the user;
- returned current-source manual validation passes all automated Phase 6 gates, with human
  accessibility/usability/scientist acceptance still stated separately and not inferred;
- no scientific or policy semantic regression, unauthorized release action, or validation artifact
  packaging occurred.

After current-source returned validation is reviewed and every required development item is genuinely
complete, mark the Phase 6 goal complete through the goal service and report the final goal usage.
If external evidence or user action remains required, keep that boundary explicit rather than closing
the goal to save tokens.
