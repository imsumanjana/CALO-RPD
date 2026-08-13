# Exact new-chat continuation prompt: Phase 6 GUI modernization

> Copy this entire file into a new Codex chat. It is a continuation handoff for the current dirty
> worktree, not a request to restart Phase 6, repeat a broad audit, discard changes, or execute
> validation. The live repository is authoritative; reconcile it narrowly before editing.

> **Consumed 2026-08-13:** this continuation was completed in the live worktree. Do not use the
> remaining-work section to restart development or repeat its static review. Current state is
> `phase_6_coding = implemented_and_phase6_validation_passed`. Authorized consolidated bundle
> `phase6-20260813-032036` passed 19/19 automated GUI/native/packaging checks. Read the current
> ledgers; do not rerun this prompt's audit or validator unless later source changes require it.

## Repository

Work only in:

`C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio`

The current branch is `main`. The clean Phase 6 baseline commit was:

`2d7130fb63c13d35d0419dd63b1d68e2050dcf72`

The worktree is intentionally dirty with in-progress Phase 6 implementation. Preserve all current
Phase 6 changes. Do not reset, checkout, clean, restore, stage, commit, push, merge, publish, or
release anything.

Two tracked SQLite runtime sidecars currently appear deleted:

- `calo_rpd_results.sqlite-shm`
- `calo_rpd_results.sqlite-wal`

They disappeared during the project owner's native application/bootstrap attempt. Treat them as
user/runtime-owned transient state. Do not restore, recreate, delete, stage, or otherwise touch
them.

## Goal-service start rule for the new chat

This continuation came from an active Phase 6 goal in another chat/account. Goal state may not
transfer between accounts. Before changing source:

1. Inspect the goal service.
2. If the exact Phase 6 goal already exists and is active, continue it without creating a duplicate.
3. If no transferred goal exists, create a Phase 6 continuation goal before stating or resuming
   numbered-phase development, using this exact objective:

   `Complete CALO-RPD Studio v12 Phase 6 GUI modernization by implementing the approved Microsoft-product-inspired tabbed ribbon workspace, one central command registry, compact contextual left input editors, central results and preview tabs, dynamic jobs/logs/truthful progress and status presentation; replace the misleading disabled legacy-training surface with an enabled, safely gated interface to the independent new-policy training workflow without auto-training, auto-qualification, or policy-semantic changes; provide first-class native Windows execution and launch guidance in addition to Docker operation; preserve all scientific, policy, CUDA/CPU, persistence, packaging, and release boundaries; update authoritative continuation and traceability records; add proportional test source; and finish with one Git-ignored noninteractive Phase 6 manual validator that the user runs locally, without the agent executing manual-capable tests or policy training/evaluation.`

Do not set a token budget. Keep the goal active until the source/deliverables are complete and the
project owner's manual evidence has been reviewed as required by the goal-service completion rules.

## Mandatory operating boundary

- Read the root `AGENTS.md` completely and only the deeper `AGENTS.md` files governing files that
  still require edits. Do not repeat the earlier repository-wide audit.
- Use the current worktree and current continuation ledgers as authority.
- Use `apply_patch` for edits.
- This is coding-only development. Do not run pytest, compileall, Ruff, formatting, mypy, schema
  checks, GUI rendering, launch/smoke checks, package builds, Docker, CUDA work, benchmarks,
  campaigns, or phase validators. The project owner will run the final validator.
- Do not run, simulate, or invoke policy training, retraining, tuning, evaluation, qualification,
  registration, activation, promotion, deletion, protected cases, or release operations.
- Do not start the GUI merely to inspect it.
- Do not use policy artifacts as a final baseline or initializer.
- Do not use subagents unless the project owner explicitly requests delegation.
- Do not request manual reviewer names, roles, PASS/FAIL answers, screen-reader declarations, or
  any `Read-Host` input. The validator must be noninteractive.
- Keep `validation/` and all generated logs Git-ignored and absent from source distributions,
  manifests, containers, and release artifacts.
- Minimize tool output. Use narrow reads/diffs and address evidence-backed issues. Do not repeatedly
  print large files or repeat broad audits.

## Evidence already accepted before Phase 6

Do not rerun or re-audit Phase 4/Phase 5 merely to start this continuation.

- Combined owner run: `validation/logs/phase4-phase5-20260813-000340`
- Phase 4 child: `phase4-20260813-000340`, 32/32 passed
- Phase 5 child: `phase5-20260813-010531`, 41/41 passed
- Both children recorded source commit
  `f800119cd3a14e2965c91040d0a8392013532089` plus the same retained dirty source identity.
- This is combined development/release-preparation evidence only. It does not choose a release
  policy scope, authorize policy work, establish a clean final source, create an RC/final release,
  authorize publication, or release v12.

Phase 6 began only after that combined pass and after the original Phase 6 goal was created. The
authoritative start checkpoint is already present in:

- `ACTIVE_DEVELOPMENT_STATUS.json`
- `docs/implementation/ACTIVE_CONTINUATION_LOG.md`
- `docs/implementation/IMPLEMENTATION_GATES.md`
- `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md`
- `docs/implementation/REQUIREMENT_TRACEABILITY.md`
- `docs/implementation/PHASE_6_NEW_CHAT_PROMPT.md`

## Scientific, policy, compute, and release invariants

- Preserve deterministic scientific behavior and exact function-evaluation/scenario accounting.
- Preserve TSH-CALO A-E as production-candidate changes. F remains experimental, separately
  feature-flagged, evidence-gated, and off by default.
- Do not change optimizer equations, convergence meaning, protected-case rules, policy ABI,
  evidence meaning, lifecycle authority, or fallback semantics for GUI convenience.
- Experiments never train, resume, retrain, qualify, register, activate, or delete policies.
- Opening or navigating to the training center never starts any process.
- A training start remains an explicit, confirmed action after a fresh successful readiness check
  from the existing independent `calo-rpd-train-tsh` contract.
- New training output remains an unqualified, inactive candidate. The GUI must not register,
  qualify, activate, promote, or select it automatically.
- The obsolete embedded legacy trainer remains disabled and hidden; do not re-enable its handler.
- Zero policies is a supported state. File presence alone never means ready, qualified, active, or
  final.
- Current execution modes remain CUDA-preferred and CPU-only. Intel XPU is non-executable.
- Configured CUDA intent, actual CUDA assignment, CPU-only execution, and explicit fallback must be
  visibly distinct.
- Safe-80 values are admission ceilings based on currently free VRAM/available RAM, not measured
  usage or forced utilization.
- Do not fabricate progress, remaining time, device, memory, performance, accessibility, human
  acceptance, scientific, policy, container, or release evidence.

## Phase 6 implementation already present in the worktree

Do not recreate these modules from scratch. Inspect and refine them narrowly.

### Central commands and ribbon

- `calo_rpd_studio/gui/command_registry.py`
  - immutable `CommandSpec` metadata;
  - one central registry and generated `QAction` objects;
  - eight categories: Home, Experiment, Algorithms, Compute, Results, Policies, View, Help;
  - stable command IDs, labels, icon names, tooltips, shortcuts, handler kinds, workspace targets,
    context kinds, and availability reasons.
- `calo_rpd_studio/gui/widgets/ribbon_bar.py`
  - registry-generated grouped icon buttons;
  - primary/secondary presentation;
  - accessible names/descriptions;
  - compact/collapsed mode and live state summary.
- `calo_rpd_studio/gui/icons/workspace_icons.py`
  - expanded application-owned SVG path set; no Microsoft/proprietary assets.

### Main shell, context, documents, activity, and status

- `calo_rpd_studio/app/main_window.py`
  - existing 16-workspace `QStackedWidget` retained for compatibility;
  - pinned scientific workspace inside the new central `DocumentWorkspace`;
  - registry/ribbon integration;
  - dockable left contextual inputs and bottom activity center;
  - independent-training and operating-guide documents;
  - workflow availability reasons;
  - F6 region cycling;
  - versioned QSettings layout persistence and Reset layout;
  - last-workspace restoration;
  - light/dark command;
  - training navigation integration.
- `calo_rpd_studio/gui/widgets/context_pane.py`
  - Inputs/Navigator tabs;
  - compact bounded-width generic, experiment, compute, and training-path editors;
  - experiment/compute edits validate a copied complete configuration before replacing shared
    state, avoiding a second scientific state authority.
- `calo_rpd_studio/gui/widgets/document_workspace.py`
  - pinned scientific workspace plus singleton closeable document tabs.
- `calo_rpd_studio/gui/widgets/activity_center.py`
  - Jobs, searchable/severity-filtered Logs, Warnings, Device, and Provenance tabs;
  - TaskStatus adapter with indeterminate progress when no truthful total exists;
  - Qt-signal logging bridge and clear-display behavior that does not delete durable evidence.
- `calo_rpd_studio/gui/widgets/global_status_bar.py`
  - task/progress/cancel plus configured mode, actual device, Safe-80 ceiling, policy state, and
    application version.
- `calo_rpd_studio/gui/themes/light.py`, `dark.py`, and `tokens.py`
  - ribbon/dock/document/activity styles and expanded layout metrics;
  - later, more-specific `QPushButton#PrimaryButton:disabled` selectors that fix the screenshot's
    blue-but-disabled primary-button defect in both themes.

### Independent new-policy interface

- `calo_rpd_studio/gui/panels/independent_training_panel.py`
  - shared `TrainingLaunchModel` for plan, development-freeze, Phase 4 acceptance, and output paths;
  - separate command previews for check and explicit start;
  - readiness invalidation whenever bound path values change;
  - start disabled until readiness passes and required output exists;
  - explicit resume checkbox, existing-output protection, foreground-task exclusion, confirmation,
    merged QProcess output, activity integration, and unqualified/inactive result wording;
  - it invokes the existing `calo_rpd_studio.scripts.train_tsh_calo` CLI and does not weaken its
    source/freeze/acceptance/clean-tree checks.
- `calo_rpd_studio/gui/panels/calo_intelligence_panel.py`
  - old `Legacy training unavailable` compatibility button remains disabled and is now hidden;
  - visible primary action is `Open new policy training`;
  - clipped legacy continuation/recovery row is replaced by a two-column new-training/import grid;
  - an `independent_training_requested` signal only opens the new center.

### Native Windows path and packaging

- `Launch-CALO-RPD.ps1`
  - routine repository launcher using only `.venv\Scripts\python.exe`;
  - clear missing-environment guidance;
  - no silent installation or dependency upgrade.
- `pyproject.toml`
  - new direct installed entry point:
    `calo-rpd-native = calo_rpd_studio.app.application:main`;
  - existing setup-aware `calo-rpd-studio` entry remains unchanged.
- `docs/NATIVE_WINDOWS_GUIDE.md`
  - first setup, normal native launch, installed-wheel launch, CUDA check, CPU-only selection,
    logs, data, shutdown, Docker/cache behavior, and troubleshooting.
- `README.md`
  - native setup/start separation and the accepted combined Phase 4/5 state were updated.
- `MANIFEST.in`
  - the interrupted last patch actually succeeded: it now includes `Launch-CALO-RPD.ps1`, the
    native guide, and the Phase 6 original prompt in the sdist while still excluding validation and
    policy artifacts.
- `calo_rpd_studio/scripts/verify_phase6_distribution.py`
  - the interrupted last patch also succeeded completely; the file exists and is approximately
    3.5 KB;
  - it requires one wheel and one sdist, the new GUI modules, the native wheel entry point, and the
    repository launcher/guide/prompt in the sdist.

### Test and render-validation source already added

- `tests/gui/test_phase6_ribbon_workspace.py`
- `tests/unit/test_phase6_command_and_native_contracts.py`
- `tests/integration/test_phase4_empty_policy_workflow.py` updated for hidden legacy plus enabled
  independent entry.
- `calo_rpd_studio/scripts/validate_phase6_gui_contracts.py`
  - offscreen light/dark/constrained renders;
  - registry/category/accessibility/dock/document/activity checks;
  - verifies training navigation starts no process and changes no execution state;
  - explicitly records no policy/scientific work and no inferred human acceptance.

No tests, render scripts, builds, validators, GUI launches, or policy workflows were executed by
the development agent after these edits.

## Current Git state to preserve

At handoff, the intentional Phase 6 edit set includes modified files:

- `ACTIVE_DEVELOPMENT_STATUS.json`
- `MANIFEST.in`
- `README.md`
- `calo_rpd_studio/app/main_window.py`
- `calo_rpd_studio/gui/icons/workspace_icons.py`
- `calo_rpd_studio/gui/panels/calo_intelligence_panel.py`
- `calo_rpd_studio/gui/themes/dark.py`
- `calo_rpd_studio/gui/themes/light.py`
- `calo_rpd_studio/gui/themes/tokens.py`
- `calo_rpd_studio/gui/widgets/global_status_bar.py`
- `calo_rpd_studio/scripts/verify_active_version.py`
- `docs/DOCUMENTATION_STATUS.md`
- `docs/implementation/ACTIVE_CONTINUATION_LOG.md`
- `docs/implementation/IMPLEMENTATION_GATES.md`
- `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md`
- `docs/implementation/REQUIREMENT_TRACEABILITY.md`
- `pyproject.toml`
- `tests/integration/test_phase4_empty_policy_workflow.py`
- `tests/unit/test_v120_phase3_gui_contracts.py`

New/untracked Phase 6 source files include:

- `Launch-CALO-RPD.ps1`
- `calo_rpd_studio/gui/command_registry.py`
- `calo_rpd_studio/gui/panels/independent_training_panel.py`
- `calo_rpd_studio/gui/widgets/activity_center.py`
- `calo_rpd_studio/gui/widgets/context_pane.py`
- `calo_rpd_studio/gui/widgets/document_workspace.py`
- `calo_rpd_studio/gui/widgets/ribbon_bar.py`
- `calo_rpd_studio/scripts/validate_phase6_gui_contracts.py`
- `calo_rpd_studio/scripts/verify_phase6_distribution.py`
- `docs/NATIVE_WINDOWS_GUIDE.md`
- `docs/implementation/PHASE_6_NEW_CHAT_PROMPT.md`
- `tests/gui/test_phase6_ribbon_workspace.py`
- `tests/unit/test_phase6_command_and_native_contracts.py`

This continuation prompt itself will also be a new tracked documentation file. Preserve unrelated
changes and do not attribute the SQLite sidecars to Phase 6 implementation.

## Exact continuation point and remaining work

The next agent should begin here, not at the original Phase 6 audit.

### 1. Perform one narrow static source review

Without executing Python or Qt, inspect the diffs and check the newly added Phase 6 modules for:

- PyQt6 enum/API usage and signal signatures;
- ownership/reparenting of the pinned workspace, training document, docks, and logging handler;
- safe QProcess `FailedToStart`, finish, and close behavior without leaving a false training lock;
- disabled-reason and workflow gating behavior;
- QSettings conversion/restoration, corrupt-state fallback, and test isolation;
- truthful status terminology and no task-state overwrite from layout/view actions;
- bounded input widths and constrained ribbon/action presentation;
- central command uniqueness and no duplicate panel-specific ribbon wiring;
- installed-wheel versus checkout guide/entry behavior;
- wheel/sdist member normalization and policy/validation exclusions;
- existing test compatibility, especially the retained `window.stack` and `window.sidebar` public
  contracts.

Make evidence-backed corrections only. Do not redesign speculative details or replace the modules.

### 2. Finish Phase 6 state/test synchronization

Current status still says:

- `phase_6_coding = started_manual_validation_pending`
- `phase_6_validation = pending_user_executed_noninteractive_phase6_validator`

After the production/test/docs/validator source is complete, update `ACTIVE_DEVELOPMENT_STATUS.json`
to an exact development-complete/manual-validation-pending value. Keep validation pending until the
owner returns evidence. Synchronize the exact expected values in:

- `calo_rpd_studio/scripts/verify_active_version.py`
- `tests/unit/test_v120_phase3_gui_contracts.py`

Do not set RC, final-release, policy-authorization, publication, or release-ready fields true.

### 3. Finish authoritative documentation

Append one concise Phase 6 implementation-complete/manual-validation-pending checkpoint to:

- `docs/implementation/ACTIVE_CONTINUATION_LOG.md`
- `docs/implementation/IMPLEMENTATION_GATES.md`
- `docs/implementation/RELEASE_READY_CONTINUATION_HANDOFF.md`
- `docs/implementation/REQUIREMENT_TRACEABILITY.md`
- `docs/DOCUMENTATION_STATUS.md` if its current-facing routing needs the native guide or this exact
  continuation prompt.

Record what was implemented, which invariants remain unchanged, that no tests/policy work were run
by Codex, and that human usability/screen-reader/scientist acceptance is not inferred.

### 4. Create the one final Git-ignored Phase 6 validator

This is the largest remaining deliverable. Create or finish:

- `validation/Validate-Phase6.ps1`
- `validation/PHASE6_VALIDATION.md`

Both must remain ignored by Git. Do not run them.

The PowerShell validator must be noninteractive, fail clearly, stream command output, retain each
command log, and create a fresh directory:

`validation/logs/phase6-YYYYMMDD-HHMMSS`

It must retain:

- environment and dependency identity;
- exact Git commit, complete dirty status, and a hash of that status;
- hashes for the validator and all declared/changed source files;
- per-command command line, start/end time, duration, exit code, PASS/FAIL, and live log path;
- a machine-readable JSON summary;
- a readable Markdown summary;
- a complete SHA-256 manifest of retained logs/evidence;
- before/after source-status identity proving that validation did not modify nonignored source;
- explicit `false` fields for policy training, evaluation, qualification, registration, activation,
  deletion, protected-case work, release, and inferred human acceptance.

Suggested proportional command sequence:

1. Python version and `pip freeze --all`.
2. `git diff --check`.
3. prove `validation/Validate-Phase6.ps1`, its instructions, and `validation/logs/` are ignored.
4. active-version verification.
5. compilation of application/test source.
6. Ruff diagnostics.
7. Ruff format check.
8. Phase 6 unit contracts.
9. Phase 6 GUI contracts.
10. affected existing GUI startup/visual regressions.
11. empty-policy/training-navigation integration regression.
12. `calo_rpd_studio.scripts.validate_phase6_gui_contracts` with output below the run directory.
13. build exactly one fresh wheel and sdist into the run directory.
14. existing distribution-stage verification.
15. `calo_rpd_studio.scripts.verify_phase6_distribution`.
16. final nonignored source-state stability comparison.

The validator must not call Docker, CUDA campaigns, policy readiness check/start, training,
evaluation, qualification, protected cases, release-scope generation, final-record generation, or
publication. Phase 4/5 already supplied the earlier container/CUDA/release-preparation evidence;
Phase 6 validation is proportional to the GUI/native/packaging changes.

If a command fails, retain a valid partial summary and stop before later expensive work. Never ask
the user for a reviewer answer.

### 5. Final handoff for this development pass

Do not run the validator. Give the owner exactly this manual command:

```powershell
cd "C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio"
& .\validation\Validate-Phase6.ps1 `
    -PythonExecutable ".\.venv\Scripts\python.exe"
```

Ask the owner to return the complete new `validation\logs\phase6-*` directory, not pasted fragments
or only the Markdown summary.

Do not claim Phase 6 validated, close the goal, or claim final release at that point. Review the
returned log read-only in the next turn, correct only evidenced failures, update the same ignored
validator if necessary, and request one fresh complete rerun.

## Definition of success for the continuation

The continuation pass is ready for owner validation only when current source shows all of these:

- one registry-generated tabbed icon ribbon;
- compact contextual left editors bound to shared validated state;
- a central pinned scientific workspace plus reusable singleton documents;
- jobs/logs/warnings/device/provenance activity with truthful determinate/indeterminate progress;
- status bar distinctions for configured mode, actual device, Safe-80 ceiling, policy state, and
  version;
- disabled primary buttons visibly disabled in both themes;
- the clipped legacy training action row removed from the visible interface;
- enabled navigation to a separate independent new-policy center without navigation side effects;
- plan/check/start separation with fresh-readiness invalidation and no implicit qualification or
  activation;
- native repository and installed-wheel launch paths that do not require Docker or reinstall on
  routine startup;
- Docker behavior and all scientific/policy/runtime/persistence boundaries preserved;
- proportional test and render-validation source;
- updated active status, continuation, gate, handoff, traceability, and documentation routing;
- one detailed ignored noninteractive Phase 6 validator plus instructions;
- no validation or policy workflow executed by the development agent.
