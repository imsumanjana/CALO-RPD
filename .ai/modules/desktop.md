# desktop

**Purpose:** Scientist-facing PyQt shell, shared application state, task/workspace management and workflow navigation.

**Important state:** AppState/workspace/task state is authoritative; panels should not keep divergent scientific copies.

**Major flow:** application -> MainWindow -> panels/workflow managers -> services -> results -> UI refresh.

**Constraints/invariants:** Scientist wording, zero-policy safety, accessibility and atomic shared state updates.

**Common failure points:** Stale UI state, duplicated ownership, task lifecycle races and backend state leaking into scientific choices.

## Primary files
- `calo_rpd_studio/app/experiment_manager.py`
- `calo_rpd_studio/gui/panels/experiment_manager_panel.py`
- `calo_rpd_studio/gui/panels/_calo_intelligence_panel_core.py`
- `calo_rpd_studio/gui/panels/_independent_training_panel_core.py`
- `calo_rpd_studio/gui/plotting/plot_format_toolbar.py`
- `calo_rpd_studio/gui/widgets/context_pane.py`
- `calo_rpd_studio/app/main_window.py`
- `calo_rpd_studio/gui/panels/live_optimization_panel.py`
- `calo_rpd_studio/gui/panels/algorithms_panel.py`
- `calo_rpd_studio/app/workflow_manager.py`
- `calo_rpd_studio/gui/panels/publication_export_panel.py`
- `calo_rpd_studio/app/execution_control.py`

## Important public/entry symbols
- `ErrorDialog` — `calo_rpd_studio/app/exception_handler.py:7-22`
- `ExecutionControlService` — `calo_rpd_studio/app/execution_control.py:22-360`
- `ExperimentWorker` — `calo_rpd_studio/app/experiment_manager.py:189-2148`
- `ExperimentWorker._run_sequential._ProgressRelay` — `calo_rpd_studio/app/experiment_manager.py:582-587`
- `ExperimentWorker._run_sequential._CancelRelay` — `calo_rpd_studio/app/experiment_manager.py:589-594`
- `ExperimentManager` — `calo_rpd_studio/app/experiment_manager.py:2151-2438`
- `WorkspaceRestoreError` — `calo_rpd_studio/app/experiment_workspace_restorer.py:18-31`
- `ExperimentWorkspaceRestorer` — `calo_rpd_studio/app/experiment_workspace_restorer.py:34-246`
- `MainWindow` — `calo_rpd_studio/app/main_window.py:78-1067`
- `ProjectManager` — `calo_rpd_studio/app/project_manager.py:6-13`
- `SessionRecoverySnapshot` — `calo_rpd_studio/app/session_recovery.py:61-86`
- `SessionRecoveryJournal` — `calo_rpd_studio/app/session_recovery.py:89-159`
- `SettingsManager` — `calo_rpd_studio/app/settings_manager.py:6-17`
- `AppState` — `calo_rpd_studio/app/state_manager.py:21-238`
- `TaskManager` — `calo_rpd_studio/app/task_manager.py:6-11`
- `TaskStatus` — `calo_rpd_studio/app/task_status.py:8-122`
- `WorkflowDescriptor` — `calo_rpd_studio/app/workflow_manager.py:15-24`
- `WorkflowManager` — `calo_rpd_studio/app/workflow_manager.py:73-598`

## Dependencies
- `calo-policy`, `compute`, `core`, `experiments`, `optimization`, `persistence`, `power-system`, `validation-release`

## Dependents
- `bootstrap`, `tests`, `validation-release`

## Associated tests
- `tests/gui/test_ci_visual_smoke.py`
- `tests/gui/test_gui_startup.py`
- `tests/gui/test_guided_workflow.py`
- `tests/gui/test_history_manager.py`
- `tests/gui/test_phase6_ribbon_workspace.py`
- `tests/gui/test_results_review_navigation.py`
- `tests/gui/test_task_status.py`
- `tests/gui/test_tsh_calo_generalization_integration.py`
- `tests/gui/test_tsh_calo_policy_library_accounting.py`
- `tests/gui/test_v620_gui_workflow.py`
- `tests/gui/test_workspace_execution_ui.py`
- `tests/gui/test_workspace_tabbed_layouts.py`
- `tests/integration/test_phase4_empty_policy_workflow.py`
- `tests/integration/test_workspace_execution_control.py`
- `tests/unit/test_calo_v41_workflow_restore.py`

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
