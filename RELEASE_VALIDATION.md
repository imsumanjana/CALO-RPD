# CALO-RPD Studio 1.0.3 — Release Validation Record

Release date: 2026-07-11

## Functional changes reviewed

- Guided workflow manager added with prerequisite locking and downstream invalidation.
- Sidebar states distinguish locked, recommended, completed, optional, and available workspaces.
- Persistent workflow guide shows the next required action.
- Power System validation controls are sequential: load case → base AC power flow → independent cross-validation.
- Experiment execution is disabled until the fairness audit passes.
- New experiments invalidate prior post-experiment workflow progression for the active session.
- Statistical Analysis, Results Explorer, Validation & Audit, and Publication Export follow a controlled post-experiment sequence.
- Publication Export remains locked until the current experiment contains an independently verified result.
- Persistent bottom task bar reports Ready/Busy/Completed/Failed state, task detail, progress, elapsed time, and safe cancellation where supported.
- CALO policy training reports epoch/episode progress and supports safe cancellation.
- Comparative and CALO analysis runs report overall progress across run items.
- Experiment cancellation no longer marks the workflow as a completed experiment; completed partial runs are retained.

## Verification performed in the build environment

- Python source compilation: passed.
- Core automated tests available in the build environment: **31 passed**.
- GUI tests: **6 skipped** because PyQt6 is not installed in the build container.
- Scientific PYPOWER cross-validation tests: **3 skipped** because PYPOWER is not installed in the build container.
- CALO training progress callback smoke check: passed (1 epoch, 1 episode, 100% progress, checkpoint written).
- The previous 1.0.2 release had already validated the GUI and PYPOWER paths in an environment containing those dependencies; version 1.0.3 changes were additionally checked by source compilation and the available automated core suite.

## Required local verification after installation

Run the complete suite in the project virtual environment, where PyQt6 and PYPOWER are installed:

```powershell
python -m pytest -q
```

Then launch:

```powershell
python main.py
```

The expected initial workflow state is Dashboard + Power System + Application Settings available, with downstream scientific workspaces locked until prerequisites are completed.
