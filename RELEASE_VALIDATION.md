# CALO-RPD Studio 1.1.0 — Release Validation Record

Version 1.1.0 adds persistent experiment-history and referenced trace-data cleanup while preserving the scientific execution, plotting, validation, and publication workflow introduced in earlier releases.

## Release verification

- 60 automated tests passed.
- PyQt6 GUI tests passed in offscreen Qt mode.
- IEEE scientific tests passed with PYPOWER available.
- Ruff static analysis passed.
- Python source compilation passed.
- Wheel and source-distribution builds completed successfully.

## History-management verification

Automated tests verify that:

- deleting one run removes only that run, its validation records, and its referenced trace file;
- deleting one experiment removes all completed runs, failed-run records, validation records, and referenced trace files belonging to that experiment;
- deleting all history removes all experiment records and referenced trace files;
- other experiments and unrelated files are preserved;
- database WAL checkpointing and compaction complete after destructive history operations;
- Results Explorer and Application Settings expose history management without source-code access;
- destructive controls are disabled while a scientific task is active.

External publication-export folders are intentionally not removed automatically because they are independent user-managed copies.
