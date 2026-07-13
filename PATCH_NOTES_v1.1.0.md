# CALO-RPD Studio 1.1.0

Version 1.1.0 adds controlled experiment-history and trace-data removal.

## Data-management capability

- **Delete selected run** removes the completed run record, its validation records, and its referenced compressed `.npz` convergence/final-population trace file.
- **Delete selected experiment** removes the experiment record, all completed runs, failed-run records, validation records, and every referenced compressed trace file owned by the experiment.
- **Delete all experiment history** requires typing `DELETE ALL` before all local experiment history is removed.
- Destructive operations are disabled while a scientific task is active.
- SQLite WAL checkpointing and `VACUUM` are performed after destructive operations so deleted records no longer remain as ordinary application history and database free pages are compacted.
- External publication-export folders are not automatically deleted because they are independent user-managed copies.

## GUI access

History management is available from:

1. **Results Explorer → Manage history**
2. **Application Settings → Experiment history → Manage experiment history**

The history manager reports completed runs, failed-run records, verified results, validation records, referenced trace files, and referenced trace storage before deletion.
