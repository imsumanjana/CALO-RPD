# Application orchestration instructions

- Power-system experiments consume only a separately qualified, explicitly activated, immutable policy snapshot.
- Experiment workflows must never start, resume, retrain, modify, qualify, or activate policy training.
- Snapshot governing policy identity and scientific protocol immutably when a run starts.
- Keep configuration changes transactional and downstream invalidation explicit.
