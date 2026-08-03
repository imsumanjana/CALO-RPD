# Experiment instructions

- Applying an experiment protocol must not read or mutate policy-training configuration.
- Store one validated global protocol and governing-policy snapshot atomically.
- Preserve paired seeds, FE budgets, failure records, case roles, and immutable started-run snapshots.
