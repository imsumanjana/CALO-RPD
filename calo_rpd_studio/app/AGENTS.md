# Application orchestration instructions

- Policy-assisted power-system experiments consume only a separately qualified, explicitly activated,
  immutable policy snapshot; non-policy baseline workflows remain available through explicit safe fallback.
- Experiment workflows must never start, resume, retrain, modify, qualify, or activate policy training.
- Snapshot governing policy identity and scientific protocol immutably when a run starts.
- Keep configuration changes transactional and downstream invalidation explicit.
- Treat zero installed/registered/active policies as a supported startup state. Keep non-policy
  workflows usable and policy-dependent formal experiments explicitly locked.
- Removing an old policy must never leave a dangling active reference or trigger regeneration,
  download, selection, qualification, registration, or activation.
