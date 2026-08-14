# Test instructions

- Tests must be deterministic and must not weaken acceptance criteria to make a component pass.
- Separate unit/invariant, parity, migration, GUI, container, hardware, ablation, falsification, and protected-campaign evidence.
- Historical release tests remain scoped to their artifacts and do not redefine the active tree.
- Development tests must not train/evaluate real policies or open protected cases. Use empty-policy
  state and deterministic temporary synthetic fixtures for lifecycle, compatibility, migration,
  deletion, qualification admission, activation, binding, and fallback.
- Test stage-neutral compatibility explicitly: software revision and development phase are
  provenance, while ABI, architecture/schema, integrity, receipt, qualification, and activation
  failures remain fail-closed.
