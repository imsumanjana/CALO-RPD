# Integration-test instructions

- Cover cross-module persistence, policy binding, workflow independence, fallback, and migration behavior.
- Use isolated temporary data and preserve user state.
- Cover empty-policy startup, stale-reference rejection, stage-neutral formal admission, locked
  invalid workflows, and dry-run policy removal with synthetic fixtures; do not train, evaluate,
  activate, bind, or delete real policies during development tests.
