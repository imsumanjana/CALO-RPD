# AI module instructions

- Keep policy artifacts versioned, immutable, checksum-bound, and schema-compatible.
- Never make policy training or experiment execution implicitly activate or mutate a policy.
- Reject incompatible artifacts explicitly and preserve deterministic safe fallback.
- Treat every existing policy as development-only, unqualified, inactive, non-final, and excluded
  from release or final-policy initialization.
- Phase 4 must support an empty policy store and may prepare inventory/dry-run removal only; actual
  deletion and completely new policy training require separate post-freeze authorization.
