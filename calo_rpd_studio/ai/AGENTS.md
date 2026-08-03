# AI module instructions

- Keep policy artifacts versioned, immutable, checksum-bound, and schema-compatible.
- Never make policy training or experiment execution implicitly activate or mutate a policy.
- Reject incompatible artifacts explicitly and preserve deterministic safe fallback.
