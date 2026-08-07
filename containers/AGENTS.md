# Container instructions

- Maintain separate reproducible CPU and CUDA profiles from the same source.
- Run non-root with read-only application filesystem, dropped capabilities, bounded temporary storage, and explicit persistent data.
- Build/install from immutable hash-locked inputs; runtime must not install packages.
- Definitions and locks are not runtime evidence; retain real digests, SBOMs, scans, and hardware results only after execution.
- Phase 4 images must start safely with an empty policy store and exclude old/generated policies.
  A future newly qualified policy remains a separately manifested post-development artifact.
