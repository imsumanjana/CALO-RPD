# Workflow instructions

- Inherit the repository scientific and release boundaries.
- Pin third-party actions to immutable full commit SHAs.
- Keep release-critical dependency installation hash-locked.
- Separate source, GUI, CPU-container, CUDA-build, physical-CUDA, compatibility, and artifact evidence.
- A workflow definition is a harness, not proof that a job or physical qualification ran.
- Phase 4 workflow lanes must validate development and empty-policy behavior without training,
  evaluating, qualifying, deleting, registering, or activating a policy.
- Release artifacts and workflow uploads must exclude old/generated policies; a future newly
  qualified policy uses a separate post-development identity and manifest.
