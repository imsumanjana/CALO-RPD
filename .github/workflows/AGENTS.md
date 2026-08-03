# Workflow instructions

- Inherit the repository scientific and release boundaries.
- Pin third-party actions to immutable full commit SHAs.
- Keep release-critical dependency installation hash-locked.
- Separate source, GUI, CPU-container, CUDA-build, physical-CUDA, compatibility, and artifact evidence.
- A workflow definition is a harness, not proof that a job or physical qualification ran.
