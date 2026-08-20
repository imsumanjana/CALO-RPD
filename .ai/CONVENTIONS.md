# Conventions

- Python >=3.11; setuptools package discovery includes `calo_rpd_studio*` and `calo_bootstrap*`.
- Ruff/Black line length: 100; mypy targets Python 3.11.
- Tests use pytest; GUI tests use pytest-qt/offscreen CI where appropriate.
- Follow the closest `AGENTS.md`; deeper instructions narrow repository-wide rules.
- Preserve deterministic ordering, exact evaluation accounting, immutable provenance and fail-closed integrity checks.
- Keep application behavior changes separate from tooling/refactoring changes.
- Do not treat indexing as auditing. Audit coverage must bind to the current content hash.
- Generated/vendored/historical artifacts are normally excluded from AI retrieval unless a task explicitly targets them.
