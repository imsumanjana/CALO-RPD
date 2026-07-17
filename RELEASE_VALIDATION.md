# CALO-RPD Studio 3.0.0 — Release Validation Record

## Scope

Version 3.0.0 introduces the PyTorch FP64 accelerator-native ORPD evaluator and canonical tensor kernels for all nineteen non-CALO baselines. CALO uses the same accelerator evaluator plus its existing cognitive/AI controller. The legacy CPU evaluator remains the independent reference.

## Checks completed in the packaging environment

- `python -m pytest -q`: **88 passed, 25 skipped**.
- `python -m compileall -q calo_bootstrap calo_rpd_studio tests`: passed.
- `python -m pip wheel . --no-deps`: passed and produced `calo_rpd_studio-3.0.0-py3-none-any.whl`.
- CPU-only PyTorch FP64 parity tests on the deterministic toy AC system passed within near-machine-precision errors.
- Batched candidate evaluation matched the scalar CPU reference for the covered toy cases.
- Smoke tests executed every one of the nineteen torch baseline kernels with the common result contract.
- A weighted 8-algorithm × 50-run plan with 100% CUDA request was verified to produce 400 CUDA lane assignments when CUDA availability is declared.
- Freeze-manifest verification passed after creation of `calo_v3_freeze.json`.

## Skipped or unavailable checks

- Physical NVIDIA CUDA execution was not available in the packaging environment; installed PyTorch was CPU-only.
- Physical Intel XPU execution was not available.
- PyQt6 was unavailable, so GUI tests were skipped.
- PYPOWER was unavailable, so three IEEE/PYPOWER cross-validation tests were skipped.
- Ruff was not available and is not claimed as executed.

The first-launch prerequisite wizard performs real CUDA/XPU tensor tests on the target workstation. Before publication use, run the v3 parity audit on every benchmark formulation and independently validate stored final solutions against the trusted CPU reference.

## Scientific boundary

This release makes all primary optimizer jobs accelerator-compatible and improves shared physical-formulation consistency and throughput. It does not guarantee that any optimizer converges better. Solution-quality claims require the complete repeated-run, validation, and statistical protocol.
