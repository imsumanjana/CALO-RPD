# CALO-RPD v6.8.0 Release Validation

## Status

- Python compileall: **PASS**
- Focused v6.8 independence/XPU suite: **5 passed**
- Combined relevant regression selection: **36 passed**
- Physical Intel XPU in build environment: **not available**
- Target-machine `xpu:0` qualification: **required after bootstrap provisioning**

## Verified contracts

1. A policy-development ExperimentConfig with `runs=1` can be scientifically valid for CALO Intelligence even when the Comparison/Portfolio publication minimum is 30 or more.
2. Full experiment/portfolio validation still rejects insufficient benchmark repetitions; only the CALO Intelligence scientific-development path is decoupled.
3. CALO Intelligence no longer auto-rehydrates its controls when other tabs emit global config changes.
4. Mixed CUDA+Intel hardware requires XPU repair when Intel hardware is detected but no verified direct/sidecar XPU runtime exists.
5. The canonical XPU sidecar can be rediscovered from disk even when serialized bootstrap state is stale.
6. Intel PnP hardware identity (`VEN_8086`) is an accepted physical detection fallback.
7. A detected Intel adapter with no verified runtime remains visible but non-schedulable.
