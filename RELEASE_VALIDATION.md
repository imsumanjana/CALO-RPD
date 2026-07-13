# CALO-RPD Studio 1.3.0 — Release Validation Record

Version 1.3.0 adds leakage-aware Historical Experience Learning and moves Live Optimization preview-series selection into Plot Tools.

## Release checks

- Python source compilation: PASS
- Automated test suite: PASS — 86 tests passed
- Ruff static analysis: PASS
- Wheel build: PASS
- Source-distribution build: PASS
- Full source archive integrity: PASS after packaging
- Incremental patch archive integrity: PASS after packaging

## Historical-learning checks

- New and migrated experiments default to EXCLUDED/not learning-eligible.
- Only experiments explicitly classified as TRAIN and marked learning-eligible are admitted to the experience repository.
- VALIDATION and TEST experiments cannot remain learning-eligible.
- Experiment learning classifications can be locked against accidental GUI editing.
- Repository payloads carry a deterministic SHA-256 checksum and are verified when loaded.
- Exact v1.3 CALO policy trajectories can contribute state, regime, operator, bounded parameter action, reward, and evaluation information.
- Legacy v1.2 CALO diagnostic histories are only reconstructed when sufficient telemetry exists; reconstructed samples are marked partial, assigned lower quality weight, and excluded from continuous parameter-action supervision.
- Historical trajectories are used only for offline supervised/value pretraining. Fresh trajectories are generated before PPO updates, preserving the on-policy PPO stage.
- Cross-algorithm historical solutions and CALO parameter priors are separate opt-in warm-start mechanisms.
- Cold Start removes historical runtime-learning parameters from the CALO configuration.
- Continual Learning rebuilds eligible repositories but does not silently retrain or promote a deployed policy.

## Live Optimization checks

- The old permanently expanded preview-series checkbox section is removed.
- Live Optimization exposes a context-sensitive Preview series icon within Plot Tools.
- Available preview checkboxes are generated dynamically from the active plot series.
- Select all, Clear all, and Restore default affect only the live preview.
- Export-series selection remains independent from preview-series selection.

## Scientific benchmark safeguards

Final benchmark/test experiments must remain classified as TEST or EXCLUDED and are not admitted to historical learning. Optional historical parameter priors and cross-algorithm population warm starts should be disabled for strict cold-start fairness unless the study protocol explicitly evaluates an experience-informed configuration.

## Hardware-validation scope

The v1.3.0 changes do not alter the accelerator-first execution architecture introduced in v1.2.4. The release environment validated the full software test suite in CPU/offscreen GUI mode. Actual CUDA/XPU throughput remains hardware-dependent and is verified by the first-launch prerequisite wizard on the target machine.
