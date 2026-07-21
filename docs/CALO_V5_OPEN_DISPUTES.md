# CALO-RPD v5.0 — Open / Partial Scientific and Performance Disputes

This file prevents unresolved work from being mistaken for completed capability.

## Critical/High open or partial items

- **P02 — PARTIAL:** PPO training and full runtime CALO are not a bit-identical shared transition implementation. Policy Qualification remains mandatory.
- **P05 — PARTIAL:** Long policy lineages and latest-vs-best are implemented, but configured periodic qualification is not automatically executed.
- **P06 — OPEN:** No bundled legacy policy has demonstrated universal/native ORPD superiority.
- **P07 — OPEN:** AI/rule/online-credit authority weights require paired IEEE 30/57 ablation and freeze before holdout use.
- **P08/P09 — DEFERRED:** CALO control/cognitive state is not fully Torch/CUDA/XPU resident. The accelerator-native evaluator does not imply end-to-end CALO control residency.
- **P10/P14 — PARTIAL:** Some per-learner/memory operations remain Python/host-side.
- **P11 — OPEN:** CUDA policy inference still materializes compact actions to host because CALO control is host-side.
- **C03 — PARTIAL:** Exact optimizer-state horizon continuation is implemented for CALO, not every baseline. Multi-algorithm higher-horizon studies use paired recompute-from-original-seed when exact baseline continuation is unavailable.
- **C08 — OPEN:** Automatic asynchronous qualification/promotion during long training is not implemented; the configured interval is advisory.

## Scientific interpretation

None of these items invalidates the implemented continuation/provenance mechanisms. They constrain what may be claimed:

- Do not claim full device-resident CALO control.
- Do not call the PPO rollout environment identical to runtime.
- Do not call a newer policy better without qualification.
- Do not call baseline recomputation exact continuation.
- Do not claim universal CALO superiority without frozen paired feasible independent evidence.
