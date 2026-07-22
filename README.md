# CALO-RPD Studio 5.6.0

**CALO-RPD Studio 5.6.0** is a Python/PyQt6 research platform for deterministic and robust optimal reactive power dispatch (ORPD), reproducible comparison of twenty optimizers, Cognitive Adaptive Learning Optimizer (CALO) research, policy qualification, independent validation, statistics, and publication evidence.

## v5.6 — Competitive Multi-Branch Policy Evolution

Version 5.6 replaces independent-policy parameter averaging with **competitive, independently resumable PPO branches** and keeps the logical Base Model separate from exact branch training state.

### Policy training modes

- **Cumulative:** the user selects a fixed number of epochs for the current session; exact resume adds the same or a newly selected fixed session length to the saved lifetime epoch.
- **Infinite:** no terminal epoch is specified. Training continues until Safe Stop.
- **Exact Resume:** restores each branch's model, optimizer, RNG, curriculum/history and epoch exactly, then continues in either Cumulative or Infinite duration mode.
- **Base-Guided Fork:** starts fresh optimizer/RNG trajectories from a selected Base Model's deployable knowledge. It is explicitly distinct from Exact Resume.

### Competitive parallel branches

- Parallel branches may use the **same seed**, `seed+1/+2/...`, `seed-1/-2/...`, or an explicit custom seed mixture selected by the user. Same-seed branches are allowed to remain identical if deterministic execution produces identical trajectories.
- Branches are independent PPO trajectories. **Their neural-network parameters are never arithmetically averaged.**
- Each branch maintains two distinct concepts: an exact resumable **working state**, which always advances, and a **Branch Champion**, which changes only when fixed validation evidence is scientifically superior.
- Branch Champion comparison uses mandatory validity/feasibility gates followed by critical-metric Pareto checks and a broad multi-metric evidence comparison including final feasible objective, convergence AUC, feasibility, violation, stability, validation return and computational overhead. Formal Policy Qualification remains a separate Candidate-vs-Reference-vs-No-AI gate.
- At session completion, the previous Base and all Branch Champions compete under the same comparator. The logical Base changes only when a superior candidate exists. Older experiment-bound policy artifacts remain immutable by SHA-256.

### Safe Stop and low-RAM exact rollback

- No permanent epoch-by-epoch checkpoint snapshots are created. During an active session, every branch keeps only a bounded rolling set of **temporary exact-state snapshots on local disk** at 10-epoch safe boundaries.
- On Safe Stop, the coordinator selects the lowest common available previous 10-epoch boundary, discards later work, writes one permanent exact resume checkpoint per branch, selects/promotes the best Base if justified, and deletes the temporary session directory.
- The disk-backed rolling window limits RAM pressure while allowing faster branches to remain a bounded distance ahead of slower branches.

### Experiment evolution and restoration

The v5 continuation architecture remains available: experiments can add independent paired runs, extend supported FE horizons with explicit provenance, preserve old horizon evidence, and restore stored workspace parameters/plots/policy bindings. Exact segmented continuation and from-start recomputation remain scientifically distinct.

## Scientific rules

CALO never receives hidden extra objective evaluations. A continuation segment starts from an authenticated optimizer checkpoint and all newly requested evaluations count normally. Increasing the number of runs preserves paired seed semantics. Historical or selectively extended evidence is never silently mixed into primary publication statistics.

CALO-RPD 5.6.0 intentionally ships with **no automatically active/default neural policy**. Policy-assisted CALO is fail-closed: the user must train or import a compatible policy, qualify it as required, explicitly activate it, and bind its immutable SHA-256 to the experiment. No random/untrained/legacy/missing-policy fallback is permitted. Policy promotion remains based on recorded Candidate vs Reference vs No-AI CALO qualification under paired equal-FE runs. IEEE 118/300 remain protected holdout systems unless a study explicitly documents otherwise.

## Run

```bash
python bootstrap.py
```

Windows launcher and dependency/bootstrap helpers remain included in the repository.


## Important v5.6.0 limitations and open research work

- CALO cognitive/control state is **not yet fully Torch/CUDA/XPU resident**; the common ORPD numerical evaluator is accelerator-capable, while portions of CALO control remain compact NumPy/Python host logic. No full-device-control claim is made.
- The lightweight PPO training environment uses the native 32-feature policy schema and CALO cognition semantics, but it is **not a bit-identical implementation of the complete runtime transition loop**. Candidate policies still require real-optimizer Policy Qualification before scientific promotion.
- Exact optimizer-state horizon continuation is currently implemented for CALO. Other algorithms can participate fairly at a larger horizon through paired recomputation from their original seeds; they are not falsely labeled exact continuations.
- `qualification_interval_epochs` is currently an advisory scheduling field. Automatic asynchronous qualification/promotion during indefinite training is not yet implemented.
- No default neural policy is bundled or implied. User-selected policies must be explicitly qualified/activated/bound; final claims require frozen, paired, feasible, independently validated evidence.
