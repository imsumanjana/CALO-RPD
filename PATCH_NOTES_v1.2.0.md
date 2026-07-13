# CALO-RPD Studio 1.2.0

Version 1.2.0 introduces CALO Core v2 and the CALO Diagnostic Framework.

## CALO Core v2

- adaptive epsilon-feasibility with decay to exact feasibility;
- separate feasible-elite and diverse constraint-boundary archives;
- per-individual operator allocation;
- feasible-elite, boundary-differential, cognitive-teacher, success-memory, mixed-variable, and diversity-recovery operators;
- environmental selection from parents and offspring;
- separate objective and constraint stagnation states;
- online operator credit blended with the AI policy;
- physically meaningful neighbouring moves for discrete taps and shunts;
- temporary recovery with cooldown instead of a permanent stagnation mode.

## Diagnostic framework

- total and component-wise constraint tracking;
- exact and epsilon-feasible population ratios;
- adaptive epsilon;
- population and elite diversity;
- CALO regime;
- operator success rates;
- evaluations to first exact feasibility;
- persistent diagnostic histories in result metadata.

## Rebuilt AI

- 24-dimensional constraint-aware state;
- hierarchical regime/operator/continuous-parameter policy;
- bounded Beta-distributed continuous actions;
- clipped PPO objective;
- generalized advantage estimation;
- entropy and value losses;
- minibatch multi-epoch optimization;
- training environment uses the same CALO Core v2 operator and selection modules as runtime.

## Live Optimization

- dynamic Preview legend with checkboxes;
- immediate selective preview without deleting raw data;
- diagnostic plot modes for constraint decomposition, feasibility, diversity, and operator success;
- square preview and square 600–2400 DPI PNG export retained.
