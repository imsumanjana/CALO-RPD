# CALO-RPD Studio 1.0.3

This release adds a guided scientific workflow and a persistent application-wide task/progress system.

## Guided workflow

The navigation sidebar now reflects the actual experiment sequence. Downstream workspaces remain locked until their prerequisites are completed and validated.

The default sequence is:

1. **Power System** — load a case, run the base AC power flow, and pass independent PYPOWER cross-validation.
2. **ORPD Formulation** — apply the common objective, decision variables, mixed-variable settings, and constraint policy.
3. **Algorithms** — select the comparison methods and apply their parameter configuration.
4. **CALO Intelligence** — required when CALO is selected; validate and apply the frozen policy configuration.
5. **Robust Scenarios** — apply deterministic or robust scenario settings.
6. **Experiment Manager** — pass the fairness audit before experiment execution becomes available.
7. **Live Optimization** — unlocks when an experiment starts.
8. **Statistical Analysis** — unlocks after a completed experiment.
9. **Results Explorer** — unlocks after statistical analysis and requires a confirmed result review.
10. **Validation & Audit** — unlocks after result review.
11. **Publication Export** — unlocks only after at least one independently verified result exists for the current experiment.

Dashboard and Application Settings remain available throughout the workflow.

Changing an upstream scientific stage invalidates dependent downstream workflow state. Starting a new experiment also invalidates prior post-experiment progression for the active session.

## Global bottom task bar

The persistent bottom bar now replaces the static Ready-only behavior with application-wide task telemetry:

- **Ready** when no task is active.
- **Busy** while training, optimization, validation, statistical analysis, or export is running.
- operation name and current detail;
- determinate percentage when available;
- indeterminate progress for tasks without a natural percentage;
- elapsed time;
- safe cancellation for CALO training and optimization experiments;
- completion or failure state before returning to Ready.

## CALO training progress

CALO policy training now reports progress by epoch and episode and supports safe cancellation between training units. The saved training metadata identifies software version 1.0.3.

## Experiment execution guidance

The Experiment Manager now requires a successful fairness audit before either comparative execution or CALO ablation execution is enabled. Experiment progress is reported globally in the bottom task bar while the run queue continues to show per-run status.
