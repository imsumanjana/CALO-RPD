# CALO-RPD Studio v1.3.0 Patch Notes

This patch upgrades **v1.2.4 → v1.3.0**.

## Live Optimization cleanup

- The permanently expanded Preview legend checkbox section has been removed from the Live Optimization workspace.
- A context-sensitive **Preview series** icon now appears inside Plot Tools for Live Optimization.
- The popup dynamically lists the currently available series and provides **Select all**, **Clear all**, and **Restore default**.
- Preview visibility affects only the displayed canvas; stored data and export-series choices remain independent.

## Historical Experience Learning

CALO Intelligence now includes a leakage-aware historical-learning workflow.

- Experiments can be classified as **TRAIN**, **VALIDATION**, **TEST**, or **EXCLUDED**.
- Only explicitly eligible TRAIN experiments can be included in an experience repository.
- Historical CALO trajectories can be used for offline policy/value pretraining before fresh on-policy PPO.
- Cross-algorithm results can contribute compatible solution exemplars for optional practical warm starts.
- Successful historical CALO training runs can contribute problem-specific parameter priors.
- Legacy v1.2 CALO runs with sufficient diagnostic histories can be reconstructed conservatively as partial, lower-weight regime/operator examples. They do not supervise CALO's continuous parameter actions.
- **Cold Start**, **Historical Warm Start**, and **Continual Learning** modes are available.
- Continual Learning rebuilds the repository from currently eligible TRAIN data; it does not silently retrain or promote a deployed policy.

## Scientific use

For strict independent final benchmark comparisons, use **Cold Start** or a policy frozen before the benchmark data were generated. Do not mark final TEST experiments as learning-eligible. Optional cross-algorithm population warm starts should be disabled when cold-start fairness is required.

## Database compatibility

Existing v1.2 databases are migrated in place with learning-role, eligibility, and classification-lock fields. Existing experiments default to excluded/not learning-eligible unless the user explicitly changes them.
