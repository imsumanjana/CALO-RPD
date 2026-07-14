# CALO-RPD Studio v2.0.2 Patch Notes

## Weighted policy training

Policy training can now collect fresh on-policy rollouts concurrently through three actor lanes:

- NVIDIA CUDA — default 50% of episodes/transitions;
- Intel XPU — default 30%;
- CPU — default 20%.

The actual integer allocation uses the largest-remainder method. For the default 12 episodes per epoch this becomes 6 CUDA, 4 XPU, and 2 CPU episodes.

Every lane receives the same policy snapshot. PPO updating starts only after all matching trajectories have returned. A stale or mismatched actor payload is rejected. CUDA/XPU actors batch policy-network inference, while environment, PYPOWER, and physical constraint calculations remain primarily CPU work.

## Frozen benchmark protection

The original v2.0.0 CALO freeze remains valid because the frozen training implementation and checkpoint are unchanged. Weighted training writes a candidate checkpoint under a new filename. Validate and create a new freeze manifest before using that candidate in a final TEST campaign.

## GUI

CALO Intelligence now includes:

- weighted heterogeneous versus legacy rollout mode;
- configurable CUDA/XPU/CPU shares;
- live requested/effective episode allocation;
- explicit PPO learner-device selection;
- a warning that transition shares are not guarantees of identical operating-system utilization.
