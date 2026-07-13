# CALO-RPD Studio 1.2.1

Version 1.2.1 makes CALO policy training resource-aware. Rollout episodes can be collected concurrently in independent CPU processes, while PPO neural-network updates run on the selected compute device. Automatic mode uses CUDA when the installed PyTorch build can access a CUDA GPU and otherwise uses CPU.

The design deliberately does not attempt to fill all system memory. Memory usage scales with rollout size and PPO batches; unused memory is not an error. GPU utilization can be bursty because the expensive optimization-environment rollouts remain CPU work while the centralized policy update is accelerated.
