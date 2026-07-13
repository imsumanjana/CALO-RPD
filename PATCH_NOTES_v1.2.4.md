# CALO-RPD Studio v1.2.4 Patch Notes

## Accelerator-first heterogeneous scheduling

Version 1.2.4 changes independent-run admission to the explicit priority:

1. NVIDIA CUDA
2. Intel XPU
3. CPU

A compatible CALO job is assigned before launch and remains on that backend for the complete run. No running optimizer state is migrated between devices.

## Mixed NVIDIA + Intel systems

The bootstrap keeps the primary CUDA-enabled PyTorch environment when NVIDIA is available. If Intel graphics is also detected and the primary runtime does not expose XPU, the wizard can provision an isolated secondary XPU virtual environment under the CALO-RPD user state directory. This prevents the Intel-XPU wheel from replacing the primary CUDA wheel while still making XPU jobs available to the scheduler.

## Admission controls

- NVIDIA CUDA compute target and VRAM limit
- Maximum simultaneous CUDA CALO jobs
- Intel XPU compute target when utilization telemetry is available
- XPU memory limit and maximum simultaneous XPU jobs
- CPU utilization target
- System-RAM safety limit

Stable XPU runtimes do not always expose a utilization percentage. In that case CALO-RPD Studio transparently uses XPU device-memory pressure plus the explicit XPU job cap rather than inventing a utilization value.

## Policy training

Automatic PPO-device selection now prefers CUDA, then direct XPU, then the verified secondary XPU runtime, then CPU. CPU rollout generation remains available in parallel because ORPD environment simulation and AC power flow are CPU workloads.

## Scientific limitations

Accelerator routing applies to compatible CALO neural-policy work. The PYPOWER AC power-flow evaluator, physical constraint evaluation, and conventional baseline optimizers remain CPU workloads. Use CPU-only, single-worker mode for strict publication-quality runtime ranking.
