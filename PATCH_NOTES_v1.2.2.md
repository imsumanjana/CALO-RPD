# CALO-RPD Studio 1.2.2

## First-launch prerequisite setup

Version 1.2.2 introduces a standard-library-only prerequisite bootstrap that runs before the PyQt6 application is imported. On first launch, or when the Python interpreter/application version changes, the setup wizard:

- checks Python and the scientific/GUI dependency stack;
- detects NVIDIA hardware and driver information through `nvidia-smi` when available;
- detects whether the installed PyTorch build can actually execute a CUDA tensor operation;
- installs or repairs the core prerequisites;
- selects an official CUDA-enabled PyTorch wheel channel compatible with the driver when possible, with ordered fallbacks and CPU fallback;
- installs the project without letting dependency resolution overwrite the selected hardware-specific PyTorch build;
- verifies the final environment before enabling **Start CALO-RPD Studio**.

The bootstrap itself does not import PyQt6 or PyTorch in the parent process, so missing prerequisites can be repaired before the scientific application starts.

## GPU-enabled CALO policy training

The v1.2.1 centralized PPO device path is retained and now benefits from the hardware-aware prerequisite setup. With a verified CUDA-enabled PyTorch installation, **Automatic** policy training uses CUDA for PPO neural-network updates while rollout environments remain parallel CPU workloads.

## Adaptive heterogeneous experiment scheduling

Primary comparison and CALO ablation execution now support three backends:

- **Adaptive hybrid CPU + GPU**
- **GPU preferred with CPU fallback**
- **CPU only**

The scheduler exposes soft admission targets for:

- GPU utilization;
- GPU memory usage;
- CPU utilization;
- maximum concurrent GPU CALO jobs;
- maximum total concurrent jobs.

CUDA-capable experimental jobs are CALO variants that actually use the neural policy. Those jobs can execute CALO policy inference on `cuda:0`. AC power flow, constraint evaluation, and the 19 non-AI baseline optimizers remain CPU workloads in v1.2.2.

The adaptive scheduler never migrates an optimizer mid-run. It assigns an independent run item to a device before launch, records that assignment in result metadata, and only uses utilization thresholds to decide whether another job should be admitted.

The default scheduling targets are:

- GPU utilization target: 70%
- CPU utilization target: 50%
- GPU memory safety limit: 85%
- maximum GPU CALO jobs: 1

These are soft throughput controls, not guarantees of exact operating-system utilization.

## Reproducibility and fairness

- Each run records the assigned compute device and execution backend.
- Experiment provenance records CUDA availability, PyTorch CUDA runtime, GPU name, and GPU count.
- The fairness audit warns that heterogeneous scheduling is suitable for equal-evaluation solution-quality studies but not strict cross-algorithm runtime ranking.
- Strict runtime comparisons should use one worker and CPU-only execution.

## Validation

- 77 automated tests passed in the release environment.
- Ruff static analysis passed.
- Python source compilation passed.
