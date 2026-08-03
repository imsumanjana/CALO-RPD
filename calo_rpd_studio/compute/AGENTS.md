# Compute instructions

- Expose only CUDA-preferred and CPU-only current execution modes.
- Do not introduce Intel XPU paths, utilization targets, task shares, or work stealing.
- Use one central device lease and at most 80% of memory free/available at admission.
- Memory stores data; GPU or CPU performs computation. Record actual execution honestly.
