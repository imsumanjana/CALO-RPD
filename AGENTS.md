# CALO-RPD agent instructions

## Scope
These instructions apply to the entire repository unless a deeper `AGENTS.md` narrows them.

## Scientific and release boundaries
- Preserve deterministic baseline behavior and exact function-evaluation accounting.
- TSH-CALO A–E are approved for production-candidate implementation. F is experimental, independently feature-flagged, and disabled by default.
- Do not change scientific semantics outside the approved A–F proposal without explicit user approval.
- Policy training is independent. Experiments may only consume separately qualified, explicitly activated, immutable, checksum-valid policies.
- Never auto-train, auto-modify, auto-qualify, or auto-activate a policy from an experiment workflow.
- Preserve explicit safe fallback for unavailable, incompatible, or rejected policies.
- Intel XPU must not be executable. Current modes are CUDA-preferred and CPU-only.
- Admission ceilings use at most 80% of currently free VRAM or currently available RAM. CUDA computes on NVIDIA GPUs; CPU fallback computes on CPUs.
- Do not fabricate hardware, container, performance, energy, thermal, or scientific evidence.
- Do not make release-ready or superiority claims before their documented gates have direct evidence.

## Workflow
- Follow `docs/implementation/IMPLEMENTATION_GATES.md` in order and keep the handoff and traceability ledger current.
- Before semantic CALO changes, prove canonical-refactor parity against the frozen baseline.
- Add unit, invariant, parity, ablation, falsification, leakage, fallback, and regression tests proportional to each change.
- Keep protected cases out of training, tuning, reward design, and checkpoint selection.
- Do not regenerate release freezes, manifests, SBOMs, image digests, or public release claims before their gates close.
- Preserve user files and unrelated changes. Do not push, merge, publish, or release without explicit approval.
