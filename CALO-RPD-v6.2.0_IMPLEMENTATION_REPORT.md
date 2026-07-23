# CALO-RPD v6.2.0 Implementation Report

## Release objective

v6.2.0 completes the planned v6 architecture on top of the v5.9 scientific-closure core and v6.0/v6.1 policy-first protected scheduling architecture. The release focuses on the requested RC/final upgrades: adaptive compute/thermal protection and staged startup; workspace migration/recovery/provenance; GUI validation contracts; and a hardware-soak/scientific-equivalence qualification framework.

## RC1 — Adaptive compute/thermal governor and staged startup

### Green / Amber / Red governor
A central `AdaptiveComputeGovernor` now evaluates the canonical Safe-80 protection profile against live telemetry.

- **GREEN:** new work may be admitted.
- **AMBER:** new admissions are blocked and protected pacing is applied where supported.
- **RED:** a protective stop is requested. Competitive policy training uses exact Safe-Stop semantics; experiment jobs stop admitting new work and cancel active uncommitted jobs while already committed runs remain immutable.

Hysteresis prevents one noisy sample from immediately changing the persistent protection state. A raw RED sample blocks new admission immediately while the configured confirmation count controls terminal protective-stop escalation.

### Telemetry
`ResourceMonitor` now carries optional trustworthy telemetry for:

- CPU/package temperature when the host runtime exposes a reliable sensor;
- accelerator temperature;
- accelerator power draw and reported power limit;
- accelerator throttle reason;
- CPU/RAM/utilization/memory state.

Unavailable telemetry remains `None`/`unavailable`; the software does not invent temperatures.

### Staged startup
Competitive training no longer launches all admitted branches at once. A configurable staged-start delay is enforced between branch admissions. The experiment scheduler uses the same staged-admission principle. This reduces simultaneous CUDA/XPU context creation, worker-pool startup and host-process power spikes.

### Protection provenance
Compute-protection decisions are written to durable append-only JSONL event chains. Each event includes the previous hash and its own SHA-256, giving tamper-evident event ordering. This is integrity/provenance evidence, not digital-signature authenticity.

### Emergency protection semantics
If a RED event occurs while a competitive session contains branches that were never started and therefore cannot form a new common exact generation without launching more work, v6.2 does not start new work under a thermal emergency. It retains the previous authoritative generation and records rollback-safe protection status/recovery evidence. This may discard uncommitted in-session progress, but preserves scientific authority and computer safety.

## RC2 — Workspace migration, recovery, provenance and GUI validation

### Workspace schema 3
Navigation remains key-based and now persists schema 3 with a stable v6.2 layout identity.

Migration supports:

- v5.9 positional workspace indexes;
- v6.0/v6.1 keyed schema-2 payloads;
- historical aliases.

Unknown/invalid navigation identities restore conservatively to Dashboard. Migration never bypasses live workflow/governing-policy gates.

### Application-session recovery
A durable integrity-sealed session journal records non-authoritative GUI/session intent:

- workspace identity;
- current experiment ID;
- policy-training-active marker;
- governing policy SHA;
- compute-profile identity.

After an unclean shutdown the application can offer conservative workspace/experiment restoration. The session journal is never used as optimizer exact-resume authority; scientific training/experiment recovery remains governed by the existing authenticated checkpoint, campaign journal and database mechanisms.

### Experiment restoration
Scientific workspace restoration continues to require validated saved configuration, exact saved `PowerFlowOptions`, successful base AC power flow and policy-binding integrity checks. The restored UI payload is passed through the versioned workspace migrator.

### GUI validation
A dependency-light structural GUI/workflow contract verifies canonical workspace order, schema version, key-based navigation and recovery/lock integration even when PyQt6 is unavailable. A real PyQt6 GUI test module is included and must execute without skip on the target-machine release gate.

## Final — Hardware soak and scientific equivalence

### Hardware-soak qualification protocol
`HardwareSoakRunner` and `scripts/validate_hardware_soak.py` implement a protected soak protocol with:

- canonical topology/Safe-80 profile capture;
- actual backend selection;
- deterministic FP64 workload;
- live governor sampling;
- RED protective abort;
- durable hash-chained provenance;
- explicit physical-qualification flag.

The runner **cannot** label a short CPU/CI run as a physical accelerator qualification. Physical qualification requires a real accelerator, the declared minimum duration, completion without protection stop, and target-machine scientific parity.

### Scientific equivalence
The release adds canonical branch scientific-identity and terminal-record comparison utilities. Scheduling/concurrency is permitted to change wall-clock order/device placement only; branch ID, seed, scientific configuration fingerprint, target and committed scientific results remain the equivalence basis.

The existing native v5.9 seeded training-vs-deployed-CALO transition-parity regression remains part of the scientific closure evidence.

## Important validation boundary

The build environment used for this generated repository does not provide physical NVIDIA CUDA, physical Intel XPU, PyQt6 or PYPOWER. Therefore the repository includes the complete qualification implementation and automated gates, but **does not falsely claim** a multi-hour target-laptop hardware certification, complete GUI execution, or physical CPU↔CUDA↔XPU equivalence pass.

Before publication/hardware qualification, run on the intended machine:

1. full PyQt6 GUI/workflow suite;
2. full PYPOWER scientific suite;
3. CPU↔CUDA↔XPU parity battery under the exact experiment formulation;
4. multi-hour `validate_hardware_soak` on each admitted accelerator;
5. queued/concurrent repeated-run scientific-equivalence campaign;
6. thermal/power protective-stop fault injection.
