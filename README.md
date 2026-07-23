# CALO-RPD Studio v6.2.1

**CALO-RPD Studio 6.2.1 — Adaptive Compute Protection, Recovery and Scientific Qualification** completes the planned v6 architecture on top of the scientifically hardened v5.9 core and the v6.0/v6.1 policy-first protected scheduler.

## v6.2.1 dashboard layout patch

The Dashboard now uses a vertically scrollable page body and three dedicated scrollable tabs: **System Readiness**, **Training Queue**, and **Scientific Context**. The three previously stacked dashboard panels no longer compress each other; the hardware/device table keeps a usable minimum height and each tab can scroll independently. Summary metric cards use a 3-column responsive grid rather than one dense 5-card row.


## Canonical workflow

Dashboard → CALO Intelligence → Power System → ORPD Formulation → Algorithms → Portfolio → Robust Scenarios → Experiment → Results/Validation/Publication.

Power System remains locked until a qualified, active, runtime-compatible and integrity-verified CALO governing policy is ready.

## v6.2 final upgrades

### Adaptive compute/thermal governor and staged startup
- One authoritative Safe-80 protection envelope is shared by Dashboard, policy training and experiment admission.
- Green/Amber/Red state machine with hysteresis.
- Live CPU/RAM/accelerator-memory protection; actual temperature/power telemetry is used only when the runtime can obtain it reliably. Missing telemetry is reported as unavailable, never invented.
- Amber blocks new admissions and applies protected pacing where supported.
- Red requests a protective Safe Stop/cancel boundary rather than continuing to add load.
- Competitive training starts branches in staged intervals instead of launching every admitted branch at once.
- Experiment admission is also governor-controlled and staged.
- Hash-chained compute-protection provenance records resource decisions and protection events.

### Workspace migration, recovery and provenance
- Workspace schema 3 uses stable workspace keys and a v6.2 layout identity.
- v5.9 positional and v6.0/v6.1 keyed workspace payloads migrate explicitly.
- Unknown historical navigation identities restore conservatively to Dashboard and still obey live workflow gates.
- Durable application-session recovery journal detects unclean shutdowns without treating UI state as authoritative optimizer state.
- Existing experiment scientific restoration still validates saved configuration, exact PowerFlowOptions and policy binding before unlocking downstream workflow.

### Hardware-soak and scientific-equivalence qualification
- Protected hardware-soak protocol/CLI with durable governor provenance.
- A run is marked physically qualified only when a real accelerator is exercised for the declared qualification duration without a protection stop. Short CI/CPU runs validate the protocol but are never mislabeled as physical hardware certification.
- Scientific-equivalence utilities verify that queue/concurrency changes preserve branch IDs, seeds, scientific configuration fingerprints and targets while allowing wall-clock/device placement to differ.
- Dependency-light GUI contract validation complements the full PyQt6 target-machine GUI suite.

## Safe-80 semantics

Safe-80 is an allocation/protection envelope, not an artificial 80% GPU-utilization cap. The system reserves approximately 20% operating capacity and continuously reassesses admission using the telemetry it can trust. Firmware/driver thermal protection is never overridden.

## Important validation boundary

The source tree contains the complete v6.2 protection, recovery and qualification implementation. Physical multi-hour CUDA/XPU laptop soak, full PyQt6 GUI execution, complete PYPOWER validation and physical CPU↔CUDA↔XPU equivalence must still be executed on the intended target hardware before a publication/hardware qualification certificate is claimed. Build environments without those devices/dependencies must report that limitation rather than fabricate a pass.

## Launch

```bash
python bootstrap.py
```

## Hardware soak qualification

```bash
python -m calo_rpd_studio.scripts.validate_hardware_soak --backend auto --duration-seconds 14400
```

Generated qualification evidence is written under `results_data/hardware_soak/`.

## Release evidence

- `CALO-RPD-v6.2.1_IMPLEMENTATION_REPORT.md`
- `CALO-RPD-v6.2.1_DEEP_POST_GENERATION_AUDIT.txt`
- `FINDINGS_CLOSURE_v6.2.1.csv`
- `calo_rpd_studio/data/frozen/calo_v621_freeze.json`
- `MANIFEST.sha256`
