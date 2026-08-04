"""Hardware-soak qualification protocol for CALO-RPD v6.2.

The module provides a safe, bounded qualification runner. A result is marked ``physical_qualified``
only when the requested physical backend was actually exercised for the requested duration without a
protection stop. Short CI/simulated runs validate the state machine but are never mislabeled as a
multi-hour hardware certification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np

from calo_rpd_studio.ai.model_io import durable_write_bytes

from .device_lease import ExclusiveDeviceLease
from .governor import AdaptiveComputeGovernor, GovernorConfig, ProtectionState
from .provenance import ComputeProvenanceRecorder
from .topology import ComputeTopologyService, SafeResourceBudgetEngine


@dataclass(frozen=True, slots=True)
class SoakConfig:
    duration_seconds: float = 4 * 3600.0
    sample_interval_seconds: float = 1.0
    backend: str = "auto"
    minimum_physical_qualification_seconds: float = 3600.0
    workload_matrix_size: int = 192

    def validate(self) -> None:
        if not math.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be finite and positive")
        if not math.isfinite(self.sample_interval_seconds) or self.sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be finite and positive")
        if self.workload_matrix_size < 16:
            raise ValueError("workload_matrix_size must be >= 16")


@dataclass(frozen=True, slots=True)
class SoakResult:
    run_id: str
    source_commit: str
    tracked_source_clean: bool
    started_at: str
    completed_at: str
    requested_backend: str
    exercised_backend: str
    duration_seconds: float
    samples: int
    protection_terminal_state: str
    protection_stop: bool
    deterministic_checksum: str
    topology_fingerprint: str
    profile_fingerprint: str
    provenance_path: str
    provenance_verification: dict
    telemetry_summary: dict
    physical_qualified: bool
    qualification_reason: str
    claim_scope: str

    def to_dict(self) -> dict:
        return asdict(self)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_backend(requested: str) -> str:
    text = str(requested or "auto").lower()
    try:
        import torch

        if text in {"auto", "cuda"} and torch.cuda.is_available():
            return "cuda:0"
    except (ImportError, RuntimeError, AttributeError):
        pass
    return "cpu"


def _workload(device: str, size: int) -> str:
    """Perform a deterministic FP64 workload and return a compact checksum."""
    if device == "cpu":
        x = np.arange(size * size, dtype=np.float64).reshape(size, size) / max(size * size, 1)
        y = x @ x.T
        digest = hashlib.sha256(np.asarray(y[:8, :8], dtype=np.float64).tobytes()).hexdigest()
        return digest
    import torch

    target = torch.device(device)
    x = torch.arange(size * size, dtype=torch.float64, device=target).reshape(size, size)
    x = x / float(max(size * size, 1))
    y = x @ x.T
    sample = y[:8, :8].detach().cpu().numpy().astype(np.float64, copy=False)
    if device.startswith("cuda"):
        torch.cuda.synchronize(target)
    return hashlib.sha256(sample.tobytes()).hexdigest()


def _numeric_summary(values: list[float]) -> dict:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "available_samples": len(finite),
        "minimum": min(finite) if finite else None,
        "maximum": max(finite) if finite else None,
        "mean": float(np.mean(finite)) if finite else None,
    }


def _summarize_telemetry(samples: list[dict]) -> dict:
    """Summarize observed sensors without inventing unavailable values."""
    cpu_temperatures: list[float] = []
    device_rows: dict[str, list[dict]] = {}
    for sample in samples:
        snapshot = dict(sample.get("snapshot", {}))
        cpu_temperature = snapshot.get("cpu_temperature_c")
        if cpu_temperature is not None:
            cpu_temperatures.append(float(cpu_temperature))
        sampled_at = float(snapshot.get("sampled_at_monotonic", 0.0) or 0.0)
        for device in snapshot.get("devices", []):
            row = dict(device)
            row["sampled_at_monotonic"] = sampled_at
            device_rows.setdefault(str(row.get("device_id", "unknown")), []).append(row)

    devices: dict[str, dict] = {}
    for device_id, rows in device_rows.items():
        temperatures = [
            float(row["temperature_c"]) for row in rows if row.get("temperature_c") is not None
        ]
        powers = [float(row["power_w"]) for row in rows if row.get("power_w") is not None]
        power_limits = [
            float(row["power_limit_w"]) for row in rows if row.get("power_limit_w") is not None
        ]
        utilization = [
            float(row["utilization_percent"])
            for row in rows
            if row.get("utilization_percent") is not None
        ]
        memory = [float(row.get("memory_percent", 0.0)) for row in rows]
        energy_joules = 0.0
        integrated_seconds = 0.0
        prior = None
        for row in rows:
            if row.get("power_w") is None:
                prior = None
                continue
            current = (
                float(row.get("sampled_at_monotonic", 0.0)),
                float(row["power_w"]),
            )
            if prior is not None:
                elapsed = max(0.0, current[0] - prior[0])
                energy_joules += elapsed * 0.5 * (prior[1] + current[1])
                integrated_seconds += elapsed
            prior = current
        devices[device_id] = {
            "name": str(rows[-1].get("name", "")),
            "backend": str(rows[-1].get("backend", "")),
            "telemetry_sources": sorted(
                {str(row.get("telemetry", "")) for row in rows if row.get("telemetry")}
            ),
            "temperature_c": _numeric_summary(temperatures),
            "power_w": _numeric_summary(powers),
            "power_limit_w": _numeric_summary(power_limits),
            "utilization_percent": _numeric_summary(utilization),
            "memory_percent": _numeric_summary(memory),
            "gpu_board_energy_estimate_joules": float(energy_joules)
            if integrated_seconds > 0
            else None,
            "gpu_board_energy_estimate_wh": float(energy_joules / 3600.0)
            if integrated_seconds > 0
            else None,
            "energy_integration_seconds": float(integrated_seconds),
        }
    return {
        "sample_count": len(samples),
        "cpu_temperature_c": _numeric_summary(cpu_temperatures),
        "devices": devices,
        "energy_scope": (
            "Trapezoidal integration of observed GPU board-power telemetry only; excludes CPU, "
            "display, PSU conversion, battery, and whole-system energy. Missing samples are not imputed."
        ),
    }


class HardwareSoakRunner:
    def __init__(
        self,
        config: SoakConfig,
        *,
        output_dir: str | Path = "results_data/hardware_soak",
        run_id: str = "",
        source_commit: str = "",
        tracked_source_clean: bool = False,
        require_physical_cuda: bool = False,
    ) -> None:
        config.validate()
        self.config = config
        self.output_dir = Path(output_dir)
        self.run_id = str(run_id).strip()
        if self.run_id and not re.fullmatch(r"[A-Za-z0-9._-]+", self.run_id):
            raise ValueError(
                "soak run ID may contain only letters, digits, dot, underscore, and dash"
            )
        self.source_commit = str(source_commit).strip().lower()
        self.tracked_source_clean = bool(tracked_source_clean)
        self.require_physical_cuda = bool(require_physical_cuda)

    def run(self, progress: Callable[[dict], None] | None = None) -> SoakResult:
        topology_service = ComputeTopologyService()
        topology = topology_service.scan()
        profile = SafeResourceBudgetEngine(allocation_limit_fraction=0.80).calculate(topology)
        governor = AdaptiveComputeGovernor(
            profile,
            monitor=topology_service.monitor,
            config=GovernorConfig(allocation_limit_fraction=0.80),
        )
        backend = _resolve_backend(self.config.backend)
        if self.require_physical_cuda and not backend.startswith("cuda"):
            raise RuntimeError("Physical CUDA soak was required but CUDA was not exercised")
        session_id = (
            self.run_id or hashlib.sha256(f"{time.time_ns()}:{backend}".encode()).hexdigest()[:20]
        )
        provenance_path = self.output_dir / f"soak_{session_id}.jsonl"
        result_path = self.output_dir / f"soak_{session_id}.json"
        if provenance_path.exists() or result_path.exists():
            raise FileExistsError(f"Refusing to overwrite hardware-soak evidence: {session_id}")
        recorder = ComputeProvenanceRecorder(
            provenance_path,
            session_id=session_id,
            metadata={
                "run_id": self.run_id,
                "source_commit": self.source_commit,
                "tracked_source_clean": self.tracked_source_clean,
                "config": asdict(self.config),
                "topology": topology.to_dict(),
                "profile": profile.to_dict(),
                "exercised_backend": backend,
                "require_physical_cuda": self.require_physical_cuda,
            },
        )
        started_at = _utc()
        start = time.monotonic()
        next_sample = start
        samples = 0
        terminal = ProtectionState.UNKNOWN
        stopped = False
        checksum = ""
        telemetry_samples: list[dict] = []
        lease = ExclusiveDeviceLease(backend) if backend.startswith("cuda") else None
        try:
            while True:
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= self.config.duration_seconds:
                    break
                if now >= next_sample:
                    decision = governor.sample(active_branches=1)
                    decision_payload = decision.to_dict()
                    terminal = decision.state
                    telemetry_samples.append(decision_payload)
                    recorder.append("GOVERNOR_SAMPLE", decision_payload)
                    samples += 1
                    if progress is not None:
                        progress(
                            {
                                "elapsed_seconds": elapsed,
                                "decision": decision_payload,
                                "backend": backend,
                            }
                        )
                    if decision.request_safe_stop or decision.state is ProtectionState.RED:
                        stopped = True
                        recorder.append("PROTECTIVE_STOP", {"reasons": list(decision.reasons)})
                        break
                    next_sample = now + self.config.sample_interval_seconds
                checksum = _workload(backend, self.config.workload_matrix_size)
                if terminal is ProtectionState.AMBER:
                    time.sleep(max(0.01, governor.config.amber_pause_seconds))
        finally:
            if lease is not None:
                lease.close()
        duration = time.monotonic() - start
        physical_candidate = (
            backend != "cpu"
            and not stopped
            and duration >= float(self.config.minimum_physical_qualification_seconds)
            and duration >= float(self.config.duration_seconds) * 0.99
        )
        if stopped:
            reason = (
                "Protection governor requested Safe Stop before qualification duration completed."
            )
        elif backend == "cpu":
            reason = "CPU/software soak completed; no physical CUDA accelerator was exercised."
        elif duration < self.config.minimum_physical_qualification_seconds:
            reason = "Short qualification run completed; duration is below the declared physical-soak minimum."
        else:
            reason = "Requested physical accelerator soak completed inside the protection envelope."
        telemetry_summary = _summarize_telemetry(telemetry_samples)
        recorder.append(
            "SOAK_TERMINAL_OBSERVED",
            {
                "duration_seconds": float(duration),
                "samples": int(samples),
                "protection_terminal_state": terminal.value,
                "protection_stop": bool(stopped),
                "physical_candidate": bool(physical_candidate),
                "deterministic_checksum": checksum,
                "telemetry_summary": telemetry_summary,
            },
        )
        verification = ComputeProvenanceRecorder.verify(provenance_path)
        physical = bool(physical_candidate and verification.get("ok"))
        claim_scope = (
            "physical CUDA soak on this source/device/duration with hash-chained protection, "
            "temperature, GPU board-power, and scoped GPU board-energy telemetry"
            if physical
            else "no physical hardware-soak qualification claim"
        )
        result = SoakResult(
            run_id=self.run_id,
            source_commit=self.source_commit,
            tracked_source_clean=self.tracked_source_clean,
            started_at=started_at,
            completed_at=_utc(),
            requested_backend=str(self.config.backend),
            exercised_backend=backend,
            duration_seconds=float(duration),
            samples=int(samples),
            protection_terminal_state=terminal.value,
            protection_stop=bool(stopped),
            deterministic_checksum=checksum,
            topology_fingerprint=topology.fingerprint,
            profile_fingerprint=profile.profile_fingerprint,
            provenance_path=str(provenance_path),
            provenance_verification=verification,
            telemetry_summary=telemetry_summary,
            physical_qualified=bool(physical),
            qualification_reason=reason,
            claim_scope=claim_scope,
        )
        durable_write_bytes(
            result_path,
            (json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            ),
        )
        return result


__all__ = ["SoakConfig", "SoakResult", "HardwareSoakRunner", "_summarize_telemetry"]
