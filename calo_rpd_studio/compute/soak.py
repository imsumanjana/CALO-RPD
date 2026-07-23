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
import time
from typing import Callable

import numpy as np

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
    physical_qualified: bool
    qualification_reason: str

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
        if text in {"auto", "xpu"} and hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu:0"
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
    elif device.startswith("xpu") and hasattr(torch, "xpu"):
        torch.xpu.synchronize(target)
    return hashlib.sha256(sample.tobytes()).hexdigest()


class HardwareSoakRunner:
    def __init__(self, config: SoakConfig, *, output_dir: str | Path = "results_data/hardware_soak") -> None:
        config.validate()
        self.config = config
        self.output_dir = Path(output_dir)

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
        session_id = hashlib.sha256(f"{time.time_ns()}:{backend}".encode()).hexdigest()[:20]
        provenance_path = self.output_dir / f"soak_{session_id}.jsonl"
        recorder = ComputeProvenanceRecorder(
            provenance_path,
            session_id=session_id,
            metadata={
                "config": asdict(self.config),
                "topology": topology.to_dict(),
                "profile": profile.to_dict(),
                "exercised_backend": backend,
            },
        )
        started_at = _utc()
        start = time.monotonic()
        next_sample = start
        samples = 0
        terminal = ProtectionState.UNKNOWN
        stopped = False
        checksum = ""
        while True:
            now = time.monotonic()
            elapsed = now - start
            if elapsed >= self.config.duration_seconds:
                break
            if now >= next_sample:
                decision = governor.sample(active_branches=1)
                terminal = decision.state
                recorder.append("GOVERNOR_SAMPLE", decision.to_dict())
                samples += 1
                if progress is not None:
                    progress({"elapsed_seconds": elapsed, "decision": decision.to_dict(), "backend": backend})
                if decision.request_safe_stop or decision.state is ProtectionState.RED:
                    stopped = True
                    recorder.append("PROTECTIVE_STOP", {"reasons": list(decision.reasons)})
                    break
                next_sample = now + self.config.sample_interval_seconds
            checksum = _workload(backend, self.config.workload_matrix_size)
            if terminal is ProtectionState.AMBER:
                time.sleep(max(0.01, governor.config.amber_pause_seconds))
        duration = time.monotonic() - start
        physical = (
            backend != "cpu"
            and not stopped
            and duration >= float(self.config.minimum_physical_qualification_seconds)
            and duration >= float(self.config.duration_seconds) * 0.99
        )
        if stopped:
            reason = "Protection governor requested Safe Stop before qualification duration completed."
        elif backend == "cpu":
            reason = "CPU/software soak completed; no physical CUDA/XPU accelerator was exercised."
        elif duration < self.config.minimum_physical_qualification_seconds:
            reason = "Short qualification run completed; duration is below the declared physical-soak minimum."
        else:
            reason = "Requested physical accelerator soak completed inside the protection envelope."
        result = SoakResult(
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
            physical_qualified=bool(physical),
            qualification_reason=reason,
        )
        recorder.append("SOAK_TERMINAL", result.to_dict())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / f"soak_{session_id}.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return result


__all__ = ["SoakConfig", "SoakResult", "HardwareSoakRunner"]
