"""Authoritative CPU/XPU/GPU topology and safe-resource profiling for v6.2 protected scheduling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
from typing import Iterable

import psutil

from .resource_scheduler import ResourceMonitor

CPU_SUPPORT_RESERVE_PER_BRANCH = 2
MINIMUM_CPU_ROLLOUT_WORKERS_PER_BRANCH = 1
MINIMUM_CPU_WORKER_EQUIVALENTS_PER_BRANCH = (
    CPU_SUPPORT_RESERVE_PER_BRANCH + MINIMUM_CPU_ROLLOUT_WORKERS_PER_BRANCH
)


def _normalise_name(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    for token in ("corporation", "graphics", "adapter", "family", "series"):
        text = text.replace(token, " ")
    return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class ComputeDevice:
    physical_id: str
    os_label: str
    runtime_id: str
    backend: str
    runtime: str
    name: str
    memory_total_bytes: int
    memory_used_percent: float
    utilization_percent: float | None
    telemetry: str
    ppo_learner: bool
    policy_actor: bool
    orpd_evaluator: bool
    full_training_branch: bool
    capability_status: str = "validated"
    capability_detail: str = ""

    @property
    def mapping_text(self) -> str:
        left = self.os_label or self.physical_id
        return f"{left} → {self.runtime_id}" if self.runtime_id else left


@dataclass(frozen=True, slots=True)
class ComputeTopologySnapshot:
    cpu_name: str
    physical_cores: int
    logical_threads: int
    ram_total_bytes: int
    ram_used_percent: float
    devices: tuple[ComputeDevice, ...]
    platform_name: str
    fingerprint: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["devices"] = [asdict(device) for device in self.devices]
        return payload


@dataclass(frozen=True, slots=True)
class ComputeProtectionProfile:
    profile_name: str
    allocation_limit_fraction: float
    reserve_fraction: float
    ready: bool
    status: str
    safe_cpu_worker_budget: int
    safe_ram_ceiling_bytes: int
    safe_parallel_branches: int
    accelerator_branch_slots: int
    cpu_support_branch_capacity: int
    ram_branch_capacity: int
    estimated_branch_ram_bytes: int
    estimated_branch_accelerator_bytes: int
    topology_fingerprint: str
    profile_fingerprint: str
    reasons: tuple[str, ...]

    @property
    def allocation_percent(self) -> int:
        return int(round(self.allocation_limit_fraction * 100.0))

    @property
    def reserve_percent(self) -> int:
        return int(round(self.reserve_fraction * 100.0))

    def to_dict(self) -> dict:
        return asdict(self)


class ComputeTopologyService:
    """Discover hardware once and expose one canonical device map to the application."""

    def __init__(self, monitor: ResourceMonitor | None = None) -> None:
        self.monitor = monitor or ResourceMonitor()

    @staticmethod
    def _cpu_name() -> str:
        name = platform.processor().strip()
        if name:
            return name
        if os.name == "nt":
            return os.environ.get("PROCESSOR_IDENTIFIER", "CPU")
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
            for line in text.splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
        return "CPU"

    @staticmethod
    def _windows_adapters() -> list[dict]:
        if os.name != "nt":
            return []
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        if not powershell:
            return []
        script = (
            "Get-CimInstance Win32_VideoController | "
            "Select-Object Name,AdapterRAM,PNPDeviceID | ConvertTo-Json -Compress"
        )
        try:
            result = subprocess.run(
                [powershell, "-NoProfile", "-Command", script],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode != 0 or not result.stdout.strip():
                return []
            payload = json.loads(result.stdout.strip())
            return [payload] if isinstance(payload, dict) else [dict(row) for row in payload]
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, TypeError, ValueError):
            return []

    @staticmethod
    def _match_adapter(name: str, adapters: Iterable[dict], used: set[int]) -> tuple[str, int]:
        target = _normalise_name(name)
        best_index = -1
        best_score = -1
        for index, row in enumerate(adapters):
            if index in used:
                continue
            candidate = _normalise_name(str(row.get("Name", "")))
            tokens = set(target.split()) & set(candidate.split())
            vendor_bonus = 0
            for vendor in ("nvidia", "intel", "amd", "radeon"):
                if vendor in target and vendor in candidate:
                    vendor_bonus = 4
            score = len(tokens) + vendor_bonus
            if score > best_score:
                best_score = score
                best_index = index
        if best_index < 0 or best_score <= 0:
            return "", -1
        used.add(best_index)
        return str(adapters[best_index].get("Name", "")), best_index

    @staticmethod
    def _runtime_memory_total(device_id: str) -> int:
        try:
            import torch

            if device_id.startswith("cuda") and torch.cuda.is_available():
                index = int(device_id.split(":", 1)[1])
                return int(torch.cuda.get_device_properties(index).total_memory)
            if device_id.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
                index = int(device_id.split(":", 1)[1])
                return int(getattr(torch.xpu.get_device_properties(index), "total_memory", 0) or 0)
        except (ImportError, RuntimeError, ValueError, AttributeError, OSError):
            return 0
        return 0

    @staticmethod
    def _fp64_runtime_smoke(device_id: str) -> tuple[bool, str]:
        """Run a tiny deterministic FP64 capability probe required by scientific ORPD branches."""
        try:
            import torch

            device = torch.device(device_id)
            a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64, device=device)
            b = a @ a.T
            value = float(b.sum().detach().cpu().item())
            if not math.isfinite(value) or abs(value - 52.0) > 1e-10:
                return False, f"FP64 smoke produced unexpected result {value!r}"
            if device_id.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(device)
            elif device_id.startswith("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.synchronize(device)
            return True, "FP64 tensor/matmul smoke passed"
        except (ImportError, RuntimeError, ValueError, AttributeError, OSError) as exc:
            return False, f"FP64 runtime smoke failed: {type(exc).__name__}: {exc}"

    def scan(self) -> ComputeTopologySnapshot:
        resource = self.monitor.sample()
        adapters = self._windows_adapters()
        used_adapters: set[int] = set()
        devices: list[ComputeDevice] = []
        for snapshot in resource.devices:
            os_name, os_index = self._match_adapter(snapshot.name, adapters, used_adapters)
            if os_index >= 0:
                os_label = f"OS adapter {os_index} — {os_name}"
                physical_id = str(adapters[os_index].get("PNPDeviceID", "") or f"os-gpu:{os_index}")
            else:
                os_label = f"OS adapter — {snapshot.name}"
                physical_id = f"runtime:{snapshot.runtime}:{snapshot.device_id}"
            direct = snapshot.runtime == "primary"
            cuda = snapshot.backend == "cuda"
            xpu = snapshot.backend == "xpu"
            if direct and (cuda or xpu):
                fp64_ok, capability_detail = self._fp64_runtime_smoke(snapshot.device_id)
                capability_status = "FP64 scientific branch validated" if fp64_ok else "restricted"
            elif xpu and snapshot.runtime == "sidecar":
                # configured_xpu_interpreter() is returned only after the bootstrap sidecar GPU smoke
                # passes. Sidecar capability is therefore explicit but intentionally limited to
                # actor/evaluator assistance until a full independent PPO-branch runtime contract exists.
                fp64_ok = True
                capability_status = "sidecar actor/evaluator validated"
                capability_detail = "Bootstrap XPU GPU smoke passed; full independent PPO branch not certified"
            else:
                fp64_ok = False
                capability_status = "detected"
                capability_detail = "No accelerator scientific capability probe"
            devices.append(
                ComputeDevice(
                    physical_id=physical_id,
                    os_label=os_label,
                    runtime_id=snapshot.device_id,
                    backend=snapshot.backend,
                    runtime=snapshot.runtime,
                    name=snapshot.name,
                    memory_total_bytes=self._runtime_memory_total(snapshot.device_id),
                    memory_used_percent=float(snapshot.memory_percent),
                    utilization_percent=snapshot.utilization_percent,
                    telemetry=snapshot.telemetry,
                    ppo_learner=bool((cuda or xpu) and direct),
                    policy_actor=bool(cuda or xpu),
                    orpd_evaluator=bool((cuda and fp64_ok) or (xpu and fp64_ok)),
                    full_training_branch=bool(direct and (cuda or xpu) and fp64_ok),
                    capability_status=capability_status,
                    capability_detail=capability_detail,
                )
            )
        memory = psutil.virtual_memory()
        physical = int(psutil.cpu_count(logical=False) or 1)
        logical = int(psutil.cpu_count(logical=True) or physical)
        identity = {
            "cpu": self._cpu_name(),
            "physical": physical,
            "logical": logical,
            "ram": int(memory.total),
            "devices": [
                {
                    "physical_id": d.physical_id,
                    "runtime_id": d.runtime_id,
                    "backend": d.backend,
                    "runtime": d.runtime,
                    "name": d.name,
                    "full_training_branch": d.full_training_branch,
                    "capability_status": d.capability_status,
                }
                for d in devices
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ComputeTopologySnapshot(
            cpu_name=self._cpu_name(),
            physical_cores=physical,
            logical_threads=logical,
            ram_total_bytes=int(memory.total),
            ram_used_percent=float(memory.percent),
            devices=tuple(devices),
            platform_name=f"{platform.system()} {platform.release()}",
            fingerprint=fingerprint,
        )


class SafeResourceBudgetEngine:
    """Calculate the v6.2 Safe-80 allocation ceiling and protected simultaneous-branch limit."""

    def __init__(
        self,
        *,
        allocation_limit_fraction: float = 0.80,
        estimated_branch_ram_bytes: int = 2 * 1024**3,
        estimated_branch_accelerator_bytes: int = 2 * 1024**3,
        minimum_cpu_workers_per_branch: int = MINIMUM_CPU_WORKER_EQUIVALENTS_PER_BRANCH,
    ) -> None:
        limit = float(allocation_limit_fraction)
        if not math.isfinite(limit) or not 0.50 <= limit <= 0.90:
            raise ValueError("allocation_limit_fraction must be finite and between 0.50 and 0.90")
        self.limit = limit
        self.estimated_branch_ram_bytes = max(512 * 1024**2, int(estimated_branch_ram_bytes))
        self.estimated_branch_accelerator_bytes = max(256 * 1024**2, int(estimated_branch_accelerator_bytes))
        self.minimum_cpu_workers_per_branch = max(1, int(minimum_cpu_workers_per_branch))

    def calculate(self, topology: ComputeTopologySnapshot) -> ComputeProtectionProfile:
        reserve = 1.0 - self.limit
        safe_workers = max(1, int(math.floor(topology.logical_threads * self.limit)))
        # Explicitly retain at least one logical thread for OS/UI/driver work on very small systems.
        if topology.logical_threads > 1:
            safe_workers = min(safe_workers, topology.logical_threads - 1)
        safe_ram_ceiling = int(topology.ram_total_bytes * self.limit)
        used_ram = int(topology.ram_total_bytes * topology.ram_used_percent / 100.0)
        allocatable_ram = max(0, safe_ram_ceiling - used_ram)
        ram_capacity = int(allocatable_ram // self.estimated_branch_ram_bytes)
        cpu_capacity = int(safe_workers // self.minimum_cpu_workers_per_branch)

        accelerator_slots = 0
        full_branch_devices = 0
        for device in topology.devices:
            if not device.full_training_branch:
                continue
            full_branch_devices += 1
            used_percent = float(device.memory_used_percent)
            if used_percent >= self.limit * 100.0:
                continue
            # Safe-80 rule: at most one independent branch per validated accelerator in v6.2.  When
            # total device memory is measurable, also require the estimated branch working set to
            # fit below the Safe-80 memory ceiling rather than using percentage alone.
            if int(device.memory_total_bytes) > 0:
                total = int(device.memory_total_bytes)
                safe_ceiling = int(total * self.limit)
                used = int(total * used_percent / 100.0)
                headroom = max(0, safe_ceiling - used)
                if headroom < self.estimated_branch_accelerator_bytes:
                    continue
            accelerator_slots += 1

        # Do not automatically spill to CPU merely because a detected accelerator is currently full
        # or above the Safe-80 memory envelope. CPU fallback is permitted only on a genuinely CPU-only
        # topology (or when the user explicitly selects CPU later in the training UI).
        compute_slots = accelerator_slots if full_branch_devices > 0 else (1 if cpu_capacity >= 1 else 0)
        safe_parallel = min(compute_slots, cpu_capacity, ram_capacity) if compute_slots else 0
        reasons: list[str] = []
        if topology.ram_used_percent >= self.limit * 100.0:
            reasons.append(
                f"System RAM is already {topology.ram_used_percent:.1f}% used, above the Safe-{int(self.limit*100)} allocation envelope."
            )
        if cpu_capacity < 1:
            reasons.append("Insufficient reserved CPU support capacity for a training branch.")
        if ram_capacity < 1:
            reasons.append("Insufficient RAM headroom inside the Safe-80 envelope for a training branch.")
        if accelerator_slots == 0:
            if full_branch_devices > 0:
                reasons.append("Validated accelerator hardware exists, but no device has sufficient Safe-80 admission headroom; automatic CPU spillover is blocked.")
            else:
                reasons.append("No validated full-branch accelerator is available; protected auto mode permits conservative CPU-only scheduling.")
        ready = safe_parallel >= 1
        status = "READY" if ready else "PROTECTED / NOT READY FOR TRAINING"
        payload = {
            "topology": topology.fingerprint,
            "limit": self.limit,
            "workers": safe_workers,
            "ram_ceiling": safe_ram_ceiling,
            "parallel": safe_parallel,
            "accelerator_slots": accelerator_slots,
            "ram_capacity": ram_capacity,
            "cpu_capacity": cpu_capacity,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ComputeProtectionProfile(
            profile_name=f"Safe {int(round(self.limit * 100))}%",
            allocation_limit_fraction=self.limit,
            reserve_fraction=reserve,
            ready=ready,
            status=status,
            safe_cpu_worker_budget=safe_workers,
            safe_ram_ceiling_bytes=safe_ram_ceiling,
            safe_parallel_branches=max(0, int(safe_parallel)),
            accelerator_branch_slots=max(0, int(accelerator_slots)),
            cpu_support_branch_capacity=max(0, int(cpu_capacity)),
            ram_branch_capacity=max(0, int(ram_capacity)),
            estimated_branch_ram_bytes=self.estimated_branch_ram_bytes,
            estimated_branch_accelerator_bytes=self.estimated_branch_accelerator_bytes,
            topology_fingerprint=topology.fingerprint,
            profile_fingerprint=fingerprint,
            reasons=tuple(reasons),
        )
