"""Heterogeneous compute discovery and soft admission control.

CALO-RPD Studio treats independent optimizer runs as schedulable jobs.  A job is assigned to a
compute device before it starts and is never migrated mid-run.  Accelerator-capable CALO jobs are
admitted in priority order:

    NVIDIA CUDA -> Intel XPU -> CPU

The CUDA/XPU labels are PyTorch backend identifiers and do not have to match Windows Task Manager's
``GPU 0``/``GPU 1`` numbering.  For example, a Windows adapter displayed as ``GPU 1`` can still be
``cuda:0`` because CUDA numbers only NVIDIA devices visible to the PyTorch runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil


@dataclass(frozen=True, slots=True)
class DeviceSnapshot:
    """One accelerator device visible to a supported runtime."""

    device_id: str
    backend: str
    index: int
    name: str
    available: bool
    utilization_percent: float | None = None
    memory_percent: float = 0.0
    telemetry: str = ""
    runtime: str = "primary"


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Host and accelerator state sampled for admission decisions.

    Legacy GPU fields are retained as properties so older callers/tests remain compatible; they
    refer to the first CUDA device in priority order.
    """

    cpu_percent: float
    devices: tuple[DeviceSnapshot, ...] = ()
    system_memory_percent: float = 0.0

    def _first_cuda(self) -> DeviceSnapshot | None:
        return next((device for device in self.devices if device.backend == "cuda" and device.available), None)

    @property
    def gpu_available(self) -> bool:
        return self._first_cuda() is not None

    @property
    def gpu_utilization(self) -> float:
        device = self._first_cuda()
        return float(device.utilization_percent or 0.0) if device else 0.0

    @property
    def gpu_memory_percent(self) -> float:
        device = self._first_cuda()
        return float(device.memory_percent) if device else 0.0

    @property
    def gpu_name(self) -> str:
        device = self._first_cuda()
        return device.name if device else ""

    def by_backend(self, backend: str) -> tuple[DeviceSnapshot, ...]:
        return tuple(device for device in self.devices if device.backend == backend and device.available)

    def get(self, device_id: str) -> DeviceSnapshot | None:
        return next((device for device in self.devices if device.device_id == device_id), None)


class ResourceMonitor:
    """Discover CUDA/XPU resources and cache moderately expensive telemetry probes."""

    def __init__(self, xpu_interpreter: str | None = None) -> None:
        psutil.cpu_percent(interval=None)
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._cuda_cache: tuple[DeviceSnapshot, ...] = ()
        self._cuda_cache_time = 0.0
        self._xpu_cache: tuple[DeviceSnapshot, ...] = ()
        self._xpu_cache_time = 0.0
        self._xpu_interpreter = xpu_interpreter or configured_xpu_interpreter()

    @staticmethod
    def torch_cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    @staticmethod
    def torch_xpu_available() -> bool:
        try:
            import torch

            return bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        except Exception:
            return False

    @property
    def xpu_interpreter(self) -> str:
        return str(self._xpu_interpreter or "")

    def _sample_cuda(self) -> tuple[DeviceSnapshot, ...]:
        now = time.monotonic()
        if now - self._cuda_cache_time < 0.5:
            return self._cuda_cache

        snapshots: list[DeviceSnapshot] = []
        if not self.torch_cuda_available():
            self._cuda_cache = ()
            self._cuda_cache_time = now
            return self._cuda_cache

        # Prefer PyTorch's own device enumeration so CUDA_VISIBLE_DEVICES remapping is respected.
        try:
            import torch

            count = int(torch.cuda.device_count())
            for index in range(count):
                name = str(torch.cuda.get_device_name(index))
                utilization: float | None = None
                telemetry = "PyTorch CUDA"
                try:
                    utilization = float(torch.cuda.utilization(index))
                except Exception:
                    pass
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                    memory_percent = 100.0 * (total_bytes - free_bytes) / max(total_bytes, 1)
                except Exception:
                    memory_percent = 0.0
                snapshots.append(
                    DeviceSnapshot(
                        device_id=f"cuda:{index}",
                        backend="cuda",
                        index=index,
                        name=name,
                        available=True,
                        utilization_percent=utilization,
                        memory_percent=float(memory_percent),
                        telemetry=telemetry,
                        runtime="primary",
                    )
                )
        except Exception:
            snapshots = []

        # When utilization is unavailable through PyTorch, supplement it from nvidia-smi.  The
        # ordering usually matches visible CUDA devices on ordinary desktop installations; names
        # are also checked before the sample is accepted.
        if snapshots and self._nvidia_smi and any(item.utilization_percent is None for item in snapshots):
            try:
                result = subprocess.run(
                    [
                        self._nvidia_smi,
                        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=4,
                    check=False,
                    creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
                )
                rows = []
                for line in result.stdout.splitlines():
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 5:
                        rows.append(parts)
                updated: list[DeviceSnapshot] = []
                for snapshot in snapshots:
                    row = rows[snapshot.index] if snapshot.index < len(rows) else None
                    if row and (
                        snapshot.name.lower() in row[1].lower() or row[1].lower() in snapshot.name.lower()
                    ):
                        used, total = float(row[3]), max(float(row[4]), 1.0)
                        updated.append(
                            DeviceSnapshot(
                                snapshot.device_id,
                                snapshot.backend,
                                snapshot.index,
                                snapshot.name,
                                snapshot.available,
                                float(row[2]),
                                100.0 * used / total,
                                "nvidia-smi",
                                snapshot.runtime,
                            )
                        )
                    else:
                        updated.append(snapshot)
                snapshots = updated
            except Exception:
                pass

        self._cuda_cache = tuple(snapshots)
        self._cuda_cache_time = now
        return self._cuda_cache

    @staticmethod
    def _direct_xpu_snapshots() -> tuple[DeviceSnapshot, ...]:
        try:
            import torch

            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                return ()
            snapshots: list[DeviceSnapshot] = []
            count = int(torch.xpu.device_count())
            for index in range(count):
                properties = torch.xpu.get_device_properties(index)
                name = str(getattr(properties, "name", f"Intel XPU {index}"))
                total = int(getattr(properties, "total_memory", 0) or 0)
                memory_percent = 0.0
                try:
                    free_bytes, total_bytes = torch.xpu.memory.mem_get_info(index)
                    memory_percent = 100.0 * (total_bytes - free_bytes) / max(total_bytes, 1)
                except Exception:
                    try:
                        allocated = int(torch.xpu.memory.memory_allocated(index))
                        memory_percent = 100.0 * allocated / max(total, 1) if total else 0.0
                    except Exception:
                        pass
                utilization: float | None = None
                # PyTorch's stable XPU API does not guarantee a utilization-percentage function.
                # Use it opportunistically if a future/runtime-specific build provides one.
                try:
                    utilization_fn = getattr(torch.xpu, "utilization", None)
                    if callable(utilization_fn):
                        utilization = float(utilization_fn(index))
                except Exception:
                    utilization = None
                snapshots.append(
                    DeviceSnapshot(
                        device_id=f"xpu:{index}",
                        backend="xpu",
                        index=index,
                        name=name,
                        available=True,
                        utilization_percent=utilization,
                        memory_percent=float(memory_percent),
                        telemetry="PyTorch XPU" if utilization is not None else "XPU memory + job-cap admission",
                        runtime="primary",
                    )
                )
            return tuple(snapshots)
        except Exception:
            return ()

    def _sidecar_xpu_snapshots(self) -> tuple[DeviceSnapshot, ...]:
        interpreter = self._xpu_interpreter
        if not interpreter or not Path(interpreter).exists():
            return ()
        try:
            result = subprocess.run(
                [interpreter, "-m", "calo_rpd_studio.compute.xpu_worker", "--telemetry"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
            )
            payload = json.loads(result.stdout.strip().splitlines()[-1])
            if not payload.get("available"):
                return ()
            devices = []
            for item in payload.get("devices", []):
                devices.append(
                    DeviceSnapshot(
                        device_id=str(item.get("device_id", "xpu:0")),
                        backend="xpu",
                        index=int(item.get("index", 0)),
                        name=str(item.get("name", "Intel XPU")),
                        available=True,
                        utilization_percent=(
                            float(item["utilization_percent"])
                            if item.get("utilization_percent") is not None
                            else None
                        ),
                        memory_percent=float(item.get("memory_percent", 0.0)),
                        telemetry=str(item.get("telemetry", "XPU sidecar")),
                        runtime="sidecar",
                    )
                )
            return tuple(devices)
        except Exception:
            return ()

    def _sample_xpu(self) -> tuple[DeviceSnapshot, ...]:
        now = time.monotonic()
        # Direct XPU telemetry is inexpensive. A sidecar sample starts a secondary interpreter and
        # imports its hardware-specific PyTorch build, so cache those samples longer and use the
        # explicit active-job cap for responsive admission between telemetry refreshes.
        direct_available = self.torch_xpu_available()
        cache_seconds = 1.0 if direct_available else 10.0
        if now - self._xpu_cache_time < cache_seconds:
            return self._xpu_cache
        direct = self._direct_xpu_snapshots() if direct_available else ()
        self._xpu_cache = direct or self._sidecar_xpu_snapshots()
        self._xpu_cache_time = now
        return self._xpu_cache

    def sample(self) -> ResourceSnapshot:
        cpu = float(psutil.cpu_percent(interval=None))
        ram = float(psutil.virtual_memory().percent)
        # Priority order is encoded by tuple order: all CUDA devices first, then XPU devices.
        devices = (*self._sample_cuda(), *self._sample_xpu())
        return ResourceSnapshot(cpu_percent=cpu, devices=tuple(devices), system_memory_percent=ram)


def configured_xpu_interpreter() -> str:
    """Return a verified secondary XPU-runtime interpreter recorded by the bootstrap wizard."""
    try:
        from calo_bootstrap.prerequisites import STATE_FILE

        if not STATE_FILE.exists():
            return ""
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        report = dict(payload.get("report", {}))
        sidecar = dict(report.get("xpu_sidecar", {}))
        interpreter = str(sidecar.get("interpreter", ""))
        if sidecar.get("xpu_available") and sidecar.get("gpu_test_passed") and Path(interpreter).exists():
            return interpreter
    except Exception:
        pass
    return ""


def item_uses_calo_ai(mode: str, item) -> bool:
    """Return whether a planned item can use accelerator policy inference."""
    if mode == "comparison":
        return str(getattr(item, "label", "")) == "CALO"
    if mode == "ablation":
        spec = getattr(item, "ablation_spec", None)
        if spec is None or str(getattr(spec, "algorithm", "")) != "CALO":
            return False
        parameters = dict(getattr(spec, "parameters", None) or {})
        return bool(parameters.get("use_ai", True))
    return False


def backend_allows_accelerators(execution_backend: str) -> bool:
    return str(execution_backend).lower() in {"weighted_split", "adaptive_hybrid", "gpu_preferred"}


def backend_allows_gpu(execution_backend: str) -> bool:
    """Backward-compatible alias retained for older callers."""
    return backend_allows_accelerators(execution_backend)


def cpu_admission_allowed(
    snapshot: ResourceSnapshot,
    target_percent: float,
    active_cpu_jobs: int,
    memory_limit_percent: float = 85.0,
) -> bool:
    """Soft CPU admission gate with process-count and host-memory safety limits."""
    if snapshot.system_memory_percent >= float(memory_limit_percent):
        return False
    if snapshot.cpu_percent >= float(target_percent):
        return False
    if active_cpu_jobs <= 0:
        return True
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
    target_jobs = max(1, int(round(physical * float(target_percent) / 100.0)))
    return active_cpu_jobs < target_jobs


def accelerator_admission_allowed(
    device: DeviceSnapshot,
    target_percent: float,
    memory_limit_percent: float,
    active_jobs: int,
    max_jobs: int,
) -> bool:
    """Return whether another independent job may be assigned to an accelerator.

    CUDA normally exposes real utilization telemetry.  Stable XPU runtimes may expose memory but no
    utilization percentage.  In that case the explicit per-device concurrency cap is the utilization
    proxy; this is reported transparently in the GUI instead of inventing a false utilization value.
    """
    if not device.available:
        return False
    if active_jobs >= max(1, int(max_jobs)):
        return False
    if device.memory_percent >= float(memory_limit_percent):
        return False
    if device.utilization_percent is not None and device.utilization_percent >= float(target_percent):
        return False
    return True


def gpu_admission_allowed(
    snapshot: ResourceSnapshot,
    target_percent: float,
    memory_limit_percent: float,
    active_gpu_jobs: int,
    max_gpu_jobs: int,
) -> bool:
    """Backward-compatible first-CUDA admission helper."""
    device = next(iter(snapshot.by_backend("cuda")), None)
    return bool(
        device
        and accelerator_admission_allowed(
            device,
            target_percent,
            memory_limit_percent,
            active_gpu_jobs,
            max_gpu_jobs,
        )
    )


def prioritized_accelerators(snapshot: ResourceSnapshot) -> tuple[DeviceSnapshot, ...]:
    """Return accelerators in the default scientific execution priority order."""
    return (*snapshot.by_backend("cuda"), *snapshot.by_backend("xpu"))

@dataclass(frozen=True, slots=True)
class WeightedAllocationSummary:
    """Static backend-share plan for one experiment.

    Shares are applied to accelerator-compatible jobs. CPU-only algorithms are always assigned to
    CPU and are reported separately so the GUI never implies that a CPU implementation is running
    on a GPU merely because a scheduler quota exists.
    """

    total_jobs: int
    accelerator_eligible_jobs: int
    cpu_only_jobs: int
    cuda_jobs: int
    xpu_jobs: int
    cpu_eligible_jobs: int
    total_cpu_jobs: int
    cuda_share: int
    xpu_share: int
    cpu_share: int
    cuda_available: bool
    xpu_available: bool

    @property
    def requested_text(self) -> str:
        return f"CUDA {self.cuda_share}% · XPU {self.xpu_share}% · CPU {self.cpu_share}%"

    @property
    def effective_text(self) -> str:
        return (
            f"CUDA {self.cuda_jobs} · XPU {self.xpu_jobs} · CPU {self.total_cpu_jobs} "
            f"({self.cpu_only_jobs} CPU-only + {self.cpu_eligible_jobs} compatible fallback)"
        )


def _largest_remainder_counts(total: int, weighted_lanes: list[tuple[str, int]]) -> dict[str, int]:
    """Allocate an integer total according to percentage weights using largest remainders."""
    total = max(0, int(total))
    positive = [(name, max(0, int(weight))) for name, weight in weighted_lanes if int(weight) > 0]
    if total == 0:
        return {name: 0 for name, _ in weighted_lanes}
    if not positive:
        return {name: (total if name == "cpu" else 0) for name, _ in weighted_lanes}
    weight_sum = sum(weight for _, weight in positive)
    raw = {name: total * weight / weight_sum for name, weight in positive}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        positive,
        key=lambda item: (raw[item[0]] - counts[item[0]], item[1], item[0] == "cuda"),
        reverse=True,
    )
    for index in range(remaining):
        counts[order[index % len(order)][0]] += 1
    return {name: counts.get(name, 0) for name, _ in weighted_lanes}


def build_weighted_lane_plan(
    plan,
    mode: str,
    *,
    cuda_available: bool,
    xpu_available: bool,
    cuda_share: int = 50,
    xpu_share: int = 30,
    cpu_share: int = 20,
) -> tuple[dict[int, str], WeightedAllocationSummary]:
    """Pre-assign experiment jobs to CUDA/XPU/CPU lanes.

    Only jobs whose implementation has accelerator-capable CALO policy inference are eligible for
    CUDA/XPU assignment. Conventional algorithms and the AC power-flow evaluator remain CPU code.
    If an accelerator is unavailable, its requested share is redistributed over the remaining
    available lanes rather than silently assigning an unusable backend.
    """
    items = list(plan)
    eligible = [item for item in items if item_uses_calo_ai(mode, item)]
    cpu_only = [item for item in items if not item_uses_calo_ai(mode, item)]

    lanes: list[tuple[str, int]] = []
    if cuda_available:
        lanes.append(("cuda", int(cuda_share)))
    if xpu_available:
        lanes.append(("xpu", int(xpu_share)))
    lanes.append(("cpu", int(cpu_share)))
    counts = _largest_remainder_counts(len(eligible), lanes)

    assignments: dict[int, str] = {int(item.job_index): "cpu" for item in cpu_only}
    # Interleave lanes deterministically so device choice is not confounded with contiguous seed
    # ranges (for example, all early repeated runs on CUDA and all late runs on CPU).
    remaining = {lane: counts.get(lane, 0) for lane in ("cuda", "xpu", "cpu")}
    assigned = {lane: 0 for lane in remaining}
    lane_sequence: list[str] = []
    eligible_total = max(1, len(eligible))
    for position in range(len(eligible)):
        candidates = [lane for lane, value in remaining.items() if value > 0]
        if not candidates:
            lane_sequence.append("cpu")
            continue
        lane = max(
            candidates,
            key=lambda name: (
                (position + 1) * counts.get(name, 0) / eligible_total - assigned[name],
                counts.get(name, 0),
                name == "cuda",
            ),
        )
        lane_sequence.append(lane)
        remaining[lane] -= 1
        assigned[lane] += 1
    for item, lane in zip(eligible, lane_sequence, strict=False):
        assignments[int(item.job_index)] = lane

    cuda_jobs = sum(1 for lane in assignments.values() if lane == "cuda")
    xpu_jobs = sum(1 for lane in assignments.values() if lane == "xpu")
    cpu_eligible_jobs = sum(
        1 for item in eligible if assignments.get(int(item.job_index), "cpu") == "cpu"
    )
    summary = WeightedAllocationSummary(
        total_jobs=len(items),
        accelerator_eligible_jobs=len(eligible),
        cpu_only_jobs=len(cpu_only),
        cuda_jobs=cuda_jobs,
        xpu_jobs=xpu_jobs,
        cpu_eligible_jobs=cpu_eligible_jobs,
        total_cpu_jobs=len(cpu_only) + cpu_eligible_jobs,
        cuda_share=int(cuda_share),
        xpu_share=int(xpu_share),
        cpu_share=int(cpu_share),
        cuda_available=bool(cuda_available),
        xpu_available=bool(xpu_available),
    )
    return assignments, summary


def weighted_worker_slots(
    total_workers: int,
    summary: WeightedAllocationSummary,
) -> dict[str, int]:
    """Create concurrent lane caps while keeping at least one slot for each non-empty lane."""
    total_workers = max(1, int(total_workers))
    nonempty = [
        ("cuda", summary.cuda_share, summary.cuda_jobs),
        ("xpu", summary.xpu_share, summary.xpu_jobs),
        ("cpu", summary.cpu_share, summary.total_cpu_jobs),
    ]
    active = [(name, weight) for name, weight, jobs in nonempty if jobs > 0]
    if not active:
        return {"cuda": 0, "xpu": 0, "cpu": total_workers}
    counts = _largest_remainder_counts(total_workers, active)
    if total_workers >= len(active):
        for name, _ in active:
            counts[name] = max(1, counts.get(name, 0))
        while sum(counts.values()) > total_workers:
            reducible = max(
                (name for name, _ in active if counts[name] > 1),
                key=lambda name: counts[name],
                default=None,
            )
            if reducible is None:
                break
            counts[reducible] -= 1
    return {"cuda": counts.get("cuda", 0), "xpu": counts.get("xpu", 0), "cpu": counts.get("cpu", 0)}
