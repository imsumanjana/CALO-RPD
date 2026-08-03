"""CUDA/CPU compute discovery and soft admission control.

CALO-RPD Studio treats independent optimizer runs as schedulable jobs.  A job is assigned to a
compute device before it starts and is never migrated mid-run.  Accelerator-compatible optimizer jobs are admitted in priority order:

    NVIDIA CUDA -> CPU

CUDA labels are PyTorch backend identifiers and do not have to match Windows Task Manager's
``GPU 0``/``GPU 1`` numbering.  For example, a Windows adapter displayed as ``GPU 1`` can still be
``cuda:0`` because CUDA numbers only NVIDIA devices visible to the PyTorch runtime.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
import os
import shutil
import subprocess
import time
import threading

import psutil

from calo_rpd_studio.compute.memory_budget import calculate_available_memory_admission


_LOG = logging.getLogger(__name__)
_WARNED_TELEMETRY_KEYS: set[str] = set()
_WARN_LOCK = threading.Lock()


def _warn_once(key: str, message: str, *args) -> None:
    """Rate-limit optional telemetry warnings across all ResourceMonitor instances."""
    with _WARN_LOCK:
        if key in _WARNED_TELEMETRY_KEYS:
            return
        _WARNED_TELEMETRY_KEYS.add(key)
    _LOG.warning(message, *args)


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
    temperature_c: float | None = None
    thermal_limit_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    throttle_reason: str = ""
    memory_total_bytes: int = 0
    memory_available_bytes: int = 0
    hardware_uuid: str = ""
    pci_bus_id: str = ""
    vendor_id: str = ""
    product_id: str = ""
    driver_version: str = ""
    runtime_version: str = ""
    fp64_test_passed: bool | None = None


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Host and accelerator state sampled for admission decisions.

    Legacy GPU fields are retained as properties so older callers/tests remain compatible; they
    refer to the first CUDA device in priority order.
    """

    cpu_percent: float
    devices: tuple[DeviceSnapshot, ...] = ()
    system_memory_percent: float = 0.0
    system_memory_total_bytes: int = 0
    system_memory_available_bytes: int = 0
    cpu_temperature_c: float | None = None
    sampled_at_monotonic: float = 0.0

    def _first_cuda(self) -> DeviceSnapshot | None:
        return next(
            (device for device in self.devices if device.backend == "cuda" and device.available),
            None,
        )

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
        return tuple(
            device for device in self.devices if device.backend == backend and device.available
        )

    def get(self, device_id: str) -> DeviceSnapshot | None:
        return next((device for device in self.devices if device.device_id == device_id), None)


class ResourceMonitor:
    """Discover CUDA resources and cache moderately expensive telemetry probes."""

    def __init__(self) -> None:
        psutil.cpu_percent(interval=None)
        self._nvidia_smi = shutil.which("nvidia-smi")
        self._cuda_cache: tuple[DeviceSnapshot, ...] = ()
        self._cuda_cache_time = 0.0

    @staticmethod
    def torch_cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except (ImportError, RuntimeError, AttributeError, OSError):
            return False

    def _sample_cuda(self) -> tuple[DeviceSnapshot, ...]:
        now = time.monotonic()
        if now - self._cuda_cache_time < 0.5:
            return self._cuda_cache

        snapshots: list[DeviceSnapshot] = []
        if not self.torch_cuda_available():
            self._cuda_cache = ()
            self._cuda_cache_time = now
            return self._cuda_cache

        # CUDA compute discovery is authoritative and independent from optional NVML telemetry.
        # A missing nvidia-ml-py package must never make an otherwise usable CUDA device disappear.
        try:
            import torch

            count = int(torch.cuda.device_count())
            for index in range(count):
                try:
                    properties = torch.cuda.get_device_properties(index)
                    name = str(torch.cuda.get_device_name(index))
                    total_memory = int(getattr(properties, "total_memory", 0) or 0)
                    free_bytes = 0
                    hardware_uuid = str(getattr(properties, "uuid", "") or "")
                    pci_bus_id = str(getattr(properties, "pci_bus_id", "") or "")
                    utilization: float | None = None
                    telemetry_parts = ["PyTorch CUDA"]
                    try:
                        utilization = float(torch.cuda.utilization(index))
                    except (
                        ImportError,
                        AttributeError,
                        RuntimeError,
                        OSError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        _warn_once(
                            f"cuda-util:{type(exc).__name__}:{exc}",
                            "CUDA utilization telemetry unavailable; CUDA compute remains enabled: %s",
                            exc,
                        )
                    try:
                        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
                        total_memory = int(total_bytes or total_memory)
                        memory_percent = 100.0 * (total_bytes - free_bytes) / max(total_bytes, 1)
                    except (
                        ImportError,
                        AttributeError,
                        RuntimeError,
                        OSError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        _warn_once(
                            f"cuda-mem:{type(exc).__name__}:{exc}",
                            "CUDA memory telemetry unavailable; CUDA compute remains enabled: %s",
                            exc,
                        )
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
                            telemetry=" + ".join(telemetry_parts),
                            runtime="primary",
                            memory_total_bytes=total_memory,
                            memory_available_bytes=int(free_bytes),
                            hardware_uuid=hardware_uuid,
                            pci_bus_id=pci_bus_id,
                            vendor_id="10DE",
                            runtime_version=str(getattr(torch.version, "cuda", "") or ""),
                        )
                    )
                except (RuntimeError, AttributeError, OSError, TypeError, ValueError) as exc:
                    _warn_once(
                        f"cuda-device-enum:{index}:{type(exc).__name__}:{exc}",
                        "CUDA device cuda:%s could not be enumerated: %s",
                        index,
                        exc,
                    )
        except (ImportError, RuntimeError, AttributeError, OSError, TypeError, ValueError) as exc:
            _warn_once(
                f"cuda-runtime-enum:{type(exc).__name__}:{exc}",
                "CUDA runtime enumeration failed: %s",
                exc,
            )

        # Supplement each already-discovered CUDA runtime device with nvidia-smi telemetry. Match by
        # stable UUID/PCI identity when available; only use name/index as conservative fallbacks.
        if snapshots and self._nvidia_smi:
            rows: list[dict[str, str]] = []
            queries = [
                (
                    "uuid,pci.bus_id,pci.device_id,index,name,utilization.gpu,memory.used,memory.total,"
                    "temperature.gpu,power.draw,power.limit,driver_version",
                    (
                        "uuid",
                        "pci_bus_id",
                        "pci_device_id",
                        "index",
                        "name",
                        "utilization",
                        "memory_used",
                        "memory_total",
                        "temperature",
                        "power",
                        "power_limit",
                        "driver_version",
                    ),
                ),
                (
                    "uuid,pci.bus_id,pci.device_id,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    (
                        "uuid",
                        "pci_bus_id",
                        "pci_device_id",
                        "index",
                        "name",
                        "utilization",
                        "memory_used",
                        "memory_total",
                        "temperature",
                    ),
                ),
                (
                    "uuid,pci.bus_id,pci.device_id,index,name,utilization.gpu,memory.used,memory.total",
                    (
                        "uuid",
                        "pci_bus_id",
                        "pci_device_id",
                        "index",
                        "name",
                        "utilization",
                        "memory_used",
                        "memory_total",
                    ),
                ),
                (
                    "uuid,pci.bus_id,index,name,utilization.gpu,memory.used,memory.total",
                    (
                        "uuid",
                        "pci_bus_id",
                        "index",
                        "name",
                        "utilization",
                        "memory_used",
                        "memory_total",
                    ),
                ),
            ]
            for query, keys in queries:
                try:
                    result = subprocess.run(
                        [
                            self._nvidia_smi,
                            f"--query-gpu={query}",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=4,
                        check=False,
                        creationflags=(
                            getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                        ),
                    )
                    if result.returncode != 0:
                        continue
                    candidate = []
                    for line in result.stdout.splitlines():
                        if not line.strip():
                            continue
                        parts = [part.strip() for part in line.split(",")]
                        candidate.append(
                            {key: parts[i] if i < len(parts) else "" for i, key in enumerate(keys)}
                        )
                    if candidate:
                        rows = candidate
                        break
                except (OSError, subprocess.SubprocessError) as exc:
                    _warn_once(
                        f"nvidia-smi:{type(exc).__name__}:{exc}",
                        "nvidia-smi telemetry unavailable; CUDA compute remains enabled: %s",
                        exc,
                    )

            def _optional_float(raw: str | None) -> float | None:
                text = str(raw or "").strip()
                if not text or text.lower() in {"n/a", "na", "not supported", "[not supported]"}:
                    return None
                try:
                    return float(text)
                except (TypeError, ValueError):
                    return None

            def _norm_pci(value: str) -> str:
                return (
                    str(value or "").strip().lower().replace("00000000:", "").replace("0000:", "")
                )

            def _row_for(snapshot: DeviceSnapshot) -> dict[str, str] | None:
                if snapshot.hardware_uuid:
                    for row in rows:
                        if (
                            row.get("uuid", "").strip().lower()
                            == snapshot.hardware_uuid.strip().lower()
                        ):
                            return row
                if snapshot.pci_bus_id:
                    target = _norm_pci(snapshot.pci_bus_id)
                    for row in rows:
                        if _norm_pci(row.get("pci_bus_id", "")) == target:
                            return row
                name_matches = [
                    row
                    for row in rows
                    if snapshot.name.lower() in row.get("name", "").lower()
                    or row.get("name", "").lower() in snapshot.name.lower()
                ]
                if len(name_matches) == 1:
                    return name_matches[0]
                # Last-resort fallback is allowed only when there is a single NVIDIA device, avoiding
                # CUDA_VISIBLE_DEVICES/index-reordering mis-association on multi-GPU systems.
                return rows[0] if len(rows) == 1 and len(snapshots) == 1 else None

            def _nvidia_product_id(raw: str) -> str:
                text = str(raw or "").strip().lower().replace("0x", "")
                # nvidia-smi commonly reports PCI device ID as DDDDVVVV (device then vendor).
                if len(text) >= 8 and all(ch in "0123456789abcdef" for ch in text[:8]):
                    return text[:4].upper()
                return text.upper()

            updated: list[DeviceSnapshot] = []
            for snapshot in snapshots:
                row = _row_for(snapshot)
                if not row:
                    updated.append(snapshot)
                    continue
                used = _optional_float(row.get("memory_used"))
                total_mib = _optional_float(row.get("memory_total"))
                memory_percent = snapshot.memory_percent
                total_bytes = snapshot.memory_total_bytes
                available_bytes = snapshot.memory_available_bytes
                if total_mib is not None and total_mib > 0:
                    total_bytes = int(total_mib * 1024**2)
                    if used is not None:
                        memory_percent = 100.0 * used / total_mib
                        available_bytes = max(0, int((total_mib - used) * 1024**2))
                utilization = _optional_float(row.get("utilization"))
                updated.append(
                    DeviceSnapshot(
                        device_id=snapshot.device_id,
                        backend=snapshot.backend,
                        index=snapshot.index,
                        name=snapshot.name,
                        available=True,
                        utilization_percent=utilization
                        if utilization is not None
                        else snapshot.utilization_percent,
                        memory_percent=float(memory_percent),
                        telemetry="nvidia-smi + PyTorch CUDA",
                        runtime=snapshot.runtime,
                        temperature_c=_optional_float(row.get("temperature")),
                        power_w=_optional_float(row.get("power")),
                        power_limit_w=_optional_float(row.get("power_limit")),
                        memory_total_bytes=total_bytes,
                        memory_available_bytes=available_bytes,
                        hardware_uuid=str(row.get("uuid", "") or snapshot.hardware_uuid),
                        pci_bus_id=str(row.get("pci_bus_id", "") or snapshot.pci_bus_id),
                        vendor_id=snapshot.vendor_id,
                        product_id=_nvidia_product_id(row.get("pci_device_id", ""))
                        or snapshot.product_id,
                        driver_version=str(
                            row.get("driver_version", "") or snapshot.driver_version
                        ),
                        runtime_version=snapshot.runtime_version,
                    )
                )
            snapshots = updated

        self._cuda_cache = tuple(snapshots)
        self._cuda_cache_time = now
        return self._cuda_cache

    @staticmethod
    def _cpu_temperature_c() -> float | None:
        """Return a trustworthy host CPU/package temperature when the OS exposes one.

        Windows commonly does not expose reliable temperatures through psutil. In that case this
        function returns ``None``; CALO-RPD never invents a temperature value.
        """
        try:
            sensors = getattr(psutil, "sensors_temperatures", None)
            if not callable(sensors):
                return None
            groups = sensors(fahrenheit=False) or {}
            preferred = []
            fallback = []
            for group_name, entries in groups.items():
                for entry in entries or ():
                    value = getattr(entry, "current", None)
                    if value is None:
                        continue
                    try:
                        temperature = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not (0.0 < temperature < 150.0):
                        continue
                    label = f"{group_name} {getattr(entry, 'label', '')}".lower()
                    fallback.append(temperature)
                    if any(
                        token in label for token in ("package", "cpu", "tctl", "tdie", "coretemp")
                    ):
                        preferred.append(temperature)
            values = preferred or fallback
            return max(values) if values else None
        except (AttributeError, OSError, RuntimeError, ValueError):
            return None

    def sample(self) -> ResourceSnapshot:
        now = time.monotonic()
        cpu = float(psutil.cpu_percent(interval=None))
        memory = psutil.virtual_memory()
        devices = self._sample_cuda()
        return ResourceSnapshot(
            cpu_percent=cpu,
            devices=tuple(devices),
            system_memory_percent=float(memory.percent),
            system_memory_total_bytes=int(memory.total),
            system_memory_available_bytes=int(memory.available),
            cpu_temperature_c=self._cpu_temperature_c(),
            sampled_at_monotonic=now,
        )


def item_uses_calo_ai(mode: str, item) -> bool:
    """Return whether a planned v3 item is accelerator-compatible.

    The historical function name is retained for API compatibility.  In v3 every primary
    algorithm and every ablation item can use the common torch FP64 power-flow/evaluator backend;
    the registered baselines additionally have torch-native canonical population kernels. CALO uses
    the same accelerator evaluator plus its neural-policy inference path.
    """
    if mode == "comparison":
        return bool(str(getattr(item, "label", "")))
    if mode == "ablation":
        return getattr(item, "ablation_spec", None) is not None
    return False


def backend_allows_accelerators(execution_backend: str) -> bool:
    return str(execution_backend).lower() == "cuda_preferred"


def cpu_admission_allowed(
    snapshot: ResourceSnapshot,
    active_cpu_jobs: int,
    max_jobs: int,
) -> bool:
    """Admit CPU work from a frozen 80%-of-currently-available RAM envelope.

    CPU utilization is intentionally irrelevant. CPU execution is bounded by the explicit worker
    count while RAM admission is calculated from bytes available at this sampling boundary.
    """
    if active_cpu_jobs >= max(1, int(max_jobs)):
        return False
    if snapshot.system_memory_total_bytes <= 0:
        return False
    admission = calculate_available_memory_admission(
        total_bytes=int(snapshot.system_memory_total_bytes),
        available_bytes=int(snapshot.system_memory_available_bytes),
        requested_fraction=0.80,
    )
    return admission.additional_allowance_bytes > 0


def accelerator_admission_allowed(
    device: DeviceSnapshot,
    active_jobs: int,
    max_jobs: int,
) -> bool:
    """Admit CUDA work from a frozen 80%-of-currently-free VRAM envelope."""
    if not device.available:
        return False
    if active_jobs >= max(1, int(max_jobs)):
        return False
    if device.memory_total_bytes <= 0:
        return False
    admission = calculate_available_memory_admission(
        total_bytes=int(device.memory_total_bytes),
        available_bytes=int(device.memory_available_bytes),
        requested_fraction=0.80,
    )
    return admission.additional_allowance_bytes > 0


def prioritized_accelerators(snapshot: ResourceSnapshot) -> tuple[DeviceSnapshot, ...]:
    """Return accelerators in the default scientific execution priority order."""
    return snapshot.by_backend("cuda")


@dataclass(frozen=True, slots=True)
class WeightedAllocationSummary:
    """Automatic CUDA-first lane plan for one experiment.

    The historical class name is retained for import compatibility. There are no user-selected
    device shares: every compatible job targets CUDA when available, otherwise CPU.
    """

    total_jobs: int
    accelerator_eligible_jobs: int
    cpu_only_jobs: int
    cuda_jobs: int
    cpu_eligible_jobs: int
    total_cpu_jobs: int
    cuda_available: bool

    @property
    def requested_text(self) -> str:
        return "Automatic CUDA-first; CPU only when required by mode or fallback policy"

    @property
    def effective_text(self) -> str:
        return (
            f"CUDA {self.cuda_jobs} · CPU {self.total_cpu_jobs} "
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
) -> tuple[dict[int, str], WeightedAllocationSummary]:
    """Pre-assign every compatible job to CUDA when available, otherwise CPU."""
    items = list(plan)
    # The v3.4 torch FP64 pipeline provides tensor-native optimizer kernels and a common
    # device-resident ORPD evaluator for every primary algorithm and CALO ablation variant.
    # The legacy CPU-reference backend never calls this planner.
    eligible = list(items)
    cpu_only = []

    assignments: dict[int, str] = {int(item.job_index): "cpu" for item in cpu_only}
    lane = "cuda" if cuda_available else "cpu"
    for item in eligible:
        assignments[int(item.job_index)] = lane

    cuda_jobs = sum(1 for lane in assignments.values() if lane == "cuda")
    cpu_eligible_jobs = sum(
        1 for item in eligible if assignments.get(int(item.job_index), "cpu") == "cpu"
    )
    summary = WeightedAllocationSummary(
        total_jobs=len(items),
        accelerator_eligible_jobs=len(eligible),
        cpu_only_jobs=len(cpu_only),
        cuda_jobs=cuda_jobs,
        cpu_eligible_jobs=cpu_eligible_jobs,
        total_cpu_jobs=len(cpu_only) + cpu_eligible_jobs,
        cuda_available=bool(cuda_available),
    )
    return assignments, summary


def weighted_worker_slots(
    total_workers: int,
    summary: WeightedAllocationSummary,
) -> dict[str, int]:
    """Create concurrent lane caps while keeping at least one slot for each non-empty lane."""
    total_workers = max(1, int(total_workers))
    nonempty = [
        ("cuda", summary.cuda_jobs, summary.cuda_jobs),
        ("cpu", summary.total_cpu_jobs, summary.total_cpu_jobs),
    ]
    active = [(name, weight) for name, weight, jobs in nonempty if jobs > 0]
    if not active:
        return {"cuda": 0, "cpu": total_workers}
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
    return {"cuda": counts.get("cuda", 0), "cpu": counts.get("cpu", 0)}


@dataclass(frozen=True, slots=True)
class ThroughputAllocationSummary:
    """Measured-throughput plan for the complete v3.4 optimizer campaign."""

    total_jobs: int
    cuda_jobs: int
    cpu_jobs: int
    lane_throughputs: dict[str, float]
    source: str = "automatic calibration"

    @property
    def effective_text(self) -> str:
        return f"CUDA {self.cuda_jobs} · CPU {self.cpu_jobs}"

    @property
    def throughput_text(self) -> str:
        return (
            " · ".join(
                f"{lane.upper()} {value:,.1f} eval/s"
                for lane, value in (
                    ("cuda", self.lane_throughputs.get("cuda", 0.0)),
                    ("cpu", self.lane_throughputs.get("cpu", 0.0)),
                )
                if value > 0
            )
            or "No successful calibration"
        )


def build_throughput_lane_plan(
    plan,
    mode: str,
    *,
    lane_throughputs: dict[str, float],
    cuda_available: bool,
) -> tuple[dict[int, str], ThroughputAllocationSummary]:
    """Allocate all v3.4 jobs in proportion to measured candidate-evaluation throughput.

    The algorithm/run plan remains run-major and deterministic.  Lane assignments are interleaved
    to avoid confounding a device with a contiguous random-seed range.
    """
    from calo_rpd_studio.accelerated.throughput_engine import measured_throughput_allocation

    items = list(plan)
    enabled = {"cuda": bool(cuda_available), "cpu": True}
    throughputs = {
        "cuda": max(0.0, float(lane_throughputs.get("cuda", 0.0))),
        "cpu": max(0.0, float(lane_throughputs.get("cpu", 0.0))),
    }
    counts = measured_throughput_allocation(len(items), throughputs, enabled=enabled)
    remaining = dict(counts)
    assigned = {lane: 0 for lane in ("cuda", "cpu")}
    assignments: dict[int, str] = {}
    total = max(1, len(items))
    for position, item in enumerate(items):
        candidates = [lane for lane in ("cuda", "cpu") if remaining.get(lane, 0) > 0]
        if not candidates:
            lane = "cpu"
        else:
            lane = max(
                candidates,
                key=lambda name: (
                    (position + 1) * counts.get(name, 0) / total - assigned[name],
                    throughputs.get(name, 0.0),
                    name == "cuda",
                ),
            )
        assignments[int(item.job_index)] = lane
        if remaining.get(lane, 0) > 0:
            remaining[lane] -= 1
        assigned[lane] += 1
    summary = ThroughputAllocationSummary(
        total_jobs=len(items),
        cuda_jobs=sum(1 for value in assignments.values() if value == "cuda"),
        cpu_jobs=sum(1 for value in assignments.values() if value == "cpu"),
        lane_throughputs=throughputs,
    )
    return assignments, summary


def throughput_worker_slots(
    total_workers: int, summary: ThroughputAllocationSummary
) -> dict[str, int]:
    """Allocate concurrent worker slots from the measured lane capacities."""
    from calo_rpd_studio.accelerated.throughput_engine import largest_remainder_counts

    active_jobs = {
        "cuda": summary.cuda_jobs,
        "cpu": summary.cpu_jobs,
    }
    weights = {
        lane: (summary.lane_throughputs.get(lane, 0.0) if active_jobs[lane] > 0 else 0.0)
        for lane in active_jobs
    }
    counts = largest_remainder_counts(max(1, int(total_workers)), weights)
    active = [lane for lane, jobs in active_jobs.items() if jobs > 0]
    if int(total_workers) >= len(active):
        for lane in active:
            counts[lane] = max(1, counts.get(lane, 0))
        while sum(counts.values()) > int(total_workers):
            lane = max(
                (name for name in active if counts[name] > 1),
                key=lambda name: counts[name],
                default=None,
            )
            if lane is None:
                break
            counts[lane] -= 1
    return {lane: int(counts.get(lane, 0)) for lane in ("cuda", "cpu")}
