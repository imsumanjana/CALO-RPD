"""v6.2 protected policy-training resource planning.

The planner separates scientific branch count from simultaneous execution capacity.  It consumes the
same authoritative ComputeTopology/Safe-80 profile shown on Dashboard and never invents an accelerator
or silently spills an accelerator request onto CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .topology import (
    ComputeDevice,
    ComputeProtectionProfile,
    ComputeTopologyService,
    ComputeTopologySnapshot,
    SafeResourceBudgetEngine,
    CPU_SUPPORT_RESERVE_PER_BRANCH,
    MINIMUM_CPU_ROLLOUT_WORKERS_PER_BRANCH,
)


@dataclass(frozen=True, slots=True)
class BranchResourceSlot:
    """One simultaneously executable branch slot."""

    slot_index: int
    primary_device: str
    backend: str
    runtime: str
    device_name: str
    cpu_worker_budget: int
    auxiliary_xpu_runtime: str = ""
    auxiliary_xpu_name: str = ""

    @property
    def uses_auxiliary_xpu(self) -> bool:
        return bool(self.auxiliary_xpu_runtime)

    def to_dict(self) -> dict:
        return {
            "slot_index": int(self.slot_index),
            "primary_device": self.primary_device,
            "backend": self.backend,
            "runtime": self.runtime,
            "device_name": self.device_name,
            "cpu_worker_budget": int(self.cpu_worker_budget),
            "auxiliary_xpu_runtime": self.auxiliary_xpu_runtime,
            "auxiliary_xpu_name": self.auxiliary_xpu_name,
        }


@dataclass(frozen=True, slots=True)
class TrainingResourcePlan:
    total_branches: int
    simultaneous_branches: int
    queued_branches: int
    global_cpu_worker_budget: int
    cpu_support_reserve_per_branch: int
    cpu_rollout_worker_budget_total: int
    slots: tuple[BranchResourceSlot, ...]
    topology_fingerprint: str
    protection_profile_fingerprint: str
    protection_profile_name: str

    def to_dict(self) -> dict:
        return {
            "total_branches": int(self.total_branches),
            "simultaneous_branches": int(self.simultaneous_branches),
            "queued_branches": int(self.queued_branches),
            "global_cpu_worker_budget": int(self.global_cpu_worker_budget),
            "cpu_support_reserve_per_branch": int(self.cpu_support_reserve_per_branch),
            "cpu_rollout_worker_budget_total": int(self.cpu_rollout_worker_budget_total),
            "slots": [slot.to_dict() for slot in self.slots],
            "topology_fingerprint": self.topology_fingerprint,
            "protection_profile_fingerprint": self.protection_profile_fingerprint,
            "protection_profile_name": self.protection_profile_name,
        }



def protected_rollout_shares(
    *,
    cuda_share: int,
    xpu_share: int,
    cpu_share: int,
    primary_device: str,
    auxiliary_xpu_runtime: str = "",
) -> dict[str, int]:
    """Return the exact protected rollout routing executed for one admitted branch slot.

    This is the single reporting/runtime authority for v6.4 Stage B. It preserves the v6.1/v6.2
    fail-closed rule: unavailable accelerator share is never silently converted into extra CPU heat.
    Instead an already-admitted accelerator primary absorbs sibling accelerator share where the
    protected runtime contract permits it.
    """

    cuda = max(0, min(100, int(cuda_share)))
    xpu = max(0, min(100, int(xpu_share)))
    cpu = max(0, min(100, int(cpu_share)))
    if cuda + xpu + cpu != 100:
        raise ValueError("Protected heterogeneous CUDA/XPU/CPU rollout shares must total exactly 100%")
    assigned = str(primary_device or "cpu").lower()
    aux_xpu = bool(str(auxiliary_xpu_runtime or "").strip())
    if assigned.startswith("cuda"):
        return {
            "cuda": cuda + (0 if aux_xpu else xpu),
            "xpu": xpu if aux_xpu else 0,
            "cpu": cpu,
        }
    if assigned.startswith("xpu"):
        return {"cuda": 0, "xpu": min(100, xpu + cuda), "cpu": cpu}
    return {
        "cuda": 0,
        "xpu": xpu if aux_xpu else 0,
        "cpu": min(100, cpu + cuda + (0 if aux_xpu else xpu)),
    }

def _device_has_safe_headroom(device: ComputeDevice, profile: ComputeProtectionProfile) -> bool:
    """Return whether a full-branch accelerator is currently admissible inside Safe-80.

    The Dashboard profile stores aggregate slot counts, but branch placement must also identify
    *which* accelerator owns that headroom.  Without this check a busy first CUDA device could be
    selected merely because a different accelerator contributed the aggregate safe slot.
    """
    if not device.full_training_branch:
        return False
    if float(device.memory_used_percent) >= float(profile.allocation_limit_fraction) * 100.0:
        return False
    total = int(device.memory_total_bytes or 0)
    if total > 0:
        safe_ceiling = int(total * float(profile.allocation_limit_fraction))
        used = int(total * float(device.memory_used_percent) / 100.0)
        if max(0, safe_ceiling - used) < int(profile.estimated_branch_accelerator_bytes):
            return False
    return True


def _primary_candidates(devices: Iterable[ComputeDevice], requested: str) -> list[ComputeDevice]:
    full = [device for device in devices if device.full_training_branch]
    cuda = [device for device in full if device.backend == "cuda"]
    direct_xpu = [
        device
        for device in full
        if device.backend == "xpu" and device.runtime == "primary"
    ]
    if requested == "auto":
        return [*cuda, *direct_xpu]
    if requested.startswith("cuda"):
        if ":" in requested:
            return [device for device in cuda if device.runtime_id == requested]
        return cuda
    if requested == "xpu" or requested.startswith("xpu:"):
        if ":" in requested:
            return [device for device in direct_xpu if device.runtime_id == requested]
        return direct_xpu
    return []


def _safe_concurrency(
    *,
    total_branches: int,
    requested_concurrency: int,
    profile: ComputeProtectionProfile,
) -> int:
    hard_limit = max(0, int(profile.safe_parallel_branches))
    if hard_limit < 1:
        raise RuntimeError(
            "Safe-80 compute protection reports no admissible simultaneous policy-training branch. "
            + " ".join(profile.reasons)
        )
    requested = int(requested_concurrency or 0)
    if requested <= 0:
        requested = min(int(total_branches), hard_limit)
    if requested > hard_limit:
        raise RuntimeError(
            f"Requested simultaneous branch concurrency ({requested}) exceeds the Dashboard Safe-80 "
            f"hard limit ({hard_limit}). Lower concurrency; total scientific branch count may remain unchanged."
        )
    return max(1, min(int(total_branches), requested, hard_limit))


def build_training_resource_plan(
    config,
    total_branches: int,
    *,
    topology: ComputeTopologySnapshot | None = None,
    profile: ComputeProtectionProfile | None = None,
) -> TrainingResourcePlan:
    """Build a fail-closed, capability-aware protected branch resource plan.

    ``parallel_runs`` is scientific diversity. ``parallel_concurrency`` is execution concurrency.
    Excess scientific branches are queued; they are never converted into implicit CPU branches.
    """

    total = max(1, int(total_branches))
    if topology is None:
        topology = ComputeTopologyService().scan()
    if profile is None:
        profile = SafeResourceBudgetEngine(allocation_limit_fraction=0.80).calculate(topology)

    expected_topology = str(getattr(config, "compute_topology_fingerprint", "") or "")
    if expected_topology and expected_topology != topology.fingerprint:
        raise RuntimeError(
            "The live CPU/XPU/GPU topology no longer matches the Dashboard hardware map used to configure training. "
            "Refresh Dashboard system mapping before launch."
        )
    # Resource pressure is intentionally re-evaluated live. A changed profile fingerprint caused only
    # by current RAM/VRAM pressure is not treated as identity drift; instead the newly calculated
    # Safe-80 concurrency is enforced below and may reduce/block admission.

    concurrency = _safe_concurrency(
        total_branches=total,
        requested_concurrency=int(getattr(config, "parallel_concurrency", 0) or 0),
        profile=profile,
    )
    requested = str(getattr(config, "ppo_device", "auto") or "auto").strip().lower()
    global_cpu = max(
        1,
        min(
            int(profile.safe_cpu_worker_budget),
            int(getattr(config, "safe_global_cpu_workers", 0) or profile.safe_cpu_worker_budget),
        ),
    )
    if global_cpu < concurrency:
        raise RuntimeError(
            f"Global protected CPU worker budget ({global_cpu}) cannot support {concurrency} simultaneous branches."
        )

    slots_devices: list[ComputeDevice | None]
    if requested == "cpu":
        # CPU is allowed only because the user explicitly selected it. This is never an automatic
        # accelerator spillover path.
        slots_devices = [None] * concurrency
    elif requested == "xpu_sidecar":
        raise RuntimeError(
            "The secondary XPU sidecar is capability-classified as an actor/evaluator runtime, not a "
            "full independent competitive PPO branch. Select Automatic/direct XPU/CUDA/CPU."
        )
    else:
        candidates = [
            device
            for device in _primary_candidates(topology.devices, requested)
            if _device_has_safe_headroom(device, profile)
        ]
        if not candidates:
            full_devices_exist = any(device.full_training_branch for device in topology.devices)
            if requested == "auto" and not full_devices_exist:
                # A genuinely CPU-only primary runtime may execute protected sequential/queued CPU
                # branches. A detected-but-full accelerator does not trigger this path.
                slots_devices = [None] * concurrency
            else:
                raise RuntimeError(
                    f"No validated full-training-branch device matches {requested!r}. Automatic heavy CPU "
                    "fallback is disabled; use a validated device or explicitly select CPU."
                )
        else:
            # Safe-80 currently admits at most one full branch per validated accelerator. If fewer
            # primary slots exist than requested concurrency, fail closed rather than creating CPU spillover.
            if len(candidates) < concurrency:
                raise RuntimeError(
                    f"Only {len(candidates)} validated primary accelerator slot(s) match {requested!r}, but "
                    f"{concurrency} simultaneous branches were requested. Lower concurrency; queued total "
                    "branch count may remain unchanged."
                )
            slots_devices = list(candidates[:concurrency])

    # One worker-equivalent per simultaneous branch is reserved for the branch coordinator and
    # accelerator/IPC support. Only the remaining global protected budget may become CPU rollout
    # workers. Thus branch process + CPU rollout pools cannot intentionally multiply the global CPU
    # budget merely because more branches are admitted.
    # Reserve two host worker-equivalents per active branch for the branch coordinator plus
    # accelerator actor/IPC/driver support. These are not rollout workers. This deliberately
    # trades a little peak host throughput for laptop-safe headroom and prevents process-count
    # multiplication from consuming the entire Safe-80 CPU allocation.
    support_reserve_per_branch = CPU_SUPPORT_RESERVE_PER_BRANCH
    support_total = concurrency * support_reserve_per_branch
    rollout_total = global_cpu - support_total
    if rollout_total < concurrency * MINIMUM_CPU_ROLLOUT_WORKERS_PER_BRANCH:
        raise RuntimeError(
            f"Global protected CPU worker budget ({global_cpu}) cannot provide protected host support reserve "
            f"and one CPU rollout worker for each of {concurrency} simultaneous branches."
        )
    base = rollout_total // concurrency
    remainder = rollout_total % concurrency
    cpu_budgets = [base + (1 if index < remainder else 0) for index in range(concurrency)]

    # XPU capability-aware auxiliary scheduling: a non-primary XPU actor/evaluator runtime may assist
    # at most one simultaneous branch. It never masquerades as an independent full branch.
    used_primary_ids = {
        device.runtime_id for device in slots_devices if device is not None
    }
    auxiliary_xpus = [
        device
        for device in topology.devices
        if device.backend == "xpu"
        and device.policy_actor
        and device.runtime_id not in used_primary_ids
    ]
    allow_aux_xpu = bool(getattr(config, "heterogeneous_rollouts", False)) and int(
        getattr(config, "xpu_rollout_share", 0) or 0
    ) > 0

    slots: list[BranchResourceSlot] = []
    aux_index = 0
    for index, device in enumerate(slots_devices):
        auxiliary = None
        if allow_aux_xpu and aux_index < len(auxiliary_xpus):
            auxiliary = auxiliary_xpus[aux_index]
            aux_index += 1
        if device is None:
            slots.append(
                BranchResourceSlot(
                    slot_index=index,
                    primary_device="cpu",
                    backend="cpu",
                    runtime="primary",
                    device_name="CPU",
                    cpu_worker_budget=max(1, int(cpu_budgets[index])),
                    auxiliary_xpu_runtime=(auxiliary.runtime if auxiliary else ""),
                    auxiliary_xpu_name=(auxiliary.name if auxiliary else ""),
                )
            )
        else:
            slots.append(
                BranchResourceSlot(
                    slot_index=index,
                    primary_device=device.runtime_id,
                    backend=device.backend,
                    runtime=device.runtime,
                    device_name=device.name,
                    cpu_worker_budget=max(1, int(cpu_budgets[index])),
                    auxiliary_xpu_runtime=(auxiliary.runtime if auxiliary else ""),
                    auxiliary_xpu_name=(auxiliary.name if auxiliary else ""),
                )
            )

    return TrainingResourcePlan(
        total_branches=total,
        simultaneous_branches=concurrency,
        queued_branches=max(0, total - concurrency),
        global_cpu_worker_budget=global_cpu,
        cpu_support_reserve_per_branch=support_reserve_per_branch,
        cpu_rollout_worker_budget_total=rollout_total,
        slots=tuple(slots),
        topology_fingerprint=topology.fingerprint,
        protection_profile_fingerprint=profile.profile_fingerprint,
        protection_profile_name=profile.profile_name,
    )


__all__ = [
    "BranchResourceSlot",
    "TrainingResourcePlan",
    "build_training_resource_plan",
]
