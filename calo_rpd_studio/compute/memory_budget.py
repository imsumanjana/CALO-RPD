"""Available-memory admission contracts shared by CPU and CUDA execution.

An admission fraction is applied once to memory that is available immediately before a protected
job is admitted.  It is a ceiling for additional application memory, not a utilization target and
not a fraction of physical capacity.  The resulting allowance remains frozen for the lifetime of
that admission so it cannot recursively grow while the job consumes memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(slots=True, frozen=True)
class AvailableMemoryAdmission:
    """Immutable memory allowance calculated at an admission boundary."""

    total_bytes: int
    available_bytes_at_admission: int
    baseline_reserved_bytes: int
    requested_available_fraction: float
    additional_allowance_bytes: int
    process_ceiling_bytes: int
    allocator_fraction_of_total: float
    absolute_ceiling_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_available_memory_admission(
    *,
    total_bytes: int,
    available_bytes: int,
    requested_fraction: float = 0.80,
    baseline_reserved_bytes: int = 0,
    absolute_ceiling_fraction: float = 0.95,
) -> AvailableMemoryAdmission:
    """Return a frozen allowance based on currently available memory.

    ``baseline_reserved_bytes`` is memory already owned by the protected process/group.  It is
    included in the allocator ceiling but not counted as newly available memory.  The absolute
    ceiling is a final safety guard; normal 80%-of-available admission will be stricter.
    """

    total = int(total_bytes)
    available = int(available_bytes)
    baseline = int(baseline_reserved_bytes)
    fraction = float(requested_fraction)
    hard_fraction = float(absolute_ceiling_fraction)
    if total <= 0:
        raise ValueError("total_bytes must be positive")
    if available < 0 or available > total:
        raise ValueError("available_bytes must lie between zero and total_bytes")
    if baseline < 0 or baseline > total:
        raise ValueError("baseline_reserved_bytes must lie between zero and total_bytes")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 0.95:
        raise ValueError("requested_fraction must be finite, positive, and no greater than 0.95")
    if not math.isfinite(hard_fraction) or not 0.0 < hard_fraction <= 1.0:
        raise ValueError("absolute_ceiling_fraction must be finite and between zero and one")

    hard_ceiling = int(math.floor(total * hard_fraction))
    if baseline > hard_ceiling:
        raise RuntimeError(
            "Memory already reserved at admission exceeds the absolute protected ceiling"
        )
    requested_allowance = int(math.floor(available * fraction))
    additional = min(requested_allowance, max(0, hard_ceiling - baseline))
    ceiling = baseline + additional
    return AvailableMemoryAdmission(
        total_bytes=total,
        available_bytes_at_admission=available,
        baseline_reserved_bytes=baseline,
        requested_available_fraction=fraction,
        additional_allowance_bytes=additional,
        process_ceiling_bytes=ceiling,
        allocator_fraction_of_total=float(ceiling / total),
        absolute_ceiling_fraction=hard_fraction,
    )
