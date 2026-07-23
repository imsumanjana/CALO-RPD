"""Adaptive compute/thermal protection governor for CALO-RPD v6.2.

The governor is deliberately conservative and fail-closed for *admission*. It never invents missing
sensor values. Safe-80 is an allocation envelope, not a promise that any specific GPU utilization or
temperature must equal 80%. Trusted temperature/power telemetry is used when available; otherwise
CPU/RAM/device-memory pressure and explicit concurrency budgets remain authoritative.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import time

from .resource_scheduler import ResourceMonitor, ResourceSnapshot
from .topology import ComputeProtectionProfile


class ProtectionState(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class GovernorConfig:
    allocation_limit_fraction: float = 0.80
    red_resource_fraction: float = 0.95
    # Conservative policy thresholds used only when a trustworthy temperature reading exists.
    # They are protection-policy thresholds, not manufacturer thermal-limit claims.
    accelerator_amber_temperature_c: float = 80.0
    accelerator_red_temperature_c: float = 88.0
    cpu_amber_temperature_c: float = 85.0
    cpu_red_temperature_c: float = 95.0
    power_amber_fraction: float = 0.80
    power_red_fraction: float = 0.97
    amber_confirm_samples: int = 2
    red_confirm_samples: int = 2
    green_recovery_samples: int = 3
    staged_startup_delay_seconds: float = 2.0
    sample_interval_seconds: float = 1.0
    amber_duty_cycle: float = 0.60
    amber_pause_seconds: float = 0.25

    def validate(self) -> None:
        for name in (
            "allocation_limit_fraction",
            "red_resource_fraction",
            "power_amber_fraction",
            "power_red_fraction",
            "amber_duty_cycle",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if self.red_resource_fraction <= self.allocation_limit_fraction:
            raise ValueError("red_resource_fraction must exceed allocation_limit_fraction")
        if self.power_red_fraction <= self.power_amber_fraction:
            raise ValueError("power_red_fraction must exceed power_amber_fraction")
        if self.accelerator_red_temperature_c <= self.accelerator_amber_temperature_c:
            raise ValueError("accelerator red temperature must exceed amber temperature")
        if self.cpu_red_temperature_c <= self.cpu_amber_temperature_c:
            raise ValueError("CPU red temperature must exceed amber temperature")
        if min(self.amber_confirm_samples, self.red_confirm_samples, self.green_recovery_samples) < 1:
            raise ValueError("governor hysteresis sample counts must be >= 1")
        if self.staged_startup_delay_seconds < 0 or self.sample_interval_seconds <= 0:
            raise ValueError("governor timing values are invalid")
        if self.amber_pause_seconds < 0:
            raise ValueError("amber_pause_seconds cannot be negative")


@dataclass(frozen=True, slots=True)
class GovernorDecision:
    state: ProtectionState
    allow_new_admission: bool
    request_safe_stop: bool
    throttle_level: int  # 0 green, 1 amber, 2 red
    throttle_fraction: float
    reasons: tuple[str, ...]
    sampled_at_monotonic: float
    active_branches: int
    profile_safe_parallel_branches: int
    snapshot: dict
    decision_fingerprint: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["state"] = self.state.value
        return payload


class AdaptiveComputeGovernor:
    """Stateful Green/Amber/Red protection with hysteresis and staged admission."""

    def __init__(
        self,
        profile: ComputeProtectionProfile,
        *,
        monitor: ResourceMonitor | None = None,
        config: GovernorConfig | None = None,
    ) -> None:
        self.profile = profile
        self.monitor = monitor or ResourceMonitor()
        self.config = config or GovernorConfig(
            allocation_limit_fraction=float(profile.allocation_limit_fraction)
        )
        self.config.validate()
        self._state = ProtectionState.UNKNOWN
        self._amber_count = 0
        self._red_count = 0
        self._green_count = 0
        self._last_launch_monotonic = -math.inf
        self._history: deque[GovernorDecision] = deque(maxlen=256)

    @property
    def state(self) -> ProtectionState:
        return self._state

    @property
    def history(self) -> tuple[GovernorDecision, ...]:
        return tuple(self._history)

    def note_branch_launch(self, when: float | None = None) -> None:
        self._last_launch_monotonic = float(time.monotonic() if when is None else when)

    def staged_delay_elapsed(self, now: float | None = None) -> bool:
        current = float(time.monotonic() if now is None else now)
        return current - self._last_launch_monotonic >= float(self.config.staged_startup_delay_seconds)

    @staticmethod
    def _snapshot_payload(snapshot: ResourceSnapshot) -> dict:
        return {
            "cpu_percent": float(snapshot.cpu_percent),
            "cpu_temperature_c": snapshot.cpu_temperature_c,
            "system_memory_percent": float(snapshot.system_memory_percent),
            "sampled_at_monotonic": float(snapshot.sampled_at_monotonic),
            "devices": [
                {
                    "device_id": d.device_id,
                    "backend": d.backend,
                    "runtime": d.runtime,
                    "name": d.name,
                    "utilization_percent": d.utilization_percent,
                    "memory_percent": float(d.memory_percent),
                    "temperature_c": d.temperature_c,
                    "power_w": d.power_w,
                    "power_limit_w": d.power_limit_w,
                    "throttle_reason": d.throttle_reason,
                    "telemetry": d.telemetry,
                }
                for d in snapshot.devices
            ],
        }

    def _classify_raw(self, snapshot: ResourceSnapshot) -> tuple[ProtectionState, list[str]]:
        cfg = self.config
        amber_limit = 100.0 * float(cfg.allocation_limit_fraction)
        red_limit = 100.0 * float(cfg.red_resource_fraction)
        amber: list[str] = []
        red: list[str] = []

        if snapshot.cpu_percent >= red_limit:
            red.append(f"CPU utilization {snapshot.cpu_percent:.1f}% >= red protection {red_limit:.0f}%")
        elif snapshot.cpu_percent >= amber_limit:
            amber.append(f"CPU utilization {snapshot.cpu_percent:.1f}% >= Safe-{amber_limit:.0f}% envelope")

        if snapshot.system_memory_percent >= red_limit:
            red.append(
                f"System RAM utilization {snapshot.system_memory_percent:.1f}% >= red protection {red_limit:.0f}%"
            )
        elif snapshot.system_memory_percent >= amber_limit:
            amber.append(
                f"System RAM utilization {snapshot.system_memory_percent:.1f}% >= Safe-{amber_limit:.0f}% envelope"
            )

        if snapshot.cpu_temperature_c is not None:
            temperature = float(snapshot.cpu_temperature_c)
            if temperature >= cfg.cpu_red_temperature_c:
                red.append(
                    f"Measured CPU/package temperature {temperature:.1f}°C >= configured red protection "
                    f"{cfg.cpu_red_temperature_c:.1f}°C"
                )
            elif temperature >= cfg.cpu_amber_temperature_c:
                amber.append(
                    f"Measured CPU/package temperature {temperature:.1f}°C >= configured amber protection "
                    f"{cfg.cpu_amber_temperature_c:.1f}°C"
                )

        for device in snapshot.devices:
            label = f"{device.device_id} ({device.name})"
            if float(device.memory_percent) >= red_limit:
                red.append(
                    f"{label} memory {device.memory_percent:.1f}% >= red protection {red_limit:.0f}%"
                )
            elif float(device.memory_percent) >= amber_limit:
                amber.append(
                    f"{label} memory {device.memory_percent:.1f}% >= Safe-{amber_limit:.0f}% envelope"
                )
            if device.temperature_c is not None:
                temp = float(device.temperature_c)
                if temp >= cfg.accelerator_red_temperature_c:
                    red.append(
                        f"Measured {label} temperature {temp:.1f}°C >= configured red protection "
                        f"{cfg.accelerator_red_temperature_c:.1f}°C"
                    )
                elif temp >= cfg.accelerator_amber_temperature_c:
                    amber.append(
                        f"Measured {label} temperature {temp:.1f}°C >= configured amber protection "
                        f"{cfg.accelerator_amber_temperature_c:.1f}°C"
                    )
            if (
                device.power_w is not None
                and device.power_limit_w is not None
                and float(device.power_limit_w) > 0
            ):
                fraction = float(device.power_w) / float(device.power_limit_w)
                if fraction >= cfg.power_red_fraction:
                    red.append(
                        f"{label} power {device.power_w:.1f} W is {100*fraction:.1f}% of reported limit"
                    )
                elif fraction >= cfg.power_amber_fraction:
                    amber.append(
                        f"{label} power {device.power_w:.1f} W is {100*fraction:.1f}% of reported limit"
                    )
            if str(device.throttle_reason or "").strip():
                amber.append(f"{label} reports throttling: {device.throttle_reason}")

        if red:
            return ProtectionState.RED, red + amber
        if amber:
            return ProtectionState.AMBER, amber
        return ProtectionState.GREEN, []

    def evaluate_snapshot(
        self, snapshot: ResourceSnapshot, *, active_branches: int = 0
    ) -> GovernorDecision:
        raw_state, reasons = self._classify_raw(snapshot)
        if raw_state is ProtectionState.RED:
            self._red_count += 1
            self._amber_count = 0
            self._green_count = 0
        elif raw_state is ProtectionState.AMBER:
            self._amber_count += 1
            self._red_count = 0
            self._green_count = 0
        else:
            self._green_count += 1
            self._red_count = 0
            self._amber_count = 0

        # Immediate first sample establishes GREEN. Protection escalation/recovery uses hysteresis.
        if self._state is ProtectionState.UNKNOWN and raw_state is ProtectionState.GREEN:
            self._state = ProtectionState.GREEN
        elif self._red_count >= self.config.red_confirm_samples:
            self._state = ProtectionState.RED
        elif self._amber_count >= self.config.amber_confirm_samples and self._state is not ProtectionState.RED:
            self._state = ProtectionState.AMBER
        elif self._green_count >= self.config.green_recovery_samples:
            self._state = ProtectionState.GREEN

        # A raw RED sample always blocks new admission immediately even before red-stop hysteresis.
        effective_for_admission = (
            ProtectionState.RED if raw_state is ProtectionState.RED else self._state
        )
        allow = effective_for_admission is ProtectionState.GREEN
        request_stop = self._state is ProtectionState.RED
        level = 2 if self._state is ProtectionState.RED else 1 if self._state is ProtectionState.AMBER else 0
        throttle_fraction = 0.0 if level == 2 else float(self.config.amber_duty_cycle) if level == 1 else 1.0
        payload = self._snapshot_payload(snapshot)
        identity = {
            "state": self._state.value,
            "raw_state": raw_state.value,
            "allow": allow,
            "stop": request_stop,
            "reasons": reasons,
            "active_branches": int(active_branches),
            "profile": self.profile.profile_fingerprint,
            "snapshot": payload,
        }
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        decision = GovernorDecision(
            state=self._state,
            allow_new_admission=bool(allow),
            request_safe_stop=bool(request_stop),
            throttle_level=level,
            throttle_fraction=throttle_fraction,
            reasons=tuple(reasons),
            sampled_at_monotonic=float(snapshot.sampled_at_monotonic or time.monotonic()),
            active_branches=max(0, int(active_branches)),
            profile_safe_parallel_branches=int(self.profile.safe_parallel_branches),
            snapshot=payload,
            decision_fingerprint=fingerprint,
        )
        self._history.append(decision)
        return decision

    def sample(self, *, active_branches: int = 0) -> GovernorDecision:
        return self.evaluate_snapshot(self.monitor.sample(), active_branches=active_branches)


__all__ = [
    "ProtectionState",
    "GovernorConfig",
    "GovernorDecision",
    "AdaptiveComputeGovernor",
]
