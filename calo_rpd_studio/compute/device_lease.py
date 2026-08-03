"""Cross-process exclusive leases for physical compute devices.

The VRAM allocator limit is process-local, so independently configured CUDA workers can otherwise
each claim the same free memory.  A device lease deliberately admits one heavy CUDA owner per
physical device.  Multiple components inside that owner share a reference-counted lease.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import BinaryIO


_LOG = logging.getLogger(__name__)


class DeviceLeaseUnavailable(RuntimeError):
    """Raised when another process owns the requested physical device."""


@dataclass(slots=True)
class _ProcessLease:
    stream: BinaryIO
    references: int


class ExclusiveDeviceLease:
    """Hold an OS-released, non-blocking exclusive lease for one device identifier."""

    _lock = threading.RLock()
    _process_leases: dict[str, _ProcessLease] = {}

    def __init__(self, device_id: str, *, root: str | Path | None = None) -> None:
        canonical = str(device_id).strip().lower().replace(":", "-")
        if not canonical or any(
            ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in canonical
        ):
            raise ValueError("device_id contains unsupported lease-path characters")
        configured_root = os.environ.get("CALO_DEVICE_LEASE_DIR", "").strip()
        lease_root = (
            Path(root)
            if root is not None
            else (
                Path(configured_root)
                if configured_root
                else Path(tempfile.gettempdir()) / "calo-rpd-device-leases"
            )
        )
        lease_root.mkdir(parents=True, exist_ok=True)
        self.device_id = str(device_id)
        self.key = str((lease_root / f"{canonical}.lock").resolve())
        self._closed = False
        with self._lock:
            existing = self._process_leases.get(self.key)
            if existing is not None:
                existing.references += 1
                return
            stream = open(self.key, "a+b")  # noqa: SIM115 - held for lease lifetime
            try:
                self._lock_stream(stream)
            except BaseException:
                stream.close()
                raise
            self._process_leases[self.key] = _ProcessLease(stream=stream, references=1)

    @staticmethod
    def _lock_stream(stream: BinaryIO) -> None:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise DeviceLeaseUnavailable(
                    "CUDA device is already leased by another CALO-RPD process"
                ) from exc
        else:
            import fcntl

            try:
                flock = getattr(fcntl, "flock")
                lock_ex = getattr(fcntl, "LOCK_EX")
                lock_nb = getattr(fcntl, "LOCK_NB")
                flock(stream.fileno(), lock_ex | lock_nb)
            except OSError as exc:
                raise DeviceLeaseUnavailable(
                    "CUDA device is already leased by another CALO-RPD process"
                ) from exc

    @staticmethod
    def _unlock_stream(stream: BinaryIO) -> None:
        if os.name == "nt":
            import msvcrt

            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            getattr(fcntl, "flock")(stream.fileno(), getattr(fcntl, "LOCK_UN"))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            current = self._process_leases.get(self.key)
            if current is None:
                return
            current.references -= 1
            if current.references > 0:
                return
            self._process_leases.pop(self.key, None)
            try:
                self._unlock_stream(current.stream)
            finally:
                current.stream.close()

    def __enter__(self) -> "ExclusiveDeviceLease":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            _LOG.debug("Unable to release the device lease during finalization", exc_info=True)
