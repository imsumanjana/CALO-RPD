"""Lease concurrency uses synthetic device names and real temporary file locks."""

from pathlib import Path
import threading
import pytest
from calo_rpd_studio.compute.device_lease import (
    ExclusiveDeviceLease as Lease,
    DeviceLeaseUnavailable,
    DeviceLeaseCancelled,
)


def test_waiting_for_one_device_does_not_block_releasing_another(tmp_path, monkeypatch):
    owner = Lease("audit-a", root=tmp_path)
    waiting, cancel, released = threading.Event(), threading.Event(), threading.Event()
    real_lock = Lease._lock_stream
    outcomes = []

    def lock(stream):
        if Path(stream.name).name.startswith("audit-b-"):
            waiting.set()
            raise DeviceLeaseUnavailable("synthetic competing owner")
        real_lock(stream)

    monkeypatch.setattr(Lease, "_lock_stream", staticmethod(lock))

    def wait_for_device():
        try:
            with Lease(
                "audit-b",
                root=tmp_path,
                wait=True,
                cancel_callback=cancel.is_set,
                poll_interval_seconds=0.01,
            ):
                outcomes.append("unexpected acquisition")
        except DeviceLeaseCancelled:
            outcomes.append("cancelled")

    def release():
        owner.close()
        released.set()

    waiter = threading.Thread(target=wait_for_device, daemon=True)
    closer = threading.Thread(target=release, daemon=True)
    waiter.start()
    try:
        assert waiting.wait(5)
        closer.start()
        closed_before_cancel = released.wait(1)
    finally:
        cancel.set()
        waiter.join(5)
        if closer.ident is not None:
            closer.join(5)
        owner.close()
    assert closed_before_cancel, "Global mutex prevented an unrelated release"
    assert not waiter.is_alive() and not closer.is_alive()
    assert outcomes == ["cancelled"]


def test_failed_constructor_cannot_release_a_later_owner(tmp_path, monkeypatch):
    failed = Lease.__new__(Lease)

    def busy(_stream):
        raise DeviceLeaseUnavailable("busy")

    with monkeypatch.context() as patch:
        patch.setattr(Lease, "_lock_stream", staticmethod(busy))
        with pytest.raises(DeviceLeaseUnavailable):
            failed.__init__("audit-fixture", root=tmp_path)
    with Lease("audit-fixture", root=tmp_path) as owner:
        failed.close()
        assert owner.key in Lease._process_leases
        assert Lease._process_leases[owner.key].references == 1


def test_close_is_idempotent_for_shared_references(tmp_path):
    first = Lease("audit-shared", root=tmp_path)
    second = Lease("audit-shared", root=tmp_path)
    try:
        assert Lease._process_leases[first.key].references == 2
        first.close()
        first.close()
        assert Lease._process_leases[second.key].references == 1
    finally:
        first.close()
        second.close()
    assert second.key not in Lease._process_leases


def test_bounded_wait_has_no_owned_reference(tmp_path, monkeypatch):
    def busy(_stream):
        raise DeviceLeaseUnavailable("busy")

    monkeypatch.setattr(Lease, "_lock_stream", staticmethod(busy))
    with pytest.raises(DeviceLeaseUnavailable, match="timeout"):
        Lease("audit-timeout", root=tmp_path, wait=True, timeout_seconds=0)
    assert not any("audit-timeout-" in key for key in Lease._process_leases)
