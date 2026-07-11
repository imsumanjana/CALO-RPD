import pytest

pytest.importorskip("PyQt6")

from calo_rpd_studio.app.task_status import TaskStatus


def test_task_status_lifecycle():
    status = TaskStatus()
    assert status.snapshot()["state"] == "Ready"
    assert status.begin("Run", progress=0, cancellable=True)
    assert status.snapshot()["busy"] is True
    status.update(42, "Working")
    assert status.snapshot()["progress"] == 42
    status.finish("Done")
    assert status.snapshot()["busy"] is False
    assert status.snapshot()["state"] == "Completed"
