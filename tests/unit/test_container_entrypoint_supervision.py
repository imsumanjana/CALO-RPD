from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _entrypoint_module():
    spec = importlib.util.spec_from_file_location(
        "calo_container_entrypoint_test", ROOT / "containers" / "entrypoint.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeProcess:
    def __init__(self, pid: int, polls: list[int | None], *, observe: Path | None = None):
        self.pid = pid
        self._polls = list(polls)
        self.returncode: int | None = None
        self.observe = observe
        self.observed_ready = False

    def poll(self):
        if self.observe is not None:
            self.observed_ready = self.observed_ready or self.observe.is_file()
        value = self._polls.pop(0) if self._polls else self.returncode
        self.returncode = value
        return value


def test_entrypoint_refuses_readiness_when_qt_app_exits_early(tmp_path):
    entrypoint = _entrypoint_module()
    app = _FakeProcess(101, [7])
    entrypoint._CHILDREN[:] = [app]
    pid_file = tmp_path / "app.pid"

    assert (
        entrypoint._supervise_children(app, pid_file, readiness_seconds=0.02, poll_seconds=0.001)
        == 7
    )
    assert not pid_file.exists()


def test_entrypoint_publishes_live_app_pid_and_removes_it_on_exit(tmp_path):
    entrypoint = _entrypoint_module()
    pid_file = tmp_path / "app.pid"
    app = _FakeProcess(202, [None, 0], observe=pid_file)
    entrypoint._CHILDREN[:] = [app]

    assert (
        entrypoint._supervise_children(app, pid_file, readiness_seconds=0.0, poll_seconds=0.001)
        == 0
    )
    assert app.observed_ready
    assert not pid_file.exists()


def test_entrypoint_fails_if_a_desktop_dependency_exits_cleanly(tmp_path):
    entrypoint = _entrypoint_module()
    app = _FakeProcess(303, [None])
    websockify = _FakeProcess(404, [0])
    entrypoint._CHILDREN[:] = [app, websockify]
    pid_file = tmp_path / "app.pid"

    assert (
        entrypoint._supervise_children(app, pid_file, readiness_seconds=0.0, poll_seconds=0.001)
        == 1
    )
    assert not pid_file.exists()
