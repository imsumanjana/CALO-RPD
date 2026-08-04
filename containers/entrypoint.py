"""Fail-closed container launcher with a browser-accessible Qt desktop."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


_CHILDREN: list[subprocess.Popen] = []


def _terminate_children() -> None:
    for process in reversed(_CHILDREN):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for process in reversed(_CHILDREN):
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.kill()


def _handle_signal(signum: int, _frame) -> None:
    _terminate_children()
    raise SystemExit(128 + signum)


def _start(*command: str) -> subprocess.Popen:
    process = subprocess.Popen(command)
    _CHILDREN.append(process)
    return process


def _first_exited() -> subprocess.Popen | None:
    return next((process for process in _CHILDREN if process.poll() is not None), None)


def _supervise_children(
    app: subprocess.Popen,
    pid_file: Path,
    *,
    readiness_seconds: float = 3.0,
    poll_seconds: float = 0.1,
) -> int:
    """Publish readiness only while every desktop process and the Qt app remain alive."""

    pid_file.unlink(missing_ok=True)
    deadline = time.monotonic() + float(readiness_seconds)
    while time.monotonic() < deadline:
        exited = _first_exited()
        if exited is not None:
            code = int(exited.returncode or 0)
            return code if exited is app and code != 0 else max(code, 1)
        time.sleep(float(poll_seconds))
    pid_file.write_text(f"{int(app.pid)}\n", encoding="ascii")
    try:
        while True:
            exited = _first_exited()
            if exited is not None:
                code = int(exited.returncode or 0)
                return code if exited is app else max(code, 1)
            time.sleep(float(poll_seconds))
    finally:
        pid_file.unlink(missing_ok=True)


def _verify_compute_mode() -> None:
    import torch

    mode = os.environ.get("CALO_COMPUTE_MODE", "cpu").strip().lower()
    if mode not in {"cpu", "cuda"}:
        raise RuntimeError("CALO_COMPUTE_MODE must be either 'cpu' or 'cuda'.")
    cuda_ready = bool(torch.cuda.is_available())
    if mode == "cpu" and cuda_ready:
        raise RuntimeError(
            "CPU container unexpectedly exposes CUDA; keep CUDA_VISIBLE_DEVICES empty."
        )
    if mode == "cuda" and not cuda_ready:
        raise RuntimeError(
            "CUDA profile was selected but PyTorch cannot access an NVIDIA GPU. Verify the host "
            "driver, Docker Desktop WSL2 GPU support, and NVIDIA Container Toolkit."
        )
    if cuda_ready:
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        print(
            "CALO CUDA runtime ready: "
            f"{torch.cuda.get_device_name(0)}; free={free_bytes}; total={total_bytes}",
            flush=True,
        )
    else:
        print("CALO CPU runtime ready; CUDA is not exposed.", flush=True)


def main() -> int:
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, _handle_signal)

    _verify_compute_mode()
    workdir = Path(os.environ.get("CALO_WORKDIR", "/data"))
    workdir.mkdir(parents=True, exist_ok=True)
    runtime_directory = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp/calo-runtime"))
    for directory in (
        Path(os.environ.get("HOME", str(workdir / "home"))),
        Path(os.environ.get("XDG_CACHE_HOME", "/tmp/calo-cache")),
        Path(os.environ.get("XDG_CONFIG_HOME", str(workdir / "home/.config"))),
        runtime_directory,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    runtime_directory.chmod(0o700)
    os.chdir(workdir)

    display = os.environ.get("DISPLAY", ":99")
    port = os.environ.get("CALO_CONTAINER_PORT", "6080")
    _start("Xvfb", display, "-screen", "0", "1600x1000x24", "-nolisten", "tcp")
    time.sleep(0.5)
    _start("openbox", "--sm-disable")
    _start(
        "x11vnc",
        "-display",
        display,
        "-forever",
        "-shared",
        "-nopw",
        "-localhost",
        "-rfbport",
        "5900",
    )
    _start(
        "websockify",
        "--web=/usr/share/novnc",
        port,
        "127.0.0.1:5900",
    )
    app = _start(sys.executable, "-m", "calo_rpd_studio.app.application")
    pid_file = Path(os.environ.get("CALO_APP_PID_FILE", "/tmp/calo-app.pid"))
    try:
        return _supervise_children(app, pid_file)
    finally:
        pid_file.unlink(missing_ok=True)
        _terminate_children()


if __name__ == "__main__":
    raise SystemExit(main())
