"""Persistent policy-rollout actor clients for CUDA and isolated Intel-XPU runtimes."""
from __future__ import annotations

import pickle
import queue
import struct
import subprocess
import threading
import time
from typing import Any, BinaryIO

_HEADER = struct.Struct("!Q")
_MAX_FRAME_BYTES = 512 * 1024 * 1024


def write_frame(stream: BinaryIO, payload: Any, lock: threading.Lock | None = None) -> None:
    data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if lock is None:
        stream.write(_HEADER.pack(len(data)))
        stream.write(data)
        stream.flush()
        return
    with lock:
        stream.write(_HEADER.pack(len(data)))
        stream.write(data)
        stream.flush()


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: BinaryIO) -> Any:
    header = read_exact(stream, _HEADER.size)
    length = int(_HEADER.unpack(header)[0])
    if length <= 0 or length > _MAX_FRAME_BYTES:
        raise ValueError(f"Invalid local worker frame length: {length} bytes")
    payload = pickle.loads(read_exact(stream, length))  # nosec B301 -- trusted same-host child process
    if not isinstance(payload, dict):
        raise ValueError("Local worker protocol requires a dictionary frame")
    return payload


class PersistentTrainingActorClient:
    """One long-lived actor interpreter with one resident accelerator context and network object."""

    def __init__(self, interpreter: str, device: str, lane: str) -> None:
        self.device = str(device)
        self.lane = str(lane)
        self._write_lock = threading.Lock()
        self._results: queue.Queue[Any] = queue.Queue()
        self._stderr_tail: list[str] = []
        self.process = subprocess.Popen(
            [
                interpreter,
                "-m",
                "calo_rpd_studio.compute.training_actor_worker",
                "--server",
                "--device",
                self.device,
                "--lane",
                self.lane,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

    def _reader(self) -> None:
        if self.process.stdout is None:
            return
        try:
            while True:
                self._results.put(read_frame(self.process.stdout))
        except Exception:
            return

    def _read_stderr(self) -> None:
        if self.process.stderr is None:
            return
        try:
            for raw in iter(self.process.stderr.readline, b""):
                text = raw.decode("utf-8", errors="replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    del self._stderr_tail[:-80]
        except Exception:
            return

    def request(self, payload: dict[str, Any], timeout: float | None = None) -> Any:
        if self.process.poll() is not None:
            stderr = ""
            if self.process.stderr is not None:
                try:
                    stderr = self.process.stderr.read().decode("utf-8", errors="replace")[-4000:]
                except Exception:
                    pass
            raise RuntimeError(
                f"Persistent {self.lane.upper()} actor exited with code {self.process.returncode}. {stderr}"
            )
        if self.process.stdin is None:
            raise RuntimeError("Persistent actor stdin is unavailable")
        write_frame(self.process.stdin, {"action": "collect", "payload": payload}, self._write_lock)
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            wait_time = 0.25
            if deadline is not None:
                wait_time = max(0.01, min(wait_time, deadline - time.monotonic()))
            try:
                response = self._results.get(timeout=wait_time)
                break
            except queue.Empty:
                if self.process.poll() is not None:
                    detail = "\n".join(self._stderr_tail[-20:])
                    raise RuntimeError(
                        f"Persistent {self.lane.upper()} actor exited with code "
                        f"{self.process.returncode}." + (f"\n{detail}" if detail else "")
                    )
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError(f"Persistent {self.lane.upper()} actor request timed out")
        if not response.get("ok", False):
            raise RuntimeError(str(response.get("error", "Persistent actor failed")))
        return response["result"]

    def close(self, timeout: float = 15.0) -> None:
        if self.process.poll() is None and self.process.stdin is not None:
            try:
                write_frame(self.process.stdin, {"action": "shutdown"}, self._write_lock)
            except Exception:
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        if self.process.poll() is None:
            self.process.kill()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
