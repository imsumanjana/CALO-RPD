"""Persistent accelerator worker for an isolated Intel-XPU Python runtime.

The bootstrap may install CUDA PyTorch in the primary environment and Intel-XPU PyTorch in a
secondary environment.  This module keeps that secondary interpreter alive for the complete
campaign and communicates with length-prefixed pickle frames over stdin/stdout.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import argparse
from copy import deepcopy
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time
from typing import Any, BinaryIO

from calo_rpd_studio.accelerated.runtime_context import clear_cross_run_broker, set_cross_run_broker
from calo_rpd_studio.accelerated.throughput_engine import CrossRunBatchBroker, GLOBAL_LEDGER, calibrate_evaluator
from calo_rpd_studio.experiments.calo_ablation import run_ablation
from calo_rpd_studio.experiments.execution_plan import ABLATION_MODE, COMPARISON_MODE
from calo_rpd_studio.experiments.experiment_runner import build_problem, failed_run_from_exception, run_single

_HEADER = struct.Struct("!Q")


def _write_frame(stream: BinaryIO, payload: Any, lock: threading.Lock | None = None) -> None:
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


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(stream: BinaryIO) -> Any:
    header = _read_exact(stream, _HEADER.size)
    length = _HEADER.unpack(header)[0]
    return pickle.loads(_read_exact(stream, int(length)))


def _configure(config, device: str):
    local = deepcopy(config)
    local.runtime_compute_device = device
    parameters = dict(local.algorithm_parameters)
    for name in tuple(getattr(local, "algorithms", ())) + ("CALO", "TLBO"):
        values = dict(parameters.get(name, {}))
        values["execution_device"] = device
        values["optimizer_backend"] = "torch"
        if name == "CALO":
            values["inference_device"] = device
        parameters[name] = values
    local.algorithm_parameters = parameters
    return local


def server(device: str, slots: int, batch_window_ms: float, max_cross_run_batch: int, cross_run_batching: bool) -> int:
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    input_stream = sys.stdin.buffer
    output_stream = sys.stdout.buffer
    write_lock = threading.Lock()
    commands: queue.Queue[Any] = queue.Queue()
    stopped = threading.Event()
    cancel_event = threading.Event()
    broker = None
    if cross_run_batching:
        broker = CrossRunBatchBroker(batch_window_ms=batch_window_ms, max_candidates=max_cross_run_batch)
        set_cross_run_broker(broker)
    executor = ThreadPoolExecutor(max_workers=max(1, slots), thread_name_prefix="CALO-XPU")
    futures = {}

    def reader() -> None:
        try:
            while not stopped.is_set():
                commands.put(_read_frame(input_stream))
        except EOFError:
            commands.put({"action": "shutdown"})
        except Exception as exc:
            commands.put({"action": "fatal", "message": str(exc)})

    threading.Thread(target=reader, daemon=True).start()

    def execute(command):
        config = _configure(command["config"], device)
        item = command["item"]
        seeds = command["seeds"]
        job_id = str(command["job_id"])
        mode = str(command["mode"])
        evaluation_span = max(1, int(config.budget.max_evaluations))
        evaluation_step = max(1, evaluation_span // 100)
        telemetry_iteration_interval = max(1, int(getattr(config, "telemetry_iteration_interval", 10)))
        last_emit = 0.0
        last_evaluations = -1
        last_iteration = -1

        def progress(payload):
            nonlocal last_emit, last_evaluations
            now = time.monotonic()
            evaluations = int(payload.get("evaluations", 0))
            if evaluations == 0 or evaluations >= evaluation_span or evaluations - last_evaluations >= evaluation_step or now - last_emit >= 0.25:
                data = dict(payload)
                data.update({"job_id": job_id, "job_index": item.job_index, "run_index": item.run_index + 1, "algorithm": item.label, "compute_device": device, "throughput_engine": "persistent_xpu_cross_run_batching"})
                _write_frame(output_stream, {"kind": "progress", "payload": data}, write_lock)
                last_emit = now
                last_evaluations = evaluations
                last_iteration = iteration

        try:
            if mode == COMPARISON_MODE:
                completed = run_single(config, item.label, item.run_index, seeds, progress, cancel_event.is_set)
            elif mode == ABLATION_MODE:
                completed = run_ablation(config, item.ablation_spec, item.run_index, seeds, progress, cancel_event.is_set)
            else:
                raise ValueError(f"Unsupported experiment mode: {mode}")
            completed.result.metadata.update({"compute_device_assignment": device, "execution_backend": str(config.execution_backend), "persistent_accelerator_worker": True, "cross_run_batching": bool(cross_run_batching), "xpu_sidecar_runtime": True, "throughput_stage_profile": GLOBAL_LEDGER.snapshot()})
            return {"kind": "completed", "job_id": job_id, "item": item, "payload": completed}
        except Exception as exc:
            return {"kind": "failed", "job_id": job_id, "item": item, "payload": failed_run_from_exception(item.label, item.run_index, seeds, exc)}

    try:
        while True:
            for future in [future for future in list(futures) if future.done()]:
                futures.pop(future, None)
                _write_frame(output_stream, future.result(), write_lock)
            try:
                command = commands.get(timeout=0.05)
            except queue.Empty:
                continue
            action = str(command.get("action", ""))
            if action in {"shutdown", "fatal"}:
                break
            if action == "cancel":
                cancel_event.set()
                continue
            if action == "calibrate":
                try:
                    problem = build_problem(_configure(command["config"], device), int(command["scenario_seed"]))
                    record = calibrate_evaluator(problem, batch_sizes=command.get("batch_sizes", (16, 32, 64, 128, 256)), repetitions=int(command.get("repetitions", 1)))
                    _write_frame(output_stream, {"kind": "calibration", "request_id": str(command["request_id"]), "device": device, "record": record}, write_lock)
                except Exception as exc:
                    _write_frame(output_stream, {"kind": "calibration_error", "request_id": str(command["request_id"]), "device": device, "message": str(exc)}, write_lock)
                continue
            if action == "job":
                future = executor.submit(execute, command)
                futures[future] = str(command["job_id"])
    finally:
        stopped.set()
        cancel_event.set()
        executor.shutdown(wait=True, cancel_futures=False)
        for future in list(futures):
            try:
                _write_frame(output_stream, future.result(), write_lock)
            except Exception as exc:
                _write_frame(output_stream, {"kind": "service_error", "message": str(exc)}, write_lock)
        if broker is not None:
            broker.close()
            clear_cross_run_broker()
    return 0


class PersistentSidecarPool:
    def __init__(self, interpreter: str, device: str, *, slots: int, progress_queue, batch_window_ms: float = 4.0, max_cross_run_batch: int = 4096, cross_run_batching: bool = True):
        self.device = device
        self.slots = max(1, int(slots))
        self.progress_queue = progress_queue
        self.active_jobs: set[str] = set()
        self.results: queue.Queue[dict[str, Any]] = queue.Queue()
        self._write_lock = threading.Lock()
        self._stderr_tail: list[str] = []
        self.process = subprocess.Popen(
            [interpreter, "-m", "calo_rpd_studio.compute.persistent_accelerator_sidecar", "--server", "--device", device, "--slots", str(self.slots), "--batch-window-ms", str(float(batch_window_ms)), "--max-batch", str(int(max_cross_run_batch)), "--cross-run-batching", "1" if cross_run_batching else "0"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        threading.Thread(target=self._reader, daemon=True).start()
        threading.Thread(target=self._stderr_reader, daemon=True).start()

    @property
    def available_slots(self) -> int:
        return max(0, self.slots - len(self.active_jobs))

    def _send(self, payload) -> None:
        if self.process.stdin is None:
            raise RuntimeError("XPU sidecar stdin is unavailable")
        _write_frame(self.process.stdin, payload, self._write_lock)

    def _reader(self) -> None:
        if self.process.stdout is None:
            return
        try:
            while True:
                message = _read_frame(self.process.stdout)
                if message.get("kind") == "progress":
                    self.progress_queue.put(message["payload"])
                    continue
                job_id = str(message.get("job_id", ""))
                if job_id:
                    self.active_jobs.discard(job_id)
                self.results.put(message)
        except Exception:
            return

    def _stderr_reader(self) -> None:
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

    def submit(self, job_id: str, config, mode: str, item, seeds) -> None:
        job_id = str(job_id)
        self.active_jobs.add(job_id)
        self._send({"action": "job", "job_id": job_id, "config": config, "mode": mode, "item": item, "seeds": seeds})

    def calibrate(self, request_id: str, config, scenario_seed: int, *, batch_sizes=(16, 32, 64, 128, 256), repetitions: int = 1) -> None:
        self._send({"action": "calibrate", "request_id": str(request_id), "config": config, "scenario_seed": int(scenario_seed), "batch_sizes": tuple(batch_sizes), "repetitions": int(repetitions)})

    def poll(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.results.get_nowait())
            except queue.Empty:
                break
        return out

    def cancel(self) -> None:
        try:
            self._send({"action": "cancel"})
        except Exception:
            pass

    def close(self, timeout: float = 30.0) -> None:
        if self.process.poll() is None:
            try:
                self._send({"action": "shutdown"})
            except Exception:
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.terminate()
        if self.process.poll() is None:
            self.process.kill()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--batch-window-ms", type=float, default=4.0)
    parser.add_argument("--max-batch", type=int, default=4096)
    parser.add_argument("--cross-run-batching", type=int, choices=(0, 1), default=1)
    args = parser.parse_args(argv)
    if not args.server:
        parser.error("Only --server mode is supported")
    return server(args.device, args.slots, args.batch_window_ms, args.max_batch, bool(args.cross_run_batching))


if __name__ == "__main__":
    raise SystemExit(main())
