"""Persistent per-device workers for the v3.4 batched throughput engine.

One process owns one CUDA/XPU context for the complete campaign.  Multiple independent optimizer
runs execute as threads inside that process and submit compatible population requests to one
``CrossRunBatchBroker``.  This removes per-run accelerator initialization and enables real
cross-run candidate batching without changing optimizer equations or evaluation budgets.
"""

from __future__ import annotations

import logging

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing as mp
import queue
import time
from typing import Any

from calo_rpd_studio.accelerated.runtime_context import clear_cross_run_broker, set_cross_run_broker
from calo_rpd_studio.accelerated.throughput_engine import (
    CrossRunBatchBroker,
    GLOBAL_LEDGER,
    calibrate_evaluator,
)
from calo_rpd_studio.experiments.calo_ablation import run_ablation
from calo_rpd_studio.experiments.execution_plan import ABLATION_MODE, COMPARISON_MODE
from calo_rpd_studio.continuation.runtime_binding import bind_exact_run_checkpoint
from calo_rpd_studio.experiments.experiment_runner import (
    build_problem,
    failed_run_from_exception,
    run_single,
)


_LOG = logging.getLogger(__name__)

def configure_item_device(config, compute_device: str, item=None):
    local = deepcopy(config)
    local.runtime_compute_device = str(compute_device)
    parameters = dict(local.algorithm_parameters)
    for algorithm_name in tuple(getattr(local, "algorithms", ())) + ("CALO", "TLBO"):
        values = dict(parameters.get(algorithm_name, {}))
        values["execution_device"] = str(compute_device)
        if str(getattr(local, "scientific_backend", "cpu_reference")) == "torch_fp64":
            values["optimizer_backend"] = "torch"
        if algorithm_name == "CALO":
            # CALO cognitive/control remains CPU/NumPy in v5.8; keep the tiny policy on CPU to
            # avoid a CUDA/XPU synchronize+host-copy every decision. Heavy ORPD evaluation still
            # uses compute_device. Explicit experimental overrides can be applied outside this
            # strict campaign binding if a fully device-resident control plane is introduced.
            values["inference_device"] = "cpu"
            values["policy_control_plane"] = "cpu_no_device_roundtrip_v57"
        parameters[algorithm_name] = values
    local.algorithm_parameters = parameters
    return bind_exact_run_checkpoint(local, item)


def _configure_numeric_threads() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)


def _worker_main(
    command_queue,
    result_queue,
    progress_queue,
    cancel_event,
    device: str,
    slots: int,
    batch_window_ms: float,
    max_cross_run_batch: int,
    cross_run_batching: bool,
):
    _configure_numeric_threads()
    broker = None
    # A broker is useful only when at least two run threads can submit concurrently. Single-slot
    # workers take the direct evaluator path and pay no microbatch wait.
    effective_cross_run_batching = bool(cross_run_batching and int(slots) > 1)
    if effective_cross_run_batching:
        broker = CrossRunBatchBroker(
            batch_window_ms=batch_window_ms,
            max_candidates=max_cross_run_batch,
        )
        set_cross_run_broker(broker)
    executor = ThreadPoolExecutor(
        max_workers=max(1, int(slots)), thread_name_prefix=f"CALO-{device}"
    )
    futures: dict[Any, str] = {}

    def run_job(command):
        job_id = str(command["job_id"])
        item = command["item"]
        config = configure_item_device(command["config"], device, item)
        seeds = command["seeds"]
        mode = str(command["mode"])
        evaluation_span = max(1, int(config.budget.max_evaluations))
        evaluation_step = max(1, evaluation_span // 100)
        telemetry_iteration_interval = max(
            1, int(getattr(config, "telemetry_iteration_interval", 10))
        )
        last_emit = 0.0
        last_evaluations = -1
        last_iteration = -1

        def cancelled() -> bool:
            return bool(cancel_event.is_set())

        def emit(payload: dict) -> None:
            nonlocal last_emit, last_evaluations, last_iteration
            now = time.monotonic()
            evaluations = int(payload.get("evaluations", 0))
            iteration = int(payload.get("iteration", 0))
            if (
                evaluations == 0
                or evaluations >= evaluation_span
                or iteration - last_iteration >= telemetry_iteration_interval
                or evaluations - last_evaluations >= evaluation_step
                or now - last_emit >= 0.50
            ):
                data = dict(payload)
                data.update(
                    {
                        "job_id": job_id,
                        "job_index": item.job_index,
                        "run_index": item.run_index + 1,
                        "algorithm": item.label,
                        "compute_device": device,
                        "throughput_engine": "device_resident_cross_run_batching_v3.4",
                    }
                )
                progress_queue.put(data)
                last_emit = now
                last_evaluations = evaluations
                last_iteration = iteration

        try:
            if mode == COMPARISON_MODE:
                completed = run_single(config, item.label, item.run_index, seeds, emit, cancelled)
            elif mode == ABLATION_MODE:
                completed = run_ablation(
                    config, item.ablation_spec, item.run_index, seeds, emit, cancelled
                )
            else:
                raise ValueError(f"Unsupported experiment mode: {mode}")
            completed.result.metadata.update(
                {
                    "compute_device_assignment": device,
                    "execution_backend": str(config.execution_backend),
                    "persistent_accelerator_worker": True,
                    "throughput_engine_version": "3.3",
                    "device_resident_execution": bool(
                        getattr(config, "device_resident_execution", True)
                    ),
                    "planned_device_share": {
                        "cuda": int(getattr(config, "cuda_task_share", 80)),
                        "xpu": int(getattr(config, "xpu_task_share", 10)),
                        "cpu": int(getattr(config, "cpu_task_share", 10)),
                    },
                    "cross_run_batching": bool(effective_cross_run_batching),
                    "cross_run_batch_window_ms": float(batch_window_ms),
                    "max_cross_run_batch": int(max_cross_run_batch),
                    "throughput_stage_profile": GLOBAL_LEDGER.snapshot(),
                }
            )
            kind = (
                "interrupted"
                if cancel_event.is_set()
                and int(completed.result.evaluations) < int(config.budget.max_evaluations)
                else "completed"
            )
            return {"kind": kind, "job_id": job_id, "item": item, "payload": completed}
        except Exception as exc:
            _LOG.exception("Persistent accelerator job failed; returning structured failure")
            return {
                "kind": "failed",
                "job_id": job_id,
                "item": item,
                "payload": failed_run_from_exception(item.label, item.run_index, seeds, exc),
            }

    try:
        while True:
            # Publish completed jobs first.
            completed_futures = [future for future in list(futures) if future.done()]
            for future in completed_futures:
                job_id = futures.pop(future)
                try:
                    result_queue.put(future.result())
                except Exception as exc:
                    result_queue.put(
                        {"kind": "service_error", "job_id": job_id, "message": str(exc)}
                    )

            try:
                command = command_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            action = str(command.get("action", ""))
            if action == "shutdown":
                break
            if action == "calibrate":
                try:
                    local = configure_item_device(command["config"], device)
                    problem = build_problem(local, int(command["scenario_seed"]))
                    record = calibrate_evaluator(
                        problem,
                        batch_sizes=command.get("batch_sizes", (16, 32, 64, 128, 256)),
                        repetitions=int(command.get("repetitions", 1)),
                        seed=int(command.get("seed", 31415)),
                    )
                    result_queue.put(
                        {
                            "kind": "calibration",
                            "request_id": str(command["request_id"]),
                            "device": device,
                            "record": record,
                        }
                    )
                except Exception as exc:
                    result_queue.put(
                        {
                            "kind": "calibration_error",
                            "request_id": str(command["request_id"]),
                            "device": device,
                            "message": str(exc),
                        }
                    )
                continue
            if action == "job":
                future = executor.submit(run_job, command)
                futures[future] = str(command["job_id"])
                continue
    finally:
        executor.shutdown(wait=True, cancel_futures=False)
        # Flush results completed during shutdown.
        for future, job_id in list(futures.items()):
            try:
                result_queue.put(future.result())
            except Exception as exc:
                result_queue.put({"kind": "service_error", "job_id": job_id, "message": str(exc)})
        if broker is not None:
            broker.close()
            clear_cross_run_broker()


class PersistentAcceleratorPool:
    """Parent-side handle for one persistent device process."""

    def __init__(
        self,
        device: str,
        *,
        slots: int,
        progress_queue,
        cancel_event,
        batch_window_ms: float = 4.0,
        max_cross_run_batch: int = 4096,
        cross_run_batching: bool = True,
        context=None,
    ) -> None:
        context = context or mp.get_context("spawn")
        self.device = str(device)
        self.slots = max(1, int(slots))
        self.command_queue = context.Queue()
        self.result_queue = context.Queue()
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.process = context.Process(
            target=_worker_main,
            args=(
                self.command_queue,
                self.result_queue,
                progress_queue,
                cancel_event,
                self.device,
                self.slots,
                float(batch_window_ms),
                int(max_cross_run_batch),
                bool(cross_run_batching),
            ),
            daemon=True,
            name=f"CALO-Persistent-{self.device}",
        )
        self.process.start()
        self.active_jobs: set[str] = set()

    @property
    def available_slots(self) -> int:
        return max(0, self.slots - len(self.active_jobs))

    def submit(self, job_id: str, config, mode: str, item, seeds) -> None:
        job_id = str(job_id)
        self.active_jobs.add(job_id)
        self.command_queue.put(
            {
                "action": "job",
                "job_id": job_id,
                "config": config,
                "mode": mode,
                "item": item,
                "seeds": seeds,
            }
        )

    def calibrate(
        self,
        request_id: str,
        config,
        scenario_seed: int,
        *,
        batch_sizes=(16, 32, 64, 128, 256),
        repetitions: int = 1,
    ) -> None:
        self.command_queue.put(
            {
                "action": "calibrate",
                "request_id": str(request_id),
                "config": config,
                "scenario_seed": int(scenario_seed),
                "batch_sizes": tuple(int(value) for value in batch_sizes),
                "repetitions": int(repetitions),
            }
        )

    def poll(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                item = self.result_queue.get_nowait()
            except queue.Empty:
                break
            job_id = str(item.get("job_id", ""))
            if job_id:
                self.active_jobs.discard(job_id)
            out.append(item)
        return out

    def close(self, timeout: float = 30.0) -> None:
        if self.process.is_alive():
            self.command_queue.put({"action": "shutdown"})
            self.process.join(timeout=timeout)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
