"""Helpers for executing jobs or policy training in the secondary Intel-XPU runtime."""

from __future__ import annotations

import logging

from dataclasses import asdict
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile

from calo_rpd_studio.compute.resource_scheduler import configured_xpu_interpreter
from calo_rpd_studio.experiments.experiment_runner import failed_run_from_exception


_LOG = logging.getLogger(__name__)

def execute_xpu_job(config, mode, item, seeds, progress_queue, cancel_event, device: str = "xpu:0"):
    """Run one optimizer job in the isolated XPU interpreter and relay progress to the parent."""
    interpreter = configured_xpu_interpreter()
    if not interpreter:
        failure = failed_run_from_exception(
            item.label,
            item.run_index,
            seeds,
            RuntimeError("No verified secondary Intel XPU runtime is configured."),
        )
        return "failed", item, failure

    with tempfile.TemporaryDirectory(prefix="calo_xpu_job_") as tmp:
        root = Path(tmp)
        input_path = root / "input.pkl"
        output_path = root / "output.pkl"
        with input_path.open("wb") as handle:
            pickle.dump(
                {
                    "config": config,
                    "mode": mode,
                    "item": item,
                    "seeds": seeds,
                    "device": device,
                },
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        process = subprocess.Popen(
            [
                interpreter,
                "-m",
                "calo_rpd_studio.compute.xpu_worker",
                str(input_path),
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
        assert process.stdout is not None
        for line in process.stdout:
            if cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                failure = failed_run_from_exception(
                    item.label,
                    item.run_index,
                    seeds,
                    RuntimeError("XPU job cancelled before completion."),
                )
                return "failed", item, failure
            line = line.strip()
            if line.startswith("CALO_PROGRESS "):
                try:
                    progress_queue.put(json.loads(line[len("CALO_PROGRESS ") :]))
                except Exception:
                    _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)
        code = process.wait()
        if code != 0 or not output_path.exists():
            failure = failed_run_from_exception(
                item.label,
                item.run_index,
                seeds,
                RuntimeError(f"Intel XPU worker exited with code {code}."),
            )
            return "failed", item, failure
        with output_path.open("rb") as handle:
            return pickle.load(handle)


def train_policy_in_xpu_sidecar(
    config, output_path: str, progress_callback=None, cancel_callback=None
) -> str:
    """Run centralized PPO updates in the secondary XPU runtime.

    Cancellation terminates the sidecar process between emitted progress updates.  No partially
    written checkpoint is treated as a successful training result.
    """
    interpreter = configured_xpu_interpreter()
    if not interpreter:
        raise RuntimeError("No verified secondary Intel XPU runtime is configured.")

    from calo_rpd_studio.algorithms.calo.training import TrainingCancelled

    with tempfile.TemporaryDirectory(prefix="calo_xpu_train_") as tmp:
        config_path = Path(tmp) / "training.json"
        payload = asdict(config)
        # JSON has no tuple type; TrainingConfig accepts the resulting list for development_cases.
        config_path.write_text(json.dumps({"config": payload}, indent=2), encoding="utf-8")
        process = subprocess.Popen(
            [
                interpreter,
                "-m",
                "calo_rpd_studio.compute.xpu_worker",
                "--train",
                str(config_path),
                str(output_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
        assert process.stdout is not None
        last_error = ""
        for line in process.stdout:
            if cancel_callback and cancel_callback():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise TrainingCancelled("CALO policy training was cancelled safely.")
            text = line.strip()
            if text.startswith("CALO_TRAIN_PROGRESS "):
                try:
                    data = json.loads(text[len("CALO_TRAIN_PROGRESS ") :])
                    if progress_callback:
                        progress_callback(int(data.get("percent", 0)), str(data.get("detail", "")))
                except Exception:
                    _LOG.debug("Suppressed non-fatal cleanup/probe exception", exc_info=True)
            elif text.startswith("CALO_TRAIN_ERROR "):
                try:
                    last_error = str(json.loads(text[len("CALO_TRAIN_ERROR ") :]).get("error", ""))
                except Exception:
                    last_error = text
        code = process.wait()
        if code != 0:
            raise RuntimeError(last_error or f"Intel XPU training worker exited with code {code}.")
    return str(output_path)
