"""Secondary Intel-XPU runtime worker and probe entry point.

A CUDA-enabled PyTorch wheel and an XPU-enabled PyTorch wheel are hardware-specific builds.  When the
bootstrap wizard provisions a secondary XPU virtual environment, the main application can launch an
independent CALO job in that interpreter without replacing the primary CUDA-enabled PyTorch install.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import pickle
from pathlib import Path
import traceback


def _probe(run_test: bool = True) -> dict:
    try:
        import torch

        available = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
        devices = []
        test_passed = False
        if available:
            for index in range(int(torch.xpu.device_count())):
                properties = torch.xpu.get_device_properties(index)
                name = str(getattr(properties, "name", f"Intel XPU {index}"))
                total = int(getattr(properties, "total_memory", 0) or 0)
                memory_percent = 0.0
                try:
                    free_bytes, total_bytes = torch.xpu.memory.mem_get_info(index)
                    memory_percent = 100.0 * (total_bytes - free_bytes) / max(total_bytes, 1)
                except Exception:
                    try:
                        allocated = int(torch.xpu.memory.memory_allocated(index))
                        memory_percent = 100.0 * allocated / max(total, 1) if total else 0.0
                    except Exception:
                        pass
                utilization = None
                try:
                    fn = getattr(torch.xpu, "utilization", None)
                    if callable(fn):
                        utilization = float(fn(index))
                except Exception:
                    utilization = None
                devices.append(
                    {
                        "device_id": f"xpu:{index}",
                        "index": index,
                        "name": name,
                        "memory_percent": float(memory_percent),
                        "utilization_percent": utilization,
                        "telemetry": "PyTorch XPU"
                        if utilization is not None
                        else "XPU memory + job-cap admission",
                    }
                )
            if run_test:
                x = torch.randn((128, 128), device="xpu:0")
                y = x @ x
                torch.xpu.synchronize()
                test_passed = bool(torch.isfinite(y).all().item())
            else:
                test_passed = True
        return {
            "available": available and test_passed,
            "xpu_available": available,
            "gpu_test_passed": test_passed,
            "torch_version": str(torch.__version__),
            "devices": devices,
            "error": "",
        }
    except Exception as exc:
        return {
            "available": False,
            "xpu_available": False,
            "gpu_test_passed": False,
            "torch_version": "",
            "devices": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def _configure_item_device(config, mode: str, item, device: str):
    from calo_rpd_studio.compute.resource_scheduler import item_uses_calo_ai
    from calo_rpd_studio.continuation.runtime_binding import bind_exact_run_checkpoint

    local_config = deepcopy(config)
    if item_uses_calo_ai(mode, item):
        parameters = dict(local_config.algorithm_parameters)
        calo_parameters = dict(parameters.get("CALO", {}))
        calo_parameters["inference_device"] = str(device)
        parameters["CALO"] = calo_parameters
        local_config.algorithm_parameters = parameters
    return bind_exact_run_checkpoint(local_config, item)


def _run_job(input_path: Path, output_path: Path) -> int:
    from calo_rpd_studio.experiments.calo_ablation import run_ablation
    from calo_rpd_studio.experiments.execution_plan import ABLATION_MODE, COMPARISON_MODE
    from calo_rpd_studio.experiments.experiment_runner import failed_run_from_exception, run_single

    with input_path.open("rb") as handle:
        payload = pickle.load(handle)
    config = payload["config"]
    mode = payload["mode"]
    item = payload["item"]
    seeds = payload["seeds"]
    device = str(payload.get("device", "xpu:0"))
    local_config = _configure_item_device(config, mode, item, device)

    def progress(data: dict) -> None:
        message = dict(data)
        message.update(
            {
                "job_index": item.job_index,
                "run_index": item.run_index + 1,
                "algorithm": item.label,
                "compute_device": device,
            }
        )
        print("CALO_PROGRESS " + json.dumps(message, default=str), flush=True)

    try:
        if mode == COMPARISON_MODE:
            completed = run_single(local_config, item.label, item.run_index, seeds, progress, None)
        elif mode == ABLATION_MODE:
            completed = run_ablation(
                local_config, item.ablation_spec, item.run_index, seeds, progress, None
            )
        else:
            raise ValueError(f"Unknown execution mode: {mode}")
        completed.result.metadata["compute_device_assignment"] = device
        completed.result.metadata["execution_backend"] = str(local_config.execution_backend)
        result = ("completed", item, completed)
    except Exception as exc:
        result = (
            "failed",
            item,
            failed_run_from_exception(item.label, item.run_index, seeds, exc),
        )
    with output_path.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


def _train(input_path: Path, output_path: Path) -> int:
    from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy

    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = TrainingConfig(**payload["config"])
    config.ppo_device = "xpu:0"

    def progress(percent: int, detail: str) -> None:
        print(
            "CALO_TRAIN_PROGRESS " + json.dumps({"percent": int(percent), "detail": str(detail)}),
            flush=True,
        )

    try:
        train_policy(config, str(output_path), progress_callback=progress)
    except Exception as exc:
        print(
            "CALO_TRAIN_ERROR "
            + json.dumps(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            ),
            flush=True,
        )
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args(argv)
    if args.probe or args.telemetry:
        print(json.dumps(_probe(run_test=not args.telemetry)), flush=True)
        return 0
    if not args.input or not args.output:
        parser.error("input and output paths are required")
    if args.train:
        return _train(Path(args.input), Path(args.output))
    return _run_job(Path(args.input), Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
