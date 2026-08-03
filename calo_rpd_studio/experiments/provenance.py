"""Machine and software provenance capture."""

from __future__ import annotations

import logging

from calo_rpd_studio.version import VERSION

import importlib.metadata as md
from pathlib import Path
import platform
import subprocess

import psutil

PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "PyQt6",
    "torch",
    "pypower",
    "PyYAML",
)


_LOG = logging.getLogger(__name__)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=Path.cwd(),
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def _torch_accelerator() -> dict:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        return {
            "cuda_available": cuda_available,
            "torch_cuda_runtime": str(torch.version.cuda or ""),
            "cuda_device_names": [
                str(torch.cuda.get_device_name(index))
                for index in range(int(torch.cuda.device_count()))
            ]
            if cuda_available
            else [],
            "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            # Backward-compatible aliases retained for existing result consumers.
            "gpu_name": str(torch.cuda.get_device_name(0)) if cuda_available else "",
            "gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
        }
    except Exception as exc:
        _LOG.warning(
            "Accelerator provenance probe failed; recording explicit unavailable/error provenance",
            exc_info=True,
        )
        return {
            "cuda_available": False,
            "torch_cuda_runtime": "",
            "cuda_device_names": [],
            "cuda_device_count": 0,
            "gpu_name": "",
            "gpu_count": 0,
            "accelerator_error": f"{type(exc).__name__}: {exc}",
        }


def collect_provenance() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "software_version": VERSION,
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_bytes": psutil.virtual_memory().total,
        "dependencies": versions,
        "accelerator": _torch_accelerator(),
    }
