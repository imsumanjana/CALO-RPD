"""Hardware-aware prerequisite detection, installation, and verification.

This module intentionally uses only the Python standard library so it can run before PyQt6,
PyTorch, NumPy, PYPOWER, or the CALO-RPD package itself are installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from typing import Callable, Iterable

try:
    from importlib.metadata import version as distribution_version

    APP_VERSION = distribution_version("calo-rpd-studio")
except Exception:
    APP_VERSION = "6.9.0"
STATE_DIR = Path.home() / ".calo_rpd_studio"
STATE_FILE = STATE_DIR / "environment_state.json"
CORE_REQUIREMENTS_FILE = "requirements-core.txt"
CORE_REQUIREMENTS: tuple[str, ...] = (
    "numpy>=1.26,<2.4",
    "scipy>=1.12,<2",
    "pandas>=2.1,<3",
    "matplotlib>=3.8,<4",
    "PyQt6>=6.6,<7",
    "PYPOWER>=5.1.18,<6",
    "PyYAML>=6,<7",
    "psutil>=5.9,<8",
    "nvidia-ml-py>=13,<14",
)

CORE_DISTRIBUTIONS: tuple[tuple[str, str], ...] = (
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "pandas"),
    ("matplotlib", "Matplotlib"),
    ("PyQt6", "PyQt6"),
    ("PYPOWER", "PYPOWER"),
    ("PyYAML", "PyYAML"),
    ("psutil", "psutil"),
    ("nvidia-ml-py", "NVIDIA NVML Python telemetry"),
)

# Candidate official PyTorch wheel channels.  The installer chooses the newest channel that does
# not exceed the maximum CUDA runtime reported by the NVIDIA driver, then falls back through older
# channels.  Keeping a fallback list is more robust than assuming one wheel channel forever.
CUDA_CHANNELS: tuple[tuple[float, str], ...] = (
    (13.2, "cu132"),
    (13.0, "cu130"),
    (12.8, "cu128"),
    (12.6, "cu126"),
    (12.4, "cu124"),
    (12.1, "cu121"),
    (11.8, "cu118"),
)
PYTORCH_INDEX_ROOT = "https://download.pytorch.org/whl"
COMPUTE_REQUIREMENTS: tuple[str, ...] = (
    "numpy>=1.26,<2.4",
    "scipy>=1.12,<2",
    "PYPOWER>=5.1.18,<6",
    "PyYAML>=6,<7",
    "psutil>=5.9,<8",
    "nvidia-ml-py>=13,<14",
)


@dataclass(slots=True)
class NvidiaInfo:
    detected: bool = False
    name: str = ""
    driver_version: str = ""
    max_cuda_version: str = ""
    error: str = ""


@dataclass(slots=True)
class TorchInfo:
    installed: bool = False
    version: str = ""
    cuda_available: bool = False
    cuda_runtime: str = ""
    device_name: str = ""
    gpu_test_passed: bool = False
    error: str = ""


@dataclass(slots=True)
class EnvironmentReport:
    python_ok: bool
    python_version: str
    interpreter: str
    virtual_environment: bool
    core_packages: dict[str, str]
    missing_core_packages: list[str]
    nvidia: NvidiaInfo
    torch: TorchInfo
    mandatory_ready: bool
    gpu_ready: bool
    recommended_backend: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class InstallProgress:
    """Structured prerequisite-installation progress for the bootstrap GUI.

    ``current_bytes``/``total_bytes`` describe the currently downloading artifact, not the
    complete pip transaction. pip discovers artifacts during dependency resolution, so a truthful
    aggregate byte total is generally unavailable before downloads begin.
    """

    phase: str = "idle"
    phase_index: int = 0
    phase_count: int = 7
    overall_percent: float = 0.0
    item: str = ""
    current_bytes: int = 0
    total_bytes: int = 0
    download_percent: float = 0.0
    speed_bytes_per_second: float = 0.0
    eta_seconds: float | None = None
    indeterminate: bool = False
    message: str = ""


def _run(
    command: list[str], timeout: int = 120, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
    )


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def detect_nvidia() -> NvidiaInfo:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return NvidiaInfo(error="nvidia-smi was not found")
    query = [
        executable,
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    result = _run(query, timeout=20)
    if result.returncode != 0 or not result.stdout.strip():
        return NvidiaInfo(error=(result.stderr or result.stdout or "nvidia-smi failed").strip())
    first = result.stdout.splitlines()[0]
    parts = [item.strip() for item in first.split(",")]
    name = parts[0] if parts else "NVIDIA GPU"
    driver = parts[1] if len(parts) > 1 else ""

    summary = _run([executable], timeout=20)
    text = f"{summary.stdout}\n{summary.stderr}"
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", text)
    cuda = match.group(1) if match else ""
    return NvidiaInfo(True, name, driver, cuda, "")


def detect_torch() -> TorchInfo:
    script = r"""
import json
try:
    import torch
    data = {
        "installed": True,
        "version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_runtime": str(torch.version.cuda or ""),
        "device_name": "",
        "gpu_test_passed": False,
        "error": "",
    }
    if data["cuda_available"]:
        data["device_name"] = str(torch.cuda.get_device_name(0))
        x = torch.randn((256, 256), device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        data["gpu_test_passed"] = bool(y.is_cuda and torch.isfinite(y).all().item())
    print(json.dumps(data))
except Exception as exc:
    print(json.dumps({"installed": False, "version": "", "cuda_available": False,
                      "cuda_runtime": "", "device_name": "", "gpu_test_passed": False,
                      "error": f"{type(exc).__name__}: {exc}"}))
"""
    result = _run([sys.executable, "-c", script], timeout=90)
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        return TorchInfo(**payload)
    except Exception:
        return TorchInfo(
            error=(result.stderr or result.stdout or "Unable to inspect PyTorch").strip()
        )


def scan_environment() -> EnvironmentReport:
    python_ok = sys.version_info >= (3, 11)
    versions: dict[str, str] = {}
    missing: list[str] = []
    for distribution, label in CORE_DISTRIBUTIONS:
        version = _distribution_version(distribution)
        versions[label] = version
        if not version:
            missing.append(label)

    nvidia = detect_nvidia()
    torch = detect_torch()

    cuda_ready = bool(torch.cuda_available and torch.gpu_test_passed)
    gpu_ready = bool(cuda_ready)
    mandatory_ready = bool(python_ok and not missing and torch.installed)

    if cuda_ready:
        recommended_backend = "cuda:0"
    else:
        recommended_backend = "cpu"

    notes: list[str] = []
    if mandatory_ready:
        notes.append("Core prerequisites are ready.")
    else:
        notes.append("Prerequisites are missing or incomplete.")
    if nvidia.detected:
        notes.append(
            "NVIDIA CUDA is ready."
            if cuda_ready
            else "NVIDIA hardware was detected, but CUDA-enabled PyTorch has not passed verification."
        )
    if not nvidia.detected:
        notes.append("No supported GPU accelerator was detected; CPU execution is available.")

    return EnvironmentReport(
        python_ok=python_ok,
        python_version=platform.python_version(),
        interpreter=sys.executable,
        virtual_environment=(getattr(sys, "base_prefix", sys.prefix) != sys.prefix),
        core_packages=versions,
        missing_core_packages=missing,
        nvidia=nvidia,
        torch=torch,
        mandatory_ready=mandatory_ready,
        gpu_ready=gpu_ready,
        recommended_backend=recommended_backend,
        message=" ".join(notes),
    )


def _cuda_version_float(value: str) -> float:
    try:
        major, minor, *_ = value.split(".") + ["0"]
        return float(f"{int(major)}.{int(minor)}")
    except Exception:
        return 0.0


def candidate_torch_channels(nvidia: NvidiaInfo) -> list[str]:
    if not nvidia.detected:
        return ["cpu"]
    maximum = _cuda_version_float(nvidia.max_cuda_version)
    channels = [channel for required, channel in CUDA_CHANNELS if maximum >= required]
    # If nvidia-smi did not expose a CUDA version, still try stable CUDA channels before CPU.
    if not channels:
        channels = ["cu126", "cu121"]
    return channels + ["cpu"]


def project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here.parents[1], Path.cwd()):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return here.parents[1]


def _emit(callback: Callable[[str], None] | None, text: str) -> None:
    if callback:
        callback(text)


def _emit_progress(
    callback: Callable[[InstallProgress], None] | None,
    progress: InstallProgress,
) -> None:
    if callback:
        callback(progress)


def _human_download_item(line: str) -> str:
    """Extract a compact artifact label from common pip download messages."""
    text = line.strip()
    match = re.search(r"(?:Downloading|Using cached)\s+(.+?)(?:\s+\([^)]*\))?$", text, re.I)
    if not match:
        return ""
    candidate = match.group(1).strip()
    # URLs and cache paths are easier to understand as wheel/archive filenames.
    candidate = candidate.split("?")[0].rstrip("/")
    tail = candidate.rsplit("/", 1)[-1]
    return tail or candidate


def _parse_pip_raw_progress(line: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"Progress\s+(\d+)\s+of\s+(\d+)", line.strip())
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _stream_command(
    command: list[str],
    callback: Callable[[str], None] | None,
    cwd: Path | None = None,
    progress_callback: Callable[[InstallProgress], None] | None = None,
    progress_template: InstallProgress | None = None,
) -> int:
    """Run a command while streaming logs and parsing pip ``--progress-bar=raw`` output.

    pip's raw progress lines are stable machine-readable records of the form
    ``Progress <downloaded> of <total>``.  We calculate transfer speed and ETA locally so the
    bootstrap window remains informative even though pip is running in a child process.
    """
    _emit(callback, "> " + " ".join(command))
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
    )
    assert process.stdout is not None

    base = progress_template or InstallProgress()
    current_item = base.item
    samples: deque[tuple[float, int]] = deque(maxlen=12)
    last_total = -1
    for raw_line in process.stdout:
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        item = _human_download_item(stripped)
        if item:
            current_item = item
            samples.clear()
            last_total = -1
            _emit_progress(
                progress_callback,
                InstallProgress(
                    phase=base.phase,
                    phase_index=base.phase_index,
                    phase_count=base.phase_count,
                    overall_percent=base.overall_percent,
                    item=current_item,
                    indeterminate=True,
                    message=f"Preparing download: {current_item}",
                ),
            )

        raw_progress = _parse_pip_raw_progress(stripped)
        if raw_progress is not None:
            current, total = raw_progress
            now = time.monotonic()
            if total != last_total or current == 0:
                samples.clear()
                last_total = total
            samples.append((now, current))
            speed = 0.0
            if len(samples) >= 2:
                elapsed = samples[-1][0] - samples[0][0]
                advanced = samples[-1][1] - samples[0][1]
                if elapsed > 0 and advanced >= 0:
                    speed = advanced / elapsed
            percent = (100.0 * current / total) if total > 0 else 0.0
            eta = ((total - current) / speed) if total > current and speed > 0 else None
            _emit_progress(
                progress_callback,
                InstallProgress(
                    phase=base.phase,
                    phase_index=base.phase_index,
                    phase_count=base.phase_count,
                    overall_percent=base.overall_percent,
                    item=current_item or "Current package",
                    current_bytes=current,
                    total_bytes=total,
                    download_percent=max(0.0, min(100.0, percent)),
                    speed_bytes_per_second=speed,
                    eta_seconds=eta,
                    indeterminate=(total <= 0),
                    message=f"Downloading {current_item or 'package'}",
                ),
            )
            # Raw progress is represented visually; keeping every 250 ms progress line out of the
            # text log prevents the GUI log from becoming enormous.
            continue

        _emit(callback, line)

    code = int(process.wait())
    return code


def _pip_for(
    executable: str | Path,
    args: Iterable[str],
    callback: Callable[[str], None] | None,
    cwd: Path | None = None,
    progress_callback: Callable[[InstallProgress], None] | None = None,
    progress_template: InstallProgress | None = None,
) -> int:
    """Run pip through a specific Python interpreter with truthful raw download telemetry."""
    pip_args = list(args)
    if (
        pip_args
        and pip_args[0] == "install"
        and not any(str(arg).startswith("--progress-bar") for arg in pip_args)
    ):
        pip_args.insert(1, "--progress-bar=raw")
    return _stream_command(
        [str(executable), "-m", "pip", *pip_args],
        callback,
        cwd,
        progress_callback=progress_callback,
        progress_template=progress_template,
    )


def _pip(
    args: Iterable[str],
    callback: Callable[[str], None] | None,
    cwd: Path | None = None,
    progress_callback: Callable[[InstallProgress], None] | None = None,
    progress_template: InstallProgress | None = None,
) -> int:
    return _pip_for(
        sys.executable,
        args,
        callback,
        cwd,
        progress_callback=progress_callback,
        progress_template=progress_template,
    )


def _phase_progress(
    progress_callback: Callable[[InstallProgress], None] | None,
    phase: str,
    phase_index: int,
    overall_percent: float,
    message: str,
    *,
    indeterminate: bool = True,
) -> InstallProgress:
    progress = InstallProgress(
        phase=phase,
        phase_index=phase_index,
        phase_count=7,
        overall_percent=overall_percent,
        indeterminate=indeterminate,
        message=message,
    )
    _emit_progress(progress_callback, progress)
    return progress


def install_or_repair(
    callback: Callable[[str], None] | None = None,
    prefer_gpu: bool = True,
    progress_callback: Callable[[InstallProgress], None] | None = None,
) -> EnvironmentReport:
    root = project_root()
    if sys.version_info < (3, 11):
        raise RuntimeError("CALO-RPD Studio requires Python 3.11 or newer.")

    phase = _phase_progress(
        progress_callback, "Update pip", 1, 5.0, "Preparing and updating pip..."
    )
    _emit(callback, "Updating pip...")
    if (
        _pip(
            ["install", "--upgrade", "pip"],
            callback,
            root,
            progress_callback=progress_callback,
            progress_template=phase,
        )
        != 0
    ):
        raise RuntimeError("Unable to update pip.")

    core_file = root / CORE_REQUIREMENTS_FILE
    phase = _phase_progress(
        progress_callback,
        "Core prerequisites",
        2,
        15.0,
        "Installing scientific and GUI prerequisites...",
    )
    _emit(callback, "Installing core scientific and GUI prerequisites...")
    core_args = (
        ["install", "-r", str(core_file)] if core_file.exists() else ["install", *CORE_REQUIREMENTS]
    )
    if (
        _pip(
            core_args,
            callback,
            root,
            progress_callback=progress_callback,
            progress_template=phase,
        )
        != 0
    ):
        raise RuntimeError("Core prerequisite installation failed.")

    _phase_progress(
        progress_callback,
        "Detect accelerators",
        3,
        35.0,
        "Detecting NVIDIA CUDA-capable hardware...",
    )
    nvidia = detect_nvidia()
    current_torch = detect_torch()

    desired_primary = "cpu"
    if prefer_gpu and nvidia.detected:
        desired_primary = "cuda"

    compatible_primary = bool(
        current_torch.installed
        and (
            (
                desired_primary == "cuda"
                and current_torch.cuda_available
                and current_torch.gpu_test_passed
            )
            or desired_primary == "cpu"
        )
    )

    if not compatible_primary:
        if current_torch.installed:
            _emit(
                callback,
                "Preparing the compatible PyTorch installation for the selected compute mode...",
            )
            _pip(["uninstall", "-y", "torch"], callback, root)

        installed = False
        if desired_primary == "cuda":
            channels = candidate_torch_channels(nvidia)
        else:
            channels = ["cpu"]

        for attempt, channel in enumerate(channels, start=1):
            index_url = f"{PYTORCH_INDEX_ROOT}/{channel}"
            attempt_base = 40.0 + min(20.0, (attempt - 1) * 4.0)
            phase = _phase_progress(
                progress_callback,
                f"Primary PyTorch ({channel})",
                4,
                attempt_base,
                f"Installing and verifying PyTorch compute support ({channel})...",
            )
            _emit(callback, f"Trying official PyTorch compute package: {channel}")
            code = _pip(
                ["install", "--upgrade", "torch>=2.5,<3", "--index-url", index_url],
                callback,
                root,
                progress_callback=progress_callback,
                progress_template=phase,
            )
            if code != 0:
                continue
            info = detect_torch()
            passed = bool(
                (channel == "cpu" and info.installed)
                or (channel.startswith("cu") and info.cuda_available and info.gpu_test_passed)
            )
            if passed:
                installed = True
                break
            _emit(
                callback,
                f"PyTorch channel {channel} installed but did not pass the requested compute test.",
            )
            _pip(["uninstall", "-y", "torch"], callback, root)
        if not installed:
            raise RuntimeError(
                "PyTorch installation failed for all compatible accelerator and CPU modes."
            )
    else:
        _emit(callback, "The existing PyTorch installation is compatible; keeping it.")
        _phase_progress(
            progress_callback,
            "Primary PyTorch",
            4,
            60.0,
            "The existing PyTorch installation is compatible; no replacement is required.",
            indeterminate=False,
        )

    phase = _phase_progress(
        progress_callback,
        "Install application",
        6,
        90.0,
        "Installing CALO-RPD Studio without changing the selected compute runtimes...",
    )
    if (root / "pyproject.toml").exists():
        if (
            _pip(
                ["install", "-e", ".", "--no-deps"],
                callback,
                root,
                progress_callback=progress_callback,
                progress_template=phase,
            )
            != 0
        ):
            raise RuntimeError("CALO-RPD Studio installation failed.")
    else:
        _emit(callback, "Installed-package mode detected; project package is already present.")

    _phase_progress(
        progress_callback,
        "Verify environment",
        7,
        97.0,
        "Verifying CUDA, CPU, package dependencies, and real accelerator computations...",
    )
    report = scan_environment()
    if not report.mandatory_ready:
        raise RuntimeError("Environment verification failed after installation.")
    save_environment_state(report)
    _emit(callback, report.message)
    _phase_progress(
        progress_callback,
        "Complete",
        7,
        100.0,
        "Installation complete. Recommended compute mode: "
        + ("NVIDIA acceleration" if report.recommended_backend.startswith("cuda") else "CPU only")
        + ".",
        indeterminate=False,
    )
    return report


def save_environment_state(report: EnvironmentReport, accepted_cpu_fallback: bool = False) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "app_version": APP_VERSION,
        "interpreter": str(Path(sys.executable).resolve()),
        "accepted_cpu_fallback": bool(accepted_cpu_fallback),
        "report": report.to_dict(),
    }
    STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cpu_fallback_is_accepted() -> bool:
    if not STATE_FILE.exists():
        return False
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return bool(
            payload.get("accepted_cpu_fallback", False)
            and str(payload.get("app_version", "")) == APP_VERSION
            and str(Path(payload.get("interpreter", "")).resolve())
            == str(Path(sys.executable).resolve())
        )
    except Exception:
        return False


def first_launch_or_version_changed() -> bool:
    if not STATE_FILE.exists():
        return True
    try:
        payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return str(payload.get("app_version", "")) != APP_VERSION or str(
            Path(payload.get("interpreter", "")).resolve()
        ) != str(Path(sys.executable).resolve())
    except Exception:
        return True
