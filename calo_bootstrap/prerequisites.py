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
import tomllib
from collections import deque
from typing import Callable, Iterable

try:
    from importlib.metadata import version as distribution_version

    APP_VERSION = distribution_version("calo-rpd-studio")
except Exception:
    from calo_rpd_studio.version import VERSION as APP_VERSION
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
    "cma>=4.4.4,<5",
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
    ("cma", "CMA-ES"),
)

CORE_IMPORTS: tuple[tuple[str, str], ...] = (
    ("NumPy", "numpy"),
    ("SciPy", "scipy"),
    ("pandas", "pandas"),
    ("Matplotlib", "matplotlib"),
    ("PyQt6", "PyQt6"),
    ("PYPOWER", "pypower"),
    ("PyYAML", "yaml"),
    ("psutil", "psutil"),
    ("NVIDIA NVML Python telemetry", "pynvml"),
    ("CMA-ES", "cma"),
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
DEFAULT_TORCH_REQUIREMENT = "torch>=2.10,<2.11"
COMPUTE_REQUIREMENTS: tuple[str, ...] = (
    "numpy>=1.26,<2.4",
    "scipy>=1.12,<2",
    "PYPOWER>=5.1.18,<6",
    "PyYAML>=6,<7",
    "psutil>=5.9,<8",
    "nvidia-ml-py>=13,<14",
    "cma>=4.4.4,<5",
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
    error_stage: str = ""
    error: str = ""


@dataclass(slots=True)
class EnvironmentReport:
    python_ok: bool
    python_version: str
    interpreter: str
    virtual_environment: bool
    core_packages: dict[str, str]
    missing_core_packages: list[str]
    core_import_errors: dict[str, str]
    nvidia: NvidiaInfo
    torch: TorchInfo
    torch_requirement: str
    torch_version_compatible: bool
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


def _last_json_dict(text: str) -> dict | None:
    for line in reversed(text.splitlines()):
        try:
            payload = json.loads(line.strip())
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _core_requirement_by_label(root: Path) -> dict[str, str]:
    core_file = root / CORE_REQUIREMENTS_FILE
    requirements: list[str] = []
    if core_file.exists():
        requirements = [
            line.strip()
            for line in core_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not requirements:
        requirements = list(CORE_REQUIREMENTS)

    by_distribution: dict[str, str] = {}
    for requirement in requirements:
        match = re.match(r"^\s*([A-Za-z0-9_.-]+)", requirement)
        if match:
            by_distribution[match.group(1).lower().replace("_", "-")] = requirement
    return {
        label: by_distribution.get(distribution.lower().replace("_", "-"), distribution)
        for distribution, label in CORE_DISTRIBUTIONS
    }


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


def detect_core_import_errors() -> dict[str, str]:
    """Return import failures for installed scientific prerequisites.

    Distribution metadata alone cannot prove that compiled extension modules are usable. Running
    the imports in a child interpreter catches broken or cross-interpreter wheels before PyTorch is
    changed, which prevents an unrelated scientific-package failure from being misdiagnosed as a
    CUDA problem.
    """
    imports = json.dumps(CORE_IMPORTS)
    script = f"""
import importlib
import json
errors = {{}}
for label, module_name in {imports}:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        errors[label] = f"{{type(exc).__name__}}: {{exc}}"
print(json.dumps(errors))
"""
    result = _run([sys.executable, "-c", script], timeout=90)
    payload = _last_json_dict(result.stdout)
    if payload is not None:
        return {str(key): str(value) for key, value in payload.items()}
    detail = (result.stderr or result.stdout or "Unable to verify scientific packages").strip()
    return {"Scientific environment": detail}


def _repair_core_import_errors(
    errors: dict[str, str],
    root: Path,
    callback: Callable[[str], None] | None,
    progress_callback: Callable[[InstallProgress], None] | None,
    progress_template: InstallProgress | None,
) -> dict[str, str]:
    """Repair only scientific packages whose imports are demonstrably broken."""
    remaining = dict(errors)
    requirements = _core_requirement_by_label(root)
    for _distribution, label in CORE_DISTRIBUTIONS:
        if label not in remaining:
            continue
        requirement = requirements.get(label)
        if not requirement:
            continue
        _emit(callback, f"Repairing scientific package that could not load: {label}")
        code = _pip(
            ["install", "--force-reinstall", "--no-cache-dir", requirement],
            callback,
            root,
            progress_callback=progress_callback,
            progress_template=progress_template,
        )
        if code != 0:
            return remaining
        remaining = detect_core_import_errors()
        if not remaining:
            break
    return remaining


def project_torch_requirement(root: Path | None = None) -> str:
    """Read the application's PyTorch requirement from project metadata.

    The source-tree requirement is authoritative when available. Installed-package metadata is the
    fallback so the bootstrap cannot silently drift to a broader, incompatible PyTorch range.
    """
    root = root or project_root()
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            dependencies = data.get("project", {}).get("dependencies", ())
            for requirement in dependencies:
                text = str(requirement).strip()
                if re.match(r"^torch(?:\s|[<>=!~])", text, re.I):
                    return text
        except Exception:
            pass
    try:
        for requirement in metadata.requires("calo-rpd-studio") or ():
            text = str(requirement).strip()
            if re.match(r"^torch(?:\s|[<>=!~])", text, re.I):
                return text
    except Exception:
        pass
    return DEFAULT_TORCH_REQUIREMENT


def _numeric_version_tuple(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^\s*(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def torch_version_satisfies_requirement(version: str, requirement: str) -> bool:
    """Evaluate the simple numeric bounds used by CALO's PyTorch requirement without packaging."""
    current = _numeric_version_tuple(version)
    if current is None:
        return False
    spec_text = re.sub(r"^\s*torch\s*", "", requirement, flags=re.I).strip()
    if not spec_text:
        return True
    for raw_spec in spec_text.split(","):
        spec = raw_spec.strip()
        match = re.fullmatch(r"(>=|<=|==|!=|>|<)\s*(\d+(?:\.\d+){0,2})", spec)
        if not match:
            return False
        operator, bound_text = match.groups()
        pieces = [int(piece) for piece in bound_text.split(".")]
        bound = tuple((pieces + [0, 0])[:3])
        ok = {
            ">=": current >= bound,
            "<=": current <= bound,
            "==": current == bound,
            "!=": current != bound,
            ">": current > bound,
            "<": current < bound,
        }[operator]
        if not ok:
            return False
    return True


def detect_torch() -> TorchInfo:
    installed_version = _distribution_version("torch")
    if not installed_version:
        return TorchInfo()
    script = r"""
import json
from importlib import metadata
version = metadata.version("torch")
data = {
    "installed": True,
    "version": version,
    "cuda_available": False,
    "cuda_runtime": "",
    "device_name": "",
    "gpu_test_passed": False,
    "error_stage": "",
    "error": "",
}
try:
    import torch
except Exception as exc:
    data["error_stage"] = "import"
    data["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(data))
    raise SystemExit(0)

data["version"] = str(torch.__version__)
data["cuda_runtime"] = str(torch.version.cuda or "")
try:
    data["cuda_available"] = bool(torch.cuda.is_available())
except Exception as exc:
    data["error_stage"] = "cuda_probe"
    data["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(data))
    raise SystemExit(0)

if data["cuda_available"]:
    try:
        data["device_name"] = str(torch.cuda.get_device_name(0))
        x = torch.randn((256, 256), device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        data["gpu_test_passed"] = bool(y.is_cuda and torch.isfinite(y).all().item())
        if not data["gpu_test_passed"]:
            data["error_stage"] = "cuda_compute"
            data["error"] = "CUDA computation did not produce a finite CUDA result."
    except Exception as exc:
        data["error_stage"] = "cuda_compute"
        data["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(data))
"""
    result = _run([sys.executable, "-c", script], timeout=90)
    payload = _last_json_dict(result.stdout)
    if payload is not None:
        try:
            return TorchInfo(**payload)
        except Exception:
            pass
    return TorchInfo(
        installed=True,
        version=installed_version,
        error_stage="inspection",
        error=(result.stderr or result.stdout or "Unable to inspect PyTorch").strip(),
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

    core_import_errors = detect_core_import_errors() if not missing else {}
    nvidia = detect_nvidia()
    torch = detect_torch()
    torch_requirement = project_torch_requirement()
    torch_version_compatible = bool(
        torch.installed and torch_version_satisfies_requirement(torch.version, torch_requirement)
    )

    cuda_ready = bool(torch.cuda_available and torch.gpu_test_passed)
    gpu_ready = bool(cuda_ready)
    torch_usable = bool(torch.installed and not torch.error_stage and torch_version_compatible)
    mandatory_ready = bool(python_ok and not missing and not core_import_errors and torch_usable)

    if cuda_ready:
        recommended_backend = "cuda:0"
    else:
        recommended_backend = "cpu"

    notes: list[str] = []
    if mandatory_ready:
        notes.append("Scientific prerequisites are ready.")
    elif core_import_errors:
        notes.append("One or more scientific libraries are installed but could not be loaded.")
    elif torch.error_stage:
        notes.append("The computation engine is installed but could not be verified.")
    elif torch.installed and not torch_version_compatible:
        notes.append(
            f"The computation engine version does not satisfy the application requirement "
            f"({torch_requirement})."
        )
    else:
        notes.append("Prerequisites are missing or incomplete.")
    if nvidia.detected:
        notes.append(
            "NVIDIA acceleration is ready."
            if cuda_ready
            else "NVIDIA hardware was detected, but GPU acceleration is not ready."
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
        core_import_errors=core_import_errors,
        nvidia=nvidia,
        torch=torch,
        torch_requirement=torch_requirement,
        torch_version_compatible=torch_version_compatible,
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

    core_import_errors = detect_core_import_errors()
    if core_import_errors:
        _emit(
            callback,
            "One or more scientific packages could not be loaded; repairing only the affected "
            "packages before GPU setup.",
        )
        core_import_errors = _repair_core_import_errors(
            core_import_errors,
            root,
            callback,
            progress_callback,
            phase,
        )
    if core_import_errors:
        detail = "; ".join(f"{name}: {error}" for name, error in core_import_errors.items())
        _emit(callback, "Scientific environment repair did not complete; PyTorch was not changed.")
        _emit(callback, detail)
        raise RuntimeError(
            "Scientific prerequisites are installed but could not be loaded after targeted repair. "
            "PyTorch was not changed. " + detail
        )

    _phase_progress(
        progress_callback,
        "Detect accelerators",
        3,
        35.0,
        "Detecting NVIDIA CUDA-capable hardware...",
    )
    nvidia = detect_nvidia()
    current_torch = detect_torch()
    torch_requirement = project_torch_requirement(root)

    if current_torch.error_stage:
        raise RuntimeError(
            "The installed computation engine could not be loaded or verified, so no replacement "
            "was attempted. " + (current_torch.error or current_torch.error_stage)
        )

    desired_primary = "cpu"
    if prefer_gpu and nvidia.detected:
        desired_primary = "cuda"

    current_version_ok = bool(
        current_torch.installed
        and torch_version_satisfies_requirement(current_torch.version, torch_requirement)
    )
    compatible_primary = bool(
        current_version_ok
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
        force_compute_reinstall = bool(current_torch.installed and current_version_ok)
        if current_torch.installed:
            _emit(
                callback,
                "Preparing a compatible PyTorch installation for the selected compute mode; "
                "the current package will be kept until a replacement installation succeeds.",
            )

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
                f"Computation engine ({channel})",
                4,
                attempt_base,
                f"Installing and verifying computation support ({channel})...",
            )
            _emit(callback, f"Trying official PyTorch compute package: {channel}")
            install_args = ["install", "--upgrade"]
            if force_compute_reinstall:
                install_args.extend(["--force-reinstall", "--no-deps"])
            install_args.extend([torch_requirement, "--index-url", index_url])
            code = _pip(
                install_args,
                callback,
                root,
                progress_callback=progress_callback,
                progress_template=phase,
            )
            if code != 0:
                continue
            info = detect_torch()
            version_ok = torch_version_satisfies_requirement(info.version, torch_requirement)
            if info.error_stage:
                _emit(callback, f"PyTorch verification stopped at {info.error_stage}: {info.error}")
                raise RuntimeError(
                    "PyTorch was installed but its verification raised an error. "
                    "No additional multi-gigabyte wheel channels were attempted and the installed "
                    "package was left in place for diagnosis. "
                    + (info.error or info.error_stage)
                )
            passed = bool(
                version_ok
                and (
                    (channel == "cpu" and info.installed)
                    or (channel.startswith("cu") and info.cuda_available and info.gpu_test_passed)
                )
            )
            if passed:
                installed = True
                break
            if info.installed and not version_ok:
                raise RuntimeError(
                    f"Installed PyTorch {info.version} does not satisfy {torch_requirement}; "
                    "stopping instead of cycling compute packages."
                )
            _emit(
                callback,
                f"PyTorch channel {channel} installed but did not expose the requested compute mode; "
                "trying the next compatible official channel.",
            )
            _pip(["uninstall", "-y", "torch"], callback, root)
            force_compute_reinstall = False
        if not installed:
            raise RuntimeError(
                "PyTorch installation failed for all compatible accelerator and CPU modes."
            )
    else:
        _emit(callback, "The existing PyTorch installation is compatible; keeping it.")
        _phase_progress(
            progress_callback,
            "Computation engine",
            4,
            60.0,
            "The existing computation engine is compatible; no replacement is required.",
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

    verify_phase = _phase_progress(
        progress_callback,
        "Verify environment",
        7,
        97.0,
        "Verifying scientific packages, GPU acceleration, and dependency consistency...",
    )
    _emit(callback, "Checking installed package dependency consistency...")
    if (
        _pip(
            ["check"],
            callback,
            root,
            progress_callback=progress_callback,
            progress_template=verify_phase,
        )
        != 0
    ):
        raise RuntimeError(
            "Installed package dependency verification failed. Review the package check details "
            "above before starting CALO-RPD Studio."
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
