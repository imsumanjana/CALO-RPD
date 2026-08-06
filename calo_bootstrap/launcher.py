"""First-launch bootstrap entry point."""

from __future__ import annotations

import sys

from calo_rpd_studio.version import DISPLAY_VERSION, VERSION

from .prerequisites import (
    cpu_fallback_is_accepted,
    first_launch_or_version_changed,
    scan_environment,
)


def accelerator_repair_required(report) -> bool:
    """Return whether detected NVIDIA hardware lacks a verified CUDA runtime."""
    cuda_ready = bool(report.torch.cuda_available and report.torch.gpu_test_passed)
    return bool(report.nvidia.detected and not cuda_ready)


def ensure_prerequisites(force_wizard: bool = False) -> bool:
    report = scan_environment()
    needs_wizard = force_wizard or first_launch_or_version_changed() or not report.mandatory_ready
    if accelerator_repair_required(report):
        if not cpu_fallback_is_accepted():
            needs_wizard = True
    if not needs_wizard:
        return True
    try:
        from .wizard import PrerequisiteWizard

        return PrerequisiteWizard().run()
    except Exception as exc:
        print(f"Prerequisite wizard could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Run: python bootstrap.py", file=sys.stderr)
        return False


def main() -> int:
    if "--version" in sys.argv or "-V" in sys.argv:
        print(f"CALO-RPD Studio {DISPLAY_VERSION} ({VERSION})")
        return 0
    force_wizard = "--setup" in sys.argv or "--repair" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in {"--setup", "--repair"}]
    if not ensure_prerequisites(force_wizard=force_wizard):
        return 1
    from calo_rpd_studio.app.application import main as app_main

    return int(app_main())


if __name__ == "__main__":
    raise SystemExit(main())
