"""First-launch bootstrap entry point."""

from __future__ import annotations

import sys

from .prerequisites import (
    cpu_fallback_is_accepted,
    first_launch_or_version_changed,
    scan_environment,
)


def accelerator_repair_required(report) -> bool:
    """Return whether any detected accelerator lacks its own verified compute runtime.

    Overall ``gpu_ready`` is insufficient on mixed NVIDIA+Intel hosts: CUDA can be healthy while
    the Intel XPU sidecar is missing.  CALO must repair each detected hardware family independently
    so the configured CUDA/XPU/CPU routing contract reflects the actual machine.
    """
    cuda_ready = bool(report.torch.cuda_available and report.torch.gpu_test_passed)
    xpu_ready = bool(
        (report.torch.xpu_available and report.torch.xpu_test_passed)
        or (report.xpu_sidecar.xpu_available and report.xpu_sidecar.gpu_test_passed)
    )
    return bool(
        (report.nvidia.detected and not cuda_ready)
        or (report.intel.detected and not xpu_ready)
    )


def ensure_prerequisites(force_wizard: bool = False) -> bool:
    report = scan_environment()
    needs_wizard = force_wizard or first_launch_or_version_changed() or not report.mandatory_ready
    # Repair missing backends per detected hardware family.  A healthy CUDA runtime must not hide a
    # missing Intel XPU runtime on mixed-GPU systems.
    if accelerator_repair_required(report):
        # A previously accepted CPU fallback may suppress repeated repair prompts on a genuinely
        # accelerator-less/broken machine, but it must not hide a missing secondary accelerator
        # when another GPU backend (for example CUDA) is already healthy.
        if report.gpu_ready or not cpu_fallback_is_accepted():
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
    force_wizard = "--setup" in sys.argv or "--repair" in sys.argv
    sys.argv = [arg for arg in sys.argv if arg not in {"--setup", "--repair"}]
    if not ensure_prerequisites(force_wizard=force_wizard):
        return 1
    from calo_rpd_studio.app.application import main as app_main

    return int(app_main())


if __name__ == "__main__":
    raise SystemExit(main())
