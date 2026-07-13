"""First-launch bootstrap entry point."""
from __future__ import annotations

import sys

from .prerequisites import cpu_fallback_is_accepted, first_launch_or_version_changed, scan_environment


def ensure_prerequisites(force_wizard: bool = False) -> bool:
    report = scan_environment()
    needs_wizard = force_wizard or first_launch_or_version_changed() or not report.mandatory_ready
    # A system with detected accelerator hardware but no verified CUDA/XPU runtime should be offered
    # automatic repair before the scientific application starts, rather than silently losing it.
    accelerator_detected = bool(report.nvidia.detected or report.intel.detected)
    if accelerator_detected and not report.gpu_ready and not cpu_fallback_is_accepted():
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
