"""Explicit one-time migration utility for trusted local pre-v5.7 exact resumes."""
from __future__ import annotations

import argparse
from pathlib import Path

from calo_rpd_studio.ai.model_io import migrate_legacy_local_resume


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate a trusted local legacy CALO exact-resume checkpoint to the v5.9 HMAC-authenticated format. "
            "Never use this for downloaded/untrusted pickle-capable files."
        )
    )
    parser.add_argument("resume", help="Legacy .resume.pt checkpoint")
    parser.add_argument("--output", help="Destination path; default creates *.v58.resume.pt")
    parser.add_argument(
        "--i-trust-this-local-file",
        action="store_true",
        help="Required explicit acknowledgement that the legacy checkpoint is a trusted local artifact",
    )
    args = parser.parse_args(argv)
    if not args.i_trust_this_local_file:
        parser.error("--i-trust-this-local-file is required; untrusted exact-resume pickle migration is refused")
    target = migrate_legacy_local_resume(
        args.resume,
        target=args.output,
        explicit_trust=True,
    )
    print(Path(target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
