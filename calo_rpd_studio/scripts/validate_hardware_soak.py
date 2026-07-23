"""Run the v6.2 protected hardware-soak qualification protocol."""

from __future__ import annotations

import argparse
import json

from calo_rpd_studio.compute.soak import HardwareSoakRunner, SoakConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=4 * 3600.0)
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "xpu"], default="auto")
    parser.add_argument("--output-dir", default="results_data/hardware_soak")
    args = parser.parse_args()
    result = HardwareSoakRunner(
        SoakConfig(duration_seconds=args.duration_seconds, backend=args.backend), output_dir=args.output_dir
    ).run()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if not result.protection_stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
