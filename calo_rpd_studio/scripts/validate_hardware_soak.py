"""Run the v6.2 protected hardware-soak qualification protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from calo_rpd_studio.accelerated.device import resolve_device
from calo_rpd_studio.compute.soak import HardwareSoakRunner, SoakConfig
from calo_rpd_studio.compute.source_identity import resolve_source_identity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=4 * 3600.0)
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--minimum-physical-qualification-seconds", type=float, default=3600.0)
    parser.add_argument("--workload-matrix-size", type=int, default=192)
    parser.add_argument("--output-dir", default="results_data/hardware_soak")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--require-physical-cuda", action="store_true")
    args = parser.parse_args()
    source_commit = ""
    source_identity_kind = "unavailable"
    tracked_source_clean = False
    if args.run_id or args.require_physical_cuda:
        if not args.run_id.strip():
            raise ValueError("--run-id is required for durable or physical soak evidence")
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            raise FileExistsError(f"Refusing to reuse hardware-soak output directory: {output_dir}")
        source_identity = resolve_source_identity(require_durable=True)
        source_commit = source_identity.source_commit
        source_identity_kind = source_identity.source_identity_kind
        tracked_source_clean = source_identity.tracked_source_clean
    if args.require_physical_cuda:
        if float(args.minimum_physical_qualification_seconds) < 3600.0:
            raise ValueError("Physical soak qualification minimum cannot be below 3600 seconds")
        selected = resolve_device("cuda", require_accelerator=True)
        if not str(selected.resolved).startswith("cuda"):
            raise RuntimeError("Physical NVIDIA CUDA was required but not selected")
    result = HardwareSoakRunner(
        SoakConfig(
            duration_seconds=args.duration_seconds,
            sample_interval_seconds=args.sample_interval_seconds,
            backend=args.backend,
            minimum_physical_qualification_seconds=args.minimum_physical_qualification_seconds,
            workload_matrix_size=args.workload_matrix_size,
        ),
        output_dir=args.output_dir,
        run_id=args.run_id,
        source_commit=source_commit,
        source_identity_kind=source_identity_kind,
        tracked_source_clean=tracked_source_clean,
        require_physical_cuda=args.require_physical_cuda,
    ).run()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if args.require_physical_cuda:
        return 0 if result.physical_qualified else 3
    return 0 if not result.protection_stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
