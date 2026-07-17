"""Run a reproducible CPU/accelerator parity audit from the command line."""
from __future__ import annotations

import argparse
import json

from calo_rpd_studio.accelerated.parity_audit import run_configuration_parity_audit
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="case30", choices=["case30", "case57", "case118", "case300"])
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, xpu, xpu:0, or cpu")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    config = ExperimentConfig(case_name=args.case, master_seed=args.seed)
    config.tensor_batch_size = args.batch_size
    report = run_configuration_parity_audit(config, device=args.device, candidates=args.candidates)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
