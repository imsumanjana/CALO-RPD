"""Standalone CUDA/XPU policy-rollout actor used by heterogeneous CALO training."""
from __future__ import annotations

import argparse
from pathlib import Path
import traceback

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    try:
        payload = torch.load(Path(args.input), map_location="cpu", weights_only=False)
        from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
            collect_actor_lane_payload,
        )

        result = collect_actor_lane_payload(payload)
        torch.save(result, Path(args.output))
        return 0
    except Exception:
        print(traceback.format_exc(), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
