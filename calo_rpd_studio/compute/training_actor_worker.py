"""Standalone and persistent CUDA/XPU rollout actor for CALO policy training."""

from __future__ import annotations

import argparse
from pathlib import Path
import traceback

import torch


def _server(device: str, lane: str) -> int:
    from calo_rpd_studio.compute.persistent_training_actor import read_frame, write_frame

    input_stream = __import__("sys").stdin.buffer
    output_stream = __import__("sys").stdout.buffer
    while True:
        try:
            message = read_frame(input_stream)
        except EOFError:
            return 0
        action = str(message.get("action", ""))
        if action == "shutdown":
            return 0
        if action != "collect":
            write_frame(output_stream, {"ok": False, "error": f"Unsupported action: {action}"})
            continue
        try:
            payload = dict(message["payload"])
            payload["device"] = device
            payload["lane"] = lane
            payload["persistent_actor"] = True
            from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
                collect_actor_lane_payload,
            )

            result = collect_actor_lane_payload(payload)
            write_frame(output_stream, {"ok": True, "result": result})
        except Exception:
            write_frame(output_stream, {"ok": False, "error": traceback.format_exc()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lane", default="cuda")
    args = parser.parse_args(argv)
    if args.server:
        return _server(args.device, args.lane)
    if not args.input or not args.output:
        parser.error("input and output are required outside --server mode")
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
