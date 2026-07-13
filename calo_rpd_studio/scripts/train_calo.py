"""Command-line CALO Core v2 PPO training entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from calo_rpd_studio.algorithms.calo.training import (
    TrainingConfig,
    available_training_devices,
    train_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the CALO Core v2 hierarchical policy on the documented constrained "
            "mixed-variable curriculum."
        )
    )
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=0,
        help="Parallel CPU rollout workers; 0 selects a conservative automatic value.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "xpu", "xpu_sidecar"],
        default="auto",
        help=(
            "Device used for centralized PPO updates. Auto prefers CUDA, then direct XPU, "
            "then the verified secondary XPU runtime, then CPU."
        ),
    )
    parser.add_argument(
        "--development-case",
        action="append",
        default=[],
        help=(
            "Optional custom ORPD development case path. Repeat the option for multiple "
            "development systems."
        ),
    )
    parser.add_argument(
        "--allow-final-benchmark-training",
        action="store_true",
        help=(
            "Explicitly allow IEEE case30/case57/case118 in training. Do not use this for final "
            "publication benchmarking."
        ),
    )
    parser.add_argument(
        "--historical-repository",
        default="",
        help="Optional v1.3 historical experience repository for offline policy pretraining.",
    )
    parser.add_argument(
        "--historical-pretraining-epochs",
        type=int,
        default=4,
        help="Offline historical pretraining epochs before fresh on-policy PPO.",
    )
    parser.add_argument(
        "--use-historical-trajectories",
        action="store_true",
        help="Use eligible CALO trajectories from the historical repository for offline pretraining.",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "trained_models"
            / "calo_policy_v2.pt"
        ),
    )
    args = parser.parse_args()
    selected_device = args.device
    device_info = available_training_devices()
    if selected_device == "auto" and device_info["recommended_device"] == "xpu_sidecar":
        selected_device = "xpu_sidecar"

    config = TrainingConfig(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes,
        horizon=args.horizon,
        seed=args.seed,
        rollout_workers=args.rollout_workers,
        ppo_device=selected_device,
        development_cases=tuple(args.development_case),
        allow_final_benchmark_training=bool(args.allow_final_benchmark_training),
        historical_repository=str(args.historical_repository),
        use_historical_trajectories=bool(args.use_historical_trajectories),
        historical_pretraining_epochs=int(args.historical_pretraining_epochs),
    )

    if selected_device == "xpu_sidecar":
        from calo_rpd_studio.compute.xpu_sidecar import train_policy_in_xpu_sidecar

        path = train_policy_in_xpu_sidecar(config, args.output)
        print(f"Saved CALO policy through secondary Intel XPU runtime: {Path(path).resolve()}")
        return 0

    path, history = train_policy(config, args.output)
    print(f"Saved CALO policy: {Path(path).resolve()}")
    print(f"Final training record: {history[-1] if history else {}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
