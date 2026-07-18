"""Command-line CALO Core v2 policy-training entry point."""
from __future__ import annotations

import argparse
from pathlib import Path

from calo_rpd_studio.algorithms.calo.heterogeneous_training import (
    HeterogeneousTrainingConfig,
    train_policy_heterogeneous,
)
from calo_rpd_studio.algorithms.calo.training import (
    TrainingConfig,
    available_training_devices,
    train_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CALO Core v2 candidate policy on the documented constrained "
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
        help="Parallel CPU actor workers; 0 selects a conservative automatic value.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "xpu", "xpu_sidecar"],
        default="auto",
        help=(
            "Central PPO learner device. Weighted mode requires a device in the primary runtime; "
            "the secondary XPU runtime remains available as an actor lane."
        ),
    )
    parser.add_argument(
        "--legacy-cpu-rollouts",
        action="store_true",
        help="Disable weighted CUDA/XPU/CPU actors and use the legacy all-CPU rollout collector.",
    )
    parser.add_argument("--cuda-share", type=int, default=100)
    parser.add_argument("--xpu-share", type=int, default=0)
    parser.add_argument("--cpu-share", type=int, default=0)
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
        help="Optional historical experience repository for offline policy pretraining.",
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
        help="Use eligible CALO trajectories for offline pretraining.",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1]
            / "data"
            / "trained_models"
            / "calo_policy_v2_candidate.pt"
        ),
    )
    args = parser.parse_args()

    device_info = available_training_devices()
    selected_device = args.device
    if (
        args.legacy_cpu_rollouts
        and selected_device == "auto"
        and device_info["recommended_device"] == "xpu_sidecar"
    ):
        selected_device = "xpu_sidecar"

    common = dict(
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

    if not args.legacy_cpu_rollouts:
        config = HeterogeneousTrainingConfig(
            **common,
            heterogeneous_rollouts=True,
            cuda_rollout_share=int(args.cuda_share),
            xpu_rollout_share=int(args.xpu_share),
            cpu_rollout_share=int(args.cpu_share),
        )
        path, history = train_policy_heterogeneous(config, args.output)
        print(f"Saved heterogeneous CALO candidate policy: {Path(path).resolve()}")
        print(f"Final training record: {history[-1] if history else {}}")
        print("Validate and re-freeze this candidate before using it in a final TEST campaign.")
        return 0

    config = TrainingConfig(**common)
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
