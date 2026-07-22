"""Command-line CALO v5.8 competitive policy-training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from calo_rpd_studio.algorithms.calo.heterogeneous_training import HeterogeneousTrainingConfig
from calo_rpd_studio.algorithms.calo.training import (
    TrainingConfig,
    available_training_devices,
    train_policy_parallel,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train a CALO v5.8 competitive multi-branch candidate policy on the documented "
            "constrained mixed-variable curriculum."
        )
    )
    parser.add_argument("--epochs", type=int, default=24, help="Epochs in this cumulative session; ignored in infinite mode.")
    parser.add_argument("--mode", choices=["cumulative", "infinite"], default="cumulative")
    parser.add_argument("--start-mode", choices=["new", "exact_resume", "base_guided_fork"], default="new")
    parser.add_argument("--base-model", default="", help="Deployable base policy used by Base-Guided Fork branches.")
    parser.add_argument("--parallel-same", type=int, default=1, help="Branches using the base seed unchanged.")
    parser.add_argument("--parallel-incremental", type=int, default=0, help="Branches using seed+1, seed+2, ...")
    parser.add_argument("--parallel-decremental", type=int, default=0, help="Branches using seed-1, seed-2, ...")
    parser.add_argument("--parallel-custom-seeds", default="", help="Comma-separated explicit branch seeds.")
    parser.add_argument("--scratch-dir", default="", help="Fast local scratch directory for rolling temporary exact-state snapshots.")
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
        training_mode=("indefinite" if args.mode == "infinite" else "cumulative"),
        parallel_same_seed_branches=max(0, int(args.parallel_same)),
        parallel_incremental_branches=max(0, int(args.parallel_incremental)),
        parallel_decremental_branches=max(0, int(args.parallel_decremental)),
        parallel_custom_seeds=tuple(int(item.strip()) for item in str(args.parallel_custom_seeds).split(",") if item.strip()),
        parallel_start_mode=str(args.start_mode),
        base_model_checkpoint=str(args.base_model),
        training_scratch_dir=str(args.scratch_dir),
        safe_snapshot_interval_epochs=10,
    )
    branch_count = (
        common["parallel_same_seed_branches"]
        + common["parallel_incremental_branches"]
        + common["parallel_decremental_branches"]
        + len(common["parallel_custom_seeds"])
    )
    if branch_count <= 0:
        parser.error("At least one parallel branch must be requested.")
    common["parallel_runs"] = branch_count
    if args.start_mode == "base_guided_fork" and not str(args.base_model).strip():
        parser.error("--base-model is required for --start-mode base_guided_fork.")

    if not args.legacy_cpu_rollouts:
        config = HeterogeneousTrainingConfig(
            **common,
            heterogeneous_rollouts=True,
            cuda_rollout_share=int(args.cuda_share),
            xpu_rollout_share=int(args.xpu_share),
            cpu_rollout_share=int(args.cpu_share),
        )
        path, history = train_policy_parallel(config, args.output, parallel_runs=branch_count)
        print(f"Saved CALO v5.8 competitive base policy: {Path(path).resolve()}")
        print(f"Parallel branches: {branch_count}; mode: {args.mode}; start mode: {args.start_mode}")
        print(f"Final coordinator record: {history[-1] if history else {}}")
        print("Formal Policy Qualification remains separate from branch-champion/base selection.")
        return 0

    config = TrainingConfig(**common)
    if selected_device == "xpu_sidecar":
        if branch_count > 1:
            parser.error("The secondary XPU sidecar supports one branch per training job; use auto/direct XPU/CUDA/CPU for competitive multi-branch training.")
        from calo_rpd_studio.compute.xpu_sidecar import train_policy_in_xpu_sidecar

        path = train_policy_in_xpu_sidecar(config, args.output)
        print(f"Saved CALO policy through secondary Intel XPU runtime: {Path(path).resolve()}")
        return 0

    path, history = train_policy_parallel(config, args.output, parallel_runs=branch_count)
    print(f"Saved CALO v5.8 competitive base policy: {Path(path).resolve()}")
    print(f"Parallel branches: {branch_count}; mode: {args.mode}; start mode: {args.start_mode}")
    print(f"Final coordinator record: {history[-1] if history else {}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
