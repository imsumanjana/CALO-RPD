from __future__ import annotations

from pathlib import Path

from calo_rpd_studio.algorithms.calo.competitive_training import competitive_progress_snapshot


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_single_branch_progress_reports_configured_target_not_epoch_array_only():
    branches = [
        {
            "branch_id": "B01",
            "start_epoch": 0,
            "scientific_session_target_epoch": 24,
            "assigned_device": "cuda:0",
        }
    ]
    percent, detail, payload = competitive_progress_snapshot(
        branches,
        [2],
        active_indices=[0],
        finished_indices=[],
        concurrency=1,
        common_safe_epoch=0,
        training_mode="cumulative",
    )
    assert percent == 8
    assert "session 2/24" in detail
    assert "epoch 2/24" in detail
    assert "cuda:0" in detail
    assert "last exact safe 0" in detail
    assert "next exact safe 10" in detail
    assert payload["total_branch_epochs"] == 24


def test_exact_resume_progress_distinguishes_session_and_cumulative_epochs():
    branches = [
        {
            "branch_id": "B01",
            "start_epoch": 20,
            "scientific_session_target_epoch": 44,
            "assigned_device": "cuda:0",
        }
    ]
    percent, detail, _ = competitive_progress_snapshot(
        branches,
        [22],
        active_indices=[0],
        finished_indices=[],
        concurrency=1,
        common_safe_epoch=20,
        training_mode="cumulative",
    )
    assert percent == 8
    assert "session 2/24" in detail
    assert "cumulative 22/44" in detail
    assert "last exact safe 20" in detail
    assert "next exact safe 30" in detail


def test_multi_branch_progress_uses_total_scientific_branch_epochs():
    branches = [
        {
            "branch_id": f"B{i:02d}",
            "start_epoch": 0,
            "scientific_session_target_epoch": 24,
            "assigned_device": "cuda:0" if i % 2 else "xpu:0",
        }
        for i in range(1, 5)
    ]
    percent, detail, payload = competitive_progress_snapshot(
        branches,
        [3, 3, 0, 0],
        active_indices=[0, 1],
        finished_indices=[],
        concurrency=2,
        common_safe_epoch=0,
        training_mode="cumulative",
    )
    assert percent == 6
    assert payload["completed_branch_epochs"] == 6
    assert payload["total_branch_epochs"] == 96
    assert "overall 6/96 branch-epochs" in detail
    assert "2/2 active" in detail
    assert "2 queued" in detail


def test_indefinite_progress_is_indeterminate_but_epoch_and_safe_state_remain_visible():
    branches = [
        {
            "branch_id": "B01",
            "start_epoch": 0,
            "scientific_session_target_epoch": 0,
            "assigned_device": "cuda:0",
        }
    ]
    percent, detail, _ = competitive_progress_snapshot(
        branches,
        [12],
        active_indices=[0],
        finished_indices=[],
        concurrency=1,
        common_safe_epoch=10,
        training_mode="indefinite",
    )
    assert percent == -1
    assert "cumulative epoch 12" in detail
    assert "indefinite" in detail
    assert "last exact safe 10" in detail
    assert "next exact safe 20" in detail


def test_stage_a_gui_contract_separates_selected_recommended_runtime_and_scope_text():
    source = (_root() / "calo_rpd_studio/gui/panels/calo_intelligence_panel.py").read_text(encoding="utf-8")
    assert 'training_form.addRow("CPU rollout process cap"' in source
    assert 'training_form.addRow("Selected rollout routing"' in source
    assert 'training_form.addRow("Recommended routing"' in source
    assert 'training_form.addRow("Safe-80 branch admission"' in source
    assert 'training_form.addRow("Runtime device mapping"' in source
    assert 'training_form.addRow("Execution scope"' in source
    assert "Advisory only — not selected automatically" in source
    assert "Planner units derive the percentages only; they are NOT counts of CUDA/XPU processes" in source
    assert "Stage-B synthetic curriculum evaluation" in source
    assert "Real ORPD development suite" in source
    assert "This count is EPISODES, not rollout-worker count" in source
    assert "not exact Task Manager utilization percentages" in source
    assert 'self.rollout_workers.valueChanged.connect(self._sync_workers_from_shares)' in source
    assert 'self.rollout_workers.valueChanged.connect(self._apply_recommended_worker_split)' not in source


def test_competitive_coordinator_no_longer_forces_normal_progress_to_zero():
    source = (_root() / "calo_rpd_studio/algorithms/calo/competitive_training.py").read_text(encoding="utf-8")
    assert "progress_callback(progress_percent, progress_detail)" in source
    assert "epochs {epoch_values} · common safe" not in source
    assert '"overall_percent": int(progress_percent)' in source
    assert '"branch_progress": progress_payload["branches"]' in source


def test_protected_rollout_share_reporting_matches_runtime_rebinding_rules():
    from calo_rpd_studio.compute.training_resources import protected_rollout_shares

    assert protected_rollout_shares(
        cuda_share=60, xpu_share=30, cpu_share=10, primary_device="cuda:0"
    ) == {"cuda": 90, "xpu": 0, "cpu": 10}
    assert protected_rollout_shares(
        cuda_share=60,
        xpu_share=30,
        cpu_share=10,
        primary_device="cuda:0",
        auxiliary_xpu_runtime="sidecar",
    ) == {"cuda": 60, "xpu": 30, "cpu": 10}
    assert protected_rollout_shares(
        cuda_share=60, xpu_share=30, cpu_share=10, primary_device="xpu:0"
    ) == {"cuda": 0, "xpu": 90, "cpu": 10}
    assert protected_rollout_shares(
        cuda_share=60, xpu_share=30, cpu_share=10, primary_device="cpu"
    ) == {"cuda": 0, "xpu": 0, "cpu": 100}


def test_single_branch_competitive_training_emits_nonzero_target_aware_progress(tmp_path):
    from calo_rpd_studio.algorithms.calo.training import TrainingConfig, train_policy_parallel

    cfg = TrainingConfig(
        epochs=2,
        episodes_per_epoch=1,
        horizon=2,
        population_size=4,
        ppo_epochs=1,
        minibatch_size=4,
        hidden_dim=16,
        seed=321,
        rollout_workers=1,
        ppo_device="cpu",
        parallel_runs=1,
        parallel_same_seed_branches=1,
        parallel_incremental_branches=0,
        champion_validation_horizon=2,
        champion_validation_episodes=1,
        champion_min_feasible_rate=0.0,
    )
    events: list[tuple[int, str]] = []
    train_policy_parallel(
        cfg,
        tmp_path / "policy.pt",
        parallel_runs=1,
        progress_callback=lambda percent, detail: events.append((int(percent), str(detail))),
    )
    assert any(percent > 0 for percent, _ in events)
    assert any("session 1/2" in detail or "session 2/2" in detail for _, detail in events)
    assert any("last exact safe" in detail for _, detail in events)
