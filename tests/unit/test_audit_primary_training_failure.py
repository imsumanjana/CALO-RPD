"""Diagnostic failures cannot replace primary training errors or enable resume."""

from types import SimpleNamespace
import pytest
from calo_rpd_studio.algorithms.calo import _tsh_calo_training_campaign_core as core


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
@pytest.mark.parametrize("provenance_fails", [False, True])
def test_member_preserves_original_error_and_conservative_resume(
    monkeypatch, error_type, provenance_fails
):
    primary = error_type("original failure")
    cleaned = []
    trainer = SimpleNamespace(close=lambda: cleaned.append(True))
    monkeypatch.setattr(core, "IndependentTSHCALOTrainer", lambda _: trainer)

    def provenance():
        if provenance_fails:
            raise ValueError("secondary provenance failure")
        return {"accounting_complete": True}

    session = SimpleNamespace(
        failed=False,
        environment=SimpleNamespace(scientific_provenance=provenance, accounting_complete=True),
    )
    runner = core.IndependentTSHCALOTrainingCampaign.__new__(
        core.IndependentTSHCALOTrainingCampaign
    )
    runner.plan = SimpleNamespace(
        members=(SimpleNamespace(episodes=("synthetic-episode",)),),
        training_config=lambda _: SimpleNamespace(),
    )
    runner._active_session = None
    runner._ensure_generalization_baseline = lambda *a, **kw: None
    runner._record_generalization_monitor = lambda *a, **kw: None
    runner._new_session = lambda *a, **kw: session

    def fail(*a, **kw):
        raise primary

    runner._advance_session = fail
    with pytest.raises(error_type) as caught:
        runner._run_member({}, 0)
    assert caught.value is primary
    assert cleaned == [True]
    assert runner._active_session is None
    assert runner._last_failure_resumable is (error_type is OSError and not provenance_fails)
    if provenance_fails:
        assert runner._last_failure_provenance is None
        assert any("secondary provenance failure" in note for note in primary.__notes__)
    else:
        assert runner._last_failure_provenance == {"accounting_complete": True}
