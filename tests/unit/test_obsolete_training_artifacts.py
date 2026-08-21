from __future__ import annotations

import json

import pytest

from calo_rpd_studio.gui.panels.obsolete_model_management import (
    ObsoleteAwareTrainingModelLibrary as TrainingModelLibrary,
)


class _Settings:
    def __init__(self) -> None:
        self.values = {}

    def value(self, key, default=None):
        return self.values.get(key, default)

    def set_value(self, key, value) -> None:
        self.values[key] = value

    def sync(self) -> None:
        return None


def _campaign(root, name: str, *, state: str | None, status_text: str | None = None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "training_plan.json").write_text(
        json.dumps({"campaign_id": name}), encoding="utf-8"
    )
    if status_text is not None:
        (directory / "training_status.json").write_text(status_text, encoding="utf-8")
    elif state is not None:
        (directory / "training_status.json").write_text(
            json.dumps({"state": state}), encoding="utf-8"
        )
    return directory


def test_obsolete_campaigns_separate_interrupted_failed_and_corrupt_from_running(tmp_path):
    library = TrainingModelLibrary(
        _Settings(), default_directory=tmp_path / "training-models"
    )
    root = library.default_directory
    interrupted = _campaign(root, "interrupted-run", state="interrupted")
    failed = _campaign(root, "failed-run", state="failed")
    corrupt = _campaign(root, "corrupt-run", state=None, status_text="{not-json")
    _campaign(root, "running-run", state="running")
    unrelated = root / "notes"
    unrelated.mkdir()
    (unrelated / "readme.txt").write_text("not a CALO training campaign", encoding="utf-8")

    records = {item["campaign_id"]: item for item in library.obsolete_campaigns()}

    assert records["interrupted-run"]["obsolete_status"] == "Interrupted training"
    assert records["interrupted-run"]["resumable"] is True
    assert records["failed-run"]["obsolete_status"] == "Failed training"
    assert records["corrupt-run"]["obsolete_status"] == "Corrupted training status"
    assert "running-run" not in records
    assert "notes" not in records
    assert library.validate_obsolete_campaign_deletion(interrupted) == interrupted.resolve()
    assert library.validate_obsolete_campaign_deletion(failed) == failed.resolve()
    assert library.validate_obsolete_campaign_deletion(corrupt) == corrupt.resolve()


def test_obsolete_deletion_requires_exact_managed_child_and_removes_only_selected_directory(tmp_path):
    library = TrainingModelLibrary(
        _Settings(), default_directory=tmp_path / "training-models"
    )
    first = _campaign(library.default_directory, "first", state="interrupted")
    second = _campaign(library.default_directory, "second", state="failed")

    with pytest.raises(ValueError, match="child"):
        library.validate_obsolete_campaign_deletion(library.default_directory)

    deleted = library.delete_obsolete_campaign(first)

    assert deleted == first.resolve()
    assert first.exists() is False
    assert second.is_dir()
    assert {item["campaign_id"] for item in library.obsolete_campaigns()} == {"second"}


def test_saved_training_can_request_exact_policy_library_focus(tmp_path):
    library = TrainingModelLibrary(
        _Settings(), default_directory=tmp_path / "training-models"
    )
    interrupted = _campaign(library.default_directory, "resume-me", state="interrupted")

    target = library.request_policy_library_focus(interrupted)

    assert target == interrupted.resolve()
    assert library.policy_library_focus_request() == str(interrupted.resolve())
    library.clear_policy_library_focus_request()
    assert library.policy_library_focus_request() == ""

    unknown = library.default_directory / "unknown"
    unknown.mkdir()
    with pytest.raises(ValueError, match="no longer in the model library"):
        library.request_policy_library_focus(unknown)
