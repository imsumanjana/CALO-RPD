from __future__ import annotations

from io import BytesIO
import struct

import pytest

from calo_rpd_studio.compute.persistent_training_actor import read_frame, write_frame


def test_persistent_actor_round_trips_multiple_progress_frames() -> None:
    stream = BytesIO()
    write_frame(stream, {"kind": "progress", "payload": {"iteration": 1, "progress": 10}})
    write_frame(stream, {"kind": "progress", "payload": {"iteration": 2, "progress": 20}})
    stream.seek(0)

    first = read_frame(stream)
    second = read_frame(stream)

    assert first["payload"]["iteration"] == 1
    assert second["payload"]["iteration"] == 2


def test_training_actor_round_trips_dictionary_frame() -> None:
    stream = BytesIO()
    write_frame(stream, {"action": "rollout", "epoch": 3})
    stream.seek(0)
    assert read_frame(stream) == {"action": "rollout", "epoch": 3}


def test_worker_protocol_rejects_oversized_frames() -> None:
    stream = BytesIO(struct.pack("!Q", 512 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="Invalid local worker frame length"):
        read_frame(stream)
