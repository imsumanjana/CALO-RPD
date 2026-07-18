from __future__ import annotations

from io import BytesIO
import struct

import pytest

from calo_rpd_studio.compute.persistent_accelerator_sidecar import _read_frame, _write_frame
from calo_rpd_studio.compute.persistent_training_actor import read_frame, write_frame


def test_accelerator_sidecar_round_trips_multiple_progress_frames() -> None:
    stream = BytesIO()
    _write_frame(stream, {"kind": "progress", "payload": {"iteration": 1, "progress": 10}})
    _write_frame(stream, {"kind": "progress", "payload": {"iteration": 2, "progress": 20}})
    stream.seek(0)

    first = _read_frame(stream)
    second = _read_frame(stream)

    assert first["payload"]["iteration"] == 1
    assert second["payload"]["iteration"] == 2


def test_training_actor_round_trips_dictionary_frame() -> None:
    stream = BytesIO()
    write_frame(stream, {"action": "rollout", "epoch": 3})
    stream.seek(0)
    assert read_frame(stream) == {"action": "rollout", "epoch": 3}


@pytest.mark.parametrize("reader", [_read_frame, read_frame])
def test_worker_protocol_rejects_oversized_frames(reader) -> None:
    stream = BytesIO(struct.pack("!Q", 512 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="Invalid local worker frame length"):
        reader(stream)
