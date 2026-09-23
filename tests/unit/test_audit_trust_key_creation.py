"""First-use key publication is single-winner without accessing real trust keys."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import subprocess
import sys
import threading
import pytest
from calo_rpd_studio.ai import model_io as io


@pytest.fixture(autouse=True)
def isolated_trust(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "_TRUST_DIR", tmp_path / "trust")
    monkeypatch.setattr(io, "_TRUST_KEY", tmp_path / "trust/key")


def test_simultaneous_creators_return_only_the_persisted_key(monkeypatch):
    barrier = threading.Barrier(8)
    publish = io.os.link

    def synchronized_publish(*args, **kwargs):
        barrier.wait(timeout=10)
        return publish(*args, **kwargs)

    monkeypatch.setattr(io.os, "link", synchronized_publish)
    with ThreadPoolExecutor(max_workers=8) as executor:
        keys = list(executor.map(lambda _: io._load_or_create_local_trust_key(), range(8)))
    assert len(set(keys)) == 1
    assert keys[0] == io._TRUST_KEY.read_bytes()
    assert not list(io._TRUST_DIR.glob("*.tmp"))


def test_existing_key_is_not_rotated(monkeypatch):
    io._TRUST_DIR.mkdir()
    io._TRUST_KEY.write_bytes(b"k" * 32)

    def forbidden(*args, **kwargs):
        pytest.fail("Existing trust key must not be regenerated")

    monkeypatch.setattr(io.secrets, "token_bytes", forbidden)
    assert io._load_or_create_local_trust_key() == b"k" * 32


def test_corrupt_existing_key_fails_without_replacement():
    io._TRUST_DIR.mkdir()
    io._TRUST_KEY.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="invalid"):
        io._load_or_create_local_trust_key()
    assert io._TRUST_KEY.read_bytes() == b"short"


def test_unsupported_publication_fails_closed(monkeypatch):
    def denied(*args, **kwargs):
        raise OSError("synthetic filesystem without atomic link support")

    monkeypatch.setattr(io.os, "link", denied)
    with pytest.raises(OSError, match="atomic link"):
        io._load_or_create_local_trust_key()
    assert not io._TRUST_KEY.exists()
    assert not list(io._TRUST_DIR.glob("*.tmp"))


def test_independent_processes_share_the_same_key(tmp_path):
    directory = tmp_path / "process-trust"
    code = (
        "from pathlib import Path; import sys,hashlib; "
        "from calo_rpd_studio.ai import model_io as io; "
        "io._TRUST_DIR=Path(sys.argv[1]); io._TRUST_KEY=io._TRUST_DIR/'key'; "
        "print(hashlib.sha256(io._load_or_create_local_trust_key()).hexdigest())"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-B", "-c", code, str(directory)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        results = [p.communicate(timeout=60) for p in processes]
        assert all(p.returncode == 0 for p in processes), results
        expected = hashlib.sha256((directory / "key").read_bytes()).hexdigest()
        assert all(stdout.strip() == expected for stdout, _stderr in results)
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=10)
