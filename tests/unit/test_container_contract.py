from __future__ import annotations

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_image_drops_root_before_runtime_entrypoint():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11-slim-bookworm@sha256:" in dockerfile
    assert "useradd --uid 10001" in dockerfile
    assert dockerfile.index("USER 10001:10001") < dockerfile.index("ENTRYPOINT")
    assert "HOME=/data/home/calo" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "${RUNTIME_LOCK}" in dockerfile
    assert "COPY containers/debian.sources" in dockerfile
    debian_sources = (ROOT / "containers/debian.sources").read_text(encoding="utf-8")
    assert debian_sources.count("20260728T000000Z") == 2
    assert "deb.debian.org" not in debian_sources


def test_compose_profiles_are_local_only_read_only_and_non_privileged():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    common = compose["x-calo-common"]
    assert common["ports"] == ["127.0.0.1:${CALO_PORT:-6080}:6080"]
    assert common["user"] == "10001:10001"
    assert common["read_only"] is True
    assert common["cap_drop"] == ["ALL"]
    assert common["security_opt"] == ["no-new-privileges:true"]
    assert common["mem_limit"] == "${CALO_HOST_MEMORY_LIMIT:-24g}"
    assert any(item.startswith("/tmp:rw,nosuid,nodev") for item in common["tmpfs"])


def test_cuda_profile_requests_one_explicit_nvidia_device():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    cuda = compose["services"]["cuda"]
    devices = cuda["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [
        {
            "driver": "nvidia",
            "device_ids": ["${CALO_GPU_DEVICE:-0}"],
            "capabilities": ["gpu"],
        }
    ]
    assert cuda["environment"]["NVIDIA_VISIBLE_DEVICES"] == "${CALO_GPU_DEVICE:-0}"
    assert cuda["build"]["args"]["RUNTIME_LOCK"] == ("requirements-lock-cuda128-py311-linux.txt")


def test_cpu_and_cuda_runtime_locks_are_hash_complete_and_backend_specific():
    expected = {
        "requirements-lock-cpu-py311-linux.txt": (
            "https://download.pytorch.org/whl/cpu",
            "torch==2.10.0+cpu",
        ),
        "requirements-lock-cuda128-py311-linux.txt": (
            "https://download.pytorch.org/whl/cu128",
            "torch==2.10.0+cu128",
        ),
    }
    requirement_pattern = re.compile(
        r"(?ms)^([A-Za-z0-9][A-Za-z0-9_.-]*==[^\s\\]+).*?(?=^[A-Za-z0-9][A-Za-z0-9_.-]*==|\Z)"
    )
    for filename, (index, torch_pin) in expected.items():
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert f"--extra-index-url {index}" in content
        assert torch_pin in content
        blocks = requirement_pattern.findall(content)
        assert len(blocks) >= 25
        for match in requirement_pattern.finditer(content):
            assert "--hash=sha256:" in match.group(0), match.group(1)


def test_ci_lock_and_workflow_are_reproducible_and_supply_chain_pinned():
    lock = (ROOT / "requirements-lock-ci-py311-linux.txt").read_text(encoding="utf-8")
    assert "uv==0.11.29" in lock
    assert "ruff==0.15.22" in lock
    assert "mypy==1.20.2" in lock
    assert "torch==2.10.0+cpu" in lock
    requirement_pattern = re.compile(
        r"(?ms)^([A-Za-z0-9][A-Za-z0-9_.-]*==[^\s\\]+).*?(?=^[A-Za-z0-9][A-Za-z0-9_.-]*==|\Z)"
    )
    for match in requirement_pattern.finditer(lock):
        assert "--hash=sha256:" in match.group(0), match.group(1)

    workflow_text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    assert set(workflow["jobs"]) == {
        "source",
        "compatibility",
        "headless-gui",
        "artifact",
        "cpu-image",
        "cuda-image",
        "physical-cuda",
    }
    action_references = re.findall(r"(?m)^\s*uses:\s*([^\s#]+)", workflow_text)
    assert action_references
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", item) for item in action_references)
    assert "--require-hashes" in workflow_text
    assert "--read-only" in workflow_text
    assert "sbom: true" in workflow_text
    assert "provenance: mode=max" in workflow_text
    assert "inputs.run_physical_cuda" in workflow_text
    compatibility = workflow["jobs"]["compatibility"]
    compatibility_text = yaml.safe_dump(compatibility, sort_keys=True)
    assert "python -m pip check" in compatibility_text
    assert "python -m pip freeze --all > compatibility-environment.txt" in compatibility_text
    assert "compatibility-${{ runner.os }}-py${{ matrix.python }}-environment" in compatibility_text
