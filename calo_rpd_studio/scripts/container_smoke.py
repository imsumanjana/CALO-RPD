"""Fail-closed runtime and filesystem smoke test for built CALO containers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.results.database import DATABASE_SCHEMA_VERSION, ResultDatabase
from calo_rpd_studio.scripts.generate_artifact_manifest import write_manifest


def _assert_root_filesystem_is_read_only() -> None:
    statvfs = getattr(os, "statvfs", None)
    if not callable(statvfs):
        raise RuntimeError("Container runtime cannot inspect filesystem mount flags.")
    readonly_flag = int(getattr(os, "ST_RDONLY", 1))
    if not int(statvfs("/").f_flag) & readonly_flag:
        raise RuntimeError("Container root mount is not flagged read-only.")
    probe = Path("/opt/calo/.container-write-probe")
    try:
        probe.write_text("unexpected", encoding="utf-8")
    except OSError:
        return
    probe.unlink(missing_ok=True)
    raise RuntimeError("Container root filesystem is writable; --read-only was not enforced.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-durable-source", action="store_true")
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Container smoke qualification requires a Linux container.")
    get_effective_uid = getattr(os, "geteuid", None)
    if not callable(get_effective_uid):
        raise RuntimeError("Container runtime does not expose a POSIX effective user ID.")
    effective_uid = int(get_effective_uid())
    if effective_uid == 0:
        raise RuntimeError("Container process is running as root.")
    _assert_root_filesystem_is_read_only()
    source_identity = resolve_source_identity(require_durable=args.require_durable_source)

    mode = os.environ.get("CALO_COMPUTE_MODE", "cpu").strip().lower()
    if mode not in {"cpu", "cuda"}:
        raise RuntimeError(f"Unsupported CALO_COMPUTE_MODE: {mode!r}")
    cuda_ready = bool(torch.cuda.is_available())
    if (mode == "cuda") != cuda_ready:
        raise RuntimeError(
            f"Compute-mode mismatch: requested={mode!r}, torch_cuda_available={cuda_ready}."
        )

    data_root = Path(os.environ.get("CALO_WORKDIR", "/data")).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="calo-smoke-", dir=data_root) as directory:
        database_path = Path(directory) / "results.sqlite"
        database = ResultDatabase(database_path)
        config = ExperimentConfig(
            execution_backend="cpu_only" if mode == "cpu" else "cuda_preferred"
        )
        experiment_id = database.create_experiment(config, {"source": "container-smoke"})
        if database.get_experiment(experiment_id) is None:
            raise RuntimeError("Container database round-trip failed.")
        if database.schema_version != DATABASE_SCHEMA_VERSION:
            raise RuntimeError(
                f"Container database schema mismatch: {database.schema_version} != "
                f"{DATABASE_SCHEMA_VERSION}."
            )

    manifest_path = data_root / "image-filesystem-manifest.json"
    write_manifest(Path("/opt/calo"), manifest_path)
    report = {
        "schema": "calo_rpd_container_smoke_v1",
        "uid": effective_uid,
        "compute_mode": mode,
        "cuda_available": cuda_ready,
        "database_schema_version": DATABASE_SCHEMA_VERSION,
        "image_manifest": str(manifest_path),
        "source_identity": source_identity.to_dict(),
    }
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
