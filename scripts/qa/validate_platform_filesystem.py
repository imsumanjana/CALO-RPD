"""Required Linux filesystem contracts using actual production manifest code, no optional skips."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def validate() -> dict:
    from calo_rpd_studio.scripts.generate_artifact_manifest import build_manifest, write_manifest

    if platform.system() != "Linux":
        raise RuntimeError("This required lane must execute on Linux, not emulate its filesystem")
    checks = []
    with tempfile.TemporaryDirectory(prefix="calo-platform-") as temporary:
        root = Path(temporary)
        stage = root / "stage"
        stage.mkdir()
        try:
            build_manifest(stage)
        except ValueError:
            checks.append("empty_stage_rejected")
        else:
            raise AssertionError("An empty stage was accepted")
        (stage / "b.whl").write_bytes(b"synthetic wheel")
        (stage / "a.tar.gz").write_bytes(b"synthetic source")
        output = stage / "artifact-manifest.json"
        write_manifest(stage, output)
        manifest = json.loads(output.read_text(encoding="utf-8"))
        assert [item["path"] for item in manifest["artifacts"]] == ["a.tar.gz", "b.whl"]
        checks.append("sorted_self_excluding_manifest")
        assert manifest["artifacts"][0]["sha256"] == hashlib.sha256(b"synthetic source").hexdigest()
        checks.append("artifact_hash_matches_bytes")
        for name, target in (("file-link", stage / "b.whl"), ("directory-link", root)):
            link = stage / name
            link.symlink_to(target, target_is_directory=target.is_dir())
            try:
                try:
                    build_manifest(stage)
                except ValueError as exc:
                    assert "Symbolic links" in str(exc), exc
                    checks.append(name + "_rejected")
                else:
                    raise AssertionError("A symbolic link was included in a release manifest")
            finally:
                link.unlink()
        source, target = root / "atomic-source", root / "atomic-target"
        source.write_bytes(b"new complete record")
        target.write_bytes(b"old complete record")
        os.replace(source, target)
        assert target.read_bytes() == b"new complete record" and not source.exists()
        checks.append("same_filesystem_atomic_replace")
    return {
        "status": "passed",
        "checks": checks,
        "skipped": [],
        "scope": "Linux manifest/symlink/filesystem contracts, not full Linux GUI or experiment qualification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = [
        Path(__file__).resolve(),
        ROOT / "calo_rpd_studio/scripts/generate_artifact_manifest.py",
    ]
    before = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    report = {
        "schema": "calo-linux-filesystem-evidence-v1",
        "platform": platform.platform(),
        "python": sys.version,
        "source_hashes": before,
        "release_ready": False,
    }
    try:
        report.update(validate())
    except Exception as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
    after = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    report["source_stable"] = before == after
    if not report["source_stable"]:
        report["status"] = "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
