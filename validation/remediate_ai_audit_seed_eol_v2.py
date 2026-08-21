#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

AUDIT_HELPERS = r'''def _git_path_clean_against_head(root: Path, path: str) -> bool:
    if not is_git_root(root):
        return False
    for args in (("diff", "--quiet", "--", path), ("diff", "--cached", "--quiet", "--", path)):
        try:
            proc = subprocess.run(
                ["git", *args], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except OSError:
            return False
        if proc.returncode != 0:
            return False
    return True


def _git_head_blob_sha256(root: Path, path: str) -> str:
    if not is_git_root(root):
        return ""
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    return sha256_bytes(proc.stdout) if proc.returncode == 0 else ""


def _legacy_review_matches_current(
    root: Path,
    path: str,
    record: dict[str, Any],
    info: dict[str, Any],
) -> bool:
    expected = str(record.get("content_hash", ""))
    if not expected:
        return False
    if expected == info.get("content_sha256"):
        return True
    # A Windows checkout can materialize CRLF while the reviewed Git blob remains LF.
    # Accept that representation difference only when the canonical HEAD blob still
    # matches the historical SHA-256 and the source has no staged or unstaged edit.
    return (
        _git_path_clean_against_head(root, path)
        and _git_head_blob_sha256(root, path) == expected
    )


def _migrate_legacy_audit_seed(root: Path, idx: dict[str, dict[str, Any]], out: dict[str, Any]) -> None:
    seed = read_json(root / AI / "audit-seed.json", {}).get("units", {})
    for path, record in seed.items():
        info = idx.get(path)
        if not info or not _legacy_review_matches_current(root, path, record, info):
            continue
        current = out.setdefault(path, {})
        if current.get("reviewed") and not current.get("re_audit_required"):
            continue
        current.update({
            "reviewed": True,
            "re_audit_required": False,
            "audit_categories": list(record.get("audit_categories", [])),
            "last_reviewed_commit": record.get("last_reviewed_commit", ""),
            "last_reviewed_content_sha256": record.get("content_hash", ""),
            "scope_note": record.get("scope_note", ""),
            "unresolved_findings": list(current.get("unresolved_findings", [])),
            "migrated_from": ".ai/audit-seed.json",
        })
'''

REGRESSION_TEST = r'''

def test_audit_seed_survives_clean_crlf_materialization_but_not_real_edit(tmp_path: Path):
    repo = make_repo(tmp_path)
    source = "calo_rpd_studio/algorithms/calo/policy_registry.py"
    path = repo / source
    canonical = path.read_text(encoding="utf-8")
    git(repo, "config", "core.autocrlf", "true")
    path.write_bytes(canonical.replace("\n", "\r\n").encode("utf-8"))
    clean = subprocess.run(["git", "diff", "--quiet", "--", source], cwd=repo)
    assert clean.returncode == 0

    run(repo, "scripts/ai-index", "init")
    audit = readj(repo, shard_path(repo, "audit", source).relative_to(repo).as_posix())["audit"]
    assert audit["reviewed"] is True
    assert audit["re_audit_required"] is False

    path.write_bytes((canonical + "\n# semantic edit\n").replace("\n", "\r\n").encode("utf-8"))
    dirty = subprocess.run(["git", "diff", "--quiet", "--", source], cwd=repo)
    assert dirty.returncode != 0
    run(repo, "scripts/ai-index", "update")
    audit = readj(repo, shard_path(repo, "audit", source).relative_to(repo).as_posix())["audit"]
    assert audit["reviewed"] is False
    assert audit["re_audit_required"] is True
'''


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    proc = subprocess.run(
        args,
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc


def git_quiet(root: Path, *args: str) -> bool:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False


def git_blob_sha256(root: Path, path: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{path}"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return ""
    if proc.returncode != 0:
        return ""
    import hashlib
    return hashlib.sha256(proc.stdout).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    engine = root / "scripts" / "ai-index"
    tests = root / "tests" / "tooling" / "test_ai_repo_intelligence_v2.py"
    seed_path = root / ".ai" / "audit-seed.json"
    if not engine.is_file() or not tests.is_file() or not seed_path.is_file():
        raise SystemExit("Installed v2 engine/tests/audit seed not found.")

    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    units = dict(seed.get("units", {}))
    if len(units) != 9:
        raise SystemExit(f"Expected exactly 9 historical audit seed units; found {len(units)}")

    # Prove the canonical committed source still matches each historical reviewed hash
    # and that there is no staged/unstaged source edit before restoring review state.
    unsafe: list[str] = []
    canonical_matches = 0
    for path, record in sorted(units.items()):
        expected = str(record.get("content_hash", ""))
        if not expected:
            unsafe.append(f"{path}: missing seed hash")
            continue
        if not git_quiet(root, "diff", "--quiet", "--", path):
            unsafe.append(f"{path}: unstaged source change")
            continue
        if not git_quiet(root, "diff", "--cached", "--quiet", "--", path):
            unsafe.append(f"{path}: staged source change")
            continue
        actual = git_blob_sha256(root, path)
        if actual != expected:
            unsafe.append(f"{path}: canonical HEAD blob does not match reviewed SHA-256")
            continue
        canonical_matches += 1
    if unsafe:
        raise SystemExit(
            "Refusing to restore historical audit state because source identity is not proven:\n- "
            + "\n- ".join(unsafe)
        )

    original = engine.read_text(encoding="utf-8")
    if "CALO_PROFILE_VERSION" not in original:
        raise SystemExit("scripts/ai-index is not the expected CALO v2 engine; refusing to patch.")
    text = original
    pattern = (
        r"def _migrate_legacy_audit_seed\(root: Path, idx: dict\[str, dict\[str, Any\]\], "
        r"out: dict\[str, Any\]\) -> None:.*?\n\n(?=def audit_state)"
    )
    if "def _legacy_review_matches_current(" not in text:
        matches = list(re.finditer(pattern, text, flags=re.S))
        if len(matches) != 1:
            raise SystemExit(
                f"Unable to identify legacy audit-seed migration function uniquely; found {len(matches)} matches."
            )
        text = re.sub(pattern, lambda _m: AUDIT_HELPERS + "\n\n", text, count=1, flags=re.S)

    try:
        compile(text, str(engine), "exec")
    except SyntaxError as exc:
        raise SystemExit(
            f"Refusing to write invalid scripts/ai-index candidate: line {exc.lineno}: {exc.msg}"
        ) from exc
    engine.write_text(text, encoding="utf-8", newline="\n")

    test_text = tests.read_text(encoding="utf-8")
    if "def test_audit_seed_survives_clean_crlf_materialization_but_not_real_edit" not in test_text:
        anchor = "\ndef test_rename_delete_and_stale_module_cleanup"
        if anchor not in test_text:
            raise SystemExit("Unable to place cross-platform audit regression test.")
        test_text = test_text.replace(anchor, REGRESSION_TEST + anchor, 1)
        tests.write_text(test_text, encoding="utf-8", newline="\n")

    run(root, sys.executable, "scripts/ai-index", "update")
    run(root, sys.executable, "scripts/ai-index", "check")

    manifest = json.loads((root / ".ai/index/manifest.json").read_text(encoding="utf-8"))
    retained = 0
    failures: list[str] = []
    raw_matches = 0
    canonical_fallbacks = 0
    for path, record in sorted(units.items()):
        meta = manifest.get("files", {}).get(path)
        if not meta:
            failures.append(f"{path}: missing from manifest")
            continue
        if meta.get("content_sha256") == record.get("content_hash"):
            raw_matches += 1
        else:
            canonical_fallbacks += 1
        audit_path = root / ".ai/index/audit" / f"{meta['key']}.json"
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))["audit"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: unreadable audit shard ({exc})")
            continue
        if audit.get("reviewed") is not True:
            failures.append(f"{path}: reviewed=false")
            continue
        if audit.get("re_audit_required") is True:
            failures.append(f"{path}: re_audit_required=true")
            continue
        if audit.get("last_reviewed_content_sha256") != record.get("content_hash"):
            failures.append(f"{path}: reviewed content hash mismatch")
            continue
        if audit.get("last_reviewed_commit") != record.get("last_reviewed_commit"):
            failures.append(f"{path}: reviewed commit mismatch")
            continue
        retained += 1
    if failures:
        raise SystemExit(
            "Historical audit preservation is still incomplete after remediation:\n- "
            + "\n- ".join(failures)
        )

    report = {
        "patch": "v2-cross-platform-audit-seed-preservation",
        "historical_seed_units": len(units),
        "canonical_identity_matches_before_patch": canonical_matches,
        "retained_reviewed_units": retained,
        "raw_worktree_sha256_matches": raw_matches,
        "canonical_git_blob_fallbacks": canonical_fallbacks,
        "engine_patched": text != original,
        "regression_test_present": True,
    }
    out = root / ".ai-tmp" / "audit-seed-eol-remediation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print("\nAudit-seed EOL remediation completed.")
    print(f"Historical audit units retained: {retained}/9")
    print(f"Raw working-tree SHA-256 matches: {raw_matches}")
    print(f"Canonical Git-blob fallbacks: {canonical_fallbacks}")
    print("Real staged/unstaged source edits remain fail-closed and require re-audit.")
    print(f"Report: {out}")
    print("No scientific workload or pytest suite was run by this remediation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
