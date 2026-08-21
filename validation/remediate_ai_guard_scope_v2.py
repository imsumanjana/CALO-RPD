#!/usr/bin/env python3
"""Undo over-broad v2 agent-policy edits and install the scoped guard safely."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXED_GUARD = HERE / "ai-agent-guard-v2-fixed.py"


def run(root: Path, *args: str, capture: bool = False) -> subprocess.CompletedProcess:
    print("+", " ".join(args))
    proc = subprocess.run(
        args,
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
        check=False,
    )
    if capture:
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc


def git_bytes(root: Path, *args: str) -> bytes | None:
    try:
        proc = subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except OSError:
        return None
    return proc.stdout if proc.returncode == 0 else None


def load_fixed_guard():
    spec = importlib.util.spec_from_file_location("calo_fixed_agent_guard", FIXED_GUARD)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {FIXED_GUARD}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if not FIXED_GUARD.is_file():
        print(f"Missing remediation dependency: {FIXED_GUARD}", file=sys.stderr)
        return 2
    guard = load_fixed_guard()
    root_proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=HERE,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if root_proc.returncode:
        print("Run this remediation from inside the CALO-RPD Git checkout.", file=sys.stderr)
        return 2
    root = Path(root_proc.stdout.strip()).resolve()
    policy_path = root / ".ai" / "agent-policy.json"
    if not policy_path.exists():
        print("Missing .ai/agent-policy.json; nothing can be safely scoped from the previous run.", file=sys.stderr)
        return 2
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    old_targets = [str(x).replace("\\", "/") for x in policy.get("agent_files", [])]
    contaminated = [rp for rp in old_targets if not guard.policy_path_allowed(rp)]

    restored_exact: list[str] = []
    restored_inverse: list[str] = []
    already_clean: list[str] = []
    unresolved: list[str] = []

    print(f"Previous guard targets: {len(old_targets)}")
    print(f"Out-of-scope targets to remediate: {len(contaminated)}")

    for rp in contaminated:
        p = root / rp
        if not p.is_file():
            continue
        current = p.read_text(encoding="utf-8", errors="replace")
        if not current.startswith(guard.BLOCK):
            already_clean.append(rp)
            continue
        head = git_bytes(root, "show", f"HEAD:{rp}")
        if head is not None:
            head_text = head.decode("utf-8", "replace")
            normalized = guard.normalize_rest(head_text)
            expected_after_old_guard = guard.BLOCK + ("\n" + normalized if normalized else "")
            if current == expected_after_old_guard:
                p.write_bytes(head)
                restored_exact.append(rp)
            else:
                unresolved.append(rp)
            continue

        rest = current[len(guard.BLOCK):]
        if rest.startswith("\n"):
            rest = rest[1:]
        p.write_text(rest, encoding="utf-8", newline="\n")
        restored_inverse.append(rp)

    if unresolved:
        report = {
            "restored_exact": restored_exact,
            "restored_inverse": restored_inverse,
            "already_clean": already_clean,
            "unresolved": unresolved,
        }
        out = root / ".ai-tmp" / "guard-scope-remediation-unresolved.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("Refusing to continue because some tracked evidence files have additional changes:", file=sys.stderr)
        for rp in unresolved:
            print(f"  {rp}", file=sys.stderr)
        print(f"Details: {out}", file=sys.stderr)
        return 3

    target_guard = root / "scripts" / "ai-agent-guard.py"
    target_guard.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(FIXED_GUARD, target_guard)

    run(root, sys.executable, "scripts/ai-agent-guard.py", "--repair", "--root", ".", "--canonical", "AGENTS.md")
    run(root, sys.executable, "scripts/ai-agent-guard.py", "--install-hook", "--root", ".")
    run(root, sys.executable, "scripts/ai-agent-guard.py", "--check", "--root", ".")
    run(root, sys.executable, "scripts/ai-index", "update")
    run(root, sys.executable, "scripts/ai-index", "check")

    new_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    remaining_bad = [rp for rp in new_policy.get("agent_files", []) if not guard.policy_path_allowed(rp)]
    if remaining_bad:
        print("Corrected policy still contains ignored targets: " + ", ".join(remaining_bad), file=sys.stderr)
        return 4

    change_path = root / ".ai" / "index" / "change-index.json"
    change = json.loads(change_path.read_text(encoding="utf-8")) if change_path.exists() else {}
    legacy = [
        name for name in (
            "file-index.json", "symbol-index.json", "dependency-graph.json",
            "test-map.json", "audit-coverage.json", "change-index.json",
        )
        if (root / ".ai" / name).exists()
    ]
    summary = {
        "previous_guard_targets": len(old_targets),
        "out_of_scope_targets": len(contaminated),
        "restored_exact_from_head": restored_exact,
        "restored_best_effort_untracked": restored_inverse,
        "already_clean": already_clean,
        "active_policy_targets": len(new_policy.get("agent_files", [])),
        "active_policy_scope_ok": not remaining_bad,
        "recent_change_initialization": change.get("initialization"),
        "recent_changed_files": change.get("changed_files", []),
        "recent_new_files": change.get("newly_indexed_files", []),
        "recent_deleted_files": change.get("deleted_files", []),
        "legacy_ai_monoliths_present": legacy,
    }
    out = root / ".ai-tmp" / "guard-scope-remediation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("\nGuard-scope remediation completed.")
    print(f"Exact tracked restorations: {len(restored_exact)}")
    print(f"Best-effort ignored/untracked reversals: {len(restored_inverse)}")
    print(f"Active agent-policy targets: {summary['active_policy_targets']}")
    print(f"Recent-change initialization flag: {summary['recent_change_initialization']}")
    print(f"Recent changed files: {len(summary['recent_changed_files'])}")
    print(f"Legacy .ai monoliths present: {', '.join(legacy) if legacy else 'none'}")
    print(f"Report: {out}")
    print("No scientific workload or pytest suite was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
