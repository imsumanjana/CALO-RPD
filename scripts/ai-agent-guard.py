#!/usr/bin/env python3
"""Protect active repository agent instructions without traversing evidence/archive trees."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = "2.1.1"
BEGIN = "<!-- REPOSITORY_INTELLIGENCE_PROTECTED:BEGIN v2 -->"
END = "<!-- REPOSITORY_INTELLIGENCE_PROTECTED:END v2 -->"
BLOCK = f"""{BEGIN}
# Repository Intelligence — Mandatory First Step

This block is a protected repository policy. It **must remain the first content** in every repository `AGENT.md` / `AGENTS.md` file. Do not delete, rewrite, relocate, summarize, weaken, or replace it. If an agent-instruction file is edited, preserve this block byte-for-byte at the top. If it is missing or damaged, immediately run `python scripts/ai-agent-guard.py --repair`.

Before broad source exploration or code changes:
1. Read `.ai/REPO_MAP.md` first, then only the relevant curated or generated `.ai/modules/**/*.md` guidance.
2. Run `python scripts/ai-index check`. If stale, run `python scripts/ai-index update` before relying on repository intelligence.
3. Run `python scripts/ai-index context \"<task/question>\" --no-semantic` before broad source searching. Semantic retrieval is optional and may be enabled with `--semantic`.
4. Architectural routing outranks similarity. Prefer `preferred_edit_target`, `public_surface`, `state_owner`, `entry_point`, and `architectural_boundary` over a higher raw text/vector match from internal implementation code.
5. Before editing, inspect relevant callers/callees, dependencies/dependents, mapped tests, `.ai/findings.json`, audit state, and recent-change state when they affect the task.
6. After code changes, run `python scripts/ai-index update` and the relevant tests/validation. Indexing or embedding **never** marks code audited and never proves correctness.
7. Do not silently bypass unsupported-mode, compatibility, authority, safety, or release gates documented by the repository or its intelligence metadata.
8. Any future agent that modifies any `AGENT.md` / `AGENTS.md` file must keep this entire protected block at the top exactly as written. Run `python scripts/ai-agent-guard.py --check` before finishing such a change.

The canonical block hash is stored in `.ai/agent-policy.json`. Local pre-commit and CI guards may reject changes that remove or alter this block.
{END}
"""
BLOCK_HASH = hashlib.sha256(BLOCK.encode("utf-8")).hexdigest()
AGENT_RE = re.compile(r"^agents?\.md$", re.IGNORECASE)
BLOCK_RE = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\s*", re.DOTALL)

POLICY_IGNORE_PREFIXES = (
    ".ai/", ".ai-cache/", ".ai-tmp/", ".git/", ".venv/", "venv/", "env/",
    "node_modules/", "build/", "dist/", "vendor/", "generated/", "__pycache__/",
    "artifacts/", "validation/", "publication_export/",
    "calo_rpd_studio/data/pglib/", "calo_rpd_studio/data/trained_models/",
    "calo_rpd_studio/data/frozen/",
)


def rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def policy_path_allowed(rp: str) -> bool:
    rp = rp.replace("\\", "/").lstrip("./")
    return not any(rp.startswith(prefix) for prefix in POLICY_IGNORE_PREFIXES)


def is_git_root(root: Path) -> bool:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True,
            encoding="utf-8", errors="replace", capture_output=True, check=False,
        )
    except OSError:
        return False
    if proc.returncode != 0:
        return False
    try:
        return Path(proc.stdout.strip()).resolve() == root.resolve()
    except OSError:
        return False


def agent_files(root: Path) -> list[Path]:
    """Return active tracked/untracked-not-ignored agent instruction files only."""
    root = root.resolve()
    out: list[Path] = []
    if is_git_root(root):
        try:
            proc = subprocess.run(
                ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            if proc.returncode == 0:
                for raw in sorted(x for x in proc.stdout.split(b"\0") if x):
                    rp = raw.decode("utf-8", "replace").replace("\\", "/")
                    p = root / rp
                    if p.is_file() and AGENT_RE.match(p.name) and policy_path_allowed(rp):
                        out.append(p)
                return sorted(out, key=lambda p: rel_posix(p, root).lower())
        except OSError:
            pass

    for base, dirs, names in os.walk(root):
        bp = Path(base)
        base_rel = "" if bp == root else rel_posix(bp, root)
        kept = []
        for d in sorted(dirs):
            rp_dir = ((base_rel + "/" + d).strip("/") + "/")
            if policy_path_allowed(rp_dir) and not (bp / d).is_symlink():
                kept.append(d)
        dirs[:] = kept
        for name in sorted(names):
            if not AGENT_RE.match(name):
                continue
            p = bp / name
            rp = rel_posix(p, root)
            if p.is_file() and policy_path_allowed(rp):
                out.append(p)
    return sorted(out, key=lambda p: rel_posix(p, root).lower())


def normalize_rest(text: str) -> str:
    text = BLOCK_RE.sub("", text)
    return text.lstrip("\ufeff\r\n ")


def repair_file(path: Path) -> bool:
    old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    rest = normalize_rest(old)
    new = BLOCK + ("\n" + rest if rest else "")
    if old == new:
        return False
    path.write_text(new, encoding="utf-8", newline="\n")
    return True


def write_policy(root: Path, files: list[Path]) -> None:
    policy = {
        "schema_version": 1,
        "policy_version": VERSION,
        "protected_block_sha256": BLOCK_HASH,
        "begin_marker": BEGIN,
        "end_marker": END,
        "agent_files": [rel_posix(p, root) for p in files],
        "ignored_prefixes": list(POLICY_IGNORE_PREFIXES),
        "selection": "git tracked plus untracked-not-ignored, filtered by active-policy prefixes",
        "rule": "Protected block must be byte-for-byte first content in every active repository AGENT.md/AGENTS.md file.",
    }
    p = root / ".ai" / "agent-policy.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def check(root: Path) -> int:
    files = agent_files(root)
    if not files:
        print("No active AGENT.md/AGENTS.md file found. Run with --repair.")
        return 1
    bad: list[str] = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        if not text.startswith(BLOCK):
            bad.append(rel_posix(p, root))
    policy_path = root / ".ai" / "agent-policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8")) if policy_path.exists() else {}
    expected_files = [rel_posix(p, root) for p in files]
    if policy.get("protected_block_sha256") != BLOCK_HASH:
        bad.append(".ai/agent-policy.json:hash")
    if policy.get("agent_files") != expected_files:
        bad.append(".ai/agent-policy.json:scope")
    if any(not policy_path_allowed(rp) for rp in policy.get("agent_files", [])):
        bad.append(".ai/agent-policy.json:ignored-target")
    if bad:
        print("Repository-intelligence agent policy is missing, modified, or mis-scoped in: " + ", ".join(bad))
        return 1
    print(f"Agent policy OK: {len(files)} active file(s), sha256={BLOCK_HASH}")
    return 0


def repair(root: Path, canonical: str) -> int:
    files = agent_files(root)
    if not files:
        files = [root / canonical]
    changed: list[str] = []
    for p in files:
        if repair_file(p):
            changed.append(rel_posix(p, root))
    files = agent_files(root)
    write_policy(root, files)
    print("Agent policy repaired." if changed else "Agent policy already correct.")
    if changed:
        print("Updated: " + ", ".join(changed))
    return check(root)


def install_hook(root: Path) -> int:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True,
            encoding="utf-8", errors="replace", capture_output=True,
        )
    except OSError as exc:
        print(f"Git unavailable; hook not installed: {exc}")
        return 1
    if proc.returncode != 0 or Path(proc.stdout.strip()).resolve() != root.resolve():
        print("Target is not a Git worktree root; hook not installed.")
        return 1
    gd = subprocess.run(
        ["git", "rev-parse", "--git-dir"], cwd=root, text=True,
        encoding="utf-8", errors="replace", capture_output=True,
    )
    if gd.returncode != 0:
        print("Unable to locate .git directory.")
        return 1
    gitdir = Path(gd.stdout.strip())
    if not gitdir.is_absolute():
        gitdir = (root / gitdir).resolve()
    hooks = gitdir / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    hook = hooks / "pre-commit"
    user_hook = hooks / "pre-commit.user"
    marker = "CALO_REPO_INTELLIGENCE_GUARD_V2"
    if hook.exists():
        current = hook.read_text(encoding="utf-8", errors="replace")
        if marker not in current:
            if user_hook.exists():
                shutil.copy2(user_hook, user_hook.with_name(user_hook.name + ".backup"))
            hook.replace(user_hook)
    hook_text = r'''#!/bin/sh
# CALO_REPO_INTELLIGENCE_GUARD_V2
PYTHON=""
if command -v python >/dev/null 2>&1; then PYTHON=python
elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
elif command -v py.exe >/dev/null 2>&1; then PYTHON="py.exe -3"
else
  echo "Python 3 is required for repository-intelligence validation." >&2
  exit 1
fi
$PYTHON scripts/ai-index precommit --no-semantic || exit $?
# Stage only canonical generated intelligence; never curated knowledge or legacy monoliths.
git add -- .ai/index .ai/INDEX_STATUS.md .ai/modules/generated 2>/dev/null || true
if [ -x "$0.user" ]; then
  "$0.user" "$@" || exit $?
elif [ -f "$0.user" ]; then
  sh "$0.user" "$@" || exit $?
fi
exit 0
'''
    hook.write_text(hook_text, encoding="utf-8", newline="\n")
    try:
        os.chmod(hook, 0o755)
    except OSError:
        pass
    print(f"Installed repository-intelligence pre-commit guard: {hook}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--repair", action="store_true")
    action.add_argument("--install-hook", action="store_true")
    parser.add_argument("--root", default=".")
    parser.add_argument("--canonical", default="AGENTS.md")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.check:
        return check(root)
    if args.repair:
        return repair(root, args.canonical)
    return install_hook(root)


if __name__ == "__main__":
    raise SystemExit(main())
