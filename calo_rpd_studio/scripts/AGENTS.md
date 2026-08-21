<!-- REPOSITORY_INTELLIGENCE_PROTECTED:BEGIN v2 -->
# Repository Intelligence — Mandatory First Step

This block is a protected repository policy. It **must remain the first content** in every repository `AGENT.md` / `AGENTS.md` file. Do not delete, rewrite, relocate, summarize, weaken, or replace it. If an agent-instruction file is edited, preserve this block byte-for-byte at the top. If it is missing or damaged, immediately run `python scripts/ai-agent-guard.py --repair`.

Before broad source exploration or code changes:
1. Read `.ai/REPO_MAP.md` first, then only the relevant curated or generated `.ai/modules/**/*.md` guidance.
2. Run `python scripts/ai-index check`. If stale, run `python scripts/ai-index update` before relying on repository intelligence.
3. Run `python scripts/ai-index context "<task/question>" --no-semantic` before broad source searching. Semantic retrieval is optional and may be enabled with `--semantic`.
4. Architectural routing outranks similarity. Prefer `preferred_edit_target`, `public_surface`, `state_owner`, `entry_point`, and `architectural_boundary` over a higher raw text/vector match from internal implementation code.
5. Before editing, inspect relevant callers/callees, dependencies/dependents, mapped tests, `.ai/findings.json`, audit state, and recent-change state when they affect the task.
6. After code changes, run `python scripts/ai-index update` and the relevant tests/validation. Indexing or embedding **never** marks code audited and never proves correctness.
7. Do not silently bypass unsupported-mode, compatibility, authority, safety, or release gates documented by the repository or its intelligence metadata.
8. Any future agent that modifies any `AGENT.md` / `AGENTS.md` file must keep this entire protected block at the top exactly as written. Run `python scripts/ai-agent-guard.py --check` before finishing such a change.

The canonical block hash is stored in `.ai/agent-policy.json`. Local pre-commit and CI guards may reject changes that remove or alter this block.
<!-- REPOSITORY_INTELLIGENCE_PROTECTED:END v2 -->

# Script instructions

- Scripts must be deterministic, non-interactive where practical, and fail closed on integrity errors.
- Never fabricate attestations or overwrite historical release evidence.
- Generation/check modes must clearly separate source inputs from derived artifacts.
