# AI-assisted development workflow

1. Read `REPO_MAP.md` and only the relevant file in `modules/`; do not read the whole repository.
2. Use `symbol-index.json` for definitions/line ranges, `dependency-graph.json` for relationships, and `test-map.json` before opening source/tests.
3. Check `DECISIONS.md`, `findings.json`, and `audit-coverage.json` before repeating architecture decisions or audits.
4. Run `./scripts/ai-index status` to see whether source changed since the committed index.
5. For a question, prefer `./scripts/ai-index context "<question>"` and inspect only the selected exact source ranges.
6. After code changes, run `./scripts/ai-index update`; unchanged-file parse facts/audit state are retained and changed code is marked for re-audit.
7. Before committing, run `./scripts/ai-index check` (PowerShell: `scripts/ai-index.ps1 check`).
8. Update authored architecture/decision text only when the architecture or repository constraints actually change.

`change-index.json` answers what files/symbols were affected by the last incremental reparse. Indexing is navigation evidence, not proof of correctness, security, test coverage or scientific validity.

The local functions deliberately map cleanly to future MCP operations: overview, symbol search/get, callers/callees, dependencies/dependents, tests, decisions, findings, recent changes, and context building. A remote MCP server is not required for correctness.
