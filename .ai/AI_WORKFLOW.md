# AI-assisted development workflow

CALO-RPD repository intelligence is a navigation and change-maintenance layer. It does not establish correctness, scientific validity, release readiness, or audit coverage.

1. Read `.ai/REPO_MAP.md` before broad repository exploration.
2. Read only the relevant curated `.ai/modules/*.md` and generated `.ai/modules/generated/*.md` guidance.
3. Check freshness with `python scripts/ai-index check`.
4. Build deterministic targeted context with `python scripts/ai-index context "<question>" --no-semantic` before opening many files.
5. Respect `preferred_edit_target`, `public_surface`, `state_owner`, `entry_point`, `authoritative_source`, `integration_layer`, `internal_only`, and `architectural_boundary`. Architectural authority outranks similarity.
6. Inspect callers/callees, dependencies/dependents, mapped tests, `.ai/findings.json`, audit state, and actual recent changes when relevant.
7. After source changes run `python scripts/ai-index update`; only affected shards should change. Unchanged audit state is retained by content identity; previously reviewed changed content requires re-audit.
8. Before commit handoff run `python scripts/ai-index precommit --no-semantic`. The local hook may stage generated intelligence shards only; curated architecture/decisions/findings are never auto-staged.
9. GitHub CI remains read-only and independently verifies agent policy, freshness, byte-stable regeneration, and the repository-specific semantic benchmark.
10. Optional local semantic retrieval uses `.ai-cache/`; deterministic retrieval remains fully functional when semantic cache/provider is absent, disabled, or corrupt.

Useful deterministic query operations:

```text
repo_overview
search_code
find_symbol
get_symbol
get_callers
get_callees
get_dependencies
get_dependents
get_tests
get_findings
get_audit_state
get_recent_changes
get_architecture
get_decisions
build_context
```

Examples:

```bash
python scripts/ai-index query get_symbol calo_rpd_studio.app.state_manager.AppState
python scripts/ai-index query get_tests calo_rpd_studio.algorithms.calo.policy_registry.PolicyRegistry
python scripts/ai-index query get_recent_changes
python scripts/ai-index embeddings update
python scripts/ai-index embeddings benchmark --check
```

## Architecture freeze contract

After the migration regression suite and fresh-session benchmark pass on the full repository checkout, freeze schema v2, the sharding model, architectural-routing contract, call-confidence policy, symbol/test mapping policy, freshness model, audit semantics, deterministic query surface, semantic/hybrid ranking contract, and agent guard policy. Generated `.ai/index/**`, module intelligence, recent-change state, and `INDEX_STATUS.md` remain live and incremental.
