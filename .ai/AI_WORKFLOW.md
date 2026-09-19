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

## Frozen v2 compatibility contract

Repository intelligence v2 was frozen on 2026-08-21 after the full migration regression suite, fresh-agent benchmark, independent source verification, and repository-wide graph/test quality audit passed.

The following are compatibility contracts and must not drift silently: schema v2, sharded physical storage, architectural-routing metadata and precedence, conservative call-confidence rules, conservative symbol/test mapping rules, freshness/recent-change semantics, audit preservation/invalidation semantics, deterministic query operations and response intent, semantic/hybrid authority ordering, and protected `AGENT.md` / `AGENTS.md` policy behavior.

The following remain live and incremental and are **not** frozen snapshots: generated `.ai/index/**`, generated and curated module intelligence, `INDEX_STATUS.md`, recent-change state, findings lifecycle, and audit coverage/content identity as source files evolve.

Any future compatibility-contract change requires an explicit versioned migration with regression, fresh-agent, and graph/test-quality validation. Do not reintroduce v1 root monolithic indexes as compatibility artifacts.

## Byte-metadata correction: indexer 2.1.1 (2026-09-19)

Schema v2, sharding, routing, scientific hashes, and the protected agent blocks are unchanged.
The additive manifest contract `byte_metadata_contract=git-clean-equivalent-v1` records the
corrected meaning of generated `content_sha256` and `size_bytes`: both describe the same
Git-clean-equivalent bytes as `content_hash`. Raw bytes are preferred; CRLF-to-LF conversion
is accepted only when its Git blob digest equals Git's clean-filtered identity. Files with
`-text`, untracked files, and non-Git files retain exact-byte metadata. Unsupported clean
filters fail explicitly. Source files and scientific/checkpoint checksums are never rewritten.

Old manifests fail freshness until `python scripts/ai-index update` regenerates derived byte
metadata. This is a versioned generated-metadata correction, not new scientific or audit
acceptance. Existing reviews survive equivalent checkout materialization; actual content
changes still invalidate them. Snapshot branch provenance is retained for content-identical
regeneration, including detached CI checkouts.

Regression definitions cover both LF/CRLF directions, exact-byte assets, review invalidation,
unknown filters, and detached checkouts. Execution of these tests, fresh-agent retrieval, and
CI regeneration remain pending for this maintenance revision. Before a later authorized
commit, stage intended source changes, then refresh repository intelligence using the normal
precommit workflow; do not publish stale untracked-file identities.
