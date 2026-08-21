# Repository Intelligence v2 Freeze Record

Date: 2026-08-21

This record freezes the **repository-intelligence compatibility contract**, not the generated intelligence snapshot and not scientific/release validity.

## Contract frozen

The following interfaces and semantics are frozen as repository-intelligence v2 compatibility contracts:

- schema v2 and canonical sharded storage under `.ai/index/`;
- architectural routing and authority metadata (`preferred_edit_target`, `public_surface`, `state_owner`, `entry_point`, `authoritative_source`, `integration_layer`, `internal_only`, `architectural_boundary`);
- conservative call-confidence policy, including unresolved arbitrary local-receiver calls unless receiver/type/import evidence exists;
- conservative executable-test classification and symbol/test mapping policy;
- content-hash freshness and recent-change semantics, including initialization separation and rename/delete handling;
- audit preservation/invalidation semantics, including cross-platform canonical Git-blob fallback for unchanged reviewed files;
- deterministic query surface (`repo_overview`, code/symbol search, callers/callees, dependencies/dependents, tests, findings, audit, recent changes, architecture, decisions, context);
- optional semantic/hybrid retrieval with architectural authority outranking similarity and deterministic operation remaining independently usable;
- protected `AGENT.md` / `AGENTS.md` policy and guard behavior;
- read-only CI verification contract.

A future change to one of these contracts requires an explicit versioned migration and renewed regression/fresh-agent/graph-test validation. It must not be introduced as silent implementation drift.

## Live, incremental state

The following remain live and should change as the repository changes:

- `.ai/index/**` generated file/symbol/dependency/test/audit/module shards;
- `.ai/index/manifest.json` and `.ai/index/change-index.json`;
- `.ai/modules/generated/**` and curated module guidance when architecture evolves;
- `.ai/INDEX_STATUS.md`;
- `.ai/findings.json` lifecycle state;
- audit coverage/content identity and recent-change state.

## Freeze evidence

The migration/freeze gate was accepted only after the following repository-local validations passed on the full CALO-RPD checkout:

- protected agent policy: 50 active targets, canonical block hash verified;
- sharded index freshness: 677 indexed files;
- historical audit preservation: 9/9 reviewed units retained with real edits fail-closed;
- byte-stable regeneration: unchanged `ai-index init` produced zero shard writes/removals;
- deterministic routing/public-vs-core ordering checks passed;
- repository semantic benchmark: 5/5 cases passed;
- v2 tooling regression suite: 17/17 passed;
- read-only CI contract passed;
- fresh-agent benchmark: 8/8 routing cases passed with independent source-symbol verification;
- repository-wide graph/test quality audit inspected 5,391 symbols, 35,454 call edges (6,817 confirmed, 172 inferred, 28,465 unresolved), 6,966 resolved repository call edges, 134 mapped executable test paths, and 1,216 symbol-to-test mapping pairs; no false resolved arbitrary-receiver calls or unsupported test mappings were detected.

These validations establish the repository-intelligence migration/freeze contract only. They do **not** establish scientific correctness, hardware/runtime qualification, GUI acceptance, release readiness, or human-scientist acceptance.

## Canonical usage after freeze

1. Read `.ai/REPO_MAP.md` first.
2. Run `python scripts/ai-index check` and update if stale.
3. Use targeted deterministic context before broad source search.
4. Use v2 query operations for graph, tests, audit and recent-change evidence; do not rely on deleted v1 root monolithic indexes.
5. After source changes, update intelligence and run relevant validation; indexing/embedding never marks code audited.
6. Before changing a frozen compatibility contract, version and migrate it explicitly.
