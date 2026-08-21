# Repository decisions worth retaining

1. **Desktop-first native workflow.** PyQt6/Python control, logging, SQLite and final host validation remain host responsibilities; accelerated numerical work may be CUDA-resident.
2. **CUDA-preferred / CPU-only execution.** Intel XPU is visible only as historical/hardware context and must not be executable in the active contract.
3. **Safe resource admission.** Admission uses at most 80% of currently free VRAM or available RAM.
4. **Policy lifecycle separation.** Training, qualification, activation and experiment consumption are distinct gates. No experiment may auto-train, auto-qualify or auto-activate a policy.
5. **Compatibility is semantic, not age-based.** Exact resume/extension and qualification depend on frozen ABI/architecture/state/schema/accounting evidence; source revision alone is not incompatibility.
6. **Historical evidence is scoped.** Older v6.x release/audit evidence does not qualify the active v12 tree.
7. **AI repository intelligence is deterministic first.** Repository-local intelligence uses hashes, ASTs, Git and lexical/graph retrieval before embeddings. Semantic retrieval is optional and must never outrank architectural authority or become a correctness dependency.
8. **Repository intelligence v2 contract is frozen after validated migration.** Schema v2, sharded physical storage, architectural-routing metadata, conservative call-confidence policy, conservative symbol/test mapping policy, freshness/recent-change semantics, audit preservation/invalidation semantics, deterministic query API, semantic/hybrid authority ordering, and protected agent-policy behavior are compatibility contracts. Generated `.ai/index/**`, generated/curated module summaries, recent-change state and `INDEX_STATUS.md` remain live and incremental. A future contract change requires an explicit versioned migration plus regression/fresh-agent validation rather than silent drift.

Sources for decisions 1-6 are repository `README.md` and `AGENTS.md`; preserve that provenance instead of inventing new policy here. Decision 8 is supported by the completed v2.1 migration validation, fresh-agent benchmark, and repository-wide graph/test quality audit performed on 2026-08-21.
