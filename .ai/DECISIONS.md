# Repository decisions worth retaining

1. **Desktop-first native workflow.** PyQt6/Python control, logging, SQLite and final host validation remain host responsibilities; accelerated numerical work may be CUDA-resident.
2. **CUDA-preferred / CPU-only execution.** Intel XPU is visible only as historical/hardware context and must not be executable in the active contract.
3. **Safe resource admission.** Admission uses at most 80% of currently free VRAM or available RAM.
4. **Policy lifecycle separation.** Training, qualification, activation and experiment consumption are distinct gates. No experiment may auto-train, auto-qualify or auto-activate a policy.
5. **Compatibility is semantic, not age-based.** Exact resume/extension and qualification depend on frozen ABI/architecture/state/schema/accounting evidence; source revision alone is not incompatibility.
6. **Historical evidence is scoped.** Older v6.x release/audit evidence does not qualify the active v12 tree.
7. **AI repository intelligence is deterministic first.** Repository-local intelligence uses hashes, ASTs, Git and lexical/graph retrieval before embeddings. Embeddings are an optional later extension, not a dependency of correct indexing.

Sources for decisions 1-6 are repository `README.md` and `AGENTS.md`; preserve that provenance instead of inventing new policy here.
