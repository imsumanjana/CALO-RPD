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

# CALO-RPD agent instructions

## Scope
These instructions apply to the entire repository unless a deeper `AGENTS.md` narrows them.

## Scientific and release boundaries
- Preserve deterministic baseline behavior and exact function-evaluation accounting.
- TSH-CALO A–E are approved for production-candidate implementation. F is experimental, independently feature-flagged, and disabled by default.
- Do not change scientific semantics outside the approved A–F proposal without explicit user approval.
- Policy training is independent. Experiments may only consume separately qualified, explicitly activated, immutable, checksum-valid policies.
- Never auto-train, auto-modify, auto-qualify, or auto-activate a policy from an experiment workflow.
- Development-phase labels and software-version identities are provenance only. They must not decide
  whether an immutable policy may be extended, formally qualified, activated, or compared.
- Admit a policy to formal qualification when its checksum-valid artifact proves the current frozen
  TSH-CALO runtime ABI, policy architecture, state/action schemas, training-environment schema,
  epistemic-ensemble structure, authenticated training receipts, and protected-case isolation.
- Exact resume or completed-training extension additionally requires the retained frozen training
  compatibility contract: policy parameter names/shapes/dtypes, persisted training-parameter schema,
  optimizer/trainer/session/environment state schemas, and exact evaluation-accounting semantics.
  A software source revision alone is not an incompatibility. Added, removed, or semantically changed
  architecture, parameters, persisted state, schemas, or evaluation accounting must fail closed.
- Existing and earlier-version policies are not presumed good, bad, qualified, or incompatible because
  of their age or originating development phase. Missing integrity or compatibility evidence must be
  reported precisely and must never be fabricated or retroactively asserted.
- Qualification is candidate-bound scientific quality evidence, not a development acceptance receipt.
  It must retain the frozen comparison plan, equal exact function-evaluation budgets, protected-case
  isolation, feasibility, objective, effect-size, significance, anytime, stability, and OOD evidence.
- A qualification pass never activates a policy. Experiment use still requires a separately qualified,
  explicitly activated, immutable, checksum-valid policy and exact qualification receipt binding.
- Policy deletion requires an exact reviewed target and explicit confirmation. Active, qualified, or
  experiment-referenced policies remain protected by the retirement workflow.
- Preserve explicit safe fallback for unavailable, incompatible, or rejected policies.
- Intel XPU must not be executable. Current modes are CUDA-preferred and CPU-only.
- Admission ceilings use at most 80% of currently free VRAM or currently available RAM. CUDA computes on NVIDIA GPUs; CPU fallback computes on CPUs.
- Do not fabricate hardware, container, performance, energy, thermal, or scientific evidence.
- Do not make release-ready or superiority claims before their documented gates have direct evidence.

## Workflow
- Follow `docs/implementation/IMPLEMENTATION_GATES.md` in order for engineering and release evidence,
  and keep the handoff and traceability ledger current. Those phase gates must not substitute for
  policy compatibility or scientific qualification evidence.
- Before stating or starting any numbered development phase, create a new phase-specific goal with
  the goal service. Do not begin source implementation until that goal exists. If another goal is
  unfinished, resolve it according to the goal-service rules before starting the new phase.
- Phase work is coding-only unless the user explicitly authorizes named execution commands in a
  later message. Implement required production code, test source, schemas, documentation, and
  validation harnesses, but do not execute tests or validation that the user can run manually.
- Do not run phase validators, pytest/tox/coverage, compile checks, schema checks, lint/format checks,
  type checks, package/build smoke tests, GUI/browser smoke tests, Docker validation, benchmarks,
  campaigns, policy training/evaluation, qualification, or protected-case workflows as part of
  phase development. Give the user the exact manual command instead.
- End every phase-development pass by creating or updating a detailed PowerShell validator under
  the Git-ignored `validation/` directory. It must exercise the checks required for that phase,
  create a newly timestamped detailed log directory, record command results and relevant source/
  validator hashes, and remain absent from Git source, manifests, distributions, and release
  artifacts. Do not run the validator; ask the user to run it and return the complete log directory.
- Phase validators must be noninteractive. Do not request or gate on reviewer names, roles,
  PASS/FAIL answers, free-form attestations, screen-reader declarations, or other `Read-Host` input.
  Use reproducible automated checks and retained artifacts only. Record explicitly when human
  screen-reader, usability, or scientist acceptance is not inferred by automation.
- Review user-returned validation logs read-only, make evidence-backed coding corrections, update the
  same ignored validator when necessary, and request a fresh user-executed noninteractive rerun.
  Minimize tool calls and output to reduce token usage; avoid redundant scans or repeated evidence
  dumps.
- Before semantic CALO changes, prove canonical-refactor parity against the frozen baseline.
- Add unit, invariant, parity, ablation, falsification, leakage, fallback, and regression tests proportional to each change.
- Keep protected cases out of training, tuning, reward design, and checkpoint selection.
- Use empty-policy behavior and synthetic temporary policy fixtures for development tests. Do not
  execute real policy training, qualification, activation, deletion, or protected-case workflows
  unless the user explicitly authorizes the named execution command in a later message.
- Do not regenerate release freezes, manifests, SBOMs, image digests, or public release claims before their gates close.
- Preserve user files and unrelated changes. Do not push, merge, publish, or release without explicit approval.
