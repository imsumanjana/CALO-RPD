# Current implementation gates

**Updated:** 2026-08-21  
**Active identity:** `12.0.0.dev1`  
**Overall status:** development; release gates open

Follow these gates in order. The detailed superseded gate history is recoverable from Git checkpoint `ba597eb`.

| Order | Gate | Current state | Exit requirement |
|---:|---|---|---|
| 1 | Source and repository integrity | Current source is retained at the active Git checkpoint; repository cleanup is unvalidated worktree change | Review cleanup diff, refresh repository intelligence, and establish exact source identity |
| 2 | Current-source engineering validation | **Pending.** `phase6-20260817-235629` passed 17/17 only for source commit `4560b2f...`; later source changes supersede it for current-source claims | Owner-run complete Phase 6 validator on the final reviewed source, with stable before/after identity and all commands passing |
| 3 | Runtime and GUI acceptance | Automated contracts exist; human usability, accessibility, and scientist acceptance are not inferred | Separate authorized acceptance against the current validated build |
| 4 | Policy and scientific qualification | No current policy qualification, activation, binding, superiority, or protected-case claim is established by development validation | Exact immutable candidate, compatible runtime/training contract, frozen scientific plan, complete candidate-bound evidence, explicit qualification decision, then separate activation |
| 5 | Physical/container/final-candidate evidence | Historical container payloads were removed from the active tree; prior metadata does not qualify a future final candidate | Repeat CPU/CUDA/container/security/parity/soak gates for one immutable final candidate and trusted CI |
| 6 | Release authorization | **Not authorized.** No final freeze, publication, tag, merge, or release approval | All prior gates closed with direct evidence plus explicit owner authorization |

## Non-negotiable boundaries

- TSH-CALO A-E are the production architecture surface; F is experimental, independently feature-gated, and disabled by default.
- CUDA-preferred and CPU-only are executable; Intel XPU is not executable.
- Training, formal qualification, explicit activation, and immutable experiment binding are separate actions.
- Policy age or software revision alone does not grant or deny compatibility or quality.
- Protected case118/case300 work remains isolated and separately authorized.
- Exact function-evaluation accounting, immutable plans, seeds, resume compatibility, and safe fallback must remain fail closed.
