# Known-issue memory

Canonical machine-readable issue memory is `.ai/findings.json`. This file is intentionally short.

- Resolved findings remain recorded so future audits do not rediscover them as open defects.
- Rejected findings must retain the rejection reason and re-open condition.
- A resolved finding may be re-opened when its affected file/symbol changes in a way that invalidates recorded validation evidence.
- Audit coverage is separate from findings: use `python scripts/ai-index query get_audit_state <file|symbol|module>` and the canonical `.ai/index/audit/` shards. A file can have no open finding and still be unaudited; do not rely on deleted v1 root `audit-coverage.json`.
