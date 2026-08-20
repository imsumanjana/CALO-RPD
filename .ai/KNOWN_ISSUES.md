# Known-issue memory

Canonical machine-readable issue memory is `.ai/findings.json`. This file is intentionally short.

- Resolved findings remain recorded so future audits do not rediscover them as open defects.
- Rejected findings must retain the rejection reason and re-open condition.
- A resolved finding may be re-opened when its affected file/symbol changes in a way that invalidates recorded validation evidence.
- `audit-coverage.json` is separate: a file can have no open finding and still be unaudited.
