# CALO-RPD v5.8 competitive-training closure contract

## Authoritative branch state

A competitive exact-resume session is a generation transaction. Worker processes never write directly into the currently authoritative branch generation. They write session-staged trusted exact states. A new immutable generation is assembled and verified first; only the final atomic root branch-manifest replacement makes that generation authoritative.

## Safe Stop contract

The session-start exact state is always a valid safe point. Rolling safe points occur every 10 completed epochs. Safe Stop resolves to the latest exact epoch available to every branch, not to an arbitrary fastest-branch epoch. If a child does not exit within the grace interval, it may be terminated and the result is marked degraded rather than completed.

## Infinite-training boundedness

Exact resume stores only a bounded recent history. Detailed epoch telemetry is append-only external data. Coordinator messages and champion decisions use bounded deques. Curriculum progression uses explicit persisted milestones and does not depend on the session-duration `epochs` field in Infinite mode.

## Champion/Base evidence contract

Branch promotion is feasibility-first. Runtime latency is diagnostic. Each evidence vector has a comparator schema and validation-bundle fingerprint. Final candidates are evaluated together under one common evidence bundle and selected with a deterministic order-independent ranking. Stale evidence from another bundle is not compared directly.

## Recovery contract

A durable recovery index identifies interrupted session scratch and the latest common safe epoch. Recovery commits a coherent branch generation and preserves the previously committed Base; unfinalized branch champions are never promoted during crash recovery.
