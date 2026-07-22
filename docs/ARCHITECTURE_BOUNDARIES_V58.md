# CALO-RPD v5.8 architecture boundaries

v5.8 freezes several service contracts even where legacy modules remain physically large.

- **Competitive session transaction:** workers write staged exact state; immutable generation assembly precedes the authoritative root manifest commit.
- **Safe checkpoint manager contract:** session-start exact state plus fixed 10-epoch rolling common-safe points; exact state and UI telemetry are separate.
- **Champion evidence contract:** comparator schema + validation-bundle fingerprint; hardware runtime is diagnostic, not a scientific quality vote.
- **Global Base contract:** all final candidates are evaluated in one common bundle and selected order-independently; provisional models are never silently promoted.
- **Resource admission contract:** each branch receives an explicit admitted device/lane assignment recorded in provenance.
- **Recovery contract:** interrupted scratch is discoverable by a durable recovery index; recovery restores exact branch state without promoting unfinalized champions.
- **Publication contract:** publication-grade verified-only mode requires the complete expected evidence set, not merely a nonempty verified subset.
- **Trust contract:** deployable weights use restricted loading; pickle-capable exact resume is trusted-local/authenticated only; legacy migration is explicit and opt-in.

Further mechanical module decomposition may occur in later releases without changing these scientific contracts.
