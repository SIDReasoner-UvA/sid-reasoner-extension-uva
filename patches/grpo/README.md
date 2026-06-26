# grpo — SIDReasoner patches for RQ4 (GRPO reward-shaping)

RQ4 reward variants are implemented entirely in `src/grpo/` of this repo and
injected via shims — no separate SIDReasoner patch is needed beyond the common one.
All SIDReasoner working-tree changes are tracked in
[`../common/sidreasoner-local.patch`](../common/sidreasoner-local.patch).

See [docs](../../docs/grpo/README.md) for experiment design and findings.
