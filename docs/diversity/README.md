# Diversity Experiments

> **RQ5: Beyond top-line accuracy — how diverse is SIDReasoner in *what* it
> recommends?** We look at how the model distributes its recommendations across the item
> catalog: whether exposure in the top-10 is concentrated on a few items (**item-side
> diversity**).

This suite steps past accuracy (covered by [reproduction](../reproduction/README.md),
[components](../components/README.md), [model_size](../model_size/README.md), and
[grpo](../grpo/README.md)) to ask *how* SIDReasoner spends its probability mass —
operationalised as exposure concentration in the top-10 recommendations.

| Axis | Question | Status |
|------|----------|--------|
| **Item-side diversity** | Is exposure in the top-10 concentrated on a few items? | results **pending** (jobs not yet committed) |

---

## Pipeline context

SIDReasoner trains in three stages; this study branches from the released checkpoints
rather than retraining:

| Stage | Name | Purpose | Checkpoint |
|-------|------|---------|------------|
| S1 | SFT | Emit valid Semantic IDs (SIDs). | `sidreasoner_sft_*` |
| S2 | Reasoning-activation (RA) | Wrap the answer in `<think>…</think>` before the SID. | `sidreasoner_ra_*` |
| S3 | GRPO RL | Reinforce reasoning toward correct recommendations. | post-GRPO |

Exposure is evaluated across **all three** stages (SFT / RA / RL).

Backbone: `Qwen/Qwen3-1.7B` (the paper's setting).

---

## Item-side exposure diversity

> **Does SIDReasoner concentrate exposure on a few recommended items?** We operationalise
> item-side diversity as **exposure concentration** in the top-10 recommendations.

Three exposure metrics are computed over the top-10 lists (added to the modified
`calc.py`):

| Metric | Meaning | Better when |
|--------|---------|-------------|
| `Coverage@10` | fraction of the catalog that appears in any top-10 | higher (broader exposure) |
| `Entropy@10` | entropy of the item-exposure distribution | higher (more evenly spread) |
| `Gini@10` | concentration of exposure among a few items | lower (less concentrated) |

We compare **SFT**, **RA**, and post-**RL** under several decoders (`beam10`,
`sample_t12_p095`, `sample_t10_k50`) across seeds `42 43 44`, on `Video Games` and
`Office Products`. The released post-RL diversity-sweep values are encoded in the table
builder, so the default evaluation only re-runs SFT and RA (`STAGES="sft ra rl"`
regenerates all three). The SASRec baseline values used in the paper are encoded in
`utils/make_rq5_diversity_tables.py`.

### Reproduce

The scripts are part of the SIDReasoner submodule (materialised by
`patches/common/sidreasoner-local.patch`). Run from the submodule root after applying the
patch; environment, data, and base checkpoints are assumed present (see the root README
and `scripts/common/data_prep/`).

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner

# 1. evaluate SFT and RA stages -> per-run JSON + manifest + summary
sbatch final_scripts/rq5_diversity/evaluate_item_side_diversity.sh

# 2. build the paper tables once the JSONs exist
sbatch final_scripts/rq5_diversity/make_diversity_tables.sh
```

Outputs land under `results/item_side_diversity/` (manifest, summary, per-run JSONs, and
the `*_sft_ra_rl_sasrec_comparison.md` tables). Per-script options and the assumed
data/checkpoint layout are documented in
`final_scripts/rq5_diversity/README.md` inside the submodule.

---

## Bottom line

- Item-side exposure-concentration metrics (Coverage/Entropy/Gini@10) across SFT/RA/RL
  and decoders are wired up but not yet run/committed.
- **Practical implication.** Diversity/exposure metrics belong alongside accuracy when
  evaluating generative recommenders — reporting HR/NDCG alone can hide popularity bias.

## Caveats

- Result tables are not yet populated — do not cite numbers until
  `results/item_side_diversity/` is committed.
- `Qwen/Qwen3-1.7B` backbone; `Video Games` and `Office Products` only.

## Reference files

| Area | Path |
|------|------|
| Launch scripts (submodule) | `baselines/repos/SIDReasoner/final_scripts/rq5_diversity/` (via `patches/common/sidreasoner-local.patch`) |
| Jobs / scripts pointers | [`../../jobs/diversity/`](../../jobs/diversity/README.md) · [`../../scripts/diversity/`](../../scripts/diversity/README.md) |
| Outputs (pending) | `results/item_side_diversity/` |
