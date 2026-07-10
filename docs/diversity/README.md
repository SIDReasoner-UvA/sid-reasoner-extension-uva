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
| **Item-side diversity** | Is exposure in the top-10 concentrated on a few items? | complete: beam search gives the best accuracy-diversity balance; stochastic decoding broadens exposure but loses accuracy |

---

## Pipeline context

SIDReasoner trains in three stages; this study branches from the released checkpoints
rather than retraining:

| Stage | Name | Purpose | Checkpoint |
|-------|------|---------|------------|
| S1 | SFT | Emit valid Semantic IDs (SIDs). | `sidreasoner_sft_*` |
| S2 | Reasoning-activation (RA) | Wrap the answer in `<think>…</think>` before the SID. | `sidreasoner_ra_*` |
| S3 | GRPO RL | Reinforce reasoning toward correct recommendations. | post-GRPO |

The scripts can evaluate exposure across **all three** stages (SFT / RA / RL).
The delivered RQ5 result reported in the paper focuses on post-RL SIDReasoner
under different decoders, compared with a SASRec baseline.

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

For the delivered RQ5 table, we compare **SASRec** with post-**RL**
SIDReasoner under three decoders (`beam10`, `sample_t12_p095`,
`sample_t10_k50`) on **Video Games**. Values are mean ± std over seeds
`42 43 44`.

| Model | Decoding | R@10 ↑ | N@10 ↑ | Coverage@10 ↑ | Entropy@10 ↑ | Gini@10 ↓ |
|-------|----------|--------|--------|----------------|--------------|-----------|
| SASRec | — | 0.092 ± 0.004 | 0.052 ± 0.004 | 0.760 ± 0.123 | 6.505 ± 0.499 | 0.830 ± 0.074 |
| SIDReasoner | `beam10` | 0.102 ± 0.000 | 0.054 ± 0.000 | 0.738 ± 0.000 | 6.855 ± 0.000 | 0.714 ± 0.000 |
| SIDReasoner | `sample_t12_p095` | 0.062 ± 0.001 | 0.035 ± 0.001 | 0.888 ± 0.002 | 7.223 ± 0.002 | 0.659 ± 0.001 |
| SIDReasoner | `sample_t10_k50` | 0.062 ± 0.002 | 0.036 ± 0.001 | 0.888 ± 0.002 | 7.138 ± 0.004 | 0.679 ± 0.001 |

Compared with SASRec, SIDReasoner with `beam10` improves R@10 by roughly
11% and N@10 by roughly 4%, while reducing exposure concentration: Gini@10 is
about 14% lower and Entropy@10 about 5% higher. Coverage@10 is slightly lower
(about 3%), so beam search spreads exposure more evenly across a marginally
smaller item set.

Stochastic decoding moves in the opposite direction: it increases Coverage@10
by about 20% relative to `beam10` and further lowers Gini@10, but R@10 drops
from 0.102 to 0.062 (about a 39% relative loss). This makes sampling useful as
an exposure-broadening control, but too costly as the default recommendation
decoder.

The broader evaluation scripts can also re-run SFT/RA and optionally regenerate
all stages (`STAGES="sft ra rl"`), including the Office Products configuration.
The released post-RL diversity-sweep values and SASRec baselines used for the
paper table are encoded in `utils/make_rq5_diversity_tables.py`.

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

- **Beam search is the best default.** On Video Games, post-RL SIDReasoner with
  `beam10` improves accuracy over SASRec while also lowering exposure
  concentration.
- **Sampling exposes more of the catalog, but the accuracy cost is severe.**
  Both sampling decoders increase Coverage@10 and reduce Gini@10 relative to
  `beam10`, but lose about 39% relative R@10.
- **Practical implication.** Diversity/exposure metrics belong alongside accuracy when
  evaluating generative recommenders: decoding is an inference-time control knob
  for the accuracy-diversity trade-off, and HR/NDCG alone can hide exposure bias.

## Caveats

- The delivered RQ5 result table is for **Video Games**. The scripts also contain
  support for Office Products, but the report's main RQ5 table does not use it.
- Diversity is measured as item-side top-10 exposure only; it does not measure
  user-side novelty, serendipity, or calibration.
- `Qwen/Qwen3-1.7B` backbone; seeds `42 43 44`.

## Reference files

| Area | Path |
|------|------|
| Launch scripts (submodule) | `baselines/repos/SIDReasoner/final_scripts/rq5_diversity/` (via `patches/common/sidreasoner-local.patch`) |
| Jobs / scripts pointers | [`../../jobs/diversity/`](../../jobs/diversity/README.md) · [`../../scripts/diversity/`](../../scripts/diversity/README.md) |
| Outputs | `results/item_side_diversity/` |
