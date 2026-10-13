# Item-exposure evaluation

Top-10 exposure metrics for SIDReasoner and SASRec on Games. The scripts also
support Office and the SFT/RA checkpoints.

## Item-side exposure diversity

Three exposure metrics are computed over the top-10 lists (added to the modified
`calc.py`):

| Metric | Meaning | Better when |
|--------|---------|-------------|
| `Coverage@10` | fraction of the catalog that appears in any top-10 | higher (broader exposure) |
| `Entropy@10` | entropy of the item-exposure distribution | higher (more evenly spread) |
| `Gini@10` | concentration of exposure among a few items | lower (less concentrated) |

Historical Games results, mean ± std over seeds 42, 43, and 44.
**The SIDReasoner Gini column uses the old exposed-item definition and awaits
rescoring with the corrected full-catalog metric.**

| Model | Decoding | R@10 ↑ | N@10 ↑ | Coverage@10 ↑ | Entropy@10 ↑ | Gini@10 ↓ |
|-------|----------|--------|--------|----------------|--------------|-----------|
| SASRec | — | 0.092 ± 0.004 | 0.052 ± 0.004 | 0.760 ± 0.123 | 6.505 ± 0.499 | 0.830 ± 0.074 |
| SIDReasoner | `beam10` | 0.102 ± 0.000 | 0.054 ± 0.000 | 0.738 ± 0.000 | 6.855 ± 0.000 | 0.714 ± 0.000 |
| SIDReasoner | `sample_t12_p095` | 0.062 ± 0.001 | 0.035 ± 0.001 | 0.888 ± 0.002 | 7.223 ± 0.002 | 0.659 ± 0.001 |
| SIDReasoner | `sample_t10_k50` | 0.062 ± 0.002 | 0.036 ± 0.001 | 0.888 ± 0.002 | 7.138 ± 0.004 | 0.679 ± 0.001 |

The evaluation launcher now defaults to all stages (`STAGES="sft ra rl"`).
The table builder recomputes every SIDReasoner row from per-seed prediction JSONs,
including RL. Missing predictions fail explicitly; there is no fallback to the old
RL aggregates. SASRec still uses its stored full-catalog baseline aggregates, which
require matching catalog and test-set provenance.

Existing predictions can be rescored without retraining or rerunning generation:
from the patched submodule root, run
`$PYTHON_CMD utils/make_rq5_diversity_tables.py` once the catalog info files and
SFT/RA/RL prediction JSONs for all seeds and decoders are in their expected paths.
This checkout contains no predictions, so corrected paper measurements are pending.

### Reproduce

The scripts are part of the SIDReasoner submodule (materialised by
`patches/common/sidreasoner-local.patch`). Run from the submodule root after applying the
patch; environment, data, and base checkpoints are assumed present (see the root README
and `scripts/common/data_prep/`).

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner

# 1. evaluate SFT, RA, and RL stages -> per-run JSON + manifest + summary
sbatch final_scripts/rq5_diversity/evaluate_item_side_diversity.sh

# 2. build the paper tables once the JSONs exist
sbatch final_scripts/rq5_diversity/make_diversity_tables.sh
```

Outputs land under `results/item_side_diversity/` (manifest, summary, per-run JSONs, and
the `*_sft_ra_rl_sasrec_comparison.md` tables). Per-script options and the assumed
data/checkpoint layout are documented in
`final_scripts/rq5_diversity/README.md` inside the submodule.

## Reference files

| Area | Path |
|------|------|
| Launch scripts (submodule) | `baselines/repos/SIDReasoner/final_scripts/rq5_diversity/` (via `patches/common/sidreasoner-local.patch`) |
| Outputs | `results/item_side_diversity/` |
