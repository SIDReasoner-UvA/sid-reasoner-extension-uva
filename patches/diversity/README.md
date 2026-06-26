# diversity — see [docs](../../docs/diversity/README.md)

The RQ5 submodule code is **not** kept as a separate per-suite patch. Because RQ2,
RQ5, and the existing baseline modifications share files (notably `calc.py`,
`snellius_env.sh`, and `sft_reasoning_activation.py`), all SIDReasoner working-tree
changes are tracked in the single common patch
[`../common/sidreasoner-local.patch`](../common/sidreasoner-local.patch), applied by
`scripts/common/apply_submodule_patch.sh`. The RQ5-specific additions captured there
are `final_scripts/rq5_diversity/`, `utils/summarize_item_side_diversity.py`,
`utils/make_rq5_diversity_tables.py`, `utils/download_pretrained_ckpts.py`, and
the exposure-metric changes to `calc.py`.
