# diversity — see [docs](../../docs/diversity/README.md)

RQ5 (item-side diversity) does not add wrapper-level entry points. Its evaluation and
table-building scripts are part of the SIDReasoner submodule (they depend on its
`split.py`/`merge.py`/`evaluate_Qwen3_think.py` and the modified `calc.py` that
reports exposure metrics) and are materialised by
`patches/common/sidreasoner-local.patch` under
`baselines/repos/SIDReasoner/final_scripts/rq5_diversity/`. The supporting Python
helpers (`utils/summarize_item_side_diversity.py`,
`utils/make_rq5_diversity_tables.py`, `utils/download_pretrained_ckpts.py`) live
there too. See [jobs/diversity](../../jobs/diversity/README.md) for how to run them.
