# diversity — see [docs](../../docs/diversity/README.md)

The RQ5 (item-side diversity) experiment **Slurm jobs** live in the SIDReasoner
submodule working tree (they call its `split.py`/`merge.py`/`evaluate_Qwen3_think.py`
and the modified `calc.py`) and are materialised by
`patches/common/sidreasoner-local.patch`:

```
baselines/repos/SIDReasoner/final_scripts/rq5_diversity/
  evaluate_item_side_diversity.sh  # SFT/RA decoding sweep -> Coverage/Entropy/Gini
  make_diversity_tables.sh    # build the RQ5 paper tables
```

Run them from the submodule root after applying the patch:

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner
sbatch final_scripts/rq5_diversity/evaluate_item_side_diversity.sh
```

See `final_scripts/rq5_diversity/README.md` (in the submodule) for per-script
options and the assumed data/checkpoint layout.
