# components — see [docs](../../docs/components/README.md)

The RQ2 (component-necessity) experiment **Slurm jobs** live in the SIDReasoner
submodule working tree (they call its `split.py`/`merge.py`/`evaluate_Qwen3_think.py`
directly) and are materialised by `patches/common/sidreasoner-local.patch`:

```
baselines/repos/SIDReasoner/final_scripts/rq2_components/
  train_ra_continuations.sh       # C1: non-RL continuations from RA
  evaluate_nonrl_vs_rl.sh         # C1: RA / extra-SFT / extra-RA / RL table
  evaluate_inference_search.sh    # C2: pre-RL decoding sweep (appendix)
  evaluate_reasoning_ablation.sh  # C3: thinking vs strict no-think
  evaluate_prompt_ablations.sh    # C4: prompt-form ablations (appendix)
```

Run them from the submodule root after applying the patch:

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner
sbatch final_scripts/rq2_components/evaluate_nonrl_vs_rl.sh
```

See `final_scripts/rq2_components/README.md` (in the submodule) for per-script
options and the assumed data/checkpoint layout.
