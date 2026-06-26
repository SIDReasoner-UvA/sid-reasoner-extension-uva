# components — see [docs](../../docs/components/README.md)

RQ2 (component-necessity) does not add wrapper-level entry points. Its training and
evaluation scripts are part of the SIDReasoner submodule (they depend on its
`split.py`/`merge.py`/`evaluate_Qwen3_think.py`) and are materialised by
`patches/common/sidreasoner-local.patch` under
`baselines/repos/SIDReasoner/final_scripts/rq2_components/`. The supporting Python
helpers (`utils/summarize_labeled_eval_table.py`, `utils/evaluate_prompt_ablation.py`,
the new `evaluate_Qwen3_strict_no_think.py`, and the modified `calc.py`) live there
too. See [jobs/components](../../jobs/components/README.md) for how to run them.
