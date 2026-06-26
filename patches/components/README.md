# components — see [docs](../../docs/components/README.md)

The RQ2 submodule code is **not** kept as a separate per-suite patch. Because RQ2,
RQ5, and the existing baseline modifications share files (notably `calc.py`,
`snellius_env.sh`, and `sft_reasoning_activation.py`), all SIDReasoner working-tree
changes are tracked in the single common patch
[`../common/sidreasoner-local.patch`](../common/sidreasoner-local.patch), applied by
`scripts/common/apply_submodule_patch.sh`. The RQ2-specific additions captured there
are `final_scripts/rq2_components/`, `evaluate_Qwen3_strict_no_think.py`,
`utils/summarize_labeled_eval_table.py`, and `utils/evaluate_prompt_ablation.py`.
