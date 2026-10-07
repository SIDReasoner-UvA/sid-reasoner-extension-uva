# Model-size Experiments

> **RQ4: How does backbone scale influence behavior?** We run SFT → RA → GRPO
> at 0.6B and 4B on Games (the original S3 full-mix data setting), comparing the
> trajectories with the original 1.7B reference.

The revised paper emphasizes **consistent training behavior across scales**: R@10
improves and plateaus during GRPO while chain-of-thought length decreases and stabilizes
(Figure 2). The 4B run shows no advantage over 0.6B under these settings. Its noisier,
slightly lower final accuracy does not establish that larger backbones are inherently worse:
these are single-seed runs without scale-specific tuning.

The historical executable path `final_scripts/rq3_model_size/` implements current **RQ4**.
It is retained for compatibility. Detailed stage scores remain in [results_table.md](results_table.md).

---

## Pipeline context

We re-run the **full** S1 SFT → S2 reasoning-activation (RA) → S3 GRPO pipeline at two new
backbone sizes and compare against the paper's single 1.7B model:

| Stage | Name | Purpose |
|-------|------|---------|
| S1 | SFT (raw align) | Teach the model to emit valid Semantic IDs. |
| S2 | RA | Wrap the answer in `<think>…</think>` (1- or 5-epoch ★ — the variant that feeds GRPO is size-dependent). |
| S3 | GRPO | Reinforce reasoning toward correct recommendations. |

Backbones: `Qwen3-0.6B` and `Qwen3-4B` (ours) vs `Qwen3-1.7B` (paper). Data: `Video
Games`, S3 full-mix.

## Experiments

### M1 — Backbone scaling (does a bigger model help?)

Compare the post-GRPO checkpoint across sizes (HR@10, our greedy think-mode constrained
beam-10 eval; see [`results_table.md`](results_table.md) for HR@5/NDCG):

| Backbone | Stage | HR@10 |
|---|---|---|
| 0.6B | SFT (raw align) | 0.0645 |
| 0.6B | **GRPO (step 500)** | **0.0734** |
| 4B | SFT (raw align) | 0.0628 |
| 4B | GRPO (step 500) | 0.0658 |
| 1.7B (paper) | GRPO | 0.1031 |

**Finding.** The 0.6B and 4B runs share the same qualitative GRPO trajectory. The
selected 4B checkpoint has no accuracy advantage (0.0658 versus 0.0734 R@10).
Both trail the original 1.7B reference, but this comparison does not isolate capacity
from retraining, initialization, or tuning differences.

### M2 — Reasoning-activation epoch count (1-ep vs 5-ep)

The RA checkpoint that feeds GRPO uses a size-dependent epoch count. Compare both:

| Backbone | RA 1-epoch HR@10 | RA 5-epoch HR@10 | ★ feeds GRPO |
|---|---|---|---|
| 0.6B | 0.0495 | 0.0501 | 5-epoch |
| 4B | 0.0532 | 0.0316 | 1-epoch |

**Finding.** **RA epoch count is size-dependent.** At 0.6B the two are ~tied (`0.0501` ≈
`0.0495`); at 4B, 5-epoch RA **reduces** performance by about 41% (`0.0316` vs `0.0532`). This motivates a size-dependent epoch choice in these runs; the mechanism is not
isolated. The 4B pipeline branches
GRPO from the **1-epoch** RA checkpoint and 0.6B from the 5-epoch one.

---

## Where the code lives

The launch scripts are part of the **SIDReasoner submodule**, materialised by
`patches/common/sidreasoner-local.patch` (same pattern as `rq2_components` /
`rq5_diversity`). After applying the patch they sit under
`baselines/repos/SIDReasoner/final_scripts/rq3_model_size/`, and the analysis tooling
under the submodule's `utils/`. Run everything from the submodule root with
`$PYTHON_CMD` (the venv) — sourcing `final_scripts/snellius_env.sh` sets it and unsets
`ROCR_VISIBLE_DEVICES` (mandatory for GRPO to launch on Snellius).

| Stage | Script (`final_scripts/rq3_model_size/`) |
|-------|------------------------------------------|
| alignment SFT | `sft_Qwen3_S3_0p6B_Games.sh` / `sft_Qwen3_S3_4B_Games.sh` |
| reasoning activation | `sft_reasoning_activation_S3_0p6B_Games.sh` (5-ep ★) / `..._1ep.sh`; `..._4B_Games.sh` (5-ep) / `..._1ep.sh` (1-ep ★) |
| GRPO | `RL_training_S3_0p6B_Games.sh` / `RL_training_S3_4B_Games.sh` |
| eval | `evaluate_Qwen3_think_S3_0p6B_Games.sh` (+`_ALL`) / `..._4B_Games.sh` |

★ = the reasoning-activation checkpoint that feeds GRPO (0.6B → 5-epoch, 4B → 1-epoch;
RA epoch count is size-dependent — see `results_table.md`).

Key config (passed as CLI flags from the launch scripts): `batch_size 1024`,
`micro_batch_size 8` (effective batch stays 1024 via gradient accumulation),
`group_by_length True`; GRPO `train_batch_size=256`, `ppo_mini_batch_size=256`,
`ppo_micro_batch_size_per_gpu=8`, `rollout.n=16`, `save_freq=100`, `test_freq=50`,
`total_epochs=10`, `n_gpus_per_node=4`, reward `R_sr + λ·R_f` (λ=0.1). The 4B activation
OOMs at `micro_batch_size 16` (by ~4.5 GiB) and is sharded over 4 GPUs.

## Reproduce

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner
source final_scripts/snellius_env.sh   # sets PYTHON_CMD; unsets ROCR_VISIBLE_DEVICES
# one-time: uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" matplotlib huggingface_hub

# Run per size; swap 0p6B <-> 4B. Each stage feeds the next.
sbatch final_scripts/rq3_model_size/sft_Qwen3_S3_0p6B_Games.sh                    # 1. alignment SFT
sbatch final_scripts/rq3_model_size/sft_reasoning_activation_S3_0p6B_Games.sh     # 2. activation (5-ep ★ for 0.6B)
sbatch final_scripts/rq3_model_size/sft_reasoning_activation_S3_0p6B_Games_1ep.sh #    (1-ep variant)
sbatch final_scripts/rq3_model_size/RL_training_S3_0p6B_Games.sh                  # 3. GRPO

# 4. merge verl checkpoints + eval every merged step
bash   scripts/merge_fsdp_ckpt_ALL.sh                                  # global_step_N/actor -> actor_merged
sbatch final_scripts/rq3_model_size/evaluate_Qwen3_think_S3_0p6B_Games_ALL.sh

# 5. collect + plot
$PYTHON_CMD utils/collect_eval_metrics.py --name_filter 0.6B --exclude act_1ep,reasoning_activation \
    --info_file data/Amazon/info/Video_Games_5_2016-10-2018-11.txt \
    --out results/figure5_0p6B.csv
# Repeat collection for the 4B run; provide both training-length CSVs before plotting.
$PYTHON_CMD utils/plot_size_comparison.py \
    --csv_06b results/figure5_0p6B.csv --csv_4b results/figure5_4B.csv \
    --cot_06b results/train_lengths_0p6B.csv --cot_4b results/train_lengths_4B.csv \
    --cot_every 50 --metric HR@10 \
    --out_recall results/recall_0p6B_vs_4B.png --out_cot results/cot_0p6B_vs_4B.png
```

Report the **validation-`mean@1`-selected final step** (step 500), not the test peak
(avoids selection bias). The 4B GRPO run is non-monotonic / possibly under-converged
(test peak `0.0697` @ step 200), so `0.0658` @ step 500 is the defensible number.

Result CSV/PNG (`figure5_*.csv`, `train_lengths_*.csv`, `recall_*.png`, `cot_*.png`)
are generated by steps 4–5 above and land in the submodule's `results/`. Copy the final
artefacts into `results/model_size/` in this repo for the write-up.

### Checkpoint hygiene
Strip optimizer state before moving checkpoints (project quota ~50 GB; a 4B verl
checkpoint is ~47 GB raw): delete `optim_*.pt` / `optimizer.pt` / `scheduler.pt` /
`rng_state*` / `trainer_state.json` — keeps the model mergeable (lean: 0.6B ~1.2 G,
4B ~7.6 G). Upload only merged dirs (`config.json` + safetensors) via
`utils/push_to_hf.py` (edit `CHECKPOINTS` + `REPO_ID`; `HF_TOKEN` env; `--dry-run` first).

---

## Training cost (Appendix D)

| Backbone | Hardware | SFT GPU-hours | RA GPU-hours | GRPO GPU-hours | Total |
|---|---|---|---|---|---|
| 0.6B (ours) | 4 × H100 | 28.5 | 2.7 | 48.3 | 79.5 |
| 4B (ours) | 4 × H100 | 50.0 | 0.3 | 148.0 | 198.3 |
| 1.7B (original authors) | 8 × H100 | 20.0 | 5.0 | 100.0 | 125.0 |

These costs are transcribed from the revised paper, not regenerated from local logs.
RA uses five epochs for 0.6B/1.7B and one for 4B; GPU counts and epoch counts are not matched.
GPU labels follow the author-confirmed H100 setup. Transformers **4.51.3** is the
confirmed environment, matching the committed lockfile and corrected paper.

## Findings

- Stage-wise training behavior persists beyond the original backbone: GRPO improves
  R@10 while CoT length decreases and stabilizes.
- 4B offers no advantage over 0.6B in these runs; single seeds and lack of scale-specific
  tuning limit broader conclusions about capacity.
- RA epoch count and checkpoint selection matter. This differs from the non-RL
  continuation controls, which start from already activated models.

## Caveats

- `Video Games` (S3 full-mix) only; single run per configuration.
- Report the **validation-`mean@1`-selected step 500**, not the test peak, to avoid
  selection bias. The 4B GRPO run is non-monotonic / possibly under-converged (test peak
  `0.0697` @ step 200), so `0.0658` @ step 500 is the defensible number.
- HR@5/NDCG@5 for the 4B GRPO row are not extracted into `results_table.md` (only
  HR@10/NDCG@10 from `figure5_4B.csv`).

## Reference files

| Area | Path |
|------|------|
| Results table + takeaways | [`results_table.md`](results_table.md) |
| Checkpoints (HF) | [`slndaniel/SIDReasoner-extensions`](https://huggingface.co/slndaniel/SIDReasoner-extensions/tree/main) (0.6B / 4B SFT · RA · GRPO) |
| Launch scripts (submodule) | `baselines/repos/SIDReasoner/final_scripts/rq3_model_size/` (via `patches/common/sidreasoner-local.patch`) |
| Analysis tooling (submodule) | `utils/collect_eval_metrics.py`, `utils/plot_size_comparison.py`, `utils/push_to_hf.py` |
| Result artifacts (submodule `results/`) | `figure5_*.csv`, `train_lengths_*.csv`, `recall_*.png`, `cot_*.png` — copy finals into [`../../results/model_size/`](../../results/model_size/) |
| Jobs / scripts pointers | [`../../jobs/model_size/`](../../jobs/model_size/README.md) · [`../../scripts/model_size/`](../../scripts/model_size/README.md) |
