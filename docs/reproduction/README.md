# Reproduction Experiments

> **RQ1: Can released artifacts recover the reported results, and can partial pre-RL
> retraining recover the early-stage trends?**

This suite combines evaluation of released checkpoints across three datasets with
independent SFT → RA retraining at Qwen3-1.7B on Games. We do not claim full independent
1.7B GRPO reproduction.

## R1 — Released-checkpoint replication

Revised Table 1 reports the original SIDReasoner score (Paper), our evaluation (Ckpt),
relative change, and two-sided bootstrap p-value:

| Dataset | Metric | Paper | Ckpt | Δ% | p |
|---|---|---|---|---|---|
| Games | R@5 | 0.0710 | 0.0689 | -3.0 | .52 |
| Games | N@5 | 0.0460 | 0.0438 | -4.8 | .32 |
| Games | R@10 | 0.1031 | 0.1024 | -0.7 | .85 |
| Games | N@10 | 0.0563 | 0.0546 | -3.0 | .46 |
| Office | R@5 | 0.1373 | 0.1365 | -0.6 | .87 |
| Office | N@5 | 0.1119 | 0.1106 | -1.2 | .75 |
| Office | R@10 | 0.1648 | 0.1661 | +0.8 | .81 |
| Office | N@10 | 0.1208 | 0.1202 | -0.5 | .89 |
| Industrial | R@5 | 0.1109 | 0.1092 | -1.5 | .72 |
| Industrial | N@5 | 0.0905 | 0.0891 | -1.6 | .72 |
| Industrial | R@10 | 0.1438 | 0.1474 | +2.5 | .50 |
| Industrial | N@10 | 0.1010 | 0.1014 | +0.4 | .92 |

Appendix A resamples test users with replacement 10,000 times at the original test-set
size and computes `p = 2 min(P(metric* <= paper), P(metric* >= paper))`. All original
scores fall inside the central 95% bootstrap intervals. These are unpaired comparisons
against published scalar scores, not paired comparisons of model predictions.
The table is transcribed from the paper; the bootstrap runner and per-user replication
artifacts are not included in this checkout, so the p-values have not been recomputed here.

**Finding.** Released-artifact scores reproduce within ±5%, with no significant bootstrap
differences. This verifies artifact consistency, not the full documented training procedure.

## R2 — Independent early-stage retraining (Games)

Retrain Qwen3-1.7B on the original S1 multi-task alignment data ablation. Here **S1 names
a data setting**, not all three training stages. Evaluation uses greedy reasoning followed
by constrained beam-10 SID decoding; HR@K is Recall@K in this single-target protocol.

| Stage | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|
| SFT (raw align) | 0.0215 | 0.0387 | 0.0129 | 0.0184 |
| RA 5-epoch | 0.0321 | 0.0602 | 0.0187 | 0.0277 |

**Finding.** RA improves R@10 from 0.0387 to 0.0602, reproducing the qualitative trend.
The original **pre-RL / RA** reference is 0.0414; it is not a separate published SFT
score. The original text specifies one RA epoch while the released code defaults to
five. The revised paper attributes the absolute-score discrepancy to this mismatch.
Cold-start RA also needs final-checkpoint selection instead of lowest SID validation loss.

## Where the code lives

The launch scripts are part of the **SIDReasoner submodule**, materialised by
`patches/common/sidreasoner-local.patch` (same pattern as `rq2_components` /
`rq5_diversity`). After applying the patch they sit under
`baselines/repos/SIDReasoner/final_scripts/rq1_reproduction/`. Run from the submodule
root with `$PYTHON_CMD`; source `final_scripts/snellius_env.sh` first.

| Stage | Script (`final_scripts/rq1_reproduction/`) |
|-------|--------------------------------------------|
| 1.7B S1 alignment SFT | `sft_Qwen3_S1_Games.sh` |
| 1.7B S1 reasoning activation (5-ep) | `sft_reasoning_activation_S1_Games.sh` |
| think-mode eval | `evaluate_Qwen3_think_S1.sh` (+ `_bestof1` for beam-1) |

> The reasoning-activation `eval_loss` **rises on purpose** during this stage (it is
> the direct-SID loss, which worsens as the model learns to think-then-answer). The
> shared `sft_reasoning_activation.py` defaults `load_best_model_at_end=False` so the
> activated final epoch is the checkpoint that is saved — do not "fix" the rising loss.

## Reproduce

```bash
bash scripts/common/apply_submodule_patch.sh
cd baselines/repos/SIDReasoner
source final_scripts/snellius_env.sh

sbatch final_scripts/rq1_reproduction/sft_Qwen3_S1_Games.sh                 # SFT  -> HR@10 0.0387
sbatch final_scripts/rq1_reproduction/sft_reasoning_activation_S1_Games.sh  # RA(5-ep) -> HR@10 0.0602
sbatch final_scripts/rq1_reproduction/evaluate_Qwen3_think_S1.sh            # standard greedy think eval
```

Use the **standard greedy think eval** (`evaluate_Qwen3_think_S1.sh`); the temperature
best-of-1 variant was only patching the previously-broken (1-epoch) model and is no
longer needed (see `S1_status.md`).

> **Environment:** Transformers **4.51.3**, as confirmed by the authors and recorded
> in the committed lockfile. The dependency constraint remains `>=4.51.0,<4.52`;
> the paper has been corrected to match this environment.

---

## Scope and reference files

- Independent early-stage retraining is Games-only and single-run. Historical Office
  debugging results in [S1_status.md](S1_status.md) are not the final reproduction result.
- Full stage-wise measurements are retained in [results_table.md](../model_size/results_table.md).
- Launch scripts live in `baselines/repos/SIDReasoner/final_scripts/rq1_reproduction/`,
  materialized by `patches/common/sidreasoner-local.patch`.
- See [jobs](../../jobs/reproduction/README.md), [scripts](../../scripts/reproduction/README.md),
  and the [alignment audit](../paper-alignment.md) for remaining artifact gaps.
