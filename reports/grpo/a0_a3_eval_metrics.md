# GRPO Reward Experiments — A0–A3 Held-out Eval (NDCG / HR, Games)

> Historical run notes. Held-out ranking numbers below have been updated to
> revised paper Table 3; other training diagnostics remain historical. For the revised paper interpretation and confirmed setup
> decisions, see [current RQ3 docs](../../docs/grpo/README.md) and [alignment audit](../../docs/paper-alignment.md).
> Earlier claims that reward shape is universally immaterial are superseded: the
> revised result distinguishes top-10 stability from rank-1 improvements. Monotone
> reward transformations do not generally cancel under GRPO normalization.


**Date:** 2026-06-14
**Backbone:** `Qwen/Qwen3-1.7B` · **Stage:** S3 (GRPO RL) from RA checkpoint `sidreasoner_ra_games`
**What this is:** the **think-mode ranking evaluation** the A0/A1, A2, and A3 reports each flagged
as their "next step." Every prior report only had the in-training reward proxy
(`val-core/rec/Video_Games/reward/mean@7`); this is the real `scripts/common/eval/evaluate.py`
(`run_eval_think`) pass on each merged S3 checkpoint, producing NDCG@k / HR@k.

> Companion to [`a0_a1_games.md`](a0_a1_games.md),
> [`a2_lambda_sweep_games.md`](a2_lambda_sweep_games.md), and
> [`a3_linear_acc_games.md`](a3_linear_acc_games.md).

---

## 1. Setup

- **Job:** `jobs/grpo/reward_eval.job` — SLURM array (`--array=0-5`), one H100 per
  checkpoint, run in parallel. Slurm job id **23816638**.
- **Eval engine:** `evaluate_Qwen3_think.py` (vLLM, thinking tokens, temperature 0,
  `max_new_tokens=256`, `tensor_parallel_size=1`), constrained to valid catalogue SID paths.
- **Test set:** full Games test split (`Video_Games_5_2016-10-2018-11.csv`, ~6.1k examples),
  `sample=-1` (no subsampling).
- **Checkpoints:** for each S3 run the latest verl FSDP actor (`global_step_100/actor`) was merged
  into a single HF model (`actor_merged`) and evaluated.
- **Beams:** `n_beam=10`, so the reported cutoffs are @1, @3, @5, @10.
- **Wall time:** ~22.5 min per checkpoint (all six in parallel).

> Why GPU and not the `rome` CPU partition: the eval loads the model with vLLM and does batched
> autoregressive generation over the full test set — a CUDA workload, not a CPU one.

**A2 λ=0.2 is absent** because it was never trained (dry-run only; no checkpoint) — see the A2 report.

---

## 2. Results — NDCG@k

| Experiment | Reward | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| **A0** baseline      | `R_sr + 0.1·R_f`            | 0.0216 | 0.0380 | 0.0438 | 0.0543 |
| **A1** no-format     | `R_sr` (λ=0)                 | 0.0308 | 0.0428 | **0.0486** | **0.0588** |
| **A2** λ=0.05        | `+0.05·R_f`                 | 0.0213 | 0.0376 | 0.0439 | 0.0544 |
| **A2** λ=0.5         | `+0.5·R_f`                  | 0.0226 | 0.0379 | 0.0438 | 0.0537 |
| **A2** λ=1.0         | `+1.0·R_f`                  | 0.0252 | 0.0399 | 0.0464 | 0.0564 |
| **A3** linear acc    | linear `R_sr + 0.1·R_f`    | 0.0290 | 0.0403 | 0.0470 | 0.0571 |

## 3. Results — HR@k

| Experiment | Reward | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---|---|---|---|
| **A0** baseline      | `R_sr + 0.1·R_f`            | 0.0216 | 0.0506 | 0.0648 | 0.0977 |
| **A1** no-format     | `R_sr` (λ=0)                 | 0.0308 | 0.0516 | 0.0658 | 0.0972 |
| **A2** λ=0.05        | `+0.05·R_f`                 | 0.0213 | 0.0506 | 0.0659 | **0.0987** |
| **A2** λ=0.5         | `+0.5·R_f`                  | 0.0226 | 0.0488 | 0.0632 | 0.0941 |
| **A2** λ=1.0         | `+1.0·R_f`                  | 0.0252 | 0.0508 | **0.0667** | 0.0978 |
| **A3** linear acc    | linear `R_sr + 0.1·R_f`    | 0.0290 | 0.0487 | 0.0650 | 0.0965 |

---

## 4. Interpretation

**All six variants land in a narrow band.** NDCG@10 ranges 0.0537–0.0588 (spread ~0.005) and HR@10
ranges 0.0941–0.0987 (spread ~0.005). The held-out ranking metrics confirm the central finding of
the whole sweep: **on Games the choice of reward shape and the format weight λ barely moves final
quality.** The accuracy ceiling is set by the data/init, not by the format term — exactly what the
GRPO group-relative advantage argument predicts (a near-constant `λ·R_f` cancels in
`Â_i = (R_i − mean(R)) / std(R)`).

**The proxy and the held-out metric disagree on the ranking of variants.** The in-training reward
proxy suggested A1 (λ=0) and A3 (linear) were *less stable* — both "regressed" to their step-0 level
by step 100, while A0's steep hierarchical reward "held" its peak. On the real held-out NDCG/HR the
picture flips:

- **A1 (λ=0) is the best on NDCG** (0.0486 @5, 0.0588 @10) and essentially tied-best on HR.
- **A3 (linear) is second on NDCG** (0.0470 @5, 0.0571 @10).
- **A0 (baseline) is at the bottom of the NDCG band** (0.0438 @5, 0.0543 @10), despite being the
  "stable" one on the proxy.

The lesson is the one the reports already cautioned about: the `mean@7` reward proxy (100 steps, 3
validation points, includes the λ offset) is **noisy and not a faithful stand-in for NDCG/HR**.
Trajectory "stability" on the proxy did not translate into better held-out ranking — if anything the
opposite here. The differences are small enough (≤0.005 NDCG@10) that they may not be significant at
this scale, so the safest statement remains: **no variant clearly beats the others on Games.**

**λ magnitude (A2) shows no monotonic trend.** Across λ ∈ {0.05, 0.5, 1.0} NDCG@10 goes
0.0544 → 0.0537 → 0.0564 — non-monotonic noise around the baseline, consistent with "λ is a
constant offset that cancels in the advantage; its magnitude is immaterial."

---

## 5. Bottom line

The think-mode NDCG/HR eval **validates the sweep's qualitative conclusion** — reward-shape and λ
choices are immaterial to final ranking quality on Games — while **correcting the variant ordering**
the noisy reward proxy implied: on held-out metrics A1 (λ=0) and A3 (linear) edge out the A0
baseline rather than under-performing it. Given the ~0.005 spread, treat all six as effectively
equivalent on Games; the proxy's "stability" signal should not be read as a quality signal.

**Caveats:** Games only, single seed, 100 training steps each, and the absolute scores are low
(NDCG@10 ≈ 0.05–0.06) as expected for this task/backbone. Confirming on Office would strengthen the
"reward shape is immaterial" claim.

---

### Source

- Job: `jobs/grpo/reward_eval.job` (Slurm array **23816638**, all tasks COMPLETED, ~22.5 min each on H100).
- Per-experiment logs: `jobs/outputs/grpo-reward-eval-23816638_{0..5}.out`.
- Predictions: `results/grpo/<exp>_games/eval_predictions.json`.

| idx | Experiment | Checkpoint |
|---|---|---|
| 0 | A0 baseline   | `a0_baseline_games/stage3_rl/global_step_100` |
| 1 | A1 no-format  | `a1_no_format_games/stage3_rl/global_step_100` |
| 2 | A2 λ=0.05     | `a2_l005_games/stage3_rl/global_step_100` |
| 3 | A2 λ=0.5      | `a2_l05_games/stage3_rl/global_step_100` |
| 4 | A2 λ=1.0      | `a2_l10_games/stage3_rl/global_step_100` |
| 5 | A3 linear acc | `a3_linear_acc_games/stage3_rl/global_step_100` |
