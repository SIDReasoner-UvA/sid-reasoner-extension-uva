# GRPO Reward Experiments — A0 vs A1 (Games)

**Date:** 2026-06-13
**Backbone:** `Qwen/Qwen3-1.7B` · **Stage:** S3 (GRPO RL) from RA checkpoint `sidreasoner_ra_games`
**Variants:** A0 (baseline, `R = R_sr + 0.1·R_f`) vs A1 (no format term, `R = R_sr`)

> See [`docs/grpo/README.md`](../../docs/grpo/README.md) for the full experiment design. A1 tests:
> *"Is the format reward actually helping, or does GRPO+KL already keep outputs valid?"*

---

## ⚠️ Caveat on what was measured

The pipeline only logged **reward-based validation** (`val-core/rec/Video_Games/reward/mean@7`),
**not NDCG@k / HR@k**. Those ranking metrics require a separate `scripts/common/eval/evaluate.py`
(`run_eval_think`) pass on the merged S3 checkpoint, which has **not** been run yet.

Everything below is the in-training reward proxy. The **absolute** A0 and A1 numbers are **not
directly comparable** because A0's reward includes the +0.1 format bonus and A1's does not.

---

## 1. Run status

| | A0 (baseline) | A1 (no format) |
|---|---|---|
| Slurm job | 23722129 | 23722130 |
| Reward formula | `R_sr + 0.1·R_f` | `R_sr` (λ = 0) |
| Steps | 100 (full run) | 100 (full run) |
| Wall time | 31,534 s (~8.8 h) | 31,487 s (~8.7 h) |
| Final checkpoint | `results/grpo/a0_baseline_games/stage3_rl` | `results/grpo/a1_no_format_games/stage3_rl` |

Both runs started from the same RA checkpoint, 4 GPUs, identical step-0 state.

---

## 2. Training reward (mean per step)

| | step 1 | step 50 | step 100 | Δ |
|---|---|---|---|---|
| A0 | 0.126 | 0.134 | 0.147 | +0.021 |
| A1 | 0.033 | 0.043 | 0.056 | +0.023 |

Both improve at essentially the same rate on the train set (+~0.02). A0 sits ~0.10 higher
in absolute terms purely because of the constant format bonus.

## 3. Validation reward (`mean@7`, logged at steps 0 / 50 / 100)

| | step 0 | step 50 | step 100 |
|---|---|---|---|
| A0 | 0.1446 | 0.1714 | 0.1714 |
| A1 | 0.0446 | 0.0714 | 0.0446 |

## 4. Secondary signals

| | A0 | A1 |
|---|---|---|
| KL-to-reference (step 100) | 0.0028 | 0.0036 |
| Response length (step 1 → 100) | 201 → 199 tok | 201 → 198 tok |

KL stayed tiny in both; A1 drifted slightly further (no constant offset → marginally more
aggressive updates). No reasoning-length inflation in either run.

---

## 5. Interpretation

**The format reward is a near-constant offset, not a learning signal.**
At step 0 the two runs share the *same* model, so the gap is purely the format bonus:

```
A0 − A1 = 0.1446 − 0.0446 ≈ 0.10 = 0.1·R_f   ⟹   R_f ≈ 1.0
```

The RA checkpoint already emits valid catalogue SIDs ~100 % of the time, so the format term
adds a flat +0.1 with essentially no discriminative gradient — exactly what A1 set out to test.

**Accuracy, format offset removed** (A0's `R_sr ≈ val − 0.10`, assuming format rate ≈ 1):

| R_sr (val) | step 0 | step 50 | step 100 |
|---|---|---|---|
| A0 | ~0.045 | ~0.071 | ~0.071 |
| A1 | 0.045 | 0.071 | 0.045 |

Both reach the same accuracy peak (~0.071) by step 50. The difference: **A0 holds it; A1
regresses to baseline by step 100.** So the format term did not raise the accuracy ceiling, but
appears to have acted as a mild **stabilizer** (A1's validation was noisier and ended where it started).

---

## 6. Bottom line

A1 supports its hypothesis: GRPO + KL keeps outputs valid on its own (validity was already
saturated at init), so the format reward contributes almost no learning signal — but removing
it left validation slightly less stable.

**Caveats:** 100 steps with only 3 validation points is noisy, and these are reward proxies,
not the NDCG@k / HR@k metrics the protocol actually calls for.

## 7. Held-out NDCG / HR eval (done)

The think-mode eval has now been run on both merged S3 checkpoints (Slurm job 23816638; see
[`a0_a3_eval_metrics.md`](a0_a3_eval_metrics.md) for the full A0–A3 table and method).
Games full test set, `n_beam=10`:

| | NDCG@5 | NDCG@10 | HR@5 | HR@10 |
|---|---|---|---|---|
| A0 (baseline) | 0.04378 | 0.05431 | 0.06480 | 0.09769 |
| A1 (no format) | **0.04865** | **0.05879** | **0.06578** | 0.09720 |

**The held-out metric reverses the proxy's "stability" story.** On the noisy `mean@7` proxy A1
looked *less* stable (it regressed to step-0 by step 100 while A0 held). On real NDCG/HR, **A1 (λ=0)
actually edges out A0** on every cutoff except HR@10 (where they tie at ~0.097). The gap is small
(~0.004 NDCG@10) and likely within noise, but it confirms two things: (i) removing the format term
did **not** hurt held-out quality, and (ii) proxy "stability" was not a quality signal here. This
strengthens A1's hypothesis — GRPO+KL keeps outputs valid on its own, and the format reward
contributes no useful learning signal on Games.

---

### Source

Metrics extracted from the Slurm logs:
- A0: `jobs/outputs/grpo-reward-sweep-23722129.out`
- A1: `jobs/outputs/grpo-reward-sweep-23722130.out`
