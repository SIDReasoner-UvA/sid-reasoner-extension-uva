# GRPO Reward Experiments — A3 Linear Accuracy (Games)

> Historical run notes. Held-out ranking numbers below have been updated to
> revised paper Table 3; other training diagnostics remain historical. For the revised paper interpretation and confirmed setup
> decisions, see [current RQ3 docs](../../docs/grpo/README.md) and [alignment audit](../../docs/paper-alignment.md).
> Earlier claims that reward shape is universally immaterial are superseded: the
> revised result distinguishes top-10 stability from rank-1 improvements. Monotone
> reward transformations do not generally cancel under GRPO normalization.


**Date:** 2026-06-14
**Backbone:** `Qwen/Qwen3-1.7B` · **Stage:** S3 (GRPO RL) from RA checkpoint `sidreasoner_ra_games`
**Variant:** A3 — linear / partial-credit accuracy, `R = R_sr + 0.1·R_f` with
`R_sr = 0.33·[a1=g1] + 0.33·[a1=g1 & a2=g2] + 0.34·[all match]` (additive, no ×2 doubling)

> See [`docs/grpo/README.md`](../../docs/grpo/README.md) for the full experiment design. A3 tests:
> *"Does softer, more linear accuracy shaping beat the steep ×2 hierarchical doubling?"*
> Compared against the A0 baseline in [`a0_a1_games.md`](a0_a1_games.md) (same `+0.1·R_f`
> offset, steep hierarchical `R_sr`).

---

## ⚠️ Caveats on what was measured

1. The pipeline only logged **reward-based validation** (`val-core/rec/Video_Games/reward/mean@7`),
   **not NDCG@k / HR@k**. Those ranking metrics require a separate `scripts/common/eval/evaluate.py`
   (`run_eval_think`) pass on the merged S3 checkpoint, which has **not** been run yet.
2. A3 keeps the same `+0.1·R_f` offset as A0, so the **absolute validation `mean@7` is directly
   comparable to A0**. However the offset-removed accuracy is **not** absolute-comparable across the
   two: A0's `R_sr` is the steep hierarchical (×2 doubling) function while A3's is linear additive
   partial credit, so the two assign different reward to the *same* prediction. Only the **trajectory
   shape** (does it climb, peak, hold, or regress?) compares meaningfully.

---

## 1. Run status

| | A3 (linear acc) | A0 (baseline, ref) |
|---|---|---|
| Slurm job | 23782814 | 23722129 |
| Reward formula | `R_sr(linear) + 0.1·R_f` | `R_sr(hierarchical) + 0.1·R_f` |
| Steps | **100** (used) | 100 (full run) |
| Wall time to step 100 | ~31,461 s (~8.7 h) | 31,534 s (~8.8 h) |
| Final checkpoint | `…/a3_linear_acc_games/stage3_rl/global_step_100` | `…/a0_baseline_games/stage3_rl` |

> **Note on the run:** this variant was launched misconfigured for **382 steps** (epoch override
> `total_epochs=2`, 191 steps/epoch) rather than the 100-step cap used by every other run in the
> sweep. It was **cancelled at step ~170** once it was clear it would otherwise exceed its 24 h wall
> limit. By then `test_freq=50` / `save_freq=100` had already logged the step-0/50/100 validation
> points and saved the `global_step_100` checkpoint, so the 100-step result below is complete and
> equivalent to the other runs. Steps 101→170 were surplus and are unused.
>
> The step-100 checkpoint lives under `global_step_100/` (canonicalised by
> `latest_checkpointed_iteration.txt = 100`), so downstream eval should target that path.

Started from the same RA checkpoint, 4 GPUs, identical step-0 state as the rest of the sweep.

---

## 2. Training reward (mean per step)

| | step 1 | step 50 | step 100 | Δ |
|---|---|---|---|---|
| A3 | 0.132 | 0.142 | 0.154 | +0.022 |
| A0 | 0.126 | 0.134 | 0.147 | +0.021 |

Both climb at the same rate (+~0.02). A3 sits slightly higher in absolute terms because linear
partial credit hands out more reward for partial SID matches.

## 3. Validation reward (`mean@7`, logged at steps 0 / 50 / 100)

Both carry the same `+0.1·R_f` offset, so these are directly comparable.

| | step 0 | step 50 | step 100 |
|---|---|---|---|
| A3 | 0.1589 | 0.1829 | **0.1589** ← regresses |
| A0 | 0.1446 | 0.1714 | 0.1714 |

A3 starts ~0.014 higher (partial-credit shaping rewards near-misses at init) and peaks at the same
step 50, but **falls back to its step-0 level by step 100** — the same regression pattern seen in
A1 (λ=0). A0's steep hierarchical reward, by contrast, **held** its step-50 peak.

## 4. Secondary signals

| | A3 | A0 |
|---|---|---|
| KL-to-reference (step 100) | 0.0035 | 0.0028 |
| Entropy (step 100) | 1.464 | — |
| Response length (step 1 → 100) | 201 → 197 tok | 201 → 199 tok |

KL stayed tiny; entropy flat (~1.46). No reasoning-length inflation.

---

## 5. Interpretation

The hypothesis was that softer, more uniform partial credit (reducing reward sparsity) would learn
better than the steep ×2 doubling. On the validation proxy, **it did not.**

- A3 reaches the **same step-50 peak** as A0 and then **regresses by step 100**, mirroring the
  unstable λ=0 (A1) behaviour rather than the stable A0 baseline.
- The denser partial-credit signal does raise the *absolute* reward (more near-misses scored), but
  the GRPO group-relative advantage `Â_i = (R_i − mean(R)) / std(R)` is scale-invariant, so a softer
  reward landscape gives no learning advantage — and here appears to be **less stable** at the
  step-100 mark.

This is consistent with the broader sweep finding: on Games, the accuracy ceiling (~step-50 peak) is
set by the data/init, and reward-shape tweaks mostly affect **stability at that peak**, not the peak
itself. The steep hierarchical shaping held the peak; the linear one did not.

---

## 6. Bottom line

**Linear/partial-credit accuracy shaping does not beat the steep ×2 hierarchical doubling on Games.**
Same peak by step 50, but A3 regresses to baseline by step 100 (like λ=0), whereas the hierarchical
A0 reward held. No reason to prefer the linear variant on this task.

**Caveats:** 100 steps with only 3 validation points is noisy, these are reward proxies (not
NDCG@k / HR@k), and the offset-removed accuracies of A3 vs A0 are not absolute-comparable (different
accuracy functions) — only the trajectory shape is.

## 7. Held-out NDCG / HR eval (done)

The think-mode eval has now been run on the merged step-100 checkpoint (Slurm job 23816638; see
[`a0_a3_eval_metrics.md`](a0_a3_eval_metrics.md) for the full A0–A3 table and method).
Games full test set, `n_beam=10`:

| | NDCG@5 | NDCG@10 | HR@5 | HR@10 |
|---|---|---|---|---|
| A3 (linear acc) | **0.0470** | **0.0571** | 0.0650 | 0.0965 |
| A0 (baseline)   | 0.0438 | 0.0543 | 0.0648 | 0.0977 |

**The held-out metric flips the §5 conclusion.** On the noisy `mean@7` proxy A3 looked *worse* than
A0 — it regressed to step-0 by step 100 while the hierarchical A0 reward held. On real NDCG/HR,
**A3 (linear) actually edges out A0** on NDCG@5/@10 and HR@5 (A0 is only ahead on HR@10). The gap is
small (~0.003 NDCG@10) and plausibly within noise, but it means the original "linear does not beat
hierarchical" headline is **not supported by the ranking metrics** — if anything the linear variant
is marginally ahead. The robust statement is the sweep-wide one: on Games the two shaping functions
are effectively equivalent on held-out quality, and the proxy's trajectory "stability" was not a
reliable quality signal.

---

### Source

Metrics extracted from the Slurm log:
- A3: `jobs/outputs/grpo-reward-sweep-23782814.out`
- A0 (comparison): see [`a0_a1_games.md`](a0_a1_games.md)
