# GRPO Reward Experiments — A2 λ Sweep (Games)

> Historical run notes. Held-out ranking numbers below have been updated to
> revised paper Table 3; other training diagnostics remain historical. For the revised paper interpretation and confirmed setup
> decisions, see [current RQ3 docs](../../docs/grpo/README.md) and [alignment audit](../../docs/paper-alignment.md).
> Earlier claims that reward shape is universally immaterial are superseded: the
> revised result distinguishes top-10 stability from rank-1 improvements. Monotone
> reward transformations do not generally cancel under GRPO normalization.


**Date:** 2026-06-13
**Backbone:** `Qwen/Qwen3-1.7B` · **Stage:** S3 (GRPO RL) from RA checkpoint `sidreasoner_ra_games`
**Variant:** A2 — sweep the format-reward weight λ in `R = R_sr + λ·R_f`

> See [`docs/grpo/README.md`](../../docs/grpo/README.md) for the full experiment design. A2 tests:
> *"Trade-off between validity pressure and accuracy."*
> This report builds on [`a0_a1_games.md`](a0_a1_games.md) (λ=0.1 baseline and λ=0).

---

## ⚠️ Caveat on what was measured

The pipeline only logged **reward-based validation** (`val-core/rec/Video_Games/reward/mean@7`),
**not NDCG@k / HR@k**. Those ranking metrics require a separate `scripts/common/eval/evaluate.py`
(`run_eval_think`) pass on the merged S3 checkpoint, which has **not** been run yet.

Everything below is the in-training reward proxy. The **absolute** numbers across λ values are
**not directly comparable** because each includes a different `λ·R_f` offset. The
offset-removed accuracy (`R_sr ≈ val − λ`, since the format rate is ≈ 1.0) is the comparable
quantity.

---

## 1. Run status

| | A2 λ=0.05 | A2 λ=0.2 | A2 λ=0.5 | A2 λ=1.0 |
|---|---|---|---|---|
| Config | `a2_lambda_sweep/games_l005.yaml` | `games_l02.yaml` | `games_l05.yaml` | `games_l10.yaml` |
| Slurm job | 23760697 | — | 23760698 | 23760699 |
| Steps | 100 (full run) | **not run** | 100 (full run) | 100 (full run) |
| Wall time (s3) | 31,289 s (~8.7 h) | — | 31,475 s (~8.7 h) | 31,562 s (~8.8 h) |
| Final checkpoint | `…/a2_l005_games/stage3_rl` | — | `…/a2_l05_games/stage3_rl` | `…/a2_l10_games/stage3_rl` |

**λ=0.2 (`a2_l02_games`) was never trained** — its `pipeline_summary.json` is a dry run
(`dry_run: true`, `s3: 0.0`, no `stage3_rl` checkpoint). The λ=0.1 point is the A0 baseline
and λ=0 is A1 (both from the previous report).

All real runs started from the same RA checkpoint, 4 GPUs, identical step-0 state.

---

## 2. Validation reward (`mean@7`, logged at steps 0 / 50 / 100)

Includes the λ·R_f offset, so larger λ ⇒ larger absolute number by construction.

| λ | step 0 | step 50 | step 100 |
|---|---|---|---|
| 0.00 (A1) | 0.0446 | 0.0714 | 0.0446 |
| 0.05 | 0.0946 | 0.1214 | 0.1214 |
| 0.10 (A0) | 0.1446 | 0.1714 | 0.1714 |
| 0.50 | 0.5446 | 0.5714 | 0.5714 |
| 1.00 | 1.0446 | 1.0714 | 1.0714 |

## 3. Accuracy with format offset removed (`R_sr ≈ val − λ`)

Assuming format rate ≈ 1.0 (confirmed in the A0/A1 analysis: the RA checkpoint emits valid
catalogue SIDs ~100 % of the time).

| λ | step 0 | step 50 | step 100 |
|---|---|---|---|
| 0.00 (A1) | 0.045 | 0.071 | **0.045** ← regresses |
| 0.05 | 0.045 | 0.071 | 0.071 |
| 0.10 (A0) | 0.045 | 0.071 | 0.071 |
| 0.50 | 0.045 | 0.071 | 0.071 |
| 1.00 | 0.045 | 0.071 | 0.071 |

**Every nonzero λ gives the identical accuracy trajectory** (peak ~0.071 by step 50, held to
step 100). Only λ=0 (A1) regresses by step 100. The *value* of λ from 0.05 → 1.0 makes no
difference to accuracy.

## 4. Secondary signals

| | λ=0.05 | λ=0.5 | λ=1.0 |
|---|---|---|---|
| Train reward (step 1 → 100) | 0.079 → 0.097 | 0.501 → 0.532 | 0.969 → 1.009 |
| KL-to-reference (step 100) | 0.0030 | 0.0030 | 0.0026 |
| Entropy (step 100) | 1.468 | 1.459 | 1.461 |
| Response length (step 1 → 100) | 201 → 198 tok | 201 → 194 tok | 201 → 200 tok |

KL stayed tiny and entropy was flat (~1.46) across all λ. No reasoning-length inflation in any
run.

---

## 5. Interpretation

**The format weight λ does not affect learning — it is a constant offset that cancels in the
GRPO advantage.** GRPO normalizes rewards within each group, `Â_i = (R_i − mean(R)) / std(R)`.
Since the format term `λ·R_f ≈ λ·1` is nearly constant across a group (validity is saturated),
it cancels in `(R_i − mean(R))` and contributes essentially no gradient — regardless of how
large λ is.

This is exactly why:
- Absolute validation reward shifts up by exactly λ across the sweep (§2), but
- Offset-removed accuracy is identical for every λ > 0 (§3).

The one real effect — already seen in A0 vs A1 — is **stabilization at the accuracy peak**:
any nonzero λ keeps validation pinned at ~0.071 through step 100, whereas λ=0 drifts back to
the step-0 level. But the *magnitude* of λ beyond "nonzero" is irrelevant on this task.

---

## 6. Bottom line

A2 confirms and sharpens the A0/A1 conclusion: on Games, **the format reward's weight is
immaterial** because validity is already saturated at the RA init, so `λ·R_f` is a constant
that cancels in the group-relative advantage. λ ∈ {0.05, 0.1, 0.5, 1.0} all yield the same
accuracy; the only role of a nonzero λ is mild stabilization vs. λ=0.

**Practical takeaway:** there is no "validity vs. accuracy trade-off" to tune here — a small
nonzero λ (the 0.1 baseline is fine) is sufficient; raising it buys nothing.

**Caveats:** 100 steps with only 3 validation points is noisy, these are reward proxies (not
NDCG@k / HR@k), and λ=0.2 was not run (it would not change the conclusion given the flat trend).

## 7. Held-out NDCG / HR eval (done)

The think-mode eval has now been run on the merged S3 checkpoints (Slurm job 23816638; see
[`a0_a3_eval_metrics.md`](a0_a3_eval_metrics.md) for the full A0–A3 table and method).
Games full test set, `n_beam=10`:

| λ | NDCG@5 | NDCG@10 | HR@5 | HR@10 |
|---|---|---|---|---|
| 0.00 (A1) | 0.0486 | 0.0588 | 0.0658 | 0.0972 |
| 0.05 | 0.0439 | 0.0544 | 0.0659 | **0.0987** |
| 0.10 (A0) | 0.0438 | 0.0543 | 0.0648 | 0.0977 |
| 0.50 | 0.0438 | 0.0537 | 0.0632 | 0.0941 |
| 1.00 | 0.0464 | 0.0564 | **0.0667** | 0.0978 |

**No monotonic trend in λ.** NDCG@10 across λ ∈ {0.05, 0.5, 1.0} goes 0.0544 → 0.0537 → 0.0564 —
noise around the baseline, not a trend. The full spread across the whole sweep is only ~0.005
NDCG@10. This confirms the §5 argument on the held-out metric: **the value of λ is immaterial**
because `λ·R_f` is a near-constant offset that cancels in the GRPO group-relative advantage. The
only notable point is that λ=0 (A1) is at the *top* of the NDCG band rather than the bottom — the
opposite of what the proxy's "λ=0 regresses" signal implied (the proxy was not a faithful
stand-in for ranking quality).

---

### Source

Metrics extracted from the Slurm logs:
- λ=0.05: `jobs/outputs/grpo-reward-sweep-23760697.out`
- λ=0.5:  `jobs/outputs/grpo-reward-sweep-23760698.out`
- λ=1.0:  `jobs/outputs/grpo-reward-sweep-23760699.out`
- λ=0.1 (A0) / λ=0 (A1): see [`a0_a1_games.md`](a0_a1_games.md)
