# GRPO Reward-Shaping Experiments (A0–A3)

> **RQ3: Is the GRPO reward formulation a sensitive design choice?** We vary reward
> formulas from the same authors' post-RA Games checkpoint, holding GRPO objective
> hyperparameters fixed.

The revised paper distinguishes **stable list-level quality (K=10)** from **sensitive
rank-1 accuracy**. A1 (no format reward) improves N@1 by about 43%, and A3 (linear
partial credit) by about 34%, over A0 in these single-seed runs. This complements
[RQ2](../components/README.md): RL supplies the main gains, while reward design can
still change which item is ranked first. KL/group-size/objective ablations are not
part of the reported experiment.

This document covers the four reward variants actually trained and evaluated: A0 (baseline),
A1 (no format term), A2 ($\lambda$ sweep), A3 (linear accuracy). Per-variant write-ups with the
full numbers live under [`../../reports/grpo/`](../../reports/grpo/).

---

## Pipeline context

SIDReasoner trains in three stages:

| Stage | Name | Purpose | Checkpoint |
|-------|------|---------|------------|
| S1 | SFT | Teach the model to emit valid Semantic IDs (SIDs). | `sidreasoner_sft_*` |
| S2 | Reasoning-activation (RA) | Teach the model to wrap its answer in `<think>…</think>` before the SID. | `sidreasoner_ra_*` |
| **S3** | **GRPO RL** | **Reinforce the policy so reasoning leads to correct recommendations.** | this experiment |

Every experiment here is **S3 only**. They **branch from the authors' S2 / RA checkpoint**
(`data/checkpoints/sidreasoner_ra_games`, Qwen3-1.7B, Video Games domain), with S1 and S2
disabled, so each run isolates exactly the reward change under test.

Backbone: `Qwen/Qwen3-1.7B` (the original paper's setting). Dataset: `Video Games`.

---

## The baseline reward (A0)

The released reward is a sum of an accuracy term and a format term:

```math
R = R_{\text{sr}} + \lambda \cdot R_{\text{f}}, \qquad \lambda = 0.1
```

The accuracy term is hierarchical (coarse-to-fine), doubling at each matched level:

```math
R_{\text{sr}} =
\begin{cases}
1.00 & \text{if } a_1 = g_1,\ a_2 = g_2,\ a_3 = g_3 \\
0.50 & \text{if } a_1 = g_1,\ a_2 = g_2,\ a_3 \neq g_3 \\
0.25 & \text{if } a_1 = g_1,\ a_2 \neq g_2 \\
0 & \text{otherwise}
\end{cases}
\qquad R_{\text{sr}} \in \{0,\ 0.25,\ 0.50,\ 1.00\}
```

```math
R_{\text{f}} =
\begin{cases}
1 & \text{if } (a_1, a_2, a_3) \text{ is a valid path in the SID catalogue trie} \\
0 & \text{otherwise}
\end{cases}
```

$a = (a_1, a_2, a_3)$ is the predicted SID, $g$ the ground truth. The GRPO advantage normalizes
rewards **within each group** of $G = 16$ rollouts:

```math
\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}
```

A reward term constant within a rollout group cancels when subtracting the group mean.
This explains reduced sensitivity to the format weight when validity has saturated; it
does not imply invariance to arbitrary monotone changes of the accuracy reward.

---

## The experimental axis

Each variant is a **single controlled change** to the reward formula above; all GRPO objective
hyperparameters are held fixed.

| Variant | What changes | Tests |
|---------|--------------|-------|
| **A0** baseline | $R = R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ (original formula). | Reference bar. |
| **A1** no-format | $R = R_{\text{sr}}$ ($\lambda = 0$; format term removed). | Whether the $0.1 \cdot R_{\text{f}}$ bonus does any real work. |
| **A2** $\lambda$-sweep | $R = R_{\text{sr}} + \lambda \cdot R_{\text{f}}$, $\lambda \in \{0.05,\ 0.5,\ 1.0\}$. | The validity-vs-accuracy trade-off; whether $\lambda = 0.1$ is special. |
| **A3** linear acc | $R_{\text{sr}} = 0.33 \cdot [a_1 = g_1] + 0.33 \cdot [a_1 = g_1,\ a_2 = g_2] + 0.34 \cdot [\text{all match}]$, plus $0.1 \cdot R_{\text{f}}$. | Whether smoother partial credit beats the steep $\times 2$ hierarchy. |

---

## Experiments

### A0 — Baseline

**Reward:** $R = R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ (original authors' formula). Reference condition; establishes
the NDCG/HR bar every other variant is compared against.

### A1 — No format term

**Reward:** $R = R_{\text{sr}}$ ($\lambda = 0$). Tests whether the format bonus matters, or whether GRPO + KL
already keep outputs catalogue-valid without it.

**Finding.** Best variant on NDCG (NDCG@10 = 0.0588) — dropping the format term did *not* hurt
validity or accuracy. The bonus is redundant signal.

### A2 — Format-weight sweep

**Reward:** $R = R_{\text{sr}} + \lambda \cdot R_{\text{f}}$, $\lambda \in \{0.05,\ 0.5,\ 1.0\}$ (three configs; $\lambda = 0.2$ was planned but
only dry-run). Quantifies the validity-pressure vs. accuracy-signal trade-off around $\lambda = 0.1$.

**Finding.** NDCG@10 across $\lambda$ is non-monotonic ($0.0544 \to 0.0537 \to 0.0564$).
The observed top-10 differences are small; these single-seed results do not establish equivalence.

### A3 — Linear partial-credit accuracy

**Reward:** additive partial credit (no $\times 2$ doubling), plus $0.1 \cdot R_{\text{f}}$. Tests whether spreading credit
more evenly across the three SID levels gives a smoother gradient.

**Finding.** A3 is within $\sim 0.003$ NDCG@10 of A0, but NDCG@1 rises from
0.0216 to 0.0290 (about 34%). Linear partial credit is not an affine rescaling of A0.
The author-confirmed linear reward uses increments **0.33/0.33/0.34**, giving
cumulative credit 0/0.33/0.66/1. The implementation is the source of truth.

---

## Results

All six trained checkpoints (A0, A1, A2$\times$3, A3) evaluated with the think-mode ranking eval
(`scripts/common/eval/evaluate.py → run_eval_think`) on the full Games held-out test set
($\sim$6.1k examples, `n_beam=10`, vLLM, `global_step_100` merged actor). Slurm job 23816638.

The reported values below follow revised **Table 3** exactly, including its four-decimal
precision. They are paper transcriptions, not freshly recomputed evaluation outputs.

### NDCG@k

| Experiment | Reward | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| **A0** baseline   | $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$  | 0.0216 | 0.0380 | 0.0438 | 0.0543 |
| **A1** no-format  | $R_{\text{sr}}$ ($\lambda = 0$)             | 0.0308 | 0.0428 | **0.0486** | **0.0588** |
| **A2** $\lambda = 0.05$ | $+\,0.05 \cdot R_{\text{f}}$         | 0.0213 | 0.0376 | 0.0439 | 0.0544 |
| **A2** $\lambda = 0.5$  | $+\,0.5 \cdot R_{\text{f}}$          | 0.0226 | 0.0379 | 0.0438 | 0.0537 |
| **A2** $\lambda = 1.0$  | $+\,1.0 \cdot R_{\text{f}}$          | 0.0252 | 0.0399 | 0.0464 | 0.0564 |
| **A3** linear acc | linear $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ | 0.0290 | 0.0403 | 0.0470 | 0.0571 |

### HR@k

| Experiment | Reward | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---|---|---|---|
| **A0** baseline   | $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$  | 0.0216 | 0.0506 | 0.0648 | 0.0977 |
| **A1** no-format  | $R_{\text{sr}}$ ($\lambda = 0$)             | 0.0308 | 0.0516 | 0.0658 | 0.0972 |
| **A2** $\lambda = 0.05$ | $+\,0.05 \cdot R_{\text{f}}$         | 0.0213 | 0.0506 | 0.0659 | **0.0987** |
| **A2** $\lambda = 0.5$  | $+\,0.5 \cdot R_{\text{f}}$          | 0.0226 | 0.0488 | 0.0632 | 0.0941 |
| **A2** $\lambda = 1.0$  | $+\,1.0 \cdot R_{\text{f}}$          | 0.0252 | 0.0508 | **0.0667** | 0.0978 |
| **A3** linear acc | linear $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ | 0.0290 | 0.0487 | 0.0650 | 0.0965 |

**Finding (overall).** NDCG@10 spans 0.0537–0.0588 and HR@10 spans 0.0941–0.0987.
At rank 1, A1 rises from A0's 0.0216 to 0.0308 (about 43%), and A3 reaches 0.0290
(about 34%). Thus the revised paper's result is top-10 robustness with rank-1 sensitivity,
not reward irrelevance at every cutoff. These are point estimates, not multi-seed
significance claims.

**Mechanism and limits.** At initialization, free-running rollouts are already almost
always catalog-valid (`R_f ≈ 1`), so the format bonus is approximately constant within
a group and cancels in mean subtraction. This is learned validity, not constrained
training-time decoding. Exact constant-offset cancellation does not prove that the
observed rank-1 gains were caused by removing that term; the paper's signal-dilution
explanation is a hypothesis. Standard-deviation normalization is invariant to positive
scaling only when the epsilon term is negligible, and not to arbitrary monotone reward
transformations such as A3.

**Proxy versus ranking.** Appendix C reports that A0 and λ=0.05 retain the step-50
proxy peak at step 100, while A1 and A3 regress. Held-out ranking reverses that ordering.
Use validation ranking metrics for model selection; the test metrics here are for reporting.
Reported ranking tables and summaries use the paper as the authoritative source;
see the [audit](../paper-alignment.md) for the alignment record.

---

## Released checkpoints

The six trained GRPO actors (merged `global_step_100`, Games) are on the Hugging Face Hub:

| Variant | Reward | Checkpoint |
|---|---|---|
| **A0** baseline   | $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$        | [`grpo-a0-baseline-games`](https://huggingface.co/SalvatoreLS/grpo-a0-baseline-games) |
| **A1** no-format  | $R_{\text{sr}}$ ($\lambda = 0$)                 | [`grpo-a1-no-format-games`](https://huggingface.co/SalvatoreLS/grpo-a1-no-format-games) |
| **A2** $\lambda = 0.05$ | $+\,0.05 \cdot R_{\text{f}}$              | [`grpo-a2-l005-games`](https://huggingface.co/SalvatoreLS/grpo-a2-l005-games) |
| **A2** $\lambda = 0.5$  | $+\,0.5 \cdot R_{\text{f}}$               | [`grpo-a2-l05-games`](https://huggingface.co/SalvatoreLS/grpo-a2-l05-games) |
| **A2** $\lambda = 1.0$  | $+\,1.0 \cdot R_{\text{f}}$               | [`grpo-a2-l10-games`](https://huggingface.co/SalvatoreLS/grpo-a2-l10-games) |
| **A3** linear acc | linear $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ | [`grpo-a3-linear-acc-games`](https://huggingface.co/SalvatoreLS/grpo-a3-linear-acc-games) |

(λ = 0.2 was only dry-run, so no A2 `l02` checkpoint.)

---

## Findings and caveats

- Reward design has small top-10 effects in this setting, but removing the format term
  or using linear credit improves rank-1 accuracy in the measured runs.
- Format saturation explains approximate cancellation of the format bonus; it does not
  make all reward formulas mathematically equivalent.
- Games only, Qwen3-1.7B, one seed, 100 GRPO steps per variant. λ=0.2 is an optional
  unreported configuration, not one of the six Table 3 conditions.
- Training uses four H100 GPUs, matching the author-confirmed setup and sweep defaults.

## Reference files

| Area | Path |
|------|------|
| Source code | [`../../src/grpo/`](../../src/grpo/) |
| Configs | [`../../configs/grpo/`](../../configs/grpo/) |
| Jobs | [`../../jobs/grpo/`](../../jobs/grpo/) |
| Scripts | [`../../scripts/grpo/`](../../scripts/grpo/) |
| Tests | [`../../tests/grpo/`](../../tests/grpo/) |
| Reports | [`../../reports/grpo/`](../../reports/grpo/) |
| Checkpoints | [Hugging Face](https://huggingface.co/SalvatoreLS) — see [Released checkpoints](#released-checkpoints) |
| Results | Unpopulated |

---

## Reproduce

All variants (A0–A3) are coded up and ready to run. They start from the S2 RA checkpoint, so no S1/S2 re-training is needed.

**Run a single variant (locally):**
```bash
python scripts/common/train/train_pipeline.py \
    --config configs/grpo/a1_no_format/games.yaml \
    [--dry-run]
```

**Run a single variant (Snellius cluster):**
```bash
sbatch --export=ALL,CONFIG=configs/grpo/a1_no_format/games.yaml jobs/grpo/reward_sweep.job
```

**Run the six paper variants sequentially (Snellius cluster):**
```bash
sbatch jobs/grpo/reward_sweep.job
```

The paper configs cap training at **100 optimizer steps**, including direct Python runs.
`GRPO_MAX_STEPS_OVERRIDE` can override the cap for additional experiments. To run the
unreported λ=0.2 condition, pass `CONFIG=configs/grpo/a2_lambda_sweep/games_l02.yaml`.

**Evaluation:** Use `scripts/common/eval/evaluate.py` (think-mode eval, `run_eval_think`) on the merged S3 actor checkpoint.

## Implementation Details

The reward modifications are implemented outside the submodule to keep the core clean:

- **Logic:** `src/grpo/reward_variants.py` contains all A0–A3 reward classes.
- **Shims:** `src/grpo/reward_shims/` contains thin files that import from `reward_variants` and expose `rule_base_reward` to the `verl` framework.
- **Adapter Changes:** `baselines/adapters/sid_reasoner.py` was extended to read `stages.s3.start_checkpoint` to inject the RA checkpoint directly into GRPO, and to map GRPO objective knobs from the YAML configs to `verl` arguments.
