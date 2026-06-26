# GRPO Reward-Shaping Experiments (A0–A3)

> **RQ4: Is the GRPO reward / objective a sensitive design choice?** We vary the reward
> formulation and related GRPO settings, starting from the authors' reasoning-activation (RA)
> checkpoint, to assess whether the method depends strongly on a particular reward specification
> or whether its behavior is stable across reasonable alternatives.

This suite asks *how much the reward shape inside the RL stage matters*. The components suite
([`../components/`](../components/README.md)) asks the prior question of whether the RL stage is
needed at all; this suite takes the RL stage as given and probes its reward. The two findings
are complementary: components shows the RL stage is necessary; GRPO shows that *given* the RL
stage, the choice of reward formula is immaterial to final ranking quality.

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

This normalization is the reason most reward edits turn out to be gauge degrees of freedom — see
the findings below.

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

**Finding.** NDCG@10 across $\lambda$ is non-monotone noise ($0.0544 \to 0.0537 \to 0.0564$), not a trend. $\lambda$
is immaterial.

### A3 — Linear partial-credit accuracy

**Reward:** additive partial credit (no $\times 2$ doubling), plus $0.1 \cdot R_{\text{f}}$. Tests whether spreading credit
more evenly across the three SID levels gives a smoother gradient.

**Finding.** Within $\sim 0.003$ NDCG@10 of A0 — second-best on NDCG, but inside the band. Accuracy
curvature is also a gauge degree of freedom.

---

## Results

All six trained checkpoints (A0, A1, A2$\times$3, A3) evaluated with the think-mode ranking eval
(`scripts/common/eval/evaluate.py → run_eval_think`) on the full Games held-out test set
($\sim$6.1k examples, `n_beam=10`, vLLM, `global_step_100` merged actor). Slurm job 23816638.

### NDCG@k

| Experiment | Reward | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|
| **A0** baseline   | $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$  | 0.02165 | 0.03800 | 0.04378 | 0.05431 |
| **A1** no-format  | $R_{\text{sr}}$ ($\lambda = 0$)             | 0.03077 | 0.04285 | **0.04865** | **0.05879** |
| **A2** $\lambda = 0.05$ | $+\,0.05 \cdot R_{\text{f}}$         | 0.02133 | 0.03762 | 0.04389 | 0.05436 |
| **A2** $\lambda = 0.5$  | $+\,0.5 \cdot R_{\text{f}}$          | 0.02263 | 0.03787 | 0.04378 | 0.05371 |
| **A2** $\lambda = 1.0$  | $+\,1.0 \cdot R_{\text{f}}$          | 0.02524 | 0.03994 | 0.04646 | 0.05639 |
| **A3** linear acc | linear $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ | 0.02898 | 0.04030 | 0.04698 | 0.05707 |

### HR@k

| Experiment | Reward | HR@1 | HR@3 | HR@5 | HR@10 |
|---|---|---|---|---|---|
| **A0** baseline   | $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$  | 0.02165 | 0.05063 | 0.06480 | 0.09769 |
| **A1** no-format  | $R_{\text{sr}}$ ($\lambda = 0$)             | 0.03077 | 0.05161 | 0.06578 | 0.09720 |
| **A2** $\lambda = 0.05$ | $+\,0.05 \cdot R_{\text{f}}$         | 0.02133 | 0.05063 | 0.06594 | **0.09866** |
| **A2** $\lambda = 0.5$  | $+\,0.5 \cdot R_{\text{f}}$          | 0.02263 | 0.04884 | 0.06317 | 0.09411 |
| **A2** $\lambda = 1.0$  | $+\,1.0 \cdot R_{\text{f}}$          | 0.02524 | 0.05080 | **0.06659** | 0.09785 |
| **A3** linear acc | linear $R_{\text{sr}} + 0.1 \cdot R_{\text{f}}$ | 0.02898 | 0.04868 | 0.06496 | 0.09655 |

**Finding (overall).** All variants land in a narrow band: NDCG@10 spans 0.0537–0.0588
(width $\sim 0.005$), HR@10 spans 0.0941–0.0987. No variant reliably separates from the baseline, and
A0 sits at the *bottom* of the NDCG band despite being the "stable" variant on the in-training
proxy.

**Finding (why).** Reward shape is nearly irrelevant *by construction*. The RA checkpoint already
emits valid catalogue SIDs $\sim 100\%$ of the time ($R_{\text{f}} \approx 1.0$ at step 0), so $\lambda \cdot R_{\text{f}}$ is an
approximately constant offset that **cancels** in the group-relative advantage
$\hat{A}_i = (R_i - \text{mean}(R)) / \text{std}(R)$ — making $\lambda$ immaterial (A2). Likewise, replacing the $\times 2$ hierarchy
with additive partial credit (A3) is a monotone rescaling absorbed by std-normalization.

**Finding (proxy disagrees).** The in-training `mean@7` reward proxy suggested A1/A3 *regressed*
by step 100 while A0 *held* — the opposite of the held-out NDCG/HR ranking. Mean reward is not
the ranking objective; its step-100 value is largely a gauge quantity and must not be used for
model selection when format validity is already saturated.

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

## Bottom line

- **Reward shape is immaterial.** From a strong reasoning-activated init, dropping, reweighting,
  or reshaping the format and accuracy terms all give equivalent ranking quality on Games.
- **The format term and accuracy curvature are gauge degrees of freedom** of GRPO's
  group-relative advantage normalization — not tunable levers.
- **Practical implication.** Reward-shaping effort is unlikely to pay off. The performance ceiling
  is set by the RA initialization and the data; effort is better spent on the GRPO *objective*
  (KL coefficient, group size, advantage norm).

## Caveats

Games domain only, single seed, 100 training steps per variant. Confirming on Office and over
multiple seeds would strengthen the "reward shape is immaterial" claim.

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
sbatch --export=CONFIG=configs/grpo/a1_no_format/games.yaml jobs/grpo/reward_sweep.job
```

**Sweep all variants at once (Snellius cluster):**
```bash
sbatch jobs/grpo/reward_sweep.job
```

**Evaluation:** Use `scripts/common/eval/evaluate.py` (think-mode eval, `run_eval_think`) on the merged S3 actor checkpoint.

## Implementation Details

The reward modifications are implemented outside the submodule to keep the core clean:

- **Logic:** `src/grpo/reward_variants.py` contains all A0–A3 reward classes.
- **Shims:** `src/grpo/reward_shims/` contains thin files that import from `reward_variants` and expose `rule_base_reward` to the `verl` framework.
- **Adapter Changes:** `baselines/adapters/sid_reasoner.py` was extended to read `stages.s3.start_checkpoint` to inject the RA checkpoint directly into GRPO, and to map GRPO objective knobs from the YAML configs to `verl` arguments.
