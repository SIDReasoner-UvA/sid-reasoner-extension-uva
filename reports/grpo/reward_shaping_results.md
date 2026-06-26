# On the (Ir)relevance of Reward Shape in GRPO for Generative Recommendation

*A study of reward-formula ablations (A0–A3) in stage S3 of the SIDReasoner pipeline.*

## Abstract

We ablate the GRPO reward function used to fine-tune a reasoning recommender (Qwen3-1.7B,
SIDReasoner stage S3) on the Amazon Video Games benchmark. Starting from the reasoning-activation
(S2) checkpoint, we train six variants that perturb the accuracy term and the format-reward weight
$\lambda$, holding every GRPO-objective hyperparameter fixed, and evaluate each on the full held-out
test set with think-mode NDCG@k and HR@k. Our central finding is **negative and robust**: across all
six reward shapes the held-out ranking quality lies in a band of $\sim 0.005$ NDCG@10
(0.0537–0.0588), with no variant reliably separating from the baseline. We trace this to a structural
property of GRPO — its group-relative advantage is invariant to constant offsets and to monotone
rescaling of the reward — and show that the one component the reward shaping was meant to control
(output validity) is already saturated at initialization, leaving the shaping with almost no gradient
to act on. We further report a methodological caution: the in-training reward proxy that is cheap to
log *disagrees* with the held-out metric on the ranking of variants, and its apparent "stability"
signal is not a quality signal.

## 1. Setup

**Pipeline and initialization.** GRPO is stage S3 of SIDReasoner (S1 SFT → S2 reasoning-activation →
S3 RL). All runs start from the authors' S2 reasoning-activation checkpoint
(`sidreasoner_ra_games`), so the model already emits `<think>…</think>` followed by a semantic-ID
(SID) recommendation before RL begins. This isolates the effect of the reward change: S1 and S2 are
disabled and shared across all runs, which begin from an identical step-0 state on 4×A100.

**Reward.** Each generated sample is parsed into its first three SID tokens $a = (a_1, a_2, a_3)$ and
scored against the ground truth $g = (g_1, g_2, g_3)$. The baseline reward is

$$R = R_{\mathrm{acc}}(a, g) + \lambda \cdot R_{\mathrm{fmt}}(a), \qquad \lambda = 0.1,$$

where $R_{\mathrm{acc}} \in \{0,\, 0.25,\, 0.50,\, 1.0\}$ is a hierarchical coarse-to-fine match (each
deeper level doubles the reward, gated on the previous level matching), and
$R_{\mathrm{fmt}} \in \{0, 1\}$ rewards a catalogue-realizable SID path. GRPO forms the advantage of
output $i$ within a group of $G = 16$ rollouts as

$$\hat{A}_i = \frac{R_i - \operatorname{mean}(R)}{\operatorname{std}(R) + \epsilon}. \tag{1}$$

**Variants.** We perturb only the reward:

| ID | Reward change |
|----|---------------|
| **A0** | baseline, $R = R_{\mathrm{acc}} + 0.1 \cdot R_{\mathrm{fmt}}$ |
| **A1** | drop the format term, $R = R_{\mathrm{acc}}$ ($\lambda = 0$) |
| **A2** | sweep the format weight, $\lambda \in \{0.05,\, 0.5,\, 1.0\}$ |
| **A3** | linear/partial-credit accuracy, $R_{\mathrm{acc}} = 0.33\,[a_1{=}g_1] + 0.33\,[a_1{=}g_1, a_2{=}g_2] + 0.34\,[\text{all}]$ (additive, no $\times 2$ doubling) |

**Training and evaluation.** Each variant trains for 100 GRPO steps (batch 256, $G = 16$) from the
shared RA init. We log the in-training reward proxy at steps 0/50/100, then merge the
`global_step_100` actor and evaluate it with the think-mode ranking eval over the full Games test
split ($\sim 6.1\text{k}$ examples, $n_{\text{beam}} = 10$), reporting NDCG@{1,3,5,10} and
HR@{1,3,5,10}.

## 2. Results

### 2.1 Held-out ranking quality (the metric of record)

| Variant | Reward | NDCG@5 | NDCG@10 | HR@5 | HR@10 |
|---|---|---|---|---|---|
| **A0** baseline   | $R_{\mathrm{acc}} + 0.1 \cdot R_{\mathrm{fmt}}$ | 0.04378 | 0.05431 | 0.06480 | 0.09769 |
| **A1** no-format  | $R_{\mathrm{acc}}$ ($\lambda{=}0$)              | **0.04865** | **0.05879** | 0.06578 | 0.09720 |
| **A2** $\lambda{=}0.05$ | $+0.05 \cdot R_{\mathrm{fmt}}$            | 0.04389 | 0.05436 | 0.06594 | **0.09866** |
| **A2** $\lambda{=}0.5$  | $+0.5 \cdot R_{\mathrm{fmt}}$             | 0.04378 | 0.05371 | 0.06317 | 0.09411 |
| **A2** $\lambda{=}1.0$  | $+1.0 \cdot R_{\mathrm{fmt}}$             | 0.04646 | 0.05639 | **0.06659** | 0.09785 |
| **A3** linear acc | linear $R_{\mathrm{acc}}$                       | 0.04698 | 0.05707 | 0.06496 | 0.09655 |

All six variants fall in a narrow band: NDCG@10 spans 0.0537–0.0588 (width $\approx 0.005$) and HR@10
spans 0.0941–0.0987 (width $\approx 0.005$). The best NDCG belongs to **A1 ($\lambda = 0$)**, the
second to **A3 (linear)**, and the baseline **A0 sits at the bottom of the NDCG band** — yet the gaps
are of the same order as the spread itself.

### 2.2 In-training reward proxy (`val-core/.../reward/mean@7`)

Offset-removed accuracy ($R_{\mathrm{acc}} \approx \text{proxy} - \lambda$, valid because the format
rate is $\approx 1.0$, see §3.2):

| Variant | step 0 | step 50 | step 100 | trajectory |
|---|---|---|---|---|
| A0 ($\lambda{=}0.1$) | 0.045 | 0.071 | 0.071 | holds the peak |
| A1 ($\lambda{=}0$)   | 0.045 | 0.071 | **0.045** | regresses |
| A2 (any $\lambda{>}0$) | 0.045 | 0.071 | 0.071 | holds the peak |
| A3 (linear) | 0.045 | 0.071 | **0.045** | regresses |

On the proxy, A0 and every $\lambda > 0$ "hold" their step-50 peak, whereas A1 and A3 "regress" to
the step-0 level by step 100. Naively this ranks A0 above A1/A3.

## 3. From numbers to interpretation

### 3.1 Why the spread is tiny: GRPO is invariant to the parts we changed

Equation (1) is the key. The advantage depends on the reward only through its *within-group*
deviation, normalized by the group std. This has two immediate consequences for our ablations.

**(i) A constant offset cancels (explains A1 vs A2).** The format term $\lambda \cdot R_{\mathrm{fmt}}$
is nearly constant across a group whenever validity is saturated ($R_{\mathrm{fmt}} \approx 1$ for all
rollouts). Then it contributes equally to every $R_i$ and to $\operatorname{mean}(R)$, so it vanishes
in $R_i - \operatorname{mean}(R)$. The *value* of $\lambda$ therefore cannot matter. The data confirm
this exactly: the absolute proxy in §2.2 shifts up by precisely $\lambda$ across the sweep, but the
offset-removed accuracy is *identical* for every $\lambda > 0$, and on held-out NDCG@10 the sweep
$\lambda \in \{0.05, 0.5, 1.0\}$ gives $0.05436 \rightarrow 0.05371 \rightarrow 0.05639$ —
non-monotone noise, not a trend.

**(ii) Monotone rescaling cancels too (explains A3).** A3 replaces the steep $\times 2$ hierarchy with
a flatter additive partial credit. This changes the *scale and curvature* of the reward landscape,
but the std-normalization in (1) makes the policy gradient scale-invariant: a uniformly "denser"
reward produces larger raw rewards (A3's train reward sits higher, as observed) yet the same
normalized advantages. So we should *not* expect A3 to separate from A0 — and on held-out metrics it
does not ($\Delta \approx 0.003$ NDCG@10, within the band).

In other words, the two knobs we turned — the additive format weight and the accuracy curvature — are
in large part *gauge degrees of freedom* of GRPO. The theory predicts they should be close to inert,
and the held-out band of $\sim 0.005$ NDCG@10 is what "close to inert" looks like in practice.

### 3.2 Why there was no validity gradient to begin with

A1 was designed to test whether the format term does real work. The cleanest evidence comes from
step 0, where A0 and A1 share the *same* model:

$$\text{proxy}(\text{A0}) - \text{proxy}(\text{A1}) = 0.1446 - 0.0446 \approx 0.10 = 0.1 \cdot R_{\mathrm{fmt}} \;\implies\; R_{\mathrm{fmt}} \approx 1.0.$$

The RA checkpoint already emits valid catalogue SIDs essentially 100 % of the time. The format
reward is thus a flat $+0.1$ with no discriminative signal: there is no population of invalid outputs
for it to push against. This is *why* §3.1(i) bites — validity saturation is the precondition that
makes $\lambda \cdot R_{\mathrm{fmt}}$ a constant. The format term is not "helping" or "hurting"; on
this init it is nearly a no-op.

### 3.3 The proxy is not the metric: why "stability" misled us

The proxy and the held-out metric **disagree on the variant ordering**, and resolving that
disagreement is the main methodological lesson. The proxy said A1 and A3 *regress* (and A0 is
*stable*); the held-out NDCG/HR say A1 and A3 are, if anything, *slightly ahead* of A0. Both cannot
be a quality ranking, and the held-out metric is the one we care about.

The reconciliation: the proxy is a 100-step, three-point trace of mean reward — and mean reward is
not the ranking objective. Under GRPO the absolute reward level and its step-100 wobble are largely
the gauge quantities of §3.1, which the advantage discards; the held-out NDCG is computed by an
entirely different procedure (constrained beam search over the catalogue, scored against held-out
purchases). There is no reason the step-100 value of a near-constant proxy should track held-out
ranking, and empirically it does not. The practical takeaway is to **distrust proxy "stability" as a
model-selection signal** here and to select on the held-out metric — which itself says the choice
barely matters.

### 3.4 What *does* set the ceiling

Every variant reaches the same accuracy peak ($\sim 0.071$ offset-removed) by step 50, and all land
within $0.005$ NDCG@10 of one another. The accuracy ceiling is therefore set not by the reward shape
but by the data and the S2 initialization. Reward-shape tweaks redistribute reward mass and at most
affect mild step-100 wobble on the proxy; they do not move the held-out ceiling. Absolute scores are
low (NDCG@10 $\approx 0.05\text{–}0.06$), as expected for this backbone and task scale.

## 4. Conclusion

On Amazon Video Games, **the shape of the GRPO reward is immaterial to held-out recommendation
quality**: dropping the format term (A1), sweeping its weight over 1.5 orders of magnitude (A2), and
replacing the hierarchical accuracy with linear partial credit (A3) all land within $\sim 0.005$
NDCG@10 of the baseline, with no reliable winner. This is not an accident of tuning but a consequence
of GRPO's group-relative advantage, which is invariant to the constant offset (the format term, given
saturated validity) and to the monotone rescaling (the accuracy curvature) that these ablations
manipulate. The corollary for practitioners is that, from a strong reasoning-activated init, effort
is better spent on the GRPO *objective* (the KL coefficient, group size, advantage normalization —
the B-series), on the initialization, or on the data, than on hand-tuning the reward formula.

**Threats to validity.** Single domain (Games), single seed, 100 training steps, and a $\sim 0.005$
held-out spread that may be within noise; the format-saturation premise is specific to this RA init.
Confirming the "reward shape is immaterial" claim on Office, and over multiple seeds, would
strengthen it. A2's $\lambda = 0.2$ point and the full B-series (objective ablations) remain to be
run.

---

*Source reports: [`a0_a1_games.md`](a0_a1_games.md),
[`a2_lambda_sweep_games.md`](a2_lambda_sweep_games.md),
[`a3_linear_acc_games.md`](a3_linear_acc_games.md), and the consolidated held-out eval
[`a0_a3_eval_metrics.md`](a0_a3_eval_metrics.md) (Slurm array 23816638). Design:
[`docs/grpo/README.md`](../../docs/grpo/README.md).*
