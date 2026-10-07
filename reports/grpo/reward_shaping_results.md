# RQ3: Reward sensitivity at different ranking cutoffs

This report aligns with §6, Table 3, and Appendix C of `ECIR_Paper_revised-3.pdf`.
Reported ranking measurements below are transcribed from revised Table 3 at the
paper's precision; they are not a fresh evaluation.

## Setup

Six reward variants start from the same authors' post-RA Games checkpoint
(Qwen3-1.7B), train for 100 GRPO steps with 256 prompts and 16 rollouts per prompt,
and use greedy thinking plus constrained beam-10 SID decoding at evaluation.
Training uses four H100 GPUs, as confirmed by the authors and configured in the launcher.

A0 uses hierarchical prefix credit `{0, 0.25, 0.5, 1}` plus `0.1 * R_f`. A1 removes
format credit; A2 uses weights 0.05, 0.5, and 1.0; A3 uses additive increments
0.33/0.33/0.34 plus `0.1 * R_f`. The unreported λ=0.2 config remains optional.

## Held-out results

| Variant | Reward | NDCG@5 | NDCG@10 | HR@5 | HR@10 |
|---|---|---|---|---|---|
| **A0** baseline   | $R_{\mathrm{acc}} + 0.1 \cdot R_{\mathrm{fmt}}$ | 0.0438 | 0.0543 | 0.0648 | 0.0977 |
| **A1** no-format  | $R_{\mathrm{acc}}$ ($\lambda{=}0$)              | **0.0486** | **0.0588** | 0.0658 | 0.0972 |
| **A2** $\lambda{=}0.05$ | $+0.05 \cdot R_{\mathrm{fmt}}$            | 0.0439 | 0.0544 | 0.0659 | **0.0987** |
| **A2** $\lambda{=}0.5$  | $+0.5 \cdot R_{\mathrm{fmt}}$             | 0.0438 | 0.0537 | 0.0632 | 0.0941 |
| **A2** $\lambda{=}1.0$  | $+1.0 \cdot R_{\mathrm{fmt}}$             | 0.0464 | 0.0564 | **0.0667** | 0.0978 |
| **A3** linear acc | linear $R_{\mathrm{acc}}$                       | 0.0470 | 0.0571 | 0.0650 | 0.0965 |


At K=10, all variants occupy a narrow range: NDCG 0.0537–0.0588 and Recall
0.0941–0.0987. At K=1, A1's 0.0308 versus A0's 0.0216 is about a **43%** gain;
A3's 0.0290 is about a **34%** gain. These are single-seed observations, not a
statistical equivalence or superiority test. The conclusion is list-level robustness
with top-rank sensitivity, not universal irrelevance of reward shaping.

## Interpretation

The initial proxy gap `0.1446 - 0.0446 ≈ 0.10` implies near-saturated format validity.
A format bonus constant within each rollout group cancels in mean subtraction.
Training rollouts are free-running; constrained SID decoding is used at evaluation.
This explains approximate insensitivity to format weight, but does not establish the
cause of the rank-1 differences. Signal dilution by the format term is the paper's
hypothesis, not a proven consequence of constant-offset cancellation.

Positive affine reward changes preserve standardized advantages when epsilon is
negligible. Arbitrary monotone transformations do not: A3 changes the relative gaps
between partial-match rewards. Its behavior therefore cannot be derived from an
invariance to reward curvature.

Appendix C compares the step-0/50/100 proxy for A0, λ=0.05, A1, and A3. The first two
retain the proxy peak while A1/A3 regress, despite better held-out NDCG. Absolute
mean reward is not a reliable ranking-quality proxy. Select checkpoints on validation
ranking metrics and reserve the test split for final reporting.

## Provenance and limitations

Games only, one seed, 100 steps. Reported numbers follow the paper. The authors
confirmed the implemented reward equations, Transformers 4.51.3, and H100 hardware;
see the [alignment audit](../../docs/paper-alignment.md). Reward computation and raw
evaluation outputs are unchanged.

Sources: [consolidated evaluation](a0_a3_eval_metrics.md), [A0/A1](a0_a1_games.md),
[A2](a2_lambda_sweep_games.md), [A3](a3_linear_acc_games.md).
The [suite README](../../docs/grpo/README.md) contains all cutoffs and run commands.
