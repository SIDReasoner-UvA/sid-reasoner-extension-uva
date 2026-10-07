# SIDReasoner reproduction — results table

HR@10 / NDCG@10, our greedy think-mode eval (constrained beam-10). ★ = the reasoning-activation
(RA) checkpoint that fed GRPO. Paper is Qwen3-1.7B; 0.6B and 4B are our scaling extension.

| Model | Dataset (ablation) | Stage | HR@5 | HR@10 | NDCG@5 | NDCG@10 |
|---|---|---|---|---|---|---|
| 0.6B | Games (S3) | SFT (raw align) | 0.0368 | 0.0645 | 0.0232 | 0.0320 |
| 0.6B | Games (S3) | RA 1-epoch      | 0.0296 | 0.0495 | 0.0181 | 0.0247 |
| 0.6B | Games (S3) | RA 5-epoch ★    | 0.0373 | 0.0501 | 0.0303 | 0.0345 |
| 0.6B | Games (S3) | GRPO (step 500) | 0.0571 | 0.0734 | 0.0448 | 0.0500 |
| 4B   | Games (S3) | SFT (raw align) | 0.0363 | 0.0628 | 0.0234 | 0.0319 |
| 4B   | Games (S3) | RA 1-epoch ★    | 0.0295 | 0.0532 | 0.0181 | 0.0256 |
| 4B   | Games (S3) | RA 5-epoch      | 0.0200 | 0.0316 | 0.0144 | 0.0181 |
| 4B   | Games (S3) | GRPO (step 500) |   —    | 0.0658 |   —    | 0.0364 |
| 1.7B | Games (S1) | SFT (raw align) | 0.0215 | 0.0387 | 0.0129 | 0.0184 |
| 1.7B | Games (S1) | RA 5-epoch      | 0.0321 | 0.0602 | 0.0187 | 0.0277 |

Paper (Qwen3-1.7B), before-RL (RA) / after-RL (GRPO), HR@10:
- Games S3: 0.0806 / 0.1031   - Games S1: 0.0414 / 0.0741

## Interpretation in the revised paper (RQ4)

- R@10 improves and plateaus during GRPO as CoT length decreases and stabilizes.
  These qualitative trends persist across the tested backbone sizes (Figure 2).
- The 4B run has no advantage over 0.6B in these single-seed, untuned comparisons.
  The absolute gap to the original 1.7B result does not by itself identify its cause.
- At 4B, five RA epochs reduce R@10 by about 41% relative to one epoch; at 0.6B the
  epoch variants are close. This motivates the size-dependent choices used here.
- The 1.7B S1 retraining reproduces the SFT → RA improvement. The original 0.0414
  reference is pre-RL/RA, not a separately reported post-SFT score.
- The 4B run is non-monotonic: report the validation-selected step 500 (0.0658),
  not the test peak at step 200 (0.0697). See the suite README for scope and costs.

Notes: HR@5/N@5 for 4B GRPO not extracted here (only HR@10/N@10 from figure5_4B.csv). All numbers reproducible via
the submodule's `scripts/evaluate_models_in_dir.sh` (cached) + `utils/collect_eval_metrics.py`.
