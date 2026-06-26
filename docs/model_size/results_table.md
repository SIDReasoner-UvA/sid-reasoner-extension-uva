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

## Takeaways
- **1-vs-5 epoch RA is size-dependent** — 0.6B Games ~tie (0.0501 vs 0.0495); 4B Games 1-ep ≫ 5-ep
  (0.0532 vs 0.0316 — 5-ep halves it). Larger models prefer fewer activation epochs (over-entrench teacher CoT).
- **No scaling benefit:** 0.6B GRPO 0.0734 > 4B GRPO 0.0658 (mild regression); raw alignment ~equal (0.0645 ≈ 0.0628).
  Addresses the paper's open scaling question (their Conclusion) with a negative result on academic-scale data.
- **Faithfulness:** Games S1 SFT 0.0387 ≈ paper 0.0414 ✓ (Games reproduces the paper).
  (Office excluded here — it showed an unexplained ~3× inflation vs paper, isolated to Office; see investigation notes.)
- **4B GRPO caveat:** non-monotonic / possibly under-converged (test peak 0.0697 @ step 200, but val mean@1 peaks at
  500, so 0.0658 is the defensible number). HR@5/N@5 for the 4B GRPO row live in results/figure5_4B.csv.

Notes: HR@5/N@5 for 4B GRPO not extracted here (only HR@10/N@10 from figure5_4B.csv). All numbers reproducible via
the submodule's `scripts/evaluate_models_in_dir.sh` (cached) + `utils/collect_eval_metrics.py`.
